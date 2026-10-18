#!/usr/bin/env python3
"""Governance copilot agent runtime.

Two engines produce the same structured card and the same trace format:

* ``rules``  – deterministic playbook executor. Calls the platform tools in the order the
               playbook prescribes and classifies the cause with fixed rules. No LLM, runs offline.
* ``llm``    – Claude tool-use loop (Anthropic SDK). The playbook markdown is the system prompt,
               the platform tools are the tools, and the final answer must come through the strict
               ``submit_card`` tool.

Guardrails run after either engine (reference validation, hard-coded escalation rules) so the
escalation policy does not depend on the model remembering it.

CLI:
  python -m agent.run --query "美国昨天误杀申诉涨了，帮我看看" [--engine rules|llm]
  python -m agent.run --health [--engine rules|llm]
  python -m agent.run --case LS-0100983
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform, Validator, TOOL_SPECS  # noqa: E402
from gov_platform.tools import METRICS  # noqa: E402

PLAYBOOK_DIR = os.path.join(ROOT, "agent", "playbooks")
TRACE_DIR = os.path.join(ROOT, "traces")
CAUSE_TYPES = ["traffic_driven", "model_drift_threshold", "strategy_change", "new_strategy_launch", "policy_update",
               "recall_drop_evasion", "guild_concentration", "guild_health_real_violations", "guild_health_unfair_deduction",
               "strategy_overlap", "zombie_strategy", "guild_health_breach", "no_anomaly",
               "insufficient_data", "unexplained", "needs_clarification"]
MODEL = "claude-opus-5-5"
PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.20}   # USD per 1M tokens (Claude Opus 5.5)
SMALL_MARKET_SESSIONS = 150
MIN_COUNT_FOR_CONCLUSION = 20


# ----------------------------------------------------------------------------- trace
class Trace:
    def __init__(self, engine, playbook, query):
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.engine, self.playbook, self.query = engine, playbook, query
        self.events, self.t0 = [], time.time()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        self.model_calls = 0

    def add(self, kind, **data):
        self.events.append({"t": round(time.time() - self.t0, 3), "kind": kind, **data})

    def tool(self, name, args, result):
        summary = {k: result[k] for k in ("ref_id", "error", "message") if isinstance(result, dict) and k in result}
        self.add("tool_call", tool=name, args=args, result_summary=summary, result=result)

    def cost_usd(self):
        u = self.usage
        return round((u["input_tokens"] * PRICE["input"] + u["output_tokens"] * PRICE["output"]
                      + u["cache_read_input_tokens"] * PRICE["cache_read"]) / 1e6, 4)

    def finish(self, card):
        self.card = card
        self.latency_s = round(time.time() - self.t0, 3)
        return self

    def to_dict(self):
        return {"run_id": self.run_id, "engine": self.engine, "playbook": self.playbook, "query": self.query,
                "latency_s": getattr(self, "latency_s", None), "model_calls": self.model_calls, "usage": self.usage,
                "cost_usd": self.cost_usd(), "tool_calls": sum(1 for e in self.events if e["kind"] == "tool_call"),
                "events": self.events, "card": getattr(self, "card", None)}

    def save(self):
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1, default=str)
        return path


# ----------------------------------------------------------------------------- query parsing
def parse_query(q, meta):
    """Lightweight intent parser used by the rules engine (and to pre-fill scope for the LLM)."""
    s = {"market": None, "scenario": None, "metric": None, "metric_candidates": [], "date": None,
         "direction": None, "session_id": None, "strategy_id": None, "guild_id": None, "window_days": 1, "raw": q,
         "guild_health": bool(re.search(r"健康分|扣分|health score", q, re.I))}
    m = re.search(r"LS-\d{7}", q)
    if m:
        s["session_id"] = m.group(0)
    m = re.search(r"S-\d{3}", q)
    if m:
        s["strategy_id"] = m.group(0)
    m = re.search(r"G-[A-Z]{2}-\d{3}", q)
    if m:
        s["guild_id"] = m.group(0)
    for code, name in meta["market_names"].items():
        if re.search(rf"(?<![A-Z]){code}(?![A-Z])", q) or name in q:
            s["market"] = code
            break
    for key, name in meta["scenario_names"].items():
        if key in q or name in q or name[:2] in q:
            s["scenario"] = key
            break
    extra = {"诈骗": "scam_fraud", "引流": "scam_fraud", "刷量": "spam_bot", "骚扰": "hate_harassment", "赌博": "gambling",
             "未成年": "minor_safety", "色情": "adult_content", "暴力": "violence", "假冒": "counterfeit"}
    if not s["scenario"]:
        for k, v in extra.items():
            if k in q:
                s["scenario"] = v
                break
    from gov_platform.tools import ALIASES
    for key in sorted(METRICS, key=len, reverse=True):
        if key in q:
            s["metric"], s["metric_candidates"] = key, [key]
            break
    if not s["metric"]:
        for alias in sorted(ALIASES, key=len, reverse=True):
            if alias in q:
                c = ALIASES[alias]
                s["metric_candidates"] = c
                s["metric"] = c[0] if len(c) == 1 else None
                break
    today = date.fromisoformat(meta["today"])
    if "昨天" in q or "yesterday" in q.lower():
        s["date"] = (today - timedelta(days=1)).isoformat()
    elif "前天" in q:
        s["date"] = (today - timedelta(days=2)).isoformat()
    elif "今天" in q:
        s["date"] = today.isoformat()
    elif re.search(r"最近|这几天|近期|这周|本周|近一周|recent|this week", q, re.I):
        s["date"] = (today - timedelta(days=1)).isoformat()
        s["window_days"] = 7
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", q) or re.search(r"(\d{1,2})月(\d{1,2})[日号]", q) or re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)", q)
    if m:
        g = m.groups()
        s["date"] = (date(int(g[0]), int(g[1]), int(g[2])) if len(g) == 3 else date(today.year, int(g[0]), int(g[1]))).isoformat()
    if re.search(r"涨|上升|升高|增加|变多|升了|高了|up|increase", q, re.I):
        s["direction"] = "up"
    elif re.search(r"跌|下降|降低|减少|变少|掉|低了|down|drop|decrease", q, re.I):
        s["direction"] = "down"
    return s


# ----------------------------------------------------------------------------- guardrails
class Guardrails:
    """Hard-coded policy applied to every card regardless of engine."""

    def __init__(self, platform):
        self.p, self.v = platform, Validator(platform)

    def apply(self, card, trace):
        card = _clean(card)
        reasons = list(card.get("escalation_reasons") or [])
        unknown = self.v.check_card(card, trace_refs(trace))
        unknown = [(k, i) for k, i in unknown if i not in (trace.query or "")]   # ids the user typed are not hallucinations
        card["validation"] = {"unknown_references": [f"{k}:{i}" for k, i in unknown], "passed": not unknown}
        if unknown:
            reasons.append(f"校验失败：引用了不存在的对象 {[i for _, i in unknown]}")
        # evidence-derived rules
        head = any(e["kind"] == "tool_call" and e["tool"] == "sample_cases" and isinstance(e.get("result"), dict)
                   and e["result"].get("contains_head_host") for e in trace.events)
        head |= any(e["kind"] == "tool_call" and e["tool"] == "reconstruct_case" and isinstance(e.get("result"), dict)
                    and any("head_host" in f for f in e["result"].get("escalation_flags", [])) for e in trace.events)
        if head and not any("头部主播" in r for r in reasons):
            reasons.append("涉及头部主播，需策略 owner 人工确认")
        errs = [e["result"].get("error") for e in trace.events if e["kind"] == "tool_call" and isinstance(e.get("result"), dict)
                and e["result"].get("error") in ("no_data", "insufficient_sample")]
        if errs and card.get("primary_cause", {}).get("type") not in ("needs_clarification", "no_anomaly"):
            reasons.append(f"部分数据缺失（{', '.join(sorted(set(errs)))}），结论只覆盖可验证部分")
        fin = [sid for sid in (card.get("primary_cause") or {}).get("strategy_ids", []) or []
               if sid in self.p.by_id and self.p.by_id[sid]["action"] in self.p.meta["financial_actions"]]
        if fin and (card.get("primary_cause") or {}).get("type") in ("model_drift_threshold", "strategy_change", "unexplained", "guild_health_unfair_deduction"):
            reasons.append(f"涉及资金侧处置策略 {fin}，误杀需解冻/释放结算，需运营与结算确认")
        if card.get("scope", {}).get("low_sample"):
            if not any("样本不足" in r for r in reasons):
                reasons.append("样本不足（小市场或计数过低）")
        if card.get("primary_cause", {}).get("type") in ("unexplained", "insufficient_data"):
            if not reasons:
                reasons.append("无法归因或样本不足，需人工介入")
        card["escalation_reasons"] = reasons
        card["needs_escalation"] = bool(reasons) or bool(card.get("needs_escalation"))
        if card["needs_escalation"] and card.get("confidence") == "high":
            card["confidence"] = "medium"
        if card.get("scope", {}).get("low_sample"):
            card["confidence"] = "low"
        if card.get("primary_cause", {}).get("type") not in CAUSE_TYPES and card.get("card_type") == "attribution":
            card.setdefault("primary_cause", {})["type"] = "unexplained"
        trace.add("guardrails", needs_escalation=card["needs_escalation"], reasons=reasons,
                  unknown_references=card["validation"]["unknown_references"])
        return card


def _clean(o):
    """Convert numpy scalars to plain Python so cards/traces serialize."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def trace_refs(trace):
    return {e["result"]["ref_id"]: e["tool"] for e in trace.events
            if e["kind"] == "tool_call" and isinstance(e.get("result"), dict) and "ref_id" in e["result"]}


# ----------------------------------------------------------------------------- rules engine
class RulesEngine:
    def __init__(self, platform):
        self.p = platform
        self.meta = platform.meta

    def call(self, trace, name, **args):
        res = self.p.call(name, args)
        trace.tool(name, args, res)
        return res

    # --- helpers
    def _series(self, trace, metric, start, end, **f):
        r = self.call(trace, "query_metrics", metric=metric, start_date=start, end_date=end, **f)
        if "error" in r:
            return None, r
        return {row["date"]: row["value"] for row in r["rows"]}, r

    @staticmethod
    def _baseline(series, day):
        vals = [v for d, v in series.items() if d < day and v is not None]
        return float(np.median(vals)) if vals else 0.0

    def attribution(self, query, scope=None):
        tr = self._attribution_impl(query, scope)
        pre = (tr.card.get("metric") or {}).pop("rate_prefix", None)
        if pre and tr.card.get("conclusion"):
            tr.card["conclusion"] = pre + tr.card["conclusion"]
        return tr

    def _attribution_impl(self, query, scope=None):
        trace = Trace("rules", "attribution", query)
        s = scope or parse_query(query, self.meta)
        W = int(s.get("window_days") or 1)
        trace.add("parse", scope=s)
        card = {"card_type": "attribution", "query": query, "scope": {}, "evidence": [], "next_steps": [],
                "contributing_causes": [], "needs_escalation": False, "escalation_reasons": [], "platform_links": []}
        if s["session_id"]:
            return self.case(query, s["session_id"])
        if s.get("guild_health") and s.get("guild_id"):
            return self.guild_health(query, s["guild_id"], s.get("date") or self.meta["end"])
        if s["strategy_id"] and (not s["market"] or (not s["metric"] and not s["metric_candidates"])):
            return self.health_check(query, strategy_ids=[s["strategy_id"]])
        missing = [k for k in ("market", "date") if not s.get(k)]
        if missing or (not s["metric"] and not s["metric_candidates"]):
            card.update(primary_cause={"type": "needs_clarification", "strategy_ids": [], "change_ids": [],
                                       "explanation": "缺少必要输入，不猜测"},
                        conclusion="请补充：" + "、".join({"market": "市场", "date": "日期/时间窗", "metric": "指标"}[k]
                                                      for k in (missing + ([] if s["metric"] or s["metric_candidates"] else ["metric"]))),
                        confidence="n/a", clarification_questions=[
                            q for k, q in (("market", "哪个市场？（US/BR/ID/GB/VN/KE）"), ("date", "看哪一天或哪个时间窗？"),
                                           ("metric", "指标是申诉推翻数、处罚数、命中数还是其他？")) if k in missing or (k == "metric" and not s["metric"] and not s["metric_candidates"])])
            return trace.finish(Guardrails(self.p).apply(card, trace))
        metric = s["metric"] or s["metric_candidates"][0]
        metric_note = None
        if not s["metric"]:
            metric_note = f"“{[a for a in s['metric_candidates']]}” 存在口径冲突，本次按 {metric}（{METRICS[metric]['definition']}）；其他口径：" + \
                          "；".join(f"{c}: {METRICS[c]['definition']}" for c in s["metric_candidates"][1:])
            card["next_steps"].append("确认口径：如果你指的是另一分母，请说明，我会重算")
        D = s["date"]
        start = (date.fromisoformat(D) - timedelta(days=7 if W == 1 else 20)).isoformat()
        spec = METRICS[metric]
        card["scope"] = {"market": s["market"], "scenario": s["scenario"], "metric": metric, "date": D, "window_days": W,
                         "baseline_window": [start, (date.fromisoformat(D) - timedelta(days=W)).isoformat()],
                         "direction": s["direction"], "note": ("近 7 天合计 vs 前 14 天日均×7" if W > 1 else "单日 vs 前 7 天中位数")}
        card["metric"] = {"key": metric, "definition": spec["definition"], "version": spec["version"], "note": metric_note}
        card["platform_links"].append(f"platform://metrics?metric={metric}&market={s['market']}&scenario={s['scenario'] or ''}&date={D}")

        # step 1/2: traffic and metric series
        sess, rs = self._series(trace, "sessions", start, D, market=s["market"])
        if sess is None:
            return self._fail(card, trace, rs)
        if metric in ("precision_est", "recall_est"):
            return self._estimate_path(card, trace, s, metric, D)
        if spec["kind"] == "ratio":
            rser, rr = self._series(trace, metric, start, D, market=s["market"], scenario=s["scenario"])
            if rser is None:
                return self._fail(card, trace, rr)
            rcur, rbase = rser.get(D) or 0, self._baseline(rser, D)
            den = next((row["denominator"] for row in rr["rows"] if row["date"] == D), 0)
            card["evidence"].append({"ref_id": rr["ref_id"], "type": "metric", "metric": metric,
                                     "what": f"{D} {metric}={rcur}（分母 {den}），前 7 天中位数 {rbase:.3f}"})
            card["metric"]["note"] = (metric_note + "\n" if metric_note else "") + f"比率指标按分子 {spec['num']} 下钻，分母 {spec['den']} 作为对照"
            se = np.sqrt(max(rbase * (1 - rbase), 0.01) / max(den, 1))
            if abs(rcur - rbase) < 2 * se:
                card["metric"]["rate_prefix"] = f"按 {metric} 口径（{rcur:.3f} vs 基线 {rbase:.3f}）未见显著变化；按分子 {spec['num']} 绝对数看："
            metric = spec["num"]
            spec = METRICS[metric]
        ser, rm = self._series(trace, metric, start, D, market=s["market"], scenario=s["scenario"])
        if ser is None:
            return self._fail(card, trace, rm)
        cur, base = self._cur_base(ser, D, W)
        sc, sb = self._cur_base(sess, D, W)
        lab = f"截至 {D} 近 7 天合计" if W > 1 else D
        blab = "前 14 天日均×7" if W > 1 else "前 7 天中位数"
        card["evidence"].append({"ref_id": rm["ref_id"], "type": "metric", "metric": metric,
                                 "what": f"{lab} {metric}={cur}，{blab} {base:.1f}"})
        card["evidence"].append({"ref_id": rs["ref_id"], "type": "metric", "metric": "sessions",
                                 "what": f"{lab} sessions={sc}，{blab} {sb:.0f}"})
        small_market = (sb / W) < SMALL_MARKET_SESSIONS
        low_count = sum(v or 0 for v in ser.values()) < MIN_COUNT_FOR_CONCLUSION
        card["scope"]["low_sample"] = bool(small_market or low_count)
        card["scope"]["observed"] = {"current": cur, "baseline_median": round(base, 1), "sessions_current": sc,
                                     "sessions_baseline_median": round(sb, 1)}
        ratio = (cur / base) if base else (float("inf") if cur else 1.0)
        min_up, max_down = (1.3, 0.75) if W > 1 else (1.5, 0.7)
        up = cur > base + 2.5 * np.sqrt(max(base, 1)) + 2 and ratio >= min_up
        down = cur < base - 2.5 * np.sqrt(max(base, 1)) and ratio <= max_down
        if s.get("direction") and ((s["direction"] == "up" and down) or (s["direction"] == "down" and up)):
            card["metric"]["rate_prefix"] = (card["metric"].get("rate_prefix") or "") + \
                f"注意：你说的是{'上升' if s['direction'] == 'up' else '下降'}，数据显示 {D} 实际{'上升' if up else '下降'}（{base:.0f}→{cur}）。"
        if W == 1 and not up and not down and not small_market and not s.get("_no_window_fallback"):
            cur7, base7 = self._cur_base(ser, D, 7) if len(ser) >= 8 else (0, 0)
            ser21, r21 = self._series(trace, metric, (date.fromisoformat(D) - timedelta(days=20)).isoformat(), D,
                                      market=s["market"], scenario=s["scenario"])
            if ser21:
                cur7, base7 = self._cur_base(ser21, D, 7)
                # automatic fallback needs stronger evidence than an explicit "最近" question (false-alarm control)
                up7 = cur7 > base7 + 2.5 * np.sqrt(max(base7, 1)) + 2 and base7 and cur7 / base7 >= 1.45
                down7 = cur7 < base7 - 2.5 * np.sqrt(max(base7, 1)) and base7 and cur7 / base7 <= 0.7
                if up7 or down7:
                    card["evidence"].append({"ref_id": r21["ref_id"], "type": "metric", "metric": metric,
                                             "what": f"单日正常，但近 7 天合计 {cur7} vs 前 14 天日均×7 {base7:.0f}，持续{'上升' if up7 else '下降'}"})
                    inner = self.attribution(s["raw"], scope=dict(s, window_days=7, direction="up" if up7 else "down", _no_window_fallback=True))
                    inner.card["evidence"] = card["evidence"][:2] + [card["evidence"][-1]] + inner.card["evidence"]
                    inner.card["conclusion"] = f"{D} 单日（{cur} vs 基线 {base:.0f}）在正常范围，但近 7 天持续{'上升' if up7 else '下降'}（{cur7} vs {base7:.0f}）：" + inner.card["conclusion"]
                    inner.events = trace.events + inner.events
                    return inner
        if low_count and not small_market and not up and not down:
            card.update(primary_cause={"type": "no_anomaly", "strategy_ids": [], "change_ids": [],
                                       "explanation": f"当日 {cur} 与基线中位数 {base:.1f} 都很小，差异在 Poisson 噪声内"},
                        conclusion=f"{s['market']} {s['scenario'] or '全场景'} {spec['zh']} {D}={cur}，前 7 天中位数 {base:.1f}，计数小且在噪声范围内，未发现异常",
                        confidence="medium")
            card["scope"]["low_sample"] = False
            card["next_steps"].append("计数小的指标建议看 7 天滚动窗口")
            return trace.finish(Guardrails(self.p).apply(card, trace))
        if small_market:
            card.update(primary_cause={"type": "insufficient_data", "strategy_ids": [], "change_ids": [],
                                       "explanation": f"{s['market']} 日均 sessions≈{sb:.0f}，{metric} 当日 {cur}、基线中位数 {base:.1f}，"
                                                      f"计数太小，单日波动没有统计意义"},
                        conclusion=f"{s['market']} 样本不足，不能下归因结论；建议看 7 天滚动窗口，并与大市场同场景趋势对照",
                        confidence="low", needs_escalation=True,
                        escalation_reasons=["样本不足（小市场或计数过低），按规则升级给策略 owner 人工判断"])
            card["next_steps"] += [f"用 7 天滚动窗口重新看 {metric}", "对照大市场同场景是否有同样变化（例如同期模型升级）"]
            # still look for a global change to hint at
            ch = self.call(trace, "get_change_log", start_date=start, end_date=D, scenario=s["scenario"])
            if ch.get("count"):
                card["contributing_causes"].append({"type": "possible_global_change", "change_ids": [c["change_id"] for c in ch["changes"]],
                                                    "explanation": "同期存在全局变更，可能相关但本市场样本不足以证实"})
                card["evidence"].append({"ref_id": ch["ref_id"], "type": "change_log", "what": "同期变更记录"})
            return trace.finish(Guardrails(self.p).apply(card, trace))
        if not up and not down and not s["scenario"] and spec["level"] != "market":
            r = self.call(trace, "query_metrics", metric=metric, start_date=start, end_date=D, market=s["market"], group_by=["scenario"])
            if "error" not in r:
                contrib = self._contrib(r["rows"], "scenario", D, W)
                # candidate scenarios whose own series is anomalous; try up to 3, prefer one with an explained cause
                cands = [c for c in contrib[:4] if c[3] >= 8 and c[3] > c[4] + 2.5 * np.sqrt(max(c[4], 1)) + 2 and c[3] / max(c[4], 0.5) >= 2]
                best = None
                for c in cands[:3]:
                    inner = self.attribution(s["raw"], scope=dict(s, scenario=c[0], direction="up"))
                    ptype = inner.card.get("primary_cause", {}).get("type")
                    inner.events = trace.events + inner.events
                    if best is None:
                        best = (c, inner)
                    if ptype not in ("unexplained", "no_anomaly", "insufficient_data"):
                        best = (c, inner)
                        break
                if best:
                    c, inner = best
                    ev = {"ref_id": r["ref_id"], "type": "drilldown", "metric": metric,
                          "what": f"市场整体平稳，但按场景下钻 {c[0]} {metric} {c[4]}→{c[3]}（增量份额 {c[2]:.0%}）；候选场景 {[x[0] for x in cands[:3]]}"}
                    inner.card["evidence"] = card["evidence"][:2] + [ev] + inner.card["evidence"]
                    inner.card["scope"]["note"] = "市场整体在正常波动内；异常集中在单一场景"
                    inner.card["conclusion"] = f"{s['market']} 整体 {spec['zh']} {cur} vs 基线 {base:.0f} 在正常范围，但 {self.meta['scenario_names'][c[0]]} 场景异常：" + inner.card["conclusion"]
                    return inner
        if not up and not down:
            card.update(primary_cause={"type": "no_anomaly", "strategy_ids": [], "change_ids": [],
                                       "explanation": f"当日 {cur} 与基线中位数 {base:.1f} 的差在 2 个标准差内"},
                        conclusion=f"{s['market']} {s['scenario'] or '全场景'} {spec['zh']} {D}={cur}，前 7 天中位数 {base:.1f}，属正常波动，未发现异常",
                        confidence="high")
            card["next_steps"].append("如需追踪，可设置超出 2σ 自动告警")
            return trace.finish(Guardrails(self.p).apply(card, trace))

        # step 2: traffic vs rate decomposition
        traffic_share = 0.0
        if base and sb and cur and sc:
            dl_count, dl_sess = np.log(cur / base), np.log(sc / sb)
            traffic_share = float(np.clip(dl_sess / dl_count, 0, 1)) if dl_count and np.sign(dl_sess) == np.sign(dl_count) else 0.0
        rate_ratio = (cur / sc) / (base / sb) if (sc and sb and base) else None
        card["scope"]["decomposition"] = {"count_ratio": round(ratio, 2), "traffic_ratio": round(sc / sb, 2) if sb else None,
                                          "rate_ratio": round(rate_ratio, 2) if rate_ratio else None,
                                          "traffic_share_of_change": round(traffic_share, 2)}
        if traffic_share >= 0.6 and rate_ratio is not None and 0.8 <= rate_ratio <= 1.25:
            card.update(primary_cause={"type": "traffic_driven", "strategy_ids": [], "change_ids": [],
                                       "explanation": f"sessions {sb:.0f}→{sc}（×{sc / sb:.2f}），{metric}/sessions 比率 ×{rate_ratio:.2f} 基本不变；"
                                                      f"流量解释了约 {traffic_share:.0%} 的变化"},
                        conclusion=f"{s['market']} {spec['zh']} {('上涨' if up else '下降')}主要由流量变化驱动（{sb:.0f}→{sc} sessions），"
                                   f"{spec['zh']}率基本不变，不是策略或模型问题",
                        confidence="high")
            card["next_steps"] += ["看板增加按 sessions 归一化的率指标，避免再次误判", "确认流量变化来源（活动/假期），无需策略动作"]
            return trace.finish(Guardrails(self.p).apply(card, trace))

        # step 3: drill down scenario -> strategy
        scenario = s["scenario"]
        if not scenario and spec["level"] != "market":
            r = self.call(trace, "query_metrics", metric=metric, start_date=start, end_date=D, market=s["market"], group_by=["scenario"])
            if "error" not in r:
                contrib = self._contrib(r["rows"], "scenario", D, W)
                card["evidence"].append({"ref_id": r["ref_id"], "type": "drilldown", "metric": metric, "what": f"按场景增量贡献: {contrib[:3]}"})
                if contrib and contrib[0][2] >= 0.4:
                    scenario = contrib[0][0]
                    card["scope"]["drill_scenario"] = scenario
        top_strategy, strat_contrib, r_st = None, [], None
        if spec["level"] == "strategy":
            r_st = self.call(trace, "query_metrics", metric=metric, start_date=start, end_date=D, market=s["market"],
                             scenario=scenario, group_by=["strategy_id"])
            if "error" not in r_st:
                strat_contrib = self._contrib(r_st["rows"], "strategy_id", D, W)
                card["evidence"].append({"ref_id": r_st["ref_id"], "type": "drilldown", "metric": metric,
                                         "what": f"按策略增量贡献（strategy_id, 增量, 份额）: {strat_contrib[:3]}"})
                if strat_contrib and abs(strat_contrib[0][1]) > 0:
                    top_strategy = strat_contrib[0][0]
        # step 4: change log
        ch = self.call(trace, "get_change_log", start_date=start, end_date=D, scenario=scenario, market=s["market"])
        changes = ch.get("changes", [])
        if changes:
            card["evidence"].append({"ref_id": ch["ref_id"], "type": "change_log",
                                     "what": "; ".join(f"{c['change_id']} {c['date']} {c['type']} {c['target']}" for c in changes)})
        model_changes = [c for c in changes if c["type"] == "model_update"]
        strat_changes = [c for c in changes if c["type"] in ("strategy_update", "strategy_launch")]
        policy_changes = [c for c in changes if c["type"] == "policy_update"]
        top_cfg = self.p.by_id.get(top_strategy) if top_strategy else None

        # --- recall-drop path (metric went down)
        if down and metric in ("hits", "enforcements", "manual_reviews") and scenario:
            rep, rr = self._series(trace, "reports", start, D, market=s["market"], scenario=scenario)
            rep_cur, rep_base = self._cur_base(rep, D, W) if rep else (0, 0)
            win_b0 = (date.fromisoformat(D) - timedelta(days=20)).isoformat()
            dr = self.call(trace, "score_drift", scenario=scenario, market=s["market"], baseline_start=win_b0, baseline_end=start,
                           current_start=(date.fromisoformat(D) - timedelta(days=6)).isoformat(), current_end=D)
            rp = dr.get("reported_population", {}) if "error" not in dr else {}
            rb, rc = rp.get("baseline", {}).get("p50"), rp.get("current", {}).get("p50")
            rec = self.call(trace, "query_metrics", metric="recall_est", start_date=(date.fromisoformat(D) - timedelta(days=6)).isoformat(),
                            end_date=D, market=s["market"], scenario=scenario, granularity="total")
            rec0 = self.call(trace, "query_metrics", metric="recall_est", start_date=win_b0, end_date=start, market=s["market"],
                             scenario=scenario, granularity="total")
            rv = rec["rows"][0]["value"] if "error" not in rec and rec["rows"] else None
            rv0 = rec0["rows"][0]["value"] if "error" not in rec0 and rec0["rows"] else None
            if rep and rep_cur >= 0.8 * rep_base and rb and rc and rc < rb - 0.1:
                sid = top_strategy or ""
                card["evidence"] += [{"ref_id": rr["ref_id"], "type": "metric", "metric": "reports", "what": f"举报 {D}={rep_cur}，基线 {rep_base:.0f}，未同步下降"},
                                     {"ref_id": dr["ref_id"], "type": "score_drift", "what": f"被举报人群 {dr['model']} 分数中位数 {rb}→{rc}"},
                                     {"ref_id": rec["ref_id"], "type": "metric", "metric": "recall_est", "what": f"recall_est 近 7 天 {rv}，此前 {rv0}"}]
                card.update(primary_cause={"type": "recall_drop_evasion", "strategy_ids": [sid] if sid else [], "change_ids": [],
                                           "explanation": f"命中下降但举报不降；被举报 session 的模型分整体下移（p50 {rb}→{rc}），"
                                                          f"固定阈值策略 {sid} 捕获不到，疑似对抗绕过，不是违规减少"},
                            conclusion=f"{s['market']} {self.meta['scenario_names'][scenario]} {spec['zh']}下降不是违规变少：举报持平、"
                                       f"被举报内容的模型分下移，recall_est {rv0}→{rv}，判断为绕过导致召回下降（主要影响 {sid}）",
                            confidence="medium")
                card["next_steps"] += ["把近 7 天被举报但未命中的 case 送模型团队做 bad case 分析", f"评估 {sid} 临时补充特征条件或下调阈值（先回测）"]
                card["platform_links"].append(f"platform://strategy/{sid}")
                return trace.finish(Guardrails(self.p).apply(card, trace))

        # --- model drift path
        if model_changes and top_cfg and top_cfg["scenario"] == scenario:
            mc = model_changes[0]
            dr = self.call(trace, "score_drift", scenario=scenario, market=None,
                           baseline_start=(date.fromisoformat(mc["date"]) - timedelta(days=7)).isoformat(),
                           baseline_end=(date.fromisoformat(mc["date"]) - timedelta(days=1)).isoformat(),
                           current_start=mc["date"], current_end=D)
            if "error" not in dr:
                th = next((t for t in dr["share_above_thresholds"] if t["strategy_id"] == top_strategy), None)
                if dr["psi"] > 0.1 and th and th["ratio"] and th["ratio"] >= 1.5:
                    cases = self.call(trace, "sample_cases", start_date=mc["date"], end_date=D, strategy_id=top_strategy,
                                      market=s["market"], n=15, overturned=True if metric in ("overturns", "appeals") else None)
                    prec = self.call(trace, "query_metrics", metric="precision_est", start_date=mc["date"], end_date=D,
                                     strategy_id=top_strategy, granularity="total")
                    prec0 = self.call(trace, "query_metrics", metric="precision_est", start_date=start,
                                      end_date=(date.fromisoformat(mc["date"]) - timedelta(days=1)).isoformat(),
                                      strategy_id=top_strategy, granularity="total")
                    pv = prec["rows"][0]["value"] if "error" not in prec else None
                    pv0 = prec0["rows"][0]["value"] if "error" not in prec0 else None
                    card["evidence"] += [{"ref_id": dr["ref_id"], "type": "score_drift",
                                          "what": f"{dr['model']} {dr['baseline']['model_versions']}→{dr['current']['model_versions']}，PSI={dr['psi']}，"
                                                  f"p50 {dr['baseline']['p50']}→{dr['current']['p50']}；超过 {top_strategy} 阈值 {th['threshold']} 的比例 "
                                                  f"{th['share_above_baseline']}→{th['share_above_current']}（×{th['ratio']}）"},
                                         {"ref_id": prec["ref_id"], "type": "metric", "metric": "precision_est", "what": f"{top_strategy} 抽检精确率 {pv0}→{pv}"}]
                    if "error" not in cases:
                        card["evidence"].append({"ref_id": cases["ref_id"], "type": "cases", "what": f"抽样 {cases['n']} 个 case，分数均值 {cases['score_summary']['mean']}，"
                                                                                              f"含头部主播={cases['contains_head_host']}"})
                        card["sample_case_ids"] = [c["session_id"] for c in cases["cases"]]
                    card.update(primary_cause={"type": "model_drift_threshold", "strategy_ids": [top_strategy], "change_ids": [mc["change_id"]],
                                               "explanation": f"{mc['change_id']}（{mc['summary']}）后分数分布上移，{top_strategy} 固定阈值 {th['threshold']} 命中增加，误杀上升"},
                                conclusion=f"{s['market']} {spec['zh']}上升（{base:.0f}→{cur}）主要来自 {top_strategy}；{mc['date']} {mc['target']} 升级后分数中位数上移，"
                                           f"固定阈值导致命中 ×{th['ratio']}，抽检精确率 {pv0}→{pv}。建议策略 owner 评估阈值重校",
                                confidence="high")
                    card["next_steps"] += [f"请 {top_cfg['owner']} 评估 {top_strategy} 阈值重校（可用 run_backtest 回放新阈值）",
                                           "与算法确认 v4 是否提供校准后分数；长期应让阈值随版本自动重校"]
                    fin = self.call(trace, "query_metrics", metric="frozen_amount", start_date=mc["date"], end_date=D, market=s["market"],
                                    scenario=scenario, group_by=["strategy_id"], granularity="total")
                    if "error" not in fin:
                        fin_rows = [r for r in fin["rows"] if r["value"] > 0 and self.p.by_id[r["strategy_id"]]["action"] in self.meta["financial_actions"]]
                        if fin_rows:
                            card["evidence"].append({"ref_id": fin["ref_id"], "type": "metric", "metric": "frozen_amount",
                                                     "what": "同一模型的资金侧策略同期冻结/延迟结算金额: " + ", ".join(f"{r['strategy_id']}={r['value']}" for r in fin_rows)})
                            card["next_steps"].append("资金侧：同一模型的 " + "、".join(r["strategy_id"] for r in fin_rows) + " 受同样漂移影响，误杀场次的冻结收益需复核后解冻（走结算复核）")
                            card["contributing_causes"].append({"type": "financial_exposure", "strategy_ids": [r["strategy_id"] for r in fin_rows], "change_ids": [],
                                                                "explanation": "资金侧动作误杀的代价不可逆性更高，需同步处理"})
                    card["platform_links"].append(f"platform://strategy/{top_strategy}")
                    if strat_changes or policy_changes:
                        card["contributing_causes"].append({"type": "other_changes", "change_ids": [c["change_id"] for c in strat_changes + policy_changes],
                                                            "explanation": "同期还有其他变更，但贡献小于模型升级"})
                    return trace.finish(Guardrails(self.p).apply(card, trace))
        # --- strategy launch / update / policy path
        related = [c for c in strat_changes if (c["strategy_id"] == top_strategy) or
                   (c["strategy_id"] and c["strategy_id"] in [x[0] for x in strat_contrib[:3]])]
        if related:
            c0 = related[0]
            sid = c0["strategy_id"]
            cfg = self.call(trace, "get_strategy_config", strategy_id=sid)
            vers = cfg["strategies"][0]["versions"] if "error" not in cfg else []
            card["evidence"].append({"ref_id": cfg["ref_id"], "type": "strategy_config", "what": f"{sid} 版本历史: " + "; ".join(
                f"{v['version']}@{v['effective_from']} thr={v['threshold']} {v['action']}" for v in vers[-3:])})
            contrib_share = next((x[2] for x in strat_contrib if x[0] == sid), 0)
            pol = [c for c in policy_changes if c["scenario"] == c0["scenario"]]
            if c0["type"] == "strategy_launch":
                ptype, expl = "new_strategy_launch", f"{sid} 于 {c0['date']} 上线（{c0['change_id']}），贡献了约 {contrib_share:.0%} 的增量"
                concl = f"{s['market']} {spec['zh']}上升主要来自新上线策略 {sid}（{c0['date']}，{c0['change_id']}），属预期内新增量；建议按上线计划观察精确率"
            elif pol:
                ptype, expl = "policy_update", f"policy {pol[0]['target']} 更新（{pol[0]['change_id']}）后 {sid} 同步调整版本（{c0['change_id']}），增量约 {contrib_share:.0%}"
                concl = f"{s['market']} {spec['zh']}上升是 policy {pol[0]['target']} 更新后 {sid} 阈值调整的预期结果（{pol[0]['change_id']} → {c0['change_id']}），不是异常"
            else:
                ptype, expl = "strategy_change", f"{sid} 版本变更（{c0['change_id']}）贡献约 {contrib_share:.0%} 的增量"
                concl = f"{s['market']} {spec['zh']}变化主要来自 {sid} 的版本变更（{c0['change_id']}）"
            prec = self.call(trace, "query_metrics", metric="precision_est", start_date=c0["date"], end_date=D, strategy_id=sid, granularity="total")
            if "error" not in prec and prec["rows"]:
                card["evidence"].append({"ref_id": prec["ref_id"], "type": "metric", "metric": "precision_est",
                                         "what": f"{sid} 变更后抽检精确率 {prec['rows'][0]['value']}（n={prec['rows'][0]['n']}）"})
            card.update(primary_cause={"type": ptype, "strategy_ids": [sid], "change_ids": [c["change_id"] for c in ([pol[0]] if pol else []) + [c0]],
                                       "explanation": expl}, conclusion=concl, confidence="high" if contrib_share >= 0.5 else "medium")
            if contrib_share < 0.5:
                card["contributing_causes"].append({"type": "other_strategies", "strategy_ids": [x[0] for x in strat_contrib[1:3]],
                                                    "explanation": "其余增量分散在其他策略"})
            card["next_steps"] += [f"确认 {sid} 变更后的精确率是否达到上线门槛", "若属预期变化，在看板标注变更日期避免重复归因"]
            card["platform_links"].append(f"platform://strategy/{sid}")
            return trace.finish(Guardrails(self.p).apply(card, trace))
        # --- guild concentration path: is the increase concentrated in one guild?
        if up and metric in ("hits", "enforcements", "manual_reviews", "appeals", "overturns"):
            rg = self.call(trace, "query_metrics", metric=metric, start_date=start, end_date=D, market=s["market"], scenario=scenario,
                           group_by=["guild_id"])
            if "error" not in rg:
                gc = self._contrib(rg["rows"], "guild_id", D, W)
                top_g = gc[0] if gc else None
                if top_g and top_g[0] not in ("(无公会)", "") and top_g[2] >= 0.5 and top_g[3] >= 8 and (top_g[4] == 0 or top_g[3] / top_g[4] >= 2):
                    gid = top_g[0]
                    gi = self.call(trace, "get_guild_info", guild_id=gid)
                    ginfo = gi["guilds"][0] if "error" not in gi else {}
                    cases = self.call(trace, "sample_cases", start_date=D if W == 1 else (date.fromisoformat(D) - timedelta(days=W - 1)).isoformat(),
                                      end_date=D, market=s["market"], scenario=scenario, guild_id=gid, n=15)
                    card["evidence"].append({"ref_id": rg["ref_id"], "type": "drilldown", "metric": metric,
                                             "what": f"按公会增量贡献: {gid} {top_g[4]}→{top_g[3]}（增量份额 {top_g[2]:.0%}）；其余公会基本持平"})
                    if "error" not in gi:
                        card["evidence"].append({"ref_id": gi["ref_id"], "type": "guild", "what": f"{gid} {ginfo.get('name')}，{ginfo.get('tier')} 公会，近 28 天 {ginfo.get('sessions_28d')} 场，头部主播 {ginfo.get('head_hosts')} 个"})
                    if "error" not in cases:
                        card["evidence"].append({"ref_id": cases["ref_id"], "type": "cases", "what": f"抽样 {cases['n']} 个 case（新设备占比高、账号年龄小），含头部主播={cases['contains_head_host']}"})
                        card["sample_case_ids"] = [c["session_id"] for c in cases["cases"]]
                    card.update(primary_cause={"type": "guild_concentration", "strategy_ids": [top_strategy] if top_strategy else [], "change_ids": [],
                                               "guild_ids": [gid],
                                               "explanation": f"同期无策略/模型/policy 变更；增量 {top_g[2]:.0%} 集中在公会 {gid}，其他公会平稳，更像该公会主播的集中违规而不是策略问题"},
                                conclusion=f"{s['market']} {self.meta['scenario_names'][scenario] if scenario else ''} {spec['zh']}上升（{base:.0f}→{cur}）集中在公会 {gid}（{ginfo.get('name', '')}，贡献 {top_g[2]:.0%}），"
                                           f"命中策略 {top_strategy or '多条'} 的分数和特征正常，不是误杀；建议公会治理跟进该公会",
                                confidence="medium")
                    card["next_steps"] += [f"公会治理：核查 {gid} 近 3 天新增主播和设备，必要时按公会维度处置",
                                           "策略侧无需改动；如果是新型批量行为，把 case 送策略团队评估是否需要公会级特征"]
                    card["platform_links"].append(f"platform://guild/{gid}")
                    card["needs_escalation"] = True
                    card["escalation_reasons"] = ["涉及公会级处置，超出只读结论范围，需公会治理确认"]
                    return trace.finish(Guardrails(self.p).apply(card, trace))
        # --- unexplained
        card.update(primary_cause={"type": "unexplained", "strategy_ids": [top_strategy] if top_strategy else [], "change_ids": [],
                                   "explanation": "流量不能解释，同期也没有可匹配的策略/模型/policy 变更"},
                    conclusion=f"{s['market']} {spec['zh']} {D} 异常（{base:.0f}→{cur}）无法用流量或已记录变更解释；最大贡献来自 {top_strategy or '未定位'}，需人工介入",
                    confidence="low", needs_escalation=True)
        card["next_steps"] += ["检查是否有未登记的变更（灰度、特征管线）", "抽样 case 人工复核"]
        return trace.finish(Guardrails(self.p).apply(card, trace))

    def _estimate_path(self, card, trace, s, metric, D):
        d = date.fromisoformat(D)
        cur = self.call(trace, "query_metrics", metric=metric, start_date=(d - timedelta(days=6)).isoformat(), end_date=D,
                        market=s["market"], scenario=s["scenario"], granularity="total")
        prev = self.call(trace, "query_metrics", metric=metric, start_date=(d - timedelta(days=20)).isoformat(),
                         end_date=(d - timedelta(days=7)).isoformat(), market=s["market"], scenario=s["scenario"], granularity="total")
        if "error" in cur or "error" in prev or not cur["rows"] or not prev["rows"]:
            return self._fail(card, trace, cur if "error" in cur else prev)
        v1, v0 = cur["rows"][0]["value"], prev["rows"][0]["value"]
        low = cur["rows"][0].get("low_sample") or prev["rows"][0].get("low_sample")
        card["scope"]["low_sample"] = bool(low)
        card["evidence"] += [{"ref_id": cur["ref_id"], "type": "metric", "metric": metric, "what": f"近 7 天 {metric}={v1}"},
                             {"ref_id": prev["ref_id"], "type": "metric", "metric": metric, "what": f"前 14 天 {metric}={v0}"}]
        if v1 is not None and v0 is not None and v0 - v1 >= 0.15 and metric == "recall_est":
            s2 = dict(s, metric="hits", direction="down")
            inner = self.attribution(s["raw"], scope=s2)
            inner.card["scope"]["metric"], inner.card["metric"] = metric, {"key": metric, "definition": METRICS[metric]["definition"], "version": METRICS[metric]["version"], "note": None}
            inner.card["evidence"] = card["evidence"] + inner.card["evidence"]
            inner.events = trace.events + inner.events
            return inner
        card.update(primary_cause={"type": "no_anomaly" if (v1 is None or v0 is None or abs(v1 - v0) < 0.15) else "unexplained",
                                   "strategy_ids": [], "change_ids": [], "explanation": f"{metric} {v0}→{v1}"},
                    conclusion=f"{s['market']} {s['scenario'] or ''} {metric} 近 7 天 {v1}，前 14 天 {v0}", confidence="medium")
        return trace.finish(Guardrails(self.p).apply(card, trace))

    def _fail(self, card, trace, err):
        card.update(primary_cause={"type": "insufficient_data", "strategy_ids": [], "change_ids": [], "explanation": err.get("message", "")},
                    conclusion=f"数据缺失，无法归因：{err.get('message')}。{err.get('hint') or ''}", confidence="low", needs_escalation=True,
                    escalation_reasons=[f"工具返回 {err.get('error')}"])
        return trace.finish(Guardrails(self.p).apply(card, trace))

    @staticmethod
    def _cur_base(ser, D, W=1):
        """Current value and baseline. W=1: day vs median of prior days. W>1: last W days sum vs prior-days mean × W."""
        if W == 1:
            vals = [v for d, v in ser.items() if d < D and v is not None]
            return (ser.get(D) or 0), (float(np.median(vals)) if vals else 0.0)
        lo = (date.fromisoformat(D) - timedelta(days=W - 1)).isoformat()
        cur = sum((v or 0) for d, v in ser.items() if lo <= d <= D)
        prior = [v or 0 for d, v in ser.items() if d < lo]
        return cur, (float(np.mean(prior)) * W if prior else 0.0)

    @staticmethod
    def _contrib(rows, key, D, W=1):
        by = {}
        for r in rows:
            by.setdefault(r[key], {})[r["date"]] = r["value"] or 0
        out = []
        for k, ser in by.items():
            cur, base = RulesEngine._cur_base(ser, D, W)
            out.append((k, cur - base, cur, base))
        tot = sum(abs(x[1]) for x in out) or 1
        out = [(k, round(dlt, 1), round(abs(dlt) / tot, 2), cur, round(base, 1)) for k, dlt, cur, base in out]
        return sorted(out, key=lambda x: -abs(x[1]))

    # --- guild health score attribution: real violations vs unfair deductions
    def guild_health(self, query, gid, D):
        trace = Trace("rules", "guild_health", query)
        d = date.fromisoformat(D)
        start = (d - timedelta(days=13)).isoformat()
        card = {"card_type": "attribution", "query": query, "scope": {"guild_id": gid, "date": D, "window": [start, D], "low_sample": False},
                "metric": {"key": "guild_health_score", "definition": self.meta["guild_health_rule"], "version": "v1", "note": "机构健康分（合成规则）"},
                "evidence": [], "next_steps": [], "contributing_causes": [], "needs_escalation": False, "escalation_reasons": [],
                "platform_links": [f"platform://guild/{gid}/health"]}
        gh = self.call(trace, "get_guild_health", start_date=start, end_date=D, guild_id=gid)
        if "error" in gh:
            return self._fail(card, trace, gh)
        drop = gh["score_start"] - gh["score_end"]
        card["evidence"].append({"ref_id": gh["ref_id"], "type": "guild_health",
                                 "what": f"{gid} 健康分 {gh['score_start']}→{gh['score_end']}（{gh['tier_now']}），窗口扣分 {gh['total_deduction']}，"
                                         f"最大扣分日 {max(gh['deductions_by_date'], key=gh['deductions_by_date'].get) if gh['deductions_by_date'] else '无'}"})
        if drop < 3 and gh["tier_now"] == "正常":
            card.update(primary_cause={"type": "no_anomaly", "strategy_ids": [], "change_ids": [], "guild_ids": [gid],
                                       "explanation": f"净变化 {drop:.1f} 分，扣分与日常恢复相抵，未越档"},
                        conclusion=f"{gid} 近 14 天健康分 {gh['score_start']}→{gh['score_end']}（{gh['tier_now']}），扣分 {gh['total_deduction']}、恢复 {gh['total_recovery']}，属正常波动", confidence="high")
            return trace.finish(Guardrails(self.p).apply(card, trace))
        by = gh["deductions_by_strategy"]
        top = by[0]
        share = top["points"] / max(gh["total_deduction"], 0.01)
        card["evidence"].append({"ref_id": gh["ref_id"], "type": "drilldown", "what": "扣分来源（策略, 分值）: " + ", ".join(f"{b['strategy_id']} {b['points']}" for b in by[:4])})
        ch = self.call(trace, "get_change_log", start_date=start, end_date=D)
        model_changes = [c for c in ch.get("changes", []) if c["type"] == "model_update"]
        unfair = None
        for cand in by[:3]:   # any top-3 contributor driven by a drifting model with >= 25% of the deduction
            if cand["points"] / max(gh["total_deduction"], 0.01) < 0.25:
                continue
            mcs = [c for c in model_changes if c["scenario"] == cand["scenario"]]
            if not mcs:
                continue
            mc = mcs[0]
            dr = self.call(trace, "score_drift", scenario=cand["scenario"], baseline_start=(date.fromisoformat(mc["date"]) - timedelta(days=7)).isoformat(),
                           baseline_end=(date.fromisoformat(mc["date"]) - timedelta(days=1)).isoformat(), current_start=mc["date"], current_end=D)
            th = next((t for t in dr.get("share_above_thresholds", []) if t["strategy_id"] == cand["strategy_id"]), None)
            if "error" not in dr and dr["psi"] > 0.1 and th and th["ratio"] and th["ratio"] >= 1.5:
                unfair = (mc, dr, th)
                top = cand
                share = cand["points"] / max(gh["total_deduction"], 0.01)
                break
        cases = self.call(trace, "sample_cases", start_date=start, end_date=D, strategy_id=top["strategy_id"], guild_id=gid, n=10)
        if "error" not in cases:
            card["evidence"].append({"ref_id": cases["ref_id"], "type": "cases", "what": f"{gid} 被 {top['strategy_id']} 命中的 case 抽样 {cases['n']} 个，含头部主播={cases['contains_head_host']}"})
            card["sample_case_ids"] = [c["session_id"] for c in cases["cases"]]
        if unfair:
            mc, dr, th = unfair
            card["evidence"] += [{"ref_id": ch["ref_id"], "type": "change_log", "what": f"{mc['change_id']} {mc['date']} {mc['summary']}"},
                                 {"ref_id": dr["ref_id"], "type": "score_drift", "what": f"{dr['model']} PSI={dr['psi']}，超过 {top['strategy_id']} 阈值的比例 ×{th['ratio']}"}]
            card.update(primary_cause={"type": "guild_health_unfair_deduction", "strategy_ids": [top["strategy_id"]], "change_ids": [mc["change_id"]], "guild_ids": [gid],
                                       "explanation": f"扣分 {share:.0%} 来自 {top['strategy_id']}，而该策略正受 {mc['change_id']} 模型漂移影响（命中 ×{th['ratio']}），扣分可能不反映真实违规"},
                        conclusion=f"{gid} 健康分下降（{gh['score_start']}→{gh['score_end']}）主要由 {top['strategy_id']} 的严重处置驱动；该策略在 {mc['date']} 模型升级后命中 ×{th['ratio']}，"
                                   f"扣分可能不合理，建议暂缓按健康分处置并在阈值重校后回补分数",
                        confidence="medium", needs_escalation=True,
                        escalation_reasons=["机构健康分扣分可能源于策略误杀，涉及公会级处置与分数回补，需公会治理 + 策略 owner 共同确认"])
            card["next_steps"] += [f"策略侧：{top['strategy_id']} 阈值重校（见归因卡片）", f"公会侧：冻结 {gid} 的档位处置直到复核完成；复核后回补误扣分数"]
        else:
            gi = self.call(trace, "get_guild_info", guild_id=gid)
            card.update(primary_cause={"type": "guild_health_real_violations", "strategy_ids": [top["strategy_id"]], "change_ids": [], "guild_ids": [gid],
                                       "explanation": f"扣分 {share:.0%} 来自 {top['strategy_id']}（{self.meta['scenario_names'][top['scenario']]}），同期无模型/策略变更，"
                                                      f"严重处置 {top['severe_enforcements']} 次集中出现，是旗下主播的真实违规"},
                        conclusion=f"{gid} 健康分下降（{gh['score_start']}→{gh['score_end']}，{gh['tier_now']}）来自旗下主播的真实违规：{top['strategy_id']} "
                                   f"严重处置 {top['severe_enforcements']} 次，占扣分 {share:.0%}；下一档 {gh['next_threshold']['action'] if gh['next_threshold'] else '无'}（{gh['next_threshold']['score'] if gh['next_threshold'] else ''} 分）",
                        confidence="high", needs_escalation=True,
                        escalation_reasons=["涉及公会级处置（约谈/暂停结算），需公会治理确认"])
            card["next_steps"] += [f"公会治理：约谈 {gid}，要求整改并核查近 3 天新增主播", "若连续扣分，将在下一档触发暂停结算；提前告知公会"]
            if "error" not in gi:
                card["evidence"].append({"ref_id": gi["ref_id"], "type": "guild", "what": f"{gi['guilds'][0]['name']}，{gi['guilds'][0]['tier']} 公会，头部主播 {gi['guilds'][0]['head_hosts']} 个"})
        return trace.finish(Guardrails(self.p).apply(card, trace))

    # --- case evidence pack
    def case(self, query, session_id):
        trace = Trace("rules", "case", query)
        r = self.call(trace, "reconstruct_case", session_id=session_id)
        card = {"card_type": "case", "query": query, "scope": {"session_id": session_id}, "evidence": [], "next_steps": [],
                "contributing_causes": [], "needs_escalation": False, "escalation_reasons": [], "platform_links": []}
        if "error" in r:
            return self._fail(card, trace, r)
        hits = [c for c in r["chain"] if c["step"] == "strategy_hit"]
        enf = next((c for c in r["chain"] if c["step"] == "enforcement"), None)
        ap = next((c for c in r["chain"] if c["step"] == "appeal"), None)
        card["evidence"].append({"ref_id": r["ref_id"], "type": "case", "what": f"命中 {[h['strategy_id'] for h in hits]}；处罚 {enf['final_action'] if enf else '无'}；申诉 {ap['result'] if ap else '无'}"})
        for c in r["related_changes"]:
            card["evidence"].append({"ref_id": r["ref_id"], "type": "change_log", "what": f"{c['change_id']} {c['date']} {c['summary']}"})
        card["case"] = r
        card["conclusion"] = (f"{session_id}（{r['session']['market']}，{r['session']['host_tier']} 主播）{r['session']['date']} 命中 "
                              + "、".join(f"{h['strategy_id']}({h['model_version']} 分 {h['score']:.3f} ≥ {h['threshold']})" for h in hits)
                              + (f"，最终 {enf['final_action']}" if enf else "，未处罚") + (f"，申诉{ap['result']}" if ap else ""))
        card["primary_cause"] = {"type": "case_reconstruction", "strategy_ids": sorted({h["strategy_id"] for h in hits}),
                                 "change_ids": [c["change_id"] for c in r["related_changes"]], "explanation": "; ".join(r["escalation_flags"]) or "无特殊标记"}
        card["confidence"] = "high"
        card["escalation_reasons"] = [f for f in r["escalation_flags"] if "head_host" in f]
        card["needs_escalation"] = bool(card["escalation_reasons"])
        card["platform_links"].append(r["platform_link"])
        card["next_steps"] = ["如申诉已推翻且有同期变更，纳入该策略的误杀复盘样本"]
        return trace.finish(Guardrails(self.p).apply(card, trace))

    # --- health check
    def health_check(self, query="策略健康巡检", strategy_ids=None, as_of=None):
        trace = Trace("rules", "health_check", query)
        D = as_of or self.meta["end"]
        d = date.fromisoformat(D)
        w7, w21 = (d - timedelta(days=6)).isoformat(), (d - timedelta(days=20)).isoformat()
        w7_prev_end = (d - timedelta(days=7)).isoformat()
        cfg = self.call(trace, "get_strategy_config", status="active")
        strategies = [s for s in cfg["strategies"] if not strategy_ids or s["strategy_id"] in strategy_ids]
        inactive = []
        if strategy_ids:
            for sid in strategy_ids:
                if sid not in {s["strategy_id"] for s in strategies}:
                    one = self.call(trace, "get_strategy_config", strategy_id=sid)
                    if "error" not in one:
                        inactive += [x for x in one["strategies"] if x["status"] != "active"]
        cur = self.call(trace, "query_metrics", metric="hits", start_date=w7, end_date=D, group_by=["strategy_id"], granularity="total")
        prev = self.call(trace, "query_metrics", metric="hits", start_date=w21, end_date=w7_prev_end, group_by=["strategy_id"], granularity="total")
        full = self.call(trace, "query_metrics", metric="hits", start_date=self.meta["start"], end_date=D, group_by=["strategy_id"], granularity="total")
        hc = {r["strategy_id"]: r["value"] for r in cur["rows"]}
        hp = {r["strategy_id"]: r["value"] for r in prev["rows"]}
        hf = {r["strategy_id"]: r["value"] for r in full["rows"]}
        changes = self.call(trace, "get_change_log", start_date=w21, end_date=D)["changes"]
        feat = self.call(trace, "get_change_log", start_date="2026-01-01", end_date=D, change_type="feature_pipeline")["changes"]
        findings, seen_pairs = [], set()
        for s in strategies:
            sid = s["strategy_id"]
            cur7, prev14 = hc.get(sid, 0) / 7, hp.get(sid, 0) / 14
            days_live = (d - date.fromisoformat(s["created_at"])).days + 1
            if hf.get(sid, 0) == 0 and days_live >= 14:
                dep = [f for f in feat if any(c["field"] == f["target"] for c in s["extra_conditions"])]
                findings.append({"finding_type": "zombie_strategy", "strategy_ids": [sid], "severity": "medium", "market": "ALL",
                                 "evidence": [full["ref_id"]] + [f["change_id"] for f in dep],
                                 "explanation": f"{sid} 近 28 天 0 命中" + (f"；依赖已下线特征 {dep[0]['target']}（{dep[0]['change_id']}）" if dep else "；条件过窄"),
                                 "suggested_action": "迁移特征或下线策略" if dep else "评估是否下线或放宽条件", "needs_human": True})
            elif prev14 >= 5 and cur7 <= 0.5 * prev14:
                findings.append({"finding_type": "effect_decline", "strategy_ids": [sid], "severity": "medium", "market": "ALL",
                                 "evidence": [cur["ref_id"], prev["ref_id"]],
                                 "explanation": f"{sid} 日均命中 {prev14:.1f} → {cur7:.1f}（-{1 - cur7 / prev14:.0%}）",
                                 "suggested_action": "按市场拆分看召回是否下降（见 recall_decline）", "needs_human": True})
            # overlap
            if hf.get(sid, 0) >= 50:
                ov = self.call(trace, "get_hit_logs", start_date=w21, end_date=D, strategy_id=sid, overlap_analysis=True, limit=1)
                for o in ov.get("overlap", []):
                    pair = tuple(sorted([sid, o["strategy_id"]]))
                    if o["share_of_this"] >= 0.8 and o["share_of_other"] >= 0.8 and o["same_action"] and pair not in seen_pairs:
                        seen_pairs.add(pair)
                        owners = {self.p.by_id[p]["owner"] for p in pair}
                        findings.append({"finding_type": "strategy_overlap", "strategy_ids": list(pair), "severity": "high", "market": "ALL",
                                         "evidence": [ov["ref_id"]],
                                         "explanation": f"{pair[0]} 与 {pair[1]} 命中重叠 {o['share_of_this']:.0%}/{o['share_of_other']:.0%}，动作相同（{s['action']}），同一 session 被重复处置",
                                         "suggested_action": f"两位 owner（{'、'.join(sorted(owners))}）确认保留哪一条，另一条下线", "needs_human": True})
        # drift per model change
        for c in [c for c in changes if c["type"] == "model_update"]:
            dr = self.call(trace, "score_drift", scenario=c["scenario"], baseline_start=(date.fromisoformat(c["date"]) - timedelta(days=7)).isoformat(),
                           baseline_end=(date.fromisoformat(c["date"]) - timedelta(days=1)).isoformat(), current_start=c["date"], current_end=D)
            if "error" in dr or dr["psi"] <= 0.1:
                continue
            for t in dr["share_above_thresholds"]:
                if t["ratio"] and t["ratio"] >= 1.5 and (not strategy_ids or t["strategy_id"] in strategy_ids):
                    prec = self.call(trace, "query_metrics", metric="precision_est", start_date=c["date"], end_date=D, strategy_id=t["strategy_id"], granularity="total")
                    prec0 = self.call(trace, "query_metrics", metric="precision_est", start_date=w21, end_date=(date.fromisoformat(c["date"]) - timedelta(days=1)).isoformat(),
                                      strategy_id=t["strategy_id"], granularity="total")
                    pv = prec["rows"][0]["value"] if "error" not in prec else None
                    pv0 = prec0["rows"][0]["value"] if "error" not in prec0 else None
                    worse = (pv is not None and pv0 is not None and pv0 - pv >= 0.1)
                    sev = ("high" if self.meta["action_severity"][t["action"]] >= 5 else "medium") if worse else "low"
                    findings.append({"finding_type": "drift_threshold", "strategy_ids": [t["strategy_id"]], "severity": sev, "market": "ALL",
                                     "evidence": [dr["ref_id"], c["change_id"], prec["ref_id"] if "error" not in prec else None],
                                     "explanation": f"{c['change_id']} 后 {dr['model']} PSI={dr['psi']}，超过 {t['strategy_id']} 阈值 {t['threshold']} 的比例 ×{t['ratio']}，抽检精确率 {pv0}→{pv}",
                                     "suggested_action": ("策略 owner 评估阈值重校（run_backtest 回放）" if worse else "分布漂移但精确率未恶化，观察 7 天"), "needs_human": worse})
        # recall per market, every scenario (a market-specific evasion does not move the global hit count much)
        decl = {f["strategy_ids"][0] for f in findings if f["finding_type"] == "effect_decline"}
        for scn in self.meta["scenarios"]:
            if strategy_ids and not any(self.p.by_id[s]["scenario"] == scn for s in strategy_ids):
                continue
            r1 = self.call(trace, "query_metrics", metric="recall_est", start_date=w7, end_date=D, scenario=scn, group_by=["market"], granularity="total")
            r0 = self.call(trace, "query_metrics", metric="recall_est", start_date=w21, end_date=w7_prev_end, scenario=scn, group_by=["market"], granularity="total")
            if "error" in r1 or "error" in r0:
                continue
            m0 = {r["market"]: r for r in r0["rows"]}
            for r in r1["rows"]:
                b = m0.get(r["market"])
                if not b or r["value"] is None or b["value"] is None or r["low_sample"] or b["low_sample"]:
                    continue
                if b["value"] - r["value"] >= 0.2:
                    h1 = self.call(trace, "query_metrics", metric="hits", start_date=w7, end_date=D, market=r["market"], scenario=scn, granularity="total")
                    h0 = self.call(trace, "query_metrics", metric="hits", start_date=w21, end_date=w7_prev_end, market=r["market"], scenario=scn, granularity="total")
                    hv1 = h1["rows"][0]["value"] / 7 if "error" not in h1 else 0
                    hv0 = h0["rows"][0]["value"] / 14 if "error" not in h0 else 0
                    if not (hv0 > 0 and hv1 <= 0.7 * hv0):
                        continue   # recall_est alone is noisy; require hits to have dropped too
                    rep1 = self.call(trace, "query_metrics", metric="reports", start_date=w7, end_date=D, market=r["market"], scenario=scn, granularity="total")
                    rep0 = self.call(trace, "query_metrics", metric="reports", start_date=w21, end_date=w7_prev_end, market=r["market"], scenario=scn, granularity="total")
                    rv1 = rep1["rows"][0]["value"] / 7 if "error" not in rep1 else None
                    rv0 = rep0["rows"][0]["value"] / 14 if "error" not in rep0 else None
                    sids = [s["strategy_id"] for s in strategies if s["scenario"] == scn and s["action"] in self.meta["enforcement_actions"]
                            and (s["markets"] == ["ALL"] or r["market"] in s["markets"])]
                    findings.append({"finding_type": "recall_decline", "strategy_ids": sids, "severity": "high", "market": r["market"],
                                     "evidence": [r1["ref_id"], r0["ref_id"], h1["ref_id"], rep1["ref_id"]],
                                     "explanation": f"{r['market']} {scn} recall_est {b['value']}→{r['value']}，命中日均 {hv0:.0f}→{hv1:.0f}，举报日均 {rv0:.0f}→{rv1:.0f} 未降，疑似绕过",
                                     "suggested_action": "bad case 送模型团队；评估临时补充条件", "needs_human": True})
        # guild health breaches (机构健康分越档)
        if not strategy_ids:
            ghm = self.call(trace, "get_guild_health", start_date=w7, end_date=D)
            for b in ghm.get("breached", []):
                g1 = self.call(trace, "get_guild_health", start_date=w21, end_date=D, guild_id=b["guild_id"])
                top = g1["deductions_by_strategy"][0] if "error" not in g1 and g1["deductions_by_strategy"] else None
                findings.append({"finding_type": "guild_health_breach", "strategy_ids": [top["strategy_id"]] if top else [], "severity": "high" if b["score"] <= 70 else "medium",
                                 "market": b["market"], "evidence": [ghm["ref_id"], g1["ref_id"] if "error" not in g1 else None],
                                 "explanation": f"{b['guild_id']} 健康分 {b['score']} 进入「{b['tier']}」档" + (f"，扣分主要来自 {top['strategy_id']}（{top['points']} 分）" if top else ""),
                                 "suggested_action": "公会治理按档位处置前，先确认扣分来源策略是否受漂移影响（问：该公会健康分为什么掉了）", "needs_human": True})
        # expected changes -> info
        for c in [c for c in changes if c["type"] in ("strategy_update", "strategy_launch") and (not strategy_ids or c["strategy_id"] in strategy_ids)]:
            findings.append({"finding_type": "expected_change", "strategy_ids": [c["strategy_id"]], "severity": "info", "market": ",".join(c["markets"]),
                             "evidence": [c["change_id"]], "explanation": f"{c['change_id']} {c['date']} {c['summary']}（预期内，看板应标注）",
                             "suggested_action": "观察变更后精确率", "needs_human": False})
        for x in inactive:
            hits28 = self.call(trace, "query_metrics", metric="hits", start_date=self.meta["start"], end_date=D,
                               strategy_id=x["strategy_id"], granularity="total")
            n = hits28["rows"][0]["value"] if "error" not in hits28 and hits28["rows"] else 0
            findings.append({"finding_type": "paused_strategy", "strategy_ids": [x["strategy_id"]], "severity": "info", "market": ",".join(x["markets"]),
                             "evidence": [one_ref for one_ref in [hits28.get("ref_id")] if one_ref],
                             "explanation": f"{x['strategy_id']}（{x['name']}）状态为 {x['status']}，近 28 天命中 {n} 次；不命中是暂停的预期结果，不是僵尸策略",
                             "suggested_action": "无需处理；如要恢复，先用 run_backtest 回放再按变更流程上线，长期不用可归档", "needs_human": False})
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        for f in findings:
            f["evidence"] = [e for e in f["evidence"] if e]
        findings.sort(key=lambda f: order[f["severity"]])
        card = {"card_type": "health_check", "query": query, "as_of": D, "scope": {"strategies_checked": len(strategies)},
                "findings": findings, "conclusion": (f"{'、'.join(x['strategy_id'] for x in inactive)} 已暂停，不命中属预期；" if inactive else "") + f"巡检 {len(strategies)} 条线上策略，{sum(f['severity'] in ('high', 'medium') for f in findings)} 条待处理，"
                                                   f"{sum(f['finding_type'] == 'expected_change' for f in findings)} 条预期内变更",
                "confidence": "high", "evidence": [{"ref_id": f["evidence"][0], "type": f["finding_type"], "what": f["explanation"]} for f in findings if f["evidence"]],
                "next_steps": [f["suggested_action"] for f in findings if f["severity"] == "high"], "needs_escalation": False,
                "escalation_reasons": [], "platform_links": [f"platform://strategy/{s}" for f in findings for s in f["strategy_ids"]][:10]}
        return trace.finish(Guardrails(self.p).apply(card, trace))


# ----------------------------------------------------------------------------- LLM engine
CARD_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "card_type": {"type": "string", "enum": ["attribution", "health_check", "case", "clarification"]},
        "conclusion": {"type": "string", "description": "一两句话结论，引用具体数字和 id"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "n/a"]},
        "scope": {"type": "object", "additionalProperties": False, "properties": {
            "market": {"type": ["string", "null"]}, "scenario": {"type": ["string", "null"]}, "metric": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"]}, "low_sample": {"type": "boolean"}},
            "required": ["market", "scenario", "metric", "date", "low_sample"]},
        "metric": {"type": "object", "additionalProperties": False, "properties": {
            "key": {"type": ["string", "null"]}, "definition": {"type": ["string", "null"]}, "note": {"type": ["string", "null"], "description": "口径说明，若有歧义写明分母"}},
            "required": ["key", "definition", "note"]},
        "primary_cause": {"type": "object", "additionalProperties": False, "properties": {
            "type": {"type": "string", "enum": CAUSE_TYPES + ["case_reconstruction"]},
            "strategy_ids": {"type": "array", "items": {"type": "string"}}, "change_ids": {"type": "array", "items": {"type": "string"}},
            "guild_ids": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"}}, "required": ["type", "strategy_ids", "change_ids", "guild_ids", "explanation"]},
        "contributing_causes": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "type": {"type": "string"}, "strategy_ids": {"type": "array", "items": {"type": "string"}},
            "change_ids": {"type": "array", "items": {"type": "string"}}, "explanation": {"type": "string"}},
            "required": ["type", "strategy_ids", "change_ids", "explanation"]}},
        "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "ref_id": {"type": "string", "description": "工具返回的 ref_id（Q-xxxx）"}, "type": {"type": "string"},
            "metric": {"type": ["string", "null"]}, "what": {"type": "string"}}, "required": ["ref_id", "type", "metric", "what"]}},
        "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "finding_type": {"type": "string", "enum": ["effect_decline", "zombie_strategy", "strategy_overlap", "drift_threshold",
                                                        "precision_decline", "recall_decline", "guild_health_breach", "paused_strategy", "expected_change"]},
            "strategy_ids": {"type": "array", "items": {"type": "string"}}, "severity": {"type": "string", "enum": ["high", "medium", "low", "info"]},
            "market": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}, "explanation": {"type": "string"},
            "suggested_action": {"type": "string"}, "needs_human": {"type": "boolean"}},
            "required": ["finding_type", "strategy_ids", "severity", "market", "evidence", "explanation", "suggested_action", "needs_human"]}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "needs_escalation": {"type": "boolean"}, "escalation_reasons": {"type": "array", "items": {"type": "string"}},
        "clarification_questions": {"type": "array", "items": {"type": "string"}},
        "sample_case_ids": {"type": "array", "items": {"type": "string"}},
        "platform_links": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["card_type", "conclusion", "confidence", "scope", "metric", "primary_cause", "contributing_causes", "evidence",
                 "findings", "next_steps", "needs_escalation", "escalation_reasons", "clarification_questions", "sample_case_ids", "platform_links"],
}
SUBMIT_TOOL = {"name": "submit_card", "description": "提交最终结构化卡片。这是唯一的结束方式；提交前确认每条证据都有真实 ref_id。",
               "strict": True, "input_schema": CARD_SCHEMA}


class LLMEngine:
    """Claude tool-use loop. Playbook markdown is the system prompt; platform tools are the tools."""

    def __init__(self, platform, model=MODEL, max_turns=25, effort="medium"):
        import anthropic
        self.p, self.model, self.max_turns, self.effort = platform, model, max_turns, effort
        self.client = anthropic.Anthropic()
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOL_SPECS] + [SUBMIT_TOOL]

    def _system(self, playbook):
        pb = open(os.path.join(PLAYBOOK_DIR, f"{playbook}.md"), encoding="utf-8").read()
        m = self.p.meta
        return (
            "你是治理平台的策略巡检与异常归因助手。你只编排平台已有的只读工具，不新建能力，不替策略团队做决定。\n"
            f"平台今天是 {m['today']}，数据范围 {m['start']} ~ {m['end']}。市场：{m['market_names']}。场景：{m['scenario_names']}。\n"
            "规则：\n"
            "1. 所有结论必须引用工具返回的 ref_id 或真实实体 id；不存在的策略/变更/指标会被校验层拦截。\n"
            "2. 工具报错时重试一次（修正参数），仍失败就在卡片里说明缺什么，不编造数字。\n"
            "3. 输入缺市场/时间/指标时，提交 card_type=clarification 并列出问题，不猜。\n"
            "4. 最后必须调用 submit_card 工具结束；不要用纯文本回答。\n"
            "5. 回答语言与用户一致（默认中文）。\n\n" + pb)

    def run(self, playbook, query, trace=None):
        trace = trace or Trace("llm", playbook, query)
        messages = [{"role": "user", "content": query}]
        system = self._system(playbook)
        card, retried_validation = None, False
        for turn in range(self.max_turns):
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=system, tools=self.tools, messages=messages,
                output_config={"effort": self.effort}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            trace.model_calls += 1
            for k in trace.usage:
                trace.usage[k] += getattr(resp.usage, k, 0) or 0
            if resp.stop_reason == "refusal":
                trace.add("model_refusal", details=str(getattr(resp, "stop_details", None)))
                break
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            texts = [b.text for b in resp.content if b.type == "text"]
            if texts:
                trace.add("model_text", text="\n".join(texts))
            if not tool_uses:
                if resp.stop_reason == "max_tokens":
                    trace.add("model_truncated")
                    break
                messages.append({"role": "user", "content": "请调用 submit_card 提交结构化卡片。"})
                continue
            results = []
            for tu in tool_uses:
                args = dict(tu.input)
                if tu.name == "submit_card":
                    card = args
                    unknown = Validator(self.p).check_card(card, trace_refs(trace))
                    trace.add("submit_card", unknown_references=[f"{k}:{i}" for k, i in unknown])
                    if unknown and not retried_validation:
                        retried_validation = True
                        card = None
                        results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                        "content": f"校验失败：以下引用不存在 {unknown}。只引用工具真实返回的 id，然后重新 submit_card。"})
                        continue
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "已收到"})
                    continue
                res = self.p.call(tu.name, args)
                trace.tool(tu.name, args, res)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": "error" in res,
                                "content": json.dumps(res, ensure_ascii=False, default=str)[:60000]})
            messages.append({"role": "user", "content": results})
            if card is not None:
                break
        if card is None:
            card = {"card_type": "attribution", "conclusion": "模型未能产出结构化卡片（超过轮数、被拒绝或被截断）", "confidence": "low",
                    "scope": {"low_sample": False}, "primary_cause": {"type": "unexplained", "strategy_ids": [], "change_ids": [],
                    "explanation": "no submit_card"}, "evidence": [], "next_steps": [], "needs_escalation": True,
                    "escalation_reasons": ["agent 未完成"], "contributing_causes": [], "findings": [], "platform_links": []}
        card["query"] = query
        return trace.finish(Guardrails(self.p).apply(card, trace))


# ----------------------------------------------------------------------------- facade
class Copilot:
    def __init__(self, engine="rules", platform=None, **kw):
        self.p = platform or Platform()
        self.engine = engine
        self.rules = RulesEngine(self.p)
        self.llm = LLMEngine(self.p, **kw) if engine == "llm" else None

    def ask(self, query, save=True):
        s = parse_query(query, self.p.meta)
        if self.engine == "llm":
            pb = "health_check" if re.search(r"巡检|health|体检|盘点", query) else "attribution"
            tr = self.llm.run(pb, query)
        elif re.search(r"巡检|health|体检|盘点", query) and not s["metric"]:
            tr = self.rules.health_check(query, strategy_ids=[s["strategy_id"]] if s["strategy_id"] else None)
        else:
            tr = self.rules.attribution(query)
        if save:
            tr.path = tr.save()
        return tr

    def health(self, save=True):
        tr = self.llm.run("health_check", "请对所有线上策略做健康巡检，输出待处理清单。") if self.engine == "llm" else self.rules.health_check()
        if save:
            tr.path = tr.save()
        return tr


def render_card(card):
    lines = [f"【结论】{card.get('conclusion')}", f"【置信度】{card.get('confidence')}   【需升级】{card.get('needs_escalation')} {card.get('escalation_reasons') or ''}"]
    pc = card.get("primary_cause")
    if pc:
        lines.append(f"【主因】{pc.get('type')}  策略={pc.get('strategy_ids')} 变更={pc.get('change_ids')}\n       {pc.get('explanation')}")
    if card.get("metric"):
        lines.append(f"【口径】{card['metric'].get('key')}: {card['metric'].get('definition')}" + (f"\n       {card['metric']['note']}" if card['metric'].get('note') else ""))
    if card.get("findings"):
        for f in card["findings"]:
            lines.append(f"  - [{f['severity']}] {f['finding_type']} {f['strategy_ids']} {f.get('market', '')}: {f['explanation']} → {f['suggested_action']}")
    if card.get("evidence"):
        lines.append("【证据】")
        for e in card["evidence"][:12]:
            lines.append(f"  - {e.get('ref_id')} [{e.get('type')}] {e.get('what', '')}")
    if card.get("next_steps"):
        lines.append("【下一步】" + " | ".join(card["next_steps"]))
    if card.get("validation"):
        lines.append(f"【校验】{'通过' if card['validation']['passed'] else '失败 ' + str(card['validation']['unknown_references'])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--engine", default="rules", choices=["rules", "llm"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    cp = Copilot(engine=a.engine)
    if a.health:
        tr = cp.health()
    elif a.case:
        tr = cp.ask(f"还原 case {a.case}")
    else:
        tr = cp.ask(a.query or "美国昨天误杀申诉涨了，帮我看看")
    print(json.dumps(tr.card, ensure_ascii=False, indent=1, default=str) if a.json else render_card(tr.card))
    print(f"\n[trace] {tr.path}  tool_calls={sum(1 for e in tr.events if e['kind'] == 'tool_call')} latency={tr.latency_s}s engine={tr.engine}")


if __name__ == "__main__":
    main()
