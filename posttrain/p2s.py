#!/usr/bin/env python3
"""Policy-to-Strategy (P2S) v1: turn a policy update into a strategy change draft, graded on held-out days.

The agent sees a policy text, reads the strategy library, replays candidate score lines on the first 14 days
(re-reviewed labels), and submits one decision: no_change, change an existing strategy, or a new
market-specific strategy. The grader re-applies the submitted rule to the LAST 14 days with the hidden
ground truth, so a threshold tuned to the replay window only passes if it generalises.

  python posttrain/p2s.py gen            # tasks for all worlds -> runs/p2s_tasks_{train,test}.jsonl, p2s_sft_train.jsonl
  python posttrain/p2s.py check          # replay the reference trajectories through the episode loop (must be 100%)
  python posttrain/p2s.py eval --name base [--adapter DIR] [--split test] [--limit N]

Task types (see the design note in reports/posttrain_report.md):
  A expand   policy widens a definition -> lower the scenario's manual-review strategy to hit coverage/precision targets
  B wording  policy only clarifies wording -> no_change
  C market   new market-specific requirement -> NEW manual_review strategy limited to that market
  D action   policy changes the action of the scenario's warn strategy; threshold unchanged; financial approval flagged
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from posttrain.harness import THINK, obs_segment  # noqa: E402,F401

WORLDS = os.path.join(ROOT, "data", "worlds")
RUNS = os.path.join(ROOT, "posttrain", "runs")
REPLAY = ("2026-09-18", "2026-10-01")
SCORE = ("2026-10-02", "2026-10-15")
TOL = 0.03
CAP_TOL = 0.08   # review-capacity slack on the scoring window (traffic varies)
HARNESS_VERSION = "p2s-v1"

SCN_ZH = {"adult_content": "色情低俗", "minor_safety": "未成年人安全", "scam_fraud": "诈骗引流", "violence": "暴力血腥",
          "spam_bot": "垃圾刷量", "gambling": "赌博", "hate_harassment": "仇恨骚扰", "counterfeit": "假冒侵权"}
MKT_ZH = {"US": "美国", "BR": "巴西", "ID": "印尼", "GB": "英国", "VN": "越南", "KE": "肯尼亚"}
ACT_ZH = {"observe": "观察", "manual_review": "转人审", "limit_traffic": "限流", "warn": "警告", "delay_settlement": "延迟结算",
          "freeze_earnings": "冻结违规场次收益", "suspend_live": "断播", "disable_gifts": "关闭礼物", "ban_7d": "封禁 7 天"}
FINANCIAL = {"delay_settlement", "freeze_earnings", "disable_gifts"}
FIELD_ZH = {"report_count": "举报", "new_device_flag": "新设备", "keyword_hits": "关键词", "legacy_brand_list_match": "旧品牌词表",
            "external_link_flag": "外链", "account_age_days": "账号天数", "gift_amount": "打赏", "viewers_peak": "观看峰值"}
MR_STRATEGY = {"adult_content": "S-103", "minor_safety": "S-106", "violence": "S-113", "gambling": "S-116", "counterfeit": "S-124",
               "scam_fraud": "S-110"}
WARN_STRATEGY = {"counterfeit": "S-109", "hate_harassment": "S-111"}
BIG_MARKETS = ["US", "BR", "ID", "GB"]
OPS = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less, "==": np.equal, "!=": np.not_equal}

# ----------------------------------------------------------------------------- prompt + tool schemas
SYSTEM = """你是直播治理平台的策略助手。输入是一条 policy 更新，你要把它落实到策略：判断要不要改、改哪条、怎么改，最后调用 submit_change 结束。
回放窗口 2026-09-18 ~ 2026-10-01：backtest_rule 只能看到这段时间的已复核标签。
精确率 = 命中里真实违规的比例；覆盖率 = 策略范围内的真实违规被命中的比例。
场景: adult_content 色情低俗, minor_safety 未成年人安全, scam_fraud 诈骗引流, violence 暴力血腥, spam_bot 垃圾刷量, gambling 赌博, hate_harassment 仇恨骚扰, counterfeit 假冒侵权。
市场: US 美国, BR 巴西, ID 印尼, GB 英国, VN 越南, KE 肯尼亚。
动作（严重度递增）: observe 观察, manual_review 转人审, limit_traffic 限流, warn 警告, delay_settlement 延迟结算*, freeze_earnings 冻结违规场次收益*, suspend_live 断播, disable_gifts 关闭礼物*, ban_7d 封禁7天。带 * 的是资金侧动作，needs_financial_approval 必须为 true，其他为 false。
做法:
1 policy 只改措辞或补充示例、执行标准不变 → decision=no_change。
2 要求调整现有策略的覆盖率/精确率/人审量 → get_strategy_config 找到该场景的转人审策略（动作 manual_review），用 backtest_rule 回放多个分数线，在同时满足覆盖率、精确率、日均人审量要求的分数线里选覆盖率最高的，decision=change，动作不变。
3 只改处置动作 → 找到该场景的警告策略（动作 warn），动作改成 policy 要求的那个，分数线保持原值，decision=change，不需要回放。
4 新的市场专项要求 → decision=new，strategy_id=NEW，markets 只填该市场，action=manual_review，用 backtest_rule(scenario, market, thresholds) 回放选分数线。
5 不要用比 policy 要求更重的动作；只引用工具返回过的策略编号。每轮只调用一个工具，格式 <tool_call>{"name": ..., "arguments": {...}}</tool_call>。"""

TOOLS = [
    {"type": "function", "function": {"name": "get_strategy_config", "description": "查策略库：条件、分数线、动作、市场。",
                                      "parameters": {"type": "object", "properties": {"scenario": {"type": "string"}, "market": {"type": "string"},
                                                                                       "strategy_id": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "backtest_rule", "description": "在回放窗口按多个分数线回放，返回日均命中、精确率、覆盖率。给 strategy_id 时沿用该策略的附加条件和市场范围；给 market 时只看该市场。",
                                      "parameters": {"type": "object", "properties": {"scenario": {"type": "string"}, "thresholds": {"type": "array"},
                                                                                       "strategy_id": {"type": "string"}, "market": {"type": "string"}},
                                                     "required": ["scenario", "thresholds"]}}},
    {"type": "function", "function": {"name": "submit_change", "description": "提交策略变更方案并结束。",
                                      "parameters": {"type": "object", "properties": {
                                          "decision": {"type": "string", "enum": ["no_change", "change", "new"]}, "strategy_id": {"type": "string"},
                                          "scenario": {"type": "string"}, "markets": {"type": "array"}, "threshold": {"type": "number"},
                                          "action": {"type": "string"}, "needs_financial_approval": {"type": "boolean"}, "rationale": {"type": "string"}},
                                          "required": ["decision", "rationale"]}}}]
TOOL_NAMES = ["get_strategy_config", "backtest_rule"]


def prefix_text(tok, query):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def assistant_text(tool, args, thought):
    return f"{thought}\n<tool_call>\n{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


# ----------------------------------------------------------------------------- world
class World:
    def __init__(self, wid):
        d = os.path.join(WORLDS, wid)
        self.wid = wid
        s = pd.read_csv(os.path.join(d, "sessions.csv"))
        truth = pd.read_csv(os.path.join(d, "_hidden_truth.csv")).set_index("session_id").true_label
        s["label"] = s.session_id.map(truth).values
        self.win = {"replay": s[(s.date >= REPLAY[0]) & (s.date <= REPLAY[1])].reset_index(drop=True),
                    "score": s[(s.date >= SCORE[0]) & (s.date <= SCORE[1])].reset_index(drop=True)}
        self.strategies = {x["strategy_id"]: x for x in json.load(open(os.path.join(d, "strategies.json")))}
        self.problems = json.load(open(os.path.join(d, "_embedded_problems.json")))

    def affected(self):
        p = self.problems
        return {p["P1_model_drift"]["scenario"], p["P2_recall_drop"]["scenario"], p["P9_guild_spike"]["scenario"]}

    def stats(self, window, scenario, thresholds, markets=None, extra=()):
        df = self.win[window]
        scope = np.ones(len(df), dtype=bool)
        if markets and markets != ["ALL"]:
            scope &= df.market.isin(markets).values
        for c in extra:
            scope &= OPS[c["op"]](df[c["field"]].values, c["value"])
        sc, pos = df[f"score_{scenario}"].values, (df.label.values == scenario)
        days = df.date.nunique()
        out = []
        for t in thresholds:
            hit = scope & (sc >= t)
            n, tp = int(hit.sum()), int((hit & pos).sum())
            out.append({"threshold": round(float(t), 3), "hits_per_day": round(n / days, 1), "precision": round(tp / n, 3) if n else None,
                        "coverage": round(tp / max(int((scope & pos).sum()), 1), 3)})
        return out

    def rule_of(self, sid):
        st = self.strategies[sid]
        return st["scenario"], st["markets"], st["extra_conditions"], st["threshold"], st["action"]


# ----------------------------------------------------------------------------- tools
def _extra_zh(extra):
    return "、".join(f"{FIELD_ZH.get(c['field'], c['field'])}{c['op'].replace('>=', '≥').replace('==', '=')}{c['value']}" for c in extra) or "无"


class Env:
    def __init__(self, world):
        self.w, self.calls, self.seen = world, [], set()

    def step(self, name, args):
        try:
            obs = getattr(self, "_" + name)(**args)
            err = None
        except TypeError as e:
            obs, err = f"错误 bad_args: {e}", "bad_args"
        except ValueError as e:
            obs, err = f"错误 {e}", str(e).split(":")[0]
        self.calls.append({"tool": name, "args": args, "error": err})
        return obs

    def _get_strategy_config(self, scenario=None, market=None, strategy_id=None):
        rows = [s for s in self.w.strategies.values() if s["status"] == "active"
                and (not scenario or s["scenario"] == scenario) and (not strategy_id or s["strategy_id"] == strategy_id)
                and (not market or s["markets"] == ["ALL"] or market in s["markets"])]
        if not rows:
            raise ValueError("not_found: 没有匹配的策略")
        self.seen.update(s["strategy_id"] for s in rows)
        return "; ".join(f"{s['strategy_id']} {SCN_ZH[s['scenario']]} 动作 {s['action']} 分数线 {s['threshold']} 附加 {_extra_zh(s['extra_conditions'])} "
                         f"市场 {','.join(s['markets'])}" for s in rows[:8])

    def _backtest_rule(self, scenario, thresholds, strategy_id=None, market=None):
        if scenario not in SCN_ZH:
            raise ValueError(f"unknown_scenario: 可选 {', '.join(SCN_ZH)}")
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds]
        ths = sorted({round(float(t), 3) for t in thresholds if 0 < float(t) < 1})[:12]
        if not ths:
            raise ValueError("bad_thresholds: 给一组 0~1 之间的分数线，例如 [0.4, 0.5, 0.6]")
        markets, extra, cur = ([market] if market else None), (), None
        if strategy_id:
            if strategy_id not in self.w.strategies:
                raise ValueError(f"unknown_strategy: {strategy_id} 不存在")
            sc2, mk, extra, cur, _ = self.w.rule_of(strategy_id)
            if sc2 != scenario:
                raise ValueError(f"scenario_mismatch: {strategy_id} 的场景是 {sc2}")
            markets = mk if not market else [market]
            self.seen.add(strategy_id)
        if market and market not in MKT_ZH:
            raise ValueError(f"unknown_market: 可选 {', '.join(MKT_ZH)}")
        rows = self.w.stats("replay", scenario, ths + ([cur] if cur is not None and cur not in ths else []), markets, extra)
        scope = f"{strategy_id or '新规则'} {SCN_ZH[scenario]} 市场 {','.join(markets or ['ALL'])} 附加 {_extra_zh(extra)}"
        body = "; ".join(f"{r['threshold']}{'(当前)' if r['threshold'] == cur else ''} 日均{r['hits_per_day']} 精确率{r['precision']} 覆盖率{r['coverage']}" for r in rows)
        return f"回放 09-18~10-01 {scope}: {body}"


# ----------------------------------------------------------------------------- parsing + gate
def parse_call(text):
    i = text.find("<tool_call>")
    body = text[i + len("<tool_call>"):] if i >= 0 else text
    j = body.find("{")
    if j < 0:
        return None, "no_tool_call"
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[j:])
        name, args = obj.get("name"), obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
    except Exception:
        return None, "bad_json"
    if name not in TOOL_NAMES + ["submit_change"] or not isinstance(args, dict):
        return None, f"bad_tool:{name}"
    return (name, normalize(args)), None


def normalize(a):
    a = {k: v for k, v in a.items() if v not in (None, "", [])}
    zh2scn = {v: k for k, v in SCN_ZH.items()}
    zh2mkt = {v: k for k, v in MKT_ZH.items()}
    if a.get("scenario") in zh2scn:
        a["scenario"] = zh2scn[a["scenario"]]
    if a.get("market") in zh2mkt:
        a["market"] = zh2mkt[a["market"]]
    if isinstance(a.get("markets"), str):
        a["markets"] = [a["markets"]]
    if isinstance(a.get("thresholds"), str):
        try:
            a["thresholds"] = json.loads(a["thresholds"])
        except Exception:
            pass
    return a


def gate(sub, env, rejects):
    w, msgs = env.w, []
    dec = sub.get("decision")
    if dec not in ("no_change", "change", "new"):
        msgs.append(("enum", "decision 只能是 no_change / change / new"))
    if sub.get("action") and sub["action"] not in ACT_ZH:
        msgs.append(("enum", "action 可选: " + ", ".join(ACT_ZH)))
    sid = sub.get("strategy_id")
    if dec == "change" and sid not in w.strategies:
        msgs.append(("ids", f"strategy_id={sid!r} 不存在；先用 get_strategy_config 查到编号"))
    if dec in ("change", "new") and sub.get("threshold") is not None:
        cur = w.strategies[sid]["threshold"] if dec == "change" and sid in w.strategies else None
        changed = cur is None or abs(float(sub["threshold"]) - cur) > 1e-6
        if changed and not any(c["tool"] == "backtest_rule" and not c["error"] for c in env.calls):
            msgs.append(("backtest", "改分数线前先用 backtest_rule 回放几个候选分数线"))
    for kind, m in msgs:
        if rejects.get(kind, 0) < 2:
            rejects[kind] = rejects.get(kind, 0) + 1
            return f"提交被退回（{kind}）: {m}"
    return None


# ----------------------------------------------------------------------------- grading (hidden: last 14 days)
def grade(task, sub, calls, world):
    e, c = task["expected"], {}
    s = sub or {}
    c["submitted"] = sub is not None
    c["decision_correct"] = s.get("decision") == e["decision"]
    known = set(world.strategies) | {"NEW"}
    sid = s.get("strategy_id")
    c["no_hallucination"] = sub is not None and (sid is None or sid in known)
    t = e["type"]
    fin = bool(s.get("needs_financial_approval"))
    if t == "B":
        c.update(target_correct=None, action_correct=None, threshold_ok=None, financial_flag_correct=None, backtested=None)
    elif t == "A":
        cur = world.strategies[e["strategy_id"]]
        c["target_correct"] = sid == e["strategy_id"] and (s.get("markets") in (None, [], ["ALL"]))
        c["action_correct"] = s.get("action") in (None, cur["action"])
        c["threshold_ok"] = _meets(world, cur["scenario"], s.get("threshold"), cur["markets"], cur["extra_conditions"], e)
        c["financial_flag_correct"] = not fin
        c["backtested"] = any(k["tool"] == "backtest_rule" and not k["error"] for k in calls)
    elif t == "C":
        c["target_correct"] = s.get("scenario") == e["scenario"] and s.get("markets") == [e["market"]] and sid in (None, "NEW")
        c["action_correct"] = s.get("action") == "manual_review"
        c["threshold_ok"] = _meets(world, e["scenario"], s.get("threshold"), [e["market"]], (), e)
        c["financial_flag_correct"] = not fin
        c["backtested"] = any(k["tool"] == "backtest_rule" and not k["error"] for k in calls)
    else:   # D
        cur = world.strategies[e["strategy_id"]]
        c["target_correct"] = sid == e["strategy_id"]
        c["action_correct"] = s.get("action") == e["action"]
        thr = s.get("threshold")
        c["threshold_ok"] = thr is None or abs(float(thr) - cur["threshold"]) < 1e-3
        c["financial_flag_correct"] = fin == (e["action"] in FINANCIAL)
        c["backtested"] = None
    applicable = {k: v for k, v in c.items() if v is not None}
    return {"criteria": c, "pass": all(applicable.values()), "score": sum(applicable.values()) / len(applicable)}


def _meets(world, scenario, thr, markets, extra, e):
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        return False
    r = world.stats("score", scenario, [thr], markets, extra)[0]
    cap_ok = "max_hits_per_day" not in e or r["hits_per_day"] <= e["max_hits_per_day"] * (1 + CAP_TOL)
    return r["precision"] is not None and r["precision"] >= e["min_precision"] - TOL and r["coverage"] >= e["min_coverage"] - TOL and cap_ok


# ----------------------------------------------------------------------------- task generation + reference solver
TEXT = {
    "A": ["Policy 更新（{pid}，2026-10-02 起生效）：{xz} 的认定范围扩大。要求对应的转人审策略覆盖率不低于 {R:.0%}，精确率不低于 {P:.0%}，且人审量不超过每天 {cap} 场（人审团队容量）。请给出策略调整方案。",
          "{xz} 的 policy 改了（{pid}），口径放宽，人审那条策略得多抓一些：覆盖率要到 {R:.0%} 以上，精确率别低于 {P:.0%}，但人审一天最多只能看 {cap} 场。帮我出个改法。"],
    "T": ["Policy 更新（{pid}，2026-10-02 起生效）：{xz} 人审资源调整，对应的转人审策略人审量不超过每天 {cap} 场，同时覆盖率不低于 {R:.0%}，精确率不低于 {P:.0%}。请给出策略调整方案。",
          "{xz} 的人审排不过来了（{pid}）：人审那条策略一天最多送 {cap} 场，覆盖率还得保持在 {R:.0%} 以上，精确率不低于 {P:.0%}。怎么调？"],
    "B": ["Policy 更新（{pid}，2026-10-02 起生效）：{xz} 条款措辞澄清并补充示例，执行标准不变。请评估是否需要调整策略。",
          "{xz} 的 policy 文案改了一下（{pid}），说只是把例子写清楚，标准没变。策略要动吗？"],
    "C": ["Policy 更新（{pid}，2026-10-02 起生效）：{mz} 市场新增 {xz} 专项要求，该市场 {xz} 中高分直播一律转人审，覆盖率不低于 {R:.0%}，精确率不低于 {P:.0%}，不得影响其他市场。请给出策略方案。",
          "{mz} 那边要单独加强 {xz}（{pid}）：只在 {mz}，中高分的都转人审，覆盖率至少 {R:.0%}，精确率至少 {P:.0%}，别动其他市场。怎么配？"],
    "D": ["Policy 更新（{pid}，2026-10-02 起生效）：{xz} 高分违规的处置由警告升级为{az}，分数线不变。请给出策略调整方案。",
          "{xz} 高分的那条警告策略，按新 policy（{pid}）要改成{az}，线不用动。帮我出变更。"],
}
GRID = [round(0.35 + 0.05 * i, 2) for i in range(12)]   # 0.35 .. 0.90 (the tool's 12-threshold cap)


def _floor2(x):
    return math.floor(x * 100) / 100


def make_tasks(world, split, rng):
    out, aff, pid_n = [], world.affected(), [10]

    def pid():
        pid_n[0] += 1
        return f"P-{pid_n[0]}"
    # A: expand (lower the line) or tighten (raise it) a manual-review strategy, with a review-capacity cap
    cands = [(x, sid) for x, sid in MR_STRATEGY.items() if x not in aff]
    for x, sid in rng.sample(cands, len(cands)):
        if sum(1 for o in out if o["type"] == "A") >= 2:
            break
        scn, mk, extra, cur, act = world.rule_of(sid)
        found = None
        for mode in rng.sample(["expand", "tighten"], 2):
            for delta in rng.sample([0.05, 0.10, 0.15, 0.20], 4):
                t_star = round(cur - delta if mode == "expand" else cur + delta, 2)
                if not 0.40 <= t_star <= 0.85:
                    continue
                probe = 0.35 if mode == "expand" else min(0.95, t_star + 0.15)
                r = world.stats("replay", scn, [cur, t_star, probe], mk, extra)
                if r[1]["precision"] is None:
                    continue
                R, P = _floor2(r[1]["coverage"] - 0.01), max(0.05, _floor2(r[1]["precision"] - 0.05))
                cap = math.ceil(r[1]["hits_per_day"] * 1.08)
                over_cap = lambda h: h > cap * (1 + CAP_TOL) + 1
                low_cov = lambda c: c < R - TOL - 0.01
                if mode == "expand" and low_cov(r[0]["coverage"]) and over_cap(r[2]["hits_per_day"]):
                    found = (mode, R, P, cap)
                if mode == "tighten" and over_cap(r[0]["hits_per_day"]) and low_cov(r[2]["coverage"]):
                    found = (mode, R, P, cap)
                if found:
                    break
            if found:
                break
        if not found:
            continue
        mode, R, P, cap = found
        out.append({"type": "A", "kind": mode, "text": "A" if mode == "expand" else "T", "fmt": dict(xz=SCN_ZH[x], R=R, P=P, cap=cap),
                    "expected": {"type": "A", "decision": "change", "strategy_id": sid, "scenario": x, "min_coverage": R, "min_precision": P,
                                 "max_hits_per_day": cap}})
    # B: wording only
    x = rng.choice([s for s in SCN_ZH if s not in aff])
    out.append({"type": "B", "kind": "wording", "fmt": dict(xz=SCN_ZH[x]), "expected": {"type": "B", "decision": "no_change", "scenario": x}})
    # C: new market-specific manual-review rule
    special = {(m, s["scenario"]) for s in world.strategies.values() if s["markets"] != ["ALL"] for m in s["markets"]}
    pairs = [(m, x) for m in BIG_MARKETS for x in SCN_ZH if x not in aff and (m, x) not in special and x != "spam_bot"]
    if pairs:
        m, x = rng.choice(pairs)
        t_star = rng.choice([0.60, 0.65, 0.70, 0.75])
        r = world.stats("replay", x, [t_star], [m])[0]
        if r["precision"] is not None:
            R, P = _floor2(r["coverage"] - 0.02), max(0.05, _floor2(r["precision"] - 0.05))
            out.append({"type": "C", "kind": "market", "fmt": dict(xz=SCN_ZH[x], mz=MKT_ZH[m], R=R, P=P),
                        "expected": {"type": "C", "decision": "new", "scenario": x, "market": m, "min_coverage": R, "min_precision": P}})
    # D: action change on a warn strategy
    dc = [(x, sid) for x, sid in WARN_STRATEGY.items() if x not in aff]
    if dc:
        x, sid = rng.choice(dc)
        act = rng.choice(["freeze_earnings", "suspend_live"])
        out.append({"type": "D", "kind": "action", "fmt": dict(xz=SCN_ZH[x], az=ACT_ZH[act]),
                    "expected": {"type": "D", "decision": "change", "strategy_id": sid, "scenario": x, "action": act}})
    tasks = []
    for k, t in enumerate(out):
        phr = ["A"] if split == "train" else ["A", "B"]
        p = pid()
        for ph in phr:
            q = TEXT[t.get("text", t["type"])][0 if ph == "A" else 1].format(pid=p, **t["fmt"])
            tasks.append({"task_id": f"{world.wid}-p{k:02d}{ph.lower()}", "world": world.wid, "split": split, "kind": t["kind"],
                          "phrasing": ph, "query": q, "expected": t["expected"]})
    return tasks


def reference(task, world):
    """Rules solver: returns list of (tool, args, thought) and the submission. Reads the structured expectation
    only for what the policy text states explicitly (scenario, market, targets, required action)."""
    e, steps = task["expected"], []
    x = e["scenario"]
    if e["type"] == "B":
        return steps, {"decision": "no_change", "scenario": x, "rationale": "policy 只澄清措辞，执行标准不变，无需调整策略。"}
    if e["type"] == "D":
        steps.append(("get_strategy_config", {"scenario": x}, "找到该场景的警告策略。"))
        sid = WARN_STRATEGY[x]
        cur = world.strategies[sid]
        return steps, {"decision": "change", "strategy_id": sid, "scenario": x, "markets": cur["markets"], "threshold": cur["threshold"],
                       "action": e["action"], "needs_financial_approval": e["action"] in FINANCIAL,
                       "rationale": f"按 policy 把 {sid} 的处置从 warn 改为 {e['action']}，分数线不变" + ("；资金侧动作需结算/财务审批。" if e["action"] in FINANCIAL else "。")}
    if e["type"] == "A":
        steps.append(("get_strategy_config", {"scenario": x}, "找到该场景的转人审策略。"))
        sid = MR_STRATEGY[x]
        scn, mk, extra, cur, act = world.rule_of(sid)
        args = {"scenario": x, "strategy_id": sid}
        markets = mk
    else:
        sid, extra, markets = None, (), [e["market"]]
        args = {"scenario": x, "market": e["market"]}
        cur = None
    coarse = world.stats("replay", x, GRID, markets, extra)
    steps.append(("backtest_rule", {**args, "thresholds": GRID}, "回放一组候选分数线。"))
    cap = e.get("max_hits_per_day", float("inf"))
    def fits(r, m):
        return r["precision"] is not None and r["precision"] >= e["min_precision"] + m and r["coverage"] >= e["min_coverage"] + m and r["hits_per_day"] <= cap * (1 - m)
    ok = [r for r in coarse if fits(r, 0.02)] or [r for r in coarse if fits(r, 0.0)]
    if ok:
        best = max(ok, key=lambda r: (r["coverage"], -r["threshold"]))
        fine = [round(best["threshold"] + d, 2) for d in (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04)]
        steps.append(("backtest_rule", {**args, "thresholds": fine}, "在最好的分数线附近细化。"))
        fr = world.stats("replay", x, fine, markets, extra)
        ok2 = [r for r in fr if fits(r, 0.02)] or [r for r in fr if fits(r, 0.0)]
        best = max(ok2, key=lambda r: (r["coverage"], -r["threshold"])) if ok2 else best
        thr = best["threshold"]
    else:
        thr = None
    if e["type"] == "A":
        sub = {"decision": "change", "strategy_id": sid, "scenario": x, "markets": markets, "threshold": thr, "action": "manual_review",
               "needs_financial_approval": False, "rationale": f"{sid} 分数线 {cur} → {thr}，回放满足覆盖率和精确率要求。"}
    else:
        sub = {"decision": "new", "strategy_id": "NEW", "scenario": x, "markets": markets, "threshold": thr, "action": "manual_review",
               "needs_financial_approval": False, "rationale": f"新增 {e['market']} 专项转人审规则，分数线 {thr}，只在该市场生效。"}
    return steps, sub


# ----------------------------------------------------------------------------- episode loop
def run_episode(policy, world, task, max_turns=10):
    env = Env(world)
    out = policy.start(prefix_text(policy.tok, task["query"]) if policy.tok else task["query"], raw=True)
    transcript = [{"role": "assistant", "text": out}]
    submit, fmt_err, rejects = None, 0, {}
    for turn in range(max_turns):
        call, err = parse_call(out)
        if err:
            fmt_err += 1
            obs = (f"格式错误 {err}：工具名只能是 {', '.join(TOOL_NAMES)} 或 submit_change。" if err.startswith("bad_tool")
                   else "格式错误：请用 <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> 调用一个工具。")
        elif call[0] == "submit_change":
            msg = gate(call[1], env, rejects)
            if msg is None:
                submit = call[1]
                break
            obs = msg
        else:
            obs = env.step(*call)
        transcript.append({"role": "tool", "text": obs})
        if turn == max_turns - 1:
            break
        out = policy.continue_(obs_segment(obs))
        transcript.append({"role": "assistant", "text": out})
    g = grade(task, submit, env.calls, world)
    return {"task_id": task["task_id"], "harness_version": HARNESS_VERSION, "submit": submit, "calls": env.calls, "transcript": transcript,
            "gate_rejects": rejects, "format_errors": fmt_err, **g}


class Replay:
    tok = None

    def __init__(self, turns):
        self.turns, self.i, self.gen_tokens = turns, 0, 0

    def start(self, text, raw=False):
        return self._next()

    def continue_(self, text):
        return self._next()

    def _next(self):
        t = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return t


def teacher_turns(task, world):
    steps, sub = reference(task, world)
    env = Env(world)
    turns = []
    for tool, args, thought in steps:
        turns.append((assistant_text(tool, args, thought), env.step(tool, args)))
    turns.append((assistant_text("submit_change", sub, "方案确定，提交。"), None))
    return turns, sub


# ----------------------------------------------------------------------------- CLI
def cmd_gen(a):
    index = json.load(open(os.path.join(WORLDS, "index.json")))
    rng = random.Random(20261018)
    files = {"train": open(os.path.join(RUNS, "p2s_tasks_train.jsonl"), "w"), "test": open(os.path.join(RUNS, "p2s_tasks_test.jsonl"), "w")}
    sft = open(os.path.join(RUNS, "p2s_sft_train.jsonl"), "w")
    stats = {"kept": {}, "dropped": {}}
    for w in index:
        world = World(w["world_id"])
        for t in make_tasks(world, w["split"], rng):
            turns, sub = teacher_turns(t, world)
            g = grade(t, sub, [{"tool": "backtest_rule", "error": None}] if sub.get("threshold") is not None else [], world)
            key = f"{w['split']}:{t['kind']}"
            if not g["pass"]:
                stats["dropped"][key] = stats["dropped"].get(key, 0) + 1
                continue
            stats["kept"][key] = stats["kept"].get(key, 0) + 1
            t["teacher_submit"] = sub
            files[w["split"]].write(json.dumps(t, ensure_ascii=False) + "\n")
            if w["split"] == "train":
                sft.write(json.dumps({"task_id": t["task_id"], "query": t["query"], "turns": turns, "prefix": "p2s"}, ensure_ascii=False) + "\n")
        print(w["world_id"], stats["kept"], flush=True)
    for f in list(files.values()) + [sft]:
        f.close()
    json.dump(stats, open(os.path.join(RUNS, "p2s_task_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


def cmd_check(a):
    tasks = [json.loads(l) for l in open(os.path.join(RUNS, f"p2s_tasks_{a.split}.jsonl"))]
    worlds, ok = {}, 0
    for t in tasks:
        w = worlds.get(t["world"]) or worlds.setdefault(t["world"], World(t["world"]))
        turns, _ = teacher_turns(t, w)
        r = run_episode(Replay([x[0] for x in turns]), w, t)
        ok += r["pass"]
        if not r["pass"]:
            print("FAIL", t["task_id"], {k: v for k, v in r["criteria"].items() if v is False}, r["gate_rejects"])
    print(f"reference replay {ok}/{len(tasks)}")


def cmd_eval(a):
    from posttrain.policy import QwenPolicy, load
    tasks = [json.loads(l) for l in open(os.path.join(RUNS, f"p2s_tasks_{a.split}.jsonl"))]
    if a.limit:
        tasks = tasks[:a.limit]
    tok, model = load(a.adapter, threads=a.threads)
    policy = QwenPolicy(tok, model)
    out_path = os.path.join(RUNS, f"p2s_eval_{a.name}_{a.split}.jsonl")
    done = {json.loads(l)["task_id"] for l in open(out_path)} if os.path.exists(out_path) else set()
    worlds = {}
    with open(out_path, "a") as f:
        for i, t in enumerate(tasks):
            if t["task_id"] in done:
                continue
            w = worlds.get(t["world"]) or worlds.setdefault(t["world"], World(t["world"]))
            t0 = time.time()
            r = run_episode(policy, w, t)
            r["latency_s"], r["gen_tokens"] = round(time.time() - t0, 1), policy.gen_tokens
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); f.flush()
            print(f"[p2s-{a.name}] {i + 1}/{len(tasks)} {t['task_id']} {t['kind']:<8} pass={r['pass']} score={r['score']:.2f} "
                  f"got={(r['submit'] or {}).get('decision')} calls={len(r['calls'])} {r['latency_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "check", "eval"])
    ap.add_argument("--name", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"gen": cmd_gen, "check": cmd_check, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
