"""Harness v2: the v1 harness plus the fixes the base-model failure triage pointed to (reports/posttrain_report.md §2).

v1 stays frozen in harness.py so results stay comparable; everything here is additive:
  * lenient parsing: a missing </tool_call> is accepted; a cause name called as a tool becomes submit_card
  * argument normalisation before execution: Chinese scenario/market/metric names, "处罚=enforcements" style
    values, empty strings, reversed or out-of-range date windows, missing score_drift windows
  * every normalisation and error is reported back with a call the model can copy, and a repeated identical
    error is blocked instead of executed again
  * a submit gate: cause_type must be a listed cause, traffic (sessions) must be checked first, and every id
    must have appeared in an observation; each gate rejects at most twice, then the submission is accepted
  * the system prompt carries the deterministically parsed scope (market / date / metric) as a hint
The grader is unchanged (harness.grade), so v1 and v2 scores measure the same thing.
"""
import json
import re
from datetime import date, timedelta

from posttrain.harness import (CAUSES, SYSTEM, TOOL_NAMES, TOOLS, Env, compact, grade, obs_segment)  # noqa: F401

HARNESS_VERSION = "v2"
DATA_START, DATA_END = "2026-09-18", "2026-10-15"
DIMS = ("strategy_id", "guild_id", "market", "scenario")
EXEMPT_TRAFFIC = {"needs_clarification", "guild_health_real_violations", "guild_health_unfair_deduction"}
ID_RE = {"strategy_ids": r"S-\d{3}", "change_ids": r"C-\d{3}", "guild_ids": r"G-[A-Z]{2}-\d{3}"}
MAX_GATE_REJECTS = 2

SYSTEM_V2 = SYSTEM.replace(
    "指标: 处罚=enforcements, 命中=hits, 人审=manual_reviews, 申诉=appeals, 申诉推翻/误杀申诉=overturns, 举报=reports, 召回=recall_est, 精确率=precision_est;",
    "指标（参数里只写英文名）: enforcements 处罚, hits 命中, manual_reviews 人审, appeals 申诉, overturns 申诉推翻/误杀申诉, reports 举报, "
    "sessions 流量, recall_est 召回, precision_est 精确率, frozen_amount 冻结金额;") + """
调用格式（每轮一个）: <tool_call>{"name": "工具名", "arguments": {...}}</tool_call>。scenario、market 参数只写英文代码。
提交前必须已经用 query_metrics 查过 sessions（澄清和公会健康分问题除外）；strategy_ids/change_ids/guild_ids 只能填工具结果里出现过的编号，没有就填 []。"""


# ----------------------------------------------------------------------------- parsing
def parse_call(text):
    """Returns ((name, args), None, note) or (None, err, None)."""
    i = text.find("<tool_call>")
    body = text[i + len("<tool_call>"):] if i >= 0 else text
    j = body.find("{")
    if j < 0:
        return None, "no_tool_call", None
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[j:])
    except Exception:
        return None, "bad_json", None
    if not isinstance(obj, dict):
        return None, "bad_json", None
    name, args = obj.get("name"), obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None, "bad_json", None
    if not isinstance(args, dict):
        return None, "bad_json", None
    note = None if (i >= 0 and "</tool_call>" in text) else "缺少 </tool_call>，已按完整调用处理"
    if name in CAUSES:   # a cause called as if it were a tool
        sub = {"cause_type": name, "strategy_ids": [], "change_ids": [], "guild_ids": [], "needs_escalation": False, "conclusion": ""}
        return ("submit_card", sub), None, f"{name} 不是工具，已按 submit_card(cause_type={name}) 处理"
    if name not in TOOL_NAMES + ["submit_card"]:
        return None, f"bad_tool:{name}", None
    return (name, args), None, note


# ----------------------------------------------------------------------------- argument normalisation
def _shift(d, n):
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def _clip(d):
    return min(max(d, DATA_START), DATA_END)


class Normalizer:
    def __init__(self, meta, metrics, aliases):
        self.scn = {v: k for k, v in meta["scenario_names"].items()}
        self.scn.update({v[:2]: k for k, v in meta["scenario_names"].items()})
        self.scn_keys = set(meta["scenario_names"])
        self.mkt = {v: k for k, v in meta["market_names"].items()}
        self.mkt_keys = set(meta["market_names"])
        self.metrics, self.aliases = set(metrics), aliases

    def scenario(self, v):
        if v in self.scn_keys or v is None:
            return v
        for name, key in self.scn.items():
            if name in str(v):
                return key
        return v

    def market(self, v):
        if v in self.mkt_keys or v is None:
            return v
        return self.mkt.get(v, next((c for c in self.mkt_keys if c in str(v)), v))

    def metric(self, v):
        if v in self.metrics or v is None:
            return v
        for part in re.split(r"[=＝/,，\s]+", str(v)):
            if part in self.metrics:
                return part
        for k, vs in self.aliases.items():
            if k in str(v) and len(vs) == 1:
                return vs[0]
        return v

    def __call__(self, tool, args, scope_date):
        a, notes = {}, []
        for k, v in args.items():
            if v is None or v == "" or v == []:
                continue
            a[k] = v
        for k, fn in (("scenario", self.scenario), ("market", self.market), ("metric", self.metric)):
            if k in a:
                new = fn(a[k])
                if new != a[k]:
                    notes.append(f"{k} {a[k]}→{new}")
                    a[k] = new
        if isinstance(a.get("group_by"), str):
            a["group_by"] = [a["group_by"]]
        if tool == "query_metrics" and a.get("metric") in DIMS:      # metric="strategy_id" means "break down by strategy"
            dim = a.pop("metric")
            a["group_by"] = sorted(set(a.get("group_by") or []) | {dim})
            a["metric"] = "enforcements"
            notes.append(f"metric={dim} 是维度不是指标，已改为 metric=enforcements, group_by={a['group_by']}")
        for s, e in (("start_date", "end_date"), ("baseline_start", "baseline_end"), ("current_start", "current_end")):
            for k in (s, e):
                if isinstance(a.get(k), str) and len(a[k]) == 10 and _clip(a[k]) != a[k]:
                    notes.append(f"{k} {a[k]}→{_clip(a[k])}（数据范围 {DATA_START}~{DATA_END}）")
                    a[k] = _clip(a[k])
            if a.get(s) and a.get(e) and a[s] > a[e]:
                notes.append(f"{s}/{e} 前后颠倒，已交换")
                a[s], a[e] = a[e], a[s]
        if tool == "score_drift" and scope_date:
            d = _clip(scope_date)
            want = {"current_end": d, "current_start": _shift(d, -2), "baseline_end": _shift(d, -3), "baseline_start": _shift(d, -9)}
            overlap = a.get("baseline_end") and a.get("current_start") and a["baseline_end"] >= a["current_start"]
            if overlap or any(k not in a for k in want):
                notes.append("score_drift 窗口缺失或重叠，已用默认窗口：基线 = 日期前 9~3 天，当前 = 日期前 2 天~当天")
                a.update({k: _clip(v) for k, v in want.items()})
        return a, notes


# ----------------------------------------------------------------------------- environment with guards
class EnvV2(Env):
    def __init__(self, platform, scope):
        super().__init__(platform)
        from gov_platform.tools import METRICS, ALIASES
        self.norm = Normalizer(platform.meta, METRICS, ALIASES)
        self.scope = scope
        self.errors = {}
        self.seen = {k: set() for k in ID_RE}
        self.notes = []
        self.blocked = 0
        self._see(scope.get("raw", ""))

    def _see(self, text):
        for k, pat in ID_RE.items():
            self.seen[k].update(re.findall(pat, text))

    def step(self, name, args):
        clean, notes = self.norm(name, args, self.scope.get("date"))
        key = (name, json.dumps(clean, sort_keys=True, ensure_ascii=False))
        if self.errors.get(key, 0) >= 2:
            self.blocked += 1
            return "已阻止：这个调用已经连续报错两次。换一个工具或参数，或者用已有证据调用 submit_card。"
        res = self.p.call(name, clean)
        obs = compact(name, clean, res)
        err = res.get("error") if isinstance(res, dict) else None
        self.calls.append({"tool": name, "args": clean, "error": err, "ref_id": res.get("ref_id") if isinstance(res, dict) else None,
                           "normalized": notes})
        if err:
            self.errors[key] = self.errors.get(key, 0) + 1
            obs += " | " + self._hint(name, clean, err)
        else:
            self._see(obs)
        if notes:
            self.notes += notes
            obs = "（已自动修正：" + "；".join(notes) + "）" + obs
        return obs

    def _hint(self, name, args, err):
        if err == "unknown_scenario":
            return "scenario 可选: " + ", ".join(sorted(self.norm.scn_keys))
        if err == "unknown_market":
            return "market 可选: " + ", ".join(sorted(self.norm.mkt_keys))
        if err == "unknown_metric":
            return "metric 可选: sessions, enforcements, hits, manual_reviews, appeals, overturns, reports, precision_est, recall_est, frozen_amount"
        if err in ("unknown_strategy", "unknown_guild"):
            k = "strategy_ids" if err == "unknown_strategy" else "guild_ids"
            seen = sorted(self.seen[k])
            return ("已出现过的编号: " + ", ".join(seen)) if seen else '先下钻拿到编号，例如 query_metrics(metric="enforcements", group_by=["strategy_id"], ...)' 
        if err == "bad_window":
            return f"日期格式 YYYY-MM-DD，范围 {DATA_START}~{DATA_END}，开始不晚于结束"
        return "检查参数后换一种调用"

    def gate(self, sub, rejects):
        """Return a rejection message, or None to accept."""
        cause = sub.get("cause_type")
        checks = []
        if cause not in CAUSES:
            checks.append(("enum", f"cause_type={cause!r} 不在可选值里。可选: " + ", ".join(CAUSES)))
        if cause not in EXEMPT_TRAFFIC and not any(c["tool"] == "query_metrics" and c["args"].get("metric") == "sessions" and not c["error"] for c in self.calls):
            sc = self.scope
            eg = {"metric": "sessions", "start_date": _shift(sc["date"], -7) if sc.get("date") else DATA_START,
                  "end_date": sc.get("date") or DATA_END, **({"market": sc["market"]} if sc.get("market") else {})}
            checks.append(("traffic", "提交前先查流量，用来排除流量因素：把流量的变化比例和该指标的变化比例对比，比例接近才是 traffic_driven，否则继续下钻找原因。先调用: "
                              + json.dumps({"name": "query_metrics", "arguments": eg}, ensure_ascii=False)))
        bad = {k: [v for v in (sub.get(k) or []) if str(v) not in self.seen[k]] for k in ID_RE}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            seen = "; ".join(f"{k}: {', '.join(sorted(self.seen[k])) or '无'}" for k in ID_RE)
            checks.append(("ids", f"这些编号没有在工具结果里出现过: {bad}。已出现过的: {seen}。没有就填 []。"))
        for kind, msg in checks:
            if rejects.get(kind, 0) < MAX_GATE_REJECTS:
                rejects[kind] = rejects.get(kind, 0) + 1
                return f"提交被退回（{kind}）: {msg}"
        return None


def prefix_text(tok, query, scope):
    hint = "系统预解析（可能有误，请核对）: " + "，".join(
        f"{k}={scope[k] if scope.get(k) else '未指定'}" for k in ("market", "date", "metric", "scenario"))
    msgs = [{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": f"{query}\n\n{hint}"}]
    return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_scope(query, meta):
    from agent.run import parse_query
    s = parse_query(query, meta)
    return {"market": s["market"], "date": s["date"], "metric": s["metric"], "scenario": s["scenario"], "raw": query}


def run_episode(policy, platform, task, max_turns=14):
    scope = parse_scope(task["query"], platform.meta)
    env = EnvV2(platform, scope)
    out = policy.start(prefix_text(policy.tok, task["query"], scope), raw=True)
    transcript = [{"role": "assistant", "text": out}]
    submit, fmt_err, rejects, parse_notes = None, 0, {}, []
    for turn in range(max_turns):
        call, err, note = parse_call(out)
        if note:
            parse_notes.append(note)
        if err:
            fmt_err += 1
            obs = (f"格式错误 {err}：工具名只能是 {', '.join(TOOL_NAMES)} 或 submit_card。"
                   if err.startswith("bad_tool") else f"格式错误 {err}：请用 <tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call> 调用一个工具。")
        elif call[0] == "submit_card":
            msg = env.gate(call[1], rejects)
            if msg is None:
                submit = call[1]
                break
            obs = msg
        else:
            obs = env.step(*call)
            if note:
                obs = f"（{note}）" + obs
        transcript.append({"role": "tool", "text": obs})
        if turn == max_turns - 1:
            break
        out = policy.continue_(obs_segment(obs))
        transcript.append({"role": "assistant", "text": out})
    g = grade(task, submit, env.calls, platform, fmt_err)
    return {"task_id": task["task_id"], "harness_version": HARNESS_VERSION, "submit": submit, "calls": env.calls,
            "transcript": transcript, "gate_rejects": rejects, "normalizations": env.notes, "parse_notes": parse_notes,
            "blocked_repeats": env.blocked, **g}
