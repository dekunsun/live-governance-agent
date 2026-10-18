"""Agent harness shared by teacher-trajectory export, SFT data, evaluation and (later) RL.

Design choices (see reports/posttrain_report.md):
  * One fixed, append-only text format for every turn. Each assistant turn is rendered as
    "<|im_start|>assistant\n<think>\n\n</think>\n\n{text}<|im_end|>" both in training and in inference,
    so the tokens the model is trained on are exactly the tokens it conditions on at inference.
  * Tool results are compacted into short Chinese summaries with the derived numbers a playbook step needs
    (today vs prior-7-day median, ratios, top contributors), so a small model is not drowned in raw rows.
  * The grader is deterministic: every criterion is checkable against planted ground truth, so the same
    grader is the SFT filter now and the RL reward later.
"""
import json
import os
import re
import sys
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform, Validator, TOOL_SPECS  # noqa: E402

TOOL_NAMES = ["query_metrics", "get_change_log", "score_drift", "sample_cases", "get_strategy_config",
              "get_guild_info", "get_guild_health"]
CAUSES = ["traffic_driven", "model_drift_threshold", "strategy_change", "new_strategy_launch", "policy_update",
          "recall_drop_evasion", "guild_concentration", "guild_health_real_violations", "guild_health_unfair_deduction",
          "no_anomaly", "insufficient_data", "unexplained", "needs_clarification"]
SUBMIT = {"name": "submit_card", "description": "提交最终归因卡片并结束。",
          "parameters": {"type": "object", "properties": {
              "cause_type": {"type": "string", "enum": CAUSES},
              "strategy_ids": {"type": "array", "items": {"type": "string"}},
              "change_ids": {"type": "array", "items": {"type": "string"}},
              "guild_ids": {"type": "array", "items": {"type": "string"}},
              "needs_escalation": {"type": "boolean"},
              "conclusion": {"type": "string"}},
              "required": ["cause_type", "strategy_ids", "change_ids", "guild_ids", "needs_escalation", "conclusion"]}}


def _short_schema(spec):
    props = {}
    for k, v in spec["input_schema"]["properties"].items():
        t = v.get("type")
        props[k] = {"type": t if isinstance(t, str) else t[0]}
        if "enum" in v:
            props[k]["enum"] = [e for e in v["enum"] if e is not None]
        if t == "array" or (isinstance(t, list) and "array" in t):
            props[k]["items"] = v.get("items", {"type": "string"})
    return {"type": "object", "properties": props, "required": spec["input_schema"].get("required", [])}


KEEP = {"query_metrics": ["metric", "start_date", "end_date", "market", "scenario", "strategy_id", "guild_id", "group_by", "granularity"],
        "get_change_log": ["start_date", "end_date", "scenario", "market", "change_type"],
        "score_drift": ["scenario", "baseline_start", "baseline_end", "current_start", "current_end", "market"],
        "sample_cases": ["start_date", "end_date", "strategy_id", "market", "scenario", "guild_id", "n", "overturned"],
        "get_strategy_config": ["strategy_id", "status"],
        "get_guild_info": ["guild_id"],
        "get_guild_health": ["start_date", "end_date", "guild_id", "market"]}
DESC = {"query_metrics": "查治理指标的日序列或汇总，可按 scenario/strategy_id/guild_id 下钻。",
        "get_change_log": "查窗口内的策略、模型、policy 变更。", "score_drift": "比较两个窗口的模型分分布。",
        "sample_cases": "抽样典型 case。", "get_strategy_config": "查策略配置和版本。", "get_guild_info": "查公会信息。",
        "get_guild_health": "查公会健康分和扣分来源，或某市场越档公会。"}


def _tool(spec):
    sch = _short_schema(spec)
    sch["properties"] = {k: v for k, v in sch["properties"].items() if k in KEEP[spec["name"]]}
    if "group_by" in sch["properties"]:
        sch["properties"]["group_by"] = {"type": "array"}
    sch["required"] = [r for r in sch["required"] if r in sch["properties"]]
    return {"type": "function", "function": {"name": spec["name"], "description": DESC[spec["name"]], "parameters": sch}}


SUBMIT_SHORT = {"name": "submit_card", "description": "提交最终归因卡片并结束。",
                "parameters": {"type": "object", "properties": {k: ({"type": "array"} if k.endswith("_ids") else
                                                                     {"type": "boolean"} if k == "needs_escalation" else {"type": "string"})
                                                                 for k in SUBMIT["parameters"]["required"]},
                               "required": SUBMIT["parameters"]["required"]}}
TOOLS = [_tool(s) for s in TOOL_SPECS if s["name"] in TOOL_NAMES] + [{"type": "function", "function": SUBMIT_SHORT}]

SYSTEM = """你是直播治理平台的归因助手，只能通过工具读取数据，最后必须调用 submit_card 结束。
今天是 2026-10-16（“昨天”=2026-10-15），数据范围 2026-09-18 ~ 2026-10-15。
市场: US 美国, BR 巴西, ID 印尼, GB 英国, VN 越南(小市场), KE 肯尼亚(小市场)。
场景: adult_content 色情低俗, minor_safety 未成年人安全, scam_fraud 诈骗引流, violence 暴力血腥, spam_bot 垃圾刷量, gambling 赌博, hate_harassment 仇恨骚扰, counterfeit 假冒侵权。
指标: 处罚=enforcements, 命中=hits, 人审=manual_reviews, 申诉=appeals, 申诉推翻/误杀申诉=overturns, 举报=reports, 召回=recall_est, 精确率=precision_est; “误杀率”有多个口径，需选定并说明分母。
步骤: 1 缺市场或日期时直接提交 needs_clarification。2 先查 sessions（流量）和该指标的日序列：流量同比例变化而比率不变=traffic_driven；当日与前7天中位数差异不显著=no_anomaly；小市场样本少=insufficient_data 并升级。3 按场景、策略、公会下钻找贡献最大的部分。4 查同期变更：模型升级且分数漂移=model_drift_threshold；新策略上线=new_strategy_launch；policy 更新后策略调整=policy_update；命中下降但举报不降且被举报人群分数下移=recall_drop_evasion；无变更且增量集中在单一公会=guild_concentration。5 抽样 case 作证据。
cause_type 取值: traffic_driven, model_drift_threshold, strategy_change, new_strategy_launch, policy_update, recall_drop_evasion, guild_concentration, guild_health_real_violations, guild_health_unfair_deduction, no_anomaly, insufficient_data, unexplained, needs_clarification。
公会健康分问题用 get_guild_health：扣分来源策略受模型漂移影响=guild_health_unfair_deduction，否则=guild_health_real_violations。
升级(needs_escalation=true): 小市场或样本不足、抽样含头部主播、公会级处置、无法解释。只引用工具返回过的编号，不要编造。每轮只调用一个工具。"""

THINK = "<think>\n\n</think>\n\n"


# ----------------------------------------------------------------------------- compaction
def _md(d):
    return d[5:] if isinstance(d, str) and len(d) == 10 else d


def _series_stats(vals, last_key):
    cur = vals.get(last_key)
    prior = [v for k, v in vals.items() if k < last_key and v is not None]
    base = median(prior) if prior else None
    return cur, base


def compact(tool, args, res, max_chars=300):
    if not isinstance(res, dict):
        return str(res)[:max_chars]
    if "error" in res:
        return f"错误 {res['error']}: {res.get('message', '')} {('提示: ' + res['hint']) if res.get('hint') else ''}"[:max_chars]
    rid = res.get("ref_id", "")
    if tool == "query_metrics":
        m, f, rows, gb = res["metric"], res["filters"], res["rows"], res["group_by"]
        mk = f["market"] if f["market"] == "ALL" else ",".join(f["market"])
        scope = " ".join(x for x in (mk if mk != "ALL" else "全市场", f.get("scenario"), f.get("strategy_id"), f.get("guild_id")) if x)
        head = f"{rid} {m}({res['metric_zh']}) {scope} {_md(f['start_date'])}~{_md(f['end_date'])}"
        keys = [g for g in gb if g != "date"]
        if "date" in gb and not keys:
            vals = {r["date"]: r["value"] for r in rows}
            last = max(vals)
            cur, base = _series_stats(vals, last)
            ser = " ".join(f"{_md(k)}:{v}" for k, v in sorted(vals.items())[-8:])
            ratio = f" ×{cur / base:.2f}" if (cur is not None and base) else ""
            extra = ""
            lr = next((r for r in rows if r["date"] == last), {})
            if "denominator" in lr:
                extra = f" 分母={lr['denominator']}"
            return f"{head} 日序列 {ser} | 当日={cur} 前7天中位数={base}{ratio}{extra}"[:max_chars]
        if "date" in gb and keys:
            k = keys[0]
            by = {}
            for r in rows:
                by.setdefault(r.get(k), {})[r["date"]] = r["value"] or 0
            last = max(r["date"] for r in rows)
            items = []
            for key, ser in by.items():
                cur, base = _series_stats(ser, last)
                items.append((key, cur or 0, base or 0))
            items.sort(key=lambda x: -abs(x[1] - x[2]))
            body = "; ".join(f"{key} {c} vs {b:g} ({c - b:+g})" for key, c, b in items[:6])
            return f"{head} 按{k}下钻（{_md(last)} vs 前7天中位数）: {body}"[:max_chars]
        if keys:
            k = keys[0]
            body = "; ".join(f"{r.get(k)}={r['value']}" + (" 低样本" if r.get("low_sample") else "") for r in rows[:8])
            return f"{head} 汇总按{k}: {body}"[:max_chars]
        r = rows[0] if rows else {}
        extra = "".join(f" {kk}={r[kk]}" for kk in ("n", "precision_est", "hit_sessions", "missed_est", "numerator", "denominator") if kk in r)
        return f"{head} 汇总 value={r.get('value')}{extra} low_sample={r.get('low_sample')}"[:max_chars]
    if tool == "get_change_log":
        ch = res["changes"]
        if not ch:
            return f"{rid} 窗口内无变更"
        body = "; ".join(f"{c['change_id']} {_md(c['date'])} {c['type']} {c['target']}（{c['scenario']}，{','.join(c['markets'])}）" for c in ch[:6])
        return f"{rid} 变更 {len(ch)} 条: {body}"[:max_chars]
    if tool == "score_drift":
        th = "; ".join(f"{t['strategy_id']}@{t['threshold']} {t['share_above_baseline']}→{t['share_above_current']}" + (f"(×{t['ratio']})" if t.get('ratio') else "")
                       for t in res["share_above_thresholds"][:4])
        rp = res.get("reported_population", {})
        rpb, rpc = rp.get("baseline", {}).get("p50"), rp.get("current", {}).get("p50")
        return (f"{rid} {res['model']} 版本 {','.join(res['baseline']['model_versions'])}→{','.join(res['current']['model_versions'])} "
                f"PSI={res['psi']} p50 {res['baseline']['p50']}→{res['current']['p50']}；超阈值比例 {th}；被举报人群 p50 {rpb}→{rpc}")[:max_chars]
    if tool == "sample_cases":
        return (f"{rid} 抽样 {res['n']}/{res['population']} 分数均值 {res['score_summary']['mean']} 含头部主播={res['contains_head_host']} "
                f"公会分布 {res.get('guild_share_in_population', {})}")[:max_chars]
    if tool == "get_strategy_config":
        body = "; ".join(f"{s['strategy_id']} {s['scenario']} 市场{','.join(s['markets'])} 阈值{s['threshold']} {s['action']} 版本 " +
                         ",".join(f"{v['version']}@{_md(v['effective_from'])}:{v['threshold']}" for v in s["versions"][-3:])
                         for s in res["strategies"][:5])
        return f"{rid} {body}"[:max_chars]
    if tool == "get_guild_info":
        body = "; ".join(f"{g['guild_id']} {g['name']} {g['tier']} 28天{g['sessions_28d']}场 头部主播{g['head_hosts']}" for g in res["guilds"][:5])
        return f"{rid} {body}"[:max_chars]
    if tool == "get_guild_health":
        if "guild_id" in res:
            ded = "; ".join(f"{d['strategy_id']}({d['scenario']}) {d['points']}" for d in res["deductions_by_strategy"][:4])
            nxt = res.get("next_threshold") or {}
            return (f"{rid} {res['guild_id']} 健康分 {res['score_start']}→{res['score_end']} 档位 {res['tier_now']} 扣分合计 {res['total_deduction']} "
                    f"恢复 {res['total_recovery']}；扣分来源: {ded}；下一档 {nxt.get('score')} {nxt.get('action')}")[:max_chars]
        br = "; ".join(f"{b['guild_id']} {b['score']} {b['tier']}" for b in res["breached"][:6]) or "无"
        return f"{rid} 越档公会: {br}"[:max_chars]
    return json.dumps(res, ensure_ascii=False)[:max_chars]


RATIONALE = {
    ("query_metrics", "sessions"): "先查流量，判断是否流量驱动。",
    ("query_metrics", "reports"): "命中下降时看举报是否同步下降。",
    ("query_metrics", "recall_est"): "估计召回变化。",
    ("query_metrics", "precision_est"): "看精确率变化。",
    ("query_metrics", "frozen_amount"): "看资金侧处置金额。",
    ("get_change_log", None): "查同期变更。",
    ("score_drift", None): "检查模型分数分布是否漂移。",
    ("sample_cases", None): "抽样 case 作为证据。",
    ("get_strategy_config", None): "查策略版本历史。",
    ("get_guild_info", None): "查公会信息。",
    ("get_guild_health", None): "查公会健康分和扣分来源。",
}


def rationale(tool, args):
    if tool == "query_metrics":
        gb = [g for g in (args.get("group_by") or []) if g != "date"]
        if gb:
            return {"scenario": "按场景下钻。", "strategy_id": "按策略下钻，找贡献最大的策略。", "guild_id": "按公会下钻。",
                    "market": "按市场汇总。"}.get(gb[0], "下钻。")
        return RATIONALE.get(("query_metrics", args.get("metric")), "查该指标的日序列和基线。")
    return RATIONALE.get((tool, None), "")


# ----------------------------------------------------------------------------- rendering
def assistant_text(tool, args, thought=None):
    args = {k: v for k, v in args.items() if v is not None}
    t = thought if thought is not None else rationale(tool, args)
    return f"{t}\n<tool_call>\n{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


def tool_turn(obs):
    return f"<|im_start|>user\n<tool_response>\n{obs}\n</tool_response><|im_end|>\n<|im_start|>assistant\n{THINK}"


def obs_segment(obs):
    """Text appended after the model's <|im_end|>: identical in training and inference."""
    return "\n" + tool_turn(obs)


def prefix_text(tok, query):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_call(text):
    m = TOOL_RE.search(text)
    if not m:
        return None, "no_tool_call"
    try:
        obj = json.loads(m.group(1))
        name, args = obj.get("name"), obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if name not in TOOL_NAMES + ["submit_card"] or not isinstance(args, dict):
            return None, "bad_tool"
        return (name, args), None
    except Exception:
        return None, "bad_json"


# ----------------------------------------------------------------------------- episode + grading
class Env:
    """One world. Executes tool calls and keeps the episode log."""

    def __init__(self, platform):
        self.p = platform
        self.p.refs.clear()
        self.p._ref_n = 0
        self.calls = []

    def step(self, name, args):
        clean = {k: v for k, v in args.items() if v is not None}
        res = self.p.call(name, clean)
        obs = compact(name, clean, res)
        self.calls.append({"tool": name, "args": clean, "error": res.get("error") if isinstance(res, dict) else None,
                           "ref_id": res.get("ref_id") if isinstance(res, dict) else None})
        return obs


def grade(task, submit, calls, platform, format_errors=0):
    """Deterministic criteria. Returns dict of criterion -> bool (None = not applicable)."""
    exp = task["expected"]
    c = {}
    c["submitted"] = submit is not None
    s = submit or {}
    c["cause_correct"] = s.get("cause_type") in exp["accepted_cause_types"]
    c["strategy_ids"] = set(exp.get("required_strategy_ids", [])) <= set(s.get("strategy_ids") or [])
    c["change_ids"] = set(exp.get("required_change_ids", [])) <= set(s.get("change_ids") or [])
    c["guild_ids"] = set(exp.get("required_guild_ids", [])) <= set(s.get("guild_ids") or [])
    c["escalation"] = None if exp.get("needs_escalation") is None else (bool(s.get("needs_escalation")) == exp["needs_escalation"])
    refs = {k["ref_id"]: 1 for k in calls if k.get("ref_id")}
    unknown = Validator(platform).check_text(json.dumps(s, ensure_ascii=False), refs) if submit else []
    unknown = [u for u in unknown if u[1] not in task["query"]]
    known = platform.known
    for field, kind in (("strategy_ids", "strategy_id"), ("change_ids", "change_id"), ("guild_ids", "guild_id")):
        for v in (s.get(field) or []):
            if str(v) not in known[kind] and (kind, str(v)) not in unknown:
                unknown.append((kind, str(v)))
    c["no_hallucination"] = not unknown if submit else False
    if exp["accepted_cause_types"][0] in ("needs_clarification", "guild_health_real_violations", "guild_health_unfair_deduction"):
        c["checked_traffic_first"] = None
    else:
        c["checked_traffic_first"] = any(k["tool"] == "query_metrics" and k["args"].get("metric") == "sessions" for k in calls)
    applicable = {k: v for k, v in c.items() if v is not None}
    passed = all(applicable.values())
    return {"criteria": c, "pass": passed, "score": sum(applicable.values()) / len(applicable), "unknown_refs": unknown,
            "format_errors": format_errors}


def run_episode(policy, platform, task, max_turns=14):
    """policy.start(prefix_text) -> str ; policy.continue_(append_text) -> str (generated assistant text)."""
    env = Env(platform)
    out = policy.start(task["query"])
    transcript = [{"role": "assistant", "text": out}]
    submit, fmt_err = None, 0
    for turn in range(max_turns):
        call, err = parse_call(out)
        if err:
            fmt_err += 1
            obs = f"格式错误 {err}：请用 <tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call> 调用一个工具。"
        elif call[0] == "submit_card":
            submit = call[1]
            break
        else:
            obs = env.step(*call)
        transcript.append({"role": "tool", "text": obs})
        if turn == max_turns - 1:
            break
        out = policy.continue_(obs_segment(obs))
        transcript.append({"role": "assistant", "text": out})
    g = grade(task, submit, env.calls, platform, fmt_err)
    return {"task_id": task["task_id"], "submit": submit, "calls": env.calls, "transcript": transcript, **g}
