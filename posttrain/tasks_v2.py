#!/usr/bin/env python3
"""Harness v2 demonstration data: the same train tasks as tasks.py, but

  * every assistant turn's thought is derived from the PREVIOUS observation (evidence -> decision), so the
    decision token has something to condition on instead of its position in the trajectory;
  * an extra training phrasing set C (B stays held out for test);
  * the two known v1 flaws are fixed (frozen_amount step rationale; secondary cause kept in the conclusion);
  * each sample records the parsed scope hint that harness v2 puts in the prompt.

Test tasks are untouched: posttrain/runs/tasks_test.jsonl is the comparison set for every harness version.

  python posttrain/tasks_v2.py        -> posttrain/runs/sft_train_v2.jsonl (+ sft_v2_stats.json)
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform  # noqa: E402
from agent.run import RulesEngine  # noqa: E402
from posttrain.harness import TOOL_NAMES, compact  # noqa: E402
from posttrain.harness_v2 import parse_scope  # noqa: E402
from posttrain import tasks as T1  # noqa: E402

RUNS = os.path.join(ROOT, "posttrain", "runs")
WORLDS = os.path.join(ROOT, "data", "worlds")

TC = {  # template set C: training only, wording distinct from A and B
    "enf_up": ["{M}{S}{D}的处罚量上来了，看下原因", "帮忙看看{M}{D}{S}处罚增加是怎么回事"],
    "overturn_up": ["{M}{D}误杀申诉数上来了，查一下", "{M}{D}申诉被推翻的变多了，什么情况"],
    "recall": ["{M}{S}这几天命中掉了不少", "{M} {S} 命中数在下降，帮我看看"],
    "traffic": ["{M}{D}处罚一下多了很多，是策略的问题吗", "{M} {D} 处罚量突然上升"],
    "policy": ["{M}仇恨骚扰{D}转人审的量上来了", "{M} 骚扰 {D} 人审增加了，看看"],
    "guild_health": ["{G} 的健康分在掉，查下原因", "看一下 {G} 健康分扣分来源"],
    "small": ["{M}昨天申诉推翻涨了", "{M}昨天处罚量上升"],
    "clar": ["{S}误杀申诉涨了", "最近{S}处罚涨了"],
}
CAUSE_ZH = {"traffic_driven": "流量驱动", "model_drift_threshold": "模型升级后分数漂移、固定阈值失效", "strategy_change": "策略版本变更",
            "new_strategy_launch": "新策略上线", "policy_update": "policy 更新（预期内）", "recall_drop_evasion": "对抗绕过导致召回下降",
            "guild_concentration": "增量集中在单一公会", "guild_health_real_violations": "公会扣分来自真实违规",
            "guild_health_unfair_deduction": "公会扣分来自策略误杀", "no_anomaly": "正常波动", "insufficient_data": "样本不足",
            "unexplained": "无法解释", "needs_clarification": "需要澄清"}


def render_c(kind, rng, **kw):
    tpl = rng.choice(TC[kind])
    fill = {"M": rng.choice([kw.get("m", ""), T1.MK.get(kw.get("m"), "")]) if kw.get("m") else "",
            "S": T1.SC.get(kw.get("s"), ""), "D": T1.fmt_date(kw["d"], rng) if kw.get("d") else "", "G": kw.get("g", "")}
    return tpl.format(**fill)


# ----------------------------------------------------------------------------- evidence -> decision thoughts
def _ratio(obs):
    m = re.search(r"×(\d+\.\d+)", obs)
    return float(m.group(1)) if m else None


def evidence(prev_tool, prev_args, prev_obs):
    """One short clause summarising what the previous observation showed."""
    if prev_obs is None:
        return ""
    o = prev_obs
    if prev_tool == "query_metrics":
        metric = prev_args.get("metric")
        gb = [g for g in (prev_args.get("group_by") or []) if g != "date"]
        if gb:
            m = re.search(r"下钻[^:：]*[:：]\s*([^;]+)", o)
            top = m.group(1).strip() if m else ""
            return f"按{gb[0]}拆开后 {top} 贡献最大" if top else f"按{gb[0]}拆开看到增量来源"
        r = _ratio(o)
        if metric == "sessions":
            if r is None:
                return "流量数据已拿到"
            return f"流量 ×{r:.2f}" + ("，基本持平" if 0.85 <= r <= 1.15 else "，变化明显")
        if metric == "reports":
            return "举报" + ("未同步下降" if "未" in o or (r and r >= 0.9) else "有变化")
        if metric in ("precision_est", "recall_est"):
            m = re.search(r"value=([\d.]+)", o)
            return f"{'精确率' if metric == 'precision_est' else '召回估计'}={m.group(1)}" if m else f"{metric} 已拿到"
        if metric == "frozen_amount":
            return "资金侧策略的冻结金额已查到"
        if r is not None:
            cur = re.search(r"当日=([\d.]+)", o)
            base = re.search(r"中位数=([\d.]+)", o)
            lvl = "明显异常" if r >= 1.5 or r <= 0.67 else "在正常波动内"
            return f"{metric} 当日 {cur.group(1) if cur else '?'} vs 基线 {base.group(1) if base else '?'}（×{r:.2f}，{lvl}）"
        return f"{metric} 序列已拿到"
    if prev_tool == "get_change_log":
        if "无变更" in o:
            return "同期无策略或模型变更"
        ids = re.findall(r"(C-\d{3}) (\d\d-\d\d) (\w+)", o)
        return "同期变更 " + "、".join(f"{c}（{t}，{d}）" for c, d, t in ids[:3]) if ids else "同期有变更"
    if prev_tool == "score_drift":
        psi = re.search(r"PSI=([\d.]+)", o)
        p50 = re.search(r"p50 ([\d.]+)→([\d.]+)", o)
        x = re.search(r"\(×([\d.]+)\)", o)
        parts = []
        if psi:
            parts.append(f"PSI={psi.group(1)}")
        if p50:
            parts.append(f"中位数 {p50.group(1)}→{p50.group(2)}")
        if x:
            parts.append(f"超阈值比例 ×{x.group(1)}")
        big = psi and float(psi.group(1)) > 0.25
        return ("分数分布" + ("明显上移" if big else "没有明显漂移") + ("（" + "，".join(parts) + "）" if parts else ""))
    if prev_tool == "sample_cases":
        head = "含头部主播" if "含头部主播=True" in o else "不含头部主播"
        m = re.search(r"分数均值 ([\d.]+)", o)
        return f"抽样{head}" + (f"，分数均值 {m.group(1)}" if m else "")
    if prev_tool == "get_strategy_config":
        return "策略版本历史已查到"
    if prev_tool == "get_guild_info":
        return "公会信息已查到"
    if prev_tool == "get_guild_health":
        m = re.search(r"扣分来源: ([^；]+)", o)
        return f"扣分来源 {m.group(1).split(';')[0].strip()} 为主" if m else "健康分和扣分来源已查到"
    return ""


def decision(tool, args):
    if tool == "query_metrics":
        gb = [g for g in (args.get("group_by") or []) if g != "date"]
        metric = args.get("metric")
        if metric == "frozen_amount":
            return "看同模型资金侧策略的冻结金额"
        if gb:
            return {"scenario": "按场景拆开找增量来源", "strategy_id": "按策略拆开找贡献最大的策略", "guild_id": "按公会拆开看是否集中",
                    "market": "按市场汇总"}.get(gb[0], "下钻")
        return {"sessions": "先查流量，排除流量因素", "reports": "看举报是否同步下降", "recall_est": "估计召回变化",
                "precision_est": "看精确率变化"}.get(metric, f"查 {metric} 的日序列和基线")
    return {"get_change_log": "查同期变更", "score_drift": "检查模型分分布是否漂移", "sample_cases": "抽样 case 作证据",
            "get_strategy_config": "查策略版本历史", "get_guild_info": "查公会信息", "get_guild_health": "查健康分和扣分来源"}.get(tool, tool)


def thought_for(prev, tool, args):
    ev = evidence(*prev) if prev else ""
    d = decision(tool, args)
    return f"{ev}，{d}。" if ev else f"{d}。"


def assistant_text(tool, args, thought):
    return f"{thought}\n<tool_call>\n{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


def teacher_turns(trace):
    calls = [e for e in trace.events if e["kind"] == "tool_call" and e["tool"] in TOOL_NAMES]
    turns, prev = [], None
    for e in calls:
        args = {k: v for k, v in e["args"].items() if v is not None}
        obs = compact(e["tool"], args, e["result"])
        turns.append((assistant_text(e["tool"], args, thought_for(prev, e["tool"], args)), obs))
        prev = (e["tool"], args, obs)
    c = trace.card
    pc = c.get("primary_cause") or {}
    concl = (c.get("conclusion") or "")[:140]
    sec = [x for x in (c.get("contributing_causes") or []) if x.get("type")]
    if sec:
        concl = (concl + "；次要：" + "、".join(f"{x['type']} {','.join(x.get('strategy_ids') or [])}".strip() for x in sec[:2]))[:200]
    sub = {"cause_type": pc.get("type"), "strategy_ids": pc.get("strategy_ids") or [], "change_ids": pc.get("change_ids") or [],
           "guild_ids": pc.get("guild_ids") or [], "needs_escalation": bool(c.get("needs_escalation")), "conclusion": concl}
    if sub["cause_type"] == "needs_clarification":
        th = "缺少市场或日期，先澄清。"
    else:
        ev = evidence(*prev) if prev else ""
        why = (pc.get("explanation") or "")[:80]
        esc = "；" + "、".join(c.get("escalation_reasons") or [])[:60] + "，需升级" if c.get("needs_escalation") else ""
        th = f"{ev}。主因 {CAUSE_ZH.get(sub['cause_type'], sub['cause_type'])}：{why}{esc}。提交。"
    turns.append((assistant_text("submit_card", sub, th), None))
    return turns, sub


# ----------------------------------------------------------------------------- build (same task ids as tasks.py)
def main():
    idx = json.load(open(os.path.join(WORLDS, "index.json")))
    existing = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(RUNS, "tasks_train.jsonl"))}
    rng = random.Random(11)   # same stream as tasks.py so task ids and A-phrasings line up
    rng_c = random.Random(1102)
    stats = {"a": 0, "c_kept": 0, "c_dropped": 0, "mismatch_with_v1": 0}
    out = []
    for spec in idx:
        p = Platform(data_dir=os.path.join(WORLDS, spec["world_id"]))
        eng = RulesEngine(p)
        for i, (kind, exp, kw, dates) in enumerate(T1.candidates(spec, rng)):
            ok = None
            for d in dates:
                q = T1.render(kind, "A", rng, d=d, **kw)
                p.refs.clear(); p._ref_n = 0
                tr = eng.attribution(q)
                turns, sub = teacher_turns(tr)
                if T1.matches(exp, sub):
                    ok = (d, q, tr, turns, sub)
                    break
            if not ok:
                continue
            d, q, tr, turns, sub = ok
            if spec["split"] == "test":
                if rng.random() < 0.5:   # consume the same rng draws as tasks.py
                    T1.render(kind, "B", rng, d=d, **kw)
                continue
            tid = f"{spec['world_id']}-{i:02d}"
            if tid not in existing or existing[tid]["query"] != q:
                stats["mismatch_with_v1"] += 1
            out.append({"task_id": tid, "phrasing": "A", "query": q, "scope": parse_scope(q, p.meta), "turns": turns, "prefix": "v2"})
            stats["a"] += 1
            qc = render_c(kind, rng_c, d=d, **kw)
            p.refs.clear(); p._ref_n = 0
            trc = eng.attribution(qc)
            turns_c, sub_c = teacher_turns(trc)
            if T1.matches(exp, sub_c):
                out.append({"task_id": tid + "c", "phrasing": "C", "query": qc, "scope": parse_scope(qc, p.meta), "turns": turns_c, "prefix": "v2"})
                stats["c_kept"] += 1
            else:
                stats["c_dropped"] += 1
        print(spec["world_id"], spec["split"], len(out), flush=True)
    with open(os.path.join(RUNS, "sft_train_v2.jsonl"), "w") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump(stats, open(os.path.join(RUNS, "sft_v2_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
