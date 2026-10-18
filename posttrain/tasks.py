#!/usr/bin/env python3
"""Build tasks per world (planted ground truth), validate each with the rules engine, export teacher trajectories.

Outputs posttrain/runs/tasks_{train,test}.jsonl and posttrain/runs/sft_train.jsonl.
A task is kept only if the rules engine, run on the canonical phrasing, reaches the planted answer
(the Mercor-style "baselining" check that a task is solvable and its label fair).
"""
import json
import os
import random
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform  # noqa: E402
from agent.run import RulesEngine  # noqa: E402
from posttrain.harness import TOOL_NAMES, assistant_text, compact  # noqa: E402

RUNS = os.path.join(ROOT, "posttrain", "runs")
WORLDS = os.path.join(ROOT, "data", "worlds")
MK = {"US": "美国", "BR": "巴西", "ID": "印尼", "GB": "英国", "VN": "越南", "KE": "肯尼亚"}
SC = {"adult_content": "色情低俗", "minor_safety": "未成年人安全", "scam_fraud": "诈骗引流", "violence": "暴力血腥",
      "spam_bot": "垃圾刷量", "gambling": "赌博", "hate_harassment": "仇恨骚扰", "counterfeit": "假冒侵权"}
HIGH = {"adult_content": "S-101", "scam_fraud": "S-102", "minor_safety": "S-105", "violence": "S-107",
        "gambling": "S-115", "spam_bot": "S-117", "hate_harassment": "S-123", "counterfeit": "S-109"}
T = {  # template set A (train and test) / B (test only)
    "enf_up": (["{M}{S}{D}处罚涨了", "{M} {S} {D} 处罚上升了，帮我看看", "{D}{M}的{S}处罚量变多了"],
               ["{S}在{M}{D}的处罚怎么突然多了？", "能帮我查一下{M}{S}{D}处罚为什么上涨吗"]),
    "overturn_up": (["{M}{D}误杀申诉涨了，帮我看看", "{M} {D} 申诉推翻涨了"], ["{M}{D}被推翻的申诉是不是变多了"]),
    "recall": (["{M}{S}命中最近下降了", "{M} {S} 最近命中下降，是违规变少了吗"], ["{S}在{M}最近命中少了很多，什么原因"]),
    "traffic": (["{M} {D} 处罚量涨了", "{M}{D}处罚涨了很多，策略是不是出问题了"], ["{D}{M}处罚为什么多了这么多"]),
    "policy": (["{M} 骚扰 {D} 人审量涨了", "{M}仇恨骚扰{D}人审上升"], ["{D}{M}仇恨骚扰转人审怎么变多了"]),
    "guild_health": (["{G} 健康分为什么掉了", "{G} 健康分最近为什么扣分"], ["帮我分析下 {G} 的健康分为什么下降"]),
    "small": (["{M}昨天误杀申诉涨了", "{M} 昨天处罚涨了"], ["{M}昨天申诉推翻怎么多了"]),
    "clar": (["昨天{S}误杀申诉涨了", "{S}最近处罚涨了，帮我看看"], ["申诉推翻最近好像多了"]),
}


def fmt_date(d, rng):
    if d == "2026-10-15" and rng.random() < 0.6:
        return "昨天"
    y, m, dd = d.split("-")
    return rng.choice([d, f"{int(m)}-{int(dd)}", f"{int(m)}月{int(dd)}日"])


def plus(d, n):
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def render(kind, which, rng, **kw):
    tpl = rng.choice(T[kind][0] if which == "A" else T[kind][1])
    fill = {"M": rng.choice([kw.get("m", ""), MK.get(kw.get("m"), "")]) if kw.get("m") else "",
            "S": SC.get(kw.get("s"), ""), "D": fmt_date(kw["d"], rng) if kw.get("d") else "", "G": kw.get("g", "")}
    return tpl.format(**fill)


def candidates(spec, rng):
    """Yield (kind, expected, kwargs, alt_dates) for one world."""
    d1, s1 = spec["drift"]["from"], spec["drift"]["scenario"]
    big = ["US", "BR", "ID", "GB"]
    out = []
    m = rng.choice(big)
    out.append(("enf_up", {"accepted_cause_types": ["model_drift_threshold"], "required_strategy_ids": [HIGH[s1]],
                           "required_change_ids": ["C-006"]}, {"m": m, "s": s1}, [plus(d1, 1), plus(d1, 2), "2026-10-15"]))
    out.append(("overturn_up", {"accepted_cause_types": ["model_drift_threshold"], "required_strategy_ids": [HIGH[s1]],
                                "required_change_ids": ["C-006"]}, {"m": rng.choice(big)}, ["2026-10-15", plus(d1, 2), plus(d1, 1)]))
    r = spec["recall"]
    out.append(("recall", {"accepted_cause_types": ["recall_drop_evasion"], "required_strategy_ids": [HIGH[r["scenario"]]]},
                {"m": r["market"], "s": r["scenario"]}, [None]))
    tr = spec["traffic"]
    out.append(("traffic", {"accepted_cause_types": ["traffic_driven"]}, {"m": tr["market"]}, list(tr["dates"])))
    pm = spec["policy"]["market"]
    out.append(("policy", {"accepted_cause_types": ["policy_update", "strategy_change"], "required_strategy_ids": ["S-112"],
                           "required_change_ids": ["C-004"]}, {"m": pm}, ["2026-10-12", "2026-10-13", "2026-10-14"]))
    L = spec["launch"]
    out.append(("enf_up", {"accepted_cause_types": ["new_strategy_launch"], "required_strategy_ids": ["S-126"],
                           "required_change_ids": ["C-005"]}, {"m": L["market"], "s": L["scenario"]},
                [plus(L["date"], 1), plus(L["date"], 2), L["date"]]))
    g = spec["guild"]
    out.append(("enf_up", {"accepted_cause_types": ["guild_concentration"], "required_guild_ids": [g["guild_id"]]},
                {"m": g["market"], "s": g["scenario"]}, ["2026-10-15", "2026-10-14"]))
    out.append(("guild_health", {"accepted_cause_types": ["guild_health_real_violations"], "required_guild_ids": [g["guild_id"]]},
                {"g": g["guild_id"]}, ["2026-10-15"]))
    planted = {(r["market"], r["scenario"]), (L["market"], L["scenario"]), (g["market"], g["scenario"]), (pm, "hate_harassment")}
    quiet = [(mm, ss) for mm in big for ss in SC if ss != s1 and (mm, ss) not in planted and ss != "hate_harassment"]
    for mm, ss in rng.sample(quiet, 3):
        out.append(("enf_up", {"accepted_cause_types": ["no_anomaly"]}, {"m": mm, "s": ss}, ["2026-10-15", "2026-10-14"]))
    out.append(("small", {"accepted_cause_types": ["insufficient_data"]}, {"m": rng.choice(["VN", "KE"])}, [None]))
    out.append(("clar", {"accepted_cause_types": ["needs_clarification"]}, {"s": rng.choice(list(SC))}, [None]))
    return out


def teacher_messages(trace):
    """Rules-engine trace -> list of (assistant_text, observation). Final element has observation None."""
    turns = []
    for e in trace.events:
        if e["kind"] != "tool_call" or e["tool"] not in TOOL_NAMES:
            continue
        args = {k: v for k, v in e["args"].items() if v is not None}
        turns.append((assistant_text(e["tool"], args), compact(e["tool"], args, e["result"])))
    c = trace.card
    pc = c.get("primary_cause") or {}
    sub = {"cause_type": pc.get("type"), "strategy_ids": pc.get("strategy_ids") or [], "change_ids": pc.get("change_ids") or [],
           "guild_ids": pc.get("guild_ids") or [], "needs_escalation": bool(c.get("needs_escalation")),
           "conclusion": (c.get("conclusion") or "")[:140]}
    turns.append((assistant_text("submit_card", sub, thought="证据充分，提交结论。" if sub["cause_type"] != "needs_clarification" else "缺少市场或日期，先澄清。"), None))
    return turns, sub


def matches(exp, sub):
    return (sub["cause_type"] in exp["accepted_cause_types"]
            and set(exp.get("required_strategy_ids", [])) <= set(sub["strategy_ids"])
            and set(exp.get("required_change_ids", [])) <= set(sub["change_ids"])
            and set(exp.get("required_guild_ids", [])) <= set(sub["guild_ids"]))


def build_world(spec, rng, stats):
    wdir = os.path.join(WORLDS, spec["world_id"])
    p = Platform(data_dir=wdir)
    eng = RulesEngine(p)
    tasks, sft = [], []
    for i, (kind, exp, kw, dates) in enumerate(candidates(spec, rng)):
        stats["candidates"] += 1
        ok = None
        for d in dates:
            q = render(kind, "A", rng, d=d, **kw)
            p.refs.clear(); p._ref_n = 0
            tr = eng.attribution(q)
            turns, sub = teacher_messages(tr)
            if matches(exp, sub):
                ok = (d, q, tr, turns, sub)
                break
        if not ok:
            stats["dropped"] += 1
            stats.setdefault("dropped_by_kind", {}).setdefault(exp["accepted_cause_types"][0], 0)
            stats["dropped_by_kind"][exp["accepted_cause_types"][0]] += 1
            continue
        d, q, tr, turns, sub = ok
        exp = dict(exp, needs_escalation=None if kind == "clar" else sub["needs_escalation"])
        phr = "A"
        if spec["split"] == "test" and rng.random() < 0.5:
            q = render(kind, "B", rng, d=d, **kw)
            phr = "B"
        task = {"task_id": f"{spec['world_id']}-{i:02d}", "world": spec["world_id"], "split": spec["split"], "kind": kind,
                "phrasing": phr, "query": q, "date": d, "expected": exp, "teacher_submit": sub, "teacher_tool_calls": len(turns) - 1}
        tasks.append(task)
        if spec["split"] == "train":
            sft.append({"task_id": task["task_id"], "query": q, "turns": turns})
    return tasks, sft


def main():
    os.makedirs(RUNS, exist_ok=True)
    idx = json.load(open(os.path.join(WORLDS, "index.json")))
    rng = random.Random(11)
    stats = {"candidates": 0, "dropped": 0}
    all_tasks, all_sft = [], []
    for spec in idx:
        t, s = build_world(spec, rng, stats)
        all_tasks += t
        all_sft += s
        print(spec["world_id"], spec["split"], "tasks", len(t), flush=True)
    for split in ("train", "test"):
        with open(os.path.join(RUNS, f"tasks_{split}.jsonl"), "w") as f:
            for t in all_tasks:
                if t["split"] == split:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(RUNS, "sft_train.jsonl"), "w") as f:
        for s in all_sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    stats["train_tasks"] = sum(t["split"] == "train" for t in all_tasks)
    stats["test_tasks"] = sum(t["split"] == "test" for t in all_tasks)
    json.dump(stats, open(os.path.join(RUNS, "task_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
