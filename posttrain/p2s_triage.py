#!/usr/bin/env python3
"""Classify Policy-to-Strategy failures by the first thing that went wrong.

  python posttrain/p2s_triage.py --name base     # reads runs/p2s_eval_base_test.jsonl
"""
import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "posttrain/runs")

LABELS = {
    "H-format": ("harness", "没有提交：输出无法解析或门控退回后用完回合"),
    "H-args": ("harness", "工具参数错误（场景、分数线列表、策略编号）"),
    "H-ids": ("harness", "提交了不存在的策略编号"),
    "T-decision": ("training", "决策类型错（该不改却改了、该新建却改旧的等）"),
    "T-target": ("training", "改错了策略或市场范围"),
    "T-action": ("training", "动作错（没按 policy 改、或用了更重的动作）"),
    "T-financial": ("training", "资金侧审批标记错"),
    "T-threshold": ("training", "分数线在评分窗口不达标"),
    "T-process": ("training", "没回放就提交"),
}


def classify(r):
    c, s = r["criteria"], r.get("submit")
    if s is None:
        return "H-format"
    if not c.get("no_hallucination"):
        return "H-ids"
    if not c.get("decision_correct"):
        return "T-decision"
    for k, lab in (("target_correct", "T-target"), ("action_correct", "T-action"), ("financial_flag_correct", "T-financial"),
                   ("backtested", "T-process"), ("threshold_ok", "T-threshold")):
        if c.get(k) is False:
            return lab
    return "T-threshold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(R, f"p2s_eval_{a.name}_{a.split}.jsonl"))]
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(R, f"p2s_tasks_{a.split}.jsonl"))}
    by_kind = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = tasks[r["task_id"]]["kind"]
        by_kind[k][0] += 1
        by_kind[k][1] += r["pass"]
    failed = [r for r in rows if not r["pass"]]
    first, ex = collections.Counter(), collections.defaultdict(list)
    for r in failed:
        lab = classify(r)
        first[lab] += 1
        t, s = tasks[r["task_id"]], r.get("submit") or {}
        ex[lab].append({"task_id": r["task_id"], "kind": t["kind"], "query": t["query"][:90],
                        "got": {k: s.get(k) for k in ("decision", "strategy_id", "markets", "threshold", "action", "needs_financial_approval")},
                        "want": t["teacher_submit"] and {k: t["teacher_submit"].get(k) for k in ("decision", "strategy_id", "markets", "threshold", "action")}})
    crit = collections.defaultdict(list)
    for r in rows:
        for k, v in r["criteria"].items():
            if v is not None:
                crit[k].append(v)
    out = {"name": a.name, "n": len(rows), "passed": sum(r["pass"] for r in rows),
           "by_kind": {k: {"n": v[0], "passed": v[1]} for k, v in sorted(by_kind.items())},
           "criteria": {k: round(sum(v) / len(v), 3) for k, v in crit.items()},
           "decisions": dict(collections.Counter((r.get("submit") or {}).get("decision") for r in rows)),
           "first_error": dict(first), "harness_first": sum(v for k, v in first.items() if LABELS[k][0] == "harness"),
           "training_first": sum(v for k, v in first.items() if LABELS[k][0] == "training"),
           "gate_rejects": dict(sum((collections.Counter(r.get("gate_rejects") or {}) for r in rows), collections.Counter())),
           "tool_errors": dict(collections.Counter(f"{c['tool']}:{c['error']}" for r in rows for c in r["calls"] if c.get("error")).most_common(6)),
           "examples": {k: v[:3] for k, v in ex.items()}, "labels": LABELS}
    json.dump(out, open(os.path.join(R, f"p2s_triage_{a.name}_{a.split}.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(f"p2s {a.name}/{a.split}: {out['passed']}/{out['n']} passed; harness-first {out['harness_first']}, training-first {out['training_first']}")
    print("by kind:", out["by_kind"])
    print("criteria:", out["criteria"])
    print("decisions:", out["decisions"], "gate:", out["gate_rejects"], "tool errors:", out["tool_errors"])
    for k, v in first.most_common():
        print(f"  {k:12s} {v:3d}  {LABELS[k][1]}")


if __name__ == "__main__":
    main()
