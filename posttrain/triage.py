#!/usr/bin/env python3
"""Classify failed episodes by their FIRST error step: harness-fixable vs needs-training.

  python posttrain/triage.py --name base            # reads runs/eval_base_test.jsonl
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from posttrain.harness import CAUSES  # noqa: E402

R = os.path.join(ROOT, "posttrain/runs")

LABELS = {
    "H-format": ("harness", "输出无法解析成工具调用"),
    "H-loop": ("harness", "同一个错误反复重试，直到回合用完"),
    "H-args": ("harness", "工具参数不合法（指标名、时间窗、场景名等）"),
    "H-enum": ("harness", "提交的主因不在可选列表里"),
    "H-premature": ("harness", "没查流量或只查一次就提交结论"),
    "H-budget": ("harness", "回合用完（14 次调用）仍未提交，一直在查"),
    "H-ids": ("harness", "提交了编造的策略/变更/公会编号"),
    "T-cause": ("training", "格式和步骤都没问题，但在证据面前选错主因"),
    "T-escalation": ("training", "主因对了，但该不该升级判断错"),
    "T-cite": ("training", "主因对了，但漏了证据里出现过的关键编号"),
}
FIX = {
    "H-format": "工具调用格式用约束解码或解析容错；解析失败时给出正确格式的完整示例",
    "H-loop": "同一错误连续出现时，在报错里给出合法取值的具体例子；连续 3 次相同错误就强制换工具或提交",
    "H-args": "参数改成枚举并在报错里列出合法值；时间窗自动裁剪到数据范围并说明",
    "H-enum": "submit_card 的 cause_type 强制枚举，不合法时拒收并列出可选值",
    "H-premature": "提交前检查必做步骤（先查流量 sessions）；没做就拒收并提示下一步",
    "H-budget": "剩余 2 次调用时在观察里提醒提交（v2.1 已加）；SFT 示范里学会何时停",
    "H-ids": "提交时校验编号必须出现在本轮观察里，否则拒收并列出已见过的编号",
    "T-cause": "SFT 示范轨迹 + RL（按主因是否正确给奖励）",
    "T-escalation": "SFT 示范 + RL 奖励里加升级判断",
    "T-cite": "SFT 示范；也可在 harness 里自动收集观察中出现的编号供选择（部分可由 harness 缓解）",
}


def residual_error(r, task):
    """Harness v2 already repairs arguments and gates the submission, so classify what is LEFT at the end."""
    s, crit = r.get("submit"), r["criteria"]
    n_asst = sum(1 for m in r["transcript"] if m["role"] == "assistant")
    if s is None:
        if not r.get("format_errors") and not r.get("gate_rejects"):
            return "H-budget", n_asst, f"{len(r.get('calls', []))} 次调用后回合结束，始终没有提交"
        return "H-format", n_asst, f"没有提交（门控退回 {r.get('gate_rejects')}，格式错误 {r.get('format_errors')}）"
    if s.get("cause_type") not in CAUSES:
        return "H-enum", n_asst, f"cause_type={s.get('cause_type')!r}（门控退回两次后仍不合法）"
    if crit.get("checked_traffic_first") is False:
        return "H-premature", n_asst, "门控退回后仍未查流量"
    if r.get("unknown_refs"):
        return "H-ids", n_asst, f"编造编号 {r['unknown_refs'][:3]}"
    if not crit.get("cause_correct"):
        return "T-cause", n_asst, f"提交 {s.get('cause_type')}，应为 {task['expected']['accepted_cause_types'][0]}"
    if crit.get("escalation") is False:
        return "T-escalation", n_asst, f"needs_escalation={s.get('needs_escalation')}"
    return "T-cite", n_asst, f"缺 {[k for k in ('strategy_ids', 'change_ids', 'guild_ids') if crit.get(k) is False]}"


def first_error(r, task):
    """Walk the episode in order and return (label, step_index, snippet)."""
    if r.get("harness_version") == "v2":
        return residual_error(r, task)
    tr, calls = r["transcript"], r["calls"]
    errs, first_arg = collections.Counter(), None
    k, step = 0, 0
    for i, m in enumerate(tr):
        if m["role"] != "assistant":
            continue
        step += 1
        obs = tr[i + 1]["text"] if i + 1 < len(tr) and tr[i + 1]["role"] == "tool" else None
        if obs is not None and obs.startswith("格式错误") and first_arg is None:
            return "H-format", step, m["text"][:160]
        if obs is not None and not obs.startswith("格式错误"):   # a tool call that got an observation
            c = calls[k] if k < len(calls) else {}
            k += 1
            if c.get("error"):
                errs[(c["tool"], c["error"])] += 1
                if first_arg is None:
                    first_arg = (step, f"{c['tool']} {c['error']} args={json.dumps(c.get('args'), ensure_ascii=False)[:120]}")
    if first_arg:   # the first error was a bad tool argument; a loop if the same error kept repeating
        (tool, err), n = errs.most_common(1)[0]
        if n >= 3:
            return "H-loop", first_arg[0], f"{tool} {err} 重复 {n} 次；首次 {first_arg[1]}"
        return "H-args", first_arg[0], first_arg[1]
    # reached submit (or ran out of turns without submitting)
    s = r.get("submit")
    if s is None:
        return "H-format", step, (tr[-1]["text"][:160] if tr else "")
    crit = r["criteria"]
    if s.get("cause_type") not in CAUSES:
        return "H-enum", step, f"cause_type={s.get('cause_type')!r}"
    if crit.get("checked_traffic_first") is False or len(calls) <= 1:
        return "H-premature", step, f"提交前只调用了 {len(calls)} 次工具；submit={json.dumps(s, ensure_ascii=False)[:120]}"
    if r.get("unknown_refs"):
        return "H-ids", step, f"编造编号 {r['unknown_refs'][:3]}"
    if not crit.get("cause_correct"):
        return "T-cause", step, f"提交 {s.get('cause_type')}，应为 {task['expected']['accepted_cause_types'][0]}"
    if crit.get("escalation") is False:
        return "T-escalation", step, f"needs_escalation={s.get('needs_escalation')}"
    return "T-cite", step, f"缺 {[k for k in ('strategy_ids', 'change_ids', 'guild_ids') if crit.get(k) is False]}"


def all_issues(r):
    s, c, out = r.get("submit"), r["criteria"], set()
    if r["format_errors"]:
        out.add("H-format")
    if any(x.get("error") for x in r["calls"]):
        out.add("H-args")
    if s is not None and s.get("cause_type") not in CAUSES:
        out.add("H-enum")
    if c.get("checked_traffic_first") is False:
        out.add("H-premature")
    if r.get("unknown_refs"):
        out.add("H-ids")
    if s is not None and s.get("cause_type") in CAUSES and not c.get("cause_correct"):
        out.add("T-cause")
    if c.get("escalation") is False:
        out.add("T-escalation")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(R, f"eval_{a.name}_{a.split}.jsonl"))]
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(R, f"tasks_{a.split}.jsonl"))}
    failed = [r for r in rows if not r["pass"]]
    firsts, examples = collections.Counter(), collections.defaultdict(list)
    multi = collections.Counter()
    for r in failed:
        t = tasks[r["task_id"]]
        lab, step, snip = first_error(r, t)
        firsts[lab] += 1
        examples[lab].append({"task_id": r["task_id"], "kind": t["kind"], "phrasing": t["phrasing"], "query": t["query"], "step": step, "detail": snip})
        for x in all_issues(r):
            multi[x] += 1
    out = {"name": a.name, "split": a.split, "n": len(rows), "passed": len(rows) - len(failed), "failed": len(failed),
           "first_error": dict(firsts), "any_issue": dict(multi),
           "harness_first": sum(v for k, v in firsts.items() if LABELS[k][0] == "harness"),
           "training_first": sum(v for k, v in firsts.items() if LABELS[k][0] == "training"),
           "examples": {k: v[:3] for k, v in examples.items()}, "labels": LABELS, "fix": FIX,
           "gate_rejects": dict(sum((collections.Counter(r.get("gate_rejects") or {}) for r in rows), collections.Counter())),
           "normalized_calls": sum(1 for r in rows for c in r["calls"] if c.get("normalized")),
           "blocked_repeats": sum(r.get("blocked_repeats", 0) for r in rows)}
    json.dump(out, open(os.path.join(R, f"triage_{a.name}_{a.split}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{a.name}/{a.split}: {out['passed']}/{out['n']} passed; failed {out['failed']}: harness-first {out['harness_first']}, training-first {out['training_first']}")
    for k, v in firsts.most_common():
        print(f"  {k:13s} {v:3d}  {LABELS[k][1]}")
    print("any-issue (multi-label):", dict(multi.most_common()))
    if out["gate_rejects"] or out["normalized_calls"]:
        print("v2 gate rejects:", out["gate_rejects"], "normalized calls:", out["normalized_calls"], "blocked repeats:", out["blocked_repeats"])


if __name__ == "__main__":
    main()
