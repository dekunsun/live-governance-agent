#!/usr/bin/env python3
"""Evaluate a policy (base Qwen, Qwen+LoRA, or the rules engine) on held-out tasks with the deterministic grader.

  python posttrain/evaluate.py --name base
  python posttrain/evaluate.py --name sft --adapter posttrain/runs/sft_lora
  python posttrain/evaluate.py --name rules --rules
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform  # noqa: E402
from posttrain.harness import grade, run_episode, TOOL_NAMES  # noqa: E402

RUNS = os.path.join(ROOT, "posttrain", "runs")
WORLDS = os.path.join(ROOT, "data", "worlds")


def rules_episode(platform, task):
    from agent.run import RulesEngine
    platform.refs.clear(); platform._ref_n = 0
    tr = RulesEngine(platform).attribution(task["query"])
    calls = [{"tool": e["tool"], "args": {k: v for k, v in e["args"].items() if v is not None},
              "ref_id": e["result"].get("ref_id") if isinstance(e["result"], dict) else None}
             for e in tr.events if e["kind"] == "tool_call" and e["tool"] in TOOL_NAMES]
    c = tr.card; pc = c.get("primary_cause") or {}
    sub = {"cause_type": pc.get("type"), "strategy_ids": pc.get("strategy_ids") or [], "change_ids": pc.get("change_ids") or [],
           "guild_ids": pc.get("guild_ids") or [], "needs_escalation": bool(c.get("needs_escalation")), "conclusion": c.get("conclusion", "")[:140]}
    return {"task_id": task["task_id"], "submit": sub, "calls": calls, "transcript": [], **grade(task, sub, calls, platform)}


def summarize(rows, tasks):
    by_id = {t["task_id"]: t for t in tasks}
    def agg(rs):
        n = len(rs)
        if not n:
            return {}
        crit = defaultdict(list)
        for r in rs:
            for k, v in r["criteria"].items():
                if v is not None:
                    crit[k].append(v)
        return {"n": n, "pass_rate": round(sum(r["pass"] for r in rs) / n, 3), "mean_score": round(sum(r["score"] for r in rs) / n, 3),
                "criteria": {k: round(sum(v) / len(v), 3) for k, v in crit.items()},
                "hallucination_rate": round(sum(bool(r["unknown_refs"]) for r in rs) / n, 3),
                "format_error_episodes": round(sum(r.get("format_errors", 0) > 0 for r in rs) / n, 3),
                "avg_tool_calls": round(sum(len(r["calls"]) for r in rs) / n, 2)}
    out = {"overall": agg(rows)}
    for key in ("phrasing", "kind"):
        groups = defaultdict(list)
        for r in rows:
            groups[by_id[r["task_id"]][key]].append(r)
        out[f"by_{key}"] = {k: agg(v) for k, v in sorted(groups.items())}
    groups = defaultdict(list)
    for r in rows:
        groups[by_id[r["task_id"]]["expected"]["accepted_cause_types"][0]].append(r)
    out["by_cause"] = {k: agg(v) for k, v in sorted(groups.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(os.path.join(RUNS, f"tasks_{a.split}.jsonl"))]
    if a.limit:
        tasks = tasks[:a.limit]
    policy = None
    if not a.rules:
        from posttrain.policy import QwenPolicy, load
        tok, model = load(a.adapter)
        policy = QwenPolicy(tok, model)
    platforms = {}
    out_path = os.path.join(RUNS, f"eval_{a.name}_{a.split}.jsonl")
    done = {}
    if os.path.exists(out_path):   # resumable
        for l in open(out_path):
            r = json.loads(l); done[r["task_id"]] = r
    rows = list(done.values())
    with open(out_path, "a") as f:
        for i, t in enumerate(tasks):
            if t["task_id"] in done:
                continue
            p = platforms.get(t["world"]) or platforms.setdefault(t["world"], Platform(data_dir=os.path.join(WORLDS, t["world"])))
            t0 = time.time()
            r = rules_episode(p, t) if a.rules else run_episode(policy, p, t)
            r["latency_s"] = round(time.time() - t0, 1)
            r["gen_tokens"] = getattr(policy, "gen_tokens", 0)
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); f.flush()
            print(f"[{a.name}] {i + 1}/{len(tasks)} {t['task_id']} {t['expected']['accepted_cause_types'][0]:<28} pass={r['pass']} "
                  f"score={r['score']:.2f} got={(r['submit'] or {}).get('cause_type')} calls={len(r['calls'])} {r['latency_s']}s", flush=True)
    summ = summarize(rows, tasks)
    json.dump(summ, open(os.path.join(RUNS, f"summary_{a.name}_{a.split}.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
