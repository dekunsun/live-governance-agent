#!/usr/bin/env python3
"""Evaluate an external API model (Anthropic / OpenRouter) on the frozen attribution test set with Harness v2.1.

Same tasks, harness, gates and grader as the GPU runs (posttrain/evaluate.py --engine vllm --harness v2); only the
policy differs (posttrain/policy_api.py). Results are resumable and go to posttrain/runs/api/.

  python posttrain/evaluate_api.py --provider openrouter --model qwen/qwen3.5-397b-a17b --name qwen35_zs --limit 10
  python posttrain/evaluate_api.py --provider anthropic  --model claude-sonnet-5-5 --name sonnet55_zs --limit 10
  python posttrain/evaluate_api.py --provider anthropic  --model claude-sonnet-5-5 --name sonnet55_fs --fewshot 4 --batch

Budget guard: after the first `--pilot` tasks the measured cost per task is extrapolated to the whole run and the
run stops if it would exceed `--budget` USD (resume with a higher --budget to continue). Every run appends its
estimate, pilot usage and final usage to posttrain/runs/api/<name>.log.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform  # noqa: E402
from posttrain.evaluate import summarize  # noqa: E402
from posttrain.harness_v2 import run_episode  # noqa: E402
from posttrain.policy_api import APIPolicy, BatchBroker, clone_platform  # noqa: E402

RUNS = os.path.join(ROOT, "posttrain", "runs")
OUT = os.path.join(RUNS, "api")
WORLDS = os.path.join(ROOT, "data", "worlds")


# ----------------------------------------------------------------------------- few-shot demos from the training set
def build_fewshot(n, path=os.path.join(RUNS, "sft_train_v2.jsonl"), seed=7):
    """n rules-engine demonstrations from the TRAIN demos (never the test set), one per cause type, rendered as text."""
    import random
    rows = [json.loads(l) for l in open(path)]
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(os.path.join(RUNS, "tasks_train.jsonl")))}
    rng = random.Random(seed)
    rng.shuffle(rows)
    want = ["model_drift_threshold", "no_anomaly", "traffic_driven", "recall_drop_evasion", "guild_concentration",
            "new_strategy_launch", "policy_update", "insufficient_data"][:max(n, 1)]
    by_cause = {}
    for r in rows:
        t = tasks.get(r["task_id"])
        if not t or t.get("split") == "test":
            continue
        by_cause.setdefault(t["expected"]["accepted_cause_types"][0], []).append((r, t))
    picked = []
    for cause in want:   # one demo per cause type, the shortest one so the prompt stays small
        cands = sorted(by_cause.get(cause, []), key=lambda x: len(x[0]["turns"]))
        if cands:
            picked.append(cands[0])
    parts = [f"下面是 {len(picked)} 个按上述步骤完成的示范（来自训练世界，编号和数字只对那个世界有效，不要照抄）。每个示范是助手与工具的交替：助手每轮先写一句依据，再调用一个工具。"]
    for i, (r, t) in enumerate(picked, 1):
        lines = [f"### 示范 {i}：{r['query']}"]
        for a, obs in r["turns"]:
            lines.append("助手：" + a.replace("<tool_call>\n", "<tool_call>").replace("\n</tool_call>", "</tool_call>"))
            if obs:
                lines.append("工具：" + obs)
        parts.append("\n".join(lines))
    return "\n\n".join(parts), [r["task_id"] for r, _ in picked]


# ----------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--provider", required=True, choices=["anthropic", "openrouter"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--fewshot", type=int, default=0, help="number of training demos to put in the system prompt")
    ap.add_argument("--effort", default=None, help="anthropic: output_config.effort (low/medium/high)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--reasoning", default=None, help='openrouter: JSON for the reasoning field, e.g. {"enabled": false}')
    ap.add_argument("--batch", action="store_true", help="anthropic: Message Batches (50%% price), one batch per turn across all episodes")
    ap.add_argument("--pilot", type=int, default=10, help="tasks after which the cost is extrapolated against --budget")
    ap.add_argument("--budget", type=float, default=12.0, help="USD cap for this run (extrapolated after the pilot)")
    ap.add_argument("--est-per-task", type=float, default=None, help="prior USD/task for the pre-run estimate (default from provider)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, f"{a.name}.log")
    lock = threading.Lock()

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with lock:
            with open(log_path, "a") as f:
                f.write(line + "\n")

    tasks = [json.loads(l) for l in open(os.path.join(RUNS, f"tasks_{a.split}.jsonl"))]
    if a.limit:
        tasks = tasks[:a.limit]
    fewshot, fs_ids = (build_fewshot(a.fewshot) if a.fewshot else (None, []))
    out_path = os.path.join(OUT, f"eval_{a.name}_{a.split}.jsonl")
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l)
            if r.get("criteria"):   # errored rows are retried
                done[r["task_id"]] = r
    todo = [t for t in tasks if t["task_id"] not in done]
    est = a.est_per_task if a.est_per_task is not None else (0.07 if a.provider == "anthropic" else 0.03)
    if a.batch:
        est *= 0.5
    log(f"run {a.name}: provider={a.provider} model={a.model} fewshot={a.fewshot} effort={a.effort} batch={a.batch} "
        f"tasks={len(tasks)} done={len(done)} todo={len(todo)} prior_estimate≈${est * len(todo):.2f} (${est}/task) budget=${a.budget}")
    if fewshot:
        log(f"fewshot demos from training set: {fs_ids} ({len(fewshot)} chars)")
        with open(os.path.join(OUT, f"{a.name}_fewshot.txt"), "w") as f:
            f.write(fewshot)
    if not todo:
        log("nothing to do")
    platforms, plock = {}, threading.Lock()

    def platform(wid):
        with plock:
            if wid not in platforms:
                platforms[wid] = Platform(data_dir=os.path.join(WORLDS, wid))
            return clone_platform(platforms[wid])

    broker = None
    client = None
    if a.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(timeout=300, max_retries=5)
        if a.batch:
            broker = BatchBroker(client, log=log)
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=300, max_retries=5)
    reasoning = json.loads(a.reasoning) if a.reasoning else None
    rows, spent, stop = list(done.values()), [sum(r.get("cost_usd", 0) for r in done.values())], {"halt": False}
    f = open(out_path, "a")
    counter = [0]

    def run_task(t):
        if stop["halt"]:
            return
        if broker is not None:
            broker.register()
        p = platform(t["world"])
        policy = APIPolicy(a.provider, a.model, client=client, fewshot=fewshot, effort=a.effort, max_tokens=a.max_tokens,
                           broker=broker, cache_ttl="1h" if a.batch else None, reasoning=reasoning)
        t0 = time.time()
        try:
            r = run_episode(policy, p, t)
        except Exception as e:
            r = {"task_id": t["task_id"], "error": f"{type(e).__name__}: {str(e)[:300]}", "submit": None, "calls": [], "transcript": [],
                 "criteria": {}, "pass": False, "score": 0.0, "unknown_refs": [], "format_errors": 0}
        finally:
            if broker is not None:
                broker.unregister()
        r.update({"provider": a.provider, "model": a.model, "fewshot": a.fewshot, "effort": a.effort, "batch": a.batch,
                  "latency_s": round(time.time() - t0, 1), "gen_tokens": policy.gen_tokens, "usage": policy.usage,
                  "cost_usd": round(policy.cost_usd, 5), "stop_reasons": policy.stop_reasons})
        with lock:
            rows.append(r)
            counter[0] += 1
            spent[0] += policy.cost_usd
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); f.flush()
            n_new = counter[0]
            print(f"[{a.name}] {n_new}/{len(todo)} {t['task_id']} {t['expected']['accepted_cause_types'][0]:<28} pass={r['pass']} "
                  f"score={r['score']:.2f} got={(r['submit'] or {}).get('cause_type')} calls={len(r['calls'])} halluc={bool(r['unknown_refs'])} "
                  f"${policy.cost_usd:.4f} {r['latency_s']}s {r.get('error', '')}", flush=True)
            if n_new == a.pilot and len(todo) > a.pilot:
                per = spent[0] / max(1, len(rows))
                proj = spent[0] + per * (len(todo) - n_new)
                pilot_rows = [x for x in rows if x.get("criteria")][-n_new:]
                log(f"pilot {n_new} tasks: spent ${spent[0]:.3f}, ${per:.4f}/task, projected total ${proj:.2f}; "
                    f"pass {sum(bool(x['pass']) for x in pilot_rows)}/{len(pilot_rows)}, hallucination {sum(bool(x['unknown_refs']) for x in pilot_rows)}, "
                    f"format_errors {sum(x.get('format_errors', 0) for x in pilot_rows)}, avg calls {sum(len(x['calls']) for x in pilot_rows) / max(1, len(pilot_rows)):.1f}, "
                    f"usage {json.dumps(_sum_usage(pilot_rows))}")
                if proj > a.budget:
                    stop["halt"] = True
                    log(f"HALT: projected ${proj:.2f} > budget ${a.budget}; rerun with a higher --budget to continue (results so far are kept)")
            if a.pilot and n_new > a.pilot and n_new % 25 == 0:
                log(f"progress {n_new}/{len(todo)}: spent ${spent[0]:.3f}")

    # worlds are independent; episodes inside a world share read-only data. In batch mode every episode runs at once.
    workers = len(todo) if a.batch else a.concurrency
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(run_task, todo))
    if broker is not None:
        broker.close()
    f.close()
    good = [r for r in rows if r.get("criteria")]
    summ = summarize(good, tasks)
    summ["provider"], summ["model"], summ["fewshot"], summ["effort"], summ["batch"] = a.provider, a.model, a.fewshot, a.effort, a.batch
    summ["cost_usd"] = round(sum(r.get("cost_usd", 0) for r in good), 3)
    summ["usage"] = _sum_usage(good)
    summ["errors"] = sum(1 for r in rows if r.get("error"))
    summ["harness"] = "v2.1"
    json.dump(summ, open(os.path.join(OUT, f"summary_{a.name}_{a.split}.json"), "w"), ensure_ascii=False, indent=1)
    log(f"done {a.name}: {len(good)} graded, errors={summ['errors']}, pass={summ['overall'].get('pass_rate')}, "
        f"cost=${summ['cost_usd']}, usage={json.dumps(summ['usage'])}")
    print(json.dumps(summ["overall"], ensure_ascii=False))


def _sum_usage(rows):
    tot = {}
    for r in rows:
        for k, v in (r.get("usage") or {}).items():
            tot[k] = tot.get(k, 0) + v
    return tot


if __name__ == "__main__":
    main()
