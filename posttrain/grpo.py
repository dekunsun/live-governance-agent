#!/usr/bin/env python3
"""GRPO on top of an SFT LoRA (GPU, vLLM + HF/PEFT in one process).

Loop per iteration:
  1. sample a batch of training tasks (attribution + Policy-to-Strategy), roll out G episodes per task through the
     Harness v2 environments with vLLM at temperature T (the current LoRA is served under a per-iteration name);
  2. reward = grader score (0..1) + 1 if the episode passes; advantage = (r - mean) / std within the task's group,
     groups where every rollout got the same reward carry no signal and are skipped (this is the "20%–60%" filter,
     applied dynamically);
  3. one on-policy policy-gradient step on the sampled assistant tokens (loss = -advantage * mean token log-prob),
     LoRA weights only, AdamW, gradient clipping;
  4. save the adapter and hot-swap it into vLLM (/v1/load_lora_adapter; the server needs
     VLLM_ALLOW_RUNTIME_LORA_UPDATING=True), then continue.
Resumable: iterations already saved under --out are skipped. The final adapter is also loaded as "grpo_final".
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "posttrain", "runs")
WORLDS = os.path.join(RUNS, "worlds")


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()


def load_tasks(limit=0):
    att = [json.loads(l) for l in open(os.path.join(RUNS, "tasks_train.jsonl"))]
    p2s = [json.loads(l) for l in open(os.path.join(RUNS, "p2s_tasks_train.jsonl"))]
    for t in att:
        t["_kind"] = "att"
    for t in p2s:
        t["_kind"] = "p2s"
    tasks = att + p2s
    return tasks[:limit] if limit else tasks


class Rollouts:
    """Runs episodes through vLLM; one Platform/World per (world, shard) thread, like evaluate.py."""

    def __init__(self, base, tok, temperature, concurrency_shards=2):
        from posttrain.harness_v2 import run_episode as att_episode
        from posttrain import p2s
        self.base, self.tok, self.T, self.shards = base, tok, temperature, concurrency_shards
        self.att_episode, self.p2s = att_episode, p2s

    def run(self, items, model):
        """items: list of (task, g). Returns list of dicts {task_id, kind, g, reward, pass, score, text, spans, calls}."""
        from posttrain.policy_vllm import VLLMPolicy
        from gov_platform import Platform
        groups, out, lock = {}, [], threading.Lock()
        for t, g in items:
            groups.setdefault((t["world"], g % self.shards), []).append((t, g))

        def run_group(key):
            wid = key[0]
            plat, world = None, None
            for t, g in groups[key]:
                pol = VLLMPolicy(self.base, model, tok=self.tok, temperature=self.T)
                t0 = time.time()
                try:
                    if t["_kind"] == "att":
                        plat = plat or Platform(data_dir=os.path.join(WORLDS, wid))
                        r = self.att_episode(pol, plat, t)
                    else:
                        world = world or self.p2s.World(wid)
                        r = self.p2s.run_episode(pol, world, t, harness="v2")
                    rec = {"task_id": t["task_id"], "kind": t["_kind"], "g": g, "pass": bool(r["pass"]), "score": float(r["score"]),
                           "reward": float(r["score"]) + (1.0 if r["pass"] else 0.0), "calls": len(r["calls"]),
                           "format_errors": r.get("format_errors", 0), "text": pol.text, "spans": pol.spans, "sec": round(time.time() - t0, 1)}
                except Exception as e:
                    rec = {"task_id": t["task_id"], "kind": t["_kind"], "g": g, "pass": False, "score": 0.0, "reward": 0.0, "calls": 0,
                           "format_errors": 0, "text": pol.text, "spans": pol.spans, "sec": round(time.time() - t0, 1), "error": str(e)[:200]}
                with lock:
                    out.append(rec)

        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
            list(ex.map(run_group, sorted(groups)))
        return out


def advantages(recs):
    """Group-normalised advantages; returns list of (rec, adv) for records whose group has signal."""
    by = {}
    for r in recs:
        by.setdefault(r["task_id"], []).append(r)
    out, used_groups = [], 0
    for tid, rs in by.items():
        rw = [r["reward"] for r in rs]
        m = sum(rw) / len(rw)
        sd = (sum((x - m) ** 2 for x in rw) / len(rw)) ** 0.5
        if sd < 1e-6:
            continue
        used_groups += 1
        for r in rs:
            if r["spans"]:
                out.append((r, (r["reward"] - m) / (sd + 1e-6)))
    return out, used_groups


def build_batch(tok, text, spans, max_len):
    """Token ids + loss mask (1 on generated tokens) from the episode text and the generated char spans."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None, None
    mask = [0] * len(ids)
    si = 0
    for i, (a, b) in enumerate(offs):
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and spans[si][0] <= a < spans[si][1]:
            mask[i] = 1
    return ids, mask


def train_step(model, tok, opt, samples, device, max_len, accum_tokens_cap=None):
    """One optimizer step over all samples (micro-batch = 1 sequence). Returns stats."""
    import torch
    model.train()
    n_seq, n_tok, tot_loss = 0, 0, 0.0
    opt.zero_grad(set_to_none=True)
    valid = []
    for rec, adv in samples:
        ids, mask = build_batch(tok, rec["text"], rec["spans"], max_len)
        if ids is None or sum(mask) == 0:
            continue
        valid.append((ids, mask, adv))
    if not valid:
        return {"seqs": 0}
    for ids, mask, adv in valid:
        x = torch.tensor([ids], device=device)
        m = torch.tensor([mask[1:]], device=device, dtype=torch.float32)   # predict token i+1 from position i
        logits = model(input_ids=x).logits[:, :-1, :].float()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        mean_logp = (logp * m).sum() / m.sum()
        loss = -(adv * mean_logp) / len(valid)
        loss.backward()
        tot_loss += float(loss.item()); n_seq += 1; n_tok += int(m.sum().item())
        del logits, logp
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return {"seqs": n_seq, "tokens": n_tok, "loss": round(tot_loss, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--init", required=True, help="SFT LoRA adapter directory (also served by vLLM as --init-name)")
    ap.add_argument("--init-name", default="sft_s1")
    ap.add_argument("--out", default=os.path.join(RUNS, "grpo"))
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--tasks-per-iter", type=int, default=48)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=7000)
    ap.add_argument("--max-seconds", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="roll out one batch, print stats, no training")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from posttrain.policy_vllm import wait_ready
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    assert wait_ready(a.base_url), f"vLLM not reachable at {a.base_url}"
    tok = AutoTokenizer.from_pretrained(a.base)
    tasks = load_tasks(a.limit)
    roll = Rollouts(a.base_url, tok, a.temperature)
    log_path = os.path.join(a.out, "grpo_log.jsonl")
    done = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    start_iter = len(done)
    cur_name = done[-1]["adapter"] if done else a.init_name
    cur_dir = os.path.join(a.out, f"iter_{start_iter - 1}") if done else a.init
    if done:   # resume: make sure the server has the last adapter
        try:
            post(a.base_url, "/v1/load_lora_adapter", {"lora_name": cur_name, "lora_path": cur_dir})
        except Exception:
            pass
    print(f"tasks {len(tasks)}; starting at iteration {start_iter} with adapter {cur_name}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, cur_dir, is_trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    t_start = time.time()
    for it in range(start_iter, a.iters):
        if time.time() - t_start > a.max_seconds:
            print("time budget reached", flush=True); break
        batch = random.sample(tasks, min(a.tasks_per_iter, len(tasks)))
        items = [(t, g) for t in batch for g in range(a.group)]
        t0 = time.time()
        recs = roll.run(items, cur_name)
        t_roll = time.time() - t0
        samples, used = advantages(recs)
        stats = {"iter": it, "adapter": cur_name, "n_rollouts": len(recs), "pass_rate": round(sum(r["pass"] for r in recs) / len(recs), 3),
                 "mean_reward": round(sum(r["reward"] for r in recs) / len(recs), 3), "groups_with_signal": used,
                 "train_samples": len(samples), "rollout_sec": round(t_roll),
                 "pass_att": round(sum(r["pass"] for r in recs if r["kind"] == "att") / max(1, sum(r["kind"] == "att" for r in recs)), 3),
                 "pass_p2s": round(sum(r["pass"] for r in recs if r["kind"] == "p2s") / max(1, sum(r["kind"] == "p2s" for r in recs)), 3)}
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        json.dump(recs, open(os.path.join(a.out, f"rollouts_{it}.json"), "w"), ensure_ascii=False)
        if a.dry_run:
            return
        t1 = time.time()
        ts = train_step(model, tok, opt, samples, device, a.max_len)
        stats.update(ts); stats["train_sec"] = round(time.time() - t1)
        new_name, new_dir = f"grpo_{it}", os.path.join(a.out, f"iter_{it}")
        model.save_pretrained(new_dir)
        post(a.base_url, "/v1/load_lora_adapter", {"lora_name": new_name, "lora_path": new_dir})
        if cur_name != a.init_name:
            try:
                post(a.base_url, "/v1/unload_lora_adapter", {"lora_name": cur_name})
            except Exception:
                pass
        stats["adapter"] = new_name
        with open(log_path, "a") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        cur_name, cur_dir = new_name, new_dir
    final = os.path.join(a.out, "final")
    model.save_pretrained(final)
    post(a.base_url, "/v1/load_lora_adapter", {"lora_name": "grpo_final", "lora_path": final})
    print(f"final adapter {cur_name} -> {final}, served as grpo_final", flush=True)


if __name__ == "__main__":
    main()
