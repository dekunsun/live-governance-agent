#!/usr/bin/env python3
"""LoRA SFT of Qwen on teacher trajectories, loss only on the model's own (assistant) tokens.

  python posttrain/train_sft.py --data posttrain/runs/sft_train.jsonl --out posttrain/runs/sft_lora --epochs 2
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from posttrain.harness import obs_segment, prefix_text  # noqa: E402
from posttrain.harness_v2 import prefix_text as prefix_text_v2  # noqa: E402
from posttrain.p2s import prefix_text_v2 as prefix_text_p2s  # noqa: E402


def build_sample(tok, s):
    ids, lab = [], []
    kind = s.get("prefix", "v1")   # which harness rendered this sample: v1 | v2 (attribution, with scope hint) | p2s
    pre = prefix_text_v2(tok, s["query"], s["scope"]) if kind == "v2" else prefix_text_p2s(tok, s["query"]) if kind == "p2s" else prefix_text(tok, s["query"])
    p = tok(pre).input_ids
    ids += p; lab += [-100] * len(p)
    for a, o in s["turns"]:
        t = tok(a + "<|im_end|>").input_ids
        ids += t; lab += t
        if o is not None:
            q = tok(obs_segment(o)).input_ids
            ids += q; lab += [-100] * len(q)
    return ids, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "posttrain/runs/sft_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "posttrain/runs/sft_lora"))
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps cap (timing runs)")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N samples (smoke tests)")
    ap.add_argument("--ckpt_every", type=int, default=1, help="save a resumable checkpoint every N optimizer steps")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); random.seed(a.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                             task_type="CAUSAL_LM"))
    inner, head = model.base_model.model.model, model.base_model.model.lm_head
    data = [build_sample(tok, json.loads(l)) for l in open(a.data)]
    if a.limit:
        data = data[:a.limit]
    total_micro = int(len(data) * a.epochs)
    total_steps = math.ceil(total_micro / a.accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.06 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    start = 0
    if os.path.exists(ckpt_path):   # resume after an interrupted run (container restarts kill background jobs)
        ck = torch.load(ckpt_path, weights_only=False)
        for n, v in ck["params"].items():
            trainable[n].data.copy_(v)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed from step {start}", flush=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a" if start else "w")
    print(f"samples={len(data)} micro={total_micro} opt_steps={total_steps} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M", flush=True)
    model.train()
    order = []
    while len(order) < total_steps * a.accum:
        perm = list(range(len(data))); random.shuffle(perm); order += perm
    t_start = time.time()
    for step in range(start, total_steps):
        t0, loss_sum, ntok = time.time(), 0.0, 0
        for k in range(a.accum):
            ids, lab = data[order[step * a.accum + k]]
            x = torch.tensor([ids])
            hidden = inner(input_ids=x).last_hidden_state[0]
            lab_t = torch.tensor(lab)
            sel = (lab_t[1:] != -100).nonzero(as_tuple=True)[0]
            logits = head(hidden[sel])
            loss = F.cross_entropy(logits.float(), lab_t[1:][sel])
            (loss / a.accum).backward()
            loss_sum += float(loss); ntok += len(sel)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        rec = {"step": step + 1, "loss": round(loss_sum / a.accum, 4), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               "trained_tokens": ntok, "elapsed_min": round((time.time() - t_start) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < total_steps:
            torch.save({"step": step + 1, "params": {n: p.detach().clone() for n, p in trainable.items()}, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": torch.get_rng_state()}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    model.save_pretrained(a.out)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
