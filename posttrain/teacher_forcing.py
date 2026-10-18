#!/usr/bin/env python3
"""Teacher-forcing diagnostic: does the SFT model reproduce its own training trajectories token by token?

For each training sample we render the exact token sequence used in training (prefix + assistant turns +
observation segments), run one forward pass, and compare the greedy argmax at every assistant position with
the token that was trained on. If agreement is ~100% but free-running episodes on the same tasks fail, the
gap is a train/inference mismatch in the harness; if agreement is well below 100%, the model simply did not
fit the data.

  python posttrain/teacher_forcing.py --adapter posttrain/runs/sft_lora --limit 20
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from posttrain.train_sft import build_sample  # noqa: E402
from posttrain.policy import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "posttrain/runs/sft_train.jsonl"))
    ap.add_argument("--adapter", default=os.path.join(ROOT, "posttrain/runs/sft_lora"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    tok, model = load(a.adapter, threads=a.threads)
    rows = [json.loads(l) for l in open(a.data)][:a.limit]
    out, tot_ok, tot_n = [], 0, 0
    for s in rows:
        ids, lab = build_sample(tok, s)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0]
        pred = logits[:-1].argmax(-1)
        tgt = torch.tensor(lab[1:])
        sel = tgt != -100
        ok = (pred[sel] == tgt[sel])
        n, k = int(sel.sum()), int(ok.sum())
        tot_ok += k; tot_n += n
        # first divergence, with context
        first = None
        if k < n:
            pos = sel.nonzero(as_tuple=True)[0][(~ok).nonzero(as_tuple=True)[0][0]].item()
            first = {"pos": pos, "expected": tok.decode([int(tgt[pos])]), "got": tok.decode([int(pred[pos])]),
                     "context": tok.decode(ids[max(0, pos - 40):pos + 1])[-160:]}
        # per-turn: which assistant turn diverges first
        out.append({"task_id": s["task_id"], "assistant_tokens": n, "agree": k, "agree_rate": round(k / n, 4), "first_divergence": first})
        print(f"{s['task_id']:10s} {k}/{n} = {k / n:.3f}" + (f"  first mismatch @{first['pos']}: expected {first['expected']!r} got {first['got']!r}" if first else ""), flush=True)
    summary = {"n_samples": len(rows), "token_agreement": round(tot_ok / tot_n, 4),
               "samples_fully_reproduced": sum(1 for o in out if o["agree"] == o["assistant_tokens"]), "per_sample": out}
    json.dump(summary, open(os.path.join(ROOT, "posttrain/runs/teacher_forcing_sft.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
