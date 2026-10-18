#!/usr/bin/env python3
"""Generate many synthetic 'worlds' for post-training: same platform, planted problems relocated per world.

Train and test worlds draw from different pools so the test set contains combinations never seen in training:
  - model drift: train in {scam_fraud, adult_content, gambling}; test in {violence, minor_safety}
  - recall drop: (market, scenario) pairs used in test never appear in train
Usage: python posttrain/worlds.py [--train 16 --test 6 --jobs 4]
"""
import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_DIR = os.path.join(ROOT, "data", "worlds")
BIG = ["US", "BR", "ID", "GB"]
RECALL_SCEN = ["adult_content", "scam_fraud", "gambling", "spam_bot"]
TEST_RECALL_PAIRS = {("GB", "adult_content"), ("ID", "gambling"), ("US", "spam_bot"), ("BR", "scam_fraud"),
                     ("GB", "scam_fraud"), ("ID", "adult_content")}
DRIFT_TRAIN = ["scam_fraud", "adult_content", "gambling"]
DRIFT_TEST = ["violence", "minor_safety"]


def day(n):  # 2026-10-n or 2026-09-n
    return f"2026-10-{n:02d}" if n >= 1 else f"2026-09-{30 + n:02d}"


def make_spec(idx, split, rng):
    for _ in range(200):
        s1 = rng.choice(DRIFT_TRAIN if split == "train" else DRIFT_TEST)
        pairs = [(m, s) for m in BIG for s in RECALL_SCEN if s != s1]
        pairs = [p for p in pairs if (p in TEST_RECALL_PAIRS) == (split == "test")]
        m2, s2 = rng.choice(pairs)
        launch_pool = [s for s in ["minor_safety", "violence", "counterfeit"] if s not in (s1, s2)]
        s7 = rng.choice(launch_pool)
        m7 = rng.choice(BIG)
        guild_pool = [s for s in ["spam_bot", "adult_content", "gambling"] if s not in (s1, s7)]
        s9 = rng.choice(guild_pool)
        m9 = rng.choice([m for m in BIG if (m, s9) != (m2, s2)])
        m5 = rng.choice([m for m in BIG if m not in (m2,)])
        m6 = rng.choice(BIG)
        if len({(m7, s7), (m9, s9), (m2, s2)}) < 3:
            continue
        t = rng.randint(-2, 8)
        return {"world_id": f"w{idx:02d}", "split": split, "seed": 20261016 + 1000 * (idx + 1),
                "drift": {"scenario": s1, "from": day(rng.randint(10, 13))},
                "recall": {"market": m2, "scenario": s2, "from": day(rng.randint(5, 9))},
                "traffic": {"market": m5, "dates": [day(t), day(t + 1)]},
                "policy": {"market": m6},
                "launch": {"market": m7, "scenario": s7, "date": day(rng.randint(9, 13))},
                "guild": {"market": m9, "guild_id": f"G-{m9}-00{rng.choice([2, 3])}", "scenario": s9},
                "head_case": {"market": rng.choice(BIG)}}
    raise RuntimeError("could not sample a consistent spec")


def build(spec):
    out = os.path.join(WORLD_DIR, spec["world_id"])
    os.makedirs(out, exist_ok=True)
    spec = dict(spec, out=out)
    with open(os.path.join(out, "spec.json"), "w") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "data", "generate.py"), "--spec", os.path.join(out, "spec.json")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{spec['world_id']} failed: {r.stderr[-800:]}")
    return spec["world_id"], r.stdout.strip().splitlines()[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(7)
    specs = [make_spec(i, "train", rng) for i in range(a.train)] + \
            [make_spec(a.train + j, "test", rng) for j in range(a.test)]
    with ThreadPoolExecutor(a.jobs) as ex:
        for wid, line in ex.map(build, specs):
            print(wid, line, flush=True)
    with open(os.path.join(WORLD_DIR, "index.json"), "w") as f:
        json.dump([{k: v for k, v in s.items()} for s in specs], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
