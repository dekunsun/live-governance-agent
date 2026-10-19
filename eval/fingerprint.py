"""Content fingerprint of the frozen test sets.

The md5 in eval/checksums.txt locks the files byte-for-byte. That is too strict for the
P2S set, whose rows also carry the *teacher's* submit (rationale text) which may be
reworded without changing the task. This fingerprint hashes only what defines a task:
task_id, query/prompt and the expected answer. Both must stay stable for published
numbers to remain comparable; the md5 may legitimately change when the teacher changes.

  python3 eval/fingerprint.py            # print fingerprints
  python3 eval/fingerprint.py --check    # compare with eval/fingerprints.txt (exit 1 on drift)
"""
import hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {
    "posttrain/runs/tasks_test.jsonl": ("task_id", "query", "expected"),
    "posttrain/runs/p2s_tasks_test.jsonl": ("task_id", "query", "expected"),
}

def fingerprint(path, keys):
    h = hashlib.sha256()
    n = 0
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h.update(json.dumps([r.get(k) for k in keys], ensure_ascii=False, sort_keys=True).encode())
            n += 1
    return f"{h.hexdigest()[:32]}  {n:4d} tasks  {path}"

def main():
    lines = [fingerprint(p, k) for p, k in SETS.items()]
    if "--check" in sys.argv:
        want = open(os.path.join(ROOT, "eval/fingerprints.txt"), encoding="utf-8").read().splitlines()
        bad = [l for l in lines if l not in want]
        for l in lines:
            print(("OK      " if l in want else "CHANGED ") + l)
        sys.exit(1 if bad else 0)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
