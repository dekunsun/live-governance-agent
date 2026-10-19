#!/usr/bin/env python3
"""Turns partner feedback from the demo page into draft evaluation tasks.

Input: a JSON export of the demo page's feedback records (list of objects; the page stores answers to the four
questions as i1..i4 plus `who` and `ts`). The third question ("哪里和你们的实际情况不一样") is the one that
produces new test cases: every non-empty answer becomes one draft task in the external-validation format
(docs/external-validation-template.md) with the fields a human still has to fill marked "TODO".

  python3 demo/feedback_to_tasks.py feedback.json -o eval/external/drafts.jsonl
"""
import argparse
import json
import re


def drafts(records):
    n = 0
    for r in records:
        text = (r.get("i3") or "").strip()
        if not text:
            continue
        n += 1
        yield {"id": f"draft-{n:03d}", "source": {"who": r.get("who") or "anonymous", "ts": r.get("ts")},
               "observation": text,
               "query": "TODO: 把这个差异写成一个用户会问的问题",
               "context": "TODO: 助手在平台上应该查到的事实（只写平台有的指标 / 变更 / 样本）",
               "expected_cause": "TODO: 10 类主因之一", "accepted_alternatives": [], "must_mention": [], "must_not_conclude": [],
               "needs_escalation": None, "difficulty": "TODO", "notes": "来自合作方反馈；先判断是规则漏了还是题目超出当前范围"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="JSON file: list of feedback records from the demo page")
    ap.add_argument("-o", "--out", default="eval/external/drafts.jsonl")
    a = ap.parse_args()
    data = json.load(open(a.export))
    records = data if isinstance(data, list) else (data.get("records") or data.get("feedback") or [])
    rows = list(drafts(records))
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(rows)} draft tasks -> {a.out} (fill the TODO fields, then review with a second person)")


if __name__ == "__main__":
    main()
