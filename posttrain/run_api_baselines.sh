#!/usr/bin/env bash
# External-API baselines on the frozen 194-task attribution set (Harness v2.1, same grader as the GPU runs).
# Needs ANTHROPIC_API_KEY (groups 1-2) and OPENROUTER_API_KEY (group 3) in the environment. Each group runs a
# 10-task pilot first (format / hallucination check + measured cost), then the rest; the budget guard stops a run
# whose projected cost exceeds --budget. Logs with usage: posttrain/runs/api/<name>.log
set -euo pipefail
cd "$(dirname "$0")/.."
make check >/dev/null   # test-set fingerprint must be unchanged
pilot() { python3 -u posttrain/evaluate_api.py "$@" --limit 10; }
full()  { python3 -u posttrain/evaluate_api.py "$@"; }

# 1. Claude Sonnet 5.5, zero-shot (prompt caching on; Message Batches for the full run)
pilot --provider anthropic --model claude-sonnet-5-5 --name sonnet55_zs --effort medium --budget 12
full  --provider anthropic --model claude-sonnet-5-5 --name sonnet55_zs --effort medium --budget 12 --batch
# 2. Claude Sonnet 5.5, few-shot: 4 rules-engine demos from the TRAIN demos (sft_train_v2.jsonl) in the system prompt
pilot --provider anthropic --model claude-sonnet-5-5 --name sonnet55_fs4 --fewshot 4 --effort medium --budget 12
full  --provider anthropic --model claude-sonnet-5-5 --name sonnet55_fs4 --fewshot 4 --effort medium --budget 12 --batch
# 3. Qwen3.5-397B-A17B via OpenRouter, zero-shot (greedy)
pilot --provider openrouter --model qwen/qwen3.5-397b-a17b --name qwen35_zs --budget 10
full  --provider openrouter --model qwen/qwen3.5-397b-a17b --name qwen35_zs --budget 10 --concurrency 8
