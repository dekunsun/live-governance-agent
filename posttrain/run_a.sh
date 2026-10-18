#!/usr/bin/env bash
# Option A pipeline (CPU): base eval -> LoRA SFT -> SFT eval (held-out) -> SFT eval on 20 seen train tasks.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
R=posttrain/runs
stage() { echo "$(date -u +%FT%TZ) $1" >> $R/PIPELINE_STATUS; }
stage "base_eval:start";  python3 posttrain/evaluate.py --name base > $R/log_eval_base.txt 2>&1;  stage "base_eval:exit=$?"
stage "train:start";      python3 posttrain/train_sft.py --epochs 2 > $R/log_train.txt 2>&1;   stage "train:exit=$?"
stage "sft_eval:start";   python3 posttrain/evaluate.py --name sft --adapter $R/sft_lora > $R/log_eval_sft.txt 2>&1; stage "sft_eval:exit=$?"
stage "sft_train20:start"; python3 posttrain/evaluate.py --name sft --adapter $R/sft_lora --split train --limit 20 > $R/log_eval_sft_train.txt 2>&1; stage "sft_train20:exit=$?"
stage "DONE"
