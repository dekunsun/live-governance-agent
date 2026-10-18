#!/usr/bin/env bash
# Option A pipeline (CPU): base eval -> LoRA SFT -> SFT eval (held-out) -> SFT eval on 20 seen train tasks.
# Idempotent: a finished stage leaves a .done_<stage> marker and is skipped; eval and training resume where they stopped,
# so after a container restart just run this script again.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
R=posttrain/runs
stage() { echo "$(date -u +%FT%TZ) $1" >> $R/PIPELINE_STATUS; }
run() {   # run <name> <log> <cmd...>
  local name=$1 logf=$2; shift 2
  [ -f "$R/.done_$name" ] && return 0
  stage "$name:start"; "$@" >> "$logf" 2>&1; local rc=$?; stage "$name:exit=$rc"
  [ $rc -eq 0 ] && touch "$R/.done_$name"
  return $rc
}
run base_eval   $R/log_eval_base.txt      python3 posttrain/evaluate.py --name base || exit 1
run train       $R/log_train.txt          python3 posttrain/train_sft.py --epochs 2 || exit 1
run sft_eval    $R/log_eval_sft.txt       python3 posttrain/evaluate.py --name sft --adapter $R/sft_lora || exit 1
run sft_train20 $R/log_eval_sft_train.txt python3 posttrain/evaluate.py --name sft --adapter $R/sft_lora --split train --limit 20 || exit 1
stage "DONE"
