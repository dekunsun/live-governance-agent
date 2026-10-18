#!/usr/bin/env bash
# Runs INSIDE a Runpod Pod (image: vllm/vllm-openai). Fire-and-forget: clones the repo, builds the synthetic worlds,
# runs the requested stage, publishes results on an HTTP port, then idles until the controller terminates the Pod.
#
# Env (set by posttrain/gpu/runpod_ctl.py create):
#   STAGE        B0 (untrained baselines) | B1 (LoRA SFT x seeds + eval) | B2 (SFT seed 1 + GRPO + eval)   default B0
#   GRPO_ITERS / GRPO_SECONDS / GRPO_TASKS / GRPO_GROUP   B2 knobs          default 12 / 7200 / 48 / 8
#   MODELS       space-separated HF ids                                       default "Qwen/Qwen3-4B Qwen/Qwen3-8B"
#   SEEDS        B1 seeds                                                     default "1 2 3"
#   REPO_URL / REPO_REF                                                       default public repo / main
#   RESULT_PORT  http.server port for results                                 default 8000
#   MAX_SECONDS  hard stop: kill the stage and mark TIMEOUT                   default 14400
#   GRACE_SECONDS seconds to keep serving results after DONE before exiting  default 2700
set -uo pipefail
STAGE=${STAGE:-B0}; MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B"}; SEEDS=${SEEDS:-"1 2 3"}
GRPO_ITERS=${GRPO_ITERS:-12}; GRPO_SECONDS=${GRPO_SECONDS:-7200}; GRPO_TASKS=${GRPO_TASKS:-48}; GRPO_GROUP=${GRPO_GROUP:-8}
REPO_URL=${REPO_URL:-https://github.com/dekunsun/live-governance-agent}; REPO_REF=${REPO_REF:-main}
RESULT_PORT=${RESULT_PORT:-8000}; MAX_SECONDS=${MAX_SECONDS:-14400}
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
R=/workspace/results; mkdir -p $R /workspace/hf; cd /workspace
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/LOG; }
status() { echo "$1" > $R/STATUS; log "STATUS $1"; }
pgrep -f "http.server $RESULT_PORT" >/dev/null || ( python3 -m http.server $RESULT_PORT --directory $R >/dev/null 2>&1 & )
if [ "$(cat $R/STATUS 2>/dev/null)" = DONE ]; then log "already DONE; serving results only"; sleep infinity; fi
status SETUP
( sleep $MAX_SECONDS; status TIMEOUT; pkill -f "vllm serve"; pkill -f "posttrain/(evaluate|p2s|train_sft|triage)" ) &

apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq git curl >/dev/null 2>&1
pip install -q pandas peft >/dev/null 2>&1
if [ ! -d repo ]; then git clone -q --depth 1 -b "$REPO_REF" "$REPO_URL" repo || { status FAILED_CLONE; sleep infinity; }; fi
cd repo && git rev-parse HEAD > $R/commit.txt
log "building worlds"
python3 data/generate.py >/dev/null && python3 posttrain/worlds.py --train 16 --test 6 --jobs 8 >/dev/null && python3 posttrain/worlds.py --extra-test 10 --jobs 8 >/dev/null \
  && python3 posttrain/tasks.py >/dev/null && python3 posttrain/tasks_v2.py >/dev/null && python3 posttrain/p2s.py gen >/dev/null || { status FAILED_DATA; sleep infinity; }
cp posttrain/runs/task_stats.json posttrain/runs/sft_v2_stats.json posttrain/runs/p2s_task_stats.json $R/ 2>/dev/null
wc -l posttrain/runs/tasks_test.jsonl posttrain/runs/p2s_tasks_test.jsonl | tee -a $R/LOG

serve() {  # serve <model> [extra vllm args...]
  local m=$1; shift
  nohup vllm serve "$m" --port 8001 --max-model-len 12288 --gpu-memory-utilization ${GPU_UTIL:-0.90} --enable-prefix-caching --served-model-name "$m" "$@" > $R/vllm_$(basename $m).log 2>&1 &
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:8001/v1/models >/dev/null && return 0; sleep 10; done
  log "vllm failed to start for $m"; tail -30 $R/vllm_$(basename $m).log >> $R/LOG; return 1
}
stop_serve() { pkill -f "vllm serve"; sleep 5; }
tag() { basename "$1" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '_'; }
evaluate() {  # evaluate <name> <served-model-name>
  python3 posttrain/evaluate.py --name "$1" --harness v2 --engine vllm --model "$2" --concurrency 32 > $R/log_eval_$1.txt 2>&1
  python3 posttrain/p2s.py eval --name "$1" --split test --harness v2 --engine vllm --model "$2" --concurrency 32 > $R/log_p2s_$1.txt 2>&1
  python3 posttrain/triage.py --name "$1" > $R/triage_$1.txt 2>&1; python3 posttrain/p2s_triage.py --name "$1" > $R/p2s_triage_$1.txt 2>&1
  cp posttrain/runs/summary_$1_test.json posttrain/runs/eval_$1_test.jsonl posttrain/runs/p2s_eval_$1_test.jsonl \
     posttrain/runs/triage_$1_test.json posttrain/runs/p2s_triage_$1_test.json $R/ 2>/dev/null
}

status RUNNING_$STAGE
case "$STAGE" in
  B0)
    for m in $MODELS; do
      log "B0 $m"; export TOKENIZER_MODEL=$m
      serve "$m" && evaluate "base_v2_$(tag $m)" "$m"; stop_serve
    done ;;
  B1)
    for m in $MODELS; do
      t=$(tag $m); export TOKENIZER_MODEL=$m BASE_MODEL=$m
      cat posttrain/runs/sft_train_v2.jsonl posttrain/runs/p2s_sft_train.jsonl > posttrain/runs/sft_train_v2_mix.jsonl
      lora_args=()
      for s in $SEEDS; do
        log "B1 train $m seed $s"
        python3 posttrain/train_sft.py --data posttrain/runs/sft_train_v2_mix.jsonl --out /workspace/lora/${t}_s$s --epochs 1 --seed $s --grad_ckpt 1 > $R/log_train_${t}_s$s.txt 2>&1
        cp /workspace/lora/${t}_s$s/train_log.jsonl $R/train_log_${t}_s$s.jsonl 2>/dev/null
        lora_args+=("sft_s$s=/workspace/lora/${t}_s$s")
      done
      serve "$m" --enable-lora --max-lora-rank 16 --lora-modules "${lora_args[@]}" || continue
      evaluate "base_v2_$t" "$m"
      for s in $SEEDS; do evaluate "sft_v2_${t}_s$s" "sft_s$s"; done
      stop_serve
    done ;;
  B2)
    m=${MODELS%% *}; t=$(tag $m); export TOKENIZER_MODEL=$m BASE_MODEL=$m
    cat posttrain/runs/sft_train_v2.jsonl posttrain/runs/p2s_sft_train.jsonl > posttrain/runs/sft_train_v2_mix.jsonl
    if [ ! -f /workspace/lora/${t}_s1/adapter_config.json ]; then
      log "B2 train $m seed 1"
      python3 posttrain/train_sft.py --data posttrain/runs/sft_train_v2_mix.jsonl --out /workspace/lora/${t}_s1 --epochs 1 --seed 1 --grad_ckpt 1 > $R/log_train_${t}_s1.txt 2>&1
      cp /workspace/lora/${t}_s1/train_log.jsonl $R/train_log_${t}_s1.jsonl 2>/dev/null
    fi
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True GPU_UTIL=0.45
    serve "$m" --enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules "sft_s1=/workspace/lora/${t}_s1" || { status FAILED_VLLM; sleep infinity; }
    log "B2 eval sft_s1 (max_new_tokens 384)"; evaluate "sft_v2_${t}_s1" sft_s1
    log "B2 grpo iters=$GRPO_ITERS budget=${GRPO_SECONDS}s tasks=$GRPO_TASKS group=$GRPO_GROUP"
    python3 posttrain/grpo.py --base $m --init /workspace/lora/${t}_s1 --init-name sft_s1 --out /workspace/grpo \
      --iters $GRPO_ITERS --max-seconds $GRPO_SECONDS --tasks-per-iter $GRPO_TASKS --group $GRPO_GROUP > $R/log_grpo_${t}.txt 2>&1
    cp /workspace/grpo/grpo_log.jsonl $R/grpo_log_${t}.jsonl 2>/dev/null; tail -3 $R/log_grpo_${t}.txt >> $R/LOG
    if curl -sf http://127.0.0.1:8001/v1/models | grep -q grpo_final; then
      log "B2 eval grpo_final"; evaluate "grpo_v2_${t}" grpo_final
    else
      log "grpo_final not served; skipping eval"
    fi
    tar czf $R/grpo_final_${t}.tgz -C /workspace/grpo final 2>/dev/null
    stop_serve ;;
  *) log "unknown STAGE $STAGE" ;;
esac
cp $R/LOG $R/LOG.txt; ls -1 $R > $R/MANIFEST.txt
tar czf /workspace/bundle.tgz -C /workspace results && mv /workspace/bundle.tgz $R/bundle.tgz; ls -la $R/bundle.tgz >> $R/LOG
status DONE
# Self-termination (optional): if the Pod was given a Runpod API key through a Runpod account secret
# (env RUNPOD_SELF_KEY = {{ RUNPOD_SECRET_<name> }}), delete itself after the grace period so an unattended Pod
# stops billing. Without a key it just idles; the controller / cost guard must terminate it.
sleep ${GRACE_SECONDS:-2700}
if [ -n "${RUNPOD_SELF_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  status SELF_TERMINATE; curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_SELF_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
fi
sleep infinity
