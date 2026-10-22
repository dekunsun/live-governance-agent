# GPU 运行手册（Runpod）

目的：在 GPU 上跑方案 B 的前两步。B0 = 不训练的 4B/8B 基线（归因 194 题 + Policy-to-Strategy 56 题，Harness v2）；B1 = 4B 的 LoRA SFT × 3 个种子 + 评测。一个 Pod 自己拉代码、造数据、跑完把结果放在 HTTP 端口上，控制脚本下载后终止 Pod。

## 一次性准备（人做）

1. Runpod 充值（预付费，余额就是硬上限）。
2. 创建 API key（Settings → API Keys，权限含 Pods 读写）。
3. 把 key 交给云环境，二选一：
   - **网络密钥**（推荐）：环境设置 → Network secrets → Add secret，域名 `rest.runpod.io`，Header `Authorization`，值 `Bearer <key>`。代理会自动加认证头，代码和日志接触不到 key。
   - **环境变量** `RUNPOD_API_KEY`：只有新开的会话能读到。
4. 新开一个会话（如果用的是环境变量），对它说："按 posttrain/gpu/README.md 跑 B0"。

不要把 key 贴到聊天里。

4.（可选，强烈建议）让 Pod 跑完后自己删除自己，防止没人收尾时一直计费：在 Runpod 控制台 Settings → Secrets 新建一个 secret（例如名字 `lga_key`，值就是 API key），然后创建 Pod 时加 `--self-key-secret lga_key`（或在云环境里设环境变量 `RUNPOD_SELF_KEY_SECRET=lga_key`）。Runpod 会在 Pod 内部把它解析成 `RUNPOD_SELF_KEY`，key 不经过这边的会话；Pod 在 DONE 之后等 `--grace-min`（默认 120 分钟）再删除自己。结果必须在这个窗口内用 `wait` 或 `results` 拉下来，否则随 Pod 一起消失；等待进程所在的会话被暂停时也会一起停，所以长任务要同时设一个定时提醒。

## 跑法（Claude 做）

```bash
python posttrain/gpu/runpod_ctl.py check                    # 认证是否通；列出现有 Pod（空列表也算通）
python posttrain/gpu/runpod_ctl.py create --stage B0        # 默认 4B + 8B；GPU 按 A40 → A6000 → L40S → 4090 → H100 的顺序找有货的
python posttrain/gpu/runpod_ctl.py wait <POD_ID>            # 每 2 分钟看一次；DONE 后下载结果到 posttrain/runs/gpu/<POD_ID>/ 并终止 Pod
python posttrain/gpu/runpod_ctl.py create --stage B1 --models Qwen/Qwen3-4B --seeds "1 2 3" --max-hours 3
```

- `create --dry-run` 只打印请求体。
- `wait` 结束时打印本次等待的大致费用；`--keep` 保留 Pod 不终止（调试用）。
- 任何时候手动收尾：`python posttrain/gpu/runpod_ctl.py terminate <POD_ID>`；或 Runpod 控制台里删除。
- Pod 里的看门狗：超过 `--max-hours`（默认 4 小时）就停止任务并标记 TIMEOUT。没有 `--self-key-secret` 时 **Pod 不会自己终止**，所以 `wait` 必须跑到结束，或者人去控制台确认 Pod 已删除。
- 下载结果用 `results <POD_ID>`；它只在核对过 Pod 列出的全部文件都拿到后才算 complete，`wait` 也只在 complete 后才终止 Pod。如果 Pod 的结果端口对 Python 返回 403（代理拦截），用 `curl -o bundle.tgz https://<POD_ID>-8000.proxy.runpod.net/bundle.tgz` 手动拿。

## 费用止损

- B0：A40 约 $0.4/h，整个 B0 约 1 小时，不到 $1；若只有 H100 有货约 $3。
- B1：4B × 3 种子训练各几分钟 + 4 次评测，约 1–1.5 小时。
- 跑完每个阶段先看结果再开下一个 Pod；花到 $35 时停下来和人确认。

## 结果在哪

`posttrain/runs/gpu/<POD_ID>/results/`：`summary_*_test.json`（总分、按问法/主因）、`eval_*_test.jsonl` 和 `p2s_eval_*_test.jsonl`（逐题轨迹）、`triage_*`（失败归类）、`train_log_*.jsonl`（训练曲线）、`LOG.txt`、`commit.txt`（跑的是哪个 commit）。

对比口径：所有结果都在 Harness v2 上；CPU 上 0.6B 的基线（`summary_base_v2_test.json`）是旧的 73 题，GPU 上是扩大后的 194 题，比较时取 task_id 交集。

## B2（GRPO）

```bash
python posttrain/gpu/runpod_ctl.py create --stage B2 --models Qwen/Qwen3-4B --max-hours 4 --self-key-secret lga_key \
  --grpo "GRPO_ITERS=12 GRPO_SECONDS=7200 GRPO_TASKS=48 GRPO_GROUP=8"
```

Pod 里的顺序：训练 SFT 种子 1（17 分钟）→ 起 vLLM（显存 45%，允许运行时换 LoRA）→ 在 384 token 上限下重测 sft_s1 → `posttrain/grpo.py`（每轮 48 题 × 8 次采样，温度 1，奖励 = 评分 + 通过加 1，组内归一化优势，只对有分歧的组做一步策略梯度，然后把新 LoRA 热换进 vLLM）→ 评测 `grpo_final`。结果里多出 `grpo_log_*.jsonl`（每轮通过率、奖励、有信号的组数、loss）和 `grpo_final_*.tgz`（最终 LoRA）。

## 数据与采样开关（`create --env "..."`）

- `EXTRA_TRAIN=16`：追加 16 个训练世界（w32–w47），测试世界不变（脚本会校验测试题的 md5）。
- `P2S_EXTRA=1`（默认）：有 `EXTRA_TRAIN` 时，P2S 示范也从追加的训练世界生成（77 → 153 条）；`P2S_EXTRA=0` 保持 16 个世界。P2S 测试集同样校验 md5。
- `AUGMENT_RARE=1`：对召回下降类和误判率上升类做过采样（多日期、多措辞），示范从 374 条变为 831 条。
- `SFT_EPOCHS=1`：SFT 的 epoch 数。`MIX_P2S=1`：SFT 混合数据里是否带 P2S 示范（0 = 只用归因示范，用来看 P2S 示范对归因的干扰）。
- `PASSK=4 PASSK_TEMP=0.7`：对第一个 SFT 种子额外做温度采样评测，报告 pass@1 / pass@4 / 一致性。
- `SEEDS_qwen3_8b=1`：按模型覆盖种子列表。
- `GRPO_V=2` + `GRPO_TEMP/LR/EPOCHS/KL/PROBE`：用 `posttrain/grpo2.py`（按通过率挑题、稀有类过采样、PPO 裁剪 + KL、探针早停）。GRPO v1（`grpo.py`）在 2 个 GPU 小时内没有可测量提升（报告第 11 节），v2 是针对其诊断的改版。

## 已知未做

- `pod_job.sh` 第一次在真实 Pod 上跑时大概率要调一两处（镜像里缺的包、vLLM 参数名随版本变化）。先用 `--max-hours 1` 跑 B0 的 4B 单模型探路：`create --stage B0 --models Qwen/Qwen3-4B --max-hours 1`。
