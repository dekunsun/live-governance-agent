# governance-copilot — 治理策略巡检与异常归因助手（可演示原型）

对应 PRD《治理策略巡检与异常归因助手（假设版）》（见 [docs/PRD.md](docs/PRD.md)）。模拟一个成熟的直播治理平台，在上面搭一个只读的对话式 agent：
**异常归因**、**策略健康巡检**、**Case 证据包**，带评估报告。全部数据合成，不含任何真实内部信息。

```
governance-copilot/
  data/generate.py          合成数据 + 埋入的 8 个问题（见文件头注释）
  data/out/                 生成结果（sessions / hit_logs / appeals / qa_samples / metrics / strategies / change_log）
  gov_platform/tools.py     模拟平台工具（9 个读工具 + 1 个只出草稿的写工具），JSON schema，返回带 ref_id
  gov_platform/validator.py 校验层：agent 引用的策略/变更/指标/session id 必须真实存在
  agent/playbooks/          异常归因、健康巡检两份 playbook（既是文档，也是 LLM 的 system prompt）
  agent/run.py              两个引擎（rules / llm）+ 护栏（升级规则写死）+ 完整 trace
  app/app.py                Streamlit 对话界面：卡片、证据、「在平台打开」、可展开 trace
  eval/golden_set.json      30 个标准答案场景 + 巡检期望发现
  eval/run_eval.py          评估：准确率、幻觉、陷阱、口径、一致性、小市场升级、巡检误报/漏报、成本延迟、全量扫描误报率
  reports/eval_report.md    评估结果表 + 失败分析
  docs/PRD.md               产品需求文档（假设版）
  tests/test_llm_mock.py    用假 client 测 Claude tool-use 回路（校验层拦截幻觉 id 后要求重提）
```

> 目录用 `gov_platform/` 而不是 PRD 里写的 `platform/`，因为 `platform` 是 Python 标准库模块名，会被遮蔽。

## 运行

```bash
pip install -r requirements.txt
python data/generate.py                      # ~10s，生成 10 万条 session
python -m agent.run -q "美国昨天误杀申诉涨了，帮我看看"   # 命令行归因（rules 引擎，离线）
python -m agent.run --health                 # 策略健康巡检
python -m agent.run --case LS-0101617        # case 证据包（头部主播误杀 case，id 见 data/out/_embedded_problems.json）
python eval/run_eval.py --engine rules       # 标准答案集评估 → reports/eval_results_rules.md
python eval/run_eval.py --engine rules --sweep   # 全量扫描误报率 → eval/sweep_rules.json
streamlit run app/app.py                     # 界面
```

### 两个引擎

| 引擎 | 是什么 | 何时用 |
|---|---|---|
| `rules` | 确定性 playbook 执行器：按 playbook 固定顺序调工具，用写死的规则分类主因 | 离线演示、回归基线、给 LLM 引擎当「裁判」对照 |
| `llm` | Claude（`claude-opus-5-5`）tool-use 回路：playbook 是 system prompt，平台工具是 tools，最后必须用 strict 的 `submit_card` 提交卡片 | 真实的对话体验；需要 `ANTHROPIC_API_KEY` |

两者共用：工具层、校验层、护栏（样本不足 / 头部主播 / 数据缺失 → 强制升级）、trace 格式、评估脚本。
这也是 PRD 里「AI / 规则 / 人工」决策框架的落地：playbook 里能写清楚的步骤用确定性代码，模糊的意图理解和叙述交给 LLM，
高影响决策留给人。

```bash
export ANTHROPIC_API_KEY=...
python -m agent.run -q "美国昨天误杀申诉涨了" --engine llm
python eval/run_eval.py --engine llm --repeats 5    # 同一输入跑 5 次看一致性；会记录 token 和成本
```

## 5 分钟演示脚本

1. **（30s）背景一句话**：policy、业务治理运营、公会治理、风险策略都在问「指标为什么变了」。平台把它做成公共能力 + agent，先只读。
2. **（60s）主场景**：界面输入「美国昨天误杀申诉涨了，帮我看看」。
   看卡片：市场整体平稳 → 下钻到诈骗场景 → S-102 贡献 100% → 同期 C-006 模型升级 → `score_drift` PSI、超阈值比例 ×3.3 → 抽检精确率 0.82→0.56 → 15 个抽样 case。
   点开 trace：每一步查了什么、依据是什么；每条证据有 ref_id。因为抽样里有头部主播，护栏自动标「需人工确认」。
3. **（45s）陷阱与公会**：「ID 10月4日处罚量涨了」→ 流量 ×1.58、处罚率 ×0.99 → 结论是流量驱动，不是策略问题。
   这是归因最常见的错误，playbook 第二步就是先分解 `sessions × rate`。
   「巴西刷量昨天处罚涨了」→ 同期没有任何变更，增量集中在一个公会 → 转公会治理，不动策略。
4. **（45s）边界**：「肯尼亚昨天误杀申诉涨了」→ 日均 15 场、计数个位数 → 不下结论，强制升级，并提示同期有全局变更可对照。
   「昨天误杀申诉涨了」（没说市场）→ 先澄清，不猜。「US 昨天误杀率涨了」→ 三种口径冲突，选定一个并写明分母。
5. **（45s）巡检**：点「运行巡检」→ 6 条待处理：S-104/S-121 重叠 100%/90%、S-102 漂移、BR 赌博召回 0.76→0.33 但举报不降（绕过）、
   S-108 依赖已下线特征的僵尸策略、S-119 条件过窄；S-112/S-126 标为预期内变更而不是告警。
6. **（45s）评估页**：`reports/eval_report.md`——先看失败模式再看准确率；门槛写死；全量扫描 672 个无埋点组合的误报率；
   以及这版评估的局限（规则引擎是照着埋入问题写的，LLM 引擎的数字需要有 key 的环境跑）。
7. **收尾**：这是我的假设，想听你们现在做到了哪一步、口径归谁定、审批边界在哪。

## 设计要点（对照 PRD）

- **agent 不新建能力，只编排平台已有能力**：所有分析都通过 `gov_platform/tools.py` 的 10 个工具；agent 看不到隐藏的真实标签，只能用 QA 抽检和申诉结果估计 precision / recall。
- **结论必须引用证据**：每个工具结果带 `ref_id`；`Validator` 检查卡片里出现的每个 id；LLM 引擎引用了不存在的 id 会被打回重提一次，仍失败则标记校验失败并升级。
- **写操作只出草稿**：`draft_change` 按影响面和可逆性分级（低：一人确认；中：策略 + 运营；高：只出建议），写到 `drafts/`，不生效。
- **升级规则写死在代码里**：样本不足、头部主播、数据缺失、口径冲突——不依赖模型记得。
- **全球化**：主干统一，市场只是参数；小市场（日均 < 150 场）自动更保守；评估按市场拆开看。
- **可观测**：每次运行的 trace（工具调用、参数、结果摘要、护栏决定、token 和成本）写到 `traces/`。

## 埋入的问题（data/generate.py）

每个 session 还带 `guild_id`（51 个公会，约 55% 的 session 属于某个公会），归因可以下钻到公会维度。

| # | 问题 | 位置 | 应被谁发现 |
|---|---|---|---|
| P1 | 模型升级后分数上移，固定阈值误杀 | scam_fraud，C-006 2026-10-13，S-102 | 归因 + 巡检 |
| P2 | 对抗绕过导致召回下降 | BR gambling，2026-10-08 起，S-115 | 归因（最近下降）+ 巡检 |
| P3 | 两条策略重叠、重复处罚 | S-104 / S-121 | 巡检 |
| P4 | 僵尸策略 | S-108（依赖已下线特征 C-007）、S-119 | 巡检 |
| P5 | 陷阱：处罚量涨只是流量涨 | ID 2026-10-03/04 +45% | 归因 |
| P6 | Policy 更新 + 策略同步调整（预期内） | GB hate_harassment，C-003 → C-004，S-112 v3 | 归因（不算异常） |
| P7 | 新策略上线 | ID minor_safety，S-126，C-005 | 归因 |
| P8 | 头部主播误杀 case | US 2026-10-15，S-102 | case 证据包 + 升级 |
| P9 | 单一公会集中刷量（无任何策略/模型变更） | BR spam_bot，G-BR-002，2026-10-13 起 | 归因：下钻到公会，转公会治理而不是改策略 |
