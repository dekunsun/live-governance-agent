# 评估结果（engine=rules）

| 评估项 | 结果 | MVP 门槛（示例） | 达标 |
|---|---|---|---|
| 归因准确率 | 100% (37/37) | ≥ 80% | ✅ |
| 幻觉率（引用不存在的 id） | 0% | 0 | ✅ |
| 陷阱场景通过率（流量涨≠策略差） | 100% | 100% | ✅ |
| 口径歧义处理 | 100% | 100% | ✅ |
| 小市场正确升级 | 100% | 100% | ✅ |
| 升级规则符合预期（全部 case） | 100% | ≥ 90% | ✅ |
| 一致性（同一输入跑 5 次主因一致） | 100% | ≥ 90% | ✅ |
| 巡检召回 / 精确率 | 100% / 100%（告警 7 条） | 重大问题不漏 / 误报可接受 | ✅ |
| 单次归因延迟（均值 / 最大） | 0.076s / 0.35s | < 3 分钟 | ✅ |
| 单次成本（均值） | $0.0 | — | — |
| 平均工具调用次数 | 5.5 | — | — |

## 按类别

| 类别 | 准确率 |
|---|---|
| model_drift | 100% |
| traffic_trap | 100% |
| recall_drop | 100% |
| policy_update | 100% |
| new_strategy | 100% |
| no_anomaly | 100% |
| clarification | 100% |
| metric_ambiguity | 100% |
| small_market | 100% |
| case | 100% |
| overlap | 100% |
| guild | 100% |
| guild_health | 100% |
| paused_strategy | 100% |

## 按市场

| 市场 | 准确率 |
|---|---|
| US | 100% |
| BR | 100% |
| GB | 100% |
| ID | 100% |
| KE | 100% |
| VN | 100% |

## 巡检

- 期望发现 6 条，发现 6 条；漏报 []；误报 []
- 工具调用 67 次，耗时 2.838s

## 逐条结果

| id | 类别 | 市场 | 结果 | 得到的主因 | 升级 | 结论 |
|---|---|---|---|---|---|---|
| G01 | model_drift | US | ✅ | model_drift_threshold ['S-102'] | True | US 整体 申诉推翻数 50 vs 基线 42 在正常范围，但 诈骗引流 场景异常：US 申诉推翻数上升（2→14）主要来自 S-102；2026-10-13 model_scam |
| G02 | model_drift | BR | ✅ | model_drift_threshold ['S-102'] | False | BR 申诉数上升（5→19）主要来自 S-102；2026-10-13 model_scam_fraud 升级后分数中位数上移，固定阈值导致命中 ×3.26，抽检精确率 0.77→ |
| G03 | model_drift | GB | ✅ | model_drift_threshold ['S-102'] | False | GB 处罚 session 数上升（10→23）主要来自 S-102；2026-10-13 model_scam_fraud 升级后分数中位数上移，固定阈值导致命中 ×3.32，抽 |
| G04 | model_drift | ID | ✅ | model_drift_threshold ['S-102'] | False | ID 申诉推翻数上升（3→12）主要来自 S-102；2026-10-13 model_scam_fraud 升级后分数中位数上移，固定阈值导致命中 ×3.26，抽检精确率 0.7 |
| G05 | model_drift | US | ✅ | model_drift_threshold ['S-102'] | True | US 处罚 session 数上升（15→47）主要来自 S-102；2026-10-13 model_scam_fraud 升级后分数中位数上移，固定阈值导致命中 ×3.25，抽 |
| G06 | traffic_trap | ID | ✅ | traffic_driven  | False | ID 处罚 session 数 上涨主要由流量变化驱动（974→1543 sessions），处罚 session 数率基本不变，不是策略或模型问题 |
| G07 | traffic_trap | ID | ✅ | traffic_driven  | False | ID 策略命中数 上涨主要由流量变化驱动（974→1728 sessions），策略命中数率基本不变，不是策略或模型问题 |
| G08 | traffic_trap | ID | ✅ | no_anomaly  | False | ID spam_bot 处罚 session 数 2026-10-04=33，前 7 天中位数 24.0，属正常波动，未发现异常 |
| G09 | recall_drop | BR | ✅ | recall_drop_evasion ['S-115'] | False | BR 赌博 策略命中数下降不是违规变少：举报持平、被举报内容的模型分下移，recall_est 0.751→0.328，判断为绕过导致召回下降（主要影响 S-115） |
| G10 | recall_drop | BR | ✅ | recall_drop_evasion ['S-115'] | False | BR 赌博 策略命中数下降不是违规变少：举报持平、被举报内容的模型分下移，recall_est 0.751→0.328，判断为绕过导致召回下降（主要影响 S-115） |
| G11 | recall_drop | BR | ✅ | recall_drop_evasion ['S-115'] | False | BR 赌博 处罚 session 数下降不是违规变少：举报持平、被举报内容的模型分下移，recall_est 0.751→0.328，判断为绕过导致召回下降（主要影响 S-115） |
| G12 | policy_update | GB | ✅ | policy_update ['S-112'] | False | GB 转人审数上升是 policy P-7 更新后 S-112 阈值调整的预期结果（C-003 → C-004），不是异常 |
| G13 | policy_update | GB | ✅ | policy_update ['S-112'] | False | GB 策略命中数上升是 policy P-7 更新后 S-112 阈值调整的预期结果（C-003 → C-004），不是异常 |
| G14 | new_strategy | ID | ✅ | new_strategy_launch ['S-126'] | False | ID 处罚 session 数上升主要来自新上线策略 S-126（2026-10-12，C-005），属预期内新增量；建议按上线计划观察精确率 |
| G15 | new_strategy | ID | ✅ | new_strategy_launch ['S-126'] | False | ID 策略命中数上升主要来自新上线策略 S-126（2026-10-12，C-005），属预期内新增量；建议按上线计划观察精确率 |
| G16 | no_anomaly | US | ✅ | no_anomaly  | False | US adult_content 申诉推翻数 2026-10-15=7，前 7 天中位数 3.0，属正常波动，未发现异常 |
| G17 | no_anomaly | GB | ✅ | no_anomaly  | False | GB violence 处罚 session 数 2026-10-15=5，前 7 天中位数 4.0，属正常波动，未发现异常 |
| G18 | no_anomaly | US | ✅ | no_anomaly  | False | US scam_fraud 申诉数 2026-10-11=3，前 7 天中位数 4.0，属正常波动，未发现异常 |
| G19 | no_anomaly | US | ✅ | no_anomaly  | False | US violence 处罚 session 数 2026-10-15=10，前 7 天中位数 12.0，属正常波动，未发现异常 |
| G20 | clarification | - | ✅ | needs_clarification  | False | 请补充：市场 |
| G21 | clarification | US | ✅ | needs_clarification  | False | 请补充：日期/时间窗、指标 |
| G22 | clarification | BR | ✅ | needs_clarification  | False | 请补充：日期/时间窗 |
| G23 | metric_ambiguity | US | ✅ | model_drift_threshold ['S-102'] | True | US 整体 申诉推翻数 50 vs 基线 42 在正常范围，但 诈骗引流 场景异常：按 overturn_rate 口径（0.778 vs 基线 0.750）未见显著变化；按分子  |
| G24 | metric_ambiguity | BR | ✅ | model_drift_threshold ['S-102'] | False | 按 overturn_rate 口径（0.857 vs 基线 0.625）未见显著变化；按分子 overturns 绝对数看：BR 申诉推翻数上升（3→12）主要来自 S-102； |
| G25 | small_market | KE | ✅ | insufficient_data  | True | KE 样本不足，不能下归因结论；建议看 7 天滚动窗口，并与大市场同场景趋势对照 |
| G26 | small_market | VN | ✅ | insufficient_data  | True | VN 样本不足，不能下归因结论；建议看 7 天滚动窗口，并与大市场同场景趋势对照 |
| G27 | small_market | KE | ✅ | insufficient_data  | True | KE 样本不足，不能下归因结论；建议看 7 天滚动窗口，并与大市场同场景趋势对照 |
| G28 | case | US | ✅ | case_reconstruction ['S-102'] | True | LS-0101617（US，head 主播）2026-10-15 命中 S-102(v4 分 0.852 ≥ 0.85)，最终 suspend_live，申诉overturned |
| G29 | case | - | ✅ | insufficient_data  | True | 数据缺失，无法归因：session LS-9999999 不存在。 |
| G30 | overlap | - | ✅ | None  | False | 巡检 1 条线上策略，1 条待处理，0 条预期内变更 |
| G31 | guild | BR | ✅ | guild_concentration ['S-117'] | True | BR 垃圾刷量 处罚 session 数上升（34→59）集中在公会 G-BR-002（巴西公会-002，贡献 64%），命中策略 S-117 的分数和特征正常，不是误杀；建议公会 |
| G32 | guild | BR | ✅ | guild_concentration ['S-104'] | True | BR 垃圾刷量 策略命中数上升（48→115）集中在公会 G-BR-002（巴西公会-002，贡献 72%），命中策略 S-104 的分数和特征正常，不是误杀；建议公会治理跟进该公 |
| G33 | guild | BR | ✅ | no_anomaly  | False | BR spam_bot 申诉数 2026-10-15=8，前 7 天中位数 4.0，属正常波动，未发现异常 |
| G34 | guild_health | BR | ✅ | guild_health_real_violations ['S-117'] | True | G-BR-002 健康分下降（98.51→75.05，警告约谈）来自旗下主播的真实违规：S-117 严重处置 46 次，占扣分 66%；下一档 暂停结算（70 分） |
| G35 | guild_health | US | ✅ | guild_health_unfair_deduction ['S-102'] | True | G-US-001 健康分下降（100.0→95.06）主要由 S-102 的严重处置驱动；该策略在 2026-10-13 模型升级后命中 ×3.26，扣分可能不合理，建议暂缓按健康 |
| G36 | guild_health | GB | ✅ | no_anomaly  | False | G-GB-001 近 14 天健康分 96.54→96.24（正常），扣分 4.95、恢复 4.0，属正常波动 |
| G37 | paused_strategy | - | ✅ | None  | False | S-127 已暂停，不命中属预期；巡检 0 条线上策略，0 条待处理，0 条预期内变更 |

## 未通过项（原始）

- 无
