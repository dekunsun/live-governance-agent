# Playbook：指标异常归因（只读）

**适用**：用户问“某市场/某场景的某个指标昨天涨/跌了，为什么”。
**输入**：指标、市场、时间（缺任一项先澄清，不猜）。场景可选。
**边界**：只读。不改策略、不下结论性处罚建议；写操作只能生成草稿（draft_change）并走审批流。

## 步骤

1. **确认口径**。用 `query_metrics` 查指标时读取返回的 `definition` 和 `metric_version`。
   若指标名有歧义（例如“误杀率”对应 overturn_rate / overturn_per_enforcement / precision_est），
   选定一个并在结论里写明分母，同时列出其他口径。
2. **分解变化**。把计数拆成 `sessions × rate`：先查 `sessions`（流量），再查指标本身。
   若流量变化解释了大部分变化且 rate 基本不变，结论是“流量驱动”，不是策略问题（常见陷阱）。
   基线用前 7 天的中位数，不用单日对比。
3. **下钻**。市场 → 场景 → 策略 → 公会：用 `group_by` 找贡献最大的部分（按绝对增量）。
   同期没有任何变更、但增量 ≥ 50% 集中在单一公会且其他公会平稳 → 更像该公会的集中违规（公会级问题），
   不是策略误杀；用 `get_guild_info` 和 `sample_cases(guild_id=...)` 取证，转公会治理，不建议改策略。
4. **查变更**。`get_change_log` 查同期（前 7 天到当天）是否有策略上线/更新、模型升级、policy 更新、特征下线。
   - 同期有模型升级 → 用 `score_drift` 看分布是否上移、超过固定阈值的比例是否变化。
   - 策略版本变化 → 对比 `get_strategy_config` 里的版本历史。
   - policy 更新 + 策略同步调整 → 这是“预期内变化”，根因是 policy。
   - 命中下降、但举报 (`reports`) 不降、被举报人群分数下移 → 怀疑对抗绕过导致召回下降。
5. **抽样**。`sample_cases` 拉 10–20 个典型 case（申诉推翻的优先）作为证据；注意 `contains_head_host`。
6. **输出归因卡片**：结论 + 置信度 + 证据（必须带 ref_id 或实体 id）+ 建议下一步 + 是否升级。

## 机构健康分为什么掉了（公会侧提问）

用 `get_guild_health(guild_id)` 看逐日分数和扣分来源（按策略）。扣分主要来自的策略若同期有模型升级且 `score_drift` 显示命中倍增 →
`guild_health_unfair_deduction`：扣分可能不反映真实违规，建议暂缓档位处置、阈值重校后回补分数。否则 → `guild_health_real_violations`：
旗下主播真实违规，转公会治理按档位处置。两种都需要升级（公会级处置）。

## 资金侧处置

动作 delay_settlement / freeze_earnings / disable_gifts 影响主播收入，误杀代价高于下播。结论涉及这类策略时：
查 `frozen_amount`，在下一步里写明"误杀场次需复核后解冻"，并标记需运营与结算确认。

## 停止与升级条件（写死，代码也会强制执行）

- 样本不足：当日指标计数与基线合计 < 20，或该市场日均 sessions < 150（小市场）→ `needs_escalation=true`，置信度不高于 low。
- 口径冲突无法解决，或工具返回 `no_data` / `insufficient_sample` → 停下说明缺什么，给出可验证的部分。
- 抽样或 case 涉及头部主播 (host_tier=head) → 升级到策略 owner 人工确认。
- 没有找到可解释的变更且无法归因 → `unexplained`，低置信度，升级。

## 结论类型（primary_cause.type）

`traffic_driven` 流量驱动 · `model_drift_threshold` 模型升级后分数漂移、固定阈值失效 ·
`strategy_change` 策略版本变更 · `new_strategy_launch` 新策略上线 · `policy_update` policy 更新引起的预期变化 ·
`recall_drop_evasion` 对抗绕过导致召回下降 · `guild_concentration` 增量集中在单一公会 · `guild_health_real_violations` 机构扣分来自真实违规 ·
`guild_health_unfair_deduction` 机构扣分可能源于策略误杀 · `no_anomaly` 波动在正常范围 · `insufficient_data` 样本不足 ·
`unexplained` 无法解释 · `needs_clarification` 需要先澄清输入

## 禁止

- 引用不存在的策略编号、变更编号、指标名或 session id（校验层会拦截）。
- 工具失败时编造数字；重试一次，仍失败就报告缺什么。
- 把“处罚量上涨”直接等同于“策略变差”，必须先看流量和 rate。
