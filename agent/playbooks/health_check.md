# Playbook：线上策略健康巡检（只读）

**适用**：定时或按需扫描全部线上策略，输出待处理清单。
**输入**：巡检日期（默认昨天），可选场景/市场过滤。
**边界**：只读；输出建议和证据，不改策略。

## 检查项

| 检查项 | 方法 | 判定（示例门槛） |
|---|---|---|
| 效果下降 effect_decline | `query_metrics(hits, group_by=strategy_id)` 近 7 天 vs 前 14 天日均 | 下降 ≥ 50% 且前期日均 ≥ 5 |
| 长期不命中 zombie | 近 28 天（或上线以来 ≥ 14 天）命中为 0 或日均 < 0.1 | 结合 `get_change_log(feature_pipeline)` 看是否依赖已下线特征 |
| 策略重叠 overlap | `get_hit_logs(strategy_id, overlap_analysis=true)` | 两个方向的重叠比例都 ≥ 0.8 且动作相同 → 重复处罚 |
| 模型分漂移导致阈值失效 drift_threshold | 同期有 `model_update` → `score_drift` | PSI > 0.1 且超阈值比例变化 ≥ 1.5 倍；再看 `precision_est`/`overturn_rate` 是否恶化 |
| 精确率下降 precision_decline | `precision_est` 近 7 天 vs 前 14 天 | 下降 ≥ 0.15 且样本 ≥ 30 |
| 召回下降 recall_decline | `recall_est` 按市场近 7 天 vs 前 14 天 | 下降 ≥ 0.2 且样本充足；同时看 `reports` 是否不降 |

## 输出

每条 finding：类型、涉及策略、市场、严重度（high/medium/low）、证据 ref_id、建议动作、是否需要人工确认。
按严重度排序；明显是“预期内”的变化（如 policy 对齐）标为 info，不算告警。

## 升级条件

- 小市场样本不足 → 只提示，不判定。
- 漂移影响 suspend_live / ban 类高严重度动作 → severity=high，建议策略 owner 当天处理。
- 重叠涉及不同 owner 的策略 → 需要两位 owner 共同确认谁下线。
