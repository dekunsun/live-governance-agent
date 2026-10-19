# 外部验证集：标注模板与说明

> 状态：模板就绪，尚无标注。这是上线前的门槛，由使用方的策略 / 治理运营同学完成；原型阶段没有标注者，所以项目里发布的全部数字只对合成测试集成立。

目的：把"和规则引擎一致"的分数变成"真的对"的分数。请 2–3 位策略 / 治理运营同学，按下面的格式各标 15–25 道贴近真实的题（合计 30–50 道）。全部用脱敏或虚构的编号与数字；不要把任何真实的策略名、公会名、主播信息写进来。

## 一道题怎么写

每道题一个 JSON 对象，一行一个，存成 `eval/external/<你的名字>.jsonl`（不要提交到公开仓库；仓库里只放模板）。

```json
{"id": "ext-001",
 "query": "印尼赌博场景这周处罚量涨了三成，帮我看看原因",
 "context": "2026-10-15 问的；同期印尼大盘流量涨了约 5%；10-12 有一次赌博模型从 v2 升到 v3；申诉推翻率没变",
 "expected_cause": "model_drift_threshold",
 "accepted_alternatives": [],
 "must_mention": ["模型升级", "固定阈值"],
 "must_not_conclude": ["traffic_driven", "no_anomaly"],
 "needs_escalation": false,
 "difficulty": "medium",
 "notes": "典型的'先排除流量再看漂移'；流量只涨 5% 解释不了 30%"}
```

字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| id | 是 | ext-三位数字 |
| query | 是 | 用户会怎么问，口语化，可以不带日期和市场 |
| context | 是 | 助手在平台上应该能查到的事实，用一两句话写清；这是标注的依据 |
| expected_cause | 是 | 10 类主因之一：model_drift_threshold / recall_drop_evasion / traffic_driven / policy_update / new_strategy_launch / guild_concentration / guild_health_real_violations / guild_health_unfair_deduction / no_anomaly / insufficient_data / needs_clarification |
| accepted_alternatives | 否 | 也算对的主因（例如 policy_update 与 strategy_change 二选一） |
| must_mention | 否 | 结论里必须出现的关键词（不是编号） |
| must_not_conclude | 否 | 常见的错误主因，答成这些算陷阱题失败 |
| needs_escalation | 是 | 按现行升级规则：资金侧、头部主播、处罚量周环比 ×2、小市场样本不足 |
| difficulty | 是 | easy / medium / hard，按你自己的判断 |
| notes | 否 | 为什么这么判，给复核的人看 |

## 请覆盖这些情况（每人至少各 1 道）

1. 处罚涨了但其实是流量涨了（陷阱）
2. 模型升级后分数漂移、固定阈值失效
3. 召回下降：违规者换了手法，命中少了但不是变干净了
4. policy 更新或策略变更导致的变化
5. 新策略上线后的首周
6. 公会集中违规 / 公会真实违规 / 公会被不合理扣分（三选一以上）
7. 小市场样本不足，正确答案是"数据不足"
8. 问题缺市场或日期，正确答案是"需要澄清"
9. 一道"其实没有异常"的题
10. 一道你觉得规则引擎会答错的题（我们最想要这种）

## 标注的两条原则

- **只写助手能查到的事实。** context 里的内容要对应平台上有的指标、变更日志、抽检样本；写"我听说"类的信息助手永远查不到，题就不公平。
- **两人独立标，不一致的题留着。** 不一致本身就是信号：要么题目有歧义，要么升级规则需要改。复核时记录分歧原因。

## 交付后我们怎么用

1. 把 context 转成一个小型合成世界的配置（市场、场景、日期、埋入的问题），生成数据后让规则引擎和模型各答一遍。
2. 规则引擎答错的题优先看：是规则漏了，还是题目超出当前能力范围。
3. 模型的结果单独报一行"外部验证集"，不和合成测试集混在一起。
