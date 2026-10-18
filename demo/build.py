import json, os, re, collections
runs = json.load(open("runs.json"))
meta = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/out/meta.json")))
MK = meta["market_names"]; SC = meta["scenario_names"]
MZ = {"sessions": "直播场次", "hits": "策略命中数", "manual_reviews": "转人审数", "enforcements": "处罚场次", "financial_actions": "资金侧处置",
      "frozen_amount": "冻结金额", "appeals": "申诉数", "overturns": "申诉推翻数", "reports": "举报数", "hit_rate": "命中率",
      "enforcement_rate": "处罚率", "overturn_rate": "申诉推翻率", "precision_est": "抽检精确率", "recall_est": "召回估计", "guild_health_score": "健康分"}
GZ = {"scenario": "风险类型", "strategy_id": "策略", "guild_id": "公会", "market": "市场", "date": "日期"}

def zh(s):
    for k, v in sorted(MZ.items(), key=lambda x: -len(x[0])):
        s = re.sub(rf"(?<![a-z_]){k}(?![a-z_])", v, s)
    for k, v in SC.items():
        s = re.sub(rf"(?<![a-z_]){k}(?![a-z_])", v, s)
    s = s.replace("recall_est", "召回估计").replace("PSI=7.6485，", "").replace("head 主播", "头部主播")
    m = re.search(r"按策略增量贡献（strategy_id, 增量, 份额）: (\[.*\])", s)
    if m:
        import ast
        rows = ast.literal_eval(m.group(1))
        parts = [f"{sid} {b:g}→{a:g}（{'+' if d >= 0 else ''}{d:g}，占变化 {sh:.0%}）" for sid, d, sh, a, b in rows if abs(d) > 0]
        s = s[:m.start()] + "按策略拆开：" + "；".join(parts) + s[m.end():]
    s = s.replace("model_update", "模型升级").replace("扣分来源（策略, 分值）: ", "扣分来源：").replace("按公会增量贡献: ", "按公会拆开：")
    s = re.sub(r"model_(\w+?) \['(v[\d.]+)'\]→\['(v[\d.]+)'\]", lambda m: f"{SC.get(m.group(1), m.group(1))}模型 {m.group(2)}→{m.group(3)}", s)
    s = re.sub(r"model_([a-z_]+)", lambda m: SC.get(m.group(1), m.group(1)) + "模型", s)
    s = s.replace("p50", "分数中位数").replace("，含头部主播=True", "，其中有头部主播").replace("，含头部主播=False", "，没有头部主播")
    s = s.replace("mid 公会", "中型公会").replace("model_update", "模型升级").replace("policy_update", "policy 更新").replace("strategy_update", "策略调整").replace("strategy_launch", "策略上线")
    s = s.replace("（可用 run_backtest 回放新阈值）", "（先用历史数据回放新阈值）").replace("run_backtest 回放", "历史回放").replace("run_backtest", "历史回放").replace("suspend_live", "断播").replace("manual_review", "人审").replace("thr=", "阈值 ").replace("overturned", "已推翻")
    s = re.sub(r"(\d+)\.0(?!\d)", r"\1", s)
    s = s.replace("场景", "风险类型").replace("候选风险类型", "可疑类型")
    return s

def step_label(tool, a):
    where = "".join(x for x in [MK.get(a.get("market") or "", ""), SC.get(a.get("scenario") or "", "")] if x)
    if a.get("strategy_id"): where += " " + a["strategy_id"]
    if a.get("guild_id"): where += " " + a["guild_id"]
    if tool == "query_metrics":
        s = f"查{where}的{MZ.get(a.get('metric'), a.get('metric'))}"
        if a.get("group_by"): s += "，按" + "、".join(GZ.get(g, g) for g in a["group_by"]) + "拆开"
        return s
    return {"get_change_log": f"查同期有没有变更（策略、模型、policy）", "score_drift": f"检查{SC.get(a.get('scenario'), '')}模型分有没有整体漂移",
            "sample_cases": f"抽 {a.get('n', '')} 个典型 case 看一看", "get_hit_logs": f"拉{where}的命中记录",
            "get_strategy_config": f"读取策略配置 {a.get('strategy_id') or ''}", "reconstruct_case": f"还原 case {a.get('session_id')} 的完整链路",
            "get_guild_info": f"查公会 {a.get('guild_id')} 的基本信息", "get_guild_health": f"查公会 {a.get('guild_id')} 的逐日健康分和扣分来源",
            "run_backtest": "用历史数据回放"}.get(tool, tool)

def steps_of(r, summarize=False):
    ev = {e["ref_id"]: zh(e["what"]) for e in (r["card"].get("evidence") or []) if e.get("ref_id")}
    calls = [e for e in r["events"] if e["kind"] == "tool_call"]
    if summarize:
        c = collections.Counter(e["tool"] for e in calls)
        names = {"get_strategy_config": "读取策略配置", "get_hit_logs": "拉命中记录", "query_metrics": "查指标", "score_drift": "检查模型分漂移",
                 "get_change_log": "查变更记录", "get_guild_health": "查公会健康分", "sample_cases": "抽样 case", "run_backtest": "回放"}
        return [{"label": f"{names.get(t, t)} × {n}", "detail": ""} for t, n in c.most_common()]
    out, seen = [], set()
    for e in calls:
        rid = (e.get("result_summary") or {}).get("ref_id")
        lab = step_label(e["tool"], e.get("args") or {})
        det = ev.get(rid, "") if rid not in seen else ""
        seen.add(rid)
        if any(o["label"] == lab and (not det or o["detail"] == det) for o in out):
            continue
        out.append({"label": lab, "detail": det, "ref": rid or ""})
    return out

PLAIN = {
 0: dict(id="drift", group="异常归因", title="误杀申诉上涨", plain="美国整体波动正常，但诈骗这一类的申诉推翻从每天约 2 次涨到 14 次，几乎都来自策略 S-102。原因是 10/13 诈骗模型升级后打分整体偏高，而 S-102 的分数线没跟着调，于是误杀变多。同一个模型还驱动两条冻结收益的策略，涉及资金，需要一起复核。",
         why="先看整个市场，再拆到风险类型、再拆到策略；找到 S-102 后查同期变更，发现模型升级，再验证分数确实整体上移、抽检精确率确实下降。",
         trap="只看到“诈骗处罚变多”，就以为是诈骗变多了。", ask="你们遇到模型升级时，阈值是谁来调？多久能调好？"),
 1: dict(id="traffic", group="陷阱题", title="命中暴涨是流量", plain="印尼 10/3 的直播量从约 970 场涨到约 1,730 场，命中数跟着涨，但命中率基本没变。这是流量变化，不是策略或模型出了问题。",
         why="第一步就把变化拆成“流量变了多少”和“比例变了多少”，比例没变就停在这里，不往下乱找原因。",
         trap="看到命中数暴涨，就去查策略。", ask="大促、节日、热点事件时，你们怎么提前标注流量变化？"),
 2: dict(id="recall", group="陷阱题", title="处罚变少≠违规变少", plain="巴西赌博的处罚变少了，但用户举报没有减少，被举报内容的模型分反而变低。更可能是违规者找到了绕过模型的办法，漏抓变多，主要影响 S-115。",
         why="处罚减少时，用一个和策略无关的信号（用户举报）做对照；举报没降，说明违规没少。",
         trap="以为治理效果变好了。", ask="你们发现“绕过”通常靠什么信号？举报、巡查还是人审？"),
 3: dict(id="policy", group="预期内变化", title="policy 更新后的人审上涨", plain="英国 10/10 更新了骚扰的定义，S-112 的分数线在 10/11 跟着从 0.80 降到 0.50，所以人审量上升，这是预期内的结果，不是异常。",
         why="查同期变更时，发现 policy 更新和策略调整前后衔接，变化方向也一致。",
         trap="把预期内的变化当作异常报警。", ask="policy 更新后，你们怎么确认线上策略已经对齐？"),
 4: dict(id="launch", group="预期内变化", title="新策略上线带来的增量", plain="印尼未成年处罚上升，主要来自 10/12 新上线的专项策略 S-126，是预期内的新增量。不过它上线后的抽检精确率只有 0.3（20 条样本），也就是抽查的处罚里约七成不违规，值得策略 owner 尽快复核。",
         why="按策略拆开后，增量几乎都来自一条刚上线的策略。",
         trap="把新策略带来的增量当作风险上升。", ask="新策略上线后，你们一般观察多久、看哪些指标？"),
 5: dict(id="guild", group="公会与 case", title="增量集中在一家公会", plain="巴西刷量处罚从 34 涨到 59，其中 64% 来自同一家公会 G-BR-002。命中的策略分数和特征都正常，不像误杀，更像这家公会旗下的批量刷量。建议转给公会治理跟进。",
         why="按策略拆完找不到问题后，再按公会拆，发现集中在一家。",
         trap="当成整个市场的问题去调策略。", ask="公会侧的异常，你们现在是怎么发现的？"),
 6: dict(id="guildhealth", group="公会与 case", title="公会健康分为什么掉了", plain="G-BR-002 的健康分三天从约 98 掉到 75，进入“警告约谈”档。扣分 66% 来自刷量策略 S-117，同期没有模型或策略变更，是旗下主播的真实违规，不是策略误扣。再往下掉到 70 会触发暂停结算。",
         why="先看扣分来自哪条策略，再检查这条策略是否正受模型漂移影响；如果是，就是误扣分，应该先暂缓处置。",
         trap="不分来源，直接按档位处罚公会。", ask="公会对扣分有异议时，你们怎么核实？"),
 7: dict(id="case", group="公会与 case", title="还原一个头部主播 case", plain="这位头部主播 10/15 被 S-102 断播，诈骗分 0.852，只比分数线 0.85 高一点，而且正处在模型升级后的漂移期，申诉已被推翻。很可能是误杀。因为涉及头部主播，必须由人确认。",
         why="还原命中了哪条策略、当时的分数和分数线、模型版本、处罚和申诉结果，再关联同期变更。",
         trap="只给结论，不提示需要人工确认。", ask="头部主播被处罚时，你们的复核流程是怎样的？"),
 8: dict(id="small", group="边界", title="小市场样本不足", plain="肯尼亚每天只有约 20 场直播，样本太少，不能下归因结论。建议看 7 天的滚动趋势，并和大市场同类风险对照。",
         why="先检查样本量，低于门槛就主动降级，而不是硬给一个原因。",
         trap="在几条数据上编出一个原因。", ask="小市场的问题，你们现在怎么判断？"),
 9: dict(id="clarify", group="边界", title="问题不清楚时先问", plain="问题里没有说是哪个市场。助手先问清楚，不去猜。",
         why="市场、时间、指标缺一个，就先澄清。", trap="猜一个市场直接回答。", ask="你们日常提问时，最常省略的是什么信息？"),
 10: dict(id="paused", group="策略巡检", title="暂停的策略不是僵尸", plain="S-127 的状态是“已暂停”，所以 28 天没有命中，这是预期的，不需要处理。如果要恢复，先回放历史数据，再按变更流程上线。",
         why="区分“暂停导致不命中”和“策略失效导致不命中”。", trap="把暂停的策略当成僵尸策略报警。", ask="你们会定期清理长期不用的策略吗？"),
 11: dict(id="health", group="策略巡检", title="全量策略健康巡检", plain="扫描了 28 条线上策略，列出 7 条待处理：两条策略几乎重复（S-104、S-121），模型升级影响了 S-102 和 S-122，巴西赌博召回下降（S-115、S-128），两条“僵尸策略”（S-108、S-119），以及一家公会健康分越档。另外 2 条是预期内的变化，只做标注。",
         why="对每条策略检查：重叠、长期不命中、模型分漂移、召回变化、公会越档、同期变更。", trap="把预期内的变更也报成问题，告警太多没人看。", ask="你们希望巡检多久跑一次、推给谁？"),
}
SEV = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
FT = {"strategy_overlap": "策略重叠", "drift_threshold": "模型漂移影响阈值", "recall_decline": "召回下降", "zombie_strategy": "僵尸策略",
      "guild_health_breach": "公会健康分越档", "expected_change": "预期内变化", "paused_strategy": "已暂停", "effect_decline": "效果下降"}
CONF = {"high": "高", "medium": "中", "low": "低", "n/a": "—"}
scenes = []
for i, r in enumerate(runs):
    c = r["card"]; p = PLAIN[i]
    raw = c.get("conclusion") or ""
    if p["id"] == "paused":
        raw = raw.split("；巡检")[0]
    if p["id"] == "case":
        r["card"]["evidence"] = [{"ref_id": "Q-0048", "what": "命中 S-102（v3，分数 0.852，分数线 0.85）→ 断播 → 申诉已推翻；主播为头部主播，账号 531 天，0 举报；同期 C-006 诈骗模型 v3→v4 升级、分数未重新校准"}]
    sc = dict(p, q=r["query"], steps=steps_of(r, summarize=p["id"] in ("health", "paused")), tool_calls=r["tool_calls"],
              raw=zh(raw), confidence=CONF.get(c.get("confidence"), "—"), escalate=bool(c.get("needs_escalation")),
              reasons=[zh(x).replace("head_host: ", "") for x in (c.get("escalation_reasons") or [])],
              next=[zh(x) for x in (c.get("next_steps") or [])], clar=c.get("clarification_questions") or [],
              findings=[{"sev": SEV.get(f["severity"], f["severity"]), "sevk": f["severity"], "type": FT.get(f["finding_type"], f["finding_type"]),
                         "ids": f["strategy_ids"], "text": zh(f["explanation"]), "act": zh(f["suggested_action"])} for f in (c.get("findings") or [])],
              ok=(c.get("validation") or {}).get("passed", True))
    scenes.append(sc)
json.dump(scenes, open("scenes.json", "w"), ensure_ascii=False)
for s in scenes:
    print(s["id"], len(s["steps"]), s["confidence"], s["escalate"], s["clar"][:1]); [print("   -", x["label"], "|", x["detail"][:90]) for x in s["steps"][:14]]
