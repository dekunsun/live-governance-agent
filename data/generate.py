#!/usr/bin/env python3
"""Synthetic live-streaming governance platform data with embedded problems.

Everything here is fictional. Strategy ids, thresholds, markets and volumes are
illustrative. Hidden ground-truth labels (``true_label``) are written to a
separate file that the agent tools never read; they exist only for evaluation.

Embedded problems (see PROBLEMS below):
  P1 model drift       scam_fraud model v3->v4 on 2026-10-13 shifts scores up;
                       S-102 keeps a fixed 0.85 threshold -> false positives, appeals
  P2 recall drop       BR gambling violators evade from 2026-10-08 -> S-115 misses them
  P3 overlap           S-104 and S-121 (spam_bot) hit the same sessions -> double action
  P4 zombie            S-108 depends on a retired feature (0 hits); S-119 nearly never fires
  P5 traffic trap      ID traffic +45% on 2026-10-03/04, violation rates unchanged
  P6 policy+strategy   GB hate_harassment: policy P-7 broadened (C-003), S-112 v3 lowers
                       threshold (C-004) -> manual review volume up (expected)
  P7 new strategy      S-126 launched in ID on 2026-10-12 (minor_safety) -> enforcement up
  P8 head-host case    a head-tier host in US suspended by S-102 on 2026-10-15 (FP)
  Financial actions    S-114 freeze_earnings, S-125 delay_settlement, S-129 freeze_earnings (scam, also drift-hit)
  Guild health score   daily per-guild score (100 start) deducted when severe enforcements exceed the market
                       baseline rate; P9 pushes G-BR-002 below 80; drift silently erodes US guilds (unfair deductions)
  P9 guild spike       guild G-BR-002 runs coordinated spam_bot sessions on 2026-10-13..15 ->
                       BR spam enforcement up, concentrated in one guild (no strategy/model change)
"""
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SEED = 20261016
START = date(2026, 9, 18)
DAYS = 28                      # 2026-09-18 .. 2026-10-15
TODAY = "2026-10-16"           # demo "today"; "yesterday" = 2026-10-15

MARKETS = {"US": 1100, "BR": 850, "ID": 950, "GB": 520, "VN": 55, "KE": 18}   # sessions/day
MARKET_NAMES = {"US": "美国", "BR": "巴西", "ID": "印尼", "GB": "英国", "VN": "越南", "KE": "肯尼亚"}
SCENARIOS = ["adult_content", "minor_safety", "scam_fraud", "violence",
             "spam_bot", "gambling", "hate_harassment", "counterfeit"]
SCENARIO_NAMES = {"adult_content": "色情低俗", "minor_safety": "未成年人安全", "scam_fraud": "诈骗引流",
                  "violence": "暴力血腥", "spam_bot": "垃圾刷量", "gambling": "赌博",
                  "hate_harassment": "仇恨骚扰", "counterfeit": "假冒侵权"}
BASE_VIOL = {"adult_content": 0.030, "minor_safety": 0.012, "scam_fraud": 0.025, "violence": 0.010,
             "spam_bot": 0.040, "gambling": 0.015, "hate_harassment": 0.012, "counterfeit": 0.008}
MARKET_VIOL_MULT = {"US": 1.0, "BR": 1.2, "ID": 1.1, "GB": 0.9, "VN": 1.3, "KE": 1.4}
SCENARIO_MARKET_MULT = {("GB", "hate_harassment"): 2.4}   # GB has a structurally higher harassment rate

ACTION_SEVERITY = {"observe": 0, "manual_review": 1, "limit_traffic": 2, "warn": 3, "delay_settlement": 4,
                   "freeze_earnings": 5, "suspend_live": 6, "disable_gifts": 7, "ban_7d": 8}
FINANCIAL_ACTIONS = {"delay_settlement", "freeze_earnings", "disable_gifts"}   # 资金侧处置
ENFORCEMENT_ACTIONS = {"limit_traffic", "warn", "suspend_live", "ban_7d"} | FINANCIAL_ACTIONS
SEVERE_ACTIONS = {"freeze_earnings", "suspend_live", "disable_gifts", "ban_7d"}   # count toward guild health deductions
GUILD_HEALTH_TIERS = [(80, "警告约谈"), (70, "暂停结算"), (60, "扣当月收益"), (50, "清退公示")]

PROBLEMS = {
    "P1_model_drift": {"scenario": "scam_fraud", "model": "model_scam_fraud", "from": "2026-10-13",
                        "shift": 0.15, "victim_strategy": "S-102", "change_id": "C-006"},
    "P2_recall_drop": {"market": "BR", "scenario": "gambling", "from": "2026-10-08",
                        "evasion_share": 0.85, "strategy": "S-115"},
    "P3_overlap": {"strategies": ["S-104", "S-121"]},
    "P4_zombie": {"strategies": ["S-108", "S-119"], "retired_feature_change": "C-007"},
    "P5_traffic_trap": {"market": "ID", "dates": ["2026-10-03", "2026-10-04"], "mult": 1.45},
    "P6_policy_strategy_change": {"market": "GB", "scenario": "hate_harassment", "strategy": "S-112",
                                   "changes": ["C-003", "C-004"]},
    "P7_new_strategy": {"market": "ID", "scenario": "minor_safety", "strategy": "S-126", "change_id": "C-005"},
    "P8_head_host_case": {"market": "US", "date": "2026-10-15", "strategy": "S-102", "session_id": None},
    "P9_guild_spike": {"market": "BR", "guild_id": "G-BR-002", "scenario": "spam_bot",
                       "dates": ["2026-10-13", "2026-10-14", "2026-10-15"], "extra_violation_prob": 0.7},
}
GUILDS_PER_MARKET = {"US": 12, "BR": 12, "ID": 12, "GB": 8, "VN": 4, "KE": 3}
GUILD_FREE_SHARE = 0.45   # share of sessions whose host is not in any guild

MODEL_VERSIONS = [  # (scenario, version, effective_from)
    {"model": "model_adult_content", "scenario": "adult_content", "version": "v5", "effective_from": "2026-06-01"},
    {"model": "model_minor_safety", "scenario": "minor_safety", "version": "v3", "effective_from": "2026-07-15"},
    {"model": "model_scam_fraud", "scenario": "scam_fraud", "version": "v3", "effective_from": "2026-05-20"},
    {"model": "model_scam_fraud", "scenario": "scam_fraud", "version": "v4", "effective_from": "2026-10-13"},
    {"model": "model_violence", "scenario": "violence", "version": "v2", "effective_from": "2026-03-01"},
    {"model": "model_spam_bot", "scenario": "spam_bot", "version": "v2", "effective_from": "2026-04-10"},
    {"model": "model_spam_bot", "scenario": "spam_bot", "version": "v2.1", "effective_from": "2026-10-05"},
    {"model": "model_gambling", "scenario": "gambling", "version": "v3", "effective_from": "2026-08-01"},
    {"model": "model_hate_harassment", "scenario": "hate_harassment", "version": "v4", "effective_from": "2026-07-01"},
    {"model": "model_counterfeit", "scenario": "counterfeit", "version": "v2", "effective_from": "2026-02-01"},
]


def S(sid, name, scenario, markets, threshold, action, owner, created, extra=None, versions=None,
      status="active", notes=""):
    """Build a strategy record. ``versions`` lists historical (version, effective_from, threshold, action)."""
    extra = extra or []
    if versions is None:
        versions = [{"version": "v1", "effective_from": created, "threshold": threshold, "action": action}]
    cur = versions[-1]
    return {
        "strategy_id": sid, "name": name, "scenario": scenario, "markets": markets,
        "score_field": f"score_{scenario}", "threshold": cur["threshold"], "action": cur["action"],
        "extra_conditions": extra, "version": cur["version"], "versions": versions,
        "owner": owner, "status": status, "created_at": created, "updated_at": cur["effective_from"],
        "notes": notes,
    }


STRATEGIES = [
    S("S-101", "色情-高分直接下播", "adult_content", ["ALL"], 0.80, "suspend_live", "策略-内容组", "2026-02-10",
      versions=[{"version": "v1", "effective_from": "2026-02-10", "threshold": 0.85, "action": "suspend_live"},
                {"version": "v2", "effective_from": "2026-06-05", "threshold": 0.80, "action": "suspend_live"}]),
    S("S-102", "诈骗-高分下播", "scam_fraud", ["ALL"], 0.85, "suspend_live", "策略-交易安全组", "2026-03-01",
      versions=[{"version": "v1", "effective_from": "2026-03-01", "threshold": 0.90, "action": "suspend_live"},
                {"version": "v2", "effective_from": "2026-05-22", "threshold": 0.88, "action": "suspend_live"},
                {"version": "v3", "effective_from": "2026-07-18", "threshold": 0.85, "action": "suspend_live"}],
      notes="固定阈值，依赖 model_scam_fraud 原始分"),
    S("S-103", "色情-中分+举报转人审", "adult_content", ["ALL"], 0.60, "manual_review", "策略-内容组", "2026-02-10",
      extra=[{"field": "report_count", "op": ">=", "value": 3}]),
    S("S-104", "刷量-新设备限流", "spam_bot", ["ALL"], 0.75, "limit_traffic", "策略-生态组", "2026-03-15",
      extra=[{"field": "new_device_flag", "op": "==", "value": 1}],
      versions=[{"version": "v1", "effective_from": "2026-03-15", "threshold": 0.80, "action": "limit_traffic"},
                {"version": "v2", "effective_from": "2026-04-20", "threshold": 0.78, "action": "limit_traffic"},
                {"version": "v3", "effective_from": "2026-06-01", "threshold": 0.75, "action": "limit_traffic"},
                {"version": "v4", "effective_from": "2026-07-01", "threshold": 0.75, "action": "limit_traffic"}]),
    S("S-105", "未成年-高分下播", "minor_safety", ["ALL"], 0.70, "suspend_live", "策略-未成年组", "2026-01-20"),
    S("S-106", "未成年-中分+举报人审", "minor_safety", ["ALL"], 0.50, "manual_review", "策略-未成年组", "2026-01-20",
      extra=[{"field": "report_count", "op": ">=", "value": 2}]),
    S("S-107", "暴力-高分下播", "violence", ["ALL"], 0.80, "suspend_live", "策略-内容组", "2026-03-01"),
    S("S-108", "假冒-品牌词表命中警告", "counterfeit", ["ALL"], 0.70, "warn", "策略-电商治理组", "2025-11-01",
      extra=[{"field": "keyword_hits", "op": ">=", "value": 1},
             {"field": "legacy_brand_list_match", "op": "==", "value": 1}],
      notes="依赖特征 legacy_brand_list_match（该特征管线已于 2026-08-01 下线，见 C-007）"),
    S("S-109", "假冒-高分警告", "counterfeit", ["ALL"], 0.85, "warn", "策略-电商治理组", "2026-01-15"),
    S("S-110", "诈骗-外链+新号人审", "scam_fraud", ["ALL"], 0.70, "manual_review", "策略-交易安全组", "2026-04-01",
      extra=[{"field": "external_link_flag", "op": "==", "value": 1},
             {"field": "account_age_days", "op": "<", "value": 30}]),
    S("S-111", "仇恨-高分警告", "hate_harassment", ["ALL"], 0.80, "warn", "策略-社区组", "2026-02-01"),
    S("S-112", "仇恨-GB 举报人审", "hate_harassment", ["GB"], 0.50, "manual_review", "策略-社区组", "2026-05-10",
      extra=[{"field": "report_count", "op": ">=", "value": 1}],
      versions=[{"version": "v1", "effective_from": "2026-05-10", "threshold": 0.85, "action": "manual_review"},
                {"version": "v2", "effective_from": "2026-08-01", "threshold": 0.80, "action": "manual_review"},
                {"version": "v3", "effective_from": "2026-10-11", "threshold": 0.50, "action": "manual_review"}],
      notes="v3 对应 policy P-7（GB 骚扰定义扩大）"),
    S("S-113", "暴力-中分+举报人审", "violence", ["ALL"], 0.60, "manual_review", "策略-内容组", "2026-03-01",
      extra=[{"field": "report_count", "op": ">=", "value": 3}]),
    S("S-114", "诈骗-高打赏新号冻结收益", "scam_fraud", ["ALL"], 0.60, "freeze_earnings", "策略-交易安全组", "2026-06-10",
      extra=[{"field": "gift_amount", "op": ">=", "value": 500},
             {"field": "account_age_days", "op": "<", "value": 14}]),
    S("S-115", "赌博-高分下播", "gambling", ["ALL"], 0.80, "suspend_live", "策略-交易安全组", "2026-02-20"),
    S("S-116", "赌博-关键词人审", "gambling", ["ALL"], 0.55, "manual_review", "策略-交易安全组", "2026-02-20",
      extra=[{"field": "keyword_hits", "op": ">=", "value": 2}]),
    S("S-117", "刷量-极高分下播", "spam_bot", ["ALL"], 0.90, "suspend_live", "策略-生态组", "2026-03-15",
      versions=[{"version": "v1", "effective_from": "2026-03-15", "threshold": 0.88, "action": "suspend_live"},
                {"version": "v2", "effective_from": "2026-09-20", "threshold": 0.90, "action": "suspend_live"}]),
    S("S-118", "色情-极高分封禁", "adult_content", ["ALL"], 0.95, "ban_7d", "策略-内容组", "2026-02-10"),
    S("S-119", "暴力-超大场高分下播", "violence", ["ALL"], 0.95, "suspend_live", "策略-内容组", "2026-04-01",
      extra=[{"field": "viewers_peak", "op": ">=", "value": 50000}], notes="条件极窄，几乎不命中"),
    S("S-120", "未成年-小市场新号观察", "minor_safety", ["VN", "KE"], 0.40, "observe", "策略-未成年组", "2026-07-01",
      extra=[{"field": "account_age_days", "op": "<", "value": 3}]),
    S("S-121", "刷量-新设备限流(v2 重写)", "spam_bot", ["ALL"], 0.72, "limit_traffic", "策略-平台生态组", "2026-08-20",
      extra=[{"field": "new_device_flag", "op": "==", "value": 1}],
      notes="由另一团队重写，与 S-104 条件几乎相同"),
    S("S-122", "诈骗-ID 中分人审", "scam_fraud", ["ID"], 0.75, "manual_review", "策略-交易安全组", "2026-05-05"),
    S("S-123", "仇恨-极高分下播", "hate_harassment", ["ALL"], 0.90, "suspend_live", "策略-社区组", "2026-02-01"),
    S("S-124", "假冒-关键词人审", "counterfeit", ["ALL"], 0.60, "manual_review", "策略-电商治理组", "2026-01-15",
      extra=[{"field": "keyword_hits", "op": ">=", "value": 2}]),
    S("S-125", "刷量-小场高打赏延迟结算", "spam_bot", ["ALL"], 0.60, "delay_settlement", "策略-生态组", "2026-05-01",
      extra=[{"field": "viewers_peak", "op": "<", "value": 20}, {"field": "gift_amount", "op": ">=", "value": 200}]),
    S("S-126", "未成年-ID 中分下播", "minor_safety", ["ID"], 0.60, "suspend_live", "策略-未成年组", "2026-10-12",
      notes="ID 市场专项，2026-10-12 上线"),
    S("S-127", "色情-举报限流（已暂停）", "adult_content", ["ALL"], 0.70, "limit_traffic", "策略-内容组", "2026-04-01",
      extra=[{"field": "report_count", "op": ">=", "value": 1}], status="paused"),
    S("S-128", "赌博-外链下播", "gambling", ["ALL"], 0.70, "suspend_live", "策略-交易安全组", "2026-06-15",
      extra=[{"field": "external_link_flag", "op": "==", "value": 1}]),
    S("S-129", "诈骗-高打赏场次冻结收益", "scam_fraud", ["ALL"], 0.75, "freeze_earnings", "策略-交易安全组", "2026-08-05",
      extra=[{"field": "gift_amount", "op": ">=", "value": 300}], notes="资金侧动作；固定阈值，同样依赖 model_scam_fraud 原始分"),
]

CHANGE_LOG = [
    {"change_id": "C-007", "date": "2026-08-01", "type": "feature_pipeline", "target": "legacy_brand_list_match",
     "scenario": "counterfeit", "markets": ["ALL"], "strategy_id": None, "model": None,
     "summary": "特征 legacy_brand_list_match 管线下线，字段恒为 0；依赖该特征的策略需迁移", "owner": "特征平台"},
    {"change_id": "C-008", "date": "2026-08-20", "type": "strategy_launch", "target": "S-121",
     "scenario": "spam_bot", "markets": ["ALL"], "strategy_id": "S-121", "model": None,
     "summary": "S-121 上线：新设备刷量限流（平台生态组重写版）", "owner": "策略-平台生态组"},
    {"change_id": "C-010", "date": "2026-08-05", "type": "strategy_launch", "target": "S-129",
     "scenario": "scam_fraud", "markets": ["ALL"], "strategy_id": "S-129", "model": None,
     "summary": "S-129 上线：诈骗高打赏场次冻结收益（资金侧处置）", "owner": "策略-交易安全组"},
    {"change_id": "C-001", "date": "2026-09-20", "type": "strategy_update", "target": "S-117",
     "scenario": "spam_bot", "markets": ["ALL"], "strategy_id": "S-117", "model": None,
     "summary": "S-117 v2：阈值 0.88 -> 0.90", "owner": "策略-生态组"},
    {"change_id": "C-002", "date": "2026-10-01", "type": "policy_update", "target": "P-3",
     "scenario": "minor_safety", "markets": ["ALL"], "strategy_id": None, "model": None,
     "summary": "Policy P-3 未成年人安全条款措辞澄清，不改变执行标准", "owner": "T&S Policy"},
    {"change_id": "C-009", "date": "2026-10-05", "type": "model_update", "target": "model_spam_bot",
     "scenario": "spam_bot", "markets": ["ALL"], "strategy_id": None, "model": "model_spam_bot",
     "summary": "model_spam_bot v2 -> v2.1：线上推理热修复，分数分布无变化", "owner": "算法-生态"},
    {"change_id": "C-003", "date": "2026-10-10", "type": "policy_update", "target": "P-7",
     "scenario": "hate_harassment", "markets": ["GB"], "strategy_id": None, "model": None,
     "summary": "Policy P-7：GB 市场骚扰定义扩大（纳入针对性嘲讽），要求策略同步放宽阈值", "owner": "T&S Policy"},
    {"change_id": "C-004", "date": "2026-10-11", "type": "strategy_update", "target": "S-112",
     "scenario": "hate_harassment", "markets": ["GB"], "strategy_id": "S-112", "model": None,
     "summary": "S-112 v3：阈值 0.80 -> 0.50（对齐 P-7），预期人审量上升", "owner": "策略-社区组"},
    {"change_id": "C-005", "date": "2026-10-12", "type": "strategy_launch", "target": "S-126",
     "scenario": "minor_safety", "markets": ["ID"], "strategy_id": "S-126", "model": None,
     "summary": "S-126 上线：ID 未成年人中分直接下播（专项）", "owner": "策略-未成年组"},
    {"change_id": "C-006", "date": "2026-10-13", "type": "model_update", "target": "model_scam_fraud",
     "scenario": "scam_fraud", "markets": ["ALL"], "strategy_id": None, "model": "model_scam_fraud",
     "summary": "model_scam_fraud v3 -> v4：召回提升；分数未重新校准，整体分布上移", "owner": "算法-交易安全"},
]

POLICIES = [
    {"policy_id": "P-3", "scenario": "minor_safety", "title": "未成年人安全", "version": "2026.10"},
    {"policy_id": "P-7", "scenario": "hate_harassment", "title": "仇恨与骚扰", "version": "2026.10-GB"},
]



HIGH_SCORE_STRATEGY = {"adult_content": "S-101", "scam_fraud": "S-102", "minor_safety": "S-105", "violence": "S-107",
                       "gambling": "S-115", "spam_bot": "S-117", "hate_harassment": "S-123", "counterfeit": "S-109"}


def _bump(version):
    major = int(version.lstrip("v").split(".")[0])
    return f"v{major + 1}"


def configure(spec):
    """Relocate the planted problems for an alternative 'world'. Without a spec the default world is unchanged.

    spec keys (all optional except out): seed, out, drift={scenario, from}, recall={market, scenario, from},
    traffic={market, dates}, policy={market}, launch={market, scenario, date}, guild={market, guild_id, scenario},
    head_case={market}.
    """
    global SEED, OUT, SCENARIO_MARKET_MULT
    SEED = int(spec.get("seed", SEED))
    OUT = spec.get("out", OUT)
    by_id = {st["strategy_id"]: i for i, st in enumerate(STRATEGIES)}
    chg = {c["change_id"]: c for c in CHANGE_LOG}
    if "drift" in spec:
        s1, d1 = spec["drift"]["scenario"], spec["drift"]["from"]
        old = [v for v in MODEL_VERSIONS if v["scenario"] == "scam_fraud"]
        MODEL_VERSIONS[:] = [v for v in MODEL_VERSIONS if not (v["scenario"] == "scam_fraud" and v["version"] == "v4")]
        prev = sorted([v for v in MODEL_VERSIONS if v["scenario"] == s1], key=lambda v: v["effective_from"])[-1]["version"]
        new = _bump(prev)
        MODEL_VERSIONS.append({"model": f"model_{s1}", "scenario": s1, "version": new, "effective_from": d1})
        PROBLEMS["P1_model_drift"].update(scenario=s1, model=f"model_{s1}", **{"from": d1},
                                          victim_strategy=HIGH_SCORE_STRATEGY[s1])
        chg["C-006"].update(date=d1, target=f"model_{s1}", scenario=s1, model=f"model_{s1}",
                            summary=f"model_{s1} {prev} -> {new}：召回提升；分数未重新校准，整体分布上移")
        PROBLEMS["P8_head_host_case"]["strategy"] = HIGH_SCORE_STRATEGY[s1]
    if "head_case" in spec:
        PROBLEMS["P8_head_host_case"]["market"] = spec["head_case"]["market"]
    if "recall" in spec:
        r = spec["recall"]
        PROBLEMS["P2_recall_drop"].update(market=r["market"], scenario=r["scenario"], **{"from": r["from"]},
                                          strategy=HIGH_SCORE_STRATEGY[r["scenario"]])
    if "traffic" in spec:
        PROBLEMS["P5_traffic_trap"].update(market=spec["traffic"]["market"], dates=spec["traffic"]["dates"])
    if "policy" in spec:
        m6 = spec["policy"]["market"]
        st = STRATEGIES[by_id["S-112"]]
        st.update(markets=[m6], name=f"仇恨-{m6} 举报人审")
        chg["C-003"].update(markets=[m6], summary=f"Policy P-7：{m6} 市场骚扰定义扩大（纳入针对性嘲讽），要求策略同步放宽阈值")
        chg["C-004"].update(markets=[m6])
        PROBLEMS["P6_policy_strategy_change"]["market"] = m6
        SCENARIO_MARKET_MULT = {(m6, "hate_harassment"): 2.4}
    if "launch" in spec:
        L = spec["launch"]
        STRATEGIES[by_id["S-126"]] = S("S-126", f"{SCENARIO_NAMES[L['scenario']]}-{L['market']} 中分下播", L["scenario"],
                                       [L["market"]], 0.60, "suspend_live", "策略-专项组", L["date"],
                                       notes=f"{L['market']} 市场专项，{L['date']} 上线")
        chg["C-005"].update(date=L["date"], scenario=L["scenario"], markets=[L["market"]],
                            summary=f"S-126 上线：{L['market']} {SCENARIO_NAMES[L['scenario']]}中分直接下播（专项）")
        PROBLEMS["P7_new_strategy"].update(market=L["market"], scenario=L["scenario"])
    if "guild" in spec:
        g = spec["guild"]
        PROBLEMS["P9_guild_spike"].update(market=g["market"], guild_id=g["guild_id"], scenario=g["scenario"])
    PROBLEMS["world_spec"] = {k: v for k, v in spec.items() if k != "out"}


def dstr(d):
    return d.isoformat()


def gen_sessions(rng):
    rows = []
    dates = [START + timedelta(days=i) for i in range(DAYS)]
    trap = PROBLEMS["P5_traffic_trap"]
    for d in dates:
        dow = 1.2 if d.weekday() >= 5 else 1.0
        for m, base in MARKETS.items():
            mult = dow * rng.normal(1.0, 0.05)
            if m == trap["market"] and dstr(d) in trap["dates"]:
                mult *= trap["mult"]
            n = int(rng.poisson(base * mult))
            rows.append((dstr(d), m, n))
    counts = pd.DataFrame(rows, columns=["date", "market", "n"])
    N = int(counts.n.sum())
    df = pd.DataFrame({
        "date": np.repeat(counts.date.values, counts.n.values),
        "market": np.repeat(counts.market.values, counts.n.values),
    })
    df["session_id"] = [f"LS-{i:07d}" for i in range(N)]
    df["host_id"] = [f"H{rng.integers(100000, 999999)}" for _ in range(N)]
    # true label (hidden): at most one violation scenario per session
    probs = np.zeros((N, len(SCENARIOS) + 1))
    for j, s in enumerate(SCENARIOS):
        probs[:, j + 1] = np.array([BASE_VIOL[s] * MARKET_VIOL_MULT[m] * SCENARIO_MARKET_MULT.get((m, s), 1.0) for m in df.market])
    probs[:, 0] = 1 - probs[:, 1:].sum(axis=1)
    cum = probs.cumsum(axis=1)
    u = rng.random(N)
    lab_idx = (u[:, None] > cum).sum(axis=1)
    labels = np.array(["none"] + SCENARIOS)[lab_idx]
    df["true_label"] = labels
    viol = labels != "none"

    tier = rng.choice(["head", "mid", "tail"], size=N, p=[0.02, 0.18, 0.80])
    df["host_tier"] = tier
    age = np.exp(rng.normal(5.0, 1.2, N)).astype(int) + 1
    young = np.isin(labels, ["scam_fraud", "spam_bot"]) & (rng.random(N) < 0.7)
    age[young] = rng.integers(0, 14, young.sum())
    df["account_age_days"] = np.clip(age, 0, 3000)
    viewers = np.exp(rng.normal(4.0, 1.3, N))
    viewers[tier == "head"] *= 60
    viewers[tier == "mid"] *= 6
    df["viewers_peak"] = viewers.astype(int)
    gift = np.where(rng.random(N) < 0.55, 0.0, np.exp(rng.normal(3.5, 1.4, N)))
    gift[labels == "scam_fraud"] *= 3
    df["gift_amount"] = gift.round(0)
    df["duration_min"] = np.clip(np.exp(rng.normal(3.6, 0.7, N)), 1, 600).astype(int)
    rc = rng.poisson(0.08, N)
    rc[viol] = rng.poisson(2.0, viol.sum())
    df["report_count"] = rc
    ext = (rng.random(N) < 0.05).astype(int)
    scam = labels == "scam_fraud"
    ext[scam] = (rng.random(scam.sum()) < 0.6).astype(int)
    gamb = labels == "gambling"
    ext[gamb] = (rng.random(gamb.sum()) < 0.35).astype(int)
    df["external_link_flag"] = ext
    nd = (rng.random(N) < 0.08).astype(int)
    spam = labels == "spam_bot"
    nd[spam] = (rng.random(spam.sum()) < 0.7).astype(int)
    df["new_device_flag"] = nd
    kw = rng.poisson(0.05, N)
    kwv = np.isin(labels, ["counterfeit", "gambling"])
    kw[kwv] = rng.poisson(1.6, kwv.sum())
    df["keyword_hits"] = kw
    df["legacy_brand_list_match"] = 0   # pipeline retired 2026-08-01 (C-007)
    # guild membership (separate RNG stream; Zipf-like sizes; head hosts almost always in a guild)
    rng_g = np.random.default_rng(SEED + 1)
    guild = np.array([""] * N, dtype=object)
    for m, k in GUILDS_PER_MARKET.items():
        idx = np.where(df.market.values == m)[0]
        w = 1.0 / np.arange(1, k + 1) ** 0.8
        w = w / w.sum()
        ids = np.array([f"G-{m}-{i:03d}" for i in range(1, k + 1)])
        pick = rng_g.choice(ids, size=len(idx), p=w)
        free = rng_g.random(len(idx)) < GUILD_FREE_SHARE
        free &= ~(tier[idx] == "head") | (rng_g.random(len(idx)) < 0.1)
        pick[free] = ""
        guild[idx] = pick
    df["guild_id"] = guild

    # model scores per scenario
    for s in SCENARIOS:
        hard = rng.random(N) < 0.12
        sc = rng.beta(1.5, 12, N)
        sc[hard] = rng.beta(5, 4, hard.sum())
        is_v = labels == s
        sc[is_v] = rng.beta(8, 2, is_v.sum())
        df[f"score_{s}"] = sc
    # P2: BR gambling evasion from 2026-10-08 — violators score lower
    p2 = PROBLEMS["P2_recall_drop"]
    ev = (labels == p2["scenario"]) & (df.market.values == p2["market"]) & (df.date.values >= p2["from"]) \
        & (rng.random(N) < p2["evasion_share"])
    df.loc[ev, f"score_{p2['scenario']}"] = rng.beta(3, 5, ev.sum())
    # P1: scam model v4 from 2026-10-13 — distribution shifts up, uncalibrated
    p1 = PROBLEMS["P1_model_drift"]
    drift = df.date.values >= p1["from"]
    col1 = f"score_{p1['scenario']}"
    df.loc[drift, col1] = np.clip(df.loc[drift, col1] + p1["shift"], 0, 0.995)
    # P9: coordinated spam from one guild (post-hoc override so other rows are untouched)
    p9 = PROBLEMS["P9_guild_spike"]
    rng9 = np.random.default_rng(SEED + 9)
    cand = (df.market.values == p9["market"]) & (df.guild_id.values == p9["guild_id"]) & df.date.isin(p9["dates"]).values \
        & (df.true_label.values == "none") & (rng9.random(N) < p9["extra_violation_prob"])
    n9 = int(cand.sum())
    df.loc[cand, "true_label"] = p9["scenario"]
    df.loc[cand, f"score_{p9['scenario']}"] = rng9.beta(10, 1.5, n9)
    df.loc[cand, "new_device_flag"] = (rng9.random(n9) < 0.9).astype(int)
    df.loc[cand, "account_age_days"] = rng9.integers(0, 14, n9)
    df.loc[cand, "report_count"] = rng9.poisson(2.0, n9)
    for s in SCENARIOS:
        df[f"score_{s}"] = df[f"score_{s}"].round(4)
    return df


OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b,
       "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def model_version_for(scenario, d):
    vs = [v for v in MODEL_VERSIONS if v["scenario"] == scenario and v["effective_from"] <= d]
    return vs[-1]["version"] if vs else "unknown"


def gen_hits(df):
    out = []
    for st in STRATEGIES:
        if st["status"] != "active":
            continue
        mask = np.ones(len(df), dtype=bool)
        if st["markets"] != ["ALL"]:
            mask &= df.market.isin(st["markets"]).values
        for c in st["extra_conditions"]:
            mask &= OPS[c["op"]](df[c["field"]].values, c["value"])
        score = df[st["score_field"]].values
        for i, v in enumerate(st["versions"]):
            end = st["versions"][i + 1]["effective_from"] if i + 1 < len(st["versions"]) else "9999-12-31"
            vmask = mask & (df.date.values >= v["effective_from"]) & (df.date.values < end) & (score >= v["threshold"])
            idx = np.where(vmask)[0]
            if len(idx) == 0:
                continue
            sub = df.iloc[idx]
            out.append(pd.DataFrame({
                "session_id": sub.session_id.values, "strategy_id": st["strategy_id"],
                "strategy_version": v["version"], "scenario": st["scenario"], "market": sub.market.values,
                "date": sub.date.values, "score": score[idx], "threshold": v["threshold"], "action": v["action"],
                "model_version": [model_version_for(st["scenario"], d) for d in sub.date.values],
            }))
    hits = pd.concat(out, ignore_index=True).sort_values(["date", "session_id", "strategy_id"]).reset_index(drop=True)
    hits["hit_id"] = [f"HT-{i:07d}" for i in range(len(hits))]
    return hits


def gen_enforcements(hits, df):
    h = hits[hits.action.isin(ENFORCEMENT_ACTIONS)].copy()
    h["sev"] = h.action.map(ACTION_SEVERITY)
    h = h.sort_values(["session_id", "sev", "strategy_id"], ascending=[True, False, True])
    grp = h.groupby("session_id")
    enf = grp.first()[["market", "date", "strategy_id", "scenario", "action"]].rename(
        columns={"strategy_id": "primary_strategy_id", "action": "final_action"})
    enf["triggered_by"] = grp.strategy_id.apply(lambda s: ",".join(sorted(set(s))))
    enf = enf.reset_index()
    enf["enforcement_id"] = [f"EN-{i:07d}" for i in range(len(enf))]
    fin_sessions = set(hits[hits.action.isin(FINANCIAL_ACTIONS)].session_id)
    enf["financial_action"] = enf.session_id.isin(fin_sessions)        # 资金侧动作可与下播并存
    enf["financial_strategy_id"] = enf.session_id.map(hits[hits.action.isin(FINANCIAL_ACTIONS)].drop_duplicates("session_id").set_index("session_id").strategy_id)
    gift = df.set_index("session_id").gift_amount
    enf["frozen_amount"] = np.where(enf.financial_action, enf.session_id.map(gift).values, 0.0).round(0)
    lab = df.set_index("session_id").true_label
    enf["_is_fp"] = enf.session_id.map(lab).values != enf.scenario.values
    return enf


def gen_appeals(enf, rng):
    fp = enf._is_fp.values
    p_app = np.where(fp, 0.70, 0.12)
    # per-session uniforms keyed by session index, so appeals do not reshuffle when unrelated sessions change
    u = np.random.default_rng(SEED + 3).random((400000, 2))
    sidx = enf.session_id.str[3:].astype(int).values
    appealed = u[sidx, 0] < p_app
    a = enf[appealed].copy()
    fp_a = a._is_fp.values
    over = u[sidx[appealed], 1] < np.where(fp_a, 0.90, 0.08)
    a["result"] = np.where(over, "overturned", "upheld")
    a["decision_date"] = a.date   # decided same day for simplicity
    a["appeal_id"] = [f"AP-{i:06d}" for i in range(len(a))]
    a["reason"] = np.where(over, "误判：直播内容不构成违规", "维持原判")
    return a[["appeal_id", "session_id", "enforcement_id", "primary_strategy_id", "scenario", "market", "date",
              "decision_date", "result", "reason"]].rename(columns={"primary_strategy_id": "strategy_id"})


def gen_qa(df, hits, enf, rng):
    """Spot-check labels. Hit samples: up to 10 per strategy/market/day.
    Non-hit samples (per market/day): stratified — reported (report_count>=2) up to 60 + random 50."""
    truth = df.set_index("session_id").true_label
    rows = []
    for (sid, m, d), g in hits.groupby(["strategy_id", "market", "date"]):
        take = g.sample(n=min(10, len(g)), random_state=int(rng.integers(0, 2**31)))
        for r in take.itertuples():
            rows.append({"session_id": r.session_id, "date": d, "market": m, "strategy_id": sid, "scenario": r.scenario,
                         "stratum": "hit", "weight": len(g) / len(take), "qa_label": truth[r.session_id]})
    hit_sessions = set(hits.session_id)
    nonhit = df[~df.session_id.isin(hit_sessions)]
    for (m, d), g in nonhit.groupby(["market", "date"]):
        rep = g[g.report_count >= 2]
        unrep = g[g.report_count < 2]
        rep_s = rep.sample(n=min(60, len(rep)), random_state=int(rng.integers(0, 2**31))) if len(rep) else rep
        unrep_s = unrep.sample(n=min(50, len(unrep)), random_state=int(rng.integers(0, 2**31))) if len(unrep) else unrep
        for sub, name, pop in ((rep_s, "nonhit_reported", len(rep)), (unrep_s, "nonhit_unreported", len(unrep))):
            for r in sub.itertuples():
                rows.append({"session_id": r.session_id, "date": d, "market": m, "strategy_id": None, "scenario": None,
                             "stratum": name, "weight": pop / max(len(sub), 1), "qa_label": truth[r.session_id]})
    qa = pd.DataFrame(rows)
    qa["qa_id"] = [f"QA-{i:06d}" for i in range(len(qa))]
    return qa


def gen_reports(df, rng):
    rep = df[df.report_count > 0]
    n = rep.report_count.values
    base = pd.DataFrame({
        "session_id": np.repeat(rep.session_id.values, n), "market": np.repeat(rep.market.values, n),
        "date": np.repeat(rep.date.values, n), "_label": np.repeat(rep.true_label.values, n)})
    rnd = rng.choice(SCENARIOS, size=len(base))
    use_true = (base._label.values != "none") & (rng.random(len(base)) < 0.9)
    base["reason"] = np.where(use_true, base._label.values, rnd)
    base["report_id"] = [f"RP-{i:07d}" for i in range(len(base))]
    return base[["report_id", "session_id", "market", "date", "reason"]]


def gen_metrics(df, hits, enf, appeals, reports):
    dates = sorted(df.date.unique())
    idx = pd.MultiIndex.from_product([dates, list(MARKETS), SCENARIOS], names=["date", "market", "scenario"])
    sess = df.groupby(["date", "market"]).size().rename("sessions")
    m = pd.DataFrame(index=idx).reset_index()
    m = m.merge(sess.reset_index(), on=["date", "market"], how="left")
    def cnt(frame, name, keys=("date", "market", "scenario")):
        return frame.groupby(list(keys)).size().rename(name).reset_index()
    m = m.merge(cnt(hits, "hits"), on=["date", "market", "scenario"], how="left")
    m = m.merge(cnt(hits[hits.action == "manual_review"], "manual_reviews"), on=["date", "market", "scenario"], how="left")
    m = m.merge(cnt(enf, "enforcements"), on=["date", "market", "scenario"], how="left")
    fe = enf[enf.financial_action].copy()
    fe["scenario"] = fe.financial_strategy_id.map({st["strategy_id"]: st["scenario"] for st in STRATEGIES})
    m = m.merge(cnt(fe, "financial_actions"), on=["date", "market", "scenario"], how="left")
    m = m.merge(fe.groupby(["date", "market", "scenario"]).frozen_amount.sum().rename("frozen_amount").reset_index(), how="left")
    m = m.merge(cnt(appeals, "appeals"), on=["date", "market", "scenario"], how="left")
    m = m.merge(cnt(appeals[appeals.result == "overturned"], "overturns"), on=["date", "market", "scenario"], how="left")
    m = m.merge(reports.groupby(["date", "market", "reason"]).size().rename("reports").reset_index()
                .rename(columns={"reason": "scenario"}), on=["date", "market", "scenario"], how="left")
    m = m.fillna(0)
    for c in ["sessions", "hits", "manual_reviews", "enforcements", "financial_actions", "appeals", "overturns", "reports", "frozen_amount"]:
        m[c] = m[c].astype(int)
    # per-strategy daily
    sidx = pd.MultiIndex.from_product([dates, list(MARKETS), [s["strategy_id"] for s in STRATEGIES]],
                                      names=["date", "market", "strategy_id"])
    ms = pd.DataFrame(index=sidx).reset_index()
    ms = ms.merge(cnt(hits, "hits", ("date", "market", "strategy_id")), how="left")
    ms = ms.merge(cnt(hits[hits.action == "manual_review"], "manual_reviews", ("date", "market", "strategy_id")), how="left")
    ms = ms.merge(cnt(enf, "enforcements", ("date", "market", "primary_strategy_id")).rename(
        columns={"primary_strategy_id": "strategy_id"}), how="left")
    fe = enf[enf.financial_action]
    ms = ms.merge(cnt(fe, "financial_actions", ("date", "market", "financial_strategy_id")).rename(
        columns={"financial_strategy_id": "strategy_id"}), how="left")
    ms = ms.merge(fe.groupby(["date", "market", "financial_strategy_id"]).frozen_amount.sum().rename("frozen_amount").reset_index()
                  .rename(columns={"financial_strategy_id": "strategy_id"}), how="left")
    ms = ms.merge(cnt(appeals, "appeals", ("date", "market", "strategy_id")), how="left")
    ms = ms.merge(cnt(appeals[appeals.result == "overturned"], "overturns", ("date", "market", "strategy_id")), how="left")
    ms = ms.fillna(0)
    for c in ["hits", "manual_reviews", "enforcements", "financial_actions", "appeals", "overturns", "frozen_amount"]:
        ms[c] = ms[c].astype(int)
    ms = ms.merge(sess.reset_index(), on=["date", "market"], how="left")
    return m, ms


def gen_guild_health(df, enf):
    """Daily guild health score. Start 100. Deduct when the guild's severe-enforcement count exceeds 1.2x what the
    market baseline rate (first 14 days) predicts; recover +0.5/day otherwise. A day's deduction is allocated to
    strategies in proportion to each strategy's own excess over its baseline, so a drifting strategy shows up."""
    dates = sorted(df.date.unique())
    base_end = dates[13]
    sev = enf[enf.final_action.isin(SEVERE_ACTIONS)].copy()
    sev["guild_id"] = sev.session_id.map(df.set_index("session_id").guild_id)
    mkt_sessions_base = df[df.date <= base_end].groupby("market").size()
    base_rate = (sev[sev.date <= base_end].groupby("market").size() / mkt_sessions_base).to_dict()
    base_rate_strat = (sev[sev.date <= base_end].groupby(["market", "primary_strategy_id"]).size()
                       / mkt_sessions_base.reindex(sev[sev.date <= base_end].groupby(["market", "primary_strategy_id"]).size().index.get_level_values(0)).values).to_dict()
    gs = df[df.guild_id != ""].groupby(["guild_id", "date"]).size()
    gm = df[df.guild_id != ""].groupby("guild_id").market.first()
    sv = sev[sev.guild_id != ""].groupby(["guild_id", "date"]).size()
    sv_strat = sev[sev.guild_id != ""].groupby(["guild_id", "date", "primary_strategy_id", "scenario"]).size()
    sv_strat_by_gd = {}
    for (g, d, sid, scn), c in sv_strat.items():
        sv_strat_by_gd.setdefault((g, d), []).append((sid, scn, int(c)))
    rows, ded_rows = [], []
    for gid in sorted(gm.index):
        score, m = 100.0, gm[gid]
        for d in dates:
            n = int(gs.get((gid, d), 0))
            k = int(sv.get((gid, d), 0))
            expected = 1.2 * base_rate.get(m, 0) * n
            excess = k - expected
            deduction = round(min(10.0, 0.5 * excess), 2) if excess > 0 else 0.0
            recovery = 0.5 if excess <= 0 else 0.0
            score = float(np.clip(score - deduction + recovery, 0, 100))
            tier = next((name for thr, name in GUILD_HEALTH_TIERS[::-1] if score <= thr), "正常")
            rows.append({"date": d, "guild_id": gid, "market": m, "sessions": n, "severe_enforcements": k,
                         "expected_severe": round(expected, 2), "deduction": deduction, "recovery": recovery,
                         "score": round(score, 2), "tier": tier})
            if deduction > 0 and k > 0:
                parts = sv_strat_by_gd.get((gid, d), [])
                exc = [(sid, scn, c, max(0.0, c - 1.2 * base_rate_strat.get((m, sid), 0) * n)) for sid, scn, c in parts]
                tot = sum(e[3] for e in exc)
                if tot <= 0:
                    exc = [(sid, scn, c, float(c)) for sid, scn, c, _ in exc]
                    tot = sum(e[3] for e in exc)
                for sid, scn, c, w in exc:
                    if w > 0:
                        ded_rows.append({"date": d, "guild_id": gid, "strategy_id": sid, "scenario": scn,
                                         "severe_enforcements": c, "points": round(deduction * w / tot, 3)})
    return pd.DataFrame(rows), pd.DataFrame(ded_rows)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    df = gen_sessions(rng)
    hits = gen_hits(df)
    enf = gen_enforcements(hits, df)
    appeals = gen_appeals(enf, rng)
    qa = gen_qa(df, hits, enf, rng)
    reports = gen_reports(df, rng)
    metrics, metrics_strategy = gen_metrics(df, hits, enf, appeals, reports)

    # P8: pick a head-tier host false-positive suspended by S-102 on 2026-10-15 in US
    p8 = PROBLEMS["P8_head_host_case"]
    cand = enf[(enf.market == p8["market"]) & (enf.date == p8["date"]) & (enf.primary_strategy_id == p8["strategy"])
               & enf._is_fp]
    cand = cand[cand.session_id.map(df.set_index("session_id").host_tier) == "head"]
    if len(cand) == 0:   # force one: promote a FP session's host to head tier
        cand = enf[(enf.market == p8["market"]) & (enf.date == p8["date"]) & (enf.primary_strategy_id == p8["strategy"]) & enf._is_fp]
        sid = cand.session_id.iloc[0]
        df.loc[df.session_id == sid, "host_tier"] = "head"
        df.loc[df.session_id == sid, "viewers_peak"] = 84000
    else:
        sid = cand.session_id.iloc[0]
    p8["session_id"] = sid
    if sid not in set(appeals.session_id):
        e = enf[enf.session_id == sid].iloc[0]
        appeals = pd.concat([appeals, pd.DataFrame([{
            "appeal_id": f"AP-{len(appeals):06d}", "session_id": sid, "enforcement_id": e.enforcement_id,
            "strategy_id": e.primary_strategy_id, "scenario": e.scenario, "market": e.market, "date": e.date,
            "decision_date": e.date, "result": "overturned", "reason": "误判：直播内容不构成违规"}])], ignore_index=True)
        metrics, metrics_strategy = gen_metrics(df, hits, enf, appeals, reports)

    guild_health, guild_ded = gen_guild_health(df, enf)
    guild_health.to_csv(os.path.join(OUT, "guild_health_daily.csv"), index=False)
    guild_ded.to_csv(os.path.join(OUT, "guild_health_deductions.csv"), index=False)
    public_cols = [c for c in df.columns if c != "true_label"]
    df[public_cols].to_csv(os.path.join(OUT, "sessions.csv"), index=False)
    df[["session_id", "true_label"]].to_csv(os.path.join(OUT, "_hidden_truth.csv"), index=False)
    hits.to_csv(os.path.join(OUT, "hit_logs.csv"), index=False)
    enf.drop(columns=["_is_fp"]).to_csv(os.path.join(OUT, "enforcements.csv"), index=False)
    appeals.to_csv(os.path.join(OUT, "appeals.csv"), index=False)
    qa.to_csv(os.path.join(OUT, "qa_samples.csv"), index=False)
    reports.to_csv(os.path.join(OUT, "reports.csv"), index=False)
    metrics.to_csv(os.path.join(OUT, "metrics_daily.csv"), index=False)
    metrics_strategy.to_csv(os.path.join(OUT, "metrics_daily_strategy.csv"), index=False)
    guilds = []
    gsz = df[df.guild_id != ""].groupby("guild_id").agg(sessions=("session_id", "size"), market=("market", "first"),
                                                       head_hosts=("host_tier", lambda x: int((x == "head").sum())))
    for gid, r in gsz.iterrows():
        guilds.append({"guild_id": gid, "market": r.market, "name": f"{MARKET_NAMES[r.market]}公会-{gid.split('-')[-1]}",
                       "sessions_28d": int(r.sessions), "head_hosts": int(r.head_hosts),
                       "tier": "large" if r.sessions >= 0.1 * MARKETS[r.market] * DAYS else ("mid" if r.sessions >= 0.03 * MARKETS[r.market] * DAYS else "small"),
                       "manager": f"公会运营-{r.market}"})
    with open(os.path.join(OUT, "guilds.json"), "w") as f:
        json.dump(guilds, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "strategies.json"), "w") as f:
        json.dump(STRATEGIES, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "change_log.json"), "w") as f:
        json.dump(sorted(CHANGE_LOG, key=lambda c: c["date"]), f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"today": TODAY, "start": dstr(START), "end": dstr(START + timedelta(days=DAYS - 1)),
                   "markets": MARKETS, "market_names": MARKET_NAMES, "scenarios": SCENARIOS,
                   "scenario_names": SCENARIO_NAMES, "model_versions": MODEL_VERSIONS, "policies": POLICIES,
                   "action_severity": ACTION_SEVERITY, "enforcement_actions": sorted(ENFORCEMENT_ACTIONS),
                   "financial_actions": sorted(FINANCIAL_ACTIONS), "severe_actions": sorted(SEVERE_ACTIONS),
                   "guild_health_tiers": GUILD_HEALTH_TIERS, "guild_health_rule": "起始 100 分；当日严重处置数超过市场基准率×1.2 的部分每条扣 0.5 分（日上限 10 分），无超额则 +0.5 分；基准率取窗口前 14 天各市场水平",
                   "small_market_threshold_sessions_per_day": 150, "seed": SEED}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "_embedded_problems.json"), "w") as f:
        json.dump(PROBLEMS, f, ensure_ascii=False, indent=1)
    print(f"sessions={len(df)} hits={len(hits)} enforcements={len(enf)} appeals={len(appeals)} qa={len(qa)} reports={len(reports)}")
    print("P8 head-host case:", sid)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--spec":
        configure(json.load(open(sys.argv[2])))
    main()
