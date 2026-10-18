"""Streamlit demo UI: chat + attribution card + evidence + platform links + expandable trace.

  streamlit run app/app.py
"""
import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.run import Copilot, render_card  # noqa: E402

st.set_page_config(page_title="治理策略巡检与异常归因助手（原型）", page_icon="🛡️", layout="wide")

EXAMPLES = ["美国昨天误杀申诉涨了，帮我看看", "ID 10月4日处罚量涨了", "巴西赌博命中最近下降了", "巴西刷量昨天处罚涨了", "G-BR-002 健康分为什么掉了", "G-US-001 健康分最近为什么扣分", "GB 骚扰 10-13 人审量涨了",
            "印尼未成年 10-13 处罚上升", "肯尼亚昨天误杀申诉涨了", "US 昨天误杀率涨了", "S-104 有没有问题"]
CAUSE_ZH = {"traffic_driven": "流量驱动", "model_drift_threshold": "模型升级后分数漂移、固定阈值失效", "strategy_change": "策略版本变更",
            "new_strategy_launch": "新策略上线", "policy_update": "Policy 更新（预期内）", "recall_drop_evasion": "对抗绕过导致召回下降", "guild_concentration": "集中在单一公会（公会级问题）", "guild_health_real_violations": "机构扣分来自旗下主播真实违规", "guild_health_unfair_deduction": "机构扣分可能源于策略误杀", "guild_health_breach": "机构健康分越档", "paused_strategy": "已暂停策略（不命中属预期）",
            "strategy_overlap": "策略重叠", "zombie_strategy": "僵尸策略", "no_anomaly": "正常波动", "insufficient_data": "样本不足",
            "unexplained": "无法解释", "needs_clarification": "需要澄清", "case_reconstruction": "Case 还原"}


@st.cache_resource
def get_copilot(engine):
    return Copilot(engine=engine)


def has_llm_credentials():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def badge(text, color):
    st.markdown(f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em'>{text}</span>", unsafe_allow_html=True)


def show_card(card, trace):
    conf_color = {"high": "#2e7d32", "medium": "#f9a825", "low": "#c62828", "n/a": "#757575"}.get(card.get("confidence"), "#757575")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        badge(f"置信度 {card.get('confidence')}", conf_color)
    with c2:
        badge("需人工升级" if card.get("needs_escalation") else "只读结论", "#c62828" if card.get("needs_escalation") else "#1565c0")
    st.markdown(f"**结论**：{card.get('conclusion')}")
    pc = card.get("primary_cause")
    if pc and card.get("card_type") != "health_check":
        st.markdown(f"**主因**：`{pc.get('type')}` {CAUSE_ZH.get(pc.get('type'), '')} ｜ 策略 {pc.get('strategy_ids') or '—'} ｜ 变更 {pc.get('change_ids') or '—'}"
                    + (f" ｜ 公会 {pc.get('guild_ids')}" if pc.get('guild_ids') else ""))
        st.caption(pc.get("explanation"))
    if card.get("escalation_reasons"):
        st.warning("升级原因：" + "；".join(card["escalation_reasons"]))
    if card.get("clarification_questions"):
        for q in card["clarification_questions"]:
            st.info(q)
    if card.get("metric") and card["metric"].get("key"):
        with st.expander(f"口径：{card['metric']['key']}（{card['metric'].get('version', '')}）", expanded=bool(card["metric"].get("note"))):
            st.write(card["metric"].get("definition"))
            if card["metric"].get("note"):
                st.write(card["metric"]["note"])
    if card.get("findings"):
        st.markdown("**待处理清单**")
        rows = [{"严重度": f["severity"], "类型": f["finding_type"], "策略": ",".join(f["strategy_ids"]), "市场": f.get("market"),
                 "说明": f["explanation"], "建议": f["suggested_action"], "需人工": f["needs_human"]} for f in card["findings"]]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if card.get("evidence"):
        st.markdown("**证据**（每条都带可追溯的 ref_id）")
        st.dataframe([{"ref_id": e.get("ref_id"), "类型": e.get("type"), "指标": e.get("metric") or "", "内容": e.get("what", "")} for e in card["evidence"]],
                     use_container_width=True, hide_index=True)
    if card.get("sample_case_ids"):
        st.caption("抽样 case：" + "、".join(card["sample_case_ids"][:10]))
    if card.get("next_steps"):
        st.markdown("**建议下一步**")
        for n in card["next_steps"]:
            st.markdown(f"- {n}")
    if card.get("case"):
        with st.expander("处罚链路", expanded=True):
            for step in card["case"]["chain"]:
                st.json(step, expanded=False)
    cols = st.columns(min(len(card.get("platform_links") or []) + 1, 4))
    for i, link in enumerate((card.get("platform_links") or [])[:3]):
        if cols[i].button(f"在平台打开 ↗ {link.split('://')[1][:28]}", key=f"link-{trace.run_id}-{i}"):
            st.toast(f"（模拟）已跳转平台页面，并带上筛选条件：{link}")
    v = card.get("validation") or {}
    st.caption(("✅ 引用校验通过" if v.get("passed") else f"❌ 引用校验失败：{v.get('unknown_references')}") +
               f" ｜ 工具调用 {sum(1 for e in trace.events if e['kind'] == 'tool_call')} 次 ｜ {trace.latency_s}s ｜ engine={trace.engine}"
               + (f" ｜ 模型调用 {trace.model_calls} 次，约 ${trace.cost_usd()}" if trace.engine == "llm" else ""))
    with st.expander("推理过程 / trace（查了什么、依据是什么）"):
        for e in trace.events:
            if e["kind"] == "tool_call":
                st.markdown(f"🔧 `{e['tool']}` {json.dumps(e['args'], ensure_ascii=False)} → {json.dumps(e['result_summary'], ensure_ascii=False)}")
            elif e["kind"] == "model_text":
                st.markdown(f"💬 {e['text']}")
            elif e["kind"] == "guardrails":
                st.markdown(f"🛡️ 护栏：升级={e['needs_escalation']} {e['reasons']} 未知引用={e['unknown_references']}")
            elif e["kind"] == "parse":
                st.markdown(f"🧭 解析：{json.dumps(e['scope'], ensure_ascii=False)}")
        st.download_button("下载完整 trace JSON", json.dumps(trace.to_dict(), ensure_ascii=False, indent=1, default=str),
                           file_name=f"{trace.run_id}.json", key=f"dl-{trace.run_id}")


# ----------------------------------------------------------------------------- layout
with st.sidebar:
    st.title("🛡️ 治理策略助手")
    st.caption("原型 · 全部合成数据 · 只读")
    llm_ok = has_llm_credentials()
    engine = st.radio("引擎", ["rules", "llm"], index=0, help="rules = 确定性 playbook 执行器（离线）；llm = Claude tool use（需 ANTHROPIC_API_KEY）",
                      disabled=not llm_ok)
    if not llm_ok:
        st.caption("未检测到 ANTHROPIC_API_KEY，llm 引擎不可用；当前用 rules 引擎演示。")
    cp = get_copilot(engine if llm_ok else "rules")
    st.markdown("**示例问题**")
    for ex in EXAMPLES:
        if st.button(ex, key="ex-" + ex, use_container_width=True):
            st.session_state["pending"] = ex
    st.divider()
    m = cp.p.meta
    st.caption(f"今天 {m['today']} ｜ 数据 {m['start']} ~ {m['end']} ｜ {len(cp.p.sessions):,} sessions ｜ {len(cp.p.strategies)} 条策略 ｜ 以 demo.user 身份、继承其权限")

tab_chat, tab_health, tab_case = st.tabs(["💬 对话归因", "🩺 策略健康巡检", "🧾 Case 证据包"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    for item in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            show_card(item["card"], item["trace"])
    q = st.chat_input("例如：某市场昨天误杀申诉涨了，帮我看看")
    if st.session_state.get("pending"):
        q, st.session_state["pending"] = st.session_state["pending"], None
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("查口径 → 分解流量 → 下钻 → 查变更 → 抽样 …"):
                tr = cp.ask(q)
            show_card(tr.card, tr)
        st.session_state["history"].append({"query": q, "card": tr.card, "trace": tr})

with tab_health:
    st.markdown("定时扫描所有线上策略：效果下降、长期不命中、互相重叠、模型分漂移导致阈值失效、召回下降。")
    if st.button("运行巡检", type="primary"):
        with st.spinner("扫描 27 条策略 …"):
            tr = cp.health()
        st.session_state["health"] = tr
    if st.session_state.get("health"):
        show_card(st.session_state["health"].card, st.session_state["health"])

with tab_case:
    default_case = json.load(open(os.path.join(cp.p.data_dir, "_embedded_problems.json")))["P8_head_host_case"]["session_id"]
    sid = st.text_input("session id", value=default_case)
    if st.button("还原 case"):
        tr = cp.ask(f"还原 case {sid}")
        show_card(tr.card, tr)
