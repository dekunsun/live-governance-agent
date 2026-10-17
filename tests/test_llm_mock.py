"""Plumbing test for the LLM engine with a scripted fake Anthropic client (no network).

Checks: tool_use -> tool_result round trip, validator rejects a hallucinated id and asks the model to
resubmit, the corrected card passes, guardrails run, and the trace records model calls / usage.
"""
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.run import LLMEngine, Platform  # noqa: E402


def block_tool(name, inp, bid):
    return NS(type="tool_use", name=name, input=inp, id=bid)


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        n = len(self.calls)
        usage = NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0)
        if n == 1:
            return NS(stop_reason="tool_use", usage=usage, content=[
                NS(type="text", text="先确认口径并查指标。"),
                block_tool("query_metrics", {"metric": "overturns", "start_date": "2026-10-08", "end_date": "2026-10-15",
                                             "market": "US", "scenario": "scam_fraud"}, "t1"),
                block_tool("get_change_log", {"start_date": "2026-10-08", "end_date": "2026-10-15", "scenario": "scam_fraud"}, "t2")])
        if n == 2:
            # hallucinated strategy id -> validator must bounce it
            return NS(stop_reason="tool_use", usage=usage, content=[block_tool("submit_card", card("S-999", "Q-0001"), "t3")])
        if n == 3:
            last_results = kw["messages"][-1]["content"]
            assert any(r.get("is_error") for r in last_results), "validator should have returned an error tool_result"
            return NS(stop_reason="tool_use", usage=usage, content=[block_tool("submit_card", card("S-102", "Q-0001"), "t4")])
        raise AssertionError("too many model calls")


def card(sid, ref):
    return {"card_type": "attribution", "conclusion": f"US 申诉推翻上升主要来自 {sid}，同期 C-006 模型升级", "confidence": "high",
            "scope": {"market": "US", "scenario": "scam_fraud", "metric": "overturns", "date": "2026-10-15", "low_sample": False},
            "metric": {"key": "overturns", "definition": "申诉推翻数", "note": None},
            "primary_cause": {"type": "model_drift_threshold", "strategy_ids": [sid], "change_ids": ["C-006"], "explanation": "分数上移"},
            "contributing_causes": [], "evidence": [{"ref_id": ref, "type": "metric", "metric": "overturns", "what": "3→13"}],
            "findings": [], "next_steps": ["阈值重校"], "needs_escalation": False, "escalation_reasons": [],
            "clarification_questions": [], "sample_case_ids": [], "platform_links": ["platform://strategy/" + sid]}


def test_llm_loop():
    p = Platform()
    eng = LLMEngine.__new__(LLMEngine)
    eng.p, eng.model, eng.max_turns, eng.effort = p, "fake", 10, "medium"
    fake = FakeMessages()
    eng.client = NS(beta=NS(messages=fake))
    from gov_platform import TOOL_SPECS
    from agent.run import SUBMIT_TOOL
    eng.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOL_SPECS] + [SUBMIT_TOOL]
    tr = eng.run("attribution", "美国昨天误杀申诉涨了")
    assert tr.model_calls == 3, tr.model_calls
    assert tr.card["validation"]["passed"], tr.card["validation"]
    assert tr.card["primary_cause"]["strategy_ids"] == ["S-102"]
    assert tr.usage["input_tokens"] == 3000 and tr.cost_usd() > 0
    tools_called = [e["tool"] for e in tr.events if e["kind"] == "tool_call"]
    assert tools_called == ["query_metrics", "get_change_log"], tools_called
    # the first request must carry system prompt, tools and the user query
    first = fake.calls[0]
    assert "submit_card" in [t["name"] for t in first["tools"]] and first["messages"][0]["content"] == "美国昨天误杀申诉涨了"
    assert first["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in first["betas"]
    print("LLM loop plumbing OK:", tr.card["conclusion"], "| validator bounced S-999 once")


if __name__ == "__main__":
    test_llm_loop()
