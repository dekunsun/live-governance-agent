"""Locks the attribution grader's existing rules (posttrain/harness.py: grade). Rules may be added, not changed:
if one of these tests fails, a published number in reports/posttrain_report.md would no longer be comparable."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gov_platform import Platform  # noqa: E402
from posttrain.harness import grade  # noqa: E402

WORLD = os.path.join(ROOT, "data", "worlds", "w16")
pytestmark = pytest.mark.skipif(not os.path.isdir(WORLD), reason="synthetic worlds not generated (python3 posttrain/worlds.py)")


@pytest.fixture(scope="module")
def platform():
    return Platform(data_dir=WORLD)


def task(**exp):
    base = {"accepted_cause_types": ["model_drift_threshold"], "required_strategy_ids": ["S-107"], "required_change_ids": ["C-006"],
            "needs_escalation": False}
    base.update(exp)
    return {"task_id": "t", "query": "英国暴力血腥10月14日处罚涨了", "expected": base}


def calls(traffic=True, refs=("Q-0001",)):
    c = [{"tool": "query_metrics", "args": {"metric": "sessions"}, "ref_id": refs[0]}] if traffic else []
    c.append({"tool": "get_change_log", "args": {}, "ref_id": "Q-0002"})
    return c


def submit(**kw):
    s = {"cause_type": "model_drift_threshold", "strategy_ids": ["S-107"], "change_ids": ["C-006"], "guild_ids": [], "needs_escalation": False, "conclusion": "x"}
    s.update(kw)
    return s


def test_all_criteria_pass(platform):
    g = grade(task(), submit(), calls(), platform)
    assert g["pass"] and g["score"] == 1.0 and g["unknown_refs"] == []
    assert set(g["criteria"]) == {"submitted", "cause_correct", "strategy_ids", "change_ids", "guild_ids", "escalation", "no_hallucination", "checked_traffic_first"}


def test_no_submit_fails_everything_applicable(platform):
    g = grade(task(), None, calls(), platform)
    assert not g["pass"] and g["criteria"]["submitted"] is False and g["criteria"]["no_hallucination"] is False
    assert g["score"] < 0.5


def test_wrong_cause_is_single_criterion(platform):
    g = grade(task(), submit(cause_type="traffic_driven"), calls(), platform)
    assert not g["pass"] and g["criteria"]["cause_correct"] is False
    assert sum(1 for v in g["criteria"].values() if v is False) == 1


def test_accepted_cause_alternatives(platform):
    t = task(accepted_cause_types=["policy_update", "strategy_change"], required_strategy_ids=["S-112"], required_change_ids=["C-004"])
    assert grade(t, submit(cause_type="strategy_change", strategy_ids=["S-112"], change_ids=["C-004"]), calls(), platform)["criteria"]["cause_correct"]


def test_required_ids_are_subset_not_equality(platform):
    g = grade(task(), submit(strategy_ids=["S-107", "S-101"]), calls(), platform)
    assert g["criteria"]["strategy_ids"] is True
    g = grade(task(), submit(strategy_ids=[]), calls(), platform)
    assert g["criteria"]["strategy_ids"] is False


def test_hallucinated_id_is_caught(platform):
    g = grade(task(), submit(strategy_ids=["S-107", "S-999"]), calls(), platform)
    assert g["criteria"]["no_hallucination"] is False and ("strategy_id", "S-999") in g["unknown_refs"]


def test_traffic_check_required_except_for_exempt_causes(platform):
    g = grade(task(), submit(), calls(traffic=False), platform)
    assert g["criteria"]["checked_traffic_first"] is False and not g["pass"]
    t = task(accepted_cause_types=["needs_clarification"], required_strategy_ids=[], required_change_ids=[], needs_escalation=None)
    g = grade(t, submit(cause_type="needs_clarification", strategy_ids=[], change_ids=[]), [], platform)
    assert g["criteria"]["checked_traffic_first"] is None and g["criteria"]["escalation"] is None and g["pass"]


def test_escalation_must_match_when_defined(platform):
    g = grade(task(needs_escalation=True), submit(needs_escalation=False), calls(), platform)
    assert g["criteria"]["escalation"] is False
    g = grade(task(needs_escalation=True), submit(needs_escalation=True), calls(), platform)
    assert g["criteria"]["escalation"] is True


def test_score_is_mean_of_applicable_criteria(platform):
    g = grade(task(), submit(cause_type="no_anomaly", strategy_ids=[], change_ids=[]), calls(), platform)
    applicable = [v for v in g["criteria"].values() if v is not None]
    assert abs(g["score"] - sum(applicable) / len(applicable)) < 1e-9


def test_format_errors_are_passed_through(platform):
    assert grade(task(), submit(), calls(), platform, format_errors=3)["format_errors"] == 3
