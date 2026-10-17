#!/usr/bin/env python3
"""Evaluate the copilot against the golden set.

  python eval/run_eval.py --engine rules [--repeats 5] [--llm-subset 8]

Outputs eval/results_<engine>.json and reports/eval_results_<engine>.md (tables + raw failure list).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.run import Copilot  # noqa: E402


def load_golden(platform):
    g = json.load(open(os.path.join(ROOT, "eval", "golden_set.json"), encoding="utf-8"))
    head = json.load(open(os.path.join(platform.data_dir, "_embedded_problems.json")))["P8_head_host_case"]["session_id"]
    for c in g["cases"]:
        c["query"] = c["query"].replace("__HEAD_HOST_CASE__", head)
    return g


def score_case(case, card):
    pc = card.get("primary_cause") or {}
    ptype = pc.get("type")
    if case.get("card_type") == "health_check":
        found = [f for f in card.get("findings", []) if f["finding_type"] == "strategy_overlap"]
        ok_type = bool(found)
        ok_s = any(set(case["required_strategy_ids"]) <= set(f["strategy_ids"]) for f in found)
        ok_c = True
    else:
        ok_type = ptype in case["accepted_cause_types"]
        ok_s = set(case["required_strategy_ids"]) <= set(pc.get("strategy_ids") or [])
        ok_c = set(case["required_change_ids"]) <= set(pc.get("change_ids") or [])
        ok_s = ok_s and set(case.get("required_guild_ids", [])) <= set(pc.get("guild_ids") or [])
    esc_ok = True if case.get("needs_escalation") is None else (bool(card.get("needs_escalation")) == case["needs_escalation"])
    metric_ok = True
    if case.get("metric_in"):
        mk = (card.get("metric") or {}).get("key")
        note = (card.get("metric") or {}).get("note") or ""
        metric_ok = mk in case["metric_in"] and ("口径" in note or "分母" in note or "definition" in note.lower())
    halluc = not (card.get("validation") or {}).get("passed", True)
    correct = ok_type and ok_s and ok_c
    return {"correct": correct, "type_ok": ok_type, "strategy_ok": ok_s, "change_ok": ok_c, "escalation_ok": esc_ok,
            "metric_ok": metric_ok, "hallucination": halluc, "got_type": ptype,
            "got_strategy_ids": pc.get("strategy_ids"), "got_change_ids": pc.get("change_ids"), "got_guild_ids": pc.get("guild_ids"),
            "got_escalation": card.get("needs_escalation"), "confidence": card.get("confidence")}


def score_health(truth, card):
    findings = [f for f in card.get("findings", []) if f["severity"] in ("high", "medium")]
    def match(exp, f):
        return f["finding_type"] == exp["finding_type"] and set(exp["strategy_ids"]) <= set(f["strategy_ids"]) \
            and (not exp.get("market") or exp["market"] == f.get("market"))
    hit = [exp for exp in truth["expected"] if any(match(exp, f) for f in findings)]
    missed = [exp for exp in truth["expected"] if exp not in hit]
    fps = [f for f in findings if not any(match(e, f) for e in truth["expected"] + truth["acceptable"])]
    return {"expected": len(truth["expected"]), "found": len(hit), "missed": missed, "false_positives": fps,
            "recall": round(len(hit) / len(truth["expected"]), 3), "precision": round((len(findings) - len(fps)) / max(len(findings), 1), 3),
            "n_alerts": len(findings)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="rules", choices=["rules", "llm"])
    ap.add_argument("--repeats", type=int, default=5, help="runs per case for the consistency check")
    ap.add_argument("--consistency-ids", default="G01,G06,G09,G12,G14,G16,G23,G25")
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--sweep", action="store_true", help="false-alarm sweep over market x scenario x date with no planted cause")
    a = ap.parse_args()
    cp = Copilot(engine=a.engine)
    if a.sweep:
        return sweep(cp)
    g = load_golden(cp.p)
    cases = g["cases"] if not a.only else [c for c in g["cases"] if c["id"] in a.only.split(",")]
    results = []
    for c in cases:
        t0 = time.time()
        tr = cp.ask(c["query"], save=False)
        sc = score_case(c, tr.card)
        results.append({**{k: c[k] for k in ("id", "category", "market", "query")}, **sc, "latency_s": round(time.time() - t0, 3),
                        "tool_calls": sum(1 for e in tr.events if e["kind"] == "tool_call"), "cost_usd": tr.cost_usd(),
                        "usage": tr.usage, "conclusion": tr.card.get("conclusion"), "escalation_reasons": tr.card.get("escalation_reasons")})
        print(f"{c['id']} {'OK ' if sc['correct'] else 'MISS'} {c['category']:<16} got={sc['got_type']} esc={sc['got_escalation']} {round(time.time() - t0, 2)}s")
    # consistency
    cons = {}
    for cid in a.consistency_ids.split(","):
        c = next((x for x in g["cases"] if x["id"] == cid), None)
        if not c:
            continue
        types = [(cp.ask(c["query"], save=False).card.get("primary_cause") or {}).get("type") for _ in range(a.repeats)]
        cons[cid] = {"types": types, "consistent": len(set(types)) == 1}
    # health
    t0 = time.time()
    htr = cp.health(save=False)
    health = score_health(g["health_truth"], htr.card)
    health.update(latency_s=round(time.time() - t0, 3), cost_usd=htr.cost_usd(), tool_calls=sum(1 for e in htr.events if e["kind"] == "tool_call"))
    # aggregates
    n = len(results)
    acc = sum(r["correct"] for r in results) / n
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
    by_mkt = defaultdict(list)
    for r in results:
        if r["market"]:
            by_mkt[r["market"]].append(r["correct"])
    summary = {
        "engine": a.engine, "n_cases": n, "accuracy": round(acc, 3),
        "hallucination_rate": round(sum(r["hallucination"] for r in results) / n, 3),
        "trap_pass_rate": round(sum(r["correct"] for r in results if r["category"] == "traffic_trap") / max(sum(1 for r in results if r["category"] == "traffic_trap"), 1), 3),
        "small_market_escalation_rate": round(sum(r["escalation_ok"] and r["correct"] for r in results if r["category"] == "small_market") / max(sum(1 for r in results if r["category"] == "small_market"), 1), 3),
        "metric_definition_handling": round(sum(r["metric_ok"] for r in results if r["category"] == "metric_ambiguity") / max(sum(1 for r in results if r["category"] == "metric_ambiguity"), 1), 3),
        "escalation_rule_accuracy": round(sum(r["escalation_ok"] for r in results) / n, 3),
        "consistency_rate": round(sum(v["consistent"] for v in cons.values()) / max(len(cons), 1), 3),
        "by_category": {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()},
        "by_market": {k: round(sum(v) / len(v), 3) for k, v in by_mkt.items()},
        "latency_s": {"mean": round(sum(r["latency_s"] for r in results) / n, 3), "max": max(r["latency_s"] for r in results)},
        "cost_usd": {"mean": round(sum(r["cost_usd"] for r in results) / n, 4), "total": round(sum(r["cost_usd"] for r in results), 4)},
        "tool_calls_mean": round(sum(r["tool_calls"] for r in results) / n, 1),
        "health": health, "consistency": cons,
    }
    out = {"summary": summary, "results": results}
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "eval", f"results_{a.engine}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    write_md(summary, results, a.engine)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("consistency",)}, ensure_ascii=False, indent=1))


PLANTED = {  # (market, scenario) -> (cause, start_date, direction, metric word the effect is on)
    ("*", "scam_fraud"): ("model_drift_threshold", "2026-10-13", "up", "处罚"),
    ("BR", "gambling"): ("recall_drop_evasion", "2026-10-08", "down", "命中"),
    ("GB", "hate_harassment"): ("policy_update", "2026-10-11", "up", "人审"),
    ("ID", "minor_safety"): ("new_strategy_launch", "2026-10-12", "up", "处罚"),
    ("BR", "spam_bot"): ("guild_concentration", "2026-10-13", "up", "处罚"),
}


def sweep(cp):
    """Ask '<market> <scenario> <date> 处罚涨了/下降了' for every combination over the last 7 days; classify answers."""
    from datetime import date, timedelta
    meta = cp.p.meta
    dates = [(date.fromisoformat(meta["end"]) - timedelta(days=i)).isoformat() for i in range(7)]
    rows, tally = [], {}
    for m in meta["markets"]:
        for sc in meta["scenarios"]:
            for d in dates:
                planted = PLANTED.get((m, sc)) or PLANTED.get(("*", sc))
                for direction, word in (("up", "涨了"), ("down", "下降了")):
                    metric_word = planted[3] if planted else "处罚"
                    card = cp.ask(f"{m} {sc} {d} {metric_word}{word}", save=False).card
                    t = (card.get("primary_cause") or {}).get("type")
                    exp = None
                    if planted and planted[2] == direction and planted[1] <= d:
                        exp = planted[0]
                    if exp and t == exp:
                        k = "planted_found"
                    elif exp and t == "insufficient_data" and m in ("VN", "KE"):
                        k = "planted_small_market_escalated"
                    elif exp and t == "no_anomaly":
                        k = "planted_missed"
                    elif exp:
                        k = "planted_wrong_cause"
                    elif t in ("no_anomaly", "traffic_driven"):
                        k = "clean_no_alarm"
                    elif t == "insufficient_data" and m in ("VN", "KE"):
                        k = "clean_small_market_escalated"
                    elif planted and t == planted[0]:
                        k = "planted_found_outside_window"   # e.g. drift still visible in a later single-day query
                    else:
                        k = "false_alarm"
                    tally[k] = tally.get(k, 0) + 1
                    rows.append({"market": m, "scenario": sc, "date": d, "direction": direction, "got": t, "expected": exp, "class": k,
                                 "escalated": card.get("needs_escalation"), "conclusion": card.get("conclusion")})
    n = len(rows)
    clean = [r for r in rows if r["expected"] is None]
    out = {"n": n, "tally": tally, "false_alarm_rate_on_clean_cells": round(tally.get("false_alarm", 0) / max(len(clean), 1), 3),
           "planted_detection_rate": round(tally.get("planted_found", 0) / max(sum(1 for r in rows if r["expected"]), 1), 3),
           "problems": [r for r in rows if r["class"] in ("false_alarm", "planted_wrong_cause", "planted_missed")]}
    with open(os.path.join(ROOT, "eval", f"sweep_{cp.engine}.json"), "w") as f:
        json.dump({"summary": out, "rows": rows}, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def write_md(s, results, engine):
    L = [f"# 评估结果（engine={engine}）", "",
         "| 评估项 | 结果 | MVP 门槛（示例） | 达标 |", "|---|---|---|---|",
         f"| 归因准确率 | {s['accuracy']:.0%} ({sum(r['correct'] for r in results)}/{s['n_cases']}) | ≥ 80% | {'✅' if s['accuracy'] >= 0.8 else '❌'} |",
         f"| 幻觉率（引用不存在的 id） | {s['hallucination_rate']:.0%} | 0 | {'✅' if s['hallucination_rate'] == 0 else '❌'} |",
         f"| 陷阱场景通过率（流量涨≠策略差） | {s['trap_pass_rate']:.0%} | 100% | {'✅' if s['trap_pass_rate'] == 1 else '❌'} |",
         f"| 口径歧义处理 | {s['metric_definition_handling']:.0%} | 100% | {'✅' if s['metric_definition_handling'] == 1 else '❌'} |",
         f"| 小市场正确升级 | {s['small_market_escalation_rate']:.0%} | 100% | {'✅' if s['small_market_escalation_rate'] == 1 else '❌'} |",
         f"| 升级规则符合预期（全部 case） | {s['escalation_rule_accuracy']:.0%} | ≥ 90% | {'✅' if s['escalation_rule_accuracy'] >= 0.9 else '❌'} |",
         f"| 一致性（同一输入跑 5 次主因一致） | {s['consistency_rate']:.0%} | ≥ 90% | {'✅' if s['consistency_rate'] >= 0.9 else '❌'} |",
         f"| 巡检召回 / 精确率 | {s['health']['recall']:.0%} / {s['health']['precision']:.0%}（告警 {s['health']['n_alerts']} 条） | 重大问题不漏 / 误报可接受 | {'✅' if s['health']['recall'] == 1 else '❌'} |",
         f"| 单次归因延迟（均值 / 最大） | {s['latency_s']['mean']}s / {s['latency_s']['max']}s | < 3 分钟 | ✅ |",
         f"| 单次成本（均值） | ${s['cost_usd']['mean']} | — | — |",
         f"| 平均工具调用次数 | {s['tool_calls_mean']} | — | — |", "",
         "## 按类别", "", "| 类别 | 准确率 |", "|---|---|"]
    L += [f"| {k} | {v:.0%} |" for k, v in s["by_category"].items()]
    L += ["", "## 按市场", "", "| 市场 | 准确率 |", "|---|---|"]
    L += [f"| {k} | {v:.0%} |" for k, v in s["by_market"].items()]
    L += ["", "## 巡检", "", f"- 期望发现 {s['health']['expected']} 条，发现 {s['health']['found']} 条；漏报 {s['health']['missed']}；误报 {[(f['finding_type'], f['strategy_ids'], f.get('market')) for f in s['health']['false_positives']]}",
          f"- 工具调用 {s['health']['tool_calls']} 次，耗时 {s['health']['latency_s']}s", "", "## 逐条结果", "",
          "| id | 类别 | 市场 | 结果 | 得到的主因 | 升级 | 结论 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['id']} | {r['category']} | {r['market'] or '-'} | {'✅' if r['correct'] else '❌'} | {r['got_type']} {r['got_strategy_ids'] or ''} | {r['got_escalation']} | {(r['conclusion'] or '')[:90].replace('|', '/')} |")
    fails = [r for r in results if not r["correct"] or r["hallucination"] or not r["escalation_ok"] or not r["metric_ok"]]
    L += ["", "## 未通过项（原始）", ""]
    L += [f"- {r['id']} {r['category']}：主因={r['got_type']} 策略={r['got_strategy_ids']} 变更={r['got_change_ids']} 升级={r['got_escalation']}（escalation_ok={r['escalation_ok']}, metric_ok={r['metric_ok']}, hallucination={r['hallucination']}）" for r in fails] or ["- 无"]
    with open(os.path.join(ROOT, "reports", f"eval_results_{engine}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
