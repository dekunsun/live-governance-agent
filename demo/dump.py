import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.run import Copilot
Q = ["美国昨天误杀申诉涨了，帮我看看", "印尼 2026-10-03 命中数涨了很多，策略是不是出问题了", "巴西赌博处罚最近少了，是不是违规变少了",
     "英国骚扰 10-13 人审量涨了", "印尼未成年 10-13 处罚上升", "巴西刷量昨天处罚涨了", "G-BR-002 健康分为什么掉了",
     "帮我还原 case LS-0101617", "肯尼亚昨天误杀申诉涨了", "昨天误杀申诉涨了", "S-127 为什么一直没有命中"]
cp = Copilot()
out = []
for q in Q:
    tr = cp.ask(q, save=False); out.append(tr.to_dict())
tr = cp.health(save=False); d = tr.to_dict(); d["query"] = "对所有线上策略做一次健康巡检"; out.append(d)
for d in out:
    for e in d["events"]:
        e.pop("result", None)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1, default=str)
for d in out:
    c = d["card"]; print("==", d["query"], "| tools", d["tool_calls"], "|", (c.get("primary_cause") or {}).get("type"), c.get("confidence"), c.get("needs_escalation")); print("  ", c.get("conclusion"))
