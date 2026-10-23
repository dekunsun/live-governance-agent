#!/usr/bin/env python3
"""Leaderboard-style result charts (docs/images/results-leaderboard.svg, results-cost-vs-accuracy.svg,
results-category-heatmap.svg). Pure Python SVG in the same style as build.py; PNGs via docs/diagrams/render.js.
All numbers come from reports/posttrain_report.md and the summary JSONs it cites (synthetic data, greedy decoding).
Model-type palette validated with the dataviz validator: orange = open small fine-tuned, blue = closed frontier,
aqua = open frontier, gray = reference / untrained. Every mark carries a text label, so color is never the only cue."""
import math
from build import SVG

ORANGE, BLUE, AQUA, GRAY = "#eb6834", "#2a78d6", "#1baf7a", "#9a978f"
INK, INK2, GRID, BAND = "#1a1a1a", "#5e5b55", "#e5e7eb", "#f6f7f4"
TYPE = {"small": (ORANGE, "开源小模型"), "closed": (BLUE, "闭源前沿"), "open": (AQUA, "开源前沿"), "ref": (GRAY, "参考 / 不训练")}

# name, type, training, pass (lo, hi) %, cause %, escalation %, hallucination, avg calls, cost per 194-question run
ROWS = [
    ("规则引擎", "ref", "写死的规则（老师与标准答案）", (100, 100), "100%", "100%", "0", "5.2", "—"),
    ("Qwen3-8B", "small", "微调，831 条示范", (96.4, 96.4), "96.4%", "98.9%", "0", "5.2", "≈ $0.03"),
    ("Qwen3-4B（当前版本）", "small", "微调，混合示范 2 epoch，两个种子", (93.8, 95.9), "94–96%", "97–99%", "0", "5.0", "≈ $0.02"),
    ("Claude Sonnet 5.5", "closed", "不训练，提示里放 4 条示范", (81.4, 81.4), "89.7%", "83.1%", "0", "4.5", "$2.82"),
    ("Claude Sonnet 5.5", "closed", "不训练，零样本", (79.4, 79.4), "89.2%", "79.2%", "0", "4.2", "$2.55"),
    ("Qwen3-0.6B", "small", "微调，374 条示范", (75.3, 75.3), "77.3%", "86.0%", "0", "4.7", "≈ $0.01"),
    ("Qwen3.5-397B-A17B", "open", "不训练，零样本", (61.3, 61.3), "76.8%", "78.7%", "0", "9.1", "$5.21"),
    ("Qwen3-4B", "ref", "不训练", (3.6, 3.6), "10.9%", "73.9%", "0", "7.0", "≈ $0.02"),
]
FOOT = ("费用 = 跑一遍 194 题的推理费用。API 为按用量记录的账单估算（Sonnet 大部分走 Batch 五折）；自部署 = 评测时 vLLM 并发 32 的 GPU 占用 × A40 每小时 0.59 美元；"
        "训练一次性另计：4B 约 0.8 美元、8B 约 0.7 美元。")


def badge(s, x, y, t):
    c, lab = TYPE[t]
    s.el.append(f'<circle cx="{x + 6}" cy="{y}" r="5" fill="{c}"/>')
    s.text(x + 16, y, lab, 12.5, anchor="start", color=INK2)


def leaderboard():
    W, top, rh = 1500, 150, 56
    H = top + len(ROWS) * rh + 90
    s = SVG(W, H)
    s.text(W / 2, 32, "归因排行榜：同一套 194 道题、同一个评分器、同一个 harness", 19, bold=True)
    s.text(W / 2, 58, "合成数据，贪心解码；按归因通过率（8 项全对）排序；颜色表示模型类型", 13, color=INK2)
    cols = [(40, "#", "start"), (80, "模型", "start"), (330, "类型", "start"), (460, "训练方式", "start"), (745, "归因通过（8 项全对）", "start"),
            (1140, "主因正确", "middle"), (1225, "升级判断", "middle"), (1295, "幻觉", "middle"), (1365, "平均调用", "middle"), (1445, "跑一遍费用", "middle")]
    for x, lab, a in cols:
        s.text(x, top - 26, lab, 13, anchor=a, bold=True, color=INK2)
    s.el.append(f'<line x1="30" y1="{top - 10}" x2="{W - 30}" y2="{top - 10}" stroke="#c9c7c0" stroke-width="1.2"/>')
    bx0, bx1 = 745, 985
    for i, (name, t, train, (lo, hi), cause, esc, hall, calls, cost) in enumerate(ROWS):
        y = top + i * rh + rh / 2 - 6
        if i % 2 == 0:
            s.el.append(f'<rect x="30" y="{y - rh / 2 + 4}" width="{W - 60}" height="{rh - 4}" fill="{BAND}" rx="6"/>')
        s.text(40, y, str(i + 1), 14, anchor="start", color=INK2)
        s.text(80, y, name, 14.5, anchor="start", bold=(name.endswith("当前版本）")))
        badge(s, 330, y, t)
        s.text(460, y, train, 12.5, anchor="start", color=INK2)
        c = TYPE[t][0]
        x_lo, x_hi = bx0 + (bx1 - bx0) * lo / 100, bx0 + (bx1 - bx0) * hi / 100
        s.el.append(f'<rect x="{bx0}" y="{y - 9}" width="{bx1 - bx0}" height="18" fill="#eceae4" rx="4"/>')
        s.el.append(f'<rect x="{bx0}" y="{y - 9}" width="{max(x_lo - bx0, 3)}" height="18" fill="{c}" rx="4"/>')
        if hi > lo:
            s.el.append(f'<rect x="{x_lo}" y="{y - 9}" width="{x_hi - x_lo}" height="18" fill="{c}" fill-opacity="0.45"/>')
        lab = f"{lo:.1f}%" if hi == lo else f"{lo:.1f}–{hi:.1f}%"
        s.text(bx1 + 12, y, lab, 13.5, anchor="start", bold=True)
        for x, v in ((1140, cause), (1225, esc), (1295, hall), (1365, calls), (1445, cost)):
            s.text(x, y, v, 13.5)
    yb = top + len(ROWS) * rh + 22
    x = 80
    for t in ("small", "closed", "open", "ref"):
        badge(s, x, yb, t); x += 170
    s.text(x + 10, yb, "浅色延伸 = 两个随机种子之间的范围", 12.5, anchor="start", color=INK2)
    s.text(80, yb + 30, FOOT, 11.5, anchor="start", color=INK2)
    s.save("results-leaderboard.svg")


def cost_vs_accuracy():
    pts = [  # label, type, cost $, pass lo, pass hi, label dx, dy
        ("Qwen3-4B 微调（当前版本）", "small", 0.02, 93.8, 95.9, 14, 22),
        ("Qwen3-8B 微调", "small", 0.03, 96.4, 96.4, 14, -14),
        ("Qwen3-0.6B 微调", "small", 0.01, 75.3, 75.3, 14, 4),
        ("Qwen3-4B 不训练", "ref", 0.02, 3.6, 3.6, 14, 4),
        ("Sonnet 5.5 零样本", "closed", 2.55, 79.4, 79.4, -14, 18),
        ("Sonnet 5.5 少样本", "closed", 2.82, 81.4, 81.4, -14, -14),
        ("Qwen3.5-397B 零样本", "open", 5.21, 61.3, 61.3, -14, 4),
    ]
    W, H = 1200, 720
    s = SVG(W, H)
    s.text(W / 2, 32, "准确率 vs 跑一遍的推理费用（越靠左上越好）", 19, bold=True)
    s.text(W / 2, 58, "合成数据；横轴是对数刻度；训练是一次性成本不计入（4B 约 0.8 美元）", 13, color=INK2)
    x0, x1, y0, y1 = 140, 1080, 620, 100
    lmin, lmax = math.log10(0.005), math.log10(10)
    X = lambda c: x0 + (x1 - x0) * (math.log10(c) - lmin) / (lmax - lmin)
    Y = lambda p: y0 - (y0 - y1) * p / 100
    for p in (0, 25, 50, 75, 100):
        s.el.append(f'<line x1="{x0}" y1="{Y(p)}" x2="{x1}" y2="{Y(p)}" stroke="{GRID}" stroke-width="1"/>')
        s.text(x0 - 12, Y(p), f"{p}%", 12, anchor="end", color=INK2)
    for c, lab in ((0.01, "$0.01"), (0.1, "$0.1"), (1, "$1"), (10, "$10")):
        s.el.append(f'<line x1="{X(c)}" y1="{y1}" x2="{X(c)}" y2="{y0}" stroke="{GRID}" stroke-width="1"/>')
        s.text(X(c), y0 + 18, lab, 12, color=INK2)
    s.text((x0 + x1) / 2, y0 + 44, "跑一遍 194 题的推理费用（美元，对数刻度）", 13, color=INK2)
    s.el.append(f'<text x="{x0 - 70}" y="{(y0 + y1) / 2}" transform="rotate(-90 {x0 - 70} {(y0 + y1) / 2})" text-anchor="middle" font-size="13" fill="{INK2}">归因通过率</text>')
    s.el.append(f'<rect x="{x0 + 4}" y="{y1 + 4}" width="{X(0.1) - x0 - 8}" height="{Y(90) - y1 - 8}" fill="{ORANGE}" fill-opacity="0.07" rx="8"/>')
    s.text(x0 + 16, y1 + 18, "又便宜又准", 12.5, anchor="start", color=INK2)
    for lab, t, c, lo, hi, dx, dy in pts:
        col = TYPE[t][0]; cx = X(c)
        if hi > lo:
            s.el.append(f'<line x1="{cx}" y1="{Y(lo)}" x2="{cx}" y2="{Y(hi)}" stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
        cy = Y((lo + hi) / 2)
        s.el.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{col}" stroke="#fff" stroke-width="2"/>')
        txt = f"{lab}  {lo:.1f}%" if hi == lo else f"{lab}  {lo:.1f}–{hi:.1f}%"
        s.text(cx + dx, cy + dy, txt, 13, anchor="start" if dx > 0 else "end")
    yb = H - 34; x = x0
    for t in ("small", "closed", "open", "ref"):
        badge(s, x, yb, t); x += 170
    s.save("results-cost-vs-accuracy.svg")


CATS = [  # label, n, 4B s1, 4B s2, 8B, Sonnet fs, Sonnet zs, Qwen3.5, 4B untrained (n may differ: v2 harness set)
    ("“误判率上升”问法", 16, 6, 9, 12, 4, 7, 9, (0, 16)),
    ("模型漂移、固定阈值失效", 32, 22, 24, 28, 20, 18, 21, (0, 32)),
    ("公会集中违规", 14, 13, 14, 14, 13, 8, 8, (0, 14)),
    ("新策略上线", 16, 15, 16, 16, 10, 14, 12, (0, 16)),
    ("召回下降（对抗绕过）", 11, 11, 11, 10, 7, 11, 10, (0, 11)),
    ("流量驱动", 9, 9, 9, 9, 9, 9, 4, (5, 9)),
    ("没有异常", 48, 48, 48, 46, 47, 46, 29, (2, 48)),
    ("数据不足", 16, 16, 16, 16, 16, 15, 11, (0, 14)),
    ("需要澄清", 16, 16, 16, 16, 16, 16, 8, (0, 16)),
    ("policy 更新", 16, 16, 16, 16, 14, 14, 14, (0, 16)),
    ("公会健康分（真实违规）", 16, 16, 16, 16, 6, 3, 2, (0, 16)),
]
HCOLS = [("Qwen3-4B 微调", "small", "两个种子"), ("Qwen3-8B 微调", "small", ""), ("Sonnet 5.5 少样本", "closed", ""),
         ("Sonnet 5.5 零样本", "closed", ""), ("Qwen3.5-397B 零样本", "open", ""), ("Qwen3-4B 不训练", "ref", "")]


def ramp(p):
    a, b = (241, 245, 249), (51, 65, 85)  # slate, light -> dark (single-hue sequential)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * p) for i in range(3))


def heatmap():
    cw, rh, x0, top = 172, 44, 300, 150
    W, H = x0 + cw * len(HCOLS) + 40, top + len(CATS) * rh + 90
    s = SVG(W, H)
    s.text(W / 2, 32, "按问题类别：每类答对的题数", 19, bold=True)
    s.text(W / 2, 58, "合成数据；颜色越深越好；微调模型几乎全对，前沿模型的失分集中在平台口径相关的几类", 13, color=INK2)
    for j, (lab, t, sub) in enumerate(HCOLS):
        cx = x0 + j * cw + cw / 2
        w = sum(12.5 if ord(ch) > 127 else 7.2 for ch in lab)
        s.el.append(f'<circle cx="{cx - w / 2 - 4}" cy="{top - 44}" r="5" fill="{TYPE[t][0]}"/>')
        s.text(cx + 6, top - 44, lab, 12.5, bold=True)
        if sub:
            s.text(cx, top - 24, sub, 11.5, color=INK2)
    for i, (lab, n, a1, a2, b8, sfs, szs, qw, base) in enumerate(CATS):
        y = top + i * rh
        s.text(x0 - 14, y + rh / 2, f"{lab}（{n}）", 13, anchor="end", bold=(i < 2))
        cells = [((a1 + a2) / 2 / n, f"{a1} / {a2}"), (b8 / n, f"{b8}"), (sfs / n, f"{sfs}"), (szs / n, f"{szs}"), (qw / n, f"{qw}"),
                 (base[0] / base[1], f"{base[0]}" + ("" if base[1] == n else f" / {base[1]}"))]
        for j, (p, txt) in enumerate(cells):
            x = x0 + j * cw
            s.el.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{rh - 4}" fill="{ramp(p)}" rx="5"/>')
            s.text(x + cw / 2, y + rh / 2, txt, 13.5, bold=True, color="#ffffff" if p >= 0.55 else INK)
    yb = top + len(CATS) * rh + 26
    for k, p in enumerate((0, 0.25, 0.5, 0.75, 1)):
        s.el.append(f'<rect x="{x0 + k * 46}" y="{yb - 9}" width="44" height="18" fill="{ramp(p)}" rx="3"/>')
    s.text(x0 + 5 * 46 + 10, yb, "0% → 100% 答对", 12, anchor="start", color=INK2)
    s.text(x0, yb + 28, "格子里是答对题数（括号里是该类总题数）；“误判率上升”问法属于模型漂移类，单独列出；不训练的 4B 来自较早的 harness 版本，个别类别题数不同，已在格内注明。", 11.5, anchor="start", color=INK2)
    s.save("results-category-heatmap.svg")


if __name__ == "__main__":
    leaderboard(); cost_vs_accuracy(); heatmap()
