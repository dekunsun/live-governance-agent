#!/usr/bin/env python3
"""Leaderboard board in the style of public LLM leaderboards (dark theme): KPI tiles + ranked table with top-3 chips,
price-vs-performance scatter with a Pareto line, failure composition, and a per-category heatmap.
Writes docs/images/results-board.svg, results-pareto.svg, results-failures.svg, results-category-heatmap.svg;
PNGs via docs/diagrams/render.js. Numbers come from reports/posttrain_report.md and the summary / per-task JSONL files it
cites (synthetic data, greedy decoding). Palettes validated with the dataviz validator on the dark surface; every mark is
also labeled in text."""
import math
from build import SVG

BG, CARD, LINE, INK, INK2, INK3 = "#111214", "#1a1b1e", "#2b2d31", "#ecebe7", "#a3a19a", "#6f6d68"
ORANGE, BLUE, AQUA, GRAY = "#d95926", "#3987e5", "#199e70", "#8a8780"
TYPE = {"small": (ORANGE, "开源小模型"), "closed": (BLUE, "闭源前沿"), "open": (AQUA, "开源前沿"), "ref": (GRAY, "参考 / 不训练")}
CHIP = [("#5b4413", "#f6c35c"), ("#5a2c1b", "#f2a27c"), ("#33363b", "#ecebe7")]  # rank 1, 2, 3 (bg, text)

# name, type, training, pass (lo, hi), cause (lo, hi), escalation (lo, hi), avg calls, cost per 194-question run (None = n/a)
MODELS = [
    ("Qwen3-8B", "small", "微调 · 831 条示范", (96.4, 96.4), (96.4, 96.4), (98.9, 98.9), 5.2, 0.03),
    ("Qwen3-4B（当前版本）", "small", "微调 · 混合示范 2 epoch · 两个种子", (93.8, 95.9), (94.3, 96.4), (96.6, 98.9), 5.0, 0.02),
    ("Claude Sonnet 5.5", "closed", "不训练 · 提示里 4 条示范", (81.4, 81.4), (89.7, 89.7), (83.1, 83.1), 4.5, 2.82),
    ("Claude Sonnet 5.5", "closed", "不训练 · 零样本", (79.4, 79.4), (89.2, 89.2), (79.2, 79.2), 4.2, 2.55),
    ("Qwen3-0.6B", "small", "微调 · 374 条示范", (75.3, 75.3), (77.3, 77.3), (86.0, 86.0), 4.7, 0.01),
    ("Qwen3.5-397B-A17B", "open", "不训练 · 零样本", (61.3, 61.3), (76.8, 76.8), (78.7, 78.7), 9.1, 5.21),
    ("Qwen3-4B", "ref", "不训练", (3.6, 3.6), (10.9, 10.9), (73.9, 73.9), 7.0, 0.02),
]


def dark(w, h):
    s = SVG(w, h)
    s.el[0] = f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>'
    return s


def card(s, x, y, w, h):
    s.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{CARD}" stroke="{LINE}" stroke-width="1"/>')


def dot(s, x, y, color, r=5):
    s.el.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')


def fmt(v):
    lo, hi = v
    return f"{lo:.1f}" if lo == hi else f"{lo:.1f}–{hi:.1f}"


def header(s, w, kicker, title, sub):
    s.text(60, 40, kicker, 12, anchor="start", color=INK3)
    s.text(60, 74, title, 26, anchor="start", bold=True, color=INK)
    s.text(60, 106, sub, 13.5, anchor="start", color=INK2)


def board():
    W = 1600
    tiles = [("归因最高", "Qwen3-8B 微调", "96.4%  ·  194 题"), ("性价比最高", "Qwen3-4B 微调", "94–96%  ·  跑一遍约 $0.02"),
             ("前沿模型最佳", "Claude Sonnet 5.5", "81.4%  ·  提示里 4 条示范"), ("幻觉", "全部配置为 0", "编号必须本轮查到过，校验层拦截"),
             ("前沿模型最弱的一类", "公会健康分", "2–6 / 16  ·  微调模型 16 / 16"), ("少样本的收益", "+4 题", "79.4% → 81.4%，还带偏了抽样")]
    th, top_t = 92, 140
    tab_y = top_t + 2 * th + 40
    rh = 64
    H = tab_y + 70 + rh * (len(MODELS) + 1) + 90
    s = dark(W, H)
    header(s, W, "治理归因 · 原型排行榜", "归因排行榜 — 同一套 194 道题、同一个评分器、同一个 harness",
           "合成数据，贪心解码。归因通过 = 8 项检查全对；颜色表示模型类型；每列前三名高亮。")
    tw = (W - 120 - 2 * 20) / 3
    for i, (k, v, sub) in enumerate(tiles):
        x = 60 + (i % 3) * (tw + 20); y = top_t + (i // 3) * th
        card(s, x, y, tw, th - 14)
        s.text(x + 22, y + 22, k, 12.5, anchor="start", color=INK2)
        s.text(x + 22, y + 48, v, 18, anchor="start", bold=True, color=INK)
        s.text(x + 22, y + 68, sub, 12, anchor="start", color=INK3)
    card(s, 60, tab_y, W - 120, rh * (len(MODELS) + 1) + 70)
    cols = [(100, "排名", "middle"), (150, "模型", "start"), (720, "归因通过 ↓", "middle"), (880, "主因正确", "middle"), (1030, "升级判断", "middle"),
            (1160, "幻觉", "middle"), (1270, "平均调用", "middle"), (1420, "推理费用 / 轮", "middle")]
    hy = tab_y + 36
    for x, lab, a in cols:
        s.text(x, hy, lab, 12.5, anchor=a, bold=True, color=INK2)
    s.el.append(f'<line x1="80" y1="{hy + 20}" x2="{W - 80}" y2="{hy + 20}" stroke="{LINE}" stroke-width="1"/>')
    mid = lambda v: (v[0] + v[1]) / 2
    ranks = {}
    for key, idx in (("pass", 3), ("cause", 4), ("esc", 5)):
        order = sorted(range(len(MODELS)), key=lambda i: -mid(MODELS[i][idx]))
        ranks[key] = {i: r for r, i in enumerate(order)}

    def chip(x, y, txt, r):
        if r < 3:
            bg, fg = CHIP[r]
            w = 9 * len(txt) + 22
            s.el.append(f'<rect x="{x - w / 2}" y="{y - 14}" width="{w}" height="28" rx="6" fill="{bg}"/>')
            s.text(x, y, txt, 14, bold=True, color=fg)
        else:
            s.text(x, y, txt, 14, color=INK)

    y0 = hy + 20 + rh / 2
    # pinned reference row
    s.text(100, y0, "参考", 12, color=INK3)
    dot(s, 160, y0 - 8, GRAY)
    s.text(174, y0 - 8, "规则引擎", 15, anchor="start", bold=True, color=INK2)
    s.text(150, y0 + 14, "写死的规则 · 老师与标准答案", 12, anchor="start", color=INK3)
    for x, v in ((720, "100.0"), (880, "100.0"), (1030, "100.0"), (1160, "0"), (1270, "5.2"), (1420, "—")):
        s.text(x, y0, v, 14, color=INK2)
    s.el.append(f'<line x1="80" y1="{y0 + rh / 2}" x2="{W - 80}" y2="{y0 + rh / 2}" stroke="{LINE}" stroke-width="1" stroke-dasharray="4,4"/>')
    for i, (name, t, train, p, c, e, calls, cost) in enumerate(MODELS):
        y = y0 + (i + 1) * rh
        r = ranks["pass"][i]
        s.text(100, y, str(r + 1), 15, bold=True, color=CHIP[r][1] if r < 3 else INK2)
        dot(s, 160, y - 8, TYPE[t][0])
        s.text(174, y - 8, name, 15, anchor="start", bold=True, color=INK)
        s.text(150, y + 14, train if t == "ref" else f"{TYPE[t][1]} · {train}", 12, anchor="start", color=INK3)
        chip(720, y, fmt(p), ranks["pass"][i]); chip(880, y, fmt(c), ranks["cause"][i]); chip(1030, y, fmt(e), ranks["esc"][i])
        s.text(1160, y, "0", 14, color=INK); s.text(1270, y, f"{calls}", 14, color=INK)
        s.text(1420, y, f"≈ ${cost:.2f}" if cost < 0.1 else f"${cost:.2f}", 14, bold=True, color="#3fbf8a" if cost < 0.1 else INK)
    fy = tab_y + rh * (len(MODELS) + 1) + 92
    x = 60
    for t in ("small", "closed", "open", "ref"):
        dot(s, x + 6, fy, TYPE[t][0]); s.text(x + 18, fy, TYPE[t][1], 12.5, anchor="start", color=INK2); x += 150
    s.text(x + 20, fy, "推理费用 / 轮 = 跑一遍 194 题。API 为用量账单估算；自部署按评测时 GPU 占用 × A40 每小时 0.59 美元折算；训练一次性另计（4B 约 0.8 美元）。",
           11.5, anchor="start", color=INK3)
    s.save("results-board.svg")


def pareto():
    pts = [("Qwen3-0.6B 微调", "small", 0.01, 75.3, 75.3, 16, 4), ("Qwen3-4B 微调（当前版本）", "small", 0.02, 93.8, 95.9, 16, 26),
           ("Qwen3-8B 微调", "small", 0.03, 96.4, 96.4, 16, -16), ("Qwen3-4B 不训练", "ref", 0.02, 3.6, 3.6, 16, 4),
           ("Sonnet 5.5 零样本", "closed", 2.55, 79.4, 79.4, -16, 20), ("Sonnet 5.5 少样本", "closed", 2.82, 81.4, 81.4, -16, -16),
           ("Qwen3.5-397B 零样本", "open", 5.21, 61.3, 61.3, -16, 4)]
    W, H = 1600, 860
    s = dark(W, H)
    header(s, W, "效率", "准确率 vs 推理费用", "横轴是跑一遍 194 题的推理费用（对数刻度），纵轴是归因通过率。绿线上的配置是帕累托最优：同样的钱拿不到更高的分。")
    card(s, 60, 140, W - 120, H - 200)
    x0, x1, y0, y1 = 170, W - 120, H - 140, 190
    lmin, lmax = math.log10(0.005), math.log10(10)
    X = lambda c: x0 + (x1 - x0) * (math.log10(c) - lmin) / (lmax - lmin)
    Y = lambda p: y0 - (y0 - y1) * p / 100
    s.el.append(f'<rect x="{x0}" y="{y1}" width="{X(0.1) - x0}" height="{Y(85) - y1}" fill="#1f3a2e" fill-opacity="0.55"/>')
    s.text(x0 + 14, y1 + 18, "EFFICIENT", 11.5, anchor="start", bold=True, color="#3fbf8a")
    for p in (0, 25, 50, 75, 100):
        s.el.append(f'<line x1="{x0}" y1="{Y(p)}" x2="{x1}" y2="{Y(p)}" stroke="{LINE}" stroke-width="1"/>')
        s.text(x0 - 14, Y(p), f"{p}%", 12, anchor="end", color=INK3)
    for c, lab in ((0.01, "$0.01"), (0.03, "$0.03"), (0.1, "$0.1"), (0.3, "$0.3"), (1, "$1"), (3, "$3"), (10, "$10")):
        s.el.append(f'<line x1="{X(c)}" y1="{y1}" x2="{X(c)}" y2="{y0}" stroke="{LINE}" stroke-width="1" stroke-dasharray="3,5"/>')
        s.text(X(c), y0 + 20, lab, 12, color=INK3)
    s.text((x0 + x1) / 2, y0 + 46, "跑一遍 194 题的推理费用（美元，对数刻度）", 12.5, color=INK2)
    s.el.append(f'<text x="{x0 - 72}" y="{(y0 + y1) / 2}" transform="rotate(-90 {x0 - 72} {(y0 + y1) / 2})" text-anchor="middle" font-size="12.5" fill="{INK2}">归因通过率</text>')
    front = [(0.01, 75.3), (0.02, 94.85), (0.03, 96.4)]
    s.el.append('<polyline points="' + " ".join(f"{X(c)},{Y(p)}" for c, p in front) + f'" fill="none" stroke="#3fbf8a" stroke-width="2.5"/>')
    s.el.append(f'<line x1="{X(0.03)}" y1="{Y(96.4)}" x2="{x1}" y2="{Y(96.4)}" stroke="#3fbf8a" stroke-width="1.5" stroke-dasharray="6,5"/>')
    for lab, t, c, lo, hi, dx, dy in pts:
        col = TYPE[t][0]; cx = X(c); cy = Y((lo + hi) / 2)
        if hi > lo:
            s.el.append(f'<line x1="{cx}" y1="{Y(lo)}" x2="{cx}" y2="{Y(hi)}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>')
        s.el.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{col}" stroke="{CARD}" stroke-width="3"/>')
        s.text(cx + dx, cy + dy, f"{lab}  {fmt((lo, hi))}%", 13, anchor="start" if dx > 0 else "end", color=INK)
    x = 80
    for t in ("small", "closed", "open", "ref"):
        dot(s, x + 6, H - 30, TYPE[t][0]); s.text(x + 18, H - 30, TYPE[t][1], 12.5, anchor="start", color=INK2); x += 150
    s.el.append(f'<line x1="{x + 10}" y1="{H - 30}" x2="{x + 40}" y2="{H - 30}" stroke="#3fbf8a" stroke-width="2.5"/>')
    s.text(x + 50, H - 30, "帕累托前沿", 12.5, anchor="start", color=INK2)
    s.text(W - 60, H - 30, "训练是一次性成本，不计入推理费用（4B 约 0.8 美元）", 11.5, anchor="end", color=INK3)
    s.save("results-pareto.svg")


FAIL = [  # model label, type, failures by first failed check: cause, escalation, not submitted, other
    ("Qwen3.5-397B 零样本", "open", 27, 29, 18, 1),
    ("Sonnet 5.5 零样本", "closed", 21, 19, 0, 0),
    ("Sonnet 5.5 少样本", "closed", 20, 16, 0, 0),
    ("Qwen3-4B 微调 种子 1", "small", 3, 1, 8, 0),
    ("Qwen3-4B 微调 种子 2", "small", 5, 1, 2, 0),
    ("Qwen3-8B 微调", "small", 6, 0, 1, 0),
]
FCOL = [("主因错", "#d55181"), ("升级判断错", "#c98500"), ("回合用完未提交", "#4b4e54"), ("编号缺漏", "#8a8d93")]


def failures():
    W, rh = 1600, 58
    H = 160 + rh * len(FAIL) + 150
    s = dark(W, H)
    header(s, W, "失分构成", "输在哪：194 道题里失败的题，按第一个没通过的检查分类",
           "合成数据。前沿模型几乎不在工具调用上失分（幻觉 0、格式错误极少），失分集中在主因和升级判断。")
    card(s, 60, 140, W - 120, rh * len(FAIL) + 60)
    x0, x1 = 420, W - 200
    scale = (x1 - x0) / 80
    for gx in (0, 20, 40, 60, 80):
        s.el.append(f'<line x1="{x0 + gx * scale}" y1="160" x2="{x0 + gx * scale}" y2="{160 + rh * len(FAIL) + 10}" stroke="{LINE}" stroke-width="1"/>')
        s.text(x0 + gx * scale, 180 + rh * len(FAIL) + 6, f"{gx} 题", 11.5, color=INK3)
    for i, (lab, t, *vals) in enumerate(FAIL):
        y = 180 + i * rh + rh / 2 - 6
        dot(s, 100, y, TYPE[t][0])
        s.text(114, y, lab, 14, anchor="start", bold=True, color=INK)
        x = x0
        for (name, col), v in zip(FCOL, vals):
            if v == 0:
                continue
            w = v * scale
            s.el.append(f'<rect x="{x}" y="{y - 15}" width="{w - 2}" height="30" rx="4" fill="{col}"/>')
            if w > 34:
                s.text(x + w / 2, y, str(v), 13, bold=True, color="#ffffff")
            x += w
        s.text(x + 14, y, f"共 {sum(vals)} 题", 13, anchor="start", color=INK2)
    ly = H - 70; x = 80
    for name, col in FCOL:
        s.el.append(f'<rect x="{x}" y="{ly - 8}" width="16" height="16" rx="3" fill="{col}"/>'); s.text(x + 24, ly, name, 12.5, anchor="start", color=INK2); x += 170
    s.text(80, ly + 34, "第 17 节复核了 40 道前沿模型的错题：约一半是模型真错，一半是平台口径造成的（累计口径没写进提示、公会题不需要日期、升级取决于规则引擎抽样）。",
           11.5, anchor="start", color=INK3)
    s.save("results-failures.svg")


CATS = [  # label, n, 4B s1, 4B s2, 8B, Sonnet fs, Sonnet zs, Qwen3.5, (4B untrained, its n)
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
    a, b = (35, 38, 43), (222, 224, 228)  # single-hue neutral, dark -> light (lighter = better on a dark surface)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * p) for i in range(3))


def heatmap():
    cw, rh, x0, top = 178, 46, 360, 200
    W, H = x0 + cw * len(HCOLS) + 80, top + len(CATS) * rh + 130
    s = dark(W, H)
    header(s, W, "按类别", "每类答对的题数", "合成数据。越亮越好。微调模型几乎全对；前沿模型的失分集中在依赖平台口径的几类。")
    card(s, 40, 140, W - 80, H - 170)
    for j, (lab, t, sub) in enumerate(HCOLS):
        cx = x0 + j * cw + cw / 2
        w = sum(12.5 if ord(ch) > 127 else 7.2 for ch in lab)
        dot(s, cx - w / 2 - 4, top - 30, TYPE[t][0])
        s.text(cx + 6, top - 30, lab, 12.5, bold=True, color=INK)
        if sub:
            s.text(cx, top - 10, sub, 11.5, color=INK3)
    for i, (lab, n, a1, a2, b8, sfs, szs, qw, base) in enumerate(CATS):
        y = top + i * rh
        s.text(x0 - 16, y + rh / 2, f"{lab}（{n}）", 13, anchor="end", bold=(i < 2), color=INK)
        cells = [((a1 + a2) / 2 / n, f"{a1} / {a2}"), (b8 / n, f"{b8}"), (sfs / n, f"{sfs}"), (szs / n, f"{szs}"), (qw / n, f"{qw}"),
                 (base[0] / base[1], f"{base[0]}" + ("" if base[1] == n else f" / {base[1]}"))]
        for j, (p, txt) in enumerate(cells):
            x = x0 + j * cw
            s.el.append(f'<rect x="{x + 3}" y="{y + 3}" width="{cw - 6}" height="{rh - 6}" fill="{ramp(p)}" rx="6"/>')
            s.text(x + cw / 2, y + rh / 2, txt, 14, bold=True, color="#111214" if p >= 0.55 else INK)
    yb = top + len(CATS) * rh + 30
    for k, p in enumerate((0, 0.25, 0.5, 0.75, 1)):
        s.el.append(f'<rect x="{x0 + k * 46}" y="{yb - 9}" width="44" height="18" fill="{ramp(p)}" rx="3"/>')
    s.text(x0 + 5 * 46 + 12, yb, "0% → 100% 答对", 12, anchor="start", color=INK2)
    s.text(x0, yb + 30, "格子里是答对题数，括号里是该类总题数；“误判率上升”问法属于模型漂移类，单独列出；不训练的 4B 来自较早的 harness 版本，个别类别题数不同，已在格内注明。",
           11.5, anchor="start", color=INK3)
    s.save("results-category-heatmap.svg")


if __name__ == "__main__":
    board(); pareto(); failures(); heatmap()
