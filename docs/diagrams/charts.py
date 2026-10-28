#!/usr/bin/env python3
"""Result charts for the technical appendix (docs/images/results-steps.svg, results-by-class.svg).
Pure Python SVG, same style as build.py; PNGs come from docs/diagrams/render.js. Numbers are copied from
reports/posttrain_report.md (synthetic data). Palette validated: #c9603f (归因) / #3a6ea5 (策略调整)."""
from build import SVG, FONT

ACC, BLU, GRAY, GRID, INK, INK2 = "#c9603f", "#3a6ea5", "#9a978f", "#e5e7eb", "#1a1a1a", "#5e5b55"


def axis(s, x0, x1, y0, y1, ticks=(0, 25, 50, 75, 100)):
    for t in ticks:
        x = x0 + (x1 - x0) * t / 100
        s.el.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
        s.text(x, y1 + 16, f"{t}%", 12, color=INK2)


def range_mark(s, x0, x1, y, lo, hi, color, label):
    xa, xb = x0 + (x1 - x0) * lo / 100, x0 + (x1 - x0) * hi / 100
    if hi - lo > 0.5:
        s.el.append(f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
    for x in {xa, xb}:
        s.el.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" stroke="#fff" stroke-width="2"/>')
    s.text(max(xa, xb) + 12, y, label, 12.5, anchor="start", color=INK)


def steps():
    rows = [  # label, 归因 (lo, hi, text), 策略调整 (lo, hi, text)
        ("4B 不训练", (3.6, 3.6, "3.6%"), (16, 16, "16%")),
        ("微调，示范 374 条（3 种子）", (79, 85, "79%–85%"), (53, 63, "53%–63%")),
        ("补稀有类示范，831 条（2 种子）", (92, 94, "92%–94%"), (69, 69, "69%")),
        ("换 8B，同样数据", (96, 96, "96%"), (70, 70, "70%")),
        ("改策略调整示范的想法（3 次运行）", (88, 98, "88%–98%"), (89, 97, "89%–97%")),
        ("混合示范 2 epoch，当前版本（2 种子）", (94, 96, "94%–96%"), (91, 98, "91%–98%")),
    ]
    W, Hh = 1200, 120 + len(rows) * 66 + 84
    s = SVG(W, Hh)
    s.text(W / 2, 30, "每一步只改一件事：归因与策略调整的通过率", 18, bold=True)
    s.text(W / 2, 54, "合成数据；横线是不同随机种子或运行之间的范围，单点表示只有一次运行", 13, color=INK2)
    x0, x1, ytop = 360, 1040, 100
    axis(s, x0, x1, ytop - 10, ytop + len(rows) * 66 - 20)
    for i, (lab, a, p) in enumerate(rows):
        y = ytop + i * 66
        if i % 2 == 0:
            s.el.append(f'<rect x="30" y="{y - 8}" width="{W - 60}" height="58" fill="#f6f7f4" rx="6"/>')
        s.text(340, y + 20, lab, 14, anchor="end", bold=(i == len(rows) - 1))
        range_mark(s, x0, x1, y + 8, a[0], a[1], ACC, a[2])
        range_mark(s, x0, x1, y + 34, p[0], p[1], BLU, p[2])
    ly = ytop + len(rows) * 66 + 14
    s.el.append(f'<circle cx="{x0}" cy="{ly}" r="5" fill="{ACC}"/>'); s.text(x0 + 12, ly, "归因（194 题）", 13, anchor="start")
    s.el.append(f'<circle cx="{x0 + 170}" cy="{ly}" r="5" fill="{BLU}"/>'); s.text(x0 + 182, ly, "策略调整（150 题）", 13, anchor="start")
    s.text(x0, ly + 26, "强化学习（GRPO，3 种子）在微调之上：归因 +4 / +1 / 0 题，策略调整 +3 / +7 / +17 题，收益远小于改示范", 12.5, anchor="start", color=INK2)
    s.save("results-steps.svg")


def by_class():
    rows = [  # label, n, seed1, seed2, untrained
        ("“误判率上升”问法（属模型漂移类）", 16, 6, 9, 0),
        ("模型漂移、固定阈值失效", 32, 22, 24, 0),
        ("公会集中违规", 14, 13, 14, 0),
        ("新策略上线", 16, 15, 16, 0),
        ("流量驱动", 9, 9, 9, 5),
        ("没有异常", 48, 48, 48, 3),
        ("数据不足", 16, 16, 16, 0),
        ("需要澄清", 16, 16, 16, 0),
        ("policy 更新", 16, 16, 16, 0),
        ("召回下降（对抗绕过）", 11, 11, 11, 0),
        ("公会真实违规", 16, 16, 16, 0),
    ]
    W, Hh = 1200, 120 + len(rows) * 40 + 70
    s = SVG(W, Hh)
    s.text(W / 2, 30, "当前版本按类别：每类答对的题数（4B，混合示范 2 epoch，两个种子）", 18, bold=True)
    s.text(W / 2, 54, "合成数据；横线两端是两个种子，灰点是不训练的 4B；弱项集中在模型漂移类和它的“误判率上升”问法", 13, color=INK2)
    x0, x1, ytop = 360, 1000, 100
    axis(s, x0, x1, ytop - 10, ytop + len(rows) * 40 - 12)
    for i, (lab, n, a, b, base) in enumerate(rows):
        y = ytop + i * 40
        if i % 2 == 0:
            s.el.append(f'<rect x="30" y="{y - 6}" width="{W - 60}" height="34" fill="#f6f7f4" rx="6"/>')
        s.text(340, y + 11, f"{lab}（{n} 题）", 13.5, anchor="end", bold=(i < 2))
        xb = x0 + (x1 - x0) * base / n
        s.el.append(f'<circle cx="{xb}" cy="{y + 11}" r="4.5" fill="{GRAY}"/>')
        lo, hi = min(a, b) / n * 100, max(a, b) / n * 100
        label = f"{a}/{n}" if a == b else f"{min(a, b)}–{max(a, b)}/{n}"
        range_mark(s, x0, x1, y + 11, lo, hi, ACC, label)
    ly = ytop + len(rows) * 40 + 24
    s.el.append(f'<circle cx="{x0}" cy="{ly}" r="5" fill="{ACC}"/>'); s.text(x0 + 12, ly, "当前版本，两个种子的范围", 13, anchor="start")
    s.el.append(f'<circle cx="{x0 + 240}" cy="{ly}" r="4.5" fill="{GRAY}"/>'); s.text(x0 + 252, ly, "不训练的 4B", 13, anchor="start")
    s.save("results-by-class.svg")


def ablation():
    steps = ["示范 374 条（3 种子）", "+ 补稀有类，831 条（2 种子）", "+ 策略调整示范写出比较过程（2 种子）",
             "+ 训练加一种问法（1 次运行）", "+ 训练 2 轮，当前版本（2 种子）"]
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, validated (light, --ordinal)
    groups = [  # label, n, 针对它的那一步 (index or None), per-step seed counts
        ("召回下降（归因）", 11, 1, [[0, 0, 1], [11, 11], [11, 7], [11], [11, 11]]),
        ("收紧分数线（策略调整）", 54, 2, [[17, 4, 7], [21, 27], [53, 53], [51], [54, 54]]),
        ("新市场规则，测试专用问法（策略调整）", 16, 3, [[8, 1, 4], [8, 1], [2, 12], [15], [13, 3]]),
        ("模型漂移（归因）", 32, None, [[18, 12, 15], [21, 21], [29, 16], [18], [22, 24]]),
        ("没有异常（归因，对照）", 48, None, [[48, 46, 36], [48, 47], [48, 48], [47], [48, 48]]),
    ]
    bw, gap, ggap = 30, 2, 64
    gw = len(steps) * (bw + gap) - gap
    x0, ytop, ybot = 110, 170, 520
    W = x0 + len(groups) * gw + (len(groups) - 1) * ggap + 60
    s = SVG(W, 720)
    s.text(W / 2, 30, "消融：每一步改动，提升落在它针对的那类题上（Qwen3-4B 微调）", 18, bold=True)
    s.text(W / 2, 54, "合成数据；柱高是各种子的平均通过率，黑点是单个种子；同一模型、同一 harness、同一评分器、同一测试集", 13, color=INK2)
    lx = x0
    for i, lab in enumerate(steps):
        s.el.append(f'<rect x="{lx}" y="{84 + (i // 3) * 26}" width="14" height="14" rx="3" fill="{ramp[i]}"/>')
        s.text(lx + 22, 91 + (i // 3) * 26, lab, 13, anchor="start")
        lx = x0 if i == 2 else lx + 300
    for t in (0, 25, 50, 75, 100):
        y = ybot - (ybot - ytop) * t / 100
        s.el.append(f'<line x1="{x0 - 10}" y1="{y}" x2="{W - 40}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        s.text(x0 - 16, y, f"{t}%", 12, anchor="end", color=INK2)
    for g, (lab, n, tgt, vals) in enumerate(groups):
        gx = x0 + g * (gw + ggap)
        for i, seeds in enumerate(vals):
            bx = gx + i * (bw + gap)
            p = sum(seeds) / len(seeds) / n
            h = max((ybot - ytop) * p, 1)
            y = ybot - h
            r = min(4, h)
            s.el.append(f'<path d="M{bx},{ybot} V{y + r} Q{bx},{y} {bx + r},{y} H{bx + bw - r} Q{bx + bw},{y} {bx + bw},{y + r} V{ybot} Z" fill="{ramp[i]}"/>')
            for v in sorted(set(seeds)):
                s.el.append(f'<circle cx="{bx + bw / 2}" cy="{ybot - (ybot - ytop) * v / n}" r="3.5" fill="{INK}" stroke="#fff" stroke-width="1.5"/>')
            if i == tgt:
                s.text(bx + bw / 2, y - 16, "针对", 11.5, bold=True, color=INK)
        s.el.append(f'<line x1="{gx - 6}" y1="{ybot}" x2="{gx + gw + 6}" y2="{ybot}" stroke="{INK2}" stroke-width="1"/>')
        name, kind = lab.rstrip("）").split("（")
        s.text(gx + gw / 2, ybot + 22, name, 13.5, bold=True)
        s.text(gx + gw / 2, ybot + 42, f"{kind} · {n} 题", 12, color=INK2)
    notes = [
        "读法：补稀有类示范让召回下降从 0–1 题到全对；策略调整示范写出“先列约束、再比较候选线”，让收紧分数线从 4–27 题到 53–54 题；",
        "训练加一种问法，让测试专用问法从此前的 1–12 题到 15 题，但训练 2 轮后两个种子又分成 13 和 3 题，这一类还不稳。",
        "模型漂移类没有哪一步专门针对它，各次运行在 12–29 题之间，是现在最大的短板；没有异常类是对照，各步基本不变（374 条那组有一个种子例外）。",
        "另一组对照：去掉策略调整示范、只训归因，策略调整总分从 136–147 题掉到 43–52 题（共 150 题），归因不变（177 / 181 题）。",
    ]
    for k, t in enumerate(notes):
        s.text(x0, ybot + 80 + k * 22, t, 12.5, anchor="start", color=INK2)
    s.save("results-ablation.svg")


if __name__ == "__main__":
    steps(); by_class(); ablation()
