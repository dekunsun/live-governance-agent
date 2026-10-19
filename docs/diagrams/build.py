#!/usr/bin/env python3
"""Draws the diagrams as SVG (docs/images/arch-layers.svg, response-flow.svg, assistant-flow.svg).
Pure Python, no dependencies; PNGs are rendered from the SVGs by docs/diagrams/render.js (Playwright)."""
import os
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(H), "images")
FONT = "'Noto Sans CJK SC','WenQuanYi Zen Hei','PingFang SC','Microsoft YaHei','Helvetica Neue',Arial,sans-serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []
        self.el.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')
        self.el.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                       '<path d="M0,0 L10,5 L0,10 z" fill="#2b2b2b"/></marker></defs>')

    def box(self, x, y, w, h, text, fill="#fff", stroke="#2b2b2b", dashed=False, size=15, bold=False, r=8, sw=1.6, color="#1a1a1a"):
        d = ' stroke-dasharray="6,4"' if dashed else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
        lines = text.split("\n")
        lh = size * 1.3; y0 = y + h / 2 - lh * (len(lines) - 1) / 2
        for i, ln in enumerate(lines):
            self.el.append(f'<text x="{x + w / 2}" y="{y0 + i * lh}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" '
                           f'font-size="{size}" font-weight="{"600" if bold else "400"}" fill="{color}">{ln}</text>')

    def group(self, x, y, w, h, title, fill="#f3f4f6", dashed=False, size=16):
        d = ' stroke-dasharray="7,5"' if dashed else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#6b7280" stroke-width="1.4"{d}/>')
        self.el.append(f'<text x="{x + w / 2}" y="{y + 20}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" font-size="{size}" font-weight="600" fill="#111">{title}</text>')

    def text(self, x, y, s, size=14, anchor="middle", bold=False, color="#1a1a1a"):
        self.el.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-family="{FONT}" font-size="{size}" '
                       f'font-weight="{"600" if bold else "400"}" fill="{color}">{s}</text>')

    def arrow(self, pts, dashed=False, label=None, lab_dx=0, lab_dy=-8):
        d = ' stroke-dasharray="6,4"' if dashed else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.el.append(f'<polyline points="{p}" fill="none" stroke="#2b2b2b" stroke-width="1.6" marker-end="url(#arr)"{d}/>')
        if label:
            mx, my = pts[len(pts) // 2] if len(pts) > 2 else ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
            self.el.append(f'<text x="{mx + lab_dx}" y="{my + lab_dy}" text-anchor="middle" font-family="{FONT}" font-size="12" fill="#374151">{label}</text>')

    def line(self, pts, dashed=False):
        d = ' stroke-dasharray="6,4"' if dashed else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.el.append(f'<polyline points="{p}" fill="none" stroke="#2b2b2b" stroke-width="1.6"{d}/>')

    def cylinder(self, x, y, w, h, text, size=13):
        ry = 9
        self.el.append(f'<path d="M{x},{y + ry} v{h - 2 * ry} a{w / 2},{ry} 0 0 0 {w},0 v-{h - 2 * ry}" fill="#fff" stroke="#2b2b2b" stroke-width="1.5"/>')
        self.el.append(f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="#fff" stroke="#2b2b2b" stroke-width="1.5"/>')
        for i, ln in enumerate(text.split("\n")):
            self.el.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 4 + i * 16}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" font-size="{size}" fill="#111">{ln}</text>')

    def person(self, x, y, label):
        self.el.append(f'<circle cx="{x}" cy="{y}" r="9" fill="none" stroke="#2b2b2b" stroke-width="2"/>')
        self.el.append(f'<path d="M{x},{y + 9} v22 M{x - 14},{y + 18} h28 M{x},{y + 31} l-11,18 M{x},{y + 31} l11,18" fill="none" stroke="#2b2b2b" stroke-width="2"/>')
        self.text(x, y + 62, label, 14, bold=True)

    def diamond(self, cx, cy, w, h, text, size=14):
        self.el.append(f'<polygon points="{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}" fill="#fff" stroke="#2b2b2b" stroke-width="1.6"/>')
        for i, ln in enumerate(text.split("\n")):
            self.text(cx, cy + (i - (len(text.split(chr(10))) - 1) / 2) * 16, ln, size)

    def open_g(self, transform):
        self.el.append(f'<g transform="{transform}">')

    def close_g(self):
        self.el.append('</g>')

    def save(self, name):
        s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">' + "".join(self.el) + "</svg>")
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, name), "w").write(s); print("wrote", os.path.join(OUT, name))


def row(svg, x, y, items, w, h, gap, **kw):
    for i, it in enumerate(items):
        dashed = it.endswith("*"); svg.box(x + i * (w + gap), y, w, h, it.rstrip("*"), dashed=dashed, **kw)


def arch():
    s = SVG(1600, 1180)
    s.person(800, 28, "策略 owner / 治理运营")
    s.box(1250, 22, 130, 34, "已完成", size=14); s.box(1400, 22, 150, 34, "规划中", dashed=True, size=14)
    # L1 session layer
    s.group(40, 110, 1230, 110, "治理助手（会话层）")
    row(s, 60, 150, ["入口与冷启动问题", "范围解析（市场/场景/日期）", "调用过程可见", "结论卡片", "修改 / 确认 / 推翻", "巡检推送*"], 190, 50, 12, size=14)
    # L2 three groups
    s.group(40, 245, 380, 230, "Agent 层")
    row(s, 60, 285, ["LLM 引擎（4B + LoRA，vLLM）"], 340, 48, 0, size=14)
    row(s, 60, 345, ["规则引擎（教师 / 参考 / 兜底）"], 340, 48, 0, size=14)
    row(s, 60, 405, ["意图路由（规则）", "LLM 意图分类*"], 164, 48, 12, size=13)
    s.group(440, 245, 400, 230, "Harness v2.1")
    row(s, 460, 285, ["参数规整", "重复错误阻断"], 180, 48, 20, size=14)
    row(s, 460, 345, ["提交门控：主因 / 流量 / 编号", "调用预算提醒"], 180, 48, 20, size=13)
    row(s, 460, 405, ["升级规则（写死，不交给模型）"], 380, 48, 0, size=14)
    s.group(860, 245, 410, 230, "训练与评测")
    row(s, 880, 285, ["合成世界生成", "出题 + 标准答案"], 185, 48, 20, size=14)
    row(s, 880, 345, ["评分器（8 项，= RL 奖励）", "教师强制诊断"], 185, 48, 20, size=13)
    row(s, 880, 405, ["SFT（LoRA）", "GRPO", "外部验证集*"], 118, 48, 15, size=13)
    # right column ops
    s.group(1290, 110, 270, 365, "运营与治理（规划）", dashed=True)
    row(s, 1310, 150, ["巡检规则配置化*"], 230, 46, 0, size=14)
    row(s, 1310, 208, ["训练样本池管理*"], 230, 46, 0, size=14)
    row(s, 1310, 266, ["发版回归看板*"], 230, 46, 0, size=14)
    row(s, 1310, 324, ["按数据等级的模型路由*"], 230, 46, 0, size=14)
    row(s, 1310, 382, ["多语言 / 多市场配置*"], 230, 46, 0, size=14)
    # L3 platform tools
    s.group(40, 500, 1520, 190, "平台工具层（10 个只读 + 1 个只出草稿；每个返回带证据编号）")
    row(s, 60, 540, ["query_metrics", "get_change_log", "score_drift", "sample_cases", "get_hit_logs", "get_strategy_config"], 240, 46, 10, size=14)
    row(s, 60, 596, ["get_guild_info", "get_guild_health", "reconstruct_case", "run_backtest", "draft_change（草稿）"], 290, 46, 12, size=14)
    s.box(60, 650, 1480, 30, "校验层：策略 / 变更 / 指标 / session 编号必须真实存在；不存在的编号被拦截并列出已见编号", fill="#fef3c7", stroke="#b45309", size=13, r=6)
    # L4 AI foundation
    s.group(40, 715, 1520, 110, "模型与推理基础")
    row(s, 60, 755, ["开源基座（Qwen3 0.6B / 4B / 8B）", "LoRA 适配器（r=16）", "vLLM（前缀缓存、LoRA 热换）", "确定性评分器", "GPU 一键作业（自删除）"], 290, 48, 12, size=13)
    # L5 data
    s.group(40, 850, 1000, 300, "数据层（合成世界 / 真实平台的同构接口）")
    row(s, 60, 890, ["指标（sessions / hits / enforcements…）", "命中日志", "申诉与推翻"], 310, 46, 15, size=13)
    row(s, 60, 946, ["抽检样本（QA）", "策略配置（29 条）", "变更日志（模型升级 / policy）"], 310, 46, 15, size=13)
    row(s, 60, 1002, ["公会与健康分（51 家）", "Policy 文本", "回放窗口 / 评分窗口"], 310, 46, 15, size=13)
    s.text(540, 1085, "6 市场 · 8 场景 · 28 天 · 约 10 万场直播 / 世界 · 32 个世界（16 训练 + 16 测试）", 13, color="#374151")
    s.group(1060, 850, 500, 300, "存储")
    s.cylinder(1085, 905, 130, 90, "合成世界\n（CSV / JSON）")
    s.cylinder(1245, 905, 130, 90, "轨迹与决策\n记录（jsonl）")
    s.cylinder(1405, 905, 130, 90, "模型与 LoRA\n（HF 格式）")
    s.cylinder(1165, 1030, 130, 90, "评测结果\n（按 commit）")
    s.cylinder(1325, 1030, 130, 90, "训练样本池*\n（经人确认）")
    s.save("arch-layers.svg")


def flow():
    s = SVG(1700, 890)
    s.text(850, 40, "治理策略助手 · 响应生成与训练回路", 22, bold=True)
    s.text(850, 72, "实线：已实现；虚线：条件分支或回流。所有策略、编号、数字均为合成。", 13, color="#374151")
    s.open_g("translate(0,-120)")
    # main spine
    s.box(30, 440, 120, 60, "用户输入", fill="#e5e7eb", r=30, size=15)
    s.box(190, 425, 170, 90, "范围解析\n市场 / 场景 / 日期\n缺失 → 先澄清", size=13)
    s.arrow([(150, 470), (190, 470)])
    s.diamond(470, 470, 150, 90, "路由\n（规则）")
    s.arrow([(360, 470), (395, 470)])
    # attribution loop group
    s.group(590, 250, 520, 330, "归因回路（最多 14 次调用）", fill="#f8fafc")
    s.box(615, 300, 200, 70, "模型\n想法 + 一个工具调用", size=13)
    s.box(880, 300, 205, 70, "Harness\n参数规整 · 重复错误阻断", size=13)
    s.box(880, 410, 205, 70, "平台工具（只读）\n校验编号，返回证据编号", size=13)
    s.box(615, 410, 200, 70, "观察拼回上下文\n（剩 2 次时提醒提交）", size=13)
    s.arrow([(815, 335), (880, 335)])
    s.arrow([(982, 370), (982, 410)])
    s.arrow([(880, 445), (815, 445)])
    s.arrow([(715, 410), (715, 370)])
    s.text(850, 545, "每轮一次；得到结论后改为调用 submit_card", 12, color="#374151")
    s.arrow([(545, 470), (590, 470)], label="归因", lab_dy=-10)
    # gate
    s.box(1150, 405, 190, 130, "提交门控\n主因在枚举内\n先查过流量\n编号在观察里出现过\n（每类最多拒收 2 次）", size=12)
    s.arrow([(1110, 470), (1150, 470)])
    s.arrow([(1150, 520), (1130, 520), (1130, 600), (715, 600), (715, 480)], dashed=True, label="拒收 → 带提示回到回路", lab_dy=14)
    # escalation
    s.box(1150, 235, 190, 85, "升级规则（护栏）\n资金侧 · 头部主播\n周环比 ×2 · 样本不足", size=12)
    s.arrow([(1245, 405), (1245, 320)])
    # card + human
    s.box(1380, 420, 150, 100, "结论卡片\n主因 · 编号 · 升级\n结论 · 建议", size=13)
    s.arrow([(1340, 470), (1380, 470)])
    s.arrow([(1340, 277), (1455, 277), (1455, 420)]); s.text(1470, 262, "需要人工复核（带完整轨迹）", 12, color="#374151")
    s.box(1570, 420, 110, 100, "人工决策\n确认\n修改后确认\n推翻（原因）", size=12)
    s.arrow([(1530, 470), (1570, 470)])
    # downstream storage
    s.cylinder(1570, 620, 110, 80, "决策记录\n+ 样本池")
    s.arrow([(1625, 520), (1625, 620)])
    # P2S branch
    s.arrow([(470, 515), (470, 700), (590, 700)]); s.text(482, 620, "策略调整", 12, anchor="start", color="#374151"); s.text(482, 636, "（policy 文本）", 12, anchor="start", color="#374151")
    s.box(590, 670, 190, 60, "get_strategy_config\n找目标策略", size=12)
    s.box(820, 670, 190, 60, "run_backtest\n一组分数线（回放窗口）", size=12)
    s.box(1050, 670, 200, 60, "最保守可行解\n覆盖率 / 精确率 / 人审量", size=12)
    s.box(1290, 670, 170, 60, "变更草稿\n（标注是否需财务审批）", size=12)
    s.arrow([(780, 700), (820, 700)]); s.arrow([(1010, 700), (1050, 700)]); s.arrow([(1250, 700), (1290, 700)])
    s.arrow([(1460, 700), (1515, 700), (1515, 660), (1570, 660)]); s.text(1488, 722, "提交审批（不直接生效）", 12, color="#374151")
    # offline training loop
    s.group(30, 790, 1650, 180, "离线：训练与评测（同一评分器既是考试也是奖励）", fill="#f9fafb")
    s.box(60, 835, 190, 60, "合成世界 + 出题\n标准答案", size=12)
    s.box(290, 835, 190, 60, "规则引擎解题\n→ 示范（想法由证据生成）", size=12)
    s.box(520, 835, 150, 60, "SFT（LoRA）", size=13)
    s.box(710, 835, 300, 60, "GRPO：vLLM 采样 → 评分器奖励 → 组内优势\n→ 一步策略梯度 → 热换 LoRA", size=12)
    s.box(1050, 835, 200, 60, "评测：194 + 150 题\n按问法 / 主因 / 市场拆开", size=12)
    s.box(1290, 835, 170, 60, "失败归类\nHarness 可修 vs 需训练", size=12)
    s.box(1500, 835, 160, 60, "模型发版\n（LoRA 热换）", size=12)
    for x1, x2 in ((250, 290), (480, 520), (670, 710), (1010, 1050), (1250, 1290), (1460, 1500)):
        s.arrow([(x1, 865), (x2, 865)])
    s.arrow([(1580, 835), (1580, 770), (565, 770), (565, 225), (715, 225), (715, 300)], dashed=True); s.text(1070, 758, "新 LoRA 进入归因回路", 12, color="#374151")
    s.arrow([(1625, 700), (1625, 790)], dashed=True)
    s.text(1612, 745, "经人确认的样本回流", 11, anchor="end", color="#374151")
    s.arrow([(1375, 895), (1375, 925), (1150, 925), (1150, 897)], dashed=True)
    s.text(1262, 942, "被推翻的题 → 新测试题", 12, color="#374151")
    s.close_g()
    s.save("response-flow.svg")


def product_flow():
    """Product-level flow for the PRD summary: ask -> read-only tools -> evidence card -> human decision -> samples; guardrails lane."""
    s = SVG(1560, 700)
    s.text(780, 34, "治理策略助手 · 产品流程", 22, bold=True)
    s.text(780, 60, "助手只读；人做决定；每一次确认都变成训练样本。所有编号、数字均为合成。", 13, color="#374151")
    # training loop (drawn first so boxes sit on top)
    s.arrow([(1245, 125), (1245, 92), (495, 92), (495, 120)], dashed=True)
    s.text(870, 80, "下一版模型：只用经人确认的样本做 SFT / RL", 12, color="#374151")
    # main spine
    s.person(70, 130, "策略 owner")
    s.box(130, 135, 200, 70, "一句话提问\n「美国昨天误杀申诉涨了？」", size=13)
    s.arrow([(330, 170), (370, 170)])
    s.box(370, 120, 250, 100, "会话式助手（只读）\n按 playbook 查证据\n最多 14 次工具调用", size=13, bold=False)
    s.arrow([(620, 170), (660, 170)])
    s.box(660, 120, 230, 100, "结论卡片\n主因 · 证据编号\n是否升级 · 建议动作", size=13)
    s.text(775, 240, "校验层：引用的编号必须真实存在", 12, color="#374151")
    s.arrow([(890, 170), (930, 170)])
    s.box(930, 120, 190, 100, "人工决策\n确认 / 修改后确认\n推翻（写原因）", size=13)
    s.text(1025, 240, "升级规则写死：小市场 · 头部主播 · 资金侧", 12, color="#374151")
    s.arrow([(1120, 170), (1160, 170)])
    s.cylinder(1160, 125, 170, 90, "训练样本池\n只收经人确认的")
    s.text(1245, 240, "决策记录可追溯", 12, color="#374151")
    # tools group
    s.group(370, 300, 520, 120, "平台只读工具（每个结果都带证据编号）", fill="#f8fafc")
    for i, t in enumerate(["查数\n指标 · 环比", "下钻\n策略 · 公会 · 市场", "查变更\n模型 · 策略 · policy", "看漂移\n分数分布 · 抽检"]):
        s.box(385 + i * 125, 340, 115, 60, t, size=12)
    s.arrow([(450, 220), (450, 300)]); s.text(425, 262, "调用", 12, color="#374151")
    s.arrow([(540, 300), (540, 220)]); s.text(590, 262, "证据 + 编号", 12, color="#374151")
    # guardrail lane
    s.group(40, 450, 1480, 220, "助手不直接改任何策略", fill="#fff7ed")
    s.box(70, 490, 190, 60, "策略修改请求\n（policy 更新 / 收紧分数线）", size=12)
    s.arrow([(260, 520), (300, 520)])
    s.box(300, 490, 230, 60, "变更草稿\n回放结果 + 建议分数线", size=12)
    s.arrow([(530, 520), (570, 520)])
    s.box(570, 490, 200, 60, "审批流（人）\n通过后才生效", size=12)
    s.box(70, 580, 190, 60, "资金类动作\n冻结收益 · 延迟结算", size=12)
    s.arrow([(260, 610), (300, 610)])
    s.box(300, 580, 230, 60, "只标注「需要财务审批」\n不生成执行指令", size=12)
    s.arrow([(530, 610), (570, 610)])
    s.box(570, 580, 200, 60, "财务审批（人）", size=12)
    s.box(840, 490, 650, 150, "三条硬规则\n1. 助手没有写工具：策略、处罚、资金一律不碰\n2. 草稿不等于生效：必须经审批流，审批人看到完整证据链\n3. 卡片里的每个编号都经过校验，编不出来的编号直接拒收", size=13, dashed=True)
    s.save("assistant-flow.svg")


if __name__ == "__main__":
    arch(); flow(); product_flow()
