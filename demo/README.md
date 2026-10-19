# 演示页与流程动画的源码

发布版（claude.ai artifact）由这里的文件生成。页面里所有回答都是原型在合成数据上真实运行的结果。

```bash
bash demo/regen.sh                      # 一键重建：dump → build → 嵌入模板（下面三步的合集）
python demo/dump.py demo/runs.json      # 用规则引擎跑 12 个示例问题，导出 trace（需先 python data/generate.py）
python demo/build.py                    # runs.json -> scenes.json（白话化的步骤与结论）；同时生成 world.json（策略格子、市场）
python - <<'EOF2'                       # 把数据嵌进模板
t = open("demo/template.html").read()
t = t.replace("__SCENES__", open("demo/scenes.json").read()).replace("__WORLD__", open("demo/world.json").read())
open("demo/governance-demo.html", "w").write(t)
EOF2
NODE_PATH=$(npm root -g) node demo/rec.js  # 逐帧录制流程动画 -> frames/；再用 ffmpeg 合成 docs/media/assistant-flow.mp4
NODE_PATH=$(npm root -g) node demo/stills.js  # 导出 docs/images/flow-*.png
```

- `template.html`：页面模板，含 SVG 流程动画（三轮对话 + 人修改 + 确认）、12 个对话回放、意见收集。
- `scenes.json` / `world.json`：当前嵌入的数据；`governance-demo.html` 是已发布的成品。
- 页面里的"白话结论"和"讲解"是为合作方改写的文案，系统原文可在页面上展开对照。
