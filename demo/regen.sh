#!/usr/bin/env bash
# Rebuild demo/governance-demo.html from the sources (rules-engine traces -> scenes/world -> template). Deterministic.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 demo/dump.py demo/runs.json >/dev/null
( cd demo && python3 build.py >/dev/null )
python3 - <<'PY'
t = open("demo/template.html").read()
t = t.replace("__SCENES__", open("demo/scenes.json").read()).replace("__WORLD__", open("demo/world.json").read())
open("demo/governance-demo.html", "w").write(t)
PY
echo "demo/governance-demo.html rebuilt"
