#!/usr/bin/env bash
# Rebuild demo/governance-demo.html from the sources (rules-engine traces -> scenes/world -> template). Deterministic.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 demo/dump.py demo/runs.json >/dev/null
( cd demo && python3 build.py >/dev/null )
# runs.json carries run ids and wall-clock timings that differ on every dump; the page does not use them.
# Keep the tracked copy unchanged when only those fields moved.
python3 - <<'PY'
import json, subprocess
def strip(rows):
    for r in rows:
        r.pop("run_id", None); r.pop("latency_s", None)
        for e in r.get("events", []): e.pop("t", None)
    return rows
try:
    old = json.loads(subprocess.check_output(["git", "show", "HEAD:demo/runs.json"]))
    new = json.load(open("demo/runs.json"))
    if strip(json.loads(json.dumps(old))) == strip(json.loads(json.dumps(new))):
        subprocess.run(["git", "checkout", "--", "demo/runs.json"], check=True)
except Exception:
    pass
PY
python3 - <<'PY'
t = open("demo/template.html").read()
t = t.replace("__SCENES__", open("demo/scenes.json").read()).replace("__WORLD__", open("demo/world.json").read())
open("demo/governance-demo.html", "w").write(t)
PY
echo "demo/governance-demo.html rebuilt"
