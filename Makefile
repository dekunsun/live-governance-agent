# One-command regression check. Run after touching the harness, grader, task generation or demo build.
#   make check        grader rules, reference-solver replays, test-set checksums, demo page byte-identity
#   make worlds       (re)generate the synthetic worlds and task sets (slow, ~10 min)
PY ?= python3

check: test-grader check-testsets check-p2s check-demo
	@echo "all checks passed"

test-grader:
	$(PY) -m pytest -q tests/test_grader.py tests/test_llm_mock.py

check-testsets:   # the published numbers are only comparable while these files are byte-identical
	@test -f posttrain/runs/tasks_test.jsonl || { echo "run: make worlds"; exit 1; }
	md5sum -c eval/checksums.txt
	$(PY) eval/fingerprint.py --check   # task_id + query + expected only; survives teacher rewording

check-p2s:        # the Policy-to-Strategy reference solver must still solve every task it generated
	$(PY) posttrain/p2s.py check | tail -1

check-demo:       # demo/governance-demo.html must be reproducible from the sources
	bash demo/regen.sh >/dev/null && git diff --quiet -- demo/governance-demo.html demo/scenes.json demo/world.json

worlds:
	$(PY) data/generate.py >/dev/null && $(PY) posttrain/worlds.py --train 16 --test 6 --jobs 4 && $(PY) posttrain/worlds.py --extra-test 10 --jobs 4 \
	  && $(PY) posttrain/tasks.py && $(PY) posttrain/tasks_v2.py && $(PY) posttrain/p2s.py gen

.PHONY: check test-grader check-testsets check-p2s check-demo worlds
