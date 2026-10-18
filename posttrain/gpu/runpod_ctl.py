#!/usr/bin/env python3
"""Controller for GPU runs on Runpod (REST v1). Creates a fire-and-forget Pod running posttrain/gpu/pod_job.sh,
waits for it, downloads the result bundle, terminates the Pod, and keeps a rough cost tally.

Auth: `Authorization: Bearer $RUNPOD_API_KEY` when the variable is set; otherwise no header is sent, which is
right when the key is a network secret the environment's proxy injects. The key is never printed or written.

  python posttrain/gpu/runpod_ctl.py check                                   # auth + list pods
  python posttrain/gpu/runpod_ctl.py create --stage B0                       # baselines, 4B + 8B
  python posttrain/gpu/runpod_ctl.py create --stage B1 --models Qwen/Qwen3-4B --seeds "1 2 3"
  python posttrain/gpu/runpod_ctl.py wait POD_ID                             # poll, download, terminate
  python posttrain/gpu/runpod_ctl.py status POD_ID | results POD_ID | terminate POD_ID | list
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "posttrain", "runs", "gpu")
IMAGE = "vllm/vllm-openai:latest"
GPU_DEFAULT = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3"]
PRICE_GUESS = {"NVIDIA A40": 0.4, "NVIDIA RTX A6000": 0.5, "NVIDIA L40S": 0.8, "NVIDIA GeForce RTX 4090": 0.7, "NVIDIA H100 80GB HBM3": 2.7}


def _headers():
    h = {"Content-Type": "application/json", "User-Agent": "live-governance-agent/1.0 (+https://github.com/dekunsun/live-governance-agent)"}
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, headers=_headers(), data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        if e.code == 401:
            sys.exit("401 from Runpod: no credential reached the API. Either set RUNPOD_API_KEY in the environment (new session) "
                     "or add it as a network secret for rest.runpod.io (header Authorization: Bearer <key>).")
        raise SystemExit(f"{method} {path} -> {e.code}: {msg}")


def cmd_check(a):
    pods = api("GET", "/pods")
    mode = "env RUNPOD_API_KEY" if os.environ.get("RUNPOD_API_KEY") else "proxy-injected secret"
    print(f"auth ok ({mode}); pods: {len(pods) if isinstance(pods, list) else pods}")
    for p in pods if isinstance(pods, list) else []:
        print(f"  {p.get('id')} {p.get('name')} {p.get('desiredStatus')} {p.get('costPerHr')}/h {p.get('gpu', {}).get('id') if isinstance(p.get('gpu'), dict) else ''}")


def cmd_list(a):
    cmd_check(a)


def cmd_create(a):
    env = {"STAGE": a.stage, "MODELS": a.models, "SEEDS": a.seeds, "REPO_URL": a.repo, "REPO_REF": a.ref, "MAX_SECONDS": str(int(a.max_hours * 3600)), "GRACE_SECONDS": str(int(a.grace_min * 60))}
    if a.self_key_secret:
        env["RUNPOD_SELF_KEY"] = "{{ RUNPOD_SECRET_%s }}" % a.self_key_secret   # resolved by Runpod inside the Pod; never passes through here
    # The vllm image's ENTRYPOINT is the API server, so we override the entrypoint with a shell. The result port is
    # served from the very first second (STATUS=BOOT) so a Pod that never reaches the job can be told apart from
    # one that is still installing.
    start = ("mkdir -p /workspace/results && (python3 -m http.server 8000 --directory /workspace/results >/dev/null 2>&1 &) "
             "&& echo BOOT > /workspace/results/STATUS && exec > >(tee -a /workspace/results/boot.log) 2>&1; "
             "apt-get update -qq; apt-get install -y -qq git curl; "
             f"git clone -q --depth 1 -b {a.ref} {a.repo} /workspace/repo; bash /workspace/repo/posttrain/gpu/pod_job.sh; sleep infinity")
    body = {"name": a.name or f"lga-{a.stage.lower()}-{int(time.time()) % 100000}", "imageName": a.image, "cloudType": a.cloud,
            "gpuTypeIds": a.gpus.split(",") if a.gpus else GPU_DEFAULT, "gpuTypePriority": "custom", "gpuCount": 1,
            "containerDiskInGb": 80, "volumeInGb": 60, "volumeMountPath": "/workspace",
            "ports": ["8000/http", "8001/http"], "env": env, "interruptible": a.spot,
            "dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": [start]}
    est = max(PRICE_GUESS.get(g, 1.0) for g in body["gpuTypeIds"]) * a.max_hours
    print(f"creating {body['name']} stage={a.stage} models={a.models!r} gpus={body['gpuTypeIds']} cap={a.max_hours}h (worst-case ≈ ${est:.0f})")
    if a.dry_run:
        print(json.dumps(body, ensure_ascii=False, indent=1)); return
    pod = api("POST", "/pods", body)
    pid = pod.get("id")
    print(f"created pod {pid}  status={pod.get('desiredStatus')}  cost/h={pod.get('costPerHr')}  gpu={pod.get('machine', {}).get('gpuTypeId') if isinstance(pod.get('machine'), dict) else pod.get('gpu')}")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"pod": pod, "created_at": time.time(), "stage": a.stage, "models": a.models}, open(os.path.join(OUT, f"{pid}.json"), "w"), indent=1, default=str)
    print(f"next: python posttrain/gpu/runpod_ctl.py wait {pid}")


def pod_status_url(pid, port=8000):
    return f"https://{pid}-{port}.proxy.runpod.net"


def fetch(url, timeout=30, binary=False, quiet=True):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; live-governance-agent/1.0)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = r.read()
            return d if binary else d.decode(errors="replace")
    except Exception as e:
        if not quiet:
            print(f"fetch {url} failed: {e}")
        return None


def cmd_status(a):
    pod = api("GET", f"/pods/{a.pod}")
    st = fetch(pod_status_url(a.pod) + "/STATUS")
    print(f"pod {a.pod}: desired={pod.get('desiredStatus')} cost/h={pod.get('costPerHr')} uptime_s={pod.get('uptimeSeconds') or pod.get('uptimeInSeconds')} job={st.strip() if st else '(no status yet)'}")
    log = fetch(pod_status_url(a.pod) + "/LOG")
    if log:
        print("--- job log (tail) ---"); print("\n".join(log.strip().splitlines()[-15:]))
    return pod, (st or "").strip()


def cmd_results(a):
    """Download everything the Pod published. Returns True only when the per-file download got the key files.
    Tries the bundle first, then falls back to every file listed by the Pod's directory index."""
    dest = os.path.join(OUT, a.pod, "results")
    os.makedirs(dest, exist_ok=True)
    base = pod_status_url(a.pod)
    data = fetch(base + "/bundle.tgz", timeout=600, binary=True, quiet=False)
    if data:
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(os.path.join(OUT, a.pod), filter="data")
            print(f"bundle -> {dest} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"bundle unreadable: {e}")
    index = fetch(base + "/", timeout=60, quiet=False) or ""
    names = [n for n in re.findall(r'href="([^"]+)"', index) if not n.endswith("/") and n != "bundle.tgz"]
    got = 0
    for n in names:
        if os.path.exists(os.path.join(dest, n)) and os.path.getsize(os.path.join(dest, n)) > 0:
            got += 1
            continue
        d = fetch(base + "/" + n, timeout=600, binary=True)
        if d is not None:
            open(os.path.join(dest, n), "wb").write(d); got += 1
        else:
            print(f"  failed: {n}")
    files = sorted(os.listdir(dest)) if os.path.isdir(dest) else []
    ok = bool(files) and any(f.startswith("summary_") for f in files) and got == len(names) and len(names) > 0
    print(f"results: {got}/{len(names)} files listed by the Pod downloaded -> {dest}  complete={ok}")
    for fn in files:
        if fn.startswith("summary_") and fn.endswith(".json"):
            try:
                o = json.load(open(os.path.join(dest, fn)))["overall"]
                print(f"  {fn}: pass {o['pass_rate']:.3f} score {o['mean_score']:.3f} n={o['n']}")
            except Exception:
                pass
    return ok


def cmd_terminate(a):
    api("DELETE", f"/pods/{a.pod}")
    print(f"terminated {a.pod}")


def cmd_wait(a):
    t0 = time.time()
    while True:
        pod, st = cmd_status(a)
        if st in ("DONE", "EXITED", "TIMEOUT") or st.startswith("FAILED"):
            ok = cmd_results(a)
            log = fetch(pod_status_url(a.pod) + "/LOG")
            if log:
                open(os.path.join(OUT, f"{a.pod}.log"), "w").write(log)
            if ok and not a.keep:
                cmd_terminate(a)
            elif not ok:
                print(f"NOT terminating {a.pod}: results incomplete. Re-run `results {a.pod}`; terminate by hand when done.")
            hrs = (time.time() - t0) / 3600
            print(f"finished with {st}; waited {hrs:.2f} h; approx cost this wait ≈ ${hrs * float(pod.get('costPerHr') or 0):.2f}")
            return
        if (time.time() - t0) / 3600 > a.max_hours:
            print("wait cap reached"); ok = cmd_results(a)
            if ok:
                cmd_terminate(a)
            else:
                print(f"results incomplete; Pod {a.pod} left running for a manual `results` + `terminate`")
            return
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("list")
    c = sub.add_parser("create")
    c.add_argument("--stage", default="B0", choices=["B0", "B1"]); c.add_argument("--models", default="Qwen/Qwen3-4B Qwen/Qwen3-8B")
    c.add_argument("--seeds", default="1 2 3"); c.add_argument("--gpus", default=None, help="comma-separated gpuTypeIds in priority order")
    c.add_argument("--image", default=IMAGE); c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    c.add_argument("--spot", action="store_true"); c.add_argument("--max-hours", type=float, default=4.0)
    c.add_argument("--repo", default="https://github.com/dekunsun/live-governance-agent"); c.add_argument("--ref", default="main")
    c.add_argument("--name", default=None); c.add_argument("--dry-run", action="store_true")
    c.add_argument("--grace-min", type=float, default=45.0, help="minutes the Pod keeps serving results after DONE before self-terminating (needs --self-key-secret)")
    c.add_argument("--self-key-secret", default=os.environ.get("RUNPOD_SELF_KEY_SECRET"), help="name of a Runpod account secret holding an API key; lets the Pod delete itself after the grace period")
    for n in ("status", "results", "terminate", "wait"):
        s = sub.add_parser(n); s.add_argument("pod")
        if n == "wait":
            s.add_argument("--poll", type=int, default=120); s.add_argument("--max-hours", type=float, default=5.0); s.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "list": cmd_list, "create": cmd_create, "status": cmd_status, "results": cmd_results, "terminate": cmd_terminate, "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    main()
