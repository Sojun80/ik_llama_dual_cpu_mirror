"""Phase 2: real Casebook labeling prompts through a separate llama-server, production 'think' profile.
Usage: serve_bench.py <label> <build> <cpulist> <mem: 0|1|0,1> <t> <tb> [--numa MODE] [--env K=V ...] [--n N]
The server runs in a cgroup cpuset scope that excludes production cores 0-15, on port 19160."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from bench import BUILDS, E, MODEL  # noqa: E402

PORT = 19160
PROFILE = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0,
           "presence_penalty": 0.0, "max_tokens": 6144, "chat_template_kwargs": {"enable_thinking": True}}

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("build")
ap.add_argument("cpus")
ap.add_argument("mem")
ap.add_argument("t", type=int)
ap.add_argument("tb", type=int)
ap.add_argument("--numa")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--extra", default="")
a = ap.parse_args()

prompts = json.load(open(E + "/prompts/bench-set.json"))[: a.n]
bindir = BUILDS[a.build]
server = [bindir + "/llama-server", "-m", MODEL, "--load-mode", "none", "-ngl", "0", "-t", str(a.t),
          "-tb", str(a.tb), "-c", "16384", "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1"]
if a.numa:
    server += ["--numa", a.numa]
server += a.extra.split()
mem = ["--interleave=" + a.mem] if "," in a.mem else ["--membind=" + a.mem]
cmd = ["sudo", "-n", "systemd-run", "--scope", "-q", "--uid=ubuntu", "--gid=ubuntu",
       "-p", "AllowedCPUs=" + a.cpus, "-p", "AllowedMemoryNodes=" + ("0,1" if "," in a.mem else a.mem), "--",
       "env", "LD_LIBRARY_PATH=" + bindir] + a.env + ["numactl", "--physcpubind=" + a.cpus] + mem + server
slug = a.label.replace(" ", "_").replace("/", "-")
log = open("%s/raw/server-%s.log" % (E, slug), "w")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            if json.loads(urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2).read()).get(
                    "status") == "ok":
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise SystemExit("server did not come up")
    spid = subprocess.run(["pgrep", "-u", "ubuntu", "-f", "llama-server.*--port %d" % PORT],
                          capture_output=True, text=True).stdout.split()[0]
    mem = {l.split(":")[0]: l.split()[1] for l in open("/proc/%s/smaps_rollup" % spid) if l.startswith(("Rss", "AnonHuge"))}
    out = open("%s/raw/serve-%s.jsonl" % (E, slug), "w")
    calls = []
    for i, p in enumerate(prompts):
        body = dict(PROFILE, model="MiniCPM5-2B-Q4_K_M", messages=p["messages"], stream=False, seed=1000 + i)
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=1500).read())
        tm = d.get("timings", {})
        rec = {"i": i, "interpretation_id": p["interpretation_id"], "wall": time.time() - t0,
               "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage"), "timings": tm}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        calls.append(rec)
        print("  %-40s #%d prompt=%d(%.0f t/s) gen=%d %.2f t/s %.1fs" % (
            a.label, i, tm.get("prompt_n", 0), tm.get("prompt_per_second", 0), tm.get("predicted_n", 0),
            tm.get("predicted_per_second", 0), rec["wall"]), flush=True)
    gen = [c["timings"]["predicted_per_second"] for c in calls]
    ntok = sum(c["timings"]["predicted_n"] for c in calls)
    tgen = sum(c["timings"]["predicted_ms"] for c in calls) / 1000
    pp = [c["timings"]["prompt_per_second"] for c in calls if c["timings"].get("prompt_n", 0) >= 256]
    summary = {"label": a.label, "build": a.build, "cpus": a.cpus, "mem": a.mem, "t": a.t, "tb": a.tb,
               "numa": a.numa, "env": a.env, "n": len(calls), "decode_tok_s_median": statistics.median(gen),
               "decode_tok_s_pooled": ntok / tgen, "completion_tokens_total": ntok,
               "prompt_tok_s_median": statistics.median(pp) if pp else None,
               "wall_total_s": sum(c["wall"] for c in calls), "server_cmd": cmd, "server_mem_kb": mem,
               "extra": a.extra}
    with open(E + "/raw/serve-summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("SUMMARY %-34s decode median %.2f pooled %.2f tok/s, pp median %s, %d tokens" % (
        a.label, summary["decode_tok_s_median"], summary["decode_tok_s_pooled"],
        "%.0f" % summary["prompt_tok_s_median"] if pp else "-", ntok), flush=True)
finally:
    # The server can hang on SIGTERM with strict-pinned threadpools; escalate so the next run gets the port.
    subprocess.run(["pkill", "-f", "llama-server.*--port %d" % PORT])
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print("  (server ignored SIGTERM for 20 s; sending SIGKILL)", flush=True)
        subprocess.run(["pkill", "-9", "-f", "llama-server.*--port %d" % PORT])
        proc.wait(timeout=30)
