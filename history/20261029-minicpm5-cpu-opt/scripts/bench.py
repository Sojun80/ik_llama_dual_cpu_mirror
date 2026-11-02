"""Phase 1: llama-bench sweep over builds, thread counts, CCD placement, NUMA modes and THP.
Each run is confined by a cgroup cpuset (systemd-run scope) that excludes production's cores 0-15,
so even --numa distribute's own affinity calls cannot land on them.
Usage: bench.py <set-name> [config-name-filter]"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from topo import ccd_histogram, cpulist, packed, spread  # noqa: E402

E = "/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt"
MODEL = "/home/ubuntu/casebook-runtime/models/MiniCPM5-2B-Q4_K_M.gguf"
BUILDS = {
    # Same build (2e0d86b, -march=native) as ~/casebook-runtime/llama-cpu/bin, which has no llama-bench.
    "prod": "/home/ubuntu/llm-evaluations/20261028-tiny-logwatch/cpu-build/bin",
    "znver2-omp": E + "/build-znver2-omp/bin",
    "znver2-noomp": E + "/build-znver2-noomp/bin",
}
BENCH_ARGS = ["-p", "1024", "-n", "128", "-d", "0,2048", "-r", "3", "-lm", "none", "-o", "jsonl"]
N0_FREE = range(4, 12)  # node 0 CCX 4-11 = cores 16-47 (CCDs 2-5)


def cfg(name, build, cpus, mem, threads=None, numa=None, env=None, extra=None):
    return dict(name=name, build=build, cpus=cpus, mem=mem, threads=threads or len(cpus), numa=numa,
                env=env or {}, extra=extra or [])


SETS = {
    "placement": [
        cfg("n1-packed16 (prod layout)", "prod", packed(1, 16), "1"),
        cfg("n1-packed16 (prod layout)", "znver2-omp", packed(1, 16), "1"),
        cfg("n1-packed8 (1 CCD)", "znver2-omp", packed(1, 8), "1"),
        cfg("n1-spread12 (1/CCX)", "znver2-omp", spread(1, 12), "1"),
        cfg("n1-spread16 (2-3/CCD)", "znver2-omp", spread(1, 16), "1"),
        cfg("n1-spread16 (2-3/CCD)", "prod", spread(1, 16), "1"),
        cfg("n1-spread18 (3/CCD)", "znver2-omp", spread(1, 18), "1"),
        cfg("n1-spread24 (2/CCX)", "znver2-omp", spread(1, 24), "1"),
        cfg("n1-spread32", "znver2-omp", spread(1, 32), "1"),
        cfg("n1-all48", "znver2-omp", spread(1, 48), "1"),
        cfg("n1-all48+SMT (96t)", "znver2-omp", spread(1, 48, smt=True), "1"),
        cfg("n1-spread24+SMT (48t)", "znver2-omp", spread(1, 24, smt=True), "1"),
    ],
    "numa": [
        # 16 + 16 threads on both sockets, 4 per CCD on node 0 CCDs 2-5 and node 1 CCDs 0-3.
        cfg("x32 interleave", "znver2-omp", spread(0, 16, N0_FREE) + spread(1, 16, range(8)), "0,1"),
        cfg("x32 --numa distribute, interleave", "znver2-omp", spread(0, 16, N0_FREE) + spread(1, 16, range(8)),
            "0,1", numa="distribute"),
        cfg("x32 --numa distribute, first-touch", "znver2-omp", spread(0, 16, N0_FREE) + spread(1, 16, range(8)),
            None, numa="distribute"),
        cfg("x64 interleave (32+32)", "znver2-omp", spread(0, 32, N0_FREE) + spread(1, 32, range(8)), "0,1"),
        cfg("n1-spread16 --numa isolate", "znver2-omp", spread(1, 16), "1", numa="isolate"),
        cfg("n1-spread16 node-local CPUs, remote memory", "znver2-omp", spread(1, 16), "0"),
    ],
    "tuning": [
        # llama-bench defaults to flash attention off; llama-server defaults to auto.
        cfg("n1-packed16 (prod layout) fa=1", "prod", packed(1, 16), "1", extra=["-fa", "1"]),
        cfg("n1-spread16 fa=1", "znver2-omp", spread(1, 16), "1", extra=["-fa", "1"]),
        cfg("n1-spread24 fa=1", "znver2-omp", spread(1, 24), "1", extra=["-fa", "1"]),
        cfg("n1-spread16 fa=1 kv=q8_0", "znver2-omp", spread(1, 16), "1",
            extra=["-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0"]),
        cfg("n1-spread16 noomp", "znver2-noomp", spread(1, 16), "1"),
        cfg("n1-spread24 noomp", "znver2-noomp", spread(1, 24), "1"),
        cfg("n1-spread16 THP (glibc hugetlb=1)", "znver2-omp", spread(1, 16), "1",
            env={"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}),
        cfg("n1-spread24 THP (glibc hugetlb=1)", "znver2-omp", spread(1, 24), "1",
            env={"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}),
        cfg("n1-spread16 OMP pinned", "znver2-omp", spread(1, 16), "1",
            env={"OMP_PROC_BIND": "true", "OMP_PLACES": "{%s}" % "},{".join(map(str, spread(1, 16)))}),
        cfg("n1-spread16 poll=0", "znver2-omp", spread(1, 16), "1", extra=["--poll", "0"]),
    ],
}


THP = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}
TG_ONLY = ["-p", "0", "-n", "128", "-d", "0,2048", "-r", "5", "-lm", "none", "-o", "jsonl"]
# Decode-only repeats, interleaved twice, to separate the THP and placement effects from run-to-run noise.
SETS["confirm"] = [dict(c, bench_args=TG_ONLY) for _ in range(2) for c in [
    cfg("A n1-packed16 (prod layout)", "prod", packed(1, 16), "1"),
    cfg("B n1-packed16 THP", "prod", packed(1, 16), "1", env=THP),
    cfg("C n1-spread16", "znver2-omp", spread(1, 16), "1"),
    cfg("D n1-spread16 THP", "znver2-omp", spread(1, 16), "1", env=THP),
    cfg("E n1-spread16 THP", "prod", spread(1, 16), "1", env=THP),
    cfg("F n1-packed16 THP", "znver2-omp", packed(1, 16), "1", env=THP),
]]


def run(c, out):
    cl = cpulist(c["cpus"])
    nodes = c["mem"] if c["mem"] else "0,1"
    cmd = ["sudo", "-n", "systemd-run", "--scope", "-q", "--uid=ubuntu", "--gid=ubuntu",
           "-p", "AllowedCPUs=" + cl, "-p", "AllowedMemoryNodes=" + nodes, "--",
           "env", "LD_LIBRARY_PATH=" + BUILDS[c["build"]]] + ["%s=%s" % kv for kv in c["env"].items()]
    cmd += ["numactl", "--physcpubind=" + cl]
    if c["mem"]:
        cmd += ["--interleave=" + c["mem"]] if "," in c["mem"] else ["--membind=" + c["mem"]]
    cmd += [BUILDS[c["build"]] + "/llama-bench", "-m", MODEL, "-t", str(c["threads"])]
    cmd += c.get("bench_args", BENCH_ARGS) + c["extra"]
    if c["numa"]:
        cmd += ["--numa", c["numa"]]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    rec = dict(c, cpulist=cl, ccds=ccd_histogram(c["cpus"]), cmd=cmd, wall=time.time() - t0,
               rc=p.returncode, stderr_tail=p.stderr[-1500:], started=t0,
               results=[{k: r.get(k) for k in ("n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts", "n_threads",
                                                "build_commit")} for r in rows])
    del rec["cpus"]
    out.write(json.dumps(rec) + "\n")
    out.flush()
    summ = " ".join("%s%d@d%d=%.1f" % ("pp" if r["n_gen"] == 0 else "tg", r["n_prompt"] or r["n_gen"],
                                       r["n_depth"], r["avg_ts"]) for r in rows)
    print("%-45s %-13s t=%-3d rc=%d %s" % (c["name"], c["build"], c["threads"], p.returncode, summ), flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    with open("%s/raw/bench-%s.jsonl" % (E, name), "a") as out:
        for c in SETS[name]:
            if filt in c["name"]:
                run(c, out)
