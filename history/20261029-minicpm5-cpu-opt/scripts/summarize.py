"""Markdown tables from raw/bench-*.jsonl and raw/serve-summary.jsonl, plus production tok/s
before vs during this evaluation (impact check). Writes raw/tables.md."""

import json
import sqlite3
import statistics
import time

E = "/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt"
START = time.mktime(time.strptime("2026-10-29 21:14", "%Y-%m-%d %H:%M"))
out = []


def bench_table(name, title):
    out.append("### %s\n" % title)
    out.append("| Config | Build | Threads | CPUs | pp1024 @d0 | pp1024 @d2048 | tg128 @d0 | tg128 @d2048 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for l in open("%s/raw/bench-%s.jsonl" % (E, name)):
        r = json.loads(l)
        v = {("pp" if x["n_gen"] == 0 else "tg", x["n_depth"]): x["avg_ts"] for x in r["results"]}
        f = lambda k: "%.1f" % v[k] if k in v else "-"
        out.append("| %s | %s | %d | `%s` | %s | %s | %s | %s |" % (
            r["name"], r["build"], r["threads"], r["cpulist"], f(("pp", 0)), f(("pp", 2048)),
            f(("tg", 0)), f(("tg", 2048))))
    out.append("")


bench_table("placement", "Phase 1a: placement and thread count (node 1, 4 KB pages)")
bench_table("tuning", "Phase 1b: build, flash attention, THP, pinning")
bench_table("numa", "Phase 1c: cross-socket / NUMA modes")
bench_table("confirm", "Phase 1d: decode-only repeats, two interleaved cycles, 5 reps each")

out.append("### Phase 2: llama-server, 6 real Casebook prompts, production think profile\n")
out.append("| Run | Decode tok/s (median of calls) | Decode tok/s (pooled) | vs prod 24.0 | Prompt tok/s (median) "
           "| Completion tokens | THP RSS |")
out.append("|---|---|---|---|---|---|---|")
for l in open(E + "/raw/serve-summary.jsonl"):
    s = json.loads(l)
    out.append("| %s | %.1f | %.1f | %+.0f%% | %.0f | %d | %.0f%% |" % (
        s["label"], s["decode_tok_s_median"], s["decode_tok_s_pooled"], 100 * (s["decode_tok_s_pooled"] / 24.05 - 1),
        s["prompt_tok_s_median"], s["completion_tokens_total"],
        100 * int(s["server_mem_kb"]["AnonHugePages"]) / int(s["server_mem_kb"]["Rss"])))
out.append("")

c = sqlite3.connect("file:/home/ubuntu/casebook-data/casebook.db?mode=ro", uri=True)
rows = c.execute("SELECT created, tok_per_s, prompt_tokens, completion_tokens FROM interpretations "
                 "WHERE decode_profile='think' AND tok_per_s IS NOT NULL").fetchall()
before = [r for r in rows if r[0] < START]
during = [r for r in rows if r[0] >= START]
out.append("### Production during this evaluation\n")
for name, rs in (("before", before), ("during", during)):
    if rs:
        out.append("- %s (%d think calls): median %.1f tok/s, median completion %d tokens, median prompt %d" % (
            name, len(rs), statistics.median(r[1] for r in rs), statistics.median(r[3] for r in rs),
            statistics.median(r[2] for r in rs)))
open(E + "/raw/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
