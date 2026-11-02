"""Snapshot the live Casebook DB (sqlite backup API, read-only on the source) and rebuild
the exact chat messages Casebook sent for recent real 'think' jobs, using Casebook's own
prompt builders. Also writes the production per-call tok/s baseline."""

import json
import sqlite3
import statistics
import sys

sys.path.insert(0, "/home/ubuntu/casebook")
from casebook import interpret, rulebook  # noqa: E402

OUT = sys.argv[1]
src = sqlite3.connect("file:/home/ubuntu/casebook-data/casebook.db?mode=ro", uri=True)
snap = sqlite3.connect(OUT + "/casebook-snapshot.db")
src.backup(snap)
src.close()
snap.row_factory = sqlite3.Row
rb = rulebook.load("/home/ubuntu/casebook-data/rulebook.json")

rows = snap.execute("SELECT * FROM interpretations WHERE decode_profile='think' AND tok_per_s IS NOT NULL "
                    "ORDER BY id DESC").fetchall()
base = {
    "n": len(rows),
    "tok_per_s_median": statistics.median(r["tok_per_s"] for r in rows),
    "tok_per_s_mean": statistics.mean(r["tok_per_s"] for r in rows),
    "completion_tokens_median": statistics.median(r["completion_tokens"] for r in rows),
    "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in rows),
    "seconds_median": statistics.median(r["seconds"] for r in rows),
    "by_kind": {},
}
for k in ("template", "case"):
    ks = [r for r in rows if r["kind"] == k]
    if ks:
        base["by_kind"][k] = {"n": len(ks), "tok_per_s_median": statistics.median(r["tok_per_s"] for r in ks),
                              "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in ks)}
json.dump(base, open(OUT + "/production-baseline.json", "w"), indent=1)

prompts, seen = [], set()
for r in rows:
    key = (r["kind"], r["target_id"])
    if key in seen:
        continue
    seen.add(key)
    if r["kind"] == "template":
        t = snap.execute("SELECT * FROM templates WHERE id=?", (r["target_id"],)).fetchone()
        msgs, _ = interpret.build_label_prompt(snap, rb, t)
    else:
        c = snap.execute("SELECT * FROM cases WHERE id=?", (r["target_id"],)).fetchone()
        msgs, _ = interpret.build_case_prompt(snap, rb, c)
    prompts.append({"interpretation_id": r["id"], "kind": r["kind"], "target_id": r["target_id"],
                    "prod_prompt_tokens": r["prompt_tokens"], "prod_completion_tokens": r["completion_tokens"],
                    "prod_tok_per_s": r["tok_per_s"], "messages": msgs})
json.dump(prompts, open(OUT + "/prompts-all.json", "w"), indent=1)
print(json.dumps(base, indent=1))
print("captured", len(prompts), "distinct jobs; kinds:",
      {k: sum(p["kind"] == k for p in prompts) for k in ("template", "case")})
