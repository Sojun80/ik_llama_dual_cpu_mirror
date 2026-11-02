#!/usr/bin/env python3
# record the casebook-off thread layouts and the speculative-decoding tests (2026-11-01) in PROGRESS.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
p = root / 'PROGRESS.md'
s = p.read_text()
assert 'Casebook stopped (2026-11-01)' not in s
s += '''
## Casebook stopped (2026-11-01, user-approved)
The user approved stopping casebook for benchmarking (`sudo systemctl stop casebook casebook-llm`; both units are
still enabled, so a reboot starts them again; restart with `sudo systemctl start casebook-llm casebook`). With it
stopped, tools/r5bench.py accepts cores 0-15 only under R5_CASEBOOK_OFF=1, which asserts casebook-llm is inactive and
widens the server's numactl mask to 0-95 (runs/68-INVALID-cpumask: before that fix, threads pinned to 0-15 piled onto
core 16). tools/r5bench.py also gained R5_ARGSFILE=<file> (extra server arguments, one per line).

Plain streaming (tools/bwprobe, casebook stopped): node 0 cores 16-47 99.6-102.5 GB/s (72-73 with casebook running),
node 0 all 48 110-114 GB/s, node 1 all 48 97.6-100.3 GB/s. So with casebook running, its memory traffic cost node 0
about 30% of its streaming bandwidth.

Thread layouts, mirror, casebook stopped (runs/69, ABBA, 4 rounds, decode tok/s; prompt tok/s in brackets):

| layout | 2.8K | 9.4K |
|---|---:|---:|
| m64: 32+32, node 0 16-47 (recommended so far) | 42.99 [532] | 38.04 [470] |
| m64s: 32+32 spread over all 6 CCDs per node | 41.92 [531] | 38.38 [468] |
| m96: 48+48, all cores | 39.94 [595] | 33.72 [533] |

m64 with casebook stopped vs running (runs/65 m64n): +3% at 2.8K, +6.5% at 9.4K. 48+48 decodes 7-11% slower but
processes prompts 12-13% faster (more threads help the compute-bound batch, while decode pays for bigger barriers).
Spreading the same 32 threads over more CCDs is neutral.

## Speculative decoding (2026-11-01)
Engine: build-mirror3 supports --spec-type draft, dflash, dspark, mtp and n-gram types. This model's GGUF has no
MTP/NextN head. z-lab's DFlash drafter targets base Qwen3.5-35B-A3B, not this fine-tune. OpenBMB ships a DSpark
draft for this exact model: models/dspark-MiniCPM-V-4_7-35B-A3B-F16.gguf (789442368 bytes, sha256 2ec640fb...e82,
matches Hugging Face's x-linked-etag; arch dflash, 5 layers, block 15, target layers 4/12/20/28/36, confidence head).

tools/spectest.py: three chat prompts (code, synthetic syslog -> JSON, prose), temperature 0, 512 tokens, mirror m64
(or "stock": node 1, 48 threads, no IK_R5 env). r5bench's random-word corpus is useless for speculation
(runs/70: 6% accepted, 10.3 tok/s).

| run | config | code | logs | prose | accepted (code/logs/prose) |
|---|---|---:|---:|---:|---|
| 71 | mirror, no draft | 45.88 | 44.01 | 45.92 | |
| 71 | mirror, dspark | 14.48 | 11.74 | 14.81 | 32% / 19% / 35% |
| 71 | mirror, dspark n_max=8 | 17.15 | 13.87 | 18.76 | 38% / 24% / 42% |
| 73 | stock, no draft | 32.95 | 32.11 | 33.06 | |
| 73 | stock, dspark n_max=8 | 18.33 | 13.31 | 18.83 | 42% / 24% / 40% |
| 74 | mirror, no draft | 46.37 | 44.01 | 46.14 | |
| 74 | mirror, ngram-mod n_max=16 | 45.72 | 51.36 | 45.55 | - / 52% / 21% |
| 74 | mirror, ngram-simple n_max=8 | 46.50 | 51.40 | 46.39 | - / 81% / 50% |

DSpark is 2.5-3.5x slower than plain decode: each draft call costs 44-65 ms (more than two target tokens; perf shows
about a third of the time in unresolved spin/sync code, then the draft's BF16 matmuls and F16 flash attention over
the whole context, which Zen 2 has no native BF16 for), and only 40% of depth-1 drafts are accepted, about 1.6 tokens
drafted per call. Stock and mirror accept the same, so the r5 fusions do not break the target features the draft
reads. Speculative outputs diverge from plain greedy output early (batched verification rounds differently).
N-gram drafting costs nothing when it has no match (code, prose unchanged) and gives +17% on the log/JSON prompt;
ngram-simple produced output identical to plain decode there. Single round each: treat as preliminary.
'''
p.write_text(s)
print('docs updated')
