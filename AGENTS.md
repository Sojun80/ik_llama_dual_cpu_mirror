# ik_llama_dual_cpu_mirror: notes for agents

Read this first. It explains what this fork is, where its history is recorded, and the rules for working on the machine
it was built for.

## What this is

A fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) at upstream commit `13cffc0`. It is tuned for fast
CPU-only decode of large MoE models on a **dual-socket AMD EPYC 7K62 (Zen 2)** box called **gigabyte5**. The test model
throughout was OpenBMB **MiniCPM-V 4.7 35B-A3B, Q4_K_M** (~21 GB).

The changes come from "round 5" of an optimization project (2026-10-31 .. 2026-11-02). Every change is gated by an
`IK_R5_*` environment variable (or `--numa mirror`), so one binary can A/B-test any feature and, with all of them off,
behaves like upstream. The central piece is a port of the NUMA **mirror** mode from
[mikechambers84/ik_llama.cpp](https://github.com/mikechambers84/ik_llama.cpp) (`numa-mirror` branch): each socket gets
its own copy of the weights and the KV cache, and 32 threads per socket work on their local copy.

## Results (512 generated tokens, temperature 0, identical prompt token IDs)

| | Decode 2.8K ctx | Decode 9.4K ctx | Prompt 2.8K | Prompt 9.4K |
|---|---:|---:|---:|---:|
| Starting point: stock ik_llama.cpp, one socket, 32 threads | 34.6 | 30.7 | 402 | 354 |
| This fork, recommended config, both sockets (runs/108) | **47.6** | **44.9** | ~535 | ~473 |

Every kept change was checked to be **bit-exact**: `completion_probabilities` identical to the previous config. The one
exception is the round-5 FA kernel (`IK_R5_FA`), which changes the reduction order (see PROGRESS.md). Speed claims come
from interleaved (ABBA) rounds at both 2.8K and 9.4K context. The machine drifts, so comparisons across sessions are
unreliable.

## Recommended command (experimental, never deployed)

```
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \
IK_R5_NONUMAB=1 IK_R5_PF=1 IK_R5_FA_SHADOW=1 \
numactl --physcpubind=16-95 build/bin/llama-server -m <model.gguf> -ngl 0 --no-mmap \
  --numa mirror --numa-mirror all -t 64 -tb 64 \
  --cpu-range $(cat history/20261031-minicpm-v47-round5/tools/m64pair.txt) -c 16384 --parallel 1
```

- `-t` must equal `-tb`.
- The cpu list puts node-0 cores first and node-1 cores second (block split), with threads 2h and 2h+1 on the same CCX.
- Cores 0-15 are left out on purpose; see the machine rules below.
- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF`, then
  `numactl --physcpubind=48-95 cmake --build build -j 48 --target llama-server`.

## Branches

| Branch | Content |
|---|---|
| `main` | `round5-mirror` plus this file and `history/` |
| `round5-mirror` | the source exactly as benchmarked (`2f69e5ee`, built on gigabyte5 as `build-mirror3`) |
| `round5` | the earlier single-socket work, before the mirror port |
| `upstream-13cffc0` | the unmodified ik_llama.cpp base |

`git log upstream-13cffc0..round5-mirror` lists the 25 `r5:` commits. Each commit message says what changed, its env
gate and its measured effect.

## history/: the whole project record

`history/<folder>/` is a copy of `gigabyte5:/home/ubuntu/llm-evaluations/<folder>/`. Builds, model files, source
checkouts, compiled binaries, perf.data files and anything over 5 MB were left out. Notes inside refer to absolute
gigabyte5 paths; map `/home/ubuntu/llm-evaluations/X` to `history/X`.

Read in this order:

1. `history/20261031-minicpm-v47-round5/MISSION.md`: the original brief and its constraints.
2. `history/20261031-minicpm-v47-round5/HANDOFF.md`: hardware facts measured before round 5 (CCX map, bandwidth).
3. `history/20261031-minicpm-v47-round5/PROGRESS.md`: **the main log.** Every commit, env flag, result, rejected idea
   and run number, in order.
4. `history/20261031-minicpm-v47-round5/FA-BOUNDARY.md`: the decode flash-attention investigation that ended with
   `IK_R5_FA_SHADOW`.
5. `history/20261031-minicpm-v47-round5/RESEARCH.md`: external leads reviewed.
6. Earlier rounds, for background:
   - `20261031-minicpm-v47-kernel-investigation/FINAL-REPORT.md`: ik vs upstream, rejected NUMA/expert splits.
   - `20261031-minicpm-v47-cpu-optimization/REPORT.md`: thread sweeps, counters, the context effect.
   - `20261031-minicpm-v47-round4/REPORT.md`: the quality evaluation of the model.
   - `20261029-minicpm5-cpu-opt/REPORT.md`: CPU tuning for the small MiniCPM5 model the casebook service runs.

Inside the round-5 folder:

- `runs/NN-*`: raw data per experiment (server logs, per-run JSON with timings and content). PROGRESS.md cites
  results as `runs/NN`.
- `tools/apply_r5*.py`: each source change was written as a patch script, applied to the source, then committed.
  - `*.NOT-KEPT.py`: tried and dropped.
  - `*.DIAG.py`: diagnostics only.
- `patches/`: `git format-patch` of the 25 commits.
- `tools/r5bench.py`: benchmark harness (interleaved configs, one model instance at a time, `--probs` for correctness).
- `tools/seqtest.py`: multi-request KV-reuse correctness test.
- `tools/profcmp.py` and `tools/r5analyze.py`: profile comparisons.
- `tools/m64pair.txt` and the other `m*.txt` files: cpu lists.
- `tools/doc_*.py`: the scripts that appended each section to PROGRESS.md.

## What made the difference (short version)

- NUMA mirror with node-local expert chunking (`IK_R5_NODECHUNK`) and K/V-only mirroring (`IK_R5_DNBIND`).
- Fused MoE tail (`IK_R5_FMOE`, `IK_R5_FMOE_UG`): 9 graph nodes become 1, with fewer barriers. Bit-exact.
- Prefetching each thread's next weights while it spins in a barrier (`IK_R5_PF`, 2 lines per spin).
- CCX-paired cpu order (`m64pair.txt`).
- `IK_R5_FA_SHADOW`: decode attention reads each thread's K/V rows from a private per-thread copy, kept current through
  a log of KV writes. Shared KV read only ~20-25 GB/s per socket however many threads read it; the private copies read
  2-2.4x faster. +10% decode at 9.4K. Why the shared reads are slow is **not proven**; the best guess is DRAM bank/row
  conflicts.

## Open items

- The hardware cause of the slow shared-KV reads (above).
- A Q8_0 KV cache: half the KV bytes, but not bit-exact, and the r5 FA kernel only handles F16, so it needs kernel work.
- Correctness was checked on synthetic corpora and the KV-reuse test only. The full log-analysis eval was not rerun.
- Nothing here is deployed.

## Machine rules (gigabyte5); binding unless the user says otherwise

- **casebook** runs on gigabyte5 (`casebook-llm` and `casebook` systemd services, `llama-server` pinned to CPUs 0-15 on
  node 0, port 19150). Never use CPUs 0-15 while it runs. Stop it only with the user's approval
  (`sudo systemctl stop casebook-llm casebook`) and restart it afterwards (`sudo systemctl start casebook-llm casebook`).
  `r5bench.py` allows cores 0-15 only with `R5_CASEBOOK_OFF=1` while the service is inactive.
- SRBMiner mines on the GPUs: never run GPU work and never disturb it.
- Run only one model instance at a time (memory).
- Leave system settings as found: `schedutil` governor, `numa_balancing` on, `amd_uncore` unloaded.
- Build on node-1 cores only (`numactl --physcpubind=48-95`).
- Do not deploy anything without the user's approval.
- Never modify the model file or earlier result folders. Work in a new dated folder under `~/llm-evaluations/`.
- In ssh commands, never use `pgrep -f`/`pkill -f` with a pattern that also appears in your own command line (it
  matches your own shell). Use `pgrep -x llama-server`.

## Updating this repo

The Gitea remote is `ssh://git@wilkinsnas.local:30009/sojun/ik_llama_dual_cpu_mirror.git`. gigabyte5 has no Gitea key,
so pushes go through the controller machine, which has one. Keep `round5-mirror` (or a new round branch) as the exact
benchmarked source. Add new notes under `history/`, and update the tables above when a result changes.
