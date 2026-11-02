# MiniCPM5-2B-Q4_K_M CPU decode optimization on gigabyte5

2026-10-29/30. Host: gigabyte5, 2x EPYC 7K62 (Zen 2; 6 CCDs x 2 CCX x 4 cores per socket), NPS1 (2 NUMA nodes), 16 x 4 GB DDR4-2666 per socket, kernel 7.0.0-38, THP `enabled=madvise`, `defrag=madvise`, governor schedutil.
Production (`casebook-llm.service`, node 0 cores 0-15) was never stopped, restarted or reconfigured. All tests ran in a cgroup cpuset (`systemd-run --scope -p AllowedCPUs=...`) that excludes cores 0-15, on port 19160. SRBMiner runs with `--disable-cpu` (GPU only, ~1% CPU) and was left running. Production's own think calls had a median of 24.1 tok/s before and 24.3 tok/s during the evaluation, so it was not disturbed.

## Result

| | Decode tok/s (real jobs) | Prompt tok/s | vs production 24.0 |
|---|---|---|---|
| Production now (node 0, cores 0-15 packed, 4 KB pages) | 24.0 (median of 44 prod calls) | ~250 | - |
| Same layout reproduced on node 1, 4 KB pages, today's fragmented memory (S0) | 7.7 | 239 | **-68%** |
| Same layout + THP (S1) | 22.4 | 253 | -7% |
| **Recommended: THP + 16 decode threads strict-pinned across all 6 CCDs + 48 batch threads (S4/S6/S9)** | **29.5-30.7** (median-of-calls 31-32) | **421-429** | **+23% to +28%** |
| Same, 24 decode threads (S10) | 32.7 (shorter completions; per-call equal to 16 threads at matched length) | 438 | +36% (not like-for-like) |

S0, S1, S2, S4, S6 and S9 produced token-identical outputs (same binary, same thread count, same seed), so their decode numbers compare the exact same 11,536 generated tokens at the same context depths.

## Recommendation

Use three changes together. They need no rebuild; the existing production binary is used.

1. **Transparent huge pages for the weights**, per process: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`. glibc then calls `madvise(MADV_HUGEPAGE)` on its mmap'd chunks, and the `--load-mode none` weight buffer ends up 99% in 2 MB pages (AnonHugePages 2.27 GB of 2.30 GB RSS in every THP run). No system setting changes.
2. **Spread the 16 decode threads over all 6 CCDs of the node** (3,3,3,3,2,2 per CCD, never two threads on one core) and pin them strictly (`-C <mask> --cpu-strict 1`). Production currently packs 16 threads onto 2 CCDs.
3. **Separate prompt-processing pool on all 48 cores of the node** (`-tb 48 -Cb <mask> --cpu-strict-batch 1`). Prompt processing goes from ~250 to ~425 tok/s, which saves ~1.5 s per job.

### Exact service command line

Recommended: node 1, which is exactly what was measured and has ~37 GB free versus ~4 GB on node 0:

```ini
[Service]
Environment=LD_LIBRARY_PATH=/home/ubuntu/casebook-runtime/llama-cpu/bin
Environment=GLIBC_TUNABLES=glibc.malloc.hugetlb=1
ExecStart=/usr/bin/numactl --physcpubind=48-95 --membind=1 /home/ubuntu/casebook-runtime/llama-cpu/bin/llama-server -m /home/ubuntu/casebook-runtime/models/MiniCPM5-2B-Q4_K_M.gguf --load-mode none -ngl 0 -t 16 -C 0x111113131313000000000000 --cpu-strict 1 -tb 48 -Cb 0xffffffffffff000000000000 --cpu-strict-batch 1 -c 16384 --host 127.0.0.1 --port 19150 --parallel 1
TimeoutStopSec=20
```

Equivalent on node 0 (same layout mirrored; decode cores 0-1,4,8-9,12,16-17,20,24-25,28,32,36,40,44):

```ini
Environment=GLIBC_TUNABLES=glibc.malloc.hugetlb=1
ExecStart=/usr/bin/numactl --physcpubind=0-47 --membind=0 /home/ubuntu/casebook-runtime/llama-cpu/bin/llama-server -m /home/ubuntu/casebook-runtime/models/MiniCPM5-2B-Q4_K_M.gguf --load-mode none -ngl 0 -t 16 -C 0x111113131313 --cpu-strict 1 -tb 48 -Cb 0xffffffffffff --cpu-strict-batch 1 -c 16384 --host 127.0.0.1 --port 19150 --parallel 1
TimeoutStopSec=20
```

`TimeoutStopSec=20`: with two strict-pinned threadpools, llama-server ignored SIGTERM in every run (S4-S10, 209 threads) and had to be SIGKILLed. Without this line, `systemctl stop/restart` waits the default 90 s before killing it.

After switching, check that the huge pages took: `grep AnonHugePages /proc/$(pgrep -f 'port 19150')/smaps_rollup` should show about 2.2 GB. Casebook's `interpretations.tok_per_s` should move from ~24 to ~30.

## Findings

### 1. Production is page-walk bound, and its 24 tok/s is luck of memory layout (most important)

With 4 KB pages the production layout ran at 29 tok/s in the 21:15 llama-bench run, then at a reproducible **8.1-8.4 tok/s** (±0.02) from 21:50 onward. Same binary, cores and memory node; only node 1's physical memory fragmentation changed. perf (root, same 16 cores, tg128 x3):

| | tok/s | cycles | instructions | dTLB L1 misses | data page walks | L2-TLB 4K misses |
|---|---|---|---|---|---|---|
| 4 KB pages | 8.13 | 2,479 G | 420 G | 216 M | **142 M** | 133 M |
| THP | 27.27 | 740 G | 391 G | 29 M | **0.7 M** | 0.6 M |

The weight stream (1.45 GiB per token) far exceeds the L2 TLB's reach with 4 KB pages, so every 4 KB of weights costs a page walk. When the backing memory happens to be physically contiguous, Zen 2's TLB page coalescing hides much of that. That explains why production, started at 20:05, still gets 24 tok/s. When it isn't contiguous, decode drops 3x. **If the production service restarts on today's fragmented node 0 (3.7 GB free, few high-order blocks in `/proc/buddyinfo`), it can be expected to fall to ~8 tok/s.** THP removes this dependence entirely, and node 1 with THP reached 22-23 tok/s even in the packed layout.

### 2. CCD placement: spread the threads, pin them

Each Zen 2 CCD has its own Infinity Fabric link to the I/O die (~33 GB/s read, from LLMExpertOptimization EXP-04 finding 5 on a 7K62). With THP, in llama-bench decode-only repeats (two interleaved cycles, 5 reps):

- packed 16 (2 CCDs): 30-32 tok/s at d0, 23-26 at d2048
- spread 16 (6 CCDs): 48-50 at d0, 29-34 at d2048

Unpinned spread threads were inconsistent on the real server (S2: one call fell to 15.9 tok/s), because the scheduler occasionally stacks two spinning threads on one core. `--cpu-strict 1` fixed it: S4/S6/S9 calls are all 28-34 tok/s. Decode threads: 12 (1 per CCX) is worse (26.4), and 16 and 24 are equal at matched length. 32 and 48 were no better in llama-bench, and SMT siblings hurt (96t: 27.3 vs 30.6 at d2048 without THP). 16 is the smallest count at the plateau.

### 3. Splitting across both sockets does not help decode

Tested in llama-bench with 4 KB pages, threads on node 0 CCDs 2-5 plus node 1 CCDs 0-3:

| | tg128 @d0 | tg128 @d2048 | pp1024 @d0 |
|---|---|---|---|
| node 1 alone, spread 16 | 42.1 | 30.6 | 269 |
| node 1 alone, all 48 | 36.8 | 30.6 | 490 |
| 16+16, memory interleaved | 37.2 | 29.0 | 434 |
| 16+16, `--numa distribute`, interleaved | 34.5 | 26.4 | 444 |
| 16+16, `--numa distribute`, first-touch | 26.7 | 21.7 | 440 |
| 32+32, interleaved | 26.3 | 22.7 | 591 |
| node 1 CPUs, node 0 memory | 17.4 | 17.9 | 249 |

- **Why it doesn't help.** Decode on one node is not limited by that node's memory controllers. 30 tok/s x 1.45 GiB is about 47 GB/s, against ~170 GB/s theoretical per socket. The limits are page walks, per-CCD fabric links and per-op barriers. A second socket adds cross-socket barrier latency and remote reads over xGMI.
- **Mirrored per-node weights** (`GGML_NUMA_STRATEGY_MIRROR`) exists in upstream llama.cpp only as an enum value. Nothing implements it (grep of 3c809f4), and `--numa` does not accept it. Even with perfect mirroring, the barrier cost remains, and the interleaved runs already show that adding the second socket's cores does not raise decode.
- **Layer split over RPC** (one rpc-server per node) was not built. For a single stream the halves run one after the other, so it can only add hop latency to what a single node already does.
- Cross-socket only pays for prompt processing (591 tok/s at 64 threads), which is ~2-4% of a Casebook job's time. The 48-core single-node batch pool already gets ~425.

### 4. Builds, flash attention, KV type, OpenMP, polling

- **Fresh upstream 3c809f4 built with `-march=znver2 -mtune=znver2`** (GGML_NATIVE=OFF, AVX2/FMA/F16C/BMI2, no AVX-512, gcc 15.2) performs the same as the production build (2e0d86b, `-march=native`). S7 30.5 vs S4 30.7 pooled. `-march=native` on this host already is znver2. No reason to switch binaries.
- **Flash attention**: llama-server's default `-fa auto` resolves to **on** for this model on CPU. S6 (`-fa on`) is token-identical to S4. `-fa off` (S5) is the same speed at matched completion length. Leave the default.
- **KV q8_0** (with FA): much worse in llama-bench (pp1024 @d2048 69 vs 213, tg 19.7 vs 24.8). Don't.
- **OpenMP vs ggml threadpool** (GGML_OPENMP=OFF): no gain (tg128 @d2048 24.0 vs ~24-30 OpenMP, within noise). OpenMP builds do honour `-C/--cpu-strict` (ggml-cpu.c applies per-worker affinity inside the OpenMP region).
- **`--poll 0`**: worse.
- **`OMP_PROC_BIND`/`OMP_PLACES`** pinning: no gain over unpinned in llama-bench. On the server, `--cpu-strict` is the way to pin.

### 5. Tricks from thed1 and LLMExpertOptimization

- The applicable one is EXP-04 finding 5 (spread threads across CCDs on Rome), confirmed and used above.
- EXP-04's fused BF16 MoE kernel does not apply to this dense Q4_K_M model.
- Its software-prefetch result (no gain on Zen 2) matches here: page walks, not prefetch distance, were the problem.
- thed1 (single-socket EPYC 7532) has `llama.cpp-fable-prefetch` and `llama.cpp-fastmtp` worktrees. The first is a detached worktree whose git metadata is missing on thed1, so its diff could not be inspected. The second is upstream-like. Neither offers a NUMA or CPU decode trick beyond the above. Nothing on thed1 was changed.

## Caveats

- **Node 1 stands in for node 0.** All results are from node 1 (symmetric to node 0). Node 0 also hosts `casebook run`, gpuwatch and sshd. With strict pinning, a decode thread sharing a core with them can stall a barrier briefly. That's why node 1 is the primary recommendation.
- **Bench numbers are noisy; trust the server runs.** llama-bench 4 KB numbers drifted with memory fragmentation over the session (see finding 1), so 4 KB-page llama-bench rows taken at different times are not comparable with each other. The THP rows, and the token-identical server runs, are the reliable comparisons.
- **Sampling makes completion lengths vary.** The runs use temperature 1.0, top-p 0.95, min-p 0, thinking on, seeds 1000-1005. Runs from different binaries or thread counts produce different completions and lengths, and decode tok/s falls with context depth (e.g. 5k-token completions in S5 ran at ~28 tok/s). Pooled tok/s is total generated tokens divided by total decode time.
- **The CPU governor stays schedutil.** It was not changed. `performance` on the service's cores might reduce variance further, but that is a system setting for the user to decide.
- **Prompt cache uses node 0 memory.** Production's llama-server holds 10.7 GB RSS, mostly the default 8 GiB `--cache-ram` prompt cache (full, evicting 90-140 MiB entries). That sits on node 0's memory and contributes to its fragmentation. `--cache-ram 1024` would likely lose little, since only the system prompt prefix is shared between jobs, but that was not measured.

## Reproduce

Everything is under `~/llm-evaluations/20261029-minicpm5-cpu-opt/`:

- `scripts/capture_prompts.py`: DB snapshot (sqlite backup API, read-only) and Casebook's own prompt builders. Writes `prompts/prompts-all.json` (43 real label jobs) and `prompts/production-baseline.json`.
- `prompts/bench-set.json`: the 6 prompts used, spanning 497-1511 production prompt tokens.
- `scripts/build.sh`: Zen 2 builds `build-znver2-omp`, `build-znver2-noomp`. Logs are in `build-logs/`.
- `scripts/topo.py`, `scripts/bench.py`: llama-bench sweeps (`placement`, `tuning`, `numa`, `confirm`). Output in `raw/bench-*.jsonl`.
- `scripts/serve_bench.py`, `scripts/phase2*.sh`: llama-server runs. Per-call results are in `raw/serve-*.jsonl`, server logs in `raw/server-*.log`, summaries in `raw/serve-summary.jsonl`.
- `scripts/summarize.py`: produces `raw/tables.md` (the full tables below).

## Full tables
### Phase 1a: placement and thread count (node 1, 4 KB pages)

| Config | Build | Threads | CPUs | pp1024 @d0 | pp1024 @d2048 | tg128 @d0 | tg128 @d2048 |
|---|---|---|---|---|---|---|---|
| n1-packed16 (prod layout) | prod | 16 | `48-63` | 267.6 | 207.8 | 29.3 | 23.6 |
| n1-packed16 (prod layout) | znver2-omp | 16 | `48-63` | 268.7 | 213.5 | 28.6 | 23.7 |
| n1-packed8 (1 CCD) | znver2-omp | 8 | `48-55` | 139.5 | 109.6 | 19.3 | 15.2 |
| n1-spread12 (1/CCX) | znver2-omp | 12 | `48,52,56,60,64,68,72,76,80,84,88,92` | 187.2 | 147.7 | 14.7 | 28.2 |
| n1-spread16 (2-3/CCD) | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 269.4 | 213.2 | 42.1 | 30.6 |
| n1-spread16 (2-3/CCD) | prod | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 210.8 | 210.6 | 36.1 | 30.0 |
| n1-spread18 (3/CCD) | znver2-omp | 18 | `48-49,52,56-57,60,64-65,68,72-73,76,80-81,84,88-89,92` | 279.4 | 211.0 | 39.4 | 22.0 |
| n1-spread24 (2/CCX) | znver2-omp | 24 | `48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77,80-81,84-85,88-89,92-93` | 332.3 | 238.1 | 38.9 | 29.7 |
| n1-spread32 | znver2-omp | 32 | `48-50,52-54,56-58,60-62,64-66,68-69,72-74,76-77,80-82,84-85,88-90,92-93` | 428.6 | 343.9 | 38.6 | 30.0 |
| n1-all48 | znver2-omp | 48 | `48-95` | 490.4 | 355.6 | 36.8 | 30.6 |
| n1-all48+SMT (96t) | znver2-omp | 96 | `48-95,144-191` | 449.8 | 270.1 | 31.0 | 27.3 |
| n1-spread24+SMT (48t) | znver2-omp | 48 | `48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77,80-81,84-85,88-89,92-93,144-145,148-149,152-153,156-157,160-161,164-165,168-169,172-173,176-177,180-181,184-185,188-189` | 337.5 | 249.3 | 33.0 | 24.7 |

### Phase 1b: build, flash attention, THP, pinning

| Config | Build | Threads | CPUs | pp1024 @d0 | pp1024 @d2048 | tg128 @d0 | tg128 @d2048 |
|---|---|---|---|---|---|---|---|
| n1-packed16 (prod layout) fa=1 | prod | 16 | `48-63` | 261.6 | 205.6 | 17.3 | 14.6 |
| n1-spread16 fa=1 | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 268.8 | 213.3 | 31.9 | 24.8 |
| n1-spread24 fa=1 | znver2-omp | 24 | `48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77,80-81,84-85,88-89,92-93` | 335.4 | 242.9 | 34.3 | 26.7 |
| n1-spread16 fa=1 kv=q8_0 | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 158.6 | 69.2 | 21.8 | 19.7 |
| n1-spread16 noomp | znver2-noomp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 247.3 | 210.0 | 30.6 | 24.0 |
| n1-spread24 noomp | znver2-noomp | 24 | `48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77,80-81,84-85,88-89,92-93` | 331.6 | 237.0 | 31.6 | 25.4 |
| n1-spread16 THP (glibc hugetlb=1) | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 271.8 | 214.1 | 57.2 | 37.7 |
| n1-spread24 THP (glibc hugetlb=1) | znver2-omp | 24 | `48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77,80-81,84-85,88-89,92-93` | 335.9 | 240.2 | 57.7 | 39.0 |
| n1-spread16 OMP pinned | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 267.4 | 210.6 | 28.9 | 23.5 |
| n1-spread16 poll=0 | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 267.9 | 210.6 | 28.7 | 16.9 |

### Phase 1c: cross-socket / NUMA modes

| Config | Build | Threads | CPUs | pp1024 @d0 | pp1024 @d2048 | tg128 @d0 | tg128 @d2048 |
|---|---|---|---|---|---|---|---|
| x32 interleave | znver2-omp | 32 | `16-17,20-21,24-25,28-29,32-33,36-37,40-41,44-45,48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77` | 433.6 | 342.6 | 37.2 | 29.0 |
| x32 --numa distribute, interleave | znver2-omp | 32 | `16-17,20-21,24-25,28-29,32-33,36-37,40-41,44-45,48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77` | 443.7 | 362.5 | 34.5 | 26.4 |
| x32 --numa distribute, first-touch | znver2-omp | 32 | `16-17,20-21,24-25,28-29,32-33,36-37,40-41,44-45,48-49,52-53,56-57,60-61,64-65,68-69,72-73,76-77` | 439.5 | 357.1 | 26.7 | 21.7 |
| x64 interleave (32+32) | znver2-omp | 64 | `16-79` | 591.1 | 440.0 | 26.3 | 22.7 |
| n1-spread16 --numa isolate | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 261.4 | 197.7 | 16.0 | 8.3 |
| n1-spread16 node-local CPUs, remote memory | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | 249.1 | 183.0 | 17.4 | 17.9 |

### Phase 1d: decode-only repeats, two interleaved cycles, 5 reps each

| Config | Build | Threads | CPUs | pp1024 @d0 | pp1024 @d2048 | tg128 @d0 | tg128 @d2048 |
|---|---|---|---|---|---|---|---|
| A n1-packed16 (prod layout) | prod | 16 | `48-63` | - | - | 8.0 | 7.2 |
| B n1-packed16 THP | prod | 16 | `48-63` | - | - | 32.3 | 26.1 |
| C n1-spread16 | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 14.1 | 9.6 |
| D n1-spread16 THP | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 49.9 | 30.1 |
| E n1-spread16 THP | prod | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 48.4 | 33.9 |
| F n1-packed16 THP | znver2-omp | 16 | `48-63` | - | - | 31.8 | 24.4 |
| A n1-packed16 (prod layout) | prod | 16 | `48-63` | - | - | 8.3 | 7.4 |
| B n1-packed16 THP | prod | 16 | `48-63` | - | - | 30.0 | 23.3 |
| C n1-spread16 | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 16.6 | 14.5 |
| D n1-spread16 THP | znver2-omp | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 48.1 | 28.6 |
| E n1-spread16 THP | prod | 16 | `48-49,52,56-57,60,64-65,68,72-73,76,80,84,88,92` | - | - | 48.7 | 30.6 |
| F n1-packed16 THP | znver2-omp | 16 | `48-63` | - | - | 29.4 | 23.4 |

### Phase 2: llama-server, 6 real Casebook prompts, production think profile

| Run | Decode tok/s (median of calls) | Decode tok/s (pooled) | vs prod 24.0 | Prompt tok/s (median) | Completion tokens | THP RSS |
|---|---|---|---|---|---|---|
| S0 prod-build packed16 4K | 7.8 | 7.7 | -68% | 239 | 11536 | 0% |
| S1 prod-build packed16 THP | 23.3 | 22.4 | -7% | 253 | 11536 | 99% |
| S2 prod-build spread16 THP | 31.7 | 24.3 | +1% | 254 | 11536 | 99% |
| S3 znver2 spread16 THP | 28.8 | 25.7 | +7% | 255 | 10154 | 99% |
| S4 prod-build spread16 tb48 THP | 32.4 | 30.7 | +27% | 421 | 11536 | 99% |
| S5 prod-build spread16 tb48 THP fa-off | 30.9 | 29.6 | +23% | 389 | 18372 | 99% |
| S6 prod-build spread16 tb48 THP fa-on | 32.0 | 30.3 | +26% | 428 | 11536 | 99% |
| S7 znver2 spread16 tb48 THP | 30.1 | 30.5 | +27% | 429 | 10154 | 99% |
| S8 prod-build spread12 tb48 THP | 26.9 | 26.4 | +10% | 425 | 11351 | 99% |
| S9 prod-build spread16 tb48 THP repeat | 31.1 | 29.5 | +23% | 429 | 11536 | 99% |
| S10 prod-build spread24 tb48 THP | 32.7 | 32.7 | +36% | 438 | 8644 | 99% |

### Production during this evaluation

- before (47 think calls): median 24.1 tok/s, median completion 1636 tokens, median prompt 1001
- during (253 think calls): median 24.3 tok/s, median completion 1700 tokens, median prompt 798
