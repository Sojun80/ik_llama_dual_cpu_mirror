# Round 5 progress notes (2026-10-31)

Source: `ik-src`, branches `round5` (single-socket work) and `round5-mirror` (round5 + NUMA mirror, current HEAD).
Every feature is env-gated, so one binary serves A/B runs. Builds: `build-prof/`, `build-dyn/`, `build-fa/` (single-socket),
`build-mirror/` (first port), `build-mirror2/` (+node-local MoE), **`build-mirror3/` (current, all patches)**.
Tools in `tools/`, raw data per experiment in `runs/NN-*` (server logs, per-run JSON with timings, content, numastat).

## Recommended experimental configuration (not deployed)

Dual-socket NUMA mirror, 32 threads per socket, node-0 cores 16-47 only (casebook's 0-15 never used):

```
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \
IK_R5_NONUMAB=1 IK_R5_PF=1 IK_R5_FA_SHADOW=1 \
numactl --physcpubind=16-95 build-mirror3/bin/llama-server -m <model.complete.gguf> -ngl 0 --no-mmap \
  --numa mirror --numa-mirror all -t 64 -tb 64 --cpu-range $(cat tools/m64pair.txt) -c 16384 --parallel 1
```

`tools/m64.txt`: first 32 entries node-0 cores (16-47, spread over CCXs), last 32 the baseline node-1 order. The thread->node
assignment is a block split (first half of the threads = node 0), so the list must keep that order, and -t must equal -tb.
Memory: ~20.3 GB weights per node + 0.38 GB KV per node (node 0 had ~40 GB of reclaimable page cache; nothing was dropped by hand).

## Results (512 tokens, temperature 0, identical token IDs; interleaved ABBA rounds)

Final A/B, runs/25-final-ab (2 rounds each, casebook idle during the run):

| Config | 2.8K decode | 9.4K decode | 2.8K prompt | 9.4K prompt |
|---|---:|---:|---:|---:|
| base32 (preserved ik baseline, node 1) | 32.36 (34.22, 30.51) | 28.08 (28.57, 27.59) | 393 | 342 |
| **m64: mirror + DYN + FA + HUGE** | **39.39** (40.01, 38.78) | **32.38** (34.26, 30.50) | **513** | **467** |
| m64 + FTZ | 37.37 | 33.64 | 521 | 468 |
| m48 (24+24) mirror | 38.75 | 31.88 | 444 | 398 |

Machine noise was unusually high today: the baseline ran 34.4-35.9 at 2.8K when the box was quiet and 30.5-32.3 in some
sessions (one slow session coincided with casebook activity; another did not). Only same-session interleaved comparisons
are meaningful. Against the cleanest baseline samples the mirror build is roughly +15% decode at 2.8K, +5..15% at 9.4K
(long-context runs are noisy), and +30-35% prompt processing.

Correctness (runs/26-final-probs, runs/15-probs): first 32 greedy tokens identical to the baseline at both contexts;
max |p_ref - p| of the chosen token 0.032 (2.8K) / 0.039 (9.4K), comparable to ik FA's own thread-count sensitivity
(0.017 when going to 48 threads). 512-token output: identical at 2.8K, diverges at char 1868/2180 at 9.4K (near-tie drift
after many identical tokens). DYN and the mirror itself are bit-exact; the FA kernel changes reduction order
(microbench max abs error vs a double reference 1.4e-8, ik's kernel 2.2e-8).

## Commits (round5-mirror, oldest first)

| Commit | Env | What | Effect |
|---|---|---|---|
| 3f44e3f | IK_R5_PROF=<file> | per-node decode profiler | tool (~2% overhead single socket, ~20% at 64 thr cross-socket) |
| ed3c47a | IK_R5_DYN=1 | 75% static rows + atomic chunk pool for decode dense matmul | +2.7% / +1.8% decode, bit-exact |
| 7685cf8 | IK_R5_FA=1 | dedicated GQA single-token FA kernel | FA -15% in-model; see open items |
| e5cac78 | IK_R5_HUGE=1 | 2 MiB-aligned MADV_HUGEPAGE CPU buffers | ~+4% prompt, decode neutral |
| c7b2ad0 | --numa mirror | port of mikechambers84/ik_llama.cpp numa-mirror (diff vs 5c2552b6), conflicts resolved by hand | see table |
| e52dff2 | IK_R5_NODECHUNK (default 1) | expert-chunking paths used node 0's weights (src0->data): now node-local; per-node chunk counters with stealing | mirror 2.8K 36.6 -> 40.5 |
| cf81e97 | IK_R5_DNBIND (default 1) | mirror only K/V (not recurrent state); proportional delta-net head split; node-bound state pages | +3.6% / +8.2% |
| 94cf62e | IK_R5_FTZ=1 | FTZ/DAZ on compute threads | neutral; off |
| ee90abe | IK_R5_FA_DEBUG=1 | per-phase FA timers | tool |
| cf5162b | IK_R5_FA_PFB (default 3) | deeper KV prefetch in the FA kernel | small (~10% of FA score phase) |

## Findings that matter for next steps

1. HANDOFF's barrier hypothesis was wrong: barrier tail + wake latency is ~1.3 ms of 27 ms single-socket (~2 ms mirrored).
2. Placement (CCD-balanced 24/30/36/48 threads) does nothing single-socket.
3. Context slowdown is flash attention. The FA kernel streams KV at only ~1.3 GB/s per thread in-model vs ~2.6 GB/s in the
   standalone benchmark with identical shapes/strides; every phase (even the 32 KB setup) is 2-3x slower in-model.
   Ruled out: THP/TLB, clock (3.25 GHz in both), L3 flush, denormals, prefetch distance. FA does not get faster from
   32 (remote) to 64 (local) threads, so a per-call cost that does not scale with work is the prime suspect. Unresolved.
4. In mirror mode the remaining non-matmul costs are: FA (2.5 / 5.5 ms), ~400 effectively single-threaded small nodes
   (~2.7 ms, cross-socket cache-line traffic), router F32 matmuls (~0.8 ms), omp fork per graph (~140 us).
5. NUMA mirror needs `-t == -tb` and an even split; prompt processing scales well with 64 threads (+30%).

## Rejected / neutral
CCD-balanced placement; FTZ/DAZ; prefetch distance 1 vs 6 blocks; weights-only mirroring at long context (KV remote reads
make 9.4K slower than single socket); KV mirroring of the recurrent state (pure overhead).

## Open
- FA in-model slowdown (above). Next: IBS sampling (perf with ibs_op) during FA only, or a microbench that reproduces the
  in-model sequence (rope writes Q, then FA) to test the Q/partials coherence theory.
- Fuse/parallelize the single-threaded small ops (MUL_MULTI_ADD, norms, softmax/top-k) - doubled cost under mirror.
- Router F32 matmuls: only 16 of 32 threads had work single-socket (DYN fixes the split; still bandwidth-poor).
- Correctness was checked on two synthetic corpora only; no log-analysis eval was run (as instructed).

| 9c (latest) | IK_R5_PINONCE=1 | pin worker threads once | start latency 152->61 us mirrored; decode neutral |

External leads reviewed: see RESEARCH.md.

FA integration-boundary investigation and small-op work: see FA-BOUNDARY.md (IK_R5_FA_2PASS, IK_R5_MMA_COLS, diagnostics IK_R5_FA_TWICE / IK_R5_FA_STREAMTEST).

| 1caa9665 | IK_R5_FMOE=1 | fused MoE tail at decode: routed down + weighted sum + residual + shared-expert down + gate + add in one node | bit-exact; mirror 2.8K +4.4%, see below |
| c878b8ab | IK_R5_FMOE_UG=1 | routed + shared up/gate folded into the fused MoE tail (9 nodes -> 1, 7 barriers -> 4) | bit-exact; mirror +4.8% 2.8K, +4.0% 9.4K, see below |
| 96a11658 | IK_R5_FMOE_UG=2 | router (norm, logits, top-k, weight normalization) folded into the fused MoE node too | bit-exact; no measurable gain, opt-in only |
| 4ab938e6 | IK_R5_NONUMAB=1 | explicit MPOL_LOCAL task policy: process opts out of automatic NUMA balancing (no system change) | bit-exact; mirror decode faults/migrations -> ~0, speed unchanged |
| 0520716c | IK_R5_FMOE_LOCAL=1 | socket-local fused MoE step: each node runs up/gate, quantization and down of its own experts with node-local barriers, one cross-node barrier per MoE layer instead of three (opt-in, default off) | bit-exact; decode ~1% slower (runs/76) |
| 2e37920d | IK_R5_PF=1 | prefetch each thread's upcoming weights (static rows of the next dense matmuls, its MoE down rows once the experts are known) while it spins in a barrier | bit-exact; mirror decode +4.5% 2.8K, +7.8% 9.4K (runs/78) |
| 87c45b2d | IK_R5_PF_FA=1 | also prefetch the thread's K/V positions of the next decode FA (opt-in) | neutral (runs/79) |
| c3d52ea4 | IK_R5_PF_LINES default 2 | 2 prefetches per barrier spin iteration instead of 4 | bit-exact (runs/84); +3.3% 2.8K, +1.0% 9.4K (runs/83) |
| 2f69e5ee | IK_R5_FA_SHADOW=1 | decode FA reads each thread's K/V rows from a private per-thread copy, kept current through a log of KV writes (mirror mode only) | bit-exact (runs/106, KV reuse runs/107); +1.7% 2.8K, +10.2% 9.4K (runs/108) |

## Fused MoE tail (IK_R5_FMOE, commit 1caa9665)
Per MoE layer the decode graph had 8 nodes after the routed up/gate (routed down MUL_MAT_ID, MUL_MULTI_ADD, ADD residual,
shared-expert gate MUL_MAT, shared FUSED_UP_GATE, shared down MUL_MAT, FUSED_MUL_UNARY, ADD). The executor now matches that
pattern and runs it as one node: shared gate and up/gate first, then every thread owns a 16-row slice of the 2048 output rows
and computes the 8 routed down rows + the shared down rows (node-local weights) and the whole combine. 8 barriers -> 5,
280 fewer nodes per token, no core gathers other cores' partials.
- Safety: the graph allocator reuses freed buffers, so in the decode graph the shared gate/up outputs can sit on top of the
  routed inputs the original order had finished with. The fused path therefore quantizes par and snapshots ids, routing
  weights and its residual slice before running them; after the last internal barrier no graph tensor is read. It only
  applies when the node pattern, shapes and work-buffer size check out (prompt graphs and the 256-expert warmup fall through).
  IK_R5_FMOE=2 also checks that no later node reads a skipped node (0 of 200 patterns); IK_R5_FMOE=3 prints misses.
- Correctness (runs/41-fmoe-probs): top-5 probabilities of 32 tokens bitwise identical to the unfused path, single socket and
  mirror, 2.8K and 9.4K.
- Profile (runs/43 vs runs/40, cross-session): MoE tail 4.54 -> 4.00 ms/token single socket, 4.33 -> 3.65 mirrored.
- Speed (runs/44 + runs/45, 6 interleaved rounds each, 512 tokens):

| Config | 2.8K base | 2.8K fused | 9.4K base | 9.4K fused |
|---|---:|---:|---:|---:|
| s32 (node 1) | 32.97 | 33.77 (+2.4%, 3/6 pairs won) | 29.47 | 29.98 (+1.7%, 4/6) |
| m64 (mirror) | 37.64 | 39.28 (+4.4%, 6/6) | 32.76 | 33.01 (+0.8%, 3/6) |

  Single-socket runs were noisy this session (+-2 tok/s between rounds), so only the mirrored 2.8K gain is clear-cut.
  Dropping the barrier between the shared gate and up/gate (IK_R5_FMOE_B=0) was not better (runs/44).

## Routed + shared up/gate in the fused MoE tail (IK_R5_FMOE_UG, commit c878b8ab)
Needs IK_R5_FMOE=1. The match now starts at the routed MOE_FUSED_UP_GATE node, so routed up/gate .. l_out (9 nodes)
run as one node in four phases:
  P0 one thread quantizes the FFN input, another computes the shared-expert gate logit (same iqk_mul_mat_4d call as the
     MUL_MAT node, into the work buffer); every thread snapshots ids, routing weights and its residual slice | barrier
  P1 one chunk pool of 8 x 16 routed up/gate chunks + 16 shared up/gate chunks (32 rows each, IK_R5_FMOE_CR), same
     iqk_moe_fused_up_gate kernel and row mapping as the original nodes, NUMA-aware claiming as before | barrier
  P2 quantize the 8 routed rows and the shared row, one row per thread | barrier
  P3 per-thread output rows: routed + shared down rows and the combine (unchanged from IK_R5_FMOE)
Nothing is written to a graph tensor before l_out, so compute-buffer reuse cannot bite. Barriers from the routed
up/gate to l_out: 7 -> 4; the shared up/gate no longer runs as its own unbalanced phase.
- Gotchas found on the way: the routed node reads a reshape view of the normed input (accept a zero-offset view);
  and the activation types here are Q8_2_X4 / Q8_0_X4, which interleave blocks, so rows must be quantized whole
  (a block-wise first attempt produced NaN logits).
- Correctness (runs/46-fmug-probs): top-5 probabilities of 32 tokens bitwise identical to IK_R5_FMOE=1, single socket
  and mirror, 2.8K and 9.4K; consumer check (IK_R5_FMOE=2) 0 external reads in 200 patterns.
- Profile (runs/49-fmug-prof, same session), routed up/gate .. l_out per token: mirrored 6.19 -> 5.36 ms,
  single socket 8.16 -> 7.77 ms.
- Speed (runs/48-fmug-bench, 4 interleaved rounds, 512 tokens):

| Config | 2.8K FMOE | 2.8K +UG | 9.4K FMOE | 9.4K +UG |
|---|---:|---:|---:|---:|
| s32 (node 1) | 33.96 | 34.00 (+0.1%, 2/4 pairs won) | 30.11 | 30.65 (+1.8%, 2/4) |
| m64 (mirror) | 40.62 | 42.58 (+4.8%, 4/4) | 34.66 | 36.04 (+4.0%, 4/4) |

  Single-socket runs stayed noisy (+-2 tok/s between rounds); the mirrored gain is consistent.

## Router in the fused MoE node (IK_R5_FMOE_UG=2, commit 96a11658) - neutral, not recommended
The 12 router nodes before the routed up/gate (FUSED_RMS_NORM, logits MUL_MAT, executor-fused SOFT_MAX..GET_ROWS,
executor-fused SUM_ROWS+DIV, reshapes) join the fused node: every thread computes the normed input itself
(ggml_compute_forward_fused_rms_norm on a private copy), the 256 logits rows are split over the threads, and after one
barrier every thread runs iqk_topk_moe and iqk_sumrows_div on private copies. Barriers from the norm to l_out 8 -> 4,
nodes per graph 863 -> 703. Only used when the MoE expert prefetch is off (the hook reads the graph's ids).
- Correctness (runs/51-fmrt-probs): bitwise identical to IK_R5_FMOE_UG=1, single socket and mirror, 2.8K and 9.4K;
  consumer check 0 external reads.
- Profile (runs/52-fmrt-prof, same session), norm .. l_out per token: mirrored 6.71 -> 6.67 ms, single socket
  9.19 -> 8.91 ms. Mean per-thread busy time rose (5.34 -> 6.08 ms mirrored): the repeated per-thread norm and top-k
  cost about what the removed barriers saved. The router logits matmul (2 MB of f32 weights per layer, ~19 us)
  dominates the router either way.
- Speed (runs/53-fmrt-bench, 4 interleaved rounds, 512 tokens), UG=1 -> UG=2: s32 2.8K 34.79 -> 33.77,
  9.4K 30.28 -> 30.56; m64 2.8K 40.56 -> 40.62, 9.4K 34.54 -> 34.62. Every config won exactly 2 of 4 pairs,
  always the one that ran second in its round, so this run also shows a run-order bias of ~1-2 tok/s that the
  ABBA order cancels.
- Kept as an opt-in for later work (e.g. sharing the top-k per CCX instead of per thread); recommended command unchanged.

## FA cold fetch (2026-11-01): see FA-BOUNDARY.md
Power-state ramp, stale coherence of re-read KV lines, lockstep channel hotspots and uneven interleave are ruled out
by A/B runs. In-model FA streams KV at about half of the plain streaming rate of the same 32-thread layout (~40 vs 80 GB/s). Mirror mode had automatic
NUMA balancing faulting on shared runtime pages; IK_R5_NONUMAB=1 removes it (bit-exact) without a measurable decode
gain. Node-0 threads read KV ~1.5-1.7x slower than node-1 threads in mirror mode; cause still open.

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

## Socket-local MoE step (2026-11-01): IK_R5_FMOE_LOCAL=1, commit 0520716c, tools/apply_r5local.py
In r5_fmug (IK_R5_FMOE_UG=1) node k now owns the experts at positions [k*n/2, (k+1)*n/2) of the selected list and half
of the shared expert's up/gate rows. It quantizes its own input copy, runs its up/gate chunks from its own pool (no
cross-node stealing), quantizes its routed rows and computes their down rows for all 2048 outputs, with node-local
barriers between the steps. One full barrier follows, then every thread computes its slice of the shared down rows
and the unchanged combine. Cross-socket barriers per MoE layer: 1 instead of 3 (plus the executor's end-of-node one).
Bit-exact against the default path at 2.8K and 9.4K (runs/75, completion_probabilities identical).
Speed (runs/76, ABBA, 4 rounds, casebook stopped): 2.8K 41.25 -> 40.84 tok/s (slower in all 4 rounds), 9.4K
36.25 -> 36.05. The saved cross-socket barriers are worth less than the cross-node work stealing they replace: with a
fixed 4+4 split the slower node (node 0 reads memory more slowly in mirror mode, see FA-BOUNDARY.md) sets the pace,
and the three extra node-local barriers are not free. Left in as an option, off by default.
Note: m64 decoded 41.3 / 36.3 here and 43.0 / 38.0 in runs/69 with the same build and settings; run-to-run
variation across sessions is larger than within one ABBA run, so compare only within a run.

## Prefetch while waiting at barriers (2026-11-01): IK_R5_PF=1, commits 2e37920d and 87c45b2d
Profile (runs/49, m64 + IK_R5_FMOE_UG=1, 2.8K): of ~22.1 ms per token each thread is idle ~5.5 ms on average, ~1.07 ms
of it in the delta-net step where few threads have work. IK_R5_DYN=1 already gives every thread a fixed share (75%) of
the rows of each single-token dense matmul before the dynamic remainder, so its next reads are known in advance.
tools/apply_r5pf.py: before each node barrier, every thread queues its static rows of the next dense MUL_MAT nodes
(lookahead IK_R5_PF_WIN=32 nodes, up to IK_R5_PF_BUDGET=256 KiB not yet prefetched) from its node-local mirror copy;
r5_fmug replaces the queue with the thread's down rows (8 routed experts + shared, its output slice) as soon as the
experts are known. Every spin iteration of the NUMA barriers issues IK_R5_PF_LINES=4 prefetcht1 (L2) before its
pause. No result changes (runs/77: completion_probabilities bitwise identical at 2.8K and 9.4K).

| config (runs/78, ABBA, 4 rounds, casebook stopped) | 2.8K | 9.4K |
|---|---:|---:|
| m64 recommended so far | 41.47 | 35.40 |
| + IK_R5_PF=1 | **43.33** (+4.5%, won all 4 rounds) | **38.17** (+7.8%, won all 4 rounds) |
| + IK_R5_PF=1, 8 lines, 384 KiB budget | 43.08 | 37.53 |

Follow-ups (runs/79, ABBA, 4 rounds, all with IK_R5_PF=1): prefetching each thread's K/V positions of the next decode
FA (tools/apply_r5pffa.py, IK_R5_PF_FA=1) 43.86 -> 44.14 at 2.8K, 37.62 -> 37.64 at 9.4K: noise, so opt-in. A larger
static share (IK_R5_DYN_STATIC=90) is not better (43.97 / 37.01).

## Decode profile with prefetch (2026-11-01, runs/81-pf-prof)
build-mirror3 (87c45b2d) has the r5prof profiler; IK_R5_PROF via `r5bench.py --prof`, 1 round per config, m64
recommended settings with and without IK_R5_PF=1. Comparison script: tools/profcmp.py (idle = wall - mean thread
compute per node class, us per token). The profiler slows decode (pf 37.2 / 36.9 tok/s). The two 2.8K runs ran first
and their output head is disturbed (7.7 / 4.4 ms instead of ~2 ms, as in runs/49 and the 9.4K runs), so read
the 9.4K pair and 2.8K without the output head.

| group (9.4K, us per token) | wall m64 | wall +PF | idle m64 | idle +PF |
|---|---:|---:|---:|---:|
| dense MUL_MAT (without output head) | 8144 | **5684** | 1685 | 1220 |
| MoE up/gate/down (fused) | 5957 | 5940 | 754 | 840 |
| flash attention | 4260 | 4373 | 95 | 131 |
| delta-net | 2415 | 2364 | 1454 | 1381 |
| output head | 2028 | 1839 | 89 | 46 |
| all other nodes (norms, adds, copies, softplus, ...) | 3323 | 3909 | 2593 | 3134 |
| per token | 26137 | 24121 | | |

- The prefetch gain is in the dense matmuls (-2.5 ms at 9.4K, -1.5 ms at 2.8K); the MoE step barely changes.
- Cost: the small nodes get ~0.6 ms slower with IK_R5_PF=1 (same at 2.8K: 3196 -> 3931 wall). They are almost all
  sync time, and a waiting thread now issues 4 prefetches before each check of the barrier; prefetches can stall
  when the miss queue is full, so release is noticed later.
- Remaining idle windows, largest first: small nodes ~3.1 ms (barrier-bound: ~860 nodes per token), delta-net
  ~1.4 ms (max - mean compute 1.34 ms: the kernel keeps only some threads busy), dense matmul ~1.2 ms,
  MoE ~0.8 ms.

## Trimming the prefetch cost (2026-11-01): IK_R5_PF_LINES default 2, commit c3d52ea4
The profile (runs/81) put the prefetch cost in the small nodes' barrier waits. First guess: the cross-node leader
(the last thread on a node, which releases its node's followers) noticing the release late. A build where that leader
only pauses (IK_R5_PF_LEADER, not kept) was tested together with fewer prefetches per spin iteration.

| tok/s, all with IK_R5_PF=1 | 2.8K | 9.4K |
|---|---:|---:|
| runs/82 (3 rounds): 4 lines, leader prefetches (previous default) | 44.47 | 39.30 |
| 4 lines, leader only pauses | 43.73 | 38.71 |
| 2 lines, leader only pauses | 45.37 | 40.12 |
| 1 line, leader only pauses | 45.34 | 39.79 |
| runs/83 (4 rounds): 4 lines, leader prefetches (previous default) | 43.52 | 39.53 |
| **2 lines, leader prefetches** | **44.94** (+3.3%, won 4/4) | **39.93** (+1.0%, won 3/4) |
| 2 lines, leader only pauses | 43.90 | 39.20 |

The leader's prefetching helps, so the leader change was dropped; the default is now 2 lines per iteration
(tools/apply_r5pf2.py). Every config drifted ~+2 tok/s over runs/82, so only within-run comparisons count.
runs/84: completion_probabilities with the new default bitwise identical to runs/77's no-prefetch reference at 2.8K
and 9.4K.

## Delta-net: splitting heads over two threads (2026-11-01, not kept) and a better CPU order (kept)
The fused delta-net step at decode gives each of the 32 heads (head_dim 128, 64 KiB of f32 state) to one thread, so
32 of 64 threads idle there (~1.4 ms per token, runs/81). Within a head every value row is independent, so
IK_R5_DN_SPLIT=1 split each head's rows over two adjacent threads (same NUMA node), bit-exact.

- Runtime row bounds (first version): bit-exact (runs/85) but decode -12.6% at 2.8K, -10.2% at 9.4K (runs/86); the
  profile shows the kernel itself 2.5x slower (~74 -> ~200 us per layer, runs/87).
- Constant 64-row loops (second version, pointers offset to the part's first row): bit-exact (runs/89), still 2.4x
  slower in the kernel (runs/90).
- Cause: the recurrent state is only 32-byte aligned: 30 of 40 printed state pointers were 32 mod 64 (temporary debug
  build, runs/91). A 512-byte state column then splits two cache lines between the halves (the midpoint and the column
  edge), whatever the split point, and the partner threads of the m64 order are on different CCDs, so those lines bounce
  over the IO die. Microbenchmark tools/dnbench.cpp (runs/88, node-1 cores, us per layer of 16 heads): one thread
  whole heads 202; two halves, constant loops, aligned state: 130 same CCX / 133 other CCD; state 32 B off alignment:
  164 same CCX / 235 other CCD.
- tools/m64pair.txt: the same 64 CPUs, reordered so threads 2h and 2h+1 share a CCX (node 0: 16..47 in order; node 1:
  14 of 16 pairs, the 4 single-CPU CCXs of the m64 set pair across CCDs). runs/92 (ABBA, 3 rounds):

| tok/s | 2.8K | 9.4K |
|---|---:|---:|
| m64 order, IK_R5_PF=1 | 44.21 | 39.18 |
| **m64pair order** | **45.40** (+2.7%, won 3/3) | **40.29** (+2.8%, won 3/3) |
| m64pair order + IK_R5_DN_SPLIT=1 | 44.99 | 39.72 |

The split no longer loses much with paired threads but does not win, so it was dropped (tools/apply_r5dnsplit.NOT-KEPT.py,
not in ik-src). The order alone is bitwise identical to runs/77 at 2.8K and 9.4K (runs/93). Its gain is not in the
delta-net step (1982 -> 1991 us per token at 9.4K, runs/94) but in the MoE step (5758 -> 5532) and the smaller
matmuls (ffn_moe_logits, linear_attn_out): ~0.57 ms per token in all. Likely neighbouring threads reading neighbouring
rows now share an L3; not verified. The recommended command now uses tools/m64pair.txt (batch uses its first 48
entries: the same 32 node-0 CPUs plus a different 16 of node 1; prompt speed unchanged or slightly higher).

## KV reads in the decode FA (2026-11-01/02): IK_R5_FA_SHADOW=1, commit 2f69e5ee
At 9.4K the decode FA reads ~189 MB of KV per token (10 attention layers x 2 KV heads x 256 dims x F16, K and V) in
~4.3 ms, ~44 GB/s, a quarter of what the weight matmuls stream. Steps, all with IK_R5_FA_DEBUG=1 per-call timings
(slowest thread = critical path), m64pair, IK_R5_PF=1, casebook stopped:
1. Node 0 vs node 1 (runs/95, tools/fa_ccx.py): the node-0 slowness of runs/66 was casebook. At 9.4K node 0 now reads
   6.1-6.8K ticks per position vs 10.3K then; node 1 6.3-7.3K. A thread alone in its CCX is only 5-10% faster than one
   of four, so not CCX bandwidth.
2. More readers (runs/97, m96 = 48+48 threads): slower, critical path 1.22M -> 1.52M ticks at 9.4K; per-position cost
   roughly doubles. So a separate 96-thread FA set would not help (not built).
3. Fewer readers (runs/98, diagnostic IK_R5_FA_STRIDE, tools/apply_r5fastride.DIAG.py): 64 / 32 / 16 readers give
   1.05M / 1.08M / 1.37M ticks at 9.4K: per-reader speed scales with 1/readers, KV throughput per node fixed at
   ~20-25 GB/s (~2.5-3 GB/s per CCX vs ~9.6 for the weights).
4. Private copy (runs/100-102, diagnostic tools/apply_r5fashadow.DIAG.py): each thread reads its rows from a copy it
   allocated and wrote itself: 2.0-2.4x faster FA (9.4K 1.07-1.19M -> 0.43-0.49M ticks; 2.8K 0.39-0.46M -> 0.22-0.32M),
   bit-identical.
5. Not the cause: remote-socket writers (each node's copy written only by its own threads, IK_R5_KVLOCAL, runs/101:
   no change); line ownership (re-storing the rows in place, IK_R5_FA_REOWN, runs/102: no change); page size and
   placement (tools/thpmap.py, runs/103-105: the node-1 KV copy is 100% THP, node-local and 2 MiB aligned; node 0's is
   split into 33 MiB attention pieces, ~95% THP with ~5% of pages on node 1, between 1 MiB delta-net state pieces bound
   alternately to both nodes by IK_R5_DNBIND; both copies are equally slow). Most likely the DRAM bank/row pattern of
   64 concurrent streams at ~152 KiB spacing inside the same few huge pages, versus one private page set per thread;
   not proven.
6. IK_R5_FA_SHADOW=1 (tools/apply_r5fashadow.py): the private copy made safe. ggml.c logs the byte range of every KV
   write in mirror mode (ggml_numa_replicate_kv_write) and whole tensors on the cold resync path; a thread re-copies
   its range when the range moves or a logged write touched rows it already copied, and otherwise only copies newly
   appended rows. Active only with the NUMA mirror. Memory: two 2 MiB-rounded buffers per thread and attention layer
   (~1.3 GB at 64 threads, 10 layers, 9.4K).
   - runs/106: completion_probabilities bitwise identical to runs/77 at 2.8K and 9.4K.
   - runs/107 (tools/seqtest.py): one server, prompt caching on, requests A, A with a different ending, an unrelated
     prompt, A again: content and top-5 probabilities of every token bitwise identical with and without the copies.
   - runs/108 (ABBA, 4 rounds):

| tok/s | 2.8K | 9.4K |
|---|---:|---:|
| m64pair, IK_R5_PF=1 | 46.78 | 40.76 |
| **+ IK_R5_FA_SHADOW=1** | **47.57** (+1.7%, won 4/4) | **44.92** (+10.2%, won 4/4) |
