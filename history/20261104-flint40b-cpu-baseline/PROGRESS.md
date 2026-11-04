# Flint 40B-A2.5B CPU baseline and round-5 carry-over (2026-11-04)

Model: RecursiveSI/flint-int4-gguf `flint-int4-q4_1.gguf` (official, Q4_1 of the trained INT4 weights), revision
9c47151036bc2c34331142dda04488bceab2bc9f, 25,992,966,976 bytes, SHA-256 43c07bac...ee1a4b verified (MODEL.txt, sha256.txt).
Architecture `flint`: 49 layers (2 dense + 47 MoE, 256 experts, 6 used + 1 shared with sigmoid gate), 32 q / 4 kv heads,
head dim 128, 35 sliding-window layers (window 513, RoPE) + 14 global layers (NoPE), attention sinks on all layers,
vocab 150,144. 39.8B params, ~2.5B active. ik_llama.cpp already supports it at 13cffc0 (src/graphs/build_flint.cpp).

Method (same as the MiniCPM-V 4.7 baseline): word-salad corpora from 20261031-minicpm-v47-kernel-investigation
(corpus-medium.txt / corpus-long.txt; Flint tokenizes them to 2,668 / 8,917 tokens), raw /completion, temperature 0,
n_predict 512, ignore_eos, cache_prompt false, GLIBC hugetlb, -c 16384, --parallel 1. Interleaved rounds.
Nothing else running: no llama-server; casebook-llm, casebook and casebookharness inactive; SRBMiner (GPU) untouched.
Tools: flintbench.py (stock, thread counts), flintab.py (configs: stock1 / r5node1 / r5mirror, --probs).

## Baseline: stock ik_llama.cpp 13cffc0 (preserved binary 20261031-minicpm-v47-cpu-optimization/ik-build), node 1 only

runs/01-baseline (2 rounds, 32 vs 48 decode threads, -tb 48, baseline cpu order):

| Decode threads | 2.8K decode | 9.4K decode | 2.8K prompt | 9.4K prompt |
|---|---:|---:|---:|---:|
| 32 (MiniCPM baseline setting) | 35.41 | 30.35 | 397 | 331 |
| 48 | 37.91 | 31.75 | 398 | 331 |

For reference, MiniCPM-V 4.7 35B-A3B on the same setup: 34.6 / 30.7. Fewer active parameters did not make Flint
faster: it has 49 layers (~1,130 executed graph nodes per token) and is overhead-bound, not bandwidth-bound.

## Round-5 build (build-mirror3, commit 2f69e5ee) on Flint

Correctness (runs/02-r5-probs, runs/03-threads-probs; 32 tokens, top-5 probabilities):
- r5 build single socket, no flags, vs stock, 32 threads: bitwise identical.
- mirror vs mirror + IK_R5_DYN/HUGE/FMOE/FMOE_UG/NONUMAB/PF: bitwise identical.
- stock 32 vs stock 48 threads diverge at token 20 (FA thread-count sensitivity). Mirror 64 matches stock 48 for all 32
  tokens at both contexts (max |dp| 0.15 at 2.8K, 0 at 9.4K), so the difference is thread count, not the mirror.
- Not applicable to Flint: the r5 decode FA kernel (IK_R5_FA) and IK_R5_FA_SHADOW require head dim 256, no sinks and
  no SWA, so they fall back to ik's FA. IK_R5_FMOE does not match: ik builds Flint's shared expert as up/gate, down,
  then the sigmoid gate matmul; r5_fmoe_match expects the gate matmul first (miss at the op-sequence check).

Speed, runs/05-ab (2 rounds, 512 tokens, interleaved):

| Config | 2.8K decode | 9.4K decode | 2.8K prompt | 9.4K prompt |
|---|---:|---:|---:|---:|
| stock, node 1, 48 threads | 39.35 | 31.94 | 397 | 330 |
| r5 mirror, 64 threads (m64pair), no flags | 43.04 | 35.76 | 531 | 453 |
| **r5 mirror + IK_R5_DYN=1 IK_R5_HUGE=1 IK_R5_NONUMAB=1 IK_R5_PF=1** | **50.45** (+28%) | **41.30** (+29%) | **547** | **463** |

Command (not deployed):
```
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_HUGE=1 IK_R5_NONUMAB=1 IK_R5_PF=1 \
numactl --physcpubind=16-95 20261031-minicpm-v47-round5/build-mirror3/bin/llama-server -m flint-int4-q4_1.gguf \
  -ngl 0 --no-mmap --numa mirror --numa-mirror all -t 64 -tb 64 \
  --cpu-range $(cat 20261031-minicpm-v47-round5/tools/m64pair.txt) -c 16384 --parallel 1
```

## Profile (runs/04-prof-files, IK_R5_PROF, mirror + flags, 256 tokens)

Per token: 19.9 ms at 2.8K, 23.7 ms at 9.4K.
- FLASH_ATTN_EXT: 3.87 ms (19%) at 2.8K, 7.86 ms (33%) at 9.4K. At 9.4K the 14 global layers take 400-480 us each
  (~19 MB of KV each, ~45 GB/s: the same slow shared-KV reads IK_R5_FA_SHADOW fixed for MiniCPM). The 35 SWA layers
  take ~55 us each, nearly constant (per-call overhead over 513 positions).
- Routed up/gate 3.25 ms, routed down 1.71 ms, Q 1.67 ms, attn out 1.54 ms, shared/dense up/gate 1.52 ms,
  output head 1.22 ms, router logits 0.65 ms, shared down 0.64 ms; small ops the rest.

## Next candidates
1. Extend the r5 decode FA kernel to Flint (head dim 128, sinks, SWA window, GQA 8) so IK_R5_FA_SHADOW applies:
   most of the 9.4K gap, plus the fixed per-call cost of the SWA layers.
2. Teach r5_fmoe_match / FMOE_UG Flint's shared-expert node order (bit-exact by design; MiniCPM gained ~9%).

## Attention work (2026-11-04), source flint-src branch `flint` (from round5-mirror), build-flint/

Supersedes "Next candidates" above for item 1.

Tried first, not kept: extending the r5 decode FA kernel to Flint (tools/apply_flintfa.NOT-KEPT.py: head dim template,
32 heads, sinks at the merge, sliding-window bound). It works but loses (runs/07-fa-ab):

| tok/s | 2.8K | 9.4K |
|---|---:|---:|
| ik FA | 50.26 | 42.30 |
| r5 FA kernel | 46.08 | 40.20 |
| r5 FA kernel + private copies | 46.06 | 43.67 |

Phase timers (runs/06-fa-probs): ~25 us setup and ~20 us merge per call. Every thread reads all 32 query heads (16 KB
just written by other cores) and keeps partials for all of them; MiniCPM has 2-4 heads and never paid this. ik's loop
splits by KV head and position chunk, which suits 4 KV heads and 32 query heads. Its probabilities also differed from
ik FA by up to 0.02 (token 4 diverged), not investigated further since the kernel was dropped.

Kept: IK_R5_FA_SHADOW=1 also covers ik's GQA decode loop (tools/apply_flintshadow.py, commit c656448). On layers without
a sliding window each thread reads its chunks from packed private copies, per layer and per step slot of the thread,
kept current by the round-5 KV write log. Same chunks, order and kernel, so results are bitwise identical to ik FA.
Rows past n_eff are not copied (they are masked); a chunk that changes thread (the padded cache length grows every 256
positions) is a refill. The 35 sliding-window layers are not covered: the window moves every token.

Correctness:
- runs/08-sh-probs: top-5 probabilities of 32 tokens bitwise identical with and without, 2.8K and 9.4K.
- runs/10-sh-seq (tools/seqtest.py, one server, prompt caching on, requests A, A with a new ending, an unrelated prompt,
  A again, which restores the saved prompt state through the resync path): content and probabilities of every token
  bitwise identical.
- Not re-tested: MiniCPM on this branch. Its recommended config uses IK_R5_FA=1, which returns before the changed loop.

Final, runs/12-final (3 interleaved rounds, 512 tokens):

| Config | 2.8K decode | 9.4K decode | 2.8K prompt | 9.4K prompt | Peak RSS |
|---|---:|---:|---:|---:|---:|
| stock ik, one socket, 48 threads | 39.13 | 31.24 | 394 | 326 | 26.3 GiB |
| mirror + DYN HUGE NONUMAB PF | 48.91 | 40.06 | 546 | 461 | 52.1 GiB |
| **+ IK_R5_FA_SHADOW=1** | **52.21** (+33%) | **47.86** (+53%) | **546** | **461** | 52.8 GiB |

Private copies alone: +6.7% at 2.8K, +19.5% at 9.4K, +0.7 GiB. The other A/B (runs/09-sh-ab) gave 52.83 / 49.20.

Profile with private copies (runs/04-prof-files/prof-*-sh.txt): 18.97 ms per token at 2.8K, 20.04 ms at 9.4K. FA 7.86 ->
4.52 ms at 9.4K: global layers ~430 -> ~165 us each; sliding-window layers still ~45 us each (1.6 ms per token).

Command (not deployed), from this folder:
```
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_HUGE=1 IK_R5_NONUMAB=1 IK_R5_PF=1 IK_R5_FA_SHADOW=1 \
numactl --physcpubind=16-95 build-flint/bin/llama-server -m flint-int4-q4_1.gguf -ngl 0 --no-mmap \
  --numa mirror --numa-mirror all -t 64 -tb 64 \
  --cpu-range $(cat ../20261031-minicpm-v47-round5/tools/m64pair.txt) -c 16384 --parallel 1
```

## Next candidates
1. Fused MoE tail on Flint: teach r5_fmoe_match / FMOE_UG Flint's shared-expert node order (bit-exact by design;
   MiniCPM gained ~9%).
2. Sliding-window layers: ~45 us each for 513 positions, almost all fixed cost. Suspected: merging 24 remote chunks per
   query head. Larger chunks would change rounding, so not bit-exact with ik FA.

## Fused MoE tail on Flint (2026-11-04), IK_R5_FMOE=1 IK_R5_FMOE_UG=1 (tools/apply_flintfmug.py)

Supersedes the earlier note that Flint's shared expert has a gate built in a different order. The real layer has no
shared gate at all (runs/04-prof-files, by-node section): routed up/gate, routed down, MUL_MULTI_ADD, ADD residual, shared
up/gate, shared down, ADD: 7 nodes against MiniCPM's 9. r5_fmug2_match matches it; r5_fmug runs both patterns on a
canonical 9-entry array with the gate nodes NULL. Without a gate the tail is (routed sum + residual) + shared down, the
same additions in the same order as the unfused nodes. IK_R5_FMOE_LOCAL stays MiniCPM only; IK_R5_FMOE_UG=2 (router
fusion) does not match Flint and was not tried.

Correctness:
- runs/13-fm-probs: top-5 probabilities of 32 tokens bitwise identical to the unfused path, 2.8K and 9.4K.
- IK_R5_FMOE=2 consumer check: 200 patterns, 0 external reads of skipped nodes.
- runs/15-fm-seq (KV reuse, prompt caching on, IK_R5_FA_SHADOW=1 in both): every token bitwise identical.
- Coverage: runs/04-prof-files/prof-long-fm.txt, FUSED_UP_GATE n=2 (the 2 dense layers), so all 47 shared up/gate nodes
  fuse. The "no match" lines of IK_R5_FMOE=3 come from multi-token graphs (warmup, prompt), which keep the stock path.

Speed, runs/14-fm-ab (3 rounds, private copies on in both): 52.62 -> 57.42 at 2.8K (+9.1%), 49.34 -> 53.45 at 9.4K (+8.3%).
Won 3/3 in both, comparing each config's rounds.

Final, runs/17-final (3 interleaved rounds, 512 tokens):

| Config | 2.8K decode | 9.4K decode | 2.8K prompt | 9.4K prompt | Peak RSS |
|---|---:|---:|---:|---:|---:|
| stock ik 13cffc0, one socket, 48 threads | 38.84 | 30.96 | 395 | 326 | 26.3 GiB |
| **flint branch, full config** | **58.81** (+51%) | **53.19** (+72%) | **548** (+39%) | **463** (+42%) | 52.8 GiB |

Profile (runs/04-prof-files/prof-long-fm.txt, 256 tokens, profiler on): 18.15 ms per token at 9.4K. Fused MoE 5.11 ms,
FA 4.72 ms, Q 1.58 ms, attn out 1.40 ms, output head 0.85 ms, router logits 0.63 ms.

Recommended command (not deployed), from this folder:
```
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_HUGE=1 IK_R5_NONUMAB=1 IK_R5_PF=1 IK_R5_FA_SHADOW=1 \
IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \
numactl --physcpubind=16-95 build-flint/bin/llama-server -m flint-int4-q4_1.gguf -ngl 0 --no-mmap \
  --numa mirror --numa-mirror all -t 64 -tb 64 \
  --cpu-range $(cat ../20261031-minicpm-v47-round5/tools/m64pair.txt) -c 16384 --parallel 1
```

## Next candidates
- Sliding-window layers: ~45 us each for 513 positions, ~1.6 ms per token, almost all fixed cost. Suspected: merging 24
  remote chunks per query head. Larger chunks change rounding, so not bit-exact with ik FA.
- Not done: quality evaluation of Flint, /v1/chat/completions, reasoning handling, parallel > 1, context > 16K, MiniCPM
  regression run on this branch.
