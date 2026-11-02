# Round 5: external leads reviewed against Gigabyte5 measurements (2026-10-31)

## 1. fairydreaming/distributed-llama (CCDs/NUMA nodes as workers)
- What the fork actually adds over b4rtaz/distributed-llama (5 commits, Nov 2024): `--numa N` via `numa_run_on_node`
  per worker process, the output-head (logits) matmul sliced across workers instead of run on the root only, and
  per-task timing. Workers still synchronize over TCP. Supports Llama 2 / Mixtral / Grok-1 only (not qwen35moe).
- Relevance: the logits lesson is already satisfied in ik (output head runs on all threads: 3.8 ms at ~108 GB/s single
  socket, 1.9 ms mirrored). On one NPS1 socket CCD-local workers cannot reduce DRAM traffic: weights are streamed once
  per token with no reuse, and all channels are equidistant. The only thing a CCD-worker design could cut is
  synchronization, which our profiler bounds at ~1.3 ms of 27 ms per token (barrier tail + wake latency).
  Placement experiments (CCD-balanced 24/30/36/48 threads) showed no effect.

## 2. ik_llama.cpp discussion #2030 (NUMA mirroring)
- Recommends weights-only mirroring; several testers found full (weights+KV) mirroring worse; small 3B-active MoE
  (Qwen3-Coder-30B-A3B) at parity, while Qwen3.6-35B-A3B got 1.28x TG; the benefit "tracks how weight-bandwidth-bound
  a model is on one socket".
- Init bug: llama-sweep-bench / llama-batched-bench skipped ggml_numa_set_mirror, so "weights-only" there was really
  full mirror. Our runs use llama-server (common init path) and the logs confirm it: weights-only runs have no KV-mirror
  line (runs/13, runs/16), full runs log "mirrored 20 KV tensors".
- Our result differs on KV: KV mirroring helps long context here (m64w vs m64kv at 9.4K: 28.3 vs 31.8 tok/s, runs/16).
  Likely because (a) our FA reads node-local K/V and (b) we stopped mirroring the recurrent state, whose replication
  cost 0.7 ms/token in the fork's version.
- 3B-active MoE: this model gains ~+15-20% decode mirrored, consistent with the 1.28x report. Our profile explains the
  ceiling: dense matmuls double their bandwidth (qkv 5.1 -> 3.0 ms, output head 3.8 -> 1.9 ms) but FA, ~400 small
  single-threaded ops and cross-socket sync do not scale.

## 3. Llaminar
- One MPI rank per NUMA node with OpenMP inside, NUMA binding verified (fails rather than run remote), tiered expert
  ownership with optional dynamic residency moves, OpenMPI Allreduce/Allgather for row-parallel projections and logits.
  AGPL-3.0; supports Qwen 3.5/3.6 MoE GGUF; README gives no CPU numbers.
- Reusable idea: expert ownership per node with work stealing - we use full mirroring plus per-node chunk counters with
  stealing instead (RAM allows it). MPI collectives would be slower than our shared-memory barrier for single-request decode.

## 4. llama.cpp discussion #11733 (prompt processing poisoning decode page layout)
- Cause there: mmap'd page cache placed by the multi-token (llamafile sgemm) warm-up/prompt path, so decode read
  remotely (numatop RMA/LMA 1.4 vs 0.0). Workarounds: single-token warm-up, drop caches, run generation first.
- Our build: checked with numa_maps after load and after a 2.8K prompt + 512-token decode (runs/27-placemap-*,
  tools/placemap.sh, tools/numamap_summary.py). Single socket: 100% node 1. Mirror: each node's weight copy (20.2 GB)
  and KV copy (384 MB) 100% on its own node. Only small first-touch regions (malloc arenas, ~50 MB activations, prompt
  checkpoints written once) are split, none of which decode streams. --no-mmap + explicit mbind removes this pathology.

## 5. Reddit four-socket design
- The thread itself could not be fetched (reddit blocks automated access). The matching public design found:
  gamozolabs/llama.cpp branch `multisocket-cpu-moe` (4x Xeon 6252N, DeepSeek-V4-Flash MXFP4 experts on CPU):
  expert-home sharding with mbind + work stealing (7.17 -> 19.85 tok/s at 96t), optional antipodal replicas,
  per-(token, expert-slot) chunk counters, persistent pinned workers (22.9 -> 25.9), and the note that barrier variants
  (two-level, tournament) tied because barrier time is mostly arrival skew - the same conclusion as our profiler.
- Tested here: pin-once workers (IK_R5_PINONCE): graph start latency 152 -> 61 us (mirror), 43 -> 35 us (single);
  decode unchanged within noise (38.0 vs 37.9 / 32.3 vs 32.4, runs/29). ik already uses a persistent OpenMP pool, so the
  large gain gamozolabs saw (from a rebuilt disposable pool) does not apply.

## Also found
- ArcLight (OpenBMB, arXiv 2603.07770, MIT): persistent pool split into thread groups, Megatron-style row/column TP with
  per-node buffers, group-local barriers with global barriers only at TP region edges, asynchronous independent
  subgraphs (~+5 tok/s on Kunpeng). Supports MiniCPM5 dense / Qwen3 / Llama, not this MoE + delta-net model.

## What the evidence says to do next (ranked)
1. Flash attention in-model inefficiency (2-3x slower per phase than the identical microbench; largest long-context cost).
2. Concurrent independent subgraphs on thread groups (ArcLight idea), e.g. shared expert || routed experts,
   router || shared-expert gate: removes several barrier-separated small nodes per layer and overlaps their latency.
3. Fuse or parallelize the single-threaded small ops (MUL_MULTI_ADD, residual adds, norms, softmax/top-k): ~0.9 ms
   single socket, ~2.7 ms mirrored.
Not recommended: CCD-local tensor-parallel workers on one socket, barrier redesign, MPI/TCP collectives, expert-home
sharding (only saves RAM, which is not the constraint).
