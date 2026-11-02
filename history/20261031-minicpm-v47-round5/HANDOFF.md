# Handoff: Zen 2 MoE decode optimization on Gigabyte5 (round 5)

You are continuing an in-progress performance engineering task. The full original brief is on Gigabyte5 at
`/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5/MISSION.md`. Read it first; its constraints are binding.
This document records what the previous session established so you do not repeat it.

## Goal in one paragraph

Make single-token decode (and, secondarily, prompt processing) of OpenBMB MiniCPM-V 4.7 35B-A3B Q4_K_M substantially faster
on **one socket (NUMA node 1)** of Gigabyte5, beyond the preserved `ik_llama.cpp` baseline, through source-level work:
topology-aware scheduling, fewer barriers, prefetch, kernel changes. Keep the routing, weights and math the same. Only keep changes
that show repeatable gains at both 2.8K and 9.4K context. Do not deploy anything.

Baseline to beat (512 generated tokens, identical input token IDs, temperature 0):

| Metric | 2,834 ctx | 9,455 ctx |
|---|---:|---:|
| Decode | 34.6 tok/s | 30.7 tok/s |
| Prompt | 402 tok/s | 354 tok/s |

## Access and safety

- SSH: `ssh gigabyte5` (user `ubuntu`, `HostName Gigabyte5.local`, key `~/.ssh/gigabyte5`). The new controller machine needs that key and
  that SSH config entry. Passwordless `sudo -n` works.
- Do not disturb the existing services. The casebook `llama-server` (pid 3136114 at handoff) is pinned to CPUs 0-15 on node 0 and serves
  port 19150. The SRBMiner process mines on the GPUs only. Never touch node 0 CPUs 0-15, and never run GPU work.
- Memory: node 1 has about 24.8 GB free. The model needs about 21 GB of anonymous memory with `--no-mmap`, so run **only one model instance at a time**.
- Work only in `/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5/`. Never modify the model, the original sources or earlier result folders.
- Leave system settings as found. The CPU governor is `schedutil` with boost on (`acpi-cpufreq`). The previous session did **not** change it. Control drift
  by interleaving A/B runs and reporting effective GHz from `perf` cycles.
- System changes made so far: installed `python3-numpy`, `hwloc-nox`, `linux-tools-7.0.0-38-generic` (the user approved tool installs).
  Loaded `amd_uncore` for DRAM counters (`amd_df/dram_channel_data_controller_0..7`), then **unloaded it** at handoff. It was unloaded
  originally. If you load it again, unload it when finished.

## Files you must read (on Gigabyte5)

- `/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation/FINAL-REPORT.md`: ik vs upstream, the worker-collision stall,
  the NUMA expert split (rejected: 24 tok/s, libgomp spin), and the chunk4/repack/passive-wait rejections.
- `/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization/REPORT.md`: thread sweeps, counters, context effect.
- `/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/REPORT.md`: the quality eval. Do **not** rerun the full eval during development.
- The baseline command is `20261031-minicpm-v47-kernel-investigation/corpus-ik-command.json`. Benchmark harness pattern: `corpus.py` in the same folder.
  The inputs are `corpus-medium.txt` / `corpus-long.txt` there (word-salad seeds 2812/9436). Confirm the token counts are 2834 / 9455 via `/tokenize`, then use
  `n_predict: 512`, `ignore_eos`, `cache_prompt: false` and `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`.

Baseline binary: `20261031-minicpm-v47-cpu-optimization/ik-build/bin/llama-server` (ik commit `13cffc0`, Release, GGML_NATIVE, OpenMP, gcc 15.2).
Model: `20261031-minicpm-v47-round4/model.complete.gguf`, SHA-256 `9ecf91e8…888e`.

## Round-5 workspace state

`/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5/`
- `ik-src/`: a clean, independent clone of the ik source on branch `round5`. **No source changes yet.** Build into a separate dir, e.g.
  `cmake -S ik-src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF` then
  `numactl --physcpubind=48-63 cmake --build build -j 16 --target llama-server llama-bench`.
- `tools/gguf_dump.py`: dependency-free GGUF header dump. `model-tensors.json` is its output.
- `tools/bwprobe.c`, `tools/bwsweep.sh`: the DRAM bandwidth probe and placement sweep. Results are in `bwsweep-1.txt`.
- `MISSION.md`: the original brief. `HANDOFF.md`: this file.

## Hardware facts (measured)

- Node 1 = CPUs 48-95, with SMT siblings 144-191. **12 CCXs of 4 cores**: CCX *c* = CPUs `48+4c … 51+4c`, each with a 16 MB L3. **6 CCDs** =
  CCX pairs (48-55, 56-63, 64-71, 72-79, 80-87, 88-95). Private L2 per core.
- Memory per socket: 8 channels × 2 DIMMs (4 GB, single-rank, DDR4-2666), NPS1. That is 170 GB/s theoretical.
- **The baseline 32-thread `--cpu-range` is uneven across CCXs.** Cores per CCX in its first 32 entries: 4,2,4,2,4,1,4,1,4,1,4,1.

Measured read bandwidth (`bwsweep-1.txt`; AVX2 loads, node-1 memory):

| Pattern | GB/s |
|---|---:|
| Streaming, 4/CCX (48 threads) | 120.6 |
| Streaming, 3/CCX (36) | 118.4 |
| Streaming, 2/CCX (24) | 114.1 |
| Streaming, 1/CCX (12) | 103.5 |
| Streaming, baseline-32 placement | 109.3 |
| One CCD (8 cores) | 37.1 (per-CCD GMI cap) |
| One core | 19.9 |
| Jobs of 64 KB + barrier, 24 thr | 16.5 (3.9 µs/job) |
| Jobs of 576 KB (one Q4_K expert matrix) + barrier, 24 thr | 74.7 |
| Jobs of 1 MB + barrier, baseline-32 | 91.5 |
| Jobs of 4.5 MB + barrier, 36 thr | 110.0 |

Cost model: **each barrier-separated job ≈ 3.5–4 µs fixed + bytes / ~115 GB/s.** Software prefetch (distance 512 B) added only ~1–2 GB/s to pure streaming.
Its value lies in hiding per-op ramp-up, not in streaming.

## Model facts (from the GGUF)

Architecture `qwen35moe`. 40 layers: 30 gated-delta-net recurrent layers and 10 full-attention layers (every 4th: 3, 7, … 39). Hidden 2048. 256 experts, 8 used,
expert FFN 512, plus one shared expert (512) with a sigmoid gate. Attention: 16 Q heads, 2 KV heads, head dim 256, mrope. SSM: inner 4096,
state 128, 16 groups, 32 value heads, conv kernel 4. Vocab 248,144.

Weights read per decode token, ≈ **1.88 GiB**:
- Routed experts: ~581 MiB. gate/up are Q4_K; down is Q6_K in 20 layers and Q4_K in 20.
- Shared expert: ~73 MiB
- **Routers `ffn_gate_inp` are F32**: 2 MiB/layer, 80 MiB total
- Delta-net projections (`attn_qkv` Q6_K or Q4_K 2048×8192, `attn_gate` 2048×4096, `ssm_out` 4096×2048): ~598 MiB
- Attention projections: ~149 MiB
- **Output head Q6_K 2048×248144: 398 MiB**

Context-dependent traffic: KV ≈ 20 KB per context position per token (10 layers × 2 KV heads × 256 × K+V × f16), ≈ 57 MB at 2.8K and ≈ 189 MB at 9.4K.
The SSM state is ~2 MiB per layer × 30, ≈ 60 MiB, read and written every token. The state is smaller than node 1's total 192 MB of L3, which
makes CCX-affine state ownership worth examining.

Implications: the baseline moves ≈ 2.1–2.3 GB per token at 34.6 / 30.7 tok/s ≈ 72 GB/s, about 60% of the 120 GB/s ceiling. Pure-bandwidth floor
≈ 18 ms/token (~55 tok/s). With roughly 1,000+ synchronized nodes per token at ~4 µs each, the barrier and ramp-up overhead alone is several ms.
**This, not raw DRAM, is the main gap.**

## How ik executes decode (code map; paths relative to `ik-src`)

- `ggml/src/ggml.c:30015-30117` `ggml_graph_compute`: a new `#pragma omp parallel` region **per graph** (per token). There is no persistent pool beyond libgomp's.
- `ggml.c:29961-30013` per-thread node loop: skip `ggml_is_noop` (5483), `ggml_compute_forward`, then **`ggml_barrier` after every non-noop node** (30001).
- `ggml.c:4964-4974`: for `n_batch <= 32` (all decode) the barrier is `#pragma omp barrier`. A custom spin barrier (4930) exists for large batches.
- `IK_PRINT_TIMING` (`ggml.c:46`) is compiled-out per-op timing. There is no other profiler.
- CPU-side fusions in `ggml_compute_forward` (`ggml.c` ~27277-27752): softmax+argsort+get_rows → `iqk_topk_moe` (27528), DELTA_NET+CPY (27669),
  SSM_CONV, and MUL_MAT chains sharing src1 (18701).
- MoE: `GGML_OP_MOE_FUSED_UP_GATE` (graph build `src/llama-build-context.cpp:1922-1937`; kernel `iqk_moe_fused_up_gate` in
  `ggml/src/iqk/iqk_mul_mat.cpp:895+`). MUL_MAT_ID splits work as (expert, row-chunk) units claimed from an atomic counter (`ggml.c:18946-18979`,
  `chunks_per_expert = MAX(1, MIN(nth, ne01/32))`). The activation is quantized to **Q8_2_X4** once per node (single-token: thread 0 only, 18881-18904), followed by a barrier.
- Expert weighted sum `GGML_OP_MUL_MULTI_ADD` (`iqk_cpu_ops.cpp:430-447`): **only one thread works for decode** (rows = n_tokens = 1).
- AVX2 kernels: Q4_K `mul_mat_qX_K_q8_2_X4_T` (`iqk_gemm_kquants.cpp:742-824`); Q6_K `mul_mat_qY_K_q8_2_X4_T` (897+). One weight row per iteration.
  **No software prefetch.** Zen 2 uses the `maddubs` path, not VNNI.
- Delta net: `ggml_compute_forward_delta_net_f32` (`ggml.c:24798`) → `iqk_fused_delta_net` (`iqk_mul_mat.cpp:1895`), split by heads.
- Graph builder: `src/graphs/build_qwen35.cpp:6-93`. Recurrent layer ops are in `src/llama-delta-net.cpp` (norm, 2 matmuls for qkv and gate, beta/alpha matmuls,
  softplus, conv, silu, l2norm, delta_net, state cpy, gated norm, ssm_out, residual). Attention: `build_std_attention` (`llama-build-context.cpp:3466`). MoE FFN:
  `llm_build_moe_ffn` (1688) and `llm_build_std_moe_ffn` (2044).
- `ggml-moe-prefetch.cpp` only faults in mmap page-cache pages (`MADV_POPULATE_READ`). It is irrelevant with `--no-mmap`.
- Affinity: `--cpu-range` → `set_cpu_thread_affinity` (`ggml.c:29424`) pins OMP thread `ith` to `cpu_affinity[ith % n]`.

## The planned next step (not started)

1. **Per-node profiler**, gated by env `IK_R5_PROF=<path>`, in `ggml_graph_compute_thread`. For graphs with `cgraph->n_batch == 1`, record rdtsc per thread
   per node (compute start/end), plus thread-0 barrier-exit timestamps. After the parallel region, reduce per node: wall (exit-to-exit),
   max-thread compute, mean compute, and wait = wall − max compute. Accumulate by node index (the decode graph is structurally constant) and by op
   name with the layer suffix stripped. Rewrite the file every 64 graphs. Use it to get the real breakdown: matmul kernel time, barrier/imbalance, and
   single-threaded ops (MUL_MULTI_ADD, quantize, top-k, norms).
2. Build a profiled binary. Run the baseline config at both contexts and collect `perf stat` (cycles, instructions, `ls_refills_from_sys.ls_mabresp_lcl_dram`,
   `ls_hw_pf_dc_fill.*`, plus `amd_df` DRAM channels if uncore is loaded) and `perf record` for libgomp spin share.
3. A cheap placement experiment: even CCX placement (3/CCX = 36 threads, 2/CCX = 24, or 32 as 3,3,3,3,3,3,3,3,2,2,2,2) against the baseline order.

## Ideas ranked by expected value (from the measurements)

1. **Fewer barriers per token.** Replace the omp barrier with a CCX-hierarchical spin barrier (4 cores → CCX leader → global). Make
   single-threaded small ops parallel or fuse them: MUL_MULTI_ADD, Q8 quantize, norms, residual adds, router softmax/top-k. Consider a specialized decode executor
   that runs whole layer stages without global barriers where data dependencies allow, such as the shared expert in parallel with routed experts.
2. **Prefetch the next op's weights during barrier waits and small ops.** Weights are static and known in advance. Expert slices become known as soon as top-k
   finishes. Each thread can prefetch its own next slice into L2/L3 so the ~4 µs ramp-up overlaps.
3. **CCX-aware work partitioning.** Give each CCX contiguous row ranges and keep the SSM state for the same heads on the same CCX every token (state
   residency in L3). Balance per-CCX thread counts.
4. Kernel-level: software prefetch in the Q4_K/Q6_K AVX2 kernels, and processing 2–4 rows per iteration to reuse activation loads.
5. Context slowdown (34.6 → 30.7): the KV read is only ~130 MB extra at 9.4K. Check whether flash-attention decode splits KV well across 32 threads
   (2 KV heads, so it may be poorly parallelized).

Already rejected (don't redo without a new reason): the NUMA expert split across sockets, `chunk4.patch` expert fragmentation, runtime repack, `OMP_WAIT_POLICY=PASSIVE`
or low `GOMP_SPINCOUNT`, and the Q8_K activation path (slower). Placing two workers on one physical core causes catastrophic stalls. Always use explicit distinct-core affinity.

## Correctness protocol

Compare greedy output and top-5 `n_probs` against the preserved ik baseline on the medium and long corpora. Different reduction orders may change
long greedy continuations. Treat that as acceptable only if early-token probabilities agree closely (and document it). For kernel changes,
also run the operator checks from the kernel-investigation folder (`opcheck*`): real `blk.0` expert matrices against dequantized float dot products.

## Deliverable

As in MISSION.md: exact source changes and build steps, benchmark commands and affinity maps, before/after at both contexts with repeated samples,
profiling evidence, rejected approaches and reasons, remaining bottlenecks. Keep everything in the round-5 folder.
