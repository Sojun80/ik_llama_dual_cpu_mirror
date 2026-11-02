# MiniCPM-V 4.7 CPU optimization — Gigabyte5

2026-10-31. The preserved upstream build, exact publisher Q4_K_M model, and CPU-only execution were used. Measurements use synthetic inputs only; no log-analysis evaluation, grading, or production service change was performed. Existing casebook CPU server and GPU miner were running throughout. The original server binary and complete working command were copied into `working-snapshot/`; the original model remains in place and its identity and hash were recorded; the model SHA-256 matched `9ecf91e814d7e9adc64325b1b65f8628a9b1bb8da4f61a92869ca7489bfd888e`.

## Findings

- **Upstream decode:** 32 threads are the best repeatable tested general-purpose medium-context choice. At long context 48 threads were consistently faster than 32 in two alternating pairs, though the differences are small. These are candidate settings, not deployed changes, and one trivial output equivalence check does not establish complete correctness.
- **Prompt pool:** Keep the existing 48-thread pool, batch 2048, microbatch 512 among the tested combinations.
- **Context effect:** At 16 threads with the exact preserved mask, synthetic decode at context depths 0/2800/9475 was 21.48/19.63/16.63 tok/s. Counters show greater work and cache misses per generated token at depth, with lower DRAM bytes per second; this supports context-dependent work and cache pressure, but does not isolate attention as a single cause or prove DRAM saturation.
- **Two sockets:** 16 physical cores on each socket with node-1-resident weights achieved 14.10 tok/s at depth 2800, versus 21.18 for 32 on node 1. The requested interleave policy was attempted with transparent huge pages and 4 KiB pages, but both readbacks placed ~20 GiB on node 1 and <0.4 GiB on node 0. Interleaved-weight performance is therefore unresolved; those results are not presented as valid interleave measurements. No system cache clearing or memory migration was forced.
- **ik_llama.cpp:** Exact GGUF loaded in CPU-only commit 13cffc006b82e03e4d89cbb8c7327f0255ae3d18. Two deterministic fixed-input server requests at 2812 tokens produced identical text and input token IDs to upstream; ik decoded 34.23/34.28 tok/s versus upstream 18.69/18.72, both with 16 threads for prompt and decode. Huge pages and node-1 residency were confirmed for both processes. At 9436 tokens its server did not return a response for over 90 seconds after logging its final prompt checkpoint. The attempt was stopped; no result is inferred for that run. This small smoke check is not correctness qualification, and the long-context behavior blocks a runtime recommendation.

## Thread sweep

One process per setting, physical cores spread over six CCDs, node 1 memory, THP via `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, CPU-only, 64 generated synthetic tokens, depth 2800, two repetitions per process. Thread count 16 in this sweep used a different spread from the exact preserved 16-core mask: last two CCDs use adjacent cores 80/81 and 88/89 instead of 80/84 and 88/92. The original mask was separately measured below.

| Physical cores | Mean decode tok/s | Samples |
|---:|---:|---|
| 8 | 16.36 | 16.35, 16.36 |
| 12 | 18.89 | 18.89, 18.90 |
| 16 | 19.59 | 19.59, 19.60 |
| 24 | 20.52 | 20.51, 20.53 |
| 32 | 21.18 | 21.17, 21.20 |
| 48 | 21.00 | 20.97, 21.02 |

A second series, in order 32/16/48, repeated two samples per depth using those same sweep layouts:

| Threads | Depth | Mean decode tok/s |
|---:|---:|---:|
| 16 | 2,800 | 19.34 |
| 16 | 9,475 | 16.44 |
| 32 | 2,800 | 20.62 |
| 32 | 9,475 | 17.86 |
| 48 | 2,800 | 20.15 |
| 48 | 9,475 | 18.07 |

At depth 9475, an additional alternating 32/48/32/48 series yielded 17.51/18.00/17.61/18.26 tok/s (means of two measurements each). Treat these as synthetic short-run results with observed between-run drift, not claims about all workload shapes.

The server was also run on the same synthetic input at temperature zero with 64 generated tokens. All outputs matched the 16-thread synthetic reference exactly. One sample per setting; all used the original 48-thread batch pool. Original 16 means its exact preserved CPU mask.

| Decode cores | Prompt tokens | Prompt tok/s | Decode tok/s | Output matches reference |
|---:|---:|---:|---:|---|
| 16 | 2,812 | 233.3 | 19.00 | yes |
| 16 | 9,436 | 204.5 | 16.17 | yes |
| 32 | 2,812 | 231.3 | 20.37 | yes |
| 32 | 9,436 | 203.7 | 17.76 | yes |
| 48 | 2,812 | 232.5 | 20.55 | yes |
| 48 | 9,436 | 204.9 | 18.64 | yes |

## Prompt processing

Synthetic 2048-token prompt, two samples, node 1, process-local THP. 48-thread batch pool unless otherwise indicated; no decode in these cases.

| Threads | Batch | Microbatch | Mean prompt tok/s |
|---:|---:|---:|---:|
| 48 | 2048 | 128 | 194.24 |
| 48 | 2048 | 256 | 219.35 |
| 48 | 2048 | 512 | 240.28 |
| 48 | 2048 | 1024 | 232.99 |
| 48 | 1024 | 512 | 234.52 |
| 32 | 2048 | 512 | 220.44 |

## Memory and counters

During the exact-mask 16-core benchmark, `smaps_rollup` showed 21,010,488 KiB anonymous and 20,996,096 KiB `AnonHugePages`; `numastat` showed 20,528 MiB on node 1 and 2.3 MiB on node 0. Both metrics and raw maps are retained. The 16-core counter experiment used a slightly different 16-core spread (80/81 and 88/89 rather than 80/84 and 88/92), identical between its two depths.

For two 512-token decode runs, `perf` sampled the same pinned physical cores and node 1 for 15 seconds during generation. The in-tree AMD uncore driver was initially unloaded; it was temporarily loaded to expose the `amd_df` controller and unloaded afterward. The NPS1 eight-channel DRAM metric is approximate, node-wide and ~50% multiplexed; core events were ~83% scheduled, so the per-token calculations are approximate. Values include any other activity on the node.

| Depth | Decode tok/s | Node 1 DRAM GiB/s | Instructions/token | Generic cache misses/token |
|---:|---:|---:|---:|---:|
| 2800 | 18.80 | 45.6 | ~1.75 billion | ~11.2 million |
| 9475 | 15.88 | 40.6 | ~2.24 billion | ~39.2 million |

Approximate memory-controller traffic per generated token rose from 2.43 to 2.56 GiB, while instructions/token rose ~28% and generic cache misses/token ~3.5x. These are counter-derived estimates using each run's average decode rate; the timing interval does not track the exact number of generated tokens. They do not separately measure attention versus other context-dependent work, and the generic `cache-misses` event is not identified here as a specific cache level. Both runs used the same alternate 16-core mask and 512 output tokens, so their sampled depths grow by up to 512 tokens.

## Reproduction and safety

Full commands and masks are in `threads-*-command.json`, `prefill-*-command.json`, `verify-*-command.json`, and `candidate-*-command.json`; per-run stdout, logs, process memory evidence, and counters remain in this directory. Original 16-core mask is `0x111113131313000000000000`; 32-core candidate is `0x1f1f1f1f3f3f000000000000`; 48-core mask and batch mask are `0xffffffffffff000000000000`. All use strict affinity. The original context is 32768; synthetic server compatibility runs used 16384 context and two fixed inputs, while first short runtime smoke used 4096. These checks cannot substitute for a broader accuracy test.

Upstream source commit 0ec4027a2a0183de87c007069adddc06e665f7de. ik's Qwen3.5 MoE support is present, but its source lacks the upstream MiniCPM-V 4.7 rope section-order implementation; upstream applies that ordering to multimodal embeddings, so this observation alone does not prove a text-only mismatch. No multimodal compatibility claim is made. The long ik attempt and first failed same-port attempt are retained; the latter was rerun with different ports. Temporary servers stopped, original CPU service and GPU miner were preserved, and the uncore module returned to its prior unloaded state. No optimized setting was installed or deployed.
