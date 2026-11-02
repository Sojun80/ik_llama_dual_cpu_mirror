# Mission: Aggressive Zen 2 CPU MoE Engine Optimization

You are taking over a successful MiniCPM-V 4.7 35B-A3B CPU inference optimization project on Gigabyte5.

**Your job is to find substantial additional performance through source-level engineering, not another superficial thread-count sweep.**

You have broad authorization to investigate, prototype, modify experimental source code, compile optimized binaries, profile execution, and benchmark alternative designs.

Do not stop at the first disappointing experiment. Follow the evidence and pursue promising architectures.

## Hardware and baseline

- Dual AMD EPYC 7K62, Zen 2, 48 physical cores per socket
- 128 GiB DDR4
- Model: OpenBMB MiniCPM-V 4.7 35B-A3B Q4_K_M
- Runtime: `ik_llama.cpp`
- Target: **One CPU socket, NUMA node 1**
- Decode: 32 threads with explicit per-worker physical-core affinity
- Prompt processing: 48 threads

Preserved, repeatable performance:

| Metric | 2,834 context | 9,455 context |
|---|---:|---:|
| Decode | 34.6 tok/s | 30.7 tok/s |
| Prompt processing | 402 tok/s | 354 tok/s |

Measurements use 512 generated tokens and identical input token IDs.

Previous investigations:

`/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation/`

`/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization/`

Read the reports, source changes, profiles, and failed experiments before beginning.

## Priority 1: Exploit the CCD/CCX topology

Map the actual processor topology, including physical core IDs, CCDs, CCXs, shared caches, memory controllers and thread placement.

Explore alternative MoE execution architectures:

- Dedicated expert worker groups pinned to particular CCXs or CCDs
- Assigning routed experts or expert weight tiles to specific worker groups
- Persistent worker pools that eliminate unnecessary OpenMP wakeups and barriers
- Scheduling independent selected experts concurrently
- Sequential or partially sequential expert execution where reduced cache contention outweighs lost parallelism
- Cache-aware partitioning of expert matrices
- Minimizing intermediate tensor movement across CCX boundaries
- Adaptive work distribution based on selected expert count and matrix dimensions

Do not assume static expert ownership is optimal. Measure routing imbalance, synchronization costs, cache residency and memory traffic.

The goal is to exploit chiplet topology while keeping all model weights resident on NUMA node 1.

## Priority 2: Memory pipelining and prefetch

Investigate whether weight movement and computation can overlap more effectively.

Explore:

- Software prefetching of quantized weight blocks
- Double-buffered or pipelined expert execution
- Cache-aware matrix tiling and data layout
- Prefetching selected experts as soon as routing results are available
- Fusing quantization, dot products and reductions where profitable
- Thread cooperation for staging weight tiles
- Reducing TLB misses, cache evictions and memory stalls
- Alternative work partitioning across memory channels

Investigate existing `ik_llama.cpp` fused MoE, quantized kernels and expert-prefetch implementations before replacing them.

Do not assume DRAM bandwidth is saturated. Use actual counters to distinguish memory stalls, cache misses, compute limitations and synchronization overhead.

Test speculative expert prefetch only if it has a measurable advantage over demand-driven prefetch.

## Priority 3: Rewrite hot execution paths

Profile the full decode path and identify its dominant costs.

You may write custom Zen 2-optimized AVX2/FMA kernels, change memory layouts, introduce specialized execution paths, modify task scheduling or replace expensive synchronization mechanisms.

Investigate:

- Q4_K and Q6_K expert matrix-vector multiplication
- Fused mixture-of-experts computation
- Router dispatch and weighted output reduction
- Thread synchronization and OpenMP barriers
- Attention and KV-cache operations at longer contexts
- Avoidable allocation, copying and graph execution overhead

Preserve the model's actual routing behavior, weights and mathematical operations. Do not achieve speedups by silently dropping experts or degrading precision.

## Priority 4: Explore unconventional execution

I explicitly want architectural experimentation.

Consider persistent per-CCD execution engines, pipelined compute stages, decentralized work queues, asynchronous expert scheduling, tiling across cache domains, and specialized single-token decode execution.

Question whether the current tensor graph execution model imposes unnecessary synchronization.

Can multiple independent expert operations proceed without global barriers?

Can memory access be reorganized so compute workers spend less time waiting?

Can work be scheduled to keep hot data within an individual CCX?

Can we reduce context-dependent decode slowdown?

You have permission to prototype substantial changes in isolated branches. Reject ideas that measurements show are slower.

## Measurements and correctness

Maintain the existing optimized `ik_llama.cpp` configuration as the primary baseline.

Benchmark both 2.8K and 9.4K context using 512 generated tokens, warmed execution and multiple repeated samples.

Measure:
- Sustained decode throughput
- Prompt processing throughput
- End-to-end latency
- CPU cycles and instructions per token
- Cache and TLB misses
- DRAM bandwidth
- OpenMP synchronization and idle time
- NUMA locality and CPU affinity

Control CPU frequency and competing workload effects where possible.

Check numerical differences against the preserved candidate, and use focused output validation to reject meaningful regressions. Different floating-point reduction orders are not automatically incorrect.

Do not run the full logging intelligence evaluation during development.

Only retain optimizations that demonstrate repeatable benefits. Avoid sacrificing long-context performance for a short-context microbenchmark.

## Constraints

- No GPU inference
- Single CPU socket for the primary target
- Preserve all model files, original sources and previous benchmark results
- Do not disturb existing CPU services or GPU mining
- Build and test only in isolated experimental locations
- No production deployment without approval
- Keep reproducible records of successful and failed experiments

## Deliverable

Produce the fastest stable, correctness-checked single-socket inference implementation you can.

Include:

1. Exact source changes and build instructions
2. Repeatable benchmark commands and CPU affinity maps
3. Before/after performance at both context lengths
4. Profiling evidence explaining meaningful speedups
5. Rejected approaches and why they failed
6. Remaining bottlenecks and possible next steps

Do not declare success based on tiny microbenchmark gains that disappear in real inference.

**The goal is not merely to tune `ik_llama.cpp`. The goal is to discover whether a topology-aware MoE execution engine can substantially outperform it on this hardware.**

Be ambitious, technically rigorous, and willing to abandon conventional implementations when the data justifies it.

Proceed independently through safe, reversible experiments. Do not request approval for routine profiling, building, or testing.

**Optimize the hell out of this machine.**
