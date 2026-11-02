# MiniCPM-V 4.7 CPU inference investigation — Gigabyte5

2026-10-31. Experimental work only; no deployment. Model `/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf`, SHA-256 `9ecf91e814d7e9adc64325b1b65f8628a9b1bb8da4f61a92869ca7489bfd888e`, reverified after testing. Upstream reference commit `0ec4027a2a0183de87c007069adddc06e665f7de`; ik commit `13cffc006b82e03e4d89cbb8c7327f0255ae3d18`, CPU-only server binary SHA-256 `e6c898de1e764d8f158f1f9d3aeede76dcb15aa1d571e0a3ad4332b25e35067c`. Existing casebook server and GPU-only miner were left running.

## Outcome and acceptance limit

The fastest reproducible single-node candidate is ik with 32 decode threads, 48 prompt threads, explicit per-worker CPU affinity, node-1 memory, `--no-mmap`, and `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`. The exact tested command is `corpus-ik-command.json`, ordered CPU list `scaling-order.json`. It is an experimental candidate: repeated synthetic performance and selected operator checks pass, but full-model numerical equivalence has not been established and varied long-prompt greedy output differs from upstream. Do not deploy or present it as correctness-qualified based on these tests.

Two repetitions each on identical fixed varied inputs, temperature zero, 512 generated tokens, no prompt cache, no log evaluation. Both runtimes received identical input token IDs; timing from their own servers. Upstream used its preserved 16-thread decode/48-thread batch configuration; ik used 32/48. Context capacity 16384 in both. `predicted_per_second` is each runtime's reported convention (upstream reports per-token timing over 511 intervals while predicted_n is 512; this is a small convention difference). These are not matched by thread count, but are matched to the preserved configuration and same input/output budget.

| prompt tokens | engine | decode tok/s, two samples | prompt tok/s, two samples |
|---:|---|---|---|
| 2834 | upstream | 18.425, 18.423 | 229.71, 226.64 |
| 2834 | ik 32 | 34.574, 34.673 | 401.14, 401.65 |
| 9455 | upstream | 15.973, 15.902 | 202.97, 205.10 |
| 9455 | ik 32 | 30.760, 30.723 | 354.06, 354.01 |

Mean gains: decode ~88% at 2.8K and ~93% at 9.4K; prompt ~76% and ~74%. Each model's repeats produced identical text to itself. Cross-runtime text matched for the 2.8K corpus, but differed on the 9.4K corpus. Earlier very repetitive 64-token smoke matched at both lengths. These tests do not establish accuracy for downstream log analysis, multimodal inputs, different templates, longer contexts, or arbitrary prompts. Full 32768 context and checkpoint-disabled recurrent multi-turn behavior were not qualified.

## Stall cause

The previous apparent long-context hang is an extreme scheduling pathology rather than evidence of a wrong result. In a clean 9436-token request, generation did finish but first tokens were anomalously slow. Another 2812-token request showed ~4.4 seconds per token before spontaneously recovering to ~29 ms/token. A ten-second perf capture during the slow phase attributed 97.3% of sampled cycles to a libgomp spin loop, with graph compute barriers in the call stacks; two active workers were sharing a physical core. The prior server used a process CPU set only, without per-worker affinity.

A deliberate isolated collision test pinned two of 16 compute workers to one core. Identical short output: distinct cores 44.47 tok/s; collision 0.26 tok/s; same collision with `OMP_WAIT_POLICY=PASSIVE` and `GOMP_SPINCOUNT=0` 19.77 tok/s. This controlled test establishes that a worker collision can cause the observed stall; it does not prove every slow interval had the same cause. Clean start with ik `--cpu-range` explicitly ordering workers on distinct physical cores eliminated the stall in the repeated runs. Separately, passive waiting without collision gave only ~10–11 tok/s at 2.8K/9.4K; spin-count 100 gave ~12/11.5, so neither is recommended for the fast pinned configuration.

## Thread scaling and numerical checks

A one-sample-per-setting screening with the same repetitive input and 64 generated tokens produced 24 threads 36.78/31.98, 32 threads 37.29/32.18, and 48 threads 36.97/31.47 tok/s at 2.8K/9.4K. This selected 32 for the repeated varied-input 256- and 512-token tests. Clean pinned 16-thread repeated samples reached 34.65–35.09/30.06–30.78. Selection is workload-specific, not an exhaustive optimum.

Cross-runtime long-output mismatch is real but is not, by itself, proof of a bug: upstream 16 vs upstream 32 also diverged after a common 13-token prefix on the same long varied input. The unmodified ik and upstream agreed through the first 13 generated tokens there, while their top-token probabilities differed already at token 0. Disabling fused MoE or fused up/gate did not resolve it. Disabling flash attention changed the early greedy output but also reduced long decode to ~22.5 tok/s and did not establish matching probabilities. The F32 KV plus ik flash-attention test aborted in `iqk_flash_attn.cpp`; that configuration is rejected. `--ctx-checkpoints 0` affected probabilities and gave ~33.4–33.5/29.98–30.03 tok/s, but did not match the full long output; recurrent multi-turn semantics remain untested.

Direct operator checks use real `blk.0.ffn_gate_exps.weight` Q4_K and `blk.0.ffn_down_exps.weight` Q6_K expert matrices from the same GGUF, matrix times three deterministic float vectors, compared to float dot products of dequantized stored weights. These test only individual 2D operators. Normalized RMSE for Q4_K: upstream 0.00349/0.02673/0.02661, ik 0.00230/0.00965/0.01246. Q6_K: upstream 0.00385/0.01866/0.02972, ik 0.00289/0.00824/0.01333. ik uses Q8_2 activation quantization on AVX2 while upstream uses Q8_K; a Q8_K diagnostic build reproduced the upstream Q4 operator errors. That path reached only ~21/19.5 tok/s in 256-token runs and still diverged from upstream in the long output; it is a diagnostic, not a promoted fix. Output and sampling differences can result from multiple operations and recurrent state; these tests do not isolate a full-model error.

## NUMA expert placement

The GGUF contains 40 layers, 256 experts per layer and eight selected per token; 120 combined expert tensors. A separate prototype applies `mbind` before loading the two halves of each expert tensor and routes expert IDs 0–127 to 16 node-0 workers and 128–255 to 16 node-1 workers during single-token decode. The initial prototype failed safely due to an incorrectly sized node mask; the v2 patch corrected it. Readback verified ~9058 MiB node 0 and ~11646 MiB node 1, 120 bind records, ~19.6 GiB AnonHugePages, and both experimental scheduler paths active.

At 2.8K/9.4K, single-socket ik 32 screening reached 37.29/32.18 tok/s. Split experts with the original global scheduler reached 24.00/21.01; split experts plus socket-local scheduling reached 24.71/21.47. The latter matched unmodified ik output on the varied 256-token corpus at both lengths, but is a prototype, not a general correctness test. A profiled medium decode attributed ~38% of sampled cycles to libgomp spin versus ~24% for the single-node sample; conditions and sampled durations differ, so this supports but does not prove synchronization as the sole cause. Node 0 shared the host with the existing CPU service; its cores 0–15 were excluded, but cross-socket and memory contention remain. This placement/scheduler is rejected for current latency goals. Earlier `numactl --interleave` attempts were invalid as split tests because model pages still resided almost entirely on node 1; the small anonymous-memory probe did split evenly.

## Other rejected experiments

- Expert task fragmentation patch (`chunk4.patch`): 35.19–35.22/31.04–31.17 tok/s versus unpatched 35.15–35.25/31.25–31.33, 256-token samples; no meaningful improvement and slightly slower long case, with identical ik output.
- Runtime repacking: 32.78–32.89/29.10–29.13 decode and ~353/315 prompt tok/s, worse than default ik; rejected.
- No-checkpoints and Q8_K diagnostic results are above. Spin/passive results are above. All scripts, exact commands, raw streams, perf profiles, failure logs, prototype patches, and memory maps remain in this directory.

## Safety and reproducibility

Reference command and report are copied here; the prior optimization directory retains its server snapshot. No model file or active service was changed. The test server used temporary loopback ports and was stopped after each experiment. No system-wide NUMA or huge-page setting was modified. The uncore module remained unloaded as found. Results are local to Gigabyte5 and its observed concurrent workload. Experimental source and binaries are isolated in `ik-chunk4-*`, `ik-q8k-*`, and `ik-numa-*`; none is deployed. `sustained-controller.log`, per-run `sustained-*.json`, `collision-controller.log`, `opcheck*-results.txt`, `numa-v2-controller.log`, and `numa-proto-v2-*-numastat.txt` are primary evidence.
