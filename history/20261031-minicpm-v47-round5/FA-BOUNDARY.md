# Flash attention "integration boundary" investigation (round 5, 2026-10-31)

Question: why does the r5fa decode kernel run 2-3x slower per phase inside llama-server than in the
standalone benchmark with identical shapes and strides, and not speed up from 32 to 64 threads?

## Answer so far
It is not dispatch, graph preparation, tensor conversion or synchronization around the kernel. It is the
first DRAM fetch of the K/V rows, which is ~3x less efficient in the server than reading other memory.

Evidence (all in-model unless noted, single socket, 9.4K context):
- Phase timers inside the kernel: the softmax phase (stack data only) runs at bench speed; scores, V and
  setup (all touching heap data) are 2-3x slower. (runs/22, 23, 24)
- Double pass (IK_R5_FA_TWICE): the second pass over the same, now cache-resident data takes ~280K ticks vs
  ~1.5M for the first - compute and cache behaviour equal the bench. (runs/33)
- IBS (perf ibs_op, latency-weighted): FA DRAM loads wait 3,529 cycles on average (median 1,627, a cluster at
  8-10.5K cycles) vs 449 for matmul DRAM loads in the same process and 596 for the same kernel in the bench;
  prefetcht1 ops sit ~10K cycles. (runs/31, runs/32)
- Plain-load stream test in the kernel (IK_R5_FA_STREAMTEST): own K/V rows 0.41-0.47 bytes/tick per thread on
  4-thread CCXs (about 5 GB/s per CCX regardless of thread count), vs 1.4-1.7 for a private cold buffer,
  ~1.0-1.2 for a private buffer read as two interleaved streams. (runs/34-36)
- Order matters: a private buffer read FIRST after the op boundary drops to 0.72-0.78 bytes/tick, and KV read
  after that warm-up improves 0.47 -> 0.57. Part of the loss is an op-boundary ramp after light ops. (runs/39)

Ruled out: NUMA-balancing hint faults / page faults (0 during decode, runs/30); TLB (IBS: TLB L1 hits);
huge pages (KV buffer 99.9% THP, IK_R5_HUGE no change); physical placement (KV and weights in the same
64-96 GiB physical range, tools/pfnmap.py); 16 MiB K->V spacing (bench, no effect); dirty-line writebacks
between calls (bench FLUSH_WRITE, no effect); multi-writer coherence (bench INIT_PAR: faster, not slower);
re-read recency (fixed private chunk stays fast, runs/38); access pattern (two-pass kernel K-stream then
V-stream, IK_R5_FA_2PASS: in-model no better, runs/37); denormals (IK_R5_FTZ); prefetch distance.

Open hypotheses: data-fabric / memory power-state ramp after the light small-op phase (would need BIOS
APBDIS / fixed DF P-state to test), and an unidentified KV-specific factor (KV reads stay slower than a
private buffer even after warm-up).

## Small single-threaded ops
451 effectively single-threaded nodes cost 2.0 ms/token mirrored (271 nodes, 1.0 ms single socket). Their cost
is mostly pulling data other cores just wrote plus the barrier; arithmetic is ~1 us per node.
- MUL_MULTI_ADD column split (IK_R5_MMA_COLS, default on, bit-exact): 269 -> 199 us/token single socket,
  383 -> 326 us mirrored. (runs/40)
- Remaining: router softmax+top-k (~0.16-0.20 ms), FFN-input RMS norm (~0.15-0.17 ms), residual ADDs
  (barrier-tail dominated), conv SiLU. Real removal needs fusion so the data never leaves the producing core,
  e.g. a fused MoE down-projection + weighted sum (+ residual) where each thread owns output rows for all
  selected experts. Upper bound for all small ops: ~4% single socket, ~8% mirrored.
- Fused MoE tail (IK_R5_FMOE, bit-exact): routed down + weighted sum + residual + shared expert down + gate + add as one
  node with per-thread output-row ownership. Tail 4.54 -> 4.00 ms/token single socket, 4.33 -> 3.65 mirrored; decode +4.4%
  mirrored at 2.8K, +0.8..2.4% elsewhere (noisy). Details in PROGRESS.md.
- Routed + shared up/gate folded into the fused tail (IK_R5_FMOE_UG, bit-exact): 9 nodes and 7 barriers -> 1 node and
  4 barriers per MoE layer. Mirrored decode +4.8% at 2.8K and +4.0% at 9.4K over IK_R5_FMOE alone; single socket within
  noise. Details in PROGRESS.md.
- Router folded into the fused MoE node as well (IK_R5_FMOE_UG=2, bit-exact): 8 -> 4 barriers per MoE layer but no
  measurable decode gain; the per-thread repeated norm/top-k offsets the saved barriers and the f32 router logits
  matmul dominates. Opt-in only; details in PROGRESS.md.

## FA cold fetch, continued (2026-11-01)
Single socket the FA read pass streams K+V at ~1.26 GB/s per thread, ~40 GB/s for 32 threads (9.4K: 608 KiB per thread
in ~480 us). Plain streaming with tools/bwprobe (2026-11-01, two rounds each): node 1 baseline-32 layout 80-82 GB/s,
node 1 all 48 cores 94-95, node 0 cores 16-47 (4 CCDs, 0-15 are casebook's) 72-73, node 1 cores 48-79 (4 CCDs) 69-71.
So in-model FA reads KV at about half of what the same 32-thread layout streams. (Corrected: an earlier version of this
note compared against a 4-CCD placement and said 60-65%.)
Ruled out (each with an A/B run):
- Fabric/memory power-state ramp: a heater thread on spare cpu 95 (tools/memheat.c: one random line per 1 us or
  0.2 us, or a 7 GB/s stream) leaves FA unchanged (runs/54-heat-*).
- Stale coherence state of re-read KV lines: clflushopt of each thread's rows after the pass makes the next read pass
  slower, not faster (2.8K 454K -> 519K, 9.4K 1.25M -> 1.40M ticks; runs/55, runs/56, tools/apply_r5faflush.py).
- Lockstep channel/bank hotspot from the fixed 0x4C000 spacing: an irregular rotated start per thread gives the same
  read time (runs/57, tools/apply_r5farot.py).
- Uneven channel interleave: DIMMs balanced (dmidecode).
Mirror mode findings:
- FA reads the node-local KV copies correctly, but automatic NUMA balancing was active in the process: 6-12K page faults
  and 200-500 migrations per 480 decode tokens (runs/61, perf), on the shared heap page with thread-pool/barrier state,
  the work buffer and the compute buffer (runs/62, tools/faultmap.py); single socket has 0 because numactl --membind
  sets an explicit policy. IK_R5_NONUMAB=1 (commit 4ab938e6) sets an explicit MPOL_LOCAL task policy: faults and
  migrations ~0, bit-exact, decode unchanged within noise (runs/63-65). Primary-copy KV pages also drifted to node 1
  before (runs/59).
- Node-0 threads read their KV ~1.5-1.7x slower per position than node-1 threads with the same CCX occupancy
  (IK_R5_FA_ONENODE diagnostic, runs/66, tools/fa_ccx.py), and both sockets at once slow node 1 by ~60%. Node 0's
  raw streaming equals node 1's (bwprobe). Writing each KV copy only from its own node (tools/apply_r5kvlocal.py,
  runs/67) was inconclusive: mirrored FA time varies 20-30% between server restarts, more than the effect tested.
Open: why node 0 (and both sockets together) stream KV slower in-model; FA would also profit from more streaming
threads than the 32 the matmuls use (node 1 has 16 idle cores during decode), which needs a separate FA thread set.

## Resolved in practice (2026-11-02): read KV from per-thread private copies (IK_R5_FA_SHADOW, commit 2f69e5ee)
With casebook stopped the node-0 difference is gone (runs/95). The decode FA reads KV at a fixed ~20-25 GB/s per node
for 8 to 48 readers per node (runs/97, runs/98), and the same rows copied into memory the reading thread allocated and
wrote read 2.0-2.4x faster (runs/100-102). Remote writers, line ownership, THP and node placement are ruled out
(runs/101-105). IK_R5_FA_SHADOW=1 keeps such copies current through a log of KV writes: bit-exact, +10.2% decode at
9.4K. The hardware reason is still open (most likely DRAM bank/row conflicts of the 64 streams in shared pages).
Details in PROGRESS.md.
