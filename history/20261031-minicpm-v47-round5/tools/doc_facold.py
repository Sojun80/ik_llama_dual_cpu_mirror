#!/usr/bin/env python3
# record the FA cold-fetch investigation (2026-11-01) in FA-BOUNDARY.md and PROGRESS.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')

fa = '''
## FA cold fetch, continued (2026-11-01)
Single socket the FA read pass streams K+V at ~1.26 GB/s per thread, ~40 GB/s for 32 threads (9.4K: 608 KiB per thread
in ~480 us). tools/bwprobe on this box streams only 64-72 GB/s with 32 threads (~2.1 GB/s per thread; 16 x 4 GiB
single-rank DDR4-2666 per socket, all 8 channels populated evenly), so in-model FA runs at ~60-65% of plain streaming;
the earlier "3x slower" was against private buffers that were partly cache-assisted.
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
'''
p = root / 'FA-BOUNDARY.md'
p.write_text(p.read_text() + fa)

p = root / 'PROGRESS.md'
s = p.read_text()
row = '| 96a11658 | IK_R5_FMOE_UG=2 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + ('| 4ab938e6 | IK_R5_NONUMAB=1 | explicit MPOL_LOCAL task policy: process opts out of automatic NUMA balancing '
             '(no system change) | bit-exact; mirror decode faults/migrations -> ~0, speed unchanged |\n') + s[a:]
s += '''
## FA cold fetch (2026-11-01): see FA-BOUNDARY.md
Power-state ramp, stale coherence of re-read KV lines, lockstep channel hotspots and uneven interleave are ruled out
by A/B runs. In-model FA streams KV at ~60-65% of this box's plain 32-thread streaming rate. Mirror mode had automatic
NUMA balancing faulting on shared runtime pages; IK_R5_NONUMAB=1 removes it (bit-exact) without a measurable decode
gain. Node-0 threads read KV ~1.5-1.7x slower than node-1 threads in mirror mode; cause still open.
'''
p.write_text(s)
print('docs updated')
