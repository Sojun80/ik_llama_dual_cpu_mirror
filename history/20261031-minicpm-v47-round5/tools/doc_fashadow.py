#!/usr/bin/env python3
# Record the KV-read investigation and IK_R5_FA_SHADOW (2026-11-01/02, runs/95-108) in PROGRESS.md and FA-BOUNDARY.md
import pathlib, sys
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
c = sys.argv[1]

p = root / 'PROGRESS.md'
s = p.read_text()
assert 'IK_R5_FA_SHADOW' not in s
old = 'IK_R5_NONUMAB=1 IK_R5_PF=1 \\\n'
assert s.count(old) == 1
s = s.replace(old, 'IK_R5_NONUMAB=1 IK_R5_PF=1 IK_R5_FA_SHADOW=1 \\\n')
row = '| IK_R5_PF_LINES default 2 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + (f'| {c} | IK_R5_FA_SHADOW=1 | decode FA reads each thread\'s K/V rows from a private per-thread copy, '
             'kept current through a log of KV writes (mirror mode only) | bit-exact (runs/106, KV reuse runs/107); '
             '+1.7% 2.8K, +10.2% 9.4K (runs/108) |\n') + s[a:]
s += f'''
## KV reads in the decode FA (2026-11-01/02): IK_R5_FA_SHADOW=1, commit {c}
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
'''
p.write_text(s)

p = root / 'FA-BOUNDARY.md'
s = p.read_text()
assert 'IK_R5_FA_SHADOW' not in s
s += f'''
## Resolved in practice (2026-11-02): read KV from per-thread private copies (IK_R5_FA_SHADOW, commit {c})
With casebook stopped the node-0 difference is gone (runs/95). The decode FA reads KV at a fixed ~20-25 GB/s per node
for 8 to 48 readers per node (runs/97, runs/98), and the same rows copied into memory the reading thread allocated and
wrote read 2.0-2.4x faster (runs/100-102). Remote writers, line ownership, THP and node placement are ruled out
(runs/101-105). IK_R5_FA_SHADOW=1 keeps such copies current through a log of KV writes: bit-exact, +10.2% decode at
9.4K. The hardware reason is still open (most likely DRAM bank/row conflicts of the 64 streams in shared pages).
Details in PROGRESS.md.
'''
p.write_text(s)
print('docs updated')
