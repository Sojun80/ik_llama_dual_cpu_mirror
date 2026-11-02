#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_FLUSH=1 evicts each thread's K/V rows (clflushopt) right after the FA partial pass,
so the next token reads them clean and uncached. Tests whether stale coherence state of the re-read KV lines
(directory probes) is the in-model KV fetch penalty. The flush cost is inside the FA node time."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_FLUSH' not in s
old = '''            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
    const uint64_t t1 = dbg ? __rdtsc() : 0;
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    static const int flush = [] { auto e = getenv("IK_R5_FA_FLUSH"); return e ? atoi(e) : 0; }();
    if (flush > 0 && j1 > j0) {
        // diagnostic: evict this thread's K/V rows (all KV heads) from every cache
        const long fk = (long)(nek2 - 1)*nbk2 + 2*D, fv = (long)(nek2 - 1)*nbv2 + 2*D;
        for (int j = j0; j < j1; ++j) {
            const char * kr = (const char *)k + (long)j*stride_k, * vr = (const char *)v + (long)j*stride_v;
            for (long o = 0; o < fk; o += 64) asm volatile("clflushopt %0" : "+m"(*(volatile char *)(kr + o)));
            for (long o = 0; o < fv; o += 64) asm volatile("clflushopt %0" : "+m"(*(volatile char *)(vr + o)));
        }
        _mm_sfence();
    }
''')
p.write_text(s)
print('patched')
