#!/usr/bin/env python3
"""IK_R5_STREAM_ORDER=1: in the FA stream test, read a cold 2-stream ring chunk FIRST (timed as
'pre'), then the KV rows, so we can tell an op-boundary effect from a KV-memory effect."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
old = '''        const long span = (long)(nek2 - 1)*nbk2 + 2*D;
        __m256 acc = _mm256_setzero_ps();
        const uint64_t a = __rdtsc();'''
new = '''        const long span = (long)(nek2 - 1)*nbk2 + 2*D;
        __m256 acc = _mm256_setzero_ps();
        static const bool ring_first = [] { auto e = getenv("IK_R5_STREAM_ORDER"); return e && atoi(e) > 0; }();
        if (ring_first) {
            static thread_local uint64_t tpre, bpre, npre;
            const size_t m = (size_t)(j1 - j0)*span*2;
            if (ring_off + m > kRing/2) ring_off = 0;
            const uint64_t x = __rdtsc();
            for (size_t o = 0; o < m/2; o += 32) {
                acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
                acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + kRing/2 + ring_off + o)));
            }
            tpre += __rdtsc() - x; bpre += m;
            ring_off += (m/2 + 4095) & ~(size_t)4095;
            if (++npre % 640 == 0) { fprintf(stderr, "r5fa-pre ith %d: ring-first 2-stream %.2f bytes/tick\\n", ith, (double)bpre/tpre); tpre = bpre = 0; }
        }
        const uint64_t a = __rdtsc();'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('patched')
