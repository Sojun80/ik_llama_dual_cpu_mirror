#!/usr/bin/env python3
"""Round-5 diagnostic v2 for IK_R5_FA_STREAMTEST: fair controls.
  KVi : own K and V rows, interleaved (two streams)            - as before
  KVs : neighbour thread's K rows then its V rows (sequential)  - one stream at a time
  R2  : private cold ring, two interleaved streams half a ring apart
  R1  : private cold ring, one stream
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
a = s.index('        const uint64_t b = __rdtsc();\n        const size_t n = (size_t)(j1 - j0)*span*2;')
b = s.index('    const uint64_t t0 = dbg ? __rdtsc() : 0;')
new = r'''        const uint64_t b = __rdtsc();
        const size_t n = (size_t)(j1 - j0)*span*2;
        if (ring_off + n > kRing/2) ring_off = 0;
        for (size_t o = 0; o < n/2; o += 32) {
            acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
            acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + kRing/2 + ring_off + o)));
        }
        const uint64_t c = __rdtsc();
        for (size_t o = n/2; o < n + n/2; o += 32) acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
        ring_off += (n + n/2 + 4095) & ~(size_t)4095;
        const uint64_t d = __rdtsc();
        const int jj0 = j1 < n_eff ? j1 : 0, jj1 = std::min(n_eff, jj0 + (j1 - j0));
        for (int j = jj0; j < jj1; ++j) { const char * kr = (const char *)k + (long)j*stride_k;
            for (long o = 0; o < span; o += 32) acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(kr + o))); }
        for (int j = jj0; j < jj1; ++j) { const char * vr = (const char *)v + (long)j*stride_v;
            for (long o = 0; o < span; o += 32) acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(vr + o))); }
        const uint64_t e = __rdtsc();
        volatile float sink = _mm256_cvtss_f32(acc); (void)sink;
        static thread_local uint64_t t_kvi, t_r2, t_r1, t_kvs, b_kvi, b_kvs;
        t_kvi += b - a; t_r2 += c - b; t_r1 += d - c; t_kvs += e - d; b_kvi += n; b_kvs += (size_t)(jj1 - jj0)*span*2;
        if (++nc % 640 == 0) {
            fprintf(stderr, "r5fa-stream2 ith %d: KVi %.2f KVs %.2f R2 %.2f R1 %.2f bytes/tick\n", ith,
                    (double)b_kvi/t_kvi, (double)b_kvs/t_kvs, (double)b_kvi/t_r2, (double)b_kvi/t_r1);
            t_kvi = t_r2 = t_r1 = t_kvs = b_kvi = b_kvs = 0;
        }
    }
'''
s = s[:a] + new + s[b:]
p.write_text(s)
print('patched')
