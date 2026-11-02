#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_STREAMTEST=1. Before r5fa::partial, each thread
  (a) streams its own K and V rows (plain AVX2 loads, all KV heads) and
  (b) streams the same number of bytes from a private, thread-allocated 64 MiB ring buffer
      (always cold: the ring advances each call and 3-4 threads x 64 MiB exceed a CCX's L3),
and reports bytes/tick for both every 640 calls. Then partial runs as usual."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_STREAMTEST' not in s
old = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk,'''
new = '''    static const bool streamtest = [] { auto e = getenv("IK_R5_FA_STREAMTEST"); return e && atoi(e) > 0; }();
    if (streamtest && j1 > j0) {
        static thread_local char * ring = nullptr;
        static thread_local size_t ring_off = 0;
        static thread_local uint64_t tk, tr, bytes, nc;
        constexpr size_t kRing = (size_t)64 << 20;
        if (!ring) { ring = (char *)aligned_alloc(2u << 20, kRing); madvise(ring, kRing, MADV_HUGEPAGE); memset(ring, 1, kRing); }
        const long span = (long)(nek2 - 1)*nbk2 + 2*D;
        __m256 acc = _mm256_setzero_ps();
        const uint64_t a = __rdtsc();
        for (int j = j0; j < j1; ++j) {
            const char * kr = (const char *)k + (long)j*stride_k;
            const char * vr = (const char *)v + (long)j*stride_v;
            for (long o = 0; o < span; o += 32) {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(kr + o)));
                acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(vr + o)));
            }
        }
        const uint64_t b = __rdtsc();
        const size_t n = (size_t)(j1 - j0)*span*2;
        if (ring_off + n > kRing) ring_off = 0;
        for (size_t o = 0; o < n; o += 32) acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
        ring_off += (n + 4095) & ~(size_t)4095;
        const uint64_t c = __rdtsc();
        volatile float sink = _mm256_cvtss_f32(acc); (void)sink;
        tk += b - a; tr += c - b; bytes += n;
        if (++nc % 640 == 0) {
            fprintf(stderr, "r5fa-stream ith %d: KV rows %.2f bytes/tick, private cold ring %.2f bytes/tick (%zu KB/call)\\n",
                    ith, (double)bytes/tk, (double)bytes/tr, n >> 10);
            tk = tr = bytes = 0;
        }
    }
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk,'''
assert s.count(old) == 1
s = s.replace(old, new)
if '#include <sys/mman.h>' not in s:
    s = s.replace('#include <immintrin.h>', '#include <immintrin.h>\n#include <sys/mman.h>', 1)
p.write_text(s)
print('patched')
