#!/usr/bin/env python3
"""Round-5 patch 4: 2 MiB-aligned, MADV_HUGEPAGE CPU backend buffers (env IK_R5_HUGE=1).

ggml's CPU buffers (KV cache, recurrent state, compute buffer) come from plain malloc, so
transparent huge page coverage is incidental (the K cache was not even page aligned).
With IK_R5_HUGE=1, buffers >= 32 MiB use aligned_alloc(2 MiB) + madvise(MADV_HUGEPAGE);
free() remains valid for aligned_alloc memory, so the buffer interface is unchanged.
"""
import sys, pathlib

p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml-backend.cpp'
s = p.read_text()
assert 'r5_huge' not in s
old = '''GGML_CALL static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
'''
new = '''#if defined(__gnu_linux__)
#include <sys/mman.h>
static bool r5_huge_enabled() {
    static const bool e = [] { const char * s = getenv("IK_R5_HUGE"); return s && atoi(s) > 0; }();
    return e;
}
#endif

GGML_CALL static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = nullptr;
#if defined(__gnu_linux__)
    if (r5_huge_enabled() && size >= ((size_t)32 << 20)) {
        const size_t huge = (size_t)2 << 20;
        const size_t rounded = (size + huge - 1) & ~(huge - 1);
        data = aligned_alloc(huge, rounded);
        if (data) {
            madvise(data, rounded, MADV_HUGEPAGE);
            size = rounded;
        }
    }
#endif
    if (!data) data = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('patched')
