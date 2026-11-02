#!/usr/bin/env python3
"""Round-5 patch 2: hybrid static/dynamic row scheduling for single-token dense matmuls.

Env IK_R5_DYN=1 enables it (default off, so one binary serves A/B runs).
  IK_R5_DYN_STATIC = percent of each thread's rows kept as a contiguous static block (default 75)
  IK_R5_DYN_CHUNK  = minimum bytes of weights per dynamic chunk (default 16384)
The remaining rows form a pool claimed in chunks via a per-node atomic counter, so threads
that finish their static block early absorb the slow threads' share.

Also adds per-thread MUL_MAT compute totals to the r5 profiler dump.
"""
import sys, pathlib

root = pathlib.Path(sys.argv[1])
ggml_c = root / 'ggml/src/ggml.c'
iqk_cpp = root / 'ggml/src/iqk/iqk_mul_mat.cpp'
iqk_h = root / 'ggml/src/iqk/iqk_mul_mat.h'

def patch(path, pairs):
    s = path.read_text()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (path.name, old[:70], n)
        s = s.replace(old, new)
    path.write_text(s)

# ---------------- iqk header
patch(iqk_h, [(
'''IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,''',
'''// round-5: set (per thread) the atomic counter used for dynamic row chunks of the next
// 2D single-column iqk_mul_mat call; NULL disables. counter must be zero before first use.
IQK_API void iqk_r5_set_dyn(void * counter);

IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,''')])

# ---------------- iqk implementation
patch(iqk_cpp, [(
'''extern "C" IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    constexpr int k_min_step = 32;

    MulMat mm;

    size_t row_size_qx = strideA; //*ggml_type_size(ggml_type(typeA));
    size_t row_size_qy = strideB; //*ggml_type_size(ggml_type(typeB));
''',
'''static thread_local std::atomic<int> * r5_dyn_counter = nullptr;

extern "C" IQK_API void iqk_r5_set_dyn(void * counter) {
    r5_dyn_counter = (std::atomic<int> *)counter;
}

namespace {
struct R5DynConfig {
    int static_pct = 75;
    long min_chunk_bytes = 16384;
    R5DynConfig() {
        if (auto s = getenv("IK_R5_DYN_STATIC")) static_pct = std::max(0, std::min(100, atoi(s)));
        if (auto s = getenv("IK_R5_DYN_CHUNK"))  min_chunk_bytes = std::max(1L, atol(s));
    }
};
const R5DynConfig & r5_dyn_config() { static R5DynConfig c; return c; }
}

extern "C" IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    constexpr int k_min_step = 32;

    MulMat mm;

    size_t row_size_qx = strideA; //*ggml_type_size(ggml_type(typeA));
    size_t row_size_qy = strideB; //*ggml_type_size(ggml_type(typeB));

    if (r5_dyn_counter && Ny == 1 && nth > 1 && MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        const long align = MulMat::num_rows(ggml_type(typeA));
        if (Nx % align == 0) {
            const auto & cfg = r5_dyn_config();
            const long units  = Nx/align;
            const long s      = units*cfg.static_pct/(100*nth);          // static units per thread
            const long pool0  = s*nth;
            long chunk = (units - pool0)/(4*nth);
            const long min_units = (cfg.min_chunk_bytes + align*row_size_qx - 1)/(align*row_size_qx);
            if (chunk < min_units) chunk = min_units;
            if (chunk < 1) chunk = 1;
            if (s > 0) {
                const long ix = ith*s*align;
                DataInfo info{C + ix, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};
                mm.mul_mat_NxM(ne00, (const char *)A + ix*row_size_qx, row_size_qx, info, s*align, Ny);
            }
            auto counter = r5_dyn_counter;
            while (true) {
                const long first = pool0 + (long)counter->fetch_add(1, std::memory_order_relaxed)*chunk;
                if (first >= units) break;
                const long n  = std::min(chunk, units - first);
                const long ix = first*align;
                DataInfo info{C + ix, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};
                mm.mul_mat_NxM(ne00, (const char *)A + ix*row_size_qx, row_size_qx, info, n*align, Ny);
            }
            return true;
        }
    }
'''),
]
)
s = iqk_cpp.read_text()
if '#include <atomic>' not in s:
    s = s.replace('#include "iqk_mul_mat.h"', '#include "iqk_mul_mat.h"\n#include <atomic>\n#include <cstdlib>', 1)
    assert '#include <atomic>' in s
iqk_cpp.write_text(s)

# ---------------- ggml.c
patch(ggml_c, [
# globals, ahead of their first use in ggml_compute_forward_mul_mat
('''static int ggml_compute_forward_mul_mat(
''',
'''// ---- round-5 dynamic row scheduling for single-token dense matmuls (env IK_R5_DYN=1) -------
#define R5D_MAX_NODES 16384
#define R5D_STRIDE    16          // ints per counter: one cache line each
static int          r5d_enabled = -1;
static volatile int r5d_active  = 0;
static atomic_int * r5d_counters;

static void r5d_begin(const struct ggml_cgraph * cgraph) {
    if (r5d_enabled < 0) {
        const char * e = getenv("IK_R5_DYN");
        r5d_enabled = e && atoi(e) > 0;
        if (r5d_enabled) {
            r5d_counters = (atomic_int *)ggml_aligned_malloc(sizeof(atomic_int)*R5D_STRIDE*R5D_MAX_NODES);
            if (!r5d_counters) r5d_enabled = 0;
        }
    }
    r5d_active = r5d_enabled && cgraph->n_batch == 1 && cgraph->n_nodes <= R5D_MAX_NODES;
    if (r5d_active) memset((void *)r5d_counters, 0, sizeof(atomic_int)*R5D_STRIDE*cgraph->n_nodes);
}

static inline void r5d_set(int node_n, const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
#if GGML_USE_IQK_MULMAT
    const bool ok = r5d_active && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                    src0->ne[2] == 1 && src0->ne[3] == 1;
    iqk_r5_set_dyn(ok ? (void *)(r5d_counters + (size_t)node_n*R5D_STRIDE) : NULL);
#else
    (void)node_n; (void)src0; (void)src1;
#endif
}

static inline void r5d_clear(void) {
#if GGML_USE_IQK_MULMAT
    iqk_r5_set_dyn(NULL);
#endif
}

static int ggml_compute_forward_mul_mat(
'''),
# first direct call (F32 x F32 etc.)
('''#if GGML_USE_IQK_MULMAT
    if (dst->type == GGML_TYPE_F32) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, nb12, nb13, nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    src1->type, src1->data, nb11,
                    (float *)dst->data, nb1/sizeof(float), ith, nth)) return node_n;
    }
#endif
''',
'''#if GGML_USE_IQK_MULMAT
    if (dst->type == GGML_TYPE_F32) {
        r5d_set(node_n, src0, src1);
        const bool done = iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, nb12, nb13, nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    src1->type, src1->data, nb11,
                    (float *)dst->data, nb1/sizeof(float), ith, nth);
        r5d_clear();
        if (done) return node_n;
    }
#endif
'''),
# quantized path + fused chain
('''    if (src1->type != vec_dot_type && dst->type == GGML_TYPE_F32) {
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);
        if (iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, row_size*ne11, row_size*ne11*ne12,
                    nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    vec_dot_type, wdata, row_size,
                    (float *)dst->data, nb1/sizeof(float), ith, nth)) {
            if (!cgraph) return node_n;''',
'''    if (src1->type != vec_dot_type && dst->type == GGML_TYPE_F32) {
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);
        r5d_set(node_n, src0, src1);
        const bool done = iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, row_size*ne11, row_size*ne11*ne12,
                    nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    vec_dot_type, wdata, row_size,
                    (float *)dst->data, nb1/sizeof(float), ith, nth);
        r5d_clear();
        if (done) {
            if (!cgraph) return node_n;'''),
('''                if (!iqk_mul_mat_4d(src0_next->ne[1], ne11, ne00,
                    src0_next->ne[2], src0_next->ne[3], ne12, ne13, src0_next->nb[2], src0_next->nb[3], row_size*ne11, row_size*ne11*ne12,
                    dst_next->nb[2]/sizeof(float), dst_next->nb[3]/sizeof(float),
                    src0_next->type, src0_next->data, src0_next->nb[1],
                    vec_dot_type, wdata, row_size,
                    (float *)dst_next->data, dst_next->nb[1]/sizeof(float), ith, nth)) break;
                ++node_n;''',
'''                r5d_set(node_n + 1, src0_next, src1);
                const bool done_next = iqk_mul_mat_4d(src0_next->ne[1], ne11, ne00,
                    src0_next->ne[2], src0_next->ne[3], ne12, ne13, src0_next->nb[2], src0_next->nb[3], row_size*ne11, row_size*ne11*ne12,
                    dst_next->nb[2]/sizeof(float), dst_next->nb[3]/sizeof(float),
                    src0_next->type, src0_next->data, src0_next->nb[1],
                    vec_dot_type, wdata, row_size,
                    (float *)dst_next->data, dst_next->nb[1]/sizeof(float), ith, nth);
                r5d_clear();
                if (!done_next) break;
                ++node_n;'''),
# graph begin
('''#ifdef R5P
    r5p_begin(cgraph, n_threads);
#endif
''',
'''#ifdef R5P
    r5p_begin(cgraph, n_threads);
#endif
    r5d_begin(cgraph);
'''),
# ---- profiler: per-thread MUL_MAT compute
('''static uint64_t          r5p_tsc_init;
''',
'''static uint64_t          r5p_tsc_init;
static double            r5p_thr_mm[R5P_MAX_THREADS], r5p_thr_all[R5P_MAX_THREADS];
static int               r5p_thr_cpu[R5P_MAX_THREADS];
'''),
('''        r5p_graphs = 0; r5p_sum_graph = r5p_sum_fork = r5p_sum_join = r5p_sum_nodes = 0;
        r5p_resets++;''',
'''        r5p_graphs = 0; r5p_sum_graph = r5p_sum_fork = r5p_sum_join = r5p_sum_nodes = 0;
        memset(r5p_thr_mm, 0, sizeof(r5p_thr_mm)); memset(r5p_thr_all, 0, sizeof(r5p_thr_all));
        r5p_resets++;'''),
('''            const uint64_t c = r->t1 - r->t0;
            if (c > maxc) maxc = c;''',
'''            const uint64_t c = r->t1 - r->t0;
            r5p_thr_all[t] += (double)c;
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) r5p_thr_mm[t] += (double)c;
            if (c > maxc) maxc = c;'''),
('''    fprintf(f, "\\n## by node (us per executed instance)\\n");''',
'''    fprintf(f, "\\n## per thread (us per graph): compute in dense MUL_MAT nodes, compute in all nodes\\n");
    for (int t = 0; t < r5p_nth; ++t) {
        fprintf(f, "thr %2d cpu %3d  mul_mat %9.1f  all %9.1f\\n", t, r5p_thr_cpu[t], r5p_thr_mm[t]/G/tpu, r5p_thr_all[t]/G/tpu);
    }
    fprintf(f, "\\n## by node (us per executed instance)\\n");'''),
('''static void r5p_end(const struct ggml_cgraph * cgraph, int nth) {
    if (!r5p_active) return;''',
'''static void r5p_end(const struct ggml_cgraph * cgraph, const struct ggml_cplan * cplan, int nth) {
    if (!r5p_active) return;
    for (int t = 0; t < nth; ++t) {
        r5p_thr_cpu[t] = cplan->cpu_affinity && cplan->n_cpu_affinity > 0 ? cplan->cpu_affinity[t % cplan->n_cpu_affinity] : -1;
    }'''),
('''    r5p_end(cgraph, state_shared.n_threads);''', '''    r5p_end(cgraph, cplan, state_shared.n_threads);'''),
])
print('patched')
