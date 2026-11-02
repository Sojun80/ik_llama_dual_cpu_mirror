#!/usr/bin/env python3
"""Round-5 patch 19: IK_R5_PF=1 prefetches upcoming weights while a thread waits at a barrier.

The decode profile (runs/49, m64 + IK_R5_FMOE_UG=1) shows each thread idle ~5.5 ms of a ~22 ms token, ~1.07 ms of it
in the delta-net step (few busy threads), while the weight reads themselves run below streaming speed. With IK_R5_DYN=1
every thread owns a fixed static share of each single-token dense matmul (75% of the rows, iqk_mul_mat's dynamic path)
before the dynamic remainder, so its next reads are known in advance.

- At every node boundary (before the executor's barrier) each thread queues the address ranges of its static rows
  in the next dense MUL_MAT nodes (lookahead window IK_R5_PF_WIN nodes, default 32) up to IK_R5_PF_BUDGET bytes of
  not-yet-prefetched data (default 256 KiB, half the Zen 2 L2), from its node-local mirror copy.
- In r5_fmug (IK_R5_FMOE_UG=1), as soon as the routed experts are known, each thread replaces its queue with its own
  P3 down rows (the 8 routed experts and the shared expert, its fixed output-row slice), consumed during the step's
  internal barrier waits.
- Every spin iteration of the NUMA barriers (hierarchical barrier and IK_R5_FMOE_LOCAL's node barrier) issues
  IK_R5_PF_LINES (default 4) prefetcht1 (L2) of the queue before its pause.
Prefetches change no result, so output is bit-identical by construction.
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])

# --- iqk: the static rows of the dynamic single-token path, exported for the planner ---
p = root / 'ggml/src/iqk/iqk_mul_mat.cpp'
s = p.read_text()
assert 'iqk_r5_dyn_static_rows' not in s
anchor = 'const R5DynConfig & r5_dyn_config() { static R5DynConfig c; return c; }\n}\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
// round-5 (IK_R5_PF): the static rows iqk_mul_mat's dynamic single-token path gives thread ith (same formula)
extern "C" IQK_API bool iqk_r5_dyn_static_rows(int typeA, int typeB, int typeB2, long ne00, long Nx, int ith, int nth,
        long * first, long * nrows) {
    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, 1) && !MulMat::prepare(typeA, typeB2, ne00, mm, 1)) return false;
    const long align = MulMat::num_rows(ggml_type(typeA));
    if (nth < 2 || align < 1 || Nx % align) return false;
    const long s = (Nx/align)*r5_dyn_config().static_pct/(100*nth);
    *first = ith*s*align;
    *nrows = s*align;
    return s > 0;
}
''')
p.write_text(s)

p = root / 'ggml/src/iqk/iqk_mul_mat.h'
s = p.read_text()
anchor = 'IQK_API void iqk_r5_set_dyn(void * counter);\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
// round-5 (IK_R5_PF): rows [*first, *first + *nrows) of an Nx-row single-column matmul that the dynamic path
// assigns statically to thread ith; typeB2 is tried if typeA x typeB is unsupported; false if none
IQK_API bool iqk_r5_dyn_static_rows(int typeA, int typeB, int typeB2, long ne00, long Nx, int ith, int nth,
        long * first, long * nrows);
''')
p.write_text(s)

# --- ggml.c ---
p = root / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_PF' not in s

# barrier spin loops: prefetch (if anything is queued), then pause
assert s.count('ggml_numa_spin_pause();') == 3, s.count('ggml_numa_spin_pause();')
s = s.replace('ggml_numa_spin_pause();', 'r5pf_spin();')
anchor = '''static inline void ggml_numa_spin_pause(void) {
#if defined(__SSE3__)
    _mm_pause();
#elif defined __ARM_NEON
    __asm__ __volatile__("isb\\n");
#endif
}
'''
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, anchor + '''
// ---- round-5: prefetch upcoming weights while waiting at barriers (IK_R5_PF, see apply_r5pf.py) ----
#define R5PF_MAXR 24
struct r5pf_range { const char * p, * e; int node; };   // [p, e) still to prefetch, needed by graph node `node`
static GGML_THREAD_LOCAL struct r5pf_range r5pf_q[R5PF_MAXR];
static GGML_THREAD_LOCAL int r5pf_h, r5pf_t;            // queue [head, tail)
static GGML_THREAD_LOCAL int r5pf_next = -1, r5pf_last = -1;
static GGML_THREAD_LOCAL const void * r5pf_g;
static int    r5pf_mode = -1, r5pf_lines = 4, r5pf_win = 32;
static size_t r5pf_budget = 256*1024;

static void r5pf_init(void) {
    const char * e;
    int lines = 4, win = 32;
    size_t budget = 256*1024;
    if ((e = getenv("IK_R5_PF_LINES")))  lines  = atoi(e) > 0 ? atoi(e) : 1;
    if ((e = getenv("IK_R5_PF_WIN")))    win    = atoi(e) > 0 ? atoi(e) : 1;
    if ((e = getenv("IK_R5_PF_BUDGET"))) budget = (size_t)(atol(e) > 4096 ? atol(e) : 4096);
    r5pf_lines = lines; r5pf_win = win; r5pf_budget = budget;
    e = getenv("IK_R5_PF");
    r5pf_mode = e ? atoi(e) : 0;
}

static inline void r5pf_spin(void) {
    if (r5pf_h < r5pf_t) {
        struct r5pf_range * r = &r5pf_q[r5pf_h];
        for (int k = 0; k < r5pf_lines; ++k) {
            __builtin_prefetch(r->p, 0, 2);   // prefetcht1: into L2
            r->p += 64;
            if (r->p >= r->e) {
                if (++r5pf_h >= r5pf_t) break;
                r = &r5pf_q[r5pf_h];
            }
        }
    }
    ggml_numa_spin_pause();
}

static inline void r5pf_push(const char * p, size_t n, int node) {
    if (r5pf_t >= R5PF_MAXR || n == 0) return;
    r5pf_q[r5pf_t++] = (struct r5pf_range){ (const char *)((uintptr_t)p & ~(uintptr_t)63), p + n, node };
}
''')

# planner (dense matmul lookahead) and the MoE down-row plan, before the socket-local MoE section
anchor = '// ---- round-5: socket-local fused MoE step (IK_R5_FMOE_LOCAL, see apply_r5local.py) ----\n'
assert s.count(anchor) == 1
s = s.replace(anchor, '''// IK_R5_PF: queue this thread's static rows of the next dense single-token matmuls (called before each node barrier)
static void r5pf_plan(const struct ggml_cgraph * cg, int i, int ith, int nth) {
    if (r5pf_mode < 0) r5pf_init();
    if (r5pf_mode <= 0 || !r5d_active) return;
    if (r5pf_g != cg || i < r5pf_last) {   // new graph or a new pass over it
        r5pf_h = r5pf_t = 0;
        r5pf_next = i + 1;
        r5pf_g = cg;
    }
    r5pf_last = i;
    while (r5pf_h < r5pf_t && r5pf_q[r5pf_h].node <= i) r5pf_h++;   // needed by nodes already done
    if (r5pf_h == r5pf_t) {
        r5pf_h = r5pf_t = 0;
    } else if (r5pf_h > 0 && r5pf_t == R5PF_MAXR) {
        memmove(r5pf_q, r5pf_q + r5pf_h, (r5pf_t - r5pf_h)*sizeof(r5pf_q[0]));
        r5pf_t -= r5pf_h;
        r5pf_h = 0;
    }
    size_t queued = 0;
    for (int k = r5pf_h; k < r5pf_t; ++k) queued += (size_t)(r5pf_q[k].e - r5pf_q[k].p);
    const int node = ggml_numa_node_for_thread(ith, nth);
    const int jend = cg->n_nodes < i + 1 + r5pf_win ? cg->n_nodes : i + 1 + r5pf_win;
    int j = r5pf_next > i + 1 ? r5pf_next : i + 1;
    for (; j < jend && r5pf_t < R5PF_MAXR && queued < r5pf_budget; ++j) {
        const struct ggml_tensor * d = cg->nodes[j];
        if (d->op != GGML_OP_MUL_MAT || d->type != GGML_TYPE_F32) continue;
        const struct ggml_tensor * a = d->src[0], * b = d->src[1];
        if (b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1 || a->ne[2] != 1 || a->ne[3] != 1) continue;
        long first, nrows;
        if (!iqk_r5_dyn_static_rows(a->type, b->type, type_traits[a->type].vec_dot_type, a->ne[0], a->ne[1], ith, nth, &first, &nrows)) continue;
        size_t n = (size_t)nrows*a->nb[1];
        if (queued + n > r5pf_budget) n = r5pf_budget - queued;
        r5pf_push((const char *)ggml_numa_tensor_data(a, node) + first*a->nb[1], n, j);
        queued += n;
    }
    r5pf_next = j;
}

// IK_R5_PF in r5_fmug: once the experts are known, this thread's P3 down rows replace the queue
static void r5pf_moe(int i_end, const struct ggml_tensor * w0, const struct ggml_tensor * ws, const int32_t * sel, int n_ids,
                     int node, int64_t r0, int64_t nr) {
    if (r5pf_mode <= 0 || !r5d_active || nr <= 0) return;
    r5pf_h = r5pf_t = 0;
    r5pf_next = i_end + 1;
    const char * a_r = (const char *)ggml_numa_tensor_data(w0, node);
    for (int j = 0; j < n_ids; ++j) {
        if (sel[j] < 0 || sel[j] >= w0->ne[2]) continue;
        r5pf_push(a_r + sel[j]*w0->nb[2] + r0*w0->nb[1], (size_t)nr*w0->nb[1], i_end);
    }
    r5pf_push((const char *)ggml_numa_tensor_data(ws, node) + r0*ws->nb[1], (size_t)nr*ws->nb[1], i_end);
}

''' + anchor)

# r5_fmug: queue the P3 down rows right after the routed experts are snapshotted (non-router form)
old = '''    if (ir < 0 && params->shared->cplan && params->shared->cplan->moe_expert_prefetch) {
        ggml_moe_prefetch_kernel_hook(ugr, ith);
    }
'''
assert s.count(old) == 1
s = s.replace(old, '''    if (ir < 0) r5pf_moe(i + 8, w0, ws, sel, n_ids, numa_node, r0, nr);
''' + old)

# executor: plan before each node barrier (decode graphs only)
old = '''        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);
'''
assert s.count(old) == 1
s = s.replace(old, '''        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        if (r5pf_mode != 0 && cgraph->n_batch == 1) r5pf_plan(cgraph, node_n, state->ith, state->shared->n_threads);
        ggml_barrier(state->shared);
''')
p.write_text(s)
print('patched')
