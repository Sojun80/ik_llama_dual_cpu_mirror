#!/usr/bin/env python3
"""Round-5 patch 14: fused MoE tail for single-token decode (env IK_R5_FMOE=1; 2 = also check consumers).

llm_build_std_moe_ffn emits, per MoE layer at decode, eight nodes after the routed up/gate:
  0 down  = MUL_MAT_ID(down_exps, par, ids)     [n_embd, n_used]
  1 mma   = MUL_MULTI_ADD(down, weights)        weighted sum of the routed experts
  2 a1    = ADD(mma, inp)                       + residual
  3 gt    = MUL_MAT(shexp_gate, x)              shared-expert gate logit (1 value)
  4 ug    = FUSED_UP_GATE(...)                  shared-expert up/gate
  5 sd    = MUL_MAT(down_shexp, ug)             shared-expert down
  6 gated = FUSED_MUL_UNARY(gt, sd)             sigmoid(gate) * sd
  7 out   = ADD(a1, gated)
Nodes 1, 2, 6, 7 are tiny and run on one thread each; 0 and 5 are two separate small matvecs.

Fused: run 3 and 4 first (they do not depend on 0-2), quantize par and ug once, then every thread
owns a 16-row-aligned slice of the n_embd output rows and computes, for its slice, the 8 routed
down rows plus the shared down rows (node-local weights under the NUMA mirror) and the whole
combine, writing `out` directly. 8 barriers become 4 and no core pulls another core's partials.

The routed sum uses the same expressions as iqk_mul_multi_add, the gate the same formula as
ggml_compute_forward_fused_mul_unary_f32, and the two residual adds stay separate roundings, so
the result should match the unfused graph bitwise. The fusion only applies when the node pattern,
shapes and buffer placement (no overlap that the reordering could break) all check out.
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])

def patch(rel, pairs):
    p = root / rel
    s = p.read_text()
    assert 'r5_fmoe' not in s and 'iqk_r5_rows' not in s and 'iqk_r5_moe_tail' not in s, rel
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)

# ---------------------------------------------------------------- iqk_mul_mat.h / .cpp
patch('ggml/src/iqk/iqk_mul_mat.h', [(
'''IQK_API void iqk_r5_set_dyn(void * counter);
''', '''IQK_API void iqk_r5_set_dyn(void * counter);

// round-5: C[0..Nx) = A rows [0,Nx) . B (one column), single-threaded; false if unsupported
IQK_API bool iqk_r5_rows(long Nx, long ne00, int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB, float * C);
''')])

patch('ggml/src/iqk/iqk_mul_mat.cpp', [(
'''extern "C" IQK_API void iqk_r5_set_dyn(void * counter) {
    r5_dyn_counter = (std::atomic<int> *)counter;
}
''', '''extern "C" IQK_API void iqk_r5_set_dyn(void * counter) {
    r5_dyn_counter = (std::atomic<int> *)counter;
}

extern "C" IQK_API bool iqk_r5_rows(long Nx, long ne00, int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB, float * C) {
    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, 1)) return false;
    if (Nx % MulMat::num_rows(ggml_type(typeA))) return false;
    DataInfo info{C, (const char *)B, (size_t)Nx, (size_t)strideB, 0, 1, nullptr, 0};
    mm.mul_mat_NxM(ne00, A, strideA, info, Nx, 1);
    return true;
}
'''), (
'''extern "C" IQK_API bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,''',
'''extern "C" IQK_API bool iqk_r5_rows(long, long, int, const void *, long, int, const void *, long, float *) {
    return false;
}

extern "C" IQK_API bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,''')])

# ---------------------------------------------------------------- iqk_cpu_ops.h / .cpp
patch('ggml/src/iqk/iqk_cpu_ops.h', [(
'''void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth);
''', '''void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth);

// round-5 fused MoE tail: out = (sum_j w[j]*e_j + inp) + gate*sh over n columns (y, z: n-float scratch)
void iqk_r5_moe_tail(int n, int ne, const float * e, long stride_e, const float * w,
        const float * inp, const float * sh, float gate, float limit, float * out, float * y, float * z);
''')])

patch('ggml/src/iqk/iqk_cpu_ops.cpp', [(
'''void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth) {
''', '''// Same expressions as iqk_mul_multi_add (routed sum) and fused_mul_unary (gate); the final loop has no
// multiply, so the two residual adds round separately exactly as the unfused ADD nodes do.
void iqk_r5_moe_tail(int n, int ne, const float * e, long stride_e, const float * w,
        const float * inp, const float * sh, float gate, float limit, float * out, float * y, float * z) {
    for (int k = 0; k < n; ++k) y[k] = e[k] * w[0];
    for (int j = 1; j < ne; ++j) {
        const float * x = e + j*stride_e;
        for (int k = 0; k < n; ++k) y[k] += x[k] * w[j];
    }
    if (limit < 1e-6f) {
        for (int k = 0; k < n; ++k) z[k] = gate * sh[k];
    } else {
        const float g = std::min(gate, limit);
        for (int k = 0; k < n; ++k) z[k] = std::max(-limit, std::min(limit, sh[k])) * g;
    }
    for (int k = 0; k < n; ++k) out[k] = (y[k] + inp[k]) + z[k];
}

void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth) {
''')])

# ---------------------------------------------------------------- ggml.c
FUSED = r'''
// ---- round-5: fused MoE tail for single-token decode (IK_R5_FMOE, see apply_r5fmoe.py) ----
static int r5_fmoe_mode(void) {
    static int mode = -1;
    if (mode < 0) { const char * e = getenv("IK_R5_FMOE"); mode = e ? atoi(e) : 0; }
    return mode;
}

static inline bool r5_ov(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const char * a0 = (const char *)a->data, * b0 = (const char *)b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static inline bool r5_in(const struct ggml_tensor * t, struct ggml_tensor * const * n, int k) {
    for (int j = 0; j < k; ++j) if (t == n[j] || (t && t->view_src == n[j])) return true;
    return false;
}

static const struct ggml_tensor * r5_fmoe_inp(const struct ggml_tensor * a1, const struct ggml_tensor * mma) {
    return a1->src[0] == mma ? a1->src[1] : a1->src[1] == mma ? a1->src[0] : NULL;
}

static bool r5_fmoe_match(const struct ggml_cgraph * g, int i, size_t wsize) {
    if (i + 7 >= g->n_nodes) return false;
    struct ggml_tensor * const * n = g->nodes + i;
    const struct ggml_tensor * down = n[0], * mma = n[1], * a1 = n[2], * gt = n[3], * ug = n[4], * sd = n[5], * gated = n[6], * out = n[7];
    if (mma->op != GGML_OP_MUL_MULTI_ADD || a1->op != GGML_OP_ADD || gt->op != GGML_OP_MUL_MAT || ug->op != GGML_OP_FUSED_UP_GATE ||
        sd->op != GGML_OP_MUL_MAT || gated->op != GGML_OP_FUSED_MUL_UNARY || out->op != GGML_OP_ADD) return false;
    const struct ggml_tensor * w0 = down->src[0], * par = down->src[1], * ids = down->src[2];
    const int64_t n_embd = down->ne[0];
    if (down->type != GGML_TYPE_F32 || down->ne[2] != 1 || down->ne[3] != 1 || w0->ne[3] != 1 || w0->ne[1] != n_embd) return false;
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] > 16 || down->ne[1] != ids->ne[0]) return false;
    if (par->type != GGML_TYPE_F32 || !ggml_is_contiguous(par) || par->ne[0] != w0->ne[0] || par->ne[2] != 1) return false;
    const struct ggml_tensor * W = mma->src[1];
    if (mma->src[0] != down || mma->src[2] || mma->src[3] || W->type != GGML_TYPE_F32 || W->ne[1] != ids->ne[0]) return false;
    const struct ggml_tensor * inp = r5_fmoe_inp(a1, mma);
    if (!inp || inp->type != GGML_TYPE_F32 || !ggml_is_contiguous(inp) || ggml_nelements(inp) != n_embd) return false;
    if (gated->src[0] != gt || gated->src[1] != sd) return false;
    if (gated->op_params[0] != GGML_UNARY_OP_SIGMOID && gated->op_params[0] != GGML_UNARY_OP_SILU) return false;
    if (gt->type != GGML_TYPE_F32 || ggml_nelements(gt) != 1) return false;
    const struct ggml_tensor * ws = sd->src[0];
    if (sd->src[1] != ug || ug->type != GGML_TYPE_F32 || !ggml_is_contiguous(ug) || ug->ne[1] != 1 || ug->ne[2] != 1) return false;
    if (ws->ne[0] != ug->ne[0] || ws->ne[1] != n_embd || ws->ne[2] != 1 || ws->ne[3] != 1 || sd->type != GGML_TYPE_F32) return false;
    if (!((out->src[0] == a1 && out->src[1] == gated) || (out->src[1] == a1 && out->src[0] == gated))) return false;
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != n_embd || n_embd % 16) return false;
    // gt and ug run first: they must not read the routed part ...
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (r5_in(gt->src[s], n, 3) || r5_in(ug->src[s], n, 3)) return false;
    }
    // ... nor be placed over anything still to be read
    const struct ggml_tensor * keep[4] = { inp, W, ids, par };
    for (int k = 0; k < 4; ++k) if (r5_ov(gt, keep[k]) || r5_ov(ug, keep[k])) return false;
    // out is written row-slice by row-slice while other threads read W, ids, gt (and inp, unless in place)
    if (r5_ov(out, W) || r5_ov(out, ids) || r5_ov(out, gt) || (out->data != inp->data && r5_ov(out, inp))) return false;
    const enum ggml_type vt_r = type_traits[w0->type].vec_dot_type, vt_s = type_traits[ws->type].vec_dot_type;
    if (vt_r == GGML_TYPE_F32 || vt_s == GGML_TYPE_F32) return false;
    const size_t rs_r = ggml_row_size(vt_r, par->ne[0]), rs_s = ggml_row_size(vt_s, ug->ne[0]);
    if (GGML_PAD(par->ne[1]*rs_r, 64) + rs_s > wsize) return false;
    return true;
}

static int ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor,
        const struct ggml_cgraph * cgraph, int i);

static int r5_fmoe(struct ggml_compute_params * params, const struct ggml_cgraph * g, int i) {
    struct ggml_tensor * const * n = g->nodes + i;
    struct ggml_tensor * down = n[0], * mma = n[1], * a1 = n[2], * gt = n[3], * ug = n[4], * sd = n[5], * gated = n[6], * out = n[7];
    const int ith = params->ith, nth = params->nth;

    if (r5_fmoe_mode() > 1 && ith == 0) {
        // debug: the skipped nodes must have no consumers outside the pattern
        static int n_checked = 0, n_bad = 0;
        if (n_checked < 200) {
            struct ggml_tensor * const skipped[5] = { n[0], n[1], n[2], n[5], n[6] };
            for (int k = i + 8; k < g->n_nodes; ++k) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (r5_in(g->nodes[k]->src[s], skipped, 5)) {
                        ++n_bad;
                        fprintf(stderr, "r5_fmoe: node %d %s reads skipped node of pattern at %d\n", k, g->nodes[k]->name, i);
                    }
                }
            }
            if (++n_checked % 40 == 0) fprintf(stderr, "r5_fmoe: consumer check %d patterns, %d external reads\n", n_checked, n_bad);
        }
    }

    // 1. shared-expert gate and up/gate (independent of the routed part)
    if (ggml_compute_forward(params, gt, g, i + 3) != i + 3) GGML_ABORT("r5_fmoe: gate node was fused");
    ggml_barrier(params->shared);
    if (ggml_compute_forward(params, ug, g, i + 4) != i + 4) GGML_ABORT("r5_fmoe: up/gate node was fused");
    ggml_barrier(params->shared);

    // 2. quantize the routed inputs (n_par rows) and the shared-expert input, spread over the threads
    const struct ggml_tensor * w0 = down->src[0], * par = down->src[1], * ids = down->src[2], * ws = sd->src[0];
    const enum ggml_type vt_r = type_traits[w0->type].vec_dot_type, vt_s = type_traits[ws->type].vec_dot_type;
    const size_t rs_r = ggml_row_size(vt_r, par->ne[0]), rs_s = ggml_row_size(vt_s, ug->ne[0]);
    const int n_par = (int)par->ne[1];
    char * wq = (char *)params->wdata;
    const size_t off_s = GGML_PAD(n_par*rs_r, 64);
    {
        const int nconv = n_par + 1;
        const int step = nth >= nconv ? nth/nconv : 1;
        for (int r = (ith % step == 0 ? ith/step : nconv); r < nconv; r += (nth >= nconv ? nconv : nth)) {
            if (r < n_par) type_traits[vt_r].from_float((const float *)((const char *)par->data + r*par->nb[1]), wq + r*rs_r, par->ne[0]);
            else           type_traits[vt_s].from_float((const float *)ug->data, wq + off_s, ug->ne[0]);
        }
    }
    ggml_barrier(params->shared);

    // 3. this thread's output rows: 8 routed + shared down rows, then the combine
    const int64_t n_embd = down->ne[0];
    const int units = (int)(n_embd/16);
    const int u0 = (int)((int64_t)ith*units/nth), u1 = (int)((int64_t)(ith + 1)*units/nth);
    if (u1 > u0) {
        const int64_t r0 = 16*(int64_t)u0, nr = 16*(int64_t)(u1 - u0);
        const int numa_node = ggml_numa_node_for_thread(ith, nth);
        const char * a_r = (const char *)ggml_numa_tensor_data(w0, numa_node);
        const char * a_s = (const char *)ggml_numa_tensor_data(ws, numa_node);
        const int n_ids = (int)ids->ne[0];
        float buf[(n_ids + 3)*nr];
        float wts[16];
        for (int j = 0; j < n_ids; ++j) {
            const int32_t a = *(const int32_t *)((const char *)ids->data + j*ids->nb[0]);
            float * c = buf + j*nr;
            wts[j] = *(const float *)((const char *)mma->src[1]->data + j*mma->src[1]->nb[1]);
            if (a < 0 || a >= w0->ne[2]) { memset(c, 0, nr*sizeof(float)); continue; }
            if (!iqk_r5_rows(nr, w0->ne[0], w0->type, a_r + a*w0->nb[2] + r0*w0->nb[1], w0->nb[1],
                             vt_r, wq + (j % n_par)*rs_r, rs_r, c)) GGML_ABORT("r5_fmoe: routed type not supported");
        }
        float * sh = buf + n_ids*nr;
        if (!iqk_r5_rows(nr, ws->ne[0], ws->type, a_s + r0*ws->nb[1], ws->nb[1], vt_s, wq + off_s, rs_s, sh))
            GGML_ABORT("r5_fmoe: shared type not supported");
        const float x = *(const float *)gt->data;
        const float gate = gated->op_params[0] == GGML_UNARY_OP_SILU ? ggml_silu_f32(x) : 1.0f/(1.0f + expf(-x));
        const float limit = *(const float *)(gated->op_params + 1);
        const struct ggml_tensor * inp = r5_fmoe_inp(a1, mma);
        iqk_r5_moe_tail((int)nr, n_ids, buf, nr, wts, (const float *)inp->data + r0, sh, gate, limit,
                        (float *)out->data + r0, sh + nr, sh + 2*nr);
    }
    return i + 7;
}

static int ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor,
        const struct ggml_cgraph * cgraph, int i) {
'''

patch('ggml/src/ggml.c', [(
'''
static int ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor,
        const struct ggml_cgraph * cgraph, int i) {
''', FUSED), (
'''        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;''', '''        case GGML_OP_MUL_MAT_ID:
            {
                if (r5_fmoe_mode() > 0 && r5_fmoe_match(cgraph, i, params->wsize)) {
                    i = r5_fmoe(params, cgraph, i);
                } else {
                    ggml_compute_forward_mul_mat_id(params, tensor);
                }
            } break;''')])
print('patched')
