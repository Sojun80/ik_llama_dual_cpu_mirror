#!/usr/bin/env python3
"""Round-5 patch 6: make MoE expert chunking NUMA-mirror aware.

1. Correctness of the mirror concept: the GGML_EXPERT_CHUNKING paths of MUL_MAT_ID and
   MOE_FUSED_UP_GATE read src0->data (node 0's copy) directly, so with --numa mirror every
   node>0 thread streamed expert weights across the socket link. Use the node-local pointers
   the numa-mirror port already computes.
2. Node-local chunk counters (env IK_R5_NODECHUNK, default on when mirroring): node k claims
   chunks from its own atomic over its share of the chunk space and steals from the other
   nodes only when that is exhausted, so the claim cache line no longer bounces between sockets.
   The up/gate loop's (last_a, last_acc) lookup cache assumed increasing chunk ids; it is reset
   whenever a stolen chunk id goes backwards.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'r5_claim' not in s

def sub(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (old[:80], c)
    s = s.replace(old, new)

# shared state: per-node counters, one cache line each
# appended at the END: the struct is initialized positionally, trailing members are zeroed
sub('''    atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads

    enum ggml_status ec;
};''', '''    atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads

    enum ggml_status ec;

    atomic_int r5_node_chunk[GGML_NUMA_MAX_NODES*16]; // round-5: per-NUMA-node chunk counters (stride: one cache line)
};''')

HELPER = '''
// ---- round-5: NUMA-node-local chunk claiming for expert chunking -----------------------------
static int r5_nodechunk_enabled(void) {
    static int e = -1;
    if (e < 0) { const char * v = getenv("IK_R5_NODECHUNK"); e = v ? atoi(v) > 0 : 1; }
    return e;
}
static inline int r5_claim_nodes(void) {
    return ggml_numa_mirror_active() && r5_nodechunk_enabled() ? ggml_numa_node_count() : 1;
}
static inline void r5_claim_reset(struct ggml_compute_state_shared * sh) {
    for (int k = 0; k < GGML_NUMA_MAX_NODES; ++k) atomic_store(&sh->r5_node_chunk[16*k], 0);
}
// next chunk in [0, total) or -1: own node's share first, then steal from the others
static inline int r5_claim(struct ggml_compute_state_shared * sh, int node, int nnodes, int total) {
    for (int s = 0; s < nnodes; ++s) {
        const int k  = (node + s) % nnodes;
        const int lo = (int)((int64_t)total*k/nnodes), hi = (int)((int64_t)total*(k + 1)/nnodes);
        if (lo >= hi) continue;
        const int c = atomic_fetch_add_explicit(&sh->r5_node_chunk[16*k], 1, memory_order_relaxed);
        if (lo + c < hi) return lo + c;
    }
    return -1;
}

static void ggml_compute_forward_mul_mat_id(
'''
sub('\nstatic void ggml_compute_forward_mul_mat_id(\n', HELPER)

# ---- MUL_MAT_ID chunking path
sub('''    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
    }

    ggml_barrier(params->shared);

#if GGML_USE_IQK_MULMAT
#if defined GGML_EXPERT_CHUNKING''', '''    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
        r5_claim_reset(params->shared);
    }

    ggml_barrier(params->shared);

#if GGML_USE_IQK_MULMAT
#if defined GGML_EXPERT_CHUNKING''')
sub('''        int chunk_id = ith;
        while (chunk_id < total_chunks) {''', '''        const int r5_nn = r5_claim_nodes();
        int chunk_id = r5_nn > 1 ? r5_claim(params->shared, numa_node, r5_nn, total_chunks) : ith;
        while (chunk_id >= 0 && chunk_id < total_chunks) {''')
sub('''            const char * src0_cur = (const char *) src0->data + cur_a*nb02;
''', '''            const char * src0_cur = src0_data + cur_a*nb02;   // node-local copy (NUMA mirror)
''')
sub('''                        matrix_rows + cur_a*ne12, local_chunk, chunks_per_expert)) goto IQK_MulMat_Not_Available0;

            chunk_id = atomic_fetch_add(&params->shared->current_chunk, 1);''', '''                        matrix_rows + cur_a*ne12, local_chunk, chunks_per_expert)) goto IQK_MulMat_Not_Available0;

            chunk_id = r5_nn > 1 ? r5_claim(params->shared, numa_node, r5_nn, total_chunks)
                                 : atomic_fetch_add(&params->shared->current_chunk, 1);''')

# ---- MOE fused up/gate chunking path
sub('''#if defined GGML_EXPERT_CHUNKING

    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
    }
''', '''#if defined GGML_EXPERT_CHUNKING

    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
        r5_claim_reset(params->shared);
    }
''')
sub('''    int last_a = 0;
    int last_acc = 0;
    int chunk_id_ug = ith;
    while (chunk_id_ug < total_chunks_ug) {
        int acc = last_acc, cur_a = -1, local_chunk = 0;''', '''    int last_a = 0;
    int last_acc = 0;
    const int r5_nn = r5_claim_nodes();
    int chunk_id_ug = r5_nn > 1 ? r5_claim(params->shared, numa_node, r5_nn, total_chunks_ug) : ith;
    while (chunk_id_ug >= 0 && chunk_id_ug < total_chunks_ug) {
        if (chunk_id_ug < last_acc) { last_a = 0; last_acc = 0; }   // stolen chunk ids can go backwards
        int acc = last_acc, cur_a = -1, local_chunk = 0;''')
sub('''        if (src0_2) {
            src0_1_cur = (const char *) src0_1->data + cur_a*nb02;
            src0_2_cur = (const char *) src0_2->data + cur_a*nb02;
            up_b_cur   = up_b   ? (const char *)up_b->data + cur_a*nb41 : NULL;
            gate_b_cur = gate_b ? (const char *)gate_b->data + cur_a*nb51 : NULL;
        } else {
            src0_2_cur = (const char *) src0_1->data + cur_a*nb02;
            src0_1_cur = src0_2_cur + nb02/2;
            if (up_b) {
                GGML_ASSERT(!gate_b);
                gate_b_cur = (const char *)up_b->data + cur_a*nb41;''', '''        // node-local copies (NUMA mirror); identical to ->data when not mirroring
        if (src0_2) {
            src0_1_cur = src0_1_data + cur_a*nb02;
            src0_2_cur = src0_2_data + cur_a*nb02;
            up_b_cur   = up_b   ? up_b_data   + cur_a*nb41 : NULL;
            gate_b_cur = gate_b ? gate_b_data + cur_a*nb51 : NULL;
        } else {
            src0_2_cur = src0_1_data + cur_a*nb02;
            src0_1_cur = src0_2_cur + nb02/2;
            if (up_b) {
                GGML_ASSERT(!gate_b);
                gate_b_cur = up_b_data + cur_a*nb41;''')
sub('''                            matrix_rows + cur_a*ne12, limit, situ_beta, situ_beta_linear, local_chunk, chunks_per_expert_ug)) GGML_ABORT("fatal error");

        chunk_id_ug = atomic_fetch_add(&params->shared->current_chunk, 1);''', '''                            matrix_rows + cur_a*ne12, limit, situ_beta, situ_beta_linear, local_chunk, chunks_per_expert_ug)) GGML_ABORT("fatal error");

        chunk_id_ug = r5_nn > 1 ? r5_claim(params->shared, numa_node, r5_nn, total_chunks_ug)
                                : atomic_fetch_add(&params->shared->current_chunk, 1);''')
p.write_text(s)
print('patched')
