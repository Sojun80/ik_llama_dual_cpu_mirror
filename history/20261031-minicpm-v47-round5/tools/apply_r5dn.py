#!/usr/bin/env python3
"""Round-5 patch 7: NUMA-mirror friendly recurrent (gated delta-net) state.

a) llama_mirror_kv_cache: mirror K/V only. The recurrent state s_l is read/written through
   tensor->data (node 0 alias) so its node>0 copies were never read, yet every conv-state CPY
   paid a barrier + replication memcpy.
b) iqk_fused_delta_net_impl: proportional head split h in [ith*H/nth, (ith+1)*H/nth). With
   nth > H the old ceil split gave every head to threads 0..H-1 (all on NUMA node 0); now the
   heads spread over all nodes. Identical for nth <= H when H % nth == 0 (e.g. 32 heads, 32 thr).
c) ggml_compute_forward_delta_net_f32 (mirror mode, IK_R5_DNBIND default 1): each thread binds
   the state pages of the heads it owns to its NUMA node once (mbind + MPOL_MF_MOVE), so the
   per-token state read/write stays node local. Results are unchanged (same per-head math).
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])

def patch(path, pairs):
    s = path.read_text()
    for old, new in pairs:
        c = s.count(old)
        assert c == 1, (path.name, old[:80], c)
        s = s.replace(old, new)
    path.write_text(s)

patch(root / 'src/llama.cpp', [(
'''            if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            const llama_kv_cache::numa_mirror_buffer * mb = nullptr;''',
'''            if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            // round-5: recurrent state is accessed via tensor->data only; mirroring it just costs
            // replication on every state write
            if (std::find(cache.s_l.begin(), cache.s_l.end(), t) != cache.s_l.end()) {
                continue;
            }
            const llama_kv_cache::numa_mirror_buffer * mb = nullptr;''')])

patch(root / 'ggml/src/iqk/iqk_mul_mat.cpp', [(
'''    const int total_heads = n_heads * n_seqs;
    const int heads_per_thread = (total_heads + nth - 1) / nth;
    const int h_start = ith * heads_per_thread;
    const int h_end = (h_start + heads_per_thread < total_heads) ? h_start + heads_per_thread : total_heads;

#ifdef __AVX2__
    static_assert(head_dim % 8 == 0);
#endif''',
'''    const int total_heads = n_heads * n_seqs;
    // round-5: proportional split so that with nth > total_heads the heads spread over all
    // threads (and NUMA nodes) instead of landing on threads 0..total_heads-1
    const int h_start = (int)((int64_t)ith * total_heads / nth);
    const int h_end   = (int)((int64_t)(ith + 1) * total_heads / nth);

#ifdef __AVX2__
    static_assert(head_dim % 8 == 0);
#endif''')])

patch(root / 'ggml/src/ggml.c', [(
'''    if (iqk_fused_delta_net(head_dim, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,''',
'''    if (ggml_numa_mirror_active()) {
        // round-5: bind the state of the heads this thread owns to its NUMA node, once per range
        static int dnbind = -1;
        if (dnbind < 0) { const char * e = getenv("IK_R5_DNBIND"); dnbind = e ? atoi(e) > 0 : 1; }
        static GGML_THREAD_LOCAL const void * bound[256];
        static GGML_THREAD_LOCAL int n_bound = 0;
        const int64_t total_heads = n_heads * n_seqs;
        const int64_t h0 = (int64_t)ith*total_heads/nth, h1 = (int64_t)(ith + 1)*total_heads/nth;
        if (dnbind && h1 > h0 && n_bound < 256) {
            const size_t hsz = (size_t)head_dim*head_dim*sizeof(float);
            const void * ptrs[2] = { (const char *)state_in + h0*hsz, (const char *)state_working + h0*hsz };
            for (int p = 0; p < 2 && n_bound < 256; ++p) {
                bool seen = false;
                for (int b = 0; b < n_bound; ++b) if (bound[b] == ptrs[p]) { seen = true; break; }
                if (seen || (p == 1 && ptrs[1] == ptrs[0])) continue;
                ggml_numa_bind((void *)ptrs[p], (size_t)(h1 - h0)*hsz, ggml_numa_node_for_thread(ith, nth));
                bound[n_bound++] = ptrs[p];
            }
        }
    }
    if (iqk_fused_delta_net(head_dim, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,''')])
print('patched')
