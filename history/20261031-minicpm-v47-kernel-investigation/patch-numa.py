import pathlib,sys
root=pathlib.Path(sys.argv[1]);p=root/'src/llama-model-loader.cpp';s=p.read_text();anchor='#include <cstring>\n';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'#if defined(__linux__)\n#include <linux/mempolicy.h>\n#include <sys/syscall.h>\n#include <unistd.h>\n#include <cerrno>\n#include <cstdlib>\n#endif\n')
anchor='''        if (ggml_backend_buffer_is_host(cur->buffer)) {
            file->seek(weight->offs, SEEK_SET);'''
assert s.count(anchor)==1
replacement='''        if (ggml_backend_buffer_is_host(cur->buffer)) {
#if defined(__linux__)
            // Experimental: put contiguous halves of each 256-expert tensor on
            // different nodes before the loader faults the pages in.  Align
            // policy boundaries to 2 MiB to retain huge-page eligibility.
            const char * ename = ggml_get_name(cur);
            if (std::getenv("IK_EXPERT_NUMA") && cur->ne[2] == 256 && cur->data &&
                    std::strstr(ename, "ffn_") && std::strstr(ename, "_exps.weight")) {
                constexpr uintptr_t align = 2u*1024u*1024u;
                uintptr_t b = (uintptr_t)cur->data;
                uintptr_t lo = (b + align - 1) & ~(align - 1);
                uintptr_t mid = (b + cur->nb[2]*128) & ~(align - 1);
                uintptr_t hi = (b + n_size) & ~(align - 1);
                if (lo < mid && mid < hi) {
                    unsigned long node0 = 1ul, node1 = 2ul;
                    if (syscall(SYS_mbind, (void *)lo, mid-lo, MPOL_BIND, &node0, 2ul, 0ul) ||
                        syscall(SYS_mbind, (void *)mid, hi-mid, MPOL_BIND, &node1, 2ul, 0ul)) {
                        throw std::runtime_error(format("expert mbind failed for %s: %s", ename, std::strerror(errno)));
                    }
                    fprintf(stderr, "NUMA_EXPERT_BIND %s node0=%zu node1=%zu\\n", ename, (size_t)(mid-lo), (size_t)(hi-mid));
                }
            }
#endif
            file->seek(weight->offs, SEEK_SET);'''
s=s.replace(anchor,replacement);p.write_text(s)
p=root/'ggml/src/ggml.c';s=p.read_text();anchor='''        int chunk_id = ith;
        while (chunk_id < total_chunks) {''';assert s.count(anchor)==1
replacement='''        // Experimental two-socket routing: first 16 workers own expert ids
        // 0..127, next 16 own ids 128..255. Each expert's output rows are
        // still disjoint; the graph-level barrier joins the two groups.
        if (getenv("IK_EXPERT_NUMA_SCHED") && ids->ne[1] == 1 && n_as == 256 && nth == 32) {
            if (ith == 0) { static bool logged = false; if (!logged) { fprintf(stderr, "NUMA_EXPERT_SCHED mul_mat_id active\\n"); logged = true; } }
            for (int a = ith < 16 ? 0 : 128; a < (ith < 16 ? 128 : 256); ++a) {
                if (!matrix_row_counts[a]) continue;
                const char * src0_cur = (const char *)src0->data + a*nb02;
                for (int chunk = ith % 16; chunk < chunks_per_expert; chunk += 16) {
                    if (!iqk_mul_mat_moe(ne01, matrix_row_counts[a], ne00, ne11,
                                src0->type, src0_cur, nb01, vec_dot_type,
                                (const char *)wdata_mm, row_size_mm, (float *)dst->data,
                                nb1, nb2, matrix_rows + a*ne12, chunk, chunks_per_expert)) {
                        GGML_ABORT("experimental NUMA mul_mat_id unsupported");
                    }
                }
            }
            return;
        }

        int chunk_id = ith;
        while (chunk_id < total_chunks) {'''
s=s.replace(anchor,replacement)
anchor='''    int last_a = 0;
    int last_acc = 0;
    int chunk_id_ug = ith;''';assert s.count(anchor)==1
replacement='''    if (getenv("IK_EXPERT_NUMA_SCHED") && ids->ne[1] == 1 && n_as == 256 && nth == 32) {
        if (ith == 0) { static bool logged = false; if (!logged) { fprintf(stderr, "NUMA_EXPERT_SCHED fused_up_gate active\\n"); logged = true; } }
        for (int a = ith < 16 ? 0 : 128; a < (ith < 16 ? 128 : 256); ++a) {
            if (!matrix_row_counts[a]) continue;
            const char *up, *gate, *up_bias = NULL, *gate_bias = NULL;
            if (src0_2) {
                up = (const char *)src0_1->data + a*nb02;
                gate = (const char *)src0_2->data + a*nb02;
                up_bias = up_b ? (const char *)up_b->data + a*nb41 : NULL;
                gate_bias = gate_b ? (const char *)gate_b->data + a*nb51 : NULL;
            } else {
                gate = (const char *)src0_1->data + a*nb02;
                up = gate + nb02/2;
                if (up_b) { GGML_ASSERT(!gate_b); gate_bias = (const char *)up_b->data + a*nb41; up_bias = gate_bias + nb41/2; }
            }
            for (int chunk = ith % 16; chunk < chunks_per_expert_ug; chunk += 16) {
                if (!iqk_moe_fused_up_gate(nr0_base, matrix_row_counts[a], ne00, ne11, dst->op_params[0],
                            type, up, gate, nb01, vec_dot_type, (const char *)wdata_ug, row_size_ug,
                            up_bias, gate_bias, (float *)dst->data, nb1, nb2,
                            matrix_rows + a*ne12, limit, situ_beta, situ_beta_linear, chunk, chunks_per_expert_ug)) {
                    GGML_ABORT("experimental NUMA fused_up_gate unsupported");
                }
            }
        }
        return;
    }

    int last_a = 0;
    int last_acc = 0;
    int chunk_id_ug = ith;'''
s=s.replace(anchor,replacement);p.write_text(s)
