#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_WHERE=1 prints, for the first FA calls of a few threads, which NUMA node holds the K/V
pages the thread actually reads (move_pages query on k_base/v_base at the thread's first position)."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_FA_WHERE' not in s
old = '''    const void * const k_base = ggml_numa_tensor_data(k, numa_node);
    const void * const v_base = ggml_numa_tensor_data(v, numa_node);
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    {
        static int where = -1;
        if (where < 0) { const char * e = getenv("IK_R5_FA_WHERE"); where = e ? atoi(e) : 0; }
        static atomic_int n_where;
        if (where > 0 && (ith == 0 || ith == nth/2 - 1 || ith == nth/2 || ith == nth - 1) && q->ne[1] == 1 &&
            atomic_fetch_add(&n_where, 1) < 80) {
            // the page each thread starts on: position ~ ith*n_kv/nth
            const long pos = (long)ith*k->ne[1]/nth;
            void * pg[4] = { (char *)k_base + pos*k->nb[1], (char *)v_base + pos*v->nb[1], (char *)k->data + pos*k->nb[1], (char *)v->data + pos*v->nb[1] };
            int st[4] = { -9, -9, -9, -9 };
            for (int z = 0; z < 4; ++z) pg[z] = (void *)((uintptr_t)pg[z] & ~(uintptr_t)4095);
            long rc = syscall(SYS_move_pages, 0, 4L, pg, NULL, st, 0);
            fprintf(stderr, "r5fa-where %s ith %d/%d thread-node %d: K local-copy node %d, V local-copy node %d, K ->data node %d, V ->data node %d (rc %ld, mirrored K %d V %d)\\n",
                    k->name, ith, nth, numa_node, st[0], st[1], st[2], st[3], rc, k_base != k->data, v_base != v->data);
        }
    }
''')
old2 = "            fprintf(stderr, \"r5fa-where %s ith %d/%d"
assert s.count(old2) == 1
scan = """            if (ith == 0) {
                // whole primary K and V buffers of this layer: pages per node (every 64 KiB) and the range's policy
                const struct ggml_tensor * rt[2] = { k, v };
                for (int z = 0; z < 2; ++z) {
                    const struct ggml_tensor * r = rt[z];
                    while (r->view_src) r = r->view_src;
                    const size_t nb = ggml_nbytes(r);
                    int cnt[4] = { 0, 0, 0, 0 };
                    for (size_t o = 0; o < nb; o += 65536) {
                        void * a = (void *)(((uintptr_t)r->data + o) & ~(uintptr_t)4095);
                        int stt = -9;
                        syscall(SYS_move_pages, 0, 1L, &a, NULL, &stt, 0);
                        cnt[stt == 0 ? 0 : stt == 1 ? 1 : stt == -14 ? 2 : 3]++;
                    }
                    int mode = -1;
                    unsigned long nm[4] = { 0, 0, 0, 0 };
                    long prc = syscall(SYS_get_mempolicy, &mode, nm, 256L, r->data, 2L);
                    fprintf(stderr, "r5fa-where-scan %s %p %.1f MiB: node0 %d node1 %d absent %d other %d (64K samples) policy mode %d mask %lx rc %ld\\n",
                            r->name, r->data, nb/1048576.0, cnt[0], cnt[1], cnt[2], cnt[3], mode, nm[0], prc);
                }
            }
"""
s = s.replace(old2, scan + old2)
if '#include <sys/syscall.h>' not in s:
    s = s.replace('#include <sys/mman.h>', '#include <sys/mman.h>\n#include <sys/syscall.h>', 1)
p.write_text(s)
print('patched')
