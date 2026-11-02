#!/usr/bin/env python3
"""Round-5 patch 17: IK_R5_NONUMAB=1 opts the process out of automatic NUMA balancing without touching system settings.

In mirror mode the server is only CPU-pinned, so its memory has the default task policy, which carries "migrate on
fault". Automatic NUMA balancing (kernel.numa_balancing=1) then keeps unmapping and migrating the shared runtime pages
that threads of both nodes touch every token: the thread-pool/barrier heap page, the work buffer and the compute buffer
(runs/61, runs/62: 6-12K page faults and 200-500 page migrations per 480 decode tokens; 0 single socket, where
numactl --membind gives the process an explicit policy). An explicit MPOL_LOCAL task policy, set in ggml_numa_init
before any worker thread exists, allocates exactly like the default (local node first) but has no migrate-on-fault
flag, so the balancer skips all memory without its own policy.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_NONUMAB' not in s
old = '''    GGML_PRINT_DEBUG("numa strategy %u\\n",g_state.numa.numa_strategy);
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '''    // round-5 (IK_R5_NONUMAB=1): explicit MPOL_LOCAL task policy for this thread and the threads it creates later.
    // Allocates like the default policy but without migrate-on-fault, so automatic NUMA balancing no longer unmaps
    // and migrates the shared runtime pages (heap, work and compute buffers) that both nodes touch every token.
    {
        const char * e = getenv("IK_R5_NONUMAB");
        if (e && atoi(e) > 0) {
            const long rc = syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, NULL, 0UL);
            fprintf(stderr, "ggml_numa_init: IK_R5_NONUMAB: MPOL_LOCAL task policy %s\\n", rc == 0 ? "set" : "FAILED");
        }
    }

''' + old)
p.write_text(s)
print('patched')
