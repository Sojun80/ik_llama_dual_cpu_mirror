#!/usr/bin/env python3
"""Round-5 patch 8: optional flush-to-zero / denormals-are-zero for graph compute threads (env IK_R5_FTZ=1).

Peaked softmax weights in attention and decaying recurrent state can produce denormal floats,
which x86 handles with slow microcode assists. ggml never sets MXCSR. With IK_R5_FTZ=1 each
compute thread sets FTZ|DAZ for the duration of the graph and restores the previous MXCSR.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_FTZ' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

sub('''static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->shared->cgraph;
    const struct ggml_cplan  * cplan  = state->shared->cplan;

    set_numa_thread_affinity(state->ith, state->shared->n_threads);
    set_cpu_thread_affinity(cplan, state->ith); // explicit affinity takes precedence over NUMA
''', '''#if defined(__SSE__)
#include <xmmintrin.h>
static int r5_ftz_enabled(void) {
    static int e = -1;
    if (e < 0) { const char * v = getenv("IK_R5_FTZ"); e = v && atoi(v) > 0; }
    return e;
}
#endif

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->shared->cgraph;
    const struct ggml_cplan  * cplan  = state->shared->cplan;

    set_numa_thread_affinity(state->ith, state->shared->n_threads);
    set_cpu_thread_affinity(cplan, state->ith); // explicit affinity takes precedence over NUMA

#if defined(__SSE__)
    const unsigned int r5_csr = _mm_getcsr();
    if (r5_ftz_enabled()) _mm_setcsr(r5_csr | 0x8040);   // FTZ (bit 15) | DAZ (bit 6)
#endif
''')

sub('''#if IK_PRINT_TIMING
    int64_t t_end = ggml_time_us();
    if (state->ith == 0) printf("ggml_barrier(...): %d us\\n", (int)(t_end - t_start - t_eval));
#endif

    return 0;
}''', '''#if IK_PRINT_TIMING
    int64_t t_end = ggml_time_us();
    if (state->ith == 0) printf("ggml_barrier(...): %d us\\n", (int)(t_end - t_start - t_eval));
#endif
#if defined(__SSE__)
    _mm_setcsr(r5_csr);
#endif

    return 0;
}''')
p.write_text(s)
print('patched')
