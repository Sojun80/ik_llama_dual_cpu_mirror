#!/usr/bin/env python3
"""Round-5 patch 11: pin worker threads once (env IK_R5_PINONCE=1).

Every graph (= every decoded token) each worker re-pinned itself with up to two
pthread_setaffinity_np calls (NUMA node set, then the explicit CPU), and the main thread saved and
restored its affinity around the graph. With IK_R5_PINONCE=1 and an explicit --cpu-range:
  * set_numa_thread_affinity still records the thread's node (needed by the hierarchical barrier)
    but skips its syscall, since the explicit CPU pin overrides it anyway;
  * set_cpu_thread_affinity skips the syscall when this thread is already on that CPU;
  * the main thread keeps its pin between graphs (no clear/restore).
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_PINONCE' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

sub('''static void set_numa_thread_affinity(int thread_n, int n_threads) {
    if (!ggml_is_numa()) {
        return;
    }
''', '''static int r5_pinonce_enabled(void) {
    static int e = -1;
    if (e < 0) { const char * v = getenv("IK_R5_PINONCE"); e = v && atoi(v) > 0; }
    return e;
}
static volatile int r5_explicit_affinity = 0;   // set per graph: cplan carries an explicit CPU list

static void set_numa_thread_affinity(int thread_n, int n_threads) {
    if (!ggml_is_numa()) {
        return;
    }
''')
sub('''    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);''', '''    if (r5_pinonce_enabled() && r5_explicit_affinity) {
        return; // node recorded above; the explicit per-thread CPU pin takes precedence
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);''')
sub('''    const int cpu = cplan->cpu_affinity[thread_n % cplan->n_cpu_affinity];
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return;
    }
''', '''    const int cpu = cplan->cpu_affinity[thread_n % cplan->n_cpu_affinity];
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return;
    }
    static GGML_THREAD_LOCAL int r5_pinned_cpu = -1;
    if (r5_pinonce_enabled()) {
        if (r5_pinned_cpu == cpu) {
            return;
        }
        r5_pinned_cpu = cpu;
    }
''')
sub('''    cpu_set_t saved_affinity;
    const bool restore_affinity =
        cplan->cpu_affinity != NULL && cplan->n_cpu_affinity > 0 &&
        pthread_getaffinity_np(pthread_self(), sizeof(saved_affinity), &saved_affinity) == 0;
#endif
''', '''    cpu_set_t saved_affinity;
    r5_explicit_affinity = cplan->cpu_affinity != NULL && cplan->n_cpu_affinity > 0;
    const bool r5_keep_pin = r5_pinonce_enabled() && r5_explicit_affinity;
    const bool restore_affinity = !r5_keep_pin &&
        cplan->cpu_affinity != NULL && cplan->n_cpu_affinity > 0 &&
        pthread_getaffinity_np(pthread_self(), sizeof(saved_affinity), &saved_affinity) == 0;
#endif
''')
sub('''    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();
''', '''    // don't leave affinity set on the main thread (unless pinned once on purpose)
#if defined(__gnu_linux__)
    if (!r5_keep_pin)
#endif
    clear_numa_thread_affinity();
''')
p.write_text(s)
print('patched')
