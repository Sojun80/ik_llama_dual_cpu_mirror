// bwprobe: DRAM read-bandwidth probe for decode-shaped work on Zen 2.
// Usage: bwprobe <cpu-list comma separated> <job_bytes> [total_mb] [prefetch_dist]
//   job_bytes = 0 -> pure streaming (no barriers), each thread owns a private slab.
//   job_bytes > 0 -> a long sequence of jobs of that size, each split evenly
//                    across threads, followed by a spin barrier (decode-like).
// Memory placement comes from the caller (numactl --membind) + first touch.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int nthr, cpus[256];
static size_t job_bytes, total_bytes;
static int pf_dist;
static uint8_t *buf;
static _Atomic int bar_count;
static _Atomic int bar_gen;
static double results[256];
static _Atomic int start_flag;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void barrier(void) {
    int gen = atomic_load_explicit(&bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel) == nthr - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&bar_gen, 1, memory_order_release);
    } else {
        while (atomic_load_explicit(&bar_gen, memory_order_acquire) == gen) _mm_pause();
    }
}

static inline __m256i read_range(const uint8_t *p, size_t n, __m256i acc) {
    // n multiple of 128
    for (size_t i = 0; i < n; i += 128) {
        if (pf_dist) _mm_prefetch((const char *)p + i + pf_dist, _MM_HINT_T0),
                     _mm_prefetch((const char *)p + i + pf_dist + 64, _MM_HINT_T0);
        __m256i a = _mm256_load_si256((const __m256i *)(p + i));
        __m256i b = _mm256_load_si256((const __m256i *)(p + i + 32));
        __m256i c = _mm256_load_si256((const __m256i *)(p + i + 64));
        __m256i d = _mm256_load_si256((const __m256i *)(p + i + 96));
        acc = _mm256_add_epi64(acc, _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)));
    }
    return acc;
}

static void *worker(void *arg) {
    int id = (int)(intptr_t)arg;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpus[id], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    __m256i acc = _mm256_setzero_si256();
    size_t slab = total_bytes / nthr / 128 * 128;
    // first touch: in job mode, touch the same pieces this thread will read
    if (job_bytes == 0) {
        memset(buf + id * slab, id + 1, slab);
    } else {
        size_t per = job_bytes / nthr / 128 * 128;
        for (size_t j = 0; j + job_bytes <= total_bytes; j += job_bytes) memset(buf + j + id * per, id + 1, per);
    }
    barrier();
    while (!atomic_load(&start_flag)) _mm_pause();
    double t0 = now();
    int reps = 0;
    size_t moved = 0;
    if (job_bytes == 0) {
        for (reps = 0; reps < 6; reps++) acc = read_range(buf + id * slab, slab, acc), moved += slab;
        barrier();
    } else {
        size_t per = job_bytes / nthr / 128 * 128;
        for (reps = 0; reps < 3; reps++)
            for (size_t j = 0; j + job_bytes <= total_bytes; j += job_bytes) {
                acc = read_range(buf + j + id * per, per, acc);
                moved += per;
                barrier();
            }
    }
    double t1 = now();
    results[id] = (double)moved;
    if (id == 0) results[255] = t1 - t0;
    volatile uint64_t sink = _mm256_extract_epi64(acc, 0);
    (void)sink;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s cpus job_bytes [total_mb] [pf]\n", argv[0]); return 1; }
    char *s = strdup(argv[1]);
    for (char *t = strtok(s, ","); t; t = strtok(NULL, ",")) cpus[nthr++] = atoi(t);
    job_bytes = strtoull(argv[2], 0, 10);
    total_bytes = (argc > 3 ? strtoull(argv[3], 0, 10) : 4096) << 20;
    pf_dist = argc > 4 ? atoi(argv[4]) : 0;
    if (job_bytes) total_bytes = total_bytes / job_bytes * job_bytes;
    buf = aligned_alloc(2 << 20, total_bytes + (4 << 20));
    pthread_t th[256];
    for (int i = 0; i < nthr; i++) pthread_create(&th[i], 0, worker, (void *)(intptr_t)i);
    // workers do first touch then wait at barrier; main is not part of barrier
    struct timespec w = {0, 200000000};
    nanosleep(&w, 0);
    atomic_store(&start_flag, 1);
    for (int i = 0; i < nthr; i++) pthread_join(th[i], 0);
    double moved = 0;
    for (int i = 0; i < nthr; i++) moved += results[i];
    double dt = results[255];
    size_t njobs = job_bytes ? total_bytes / job_bytes * 3 : 0;
    printf("threads=%d job_bytes=%zu pf=%d GBps=%.2f", nthr, job_bytes, pf_dist, moved / dt / 1e9);
    if (njobs) printf(" us_per_job=%.3f", dt / njobs * 1e6);
    printf("\n");
    return 0;
}
