// memheat: keep the memory fabric busy from one spare core while the model decodes (FA cold-fetch test).
// Usage (placement from the caller): numactl --physcpubind=<cpu> --membind=<node> memheat <mode> <arg> [seconds]
//   tick <ns>    one dependent load of a random cache line of a 1 GiB buffer every <ns> ns (~64 B per tick)
//   stream <mb>  sequential reads over a <mb> MiB buffer as fast as one core can
// Prints the achieved rate every 10 s to stderr. Stops after [seconds] (default 3600) or on SIGTERM.
#define _GNU_SOURCE
#include <immintrin.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <x86intrin.h>

static volatile sig_atomic_t stop;
static void on_term(int s) { (void)s; stop = 1; }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: memheat tick <ns> | stream <mb> [seconds]\n"); return 1; }
    const int tick = strcmp(argv[1], "tick") == 0;
    const long arg = atol(argv[2]);
    const double secs = argc > 3 ? atof(argv[3]) : 3600.0;
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);

    const size_t n = tick ? (size_t)1 << 30 : (size_t)arg << 20;
    uint8_t * buf = aligned_alloc(2u << 20, n);
    madvise(buf, n, MADV_HUGEPAGE);
    memset(buf, 1, n);

    // calibrate TSC ticks per ns
    const double t0 = now(); const uint64_t c0 = __rdtsc();
    while (now() - t0 < 0.05) {}
    const double tpn = (double)(__rdtsc() - c0)/((now() - t0)*1e9);

    uint64_t x = 88172645463325252ull, sink = 0, ops = 0;
    double tr = now(), tend = tr + secs;
    while (!stop && now() < tend) {
        if (tick) {
            const uint64_t gap = (uint64_t)(arg*tpn);
            for (int i = 0; i < 1000 && !stop; ++i) {
                const uint64_t ts = __rdtsc();
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                sink += buf[(x % (n/64))*64 + (sink & 1)];
                ++ops;
                while (__rdtsc() - ts < gap) _mm_pause();
            }
        } else {
            __m256i acc = _mm256_setzero_si256();
            for (size_t o = 0; o < n && !stop; o += 64) acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(buf + o)));
            sink += (uint64_t)_mm256_extract_epi64(acc, 0);
            ops += n/64;
        }
        const double t = now();
        if (t - tr >= 10.0) {
            fprintf(stderr, "memheat %s: %.1f MB/s (%.0f lines/s)\n", argv[1], ops*64/(t - tr)/1e6, ops/(t - tr));
            ops = 0; tr = t;
        }
    }
    printf("%llu\n", (unsigned long long)(sink & 1));
    return 0;
}
