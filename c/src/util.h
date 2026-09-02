#ifndef QWEN_UTIL_H
#define QWEN_UTIL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char msg[512];
} err_t;

#define err_set(e, ...)                                                       \
    do {                                                                      \
        if (e) snprintf((e)->msg, sizeof((e)->msg), __VA_ARGS__);             \
    } while (0)

#define err_clear(e)                                                          \
    do {                                                                      \
        if (e) (e)->msg[0] = 0;                                               \
    } while (0)

static inline int err_ok(const err_t *e) {
    return e == NULL || e->msg[0] == 0;
}

static inline void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

static inline void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) {
        fprintf(stderr, "out of memory (%zu x %zu bytes)\n", n, sz);
        abort();
    }
    return p;
}

static inline void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t r = (bits >> 16) & 0xffff;
    uint32_t rem = bits & 0xffff;
    uint32_t lsb = r & 1;
    if (rem > 0x8000 || (rem == 0x8000 && lsb)) r++;
    return (uint16_t)r;
}

static inline double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif
