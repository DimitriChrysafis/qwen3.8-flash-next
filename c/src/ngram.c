#include "ngram.h"

#include <math.h>
#include <stdlib.h>

#include "util.h"

static const uint64_t GAMMA = 0x9E3779B97F4A7C15ull;
static const uint64_t M1 = 0xBF58476D1CE4E5B9ull;
static const uint64_t M2 = 0x94D049BB133111EBull;

uint64_t ngram_splitmix64(uint64_t v) {
    v = (v + GAMMA) & 0xFFFFFFFFFFFFFFFFull;
    v = ((v ^ (v >> 30)) * M1) & 0xFFFFFFFFFFFFFFFFull;
    v = ((v ^ (v >> 27)) * M2) & 0xFFFFFFFFFFFFFFFFull;
    return (v ^ (v >> 31)) & 0xFFFFFFFFFFFFFFFFull;
}

static int is_prime(int64_t v) {
    if (v < 2) return 0;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2) {
        if (v % d == 0) return 0;
    }
    return 1;
}

static int64_t nth_prime_after(int64_t start, int64_t count) {
    int64_t p = start;
    for (int64_t i = 0; i < count; i++) {
        p++;
        while (!is_prime(p)) p++;
    }
    return p;
}

void ngram_geo_compute(ngram_geo *g, int64_t ngram_vocab_size_base,
                       int64_t seed, int ple_layer_index) {
    g->ngram_heads = (g->ngram_size - 1) * g->heads_per_ngram;
    if (g->ngram_heads > NGRAM_MAX_HEADS) {
        fprintf(stderr, "ngram_geo_compute: too many heads\n");
        abort();
    }
    int64_t offset = 0;
    for (int h = 0; h < g->ngram_heads; h++) {
        int64_t size = nth_prime_after(
            ngram_vocab_size_base - 1, ple_layer_index * g->ngram_heads + h + 1);
        g->sizes[h] = size;
        g->offsets[h] = offset;
        offset += size;
    }
    g->total_rows = offset;
    int64_t half = ((int64_t)((1ull << 63) - 1) / (g->vocab_size > 0 ? g->vocab_size : 1)) / 2;
    if (half < 1) half = 1;
    uint64_t s = (uint64_t)seed;
    for (int i = 0; i < g->ngram_size; i++) {
        uint64_t v = (s + GAMMA * (uint64_t)(i + 1)) & 0xFFFFFFFFFFFFFFFFull;
        g->multipliers[i] = 2 * (ngram_splitmix64(v) % (uint64_t)half) + 1;
    }
}

void ngram_indices(const ngram_geo *g, const int64_t *tokens, size_t n,
                   int64_t *out) {
    int hpn = g->heads_per_ngram;
    int64_t *prev = xmalloc(n * sizeof(int64_t));
    int64_t last_eos = -1;
    for (size_t t = 0; t < n; t++) {
        prev[t] = last_eos;
        if (tokens[t] == g->eos_token_id) last_eos = (int64_t)t;
    }
    for (size_t t = 0; t < n; t++) {
        int64_t *o = out + t * g->ngram_heads;
        for (int p = 1; p < g->ngram_size; p++) {
            int64_t mixed = 0;
            for (int q = 0; q <= p; q++) {
                int64_t tok = g->eos_token_id;
                int64_t src = (int64_t)t - q;
                // shift-right: use tokens[t-q] only when t-q >= 0 and the
                // previous q tokens do not cross an eos boundary
                if (src >= 0 && (int64_t)t - (prev[t] + 1) >= q) {
                    tok = tokens[src];
                }
                mixed ^= tok * g->multipliers[q];
            }
            for (int h = 0; h < hpn; h++) {
                int64_t head = (p - 1) * hpn + h;
                int64_t m = mixed % g->sizes[head];
                if (m < 0) m += g->sizes[head];
                o[head] = m + g->offsets[head];
            }
        }
    }
    free(prev);
}
