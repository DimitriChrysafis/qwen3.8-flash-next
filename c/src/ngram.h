#ifndef QWEN_NGRAM_H
#define QWEN_NGRAM_H

#include <stddef.h>
#include <stdint.h>

#define NGRAM_MAX_HEADS 16

// n-gram embedding table geometry. sizes/offsets/multipliers are stored in
// the model file but can be recomputed for validation.
typedef struct {
    int ngram_size;         // 3
    int heads_per_ngram;    // 8
    int ngram_heads;        // (ngram_size - 1) * heads_per_ngram
    int64_t vocab_size;     // token vocab
    int64_t eos_token_id;
    int64_t sizes[NGRAM_MAX_HEADS];
    int64_t offsets[NGRAM_MAX_HEADS];
    int64_t total_rows;
    int64_t multipliers[8]; // ngram_size entries
} ngram_geo;

// recompute sizes/offsets/multipliers from config (for validation and
// models that do not store them)
void ngram_geo_compute(ngram_geo *g, int64_t ngram_vocab_size_base,
                       int64_t seed, int ple_layer_index);

// compute the gid for each head for a sequence of tokens. history has length
// n (last ngram_size - 1 tokens followed by the current tokens). out has
// ngram_heads entries per position.
void ngram_indices(const ngram_geo *g, const int64_t *tokens, size_t n,
                   int64_t *out);

// splitmix64 (mlx-compatible)
uint64_t ngram_splitmix64(uint64_t v);

#endif
