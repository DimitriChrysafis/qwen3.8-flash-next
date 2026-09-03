#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "cache.h"
#include "ngram.h"
#include "safetensors.h"
#include "util.h"

#define QMODEL_MAX_LAYERS 128
#define QMODEL_MAX_HEADS 64
#define QMODEL_KV_CAP 2048

typedef struct qmodel qmodel;

typedef struct {
    uint64_t expert_loads, expert_bytes;
    uint64_t ple_loads, ple_bytes;
    double expert_wait_seconds, ple_wait_seconds;
    uint64_t prefill_calls, decode_calls;
    double forward_seconds;
} qstats;

// load config.json + all non-streamed weights. expert/ple rows stay on disk.
qmodel *model_load(const char *dir, int io_workers, uint64_t expert_budget,
                   uint64_t ple_budget, err_t *err);

void model_free(qmodel *m);

// run the model on `n` tokens (n >= 1). when `decode` is set, the caches are
// updated incrementally and exactly one token is expected. logits must hold
// n * vocab floats.
int model_forward(qmodel *m, const int64_t *tokens, size_t n, int decode,
                  float *logits, err_t *err);

// reset kv and state caches (fresh context)
void model_reset_caches(qmodel *m);

int64_t model_vocab(const qmodel *m);
int64_t model_eos(const qmodel *m);
const qstats *model_stats(const qmodel *m);

// tokenizer is a separate module; the model exposes a register hook so the
// cli can attach one for text mode.
typedef struct tokenizer tokenizer;
void model_set_tokenizer(qmodel *m, tokenizer *tok);

#endif
