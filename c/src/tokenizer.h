#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer;

// load a tokenizer from a model dir (tokenizer.json). returns NULL when the
// file is missing.
tokenizer *tokenizer_load(const char *dir);

void tokenizer_free(tokenizer *t);

// encode text into token ids; returns count, -1 on error
int tokenizer_encode(tokenizer *t, const char *text, int64_t *ids, size_t cap);

// decode token ids into utf-8; returns malloc'd string
char *tokenizer_decode(tokenizer *t, const int64_t *ids, size_t n);

int64_t tokenizer_eos(const tokenizer *t);

#endif
