// placeholder tokenizer; real BPE lands in a later commit.
#include "tokenizer.h"

#include <stdlib.h>

#include "util.h"

struct tokenizer {
    int dummy;
};

tokenizer *tokenizer_load(const char *dir) {
    (void)dir;
    return NULL;
}

void tokenizer_free(tokenizer *t) {
    free(t);
}

int tokenizer_encode(tokenizer *t, const char *text, int64_t *ids, size_t cap) {
    (void)t;
    (void)text;
    (void)ids;
    (void)cap;
    return -1;
}

char *tokenizer_decode(tokenizer *t, const int64_t *ids, size_t n) {
    (void)t;
    (void)ids;
    (void)n;
    return NULL;
}

int64_t tokenizer_eos(const tokenizer *t) {
    (void)t;
    return 0;
}
