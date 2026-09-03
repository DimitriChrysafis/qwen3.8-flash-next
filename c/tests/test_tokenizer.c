#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "tokenizer.h"
#include "util.h"

// tiny tokenizer.json with the same byte-unicode vocab layout as the real
// qwen files: space is "Ġ" (u+0120), merges are "left right" strings
static const char *TINY_JSON =
    "{\n"
    "  \"version\": \"1.0\",\n"
    "  \"added_tokens\": [\n"
    "    {\"id\": 10, \"content\": \"<|endoftext|>\", \"single_word\": false,\n"
    "     \"lstrip\": false, \"rstrip\": false, \"normalized\": false,\n"
    "     \"special\": true},\n"
    "    {\"id\": 11, \"content\": \"<|im_end|>\", \"single_word\": false,\n"
    "     \"lstrip\": false, \"rstrip\": false, \"normalized\": false,\n"
    "     \"special\": true}\n"
    "  ],\n"
    "  \"model\": {\n"
    "    \"type\": \"BPE\",\n"
    "    \"vocab\": {\n"
    "      \"!\": 0, \"H\": 1, \"i\": 2, \"\\u0120\": 3, \"Hi\": 4,\n"
    "      \"\\u0120i\": 5, \"\\u0120Hi\": 6, \"\\u00c3\": 7, \"\\u00a9\": 8,\n"
    "      \"x\": 9\n"
    "    },\n"
    "    \"merges\": [\n"
    "      \"H i\",\n"
    "      \"\\u0120 i\",\n"
    "      \"\\u0120 Hi\"\n"
    "    ]\n"
    "  }\n"
    "}\n";

static char *write_tiny_tokenizer(void) {
    const char *dir = "/tmp/qwen_tok_test";
    char path[256];
    snprintf(path, sizeof(path), "mkdir -p %s", dir);
    if (system(path) != 0) return NULL;
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    fwrite(TINY_JSON, 1, strlen(TINY_JSON), f);
    fclose(f);
    return strdup(dir);
}

static void test_load_and_roundtrip(void) {
    char *dir = write_tiny_tokenizer();
    CHECK(dir != NULL);
    if (!dir) return;
    tokenizer *t = tokenizer_load(dir);
    CHECK_MSG(t != NULL, "load failed");
    if (!t) {
        free(dir);
        return;
    }
    CHECK(tokenizer_eos(t) == 10);

    int64_t ids[64];
    int n = tokenizer_encode(t, "Hi", ids, 64);
    CHECK(n == 1 && ids[0] == 4); // "Hi" merged
    n = tokenizer_encode(t, " i", ids, 64);
    CHECK(n == 1 && ids[0] == 5); // "Ġi" merged
    n = tokenizer_encode(t, " Hi", ids, 64);
    CHECK(n == 1 && ids[0] == 6); // "ĠHi" merged twice
    n = tokenizer_encode(t, "Hi i", ids, 64);
    CHECK(n == 2 && ids[0] == 4 && ids[1] == 5); // pre-token split
    n = tokenizer_encode(t, "x!", ids, 64);
    CHECK(n == 2 && ids[0] == 9 && ids[1] == 0); // punct split

    // utf-8: é is two bytes -> "Ã" "©" (unmerged)
    n = tokenizer_encode(t, "\xc3\xa9", ids, 64);
    CHECK(n == 2 && ids[0] == 7 && ids[1] == 8);

    // decode roundtrip
    n = tokenizer_encode(t, "Hi i!\xc3\xa9", ids, 64);
    char *s = tokenizer_decode(t, ids, (size_t)n);
    CHECK(s && strcmp(s, "Hi i!\xc3\xa9") == 0);
    free(s);

    // added tokens decode as raw text
    int64_t special[2] = {4, 11};
    s = tokenizer_decode(t, special, 2);
    CHECK(s && strcmp(s, "Hi<|im_end|>") == 0);
    free(s);

    // unknown id is skipped, not crashed on
    int64_t bad[2] = {4, 999};
    s = tokenizer_decode(t, bad, 2);
    CHECK(s && strcmp(s, "Hi") == 0);
    free(s);

    tokenizer_free(t);
    free(dir);
}

static void test_cap_overflow(void) {
    char *dir = write_tiny_tokenizer();
    if (!dir) return;
    tokenizer *t = tokenizer_load(dir);
    CHECK(t != NULL);
    if (!t) {
        free(dir);
        return;
    }
    int64_t ids[1];
    CHECK(tokenizer_encode(t, "Hi i", ids, 1) == -1); // needs 2, cap 1
    CHECK(tokenizer_encode(t, "Hi", ids, 1) == 1);
    tokenizer_free(t);
    free(dir);
}

int main(void) {
    RUN_TEST(test_load_and_roundtrip);
    RUN_TEST(test_cap_overflow);
    return test_summary("test_tokenizer");
}
