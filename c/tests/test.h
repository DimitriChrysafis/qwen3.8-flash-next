#ifndef QWEN_TEST_H
#define QWEN_TEST_H

#include <stdio.h>
#include <stdlib.h>

static int test_failures = 0;
static int test_checks = 0;

#define CHECK(cond)                                                           \
    do {                                                                      \
        test_checks++;                                                        \
        if (!(cond)) {                                                        \
            test_failures++;                                                  \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
        }                                                                     \
    } while (0)

#define CHECK_MSG(cond, ...)                                                  \
    do {                                                                      \
        test_checks++;                                                        \
        if (!(cond)) {                                                        \
            test_failures++;                                                  \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);   \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

#define RUN_TEST(fn)                                                          \
    do {                                                                      \
        int before = test_failures;                                           \
        fn();                                                                 \
        if (test_failures == before)                                          \
            printf("ok   %s\n", #fn);                                         \
        else                                                                  \
            printf("FAIL %s\n", #fn);                                         \
    } while (0)

static inline int test_summary(const char *name) {
    printf("%s: %d checks, %d failures\n", name, test_checks, test_failures);
    return test_failures ? 1 : 0;
}

#endif
