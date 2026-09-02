#include <pthread.h>
#include <stdlib.h>

#include "cache.h"
#include "test.h"

static void *dummy(uint64_t v) {
    return (void *)(uintptr_t)(v + 1);
}

static void test_basic(void) {
    cache_t c;
    cache_init(&c, 1000, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 0, 0) == 1);
    CHECK(cache_put(&c, 2, dummy(2), 100, 0, 0) == 1);
    CHECK(cache_peek(&c, 1) == dummy(1));
    CHECK(cache_get(&c, 2) == dummy(2));
    CHECK(cache_get(&c, 3) == NULL);
    cache_stats s = cache_snapshot(&c);
    CHECK(s.hits == 1 && s.misses == 1 && s.size == 2 && s.bytes == 200);
    cache_destroy(&c);
}

static void test_lru(void) {
    cache_t c;
    cache_init(&c, 300, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 0, 0));
    CHECK(cache_put(&c, 2, dummy(2), 100, 0, 0));
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0));
    // touch 1 so 2 becomes the oldest
    CHECK(cache_get(&c, 1) == dummy(1));
    CHECK(cache_put(&c, 4, dummy(4), 100, 0, 0));
    CHECK(cache_peek(&c, 2) == NULL); // evicted
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 3) != NULL);
    CHECK(cache_peek(&c, 4) != NULL);
    cache_destroy(&c);
}

static void test_lfu(void) {
    cache_t c;
    cache_init(&c, 300, CACHE_POLICY_LFU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 0, 0));
    CHECK(cache_put(&c, 2, dummy(2), 100, 0, 0));
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0));
    cache_get(&c, 1);
    cache_get(&c, 1);
    cache_get(&c, 2);
    CHECK(cache_put(&c, 4, dummy(4), 100, 0, 0));
    CHECK(cache_peek(&c, 3) == NULL); // lowest frequency evicted
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 2) != NULL);
    cache_destroy(&c);
}

static void test_adaptive(void) {
    cache_t c;
    cache_init(&c, 300, CACHE_POLICY_ADAPTIVE, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 0, 1)); // hot, small
    CHECK(cache_put(&c, 2, dummy(2), 200, 0, 0)); // cold, large
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0));
    // 300 budget: adding 3 (100) evicts the worst: 2 (cold, big, score 0)
    CHECK(cache_peek(&c, 2) == NULL);
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 3) != NULL);
    cache_destroy(&c);
}

static void test_pin(void) {
    cache_t c;
    cache_init(&c, 200, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 1, 0)); // pinned
    CHECK(cache_put(&c, 2, dummy(2), 100, 0, 0));
    // pin the second too: now nothing is evictable
    CHECK(cache_pin(&c, 2) == 1);
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0) == 0);
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 2) != NULL);
    CHECK(cache_peek(&c, 3) == NULL);
    // unpin one and retry: evicts the unpinned entry
    CHECK(cache_unpin(&c, 2) == 1);
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0) == 1);
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 2) == NULL);
    CHECK(cache_peek(&c, 3) != NULL);
    cache_stats s = cache_snapshot(&c);
    CHECK(s.pinned == 1);
    CHECK(s.rejected == 1);
    cache_destroy(&c);
}

static void test_hot(void) {
    cache_t c;
    cache_init(&c, 200, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 0, 1)); // hot
    CHECK(cache_put(&c, 2, dummy(2), 100, 0, 0));
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0));
    // cold entries evicted first: 2 is the oldest cold entry
    CHECK(cache_peek(&c, 1) != NULL);
    CHECK(cache_peek(&c, 2) == NULL);
    CHECK(cache_peek(&c, 3) != NULL);
    cache_destroy(&c);
}

static void test_partitions(void) {
    cache_t c;
    cache_init(&c, 200, CACHE_POLICY_LRU, 2);
    // partition 0: two entries (fair share 100 each total 100)
    CHECK(cache_put(&c, 0x100000001ull, dummy(1), 100, 0, 0));
    CHECK(cache_put(&c, 0x100000002ull, dummy(2), 100, 0, 0));
    // partition 1: one entry
    CHECK(cache_put(&c, 0x200000001ull, dummy(3), 100, 0, 0));
    // budget 200: partition 0 already at 200 (over fair 100), partition 1 at 100
    CHECK(cache_put(&c, 0x200000002ull, dummy(4), 100, 0, 0));
    // partition 1 has usage 100 <= fair 100, so victim comes from partition 0
    CHECK(cache_peek(&c, 0x100000001ull) == NULL || cache_peek(&c, 0x100000002ull) == NULL);
    CHECK(cache_peek(&c, 0x200000001ull) != NULL);
    CHECK(cache_peek(&c, 0x200000002ull) != NULL);
    cache_destroy(&c);
}

static void test_replace(void) {
    cache_t c;
    cache_init(&c, 1000, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 100, 1, 0));
    // replacing keeps pin
    CHECK(cache_put(&c, 1, dummy(9), 100, 0, 0));
    CHECK(cache_peek(&c, 1) == dummy(9));
    cache_stats s = cache_snapshot(&c);
    CHECK(s.pinned == 1);
    CHECK(s.size == 1 && s.bytes == 100);
    cache_destroy(&c);
}

static void test_oversized(void) {
    cache_t c;
    cache_init(&c, 100, CACHE_POLICY_LRU, 0);
    CHECK(cache_put(&c, 1, dummy(1), 200, 0, 0) == 0);
    CHECK(cache_put(&c, 2, dummy(2), 50, 0, 0) == 1);
    CHECK(cache_put(&c, 3, dummy(3), 100, 0, 0) == 1); // evicts 2
    CHECK(cache_peek(&c, 2) == NULL);
    cache_destroy(&c);
}

// hash table stress: insert many, verify findability
static void test_table_stress(void) {
    cache_t c;
    cache_init(&c, 1ull << 40, CACHE_POLICY_LRU, 0); // effectively unbounded
    enum { N = 2000 };
    uint64_t keys[N];
    for (int i = 0; i < N; i++) {
        keys[i] = ((uint64_t)(i * 2654435761u) << 32) ^ (uint32_t)(i * 40503u);
        CHECK(cache_put(&c, keys[i], dummy(keys[i]), 1, 0, 0));
    }
    for (int round = 0; round < 5; round++) {
        for (int i = 0; i < N; i++) {
            CHECK(cache_peek(&c, keys[i]) == dummy(keys[i]));
        }
    }
    cache_destroy(&c);
}

// actual delete path exercised through eviction with a small budget
static void test_table_evict_delete(void) {
    enum { N = 600 };
    cache_t c;
    cache_init(&c, N / 2, CACHE_POLICY_LRU, 0);
    uint64_t keys[N];
    for (int i = 0; i < N; i++) {
        keys[i] = ((uint64_t)(i * 2654435761u) << 32) ^ (uint32_t)(i * 40503u);
        cache_put(&c, keys[i], dummy(keys[i]), 1, 0, 0);
    }
    cache_stats s = cache_snapshot(&c);
    CHECK(s.size == (uint64_t)N / 2);
    CHECK(s.bytes == (uint64_t)N / 2);
    // every surviving key must be findable, evicted ones must not
    for (int i = 0; i < N; i++) {
        if (i < N / 2) CHECK(cache_peek(&c, keys[i]) == NULL);
        else CHECK(cache_peek(&c, keys[i]) == dummy(keys[i]));
    }
    // churn: insert new keys and verify lookups still work
    for (int i = 0; i < N; i++) {
        cache_put(&c, keys[i] ^ 0xdeadbeef, dummy(keys[i]), 1, 0, 0);
    }
    for (int i = 0; i < N; i++) {
        // only the last N/2 inserted survive
        if (i >= N / 2) {
            CHECK(cache_peek(&c, keys[i] ^ 0xdeadbeef) == dummy(keys[i]));
        } else {
            CHECK(cache_peek(&c, keys[i] ^ 0xdeadbeef) == NULL);
        }
        // keys from the first wave are all evicted by now
        CHECK(cache_peek(&c, keys[i]) == NULL);
    }
    cache_destroy(&c);
}

static void *thread_worker(void *arg) {
    cache_t *c = arg;
    for (uint64_t i = 0; i < 5000; i++) {
        uint64_t key = (i % 100) + 1;
        cache_put(c, key, dummy(key), 10, 0, 0);
        cache_get(c, key);
        cache_peek(c, key);
    }
    return NULL;
}

static void test_concurrent(void) {
    cache_t c;
    cache_init(&c, 500, CACHE_POLICY_ADAPTIVE, 4);
    pthread_t threads[8];
    for (int i = 0; i < 8; i++) pthread_create(&threads[i], NULL, thread_worker, &c);
    for (int i = 0; i < 8; i++) pthread_join(threads[i], NULL);
    cache_stats s = cache_snapshot(&c);
    CHECK(s.bytes <= 500);
    cache_destroy(&c);
}

int main(void) {
    RUN_TEST(test_basic);
    RUN_TEST(test_lru);
    RUN_TEST(test_lfu);
    RUN_TEST(test_adaptive);
    RUN_TEST(test_pin);
    RUN_TEST(test_hot);
    RUN_TEST(test_partitions);
    RUN_TEST(test_replace);
    RUN_TEST(test_oversized);
    RUN_TEST(test_table_stress);
    RUN_TEST(test_table_evict_delete);
    RUN_TEST(test_concurrent);
    return test_summary("cache");
}
