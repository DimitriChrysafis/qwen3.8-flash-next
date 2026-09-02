#ifndef QWEN_CACHE_H
#define QWEN_CACHE_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#define CACHE_POLICY_LRU 0
#define CACHE_POLICY_LFU 1
#define CACHE_POLICY_ADAPTIVE 2

typedef struct cache_entry cache_entry;

struct cache_entry {
    uint64_t key; // partition in high 32 bits, item in low 32 bits
    void *value;
    uint64_t nbytes;
    int pinned;
    int hot;
    uint64_t freq;
    uint64_t touched; // monotonic ticks
    cache_entry *prev, *next;
};

typedef struct {
    uint64_t max_bytes;
    int policy;
    int partitions; // 0 = no partition fairness
    cache_entry **slots;
    size_t nslots, count;
    cache_entry *head, *tail; // recency list, head is most recent
    uint64_t bytes, peak_bytes;
    uint64_t hits, misses, evictions, rejected, pinned_count;
    pthread_mutex_t lock;
} cache_t;

void cache_init(cache_t *c, uint64_t max_bytes, int policy, int partitions);
void cache_destroy(cache_t *c);

// look up without touching stats or recency
void *cache_peek(cache_t *c, uint64_t key);

// look up, counts a hit and moves the entry to the front
void *cache_get(cache_t *c, uint64_t key);

// insert or replace. evicts as needed. returns 1 when inserted.
int cache_put(cache_t *c, uint64_t key, void *value, uint64_t nbytes, int pin,
              int hot);

int cache_pin(cache_t *c, uint64_t key);
int cache_unpin(cache_t *c, uint64_t key);
int cache_mark_hot(cache_t *c, uint64_t key, int hot);

typedef struct {
    uint64_t hits, misses, evictions, rejected, pinned;
    uint64_t bytes, peak_bytes, size, hot;
} cache_stats;

cache_stats cache_snapshot(cache_t *c);

#endif
