#include "cache.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util.h"

static uint64_t now_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t hash_key(uint64_t key) {
    // splitmix64 finalizer
    uint64_t z = key + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)(z ^ (z >> 31));
}

static size_t slot_for(cache_t *c, uint64_t key) {
    return (size_t)hash_key(key) & (c->nslots - 1);
}

static cache_entry *find_slot(cache_t *c, uint64_t key) {
    size_t i = slot_for(c, key);
    while (c->slots[i]) {
        if (c->slots[i]->key == key) return c->slots[i];
        i = (i + 1) & (c->nslots - 1);
    }
    return NULL;
}

static void unlink_entry(cache_t *c, cache_entry *e) {
    if (e->prev) e->prev->next = e->next;
    else c->head = e->next;
    if (e->next) e->next->prev = e->prev;
    else c->tail = e->prev;
    e->prev = e->next = NULL;
}

static void link_front(cache_t *c, cache_entry *e) {
    e->prev = NULL;
    e->next = c->head;
    if (c->head) c->head->prev = e;
    c->head = e;
    if (!c->tail) c->tail = e;
}

// remove e from the open-addressing table, backshifting cluster entries so
// probing stays correct. entries whose home sits inside (i, j] stay put;
// everything else shifts into the hole. the scan only stops at an empty slot.
static void remove_from_table(cache_t *c, cache_entry *e) {
    size_t i = slot_for(c, e->key);
    while (c->slots[i] && c->slots[i] != e) i = (i + 1) & (c->nslots - 1);
    if (!c->slots[i]) return;
    c->slots[i] = NULL;
    size_t j = (i + 1) & (c->nslots - 1);
    while (c->slots[j]) {
        size_t h = slot_for(c, c->slots[j]->key);
        // entry at j can move into the hole at i iff its home h is not
        // strictly within the circular range (i, j]
        size_t dist_ij = (j - i) & (c->nslots - 1);
        size_t dist_ih = (h - i) & (c->nslots - 1);
        if (dist_ih == 0 || dist_ih > dist_ij) {
            c->slots[i] = c->slots[j];
            c->slots[j] = NULL;
            i = j;
        }
        j = (j + 1) & (c->nslots - 1);
    }
}

static uint32_t partition_of(uint64_t key) {
    return (uint32_t)(key >> 32);
}

// grow the table when the load factor gets high so probing always finds an
// empty slot.
static void grow_table(cache_t *c) {
    size_t ncap = c->nslots * 2;
    cache_entry **ns = xcalloc(ncap, sizeof(cache_entry *));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < c->nslots; i++) {
        cache_entry *e = c->slots[i];
        if (!e) continue;
        size_t j = (size_t)hash_key(e->key) & mask;
        while (ns[j]) j = (j + 1) & mask;
        ns[j] = e;
    }
    free(c->slots);
    c->slots = ns;
    c->nslots = ncap;
}

// choose a victim, or NULL when nothing evictable exists.
static cache_entry *pick_victim(cache_t *c) {
    cache_entry *best = NULL;
    double best_score = 0;
    uint64_t best_touched = 0;
    uint64_t fair = c->partitions > 0 ? c->max_bytes / (uint64_t)c->partitions : 0;
    // partition fairness only kicks in when some partition is over its share
    int any_over = 0;
    if (c->partitions > 0) {
        for (cache_entry *e = c->tail; e; e = e->prev) {
            if (e->pinned) continue;
            uint64_t usage = 0;
            uint32_t part = partition_of(e->key);
            for (cache_entry *x = c->head; x; x = x->next) {
                if (partition_of(x->key) == part) usage += x->nbytes;
            }
            if (usage > fair) {
                any_over = 1;
                break;
            }
        }
    }
    for (cache_entry *e = c->tail; e; e = e->prev) {
        if (e->pinned) continue;
        if (c->partitions > 0 && any_over) {
            uint64_t usage = 0;
            uint32_t part = partition_of(e->key);
            for (cache_entry *x = c->head; x; x = x->next) {
                if (partition_of(x->key) == part) usage += x->nbytes;
            }
            if (usage <= fair) continue; // partition under its fair share
        }
        // cold entries preferred over hot ones
        if (best && !e->hot && best->hot) {
            best = e;
            best_score = 0;
            best_touched = e->touched;
            continue;
        }
        if (best && e->hot && !best->hot) continue;
        double score;
        if (c->policy == CACHE_POLICY_LRU) {
            score = 0;
        } else if (c->policy == CACHE_POLICY_LFU) {
            score = (double)e->freq;
        } else {
            score = (double)(e->freq + (e->hot ? 4 : 0)) /
                    (double)(e->nbytes ? e->nbytes : 1);
        }
        int better;
        if (!best) {
            better = 1;
        } else if (c->policy == CACHE_POLICY_LRU) {
            better = e->touched < best_touched;
        } else {
            better = score < best_score ||
                     (score == best_score && e->touched < best_touched);
        }
        if (better) {
            best = e;
            best_score = score;
            best_touched = e->touched;
        }
    }
    return best;
}

void cache_init(cache_t *c, uint64_t max_bytes, int policy, int partitions) {
    memset(c, 0, sizeof(*c));
    c->max_bytes = max_bytes;
    c->policy = policy;
    c->partitions = partitions;
    c->nslots = 1024;
    c->slots = xcalloc(c->nslots, sizeof(cache_entry *));
    pthread_mutex_init(&c->lock, NULL);
}

void cache_destroy(cache_t *c) {
    for (size_t i = 0; i < c->nslots; i++) free(c->slots[i]);
    free(c->slots);
    pthread_mutex_destroy(&c->lock);
    memset(c, 0, sizeof(*c));
}

void *cache_peek(cache_t *c, uint64_t key) {
    pthread_mutex_lock(&c->lock);
    cache_entry *e = find_slot(c, key);
    void *v = e ? e->value : NULL;
    pthread_mutex_unlock(&c->lock);
    return v;
}

void *cache_get(cache_t *c, uint64_t key) {
    pthread_mutex_lock(&c->lock);
    cache_entry *e = find_slot(c, key);
    void *v = NULL;
    if (e) {
        e->freq++;
        e->touched = now_ticks();
        unlink_entry(c, e);
        link_front(c, e);
        c->hits++;
        v = e->value;
    } else {
        c->misses++;
    }
    pthread_mutex_unlock(&c->lock);
    return v;
}

int cache_put(cache_t *c, uint64_t key, void *value, uint64_t nbytes, int pin,
              int hot) {
    pthread_mutex_lock(&c->lock);
    if (c->max_bytes > 0 && nbytes > c->max_bytes) {
        c->rejected++;
        pthread_mutex_unlock(&c->lock);
        return 0;
    }
    cache_entry *old = find_slot(c, key);
    if (old) {
        pin = pin || old->pinned;
        hot = hot || old->hot;
        c->bytes -= old->nbytes;
        if (old->pinned) c->pinned_count--;
        unlink_entry(c, old);
        remove_from_table(c, old);
        free(old);
        c->count--;
    }
    uint64_t evicted = 0;
    while (c->max_bytes > 0 && c->bytes + nbytes > c->max_bytes) {
        cache_entry *victim = pick_victim(c);
        if (!victim) {
            c->rejected++;
            pthread_mutex_unlock(&c->lock);
            return 0;
        }
        c->bytes -= victim->nbytes;
        if (victim->pinned) c->pinned_count--;
        unlink_entry(c, victim);
        remove_from_table(c, victim);
        free(victim);
        c->count--;
        evicted++;
    }
    cache_entry *e = xcalloc(1, sizeof(cache_entry));
    if (c->count + 1 > c->nslots * 3 / 4) grow_table(c);
    e->key = key;
    e->value = value;
    e->nbytes = nbytes;
    e->pinned = pin;
    e->hot = hot;
    e->freq = 1;
    e->touched = now_ticks();
    size_t i = slot_for(c, key);
    while (c->slots[i]) i = (i + 1) & (c->nslots - 1);
    c->slots[i] = e;
    link_front(c, e);
    c->bytes += nbytes;
    if (c->bytes > c->peak_bytes) c->peak_bytes = c->bytes;
    c->count++;
    if (pin) c->pinned_count++;
    c->evictions += evicted;
    pthread_mutex_unlock(&c->lock);
    return 1;
}

int cache_pin(cache_t *c, uint64_t key) {
    pthread_mutex_lock(&c->lock);
    cache_entry *e = find_slot(c, key);
    int ok = 0;
    if (e && !e->pinned) {
        e->pinned = 1;
        c->pinned_count++;
        ok = 1;
    }
    pthread_mutex_unlock(&c->lock);
    return ok;
}

int cache_unpin(cache_t *c, uint64_t key) {
    pthread_mutex_lock(&c->lock);
    cache_entry *e = find_slot(c, key);
    int ok = 0;
    if (e && e->pinned) {
        e->pinned = 0;
        c->pinned_count--;
        ok = 1;
    }
    pthread_mutex_unlock(&c->lock);
    return ok;
}

int cache_mark_hot(cache_t *c, uint64_t key, int hot) {
    pthread_mutex_lock(&c->lock);
    cache_entry *e = find_slot(c, key);
    int ok = 0;
    if (e) {
        e->hot = hot;
        ok = 1;
    }
    pthread_mutex_unlock(&c->lock);
    return ok;
}

cache_stats cache_snapshot(cache_t *c) {
    pthread_mutex_lock(&c->lock);
    cache_stats s = {
        .hits = c->hits,
        .misses = c->misses,
        .evictions = c->evictions,
        .rejected = c->rejected,
        .pinned = c->pinned_count,
        .bytes = c->bytes,
        .peak_bytes = c->peak_bytes,
        .size = c->count,
    };
    for (cache_entry *e = c->head; e; e = e->next) {
        if (e->hot) s.hot++;
    }
    pthread_mutex_unlock(&c->lock);
    return s;
}
