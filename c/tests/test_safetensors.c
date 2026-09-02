#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "safetensors.h"
#include "test.h"
#include "util.h"

// write a safetensors file with the given entries.
// entries: name, dtype, ndim, shape, data (bytes, exactly prod(shape)*size)
typedef struct {
    const char *name;
    const char *dtype;
    int ndim;
    int64_t shape[4];
    const uint8_t *data;
    size_t data_len;
} st_entry;

static void write_safetensors(const char *path, const st_entry *entries, size_t n,
                              const char *force_header) {
    // compute offsets and build header
    size_t total = 0;
    char header[65536];
    size_t hp = 0;
    hp += (size_t)snprintf(header + hp, sizeof(header) - hp, "{");
    for (size_t i = 0; i < n; i++) {
        size_t len = entries[i].data_len;
        if (i) hp += (size_t)snprintf(header + hp, sizeof(header) - hp, ",");
        hp += (size_t)snprintf(header + hp, sizeof(header) - hp, "\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                               entries[i].name, entries[i].dtype);
        for (int d = 0; d < entries[i].ndim; d++) {
            if (d) hp += (size_t)snprintf(header + hp, sizeof(header) - hp, ",");
            hp += (size_t)snprintf(header + hp, sizeof(header) - hp, "%lld",
                                   (long long)entries[i].shape[d]);
        }
        hp += (size_t)snprintf(header + hp, sizeof(header) - hp, "],\"data_offsets\":[%zu,%zu]}",
                               total, total + len);
        total += len;
    }
    hp += (size_t)snprintf(header + hp, sizeof(header) - hp, "}");
    if (force_header) {
        hp = strlen(force_header);
        memcpy(header, force_header, hp);
    }
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    CHECK_MSG(fd >= 0, "open %s", path);
    uint64_t hlen = hp;
    CHECK(write(fd, &hlen, 8) == 8);
    CHECK(write(fd, header, hp) == (ssize_t)hp);
    for (size_t i = 0; i < n; i++) {
        CHECK(write(fd, entries[i].data, entries[i].data_len) ==
              (ssize_t)entries[i].data_len);
    }
    close(fd);
}

static uint8_t *pattern_data(size_t n, uint8_t seed) {
    uint8_t *d = xmalloc(n);
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)(seed + i * 7);
    return d;
}

static void test_basic_index(void) {
    uint8_t *a = pattern_data(128, 1);  // [4,16] bf16
    uint8_t *b = pattern_data(96, 9);   // [3,4,8] u8
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/st_basic_%d", "/tmp", getpid());
    mkdir(dir, 0700);
    st_entry entries[] = {
        {"language_model.model.layers.0.weight", "BF16", 2, {4, 16}, a, 128},
        {"language_model.other", "U8", 3, {3, 4, 8}, b, 96},
    };
    char path[600];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    write_safetensors(path, entries, 2, NULL);

    err_t err = {0};
    st_index ix;
    CHECK(st_index_open(&ix, dir, 2, &err) == 0);
    if (err_ok(&err)) {
        st_tensor *t0 = st_find(&ix, "model.layers.0.weight");
        CHECK(t0 != NULL);
        if (t0) {
            CHECK(t0->dtype == ST_BF16);
            CHECK(t0->ndim == 2 && t0->shape[0] == 4 && t0->shape[1] == 16);
            CHECK(t0->nbytes == 128);
            CHECK(st_find(&ix, "other") != NULL);
            CHECK(st_find(&ix, "model.missing") == NULL);
            // read whole tensor
            uint8_t *out = xmalloc(128);
            CHECK(st_read_tensor(&ix, t0, out, &err) == 0);
            CHECK(memcmp(out, a, 128) == 0);
            free(out);
        }
        st_index_close(&ix);
    } else {
        fprintf(stderr, "open failed: %s\n", err.msg);
    }
    free(a);
    free(b);
    unlink(path);
    rmdir(dir);
}

static void test_malformed(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/st_bad_%d", "/tmp", getpid());
    mkdir(dir, 0700);
    // truncated header length
    char path[600];
    snprintf(path, sizeof(path), "%s/bad.safetensors", dir);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    uint64_t hlen = 10000;
    write(fd, &hlen, 8);
    close(fd);
    err_t err = {0};
    st_index ix;
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);
    CHECK(!err_ok(&err));

    // bad dtype
    const uint8_t dummy[16] = {0};
    st_entry bad[] = {{"t", "F32", 1, {4}, dummy, 16}};
    snprintf(path, sizeof(path), "%s/bad2.safetensors", dir);
    write_safetensors(path, bad, 1, "{\"t\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,16]}}");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    // overlapping ranges
    snprintf(path, sizeof(path), "%s/bad3.safetensors", dir);
    write_safetensors(path, bad, 1, "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,24]}}");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    // offsets beyond file
    snprintf(path, sizeof(path), "%s/bad4.safetensors", dir);
    write_safetensors(path, bad, 1, "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,100]}}");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    // mismatched size
    snprintf(path, sizeof(path), "%s/bad5.safetensors", dir);
    write_safetensors(path, bad, 1, "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    // duplicate names
    snprintf(path, sizeof(path), "%s/bad6.safetensors", dir);
    write_safetensors(path, bad, 1, "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},\"language_model.t\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[16,32]}}");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    // invalid json
    snprintf(path, sizeof(path), "%s/bad7.safetensors", dir);
    write_safetensors(path, bad, 1, "{not json");
    memset(&err, 0, sizeof(err));
    CHECK(st_index_open(&ix, dir, 1, &err) != 0);

    unlink(path);
    rmdir(dir);
}

int main(void) {
    RUN_TEST(test_basic_index);
    RUN_TEST(test_malformed);
    return test_summary("safetensors");
}
