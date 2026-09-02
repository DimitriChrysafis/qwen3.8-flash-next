#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "reader.h"
#include "safetensors.h"
#include "test.h"
#include "util.h"

// craft a tensor file with known row content: row i is filled with byte i.
static void make_rows_file(const char *path, int64_t nrows, uint64_t row_bytes) {
    char header[512];
    size_t data = (size_t)nrows * row_bytes;
    snprintf(header, sizeof(header),
             "{\"rows\":{\"dtype\":\"U8\",\"shape\":[%lld,%llu],\"data_offsets\":[0,%zu]}}",
             (long long)nrows, (unsigned long long)row_bytes, data);
    FILE *f = fopen(path, "wb");
    uint64_t hlen = strlen(header);
    fwrite(&hlen, 8, 1, f);
    fwrite(header, 1, hlen, f);
    uint8_t *buf = xmalloc(row_bytes);
    for (int64_t i = 0; i < nrows; i++) {
        memset(buf, (int)i, (size_t)row_bytes);
        fwrite(buf, 1, row_bytes, f);
    }
    free(buf);
    fclose(f);
}

static void test_batched_rows(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/rd_%d", "/tmp", getpid());
    mkdir(dir, 0700);
    char path[600];
    snprintf(path, sizeof(path), "%s/m.safetensors", dir);
    make_rows_file(path, 1000, 32);

    err_t err = {0};
    st_index ix;
    CHECK(st_index_open(&ix, dir, 4, &err) == 0);
    st_tensor *t = st_find(&ix, "rows");
    CHECK(t != NULL);
    if (!t) return;

    // out-of-order + duplicates
    int64_t rows[] = {999, 0, 5, 5, 100, 3, 999, 42};
    uint8_t out[8 * 32];
    CHECK(st_read_rows(&ix, t, rows, 8, out, &err) == 0);
    int64_t expect[] = {999, 0, 5, 5, 100, 3, 999, 42};
    for (size_t i = 0; i < 8; i++) {
        for (uint64_t b = 0; b < 32; b++) {
            CHECK_MSG(out[i * 32 + b] == (uint8_t)expect[i], "row %lld byte %llu: %d != %d",
                      (long long)expect[i], (unsigned long long)b, out[i * 32 + b],
                      (uint8_t)expect[i]);
        }
    }

    // single row (byte value wraps mod 256)
    int64_t one[] = {777};
    uint8_t out1[32];
    CHECK(st_read_rows(&ix, t, one, 1, out1, &err) == 0);
    CHECK(out1[0] == 9 && out1[31] == 9);

    // contiguous block read
    int64_t block[4] = {10, 11, 12, 13};
    uint8_t outb[4 * 32];
    CHECK(st_read_rows(&ix, t, block, 4, outb, &err) == 0);
    for (int i = 0; i < 4; i++) CHECK(outb[i * 32] == (uint8_t)(10 + i));

    // out of range
    int64_t bad[] = {1000};
    CHECK(st_read_rows(&ix, t, bad, 1, out1, &err) != 0);
    int64_t bad2[] = {-1};
    memset(&err, 0, sizeof(err));
    CHECK(st_read_rows(&ix, t, bad2, 1, out1, &err) != 0);

    CHECK(ix.rows_read == 8 + 1 + 4);
    st_index_close(&ix);
    unlink(path);
    rmdir(dir);
}

static void test_parallel_reads(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/rdp_%d", "/tmp", getpid());
    mkdir(dir, 0700);
    char path[600];
    snprintf(path, sizeof(path), "%s/m.safetensors", dir);
    make_rows_file(path, 10000, 64);

    err_t err = {0};
    st_index ix;
    CHECK(st_index_open(&ix, dir, 6, &err) == 0);
    st_tensor *t = st_find(&ix, "rows");
    CHECK(t != NULL);

    // submit 64 interleaved reads from the main thread
    enum { N = 64 };
    int64_t rows[N];
    uint8_t *outs[N];
    io_job *jobs[N];
    for (int i = 0; i < N; i++) {
        rows[i] = (i * 137) % 10000;
        outs[i] = xmalloc(64);
        io_job *job = xcalloc(1, sizeof(io_job));
        job->fd = ix.fds[t->file];
        job->base = t->offset;
        job->row_bytes = 64;
        job->rows = &rows[i];
        job->nrows = 1;
        job->out = outs[i];
        io_pool_submit(&ix.pool, job);
        jobs[i] = job;
    }
    for (int i = 0; i < N; i++) {
        CHECK(io_job_wait(jobs[i]) == 0);
        CHECK(outs[i][0] == (uint8_t)((i * 137) % 10000));
        CHECK(outs[i][63] == (uint8_t)((i * 137) % 10000));
        free(jobs[i]);
        free(outs[i]);
    }
    st_index_close(&ix);
    unlink(path);
    rmdir(dir);
}

int main(void) {
    RUN_TEST(test_batched_rows);
    RUN_TEST(test_parallel_reads);
    return test_summary("reader");
}
