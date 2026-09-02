#ifndef QWEN_SAFETENSORS_H
#define QWEN_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#include "reader.h"
#include "util.h"

typedef enum {
    ST_BOOL,
    ST_U8,
    ST_I8,
    ST_U16,
    ST_I16,
    ST_U32,
    ST_I32,
    ST_U64,
    ST_I64,
    ST_F16,
    ST_BF16,
    ST_F32,
    ST_F64,
    ST_COUNT,
} st_dtype;

typedef struct {
    char *name; // normalized name, owned
    int file;   // index into st_index.paths
    st_dtype dtype;
    size_t ndim;
    int64_t shape[8];
    uint64_t offset; // absolute data offset in file
    uint64_t nbytes;
} st_tensor;

typedef struct {
    char **paths;
    int *fds;
    int nfiles;
    st_tensor *tensors;
    size_t ntensors, cap;
    io_pool pool;
    // io stats
    uint64_t pread_calls;
    uint64_t bytes_read;
    uint64_t rows_read;
    uint64_t tensors_read;
    double read_seconds;
} st_index;

int st_dtype_size(st_dtype d);

// scan dir for *.safetensors and index all tensors. opens fds and starts the
// io pool. returns 0 on success.
int st_index_open(st_index *ix, const char *dir, int io_workers, err_t *err);

// release fds, pool threads, tensor table.
void st_index_close(st_index *ix);

st_tensor *st_find(st_index *ix, const char *name);

// read an entire tensor into a caller-allocated buffer of nbytes.
int st_read_tensor(st_index *ix, st_tensor *t, void *out, err_t *err);

// read rows of a tensor into out (nrows * row_bytes). preserves request
// order and dedupes reads. rows may repeat.
int st_read_rows(st_index *ix, st_tensor *t, const int64_t *rows, size_t nrows,
                 void *out, err_t *err);

#endif
