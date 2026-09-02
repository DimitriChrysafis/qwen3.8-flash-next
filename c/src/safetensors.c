#include "safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"

static const struct {
    const char *name;
    int size;
    st_dtype dtype;
} DTYPES[] = {
    {"BOOL", 1, ST_BOOL}, {"U8", 1, ST_U8},     {"I8", 1, ST_I8},
    {"U16", 2, ST_U16},   {"I16", 2, ST_I16},   {"U32", 4, ST_U32},
    {"I32", 4, ST_I32},   {"U64", 8, ST_U64},   {"I64", 8, ST_I64},
    {"F16", 2, ST_F16},   {"BF16", 2, ST_BF16}, {"F32", 4, ST_F32},
    {"F64", 8, ST_F64},
};

int st_dtype_size(st_dtype d) {
    if (d < 0 || d >= ST_COUNT) return 0;
    return DTYPES[d].size;
}

static st_dtype dtype_from_name(const char *name) {
    for (size_t i = 0; i < sizeof(DTYPES) / sizeof(DTYPES[0]); i++) {
        if (strcmp(DTYPES[i].name, name) == 0) return DTYPES[i].dtype;
    }
    return ST_COUNT;
}

static st_tensor *add_tensor(st_index *ix, const char *name) {
    if (ix->ntensors == ix->cap) {
        ix->cap = ix->cap ? ix->cap * 2 : 256;
        ix->tensors = xrealloc(ix->tensors, ix->cap * sizeof(st_tensor));
    }
    st_tensor *t = &ix->tensors[ix->ntensors++];
    memset(t, 0, sizeof(*t));
    t->name = xmalloc(strlen(name) + 1);
    strcpy(t->name, name);
    return t;
}

static int parse_file(st_index *ix, const char *path, int file_idx, err_t *err) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        err_set(err, "cannot open %s: %s", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        err_set(err, "cannot stat %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    uint64_t file_size = (uint64_t)st.st_size;

    uint8_t lenbuf[8];
    ssize_t n = pread(fd, lenbuf, 8, 0);
    if (n != 8) {
        err_set(err, "truncated safetensors header: %s", path);
        close(fd);
        return -1;
    }
    uint64_t header_len;
    memcpy(&header_len, lenbuf, 8);
    if (header_len > file_size - 8) {
        err_set(err, "truncated safetensors json: %s", path);
        close(fd);
        return -1;
    }
    char *header = xmalloc((size_t)header_len + 1);
    size_t got = 0;
    while (got < header_len) {
        ssize_t r = pread(fd, header + got, header_len - got, 8 + got);
        if (r < 0) {
            if (errno == EINTR) continue;
            err_set(err, "read failed %s: %s", path, strerror(errno));
            free(header);
            close(fd);
            return -1;
        }
        if (r == 0) {
            err_set(err, "truncated safetensors json: %s", path);
            free(header);
            close(fd);
            return -1;
        }
        got += (size_t)r;
    }
    header[header_len] = 0;

    char jerr[256];
    json_value *root = json_parse(header, (size_t)header_len, jerr, sizeof(jerr));
    free(header);
    if (!root) {
        err_set(err, "invalid safetensors json %s: %s", path, jerr);
        close(fd);
        return -1;
    }
    if (root->type != JSON_OBJ) {
        err_set(err, "safetensors header must be an object: %s", path);
        json_free(root);
        close(fd);
        return -1;
    }

    uint64_t base = 8 + header_len;
    int status = -1;

    // first pass: validate entries and record ranges
    typedef struct {
        uint64_t lo, hi;
        const char *name;
    } range_t;
    range_t *ranges = xcalloc(root->u.obj.n ? root->u.obj.n : 1, sizeof(range_t));
    size_t nranges = 0;
    int failed = 0;

    for (size_t i = 0; i < root->u.obj.n && !failed; i++) {
        const char *stored = root->u.obj.keys[i];
        json_value *meta = root->u.obj.vals[i];
        if (strcmp(stored, "__metadata__") == 0) continue;
        if (meta->type != JSON_OBJ) {
            err_set(err, "invalid safetensors entry in %s", path);
            failed = 1;
            break;
        }
        const char *name = stored;
        if (strncmp(name, "language_model.", 15) == 0) name += 15;
        json_value *jdtype = json_obj_get(meta, "dtype");
        json_value *joffs = json_obj_get(meta, "data_offsets");
        json_value *jshape = json_obj_get(meta, "shape");
        if (!jdtype || jdtype->type != JSON_STR || !joffs || joffs->type != JSON_ARR ||
            joffs->u.arr.n != 2 || !jshape || jshape->type != JSON_ARR) {
            err_set(err, "invalid safetensors metadata for %s", stored);
            failed = 1;
            break;
        }
        st_dtype dtype = dtype_from_name(jdtype->u.str.s);
        if (dtype == ST_COUNT) {
            err_set(err, "unsupported safetensors dtype %s for %s", jdtype->u.str.s, stored);
            failed = 1;
            break;
        }
        json_value *lo_v = joffs->u.arr.items[0];
        json_value *hi_v = joffs->u.arr.items[1];
        if (lo_v->type != JSON_NUM || hi_v->type != JSON_NUM) {
            err_set(err, "invalid data offsets for %s", stored);
            failed = 1;
            break;
        }
        uint64_t lo = (uint64_t)json_num_i64(lo_v);
        uint64_t hi = (uint64_t)json_num_i64(hi_v);

        st_tensor *t = add_tensor(ix, name);
        t->file = file_idx;
        t->dtype = dtype;
        t->ndim = jshape->u.arr.n;
        if (t->ndim > 8) {
            err_set(err, "tensor %s has too many dims", stored);
            failed = 1;
            break;
        }
        uint64_t elems = 1;
        for (size_t d = 0; d < t->ndim; d++) {
            json_value *sv = jshape->u.arr.items[d];
            if (sv->type != JSON_NUM) {
                err_set(err, "invalid shape for %s", stored);
                failed = 1;
                break;
            }
            int64_t dim = json_num_i64(sv);
            if (dim < 0) {
                err_set(err, "invalid shape for %s", stored);
                failed = 1;
                break;
            }
            t->shape[d] = dim;
            elems *= (uint64_t)dim;
        }
        if (failed) break;
        uint64_t expect = elems * (uint64_t)st_dtype_size(dtype);
        if (hi < lo || hi - lo != expect || hi > file_size - base) {
            err_set(err, "invalid safetensors range for %s", stored);
            failed = 1;
            break;
        }
        // duplicate check (table not sorted yet, so linear scan)
        for (size_t k = 0; k + 1 < ix->ntensors; k++) {
            if (strcmp(ix->tensors[k].name, t->name) == 0) {
                err_set(err, "duplicate tensor %s", name);
                failed = 1;
                break;
            }
        }
        if (failed) break;
        t->offset = base + lo;
        t->nbytes = hi - lo;
        ranges[nranges].lo = lo;
        ranges[nranges].hi = hi;
        ranges[nranges].name = stored;
        nranges++;
    }

    if (!failed) {
        // overlap check
        for (size_t i = 0; i < nranges; i++) {
            for (size_t j = i + 1; j < nranges; j++) {
                if (ranges[i].lo < ranges[j].hi && ranges[j].lo < ranges[i].hi) {
                    err_set(err, "overlapping safetensors ranges for %s", ranges[i].name);
                    failed = 1;
                    break;
                }
            }
            if (failed) break;
        }
    }

    free(ranges);
    json_free(root);
    if (failed) {
        // drop tensors added from this file
        while (ix->ntensors > 0 && ix->tensors[ix->ntensors - 1].file == file_idx) {
            free(ix->tensors[--ix->ntensors].name);
        }
        close(fd);
        return -1;
    }

    ix->paths[file_idx] = xmalloc(strlen(path) + 1);
    strcpy(ix->paths[file_idx], path);
    ix->fds[file_idx] = fd;
    status = 0;
    return status;
}

static int cmp_tensor_name(const void *a, const void *b) {
    const st_tensor *ta = a, *tb = b;
    return strcmp(ta->name, tb->name);
}

int st_index_open(st_index *ix, const char *dir, int io_workers, err_t *err) {
    memset(ix, 0, sizeof(*ix));
    DIR *d = opendir(dir);
    if (!d) {
        err_set(err, "cannot open model dir %s: %s", dir, strerror(errno));
        return -1;
    }
    int nfiles = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        size_t len = strlen(de->d_name);
        if (len > 12 && strcmp(de->d_name + len - 12, ".safetensors") == 0) nfiles++;
    }
    if (nfiles == 0) {
        err_set(err, "no safetensors in %s", dir);
        closedir(d);
        return -1;
    }
    ix->paths = xcalloc(nfiles, sizeof(char *));
    ix->fds = xcalloc(nfiles, sizeof(int));
    for (int i = 0; i < nfiles; i++) ix->fds[i] = -1;
    ix->nfiles = nfiles;

    int file_idx = 0;
    rewinddir(d);
    int status = -1;
    while ((de = readdir(d)) != NULL) {
        size_t len = strlen(de->d_name);
        if (len > 12 && strcmp(de->d_name + len - 12, ".safetensors") == 0) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
            if (parse_file(ix, path, file_idx, err) != 0) goto out;
            file_idx++;
        }
    }
    // sort by name for binary search
    qsort(ix->tensors, ix->ntensors, sizeof(st_tensor), cmp_tensor_name);
    io_pool_init(&ix->pool, io_workers);
    status = 0;
out:
    closedir(d);
    if (status != 0) st_index_close(ix);
    return status;
}

void st_index_close(st_index *ix) {
    if (ix->pool.nthreads) io_pool_destroy(&ix->pool);
    for (int i = 0; i < ix->nfiles; i++) {
        if (ix->fds[i] >= 0) close(ix->fds[i]);
        free(ix->paths[i]);
    }
    free(ix->paths);
    free(ix->fds);
    for (size_t i = 0; i < ix->ntensors; i++) free(ix->tensors[i].name);
    free(ix->tensors);
    memset(ix, 0, sizeof(*ix));
}

st_tensor *st_find(st_index *ix, const char *name) {
    // binary search over sorted names
    size_t lo = 0, hi = ix->ntensors;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        int c = strcmp(ix->tensors[mid].name, name);
        if (c == 0) return &ix->tensors[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

int st_read_tensor(st_index *ix, st_tensor *t, void *out, err_t *err) {
    uint64_t got = 0;
    uint8_t *p = out;
    while (got < t->nbytes) {
        ssize_t r = pread(ix->fds[t->file], p + got, t->nbytes - got, t->offset + got);
        if (r < 0) {
            if (errno == EINTR) continue;
            err_set(err, "read failed for %s: %s", t->name, strerror(errno));
            return -1;
        }
        if (r == 0) {
            err_set(err, "short read for %s", t->name);
            return -1;
        }
        got += (uint64_t)r;
    }
    ix->pread_calls++;
    ix->bytes_read += t->nbytes;
    ix->tensors_read++;
    return 0;
}

int st_read_rows(st_index *ix, st_tensor *t, const int64_t *rows, size_t nrows,
                 void *out, err_t *err) {
    if (t->ndim == 0 || t->shape[0] == 0) {
        err_set(err, "tensor %s has no rows", t->name);
        return -1;
    }
    for (size_t i = 0; i < nrows; i++) {
        if (rows[i] < 0 || rows[i] >= t->shape[0]) {
            err_set(err, "row %lld outside %s shape %lld", (long long)rows[i],
                    t->name, (long long)t->shape[0]);
            return -1;
        }
    }
    uint64_t row_bytes = t->nbytes / (uint64_t)t->shape[0];
    io_job job;
    memset(&job, 0, sizeof(job));
    job.fd = ix->fds[t->file];
    job.base = t->offset;
    job.row_bytes = row_bytes;
    job.rows = rows;
    job.nrows = nrows;
    job.out = out;
    io_pool_submit(&ix->pool, &job);
    int status = io_job_wait(&job);
    if (status != 0) {
        err_set(err, "row read failed for %s: %s", t->name, strerror(status));
        return -1;
    }
    ix->rows_read += nrows;
    return 0;
}
