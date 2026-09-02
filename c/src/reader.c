#include "reader.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// merge contiguous runs of unique row indices and pread each run in one
// syscall, then scatter into the output buffer in request order. mirrors the
// rowio c extension behavior.
static int read_rows_impl(int fd, uint64_t base, uint64_t row_bytes,
                          const int64_t *rows, size_t nrows, uint8_t *out,
                          uint64_t *pread_calls_out, uint64_t *bytes_out) {
    if (nrows == 0) {
        *pread_calls_out = 0;
        *bytes_out = 0;
        return 0;
    }
    if (row_bytes == 0) {
        *pread_calls_out = 1;
        *bytes_out = 0;
        return 0;
    }

    size_t *order = malloc(nrows * sizeof(size_t));
    if (!order) return ENOMEM;
    for (size_t i = 0; i < nrows; i++) order[i] = i;
    // sort indices by row, stable
    for (size_t i = 1; i < nrows; i++) {
        size_t key = order[i];
        size_t j = i;
        while (j > 0 && rows[order[j - 1]] > rows[key]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = key;
    }

    // largest contiguous run decides the scratch block size
    uint64_t max_run = row_bytes;
    {
        size_t i = 0;
        while (i < nrows) {
            int64_t start_row = rows[order[i]];
            int64_t expect = start_row;
            while (i < nrows && rows[order[i]] == expect) {
                size_t k = i;
                while (k < nrows && rows[order[k]] == expect) k++;
                i = k;
                expect++;
            }
            uint64_t run_bytes = (uint64_t)(expect - start_row) * row_bytes;
            if (run_bytes > max_run) max_run = run_bytes;
        }
    }

    uint8_t *block = malloc((size_t)max_run);
    if (!block) {
        free(order);
        return ENOMEM;
    }

    uint64_t calls = 0, total = 0;
    size_t i = 0;
    while (i < nrows) {
        int64_t start_row = rows[order[i]];
        // find run of consecutive unique rows
        size_t j = i;
        int64_t expect = start_row;
        while (j < nrows && rows[order[j]] == expect) {
            // skip duplicates within the run
            size_t k = j;
            while (k < nrows && rows[order[k]] == expect) k++;
            j = k;
            expect++;
        }
        uint64_t run_rows = (uint64_t)(expect - start_row);
        uint64_t run_bytes = run_rows * row_bytes;
        size_t off = 0;
        while (off < run_bytes) {
            ssize_t n = pread(fd, block, run_bytes - off,
                              (off_t)(base + (uint64_t)start_row * row_bytes + off));
            if (n < 0) {
                if (errno == EINTR) continue;
                free(block);
                free(order);
                return errno;
            }
            if (n == 0) {
                free(block);
                free(order);
                return EIO;
            }
            off += (size_t)n;
            calls++;
        }
        total += run_bytes;
        // scatter: for each request in the run (including dupes), copy its row
        size_t r = i;
        while (r < nrows && rows[order[r]] < expect) {
            size_t row = (size_t)rows[order[r]];
            memcpy(out + order[r] * row_bytes,
                   block + (row - (size_t)start_row) * row_bytes, row_bytes);
            r++;
        }
        i = r;
    }
    free(block);
    free(order);
    *pread_calls_out = calls;
    *bytes_out = total;
    return 0;
}

static void *worker_main(void *arg) {
    io_pool *p = arg;
    for (;;) {
        pthread_mutex_lock(&p->lock);
        while (p->head == p->tail && !p->stop) {
            pthread_cond_wait(&p->cv_jobs, &p->lock);
        }
        if (p->stop && p->head == p->tail) {
            pthread_mutex_unlock(&p->lock);
            return NULL;
        }
        io_job *job = p->queue[p->head];
        p->head = (p->head + 1) % p->cap;
        pthread_mutex_unlock(&p->lock);

        uint64_t calls = 0, bytes = 0;
        job->status = read_rows_impl(job->fd, job->base, job->row_bytes,
                                     job->rows, job->nrows, job->out, &calls,
                                     &bytes);

        // signal under the job lock so the waiter's check-and-wait cannot
        // race with our set-and-broadcast (lost wakeup otherwise)
        pthread_mutex_lock(&job->lock);
        job->done = 1;
        pthread_cond_broadcast(&job->cv);
        pthread_mutex_unlock(&job->lock);

        pthread_mutex_lock(&p->lock);
        p->inflight--;
        if (p->inflight == 0) pthread_cond_broadcast(&p->cv_idle);
        pthread_mutex_unlock(&p->lock);
    }
}

void io_pool_init(io_pool *p, int nthreads) {
    memset(p, 0, sizeof(*p));
    if (nthreads < 1) nthreads = 1;
    p->nthreads = nthreads;
    p->cap = 256;
    p->queue = calloc(p->cap, sizeof(io_job *));
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->cv_jobs, NULL);
    pthread_cond_init(&p->cv_idle, NULL);
    p->threads = calloc(nthreads, sizeof(pthread_t));
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&p->threads[i], NULL, worker_main, p);
    }
}

void io_pool_submit(io_pool *p, io_job *job) {
    job->done = 0;
    job->status = 0;
    pthread_mutex_init(&job->lock, NULL);
    pthread_cond_init(&job->cv, NULL);
    pthread_mutex_lock(&p->lock);
    while ((p->tail + 1) % p->cap == p->head) {
        // grow ring
        int ncap = p->cap * 2;
        io_job **nq = calloc(ncap, sizeof(io_job *));
        int n = 0;
        for (int i = p->head; i != p->tail; i = (i + 1) % p->cap) nq[n++] = p->queue[i];
        p->head = 0;
        p->tail = n;
        p->cap = ncap;
        free(p->queue);
        p->queue = nq;
    }
    p->queue[p->tail] = job;
    p->tail = (p->tail + 1) % p->cap;
    p->inflight++;
    pthread_cond_signal(&p->cv_jobs);
    pthread_mutex_unlock(&p->lock);
}

int io_job_wait(io_job *job) {
    pthread_mutex_lock(&job->lock);
    while (!job->done) pthread_cond_wait(&job->cv, &job->lock);
    pthread_mutex_unlock(&job->lock);
    return job->status;
}

void io_pool_destroy(io_pool *p) {
    pthread_mutex_lock(&p->lock);
    p->stop = 1;
    pthread_cond_broadcast(&p->cv_jobs);
    pthread_mutex_unlock(&p->lock);
    for (int i = 0; i < p->nthreads; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->queue);
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->cv_jobs);
    pthread_cond_destroy(&p->cv_idle);
}
