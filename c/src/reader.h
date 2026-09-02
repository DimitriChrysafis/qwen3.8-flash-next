#ifndef QWEN_READER_H
#define QWEN_READER_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

typedef struct io_job io_job;

struct io_job {
    int fd;
    uint64_t base;     // data offset of first row in file
    uint64_t row_bytes;
    const int64_t *rows; // requested row indices, may repeat, any order
    size_t nrows;
    uint8_t *out;      // caller buffer of nrows * row_bytes
    int status;        // 0 on success, errno otherwise
    int done;
    pthread_mutex_t lock;
    pthread_cond_t cv;
};

typedef struct {
    pthread_t *threads;
    int nthreads;
    io_job **queue;
    int head, tail, cap;
    int inflight;
    int stop;
    pthread_mutex_t lock;
    pthread_cond_t cv_jobs;
    pthread_cond_t cv_idle;
} io_pool;

void io_pool_init(io_pool *p, int nthreads);

// enqueue a job; job must stay alive until io_job_wait returns.
void io_pool_submit(io_pool *p, io_job *job);

// block until the job is done; returns status.
int io_job_wait(io_job *job);

// wait for all in-flight jobs, stop workers, join.
void io_pool_destroy(io_pool *p);

#endif
