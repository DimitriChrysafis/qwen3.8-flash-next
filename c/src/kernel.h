#ifndef QWEN_KERNEL_H
#define QWEN_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// ---- elementwise ----

// y[i] = x[i] / rms(x) * w[i]; w may be NULL
void kernel_rmsnorm(const float *x, const float *w, size_t n, float eps,
                    float *y);

// grouped rmsnorm: x has [rows * groups_per_row * group] elements; each group
// is normalized independently. w holds groups_per_row * group weights that
// are broadcast across rows (indexed by group and element, not by row).
void kernel_rmsnorm_grouped(const float *x, const float *w, size_t n,
                            size_t group, size_t rows, float eps, float *y);

void kernel_silu(const float *x, size_t n, float *y);
void kernel_sigmoid(const float *x, size_t n, float *y);
void kernel_add(const float *a, const float *b, size_t n, float *y);
void kernel_scale(const float *x, float s, size_t n, float *y);
void kernel_mul(const float *a, const float *b, size_t n, float *y);
void kernel_softmax(float *x, size_t n); // in place over one row
void kernel_l2norm(const float *x, size_t n, float eps, float *y);

// ---- rope ----

// rotate the first d/2 pairs of each row of x (rows of len n) by cos/sin
// (cos/sin have length d/2 each? no: cos/sin have length d). partial rotary:
// only the first d dims rotate.
void kernel_rope_partial(const float *x, size_t rows, size_t n, size_t d,
                         const float *cos, const float *sin, float *y);

// ---- topk ----

// return the indices of the k largest values of x (order within top-k
// unspecified). out must hold k ints.
void kernel_topk_indices(const float *x, size_t n, size_t k, int *out);

// ---- dequant ----

// dequantize 4-bit rows: q packed as u32 (8 nibbles per word, low nibble
// first), scales/biases per group. out is rows x cols fp32.
void kernel_dequant_q4(const uint32_t *q, const float *scales,
                       const float *biases, size_t rows, size_t cols,
                       size_t group_size, float *out);

// dequantize a single row (used for gathers)
void kernel_dequant_q4_row(const uint32_t *q, const float *scales,
                           const float *biases, size_t cols,
                           size_t group_size, float *out);

// ---- gemm ----

// C[M,N] = A[M,K] @ W^T[N,K], W rows are filters
void gemm_fp32(int M, int N, int K, const float *A, const float *W, float *C);

// C[M,N] = A[M,K] @ W^T[N,K] with bf16 weights, fp32 accumulate
void gemm_bf16(int M, int N, int K, const float *A, const uint16_t *W,
               float *C);

// C[M,N] = A[M,K] @ W^T[N,K] with 4-bit group-quantized weights
void gemm_q4(int M, int N, int K, const float *A, const uint32_t *W,
             const float *scales, const float *biases, int group_size,
             float *C);

// ---- conv ----

// depthwise conv1d, groups = channels, kernel k, dilation d.
// x: [seq, ch]; w: [ch, k] bf16 (one weight per channel per tap);
// out: [seq, ch]; output[t] = sum_{i=0}^{k-1} w[c][i] * x[t - (k-1-i)*d]
void kernel_conv1d_depthwise(const float *x, size_t seq, size_t ch,
                             const uint16_t *w, size_t k, size_t d, float *y);

#endif
