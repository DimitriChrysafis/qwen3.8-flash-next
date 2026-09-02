#include "kernel.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void kernel_rmsnorm(const float *x, const float *w, size_t n, float eps,
                    float *y) {
    float ss = 0;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (size_t i = 0; i < n; i++) y[i] = x[i] * r * w[i];
    } else {
        for (size_t i = 0; i < n; i++) y[i] = x[i] * r;
    }
}

void kernel_rmsnorm_grouped(const float *x, const float *w, size_t n,
                            size_t group, size_t rows, float eps, float *y) {
    size_t groups = n / group;
    size_t groups_per_row = rows ? groups / rows : groups;
    for (size_t g = 0; g < groups; g++) {
        const float *xg = x + g * group;
        float ss = 0;
        for (size_t i = 0; i < group; i++) ss += xg[i] * xg[i];
        float r = 1.0f / sqrtf(ss / (float)group + eps);
        const float *wg = w ? w + (g % groups_per_row) * group : NULL;
        for (size_t i = 0; i < group; i++) {
            y[g * group + i] = xg[i] * r * (wg ? wg[i] : 1.0f);
        }
    }
}

void kernel_silu(const float *x, size_t n, float *y) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

void kernel_sigmoid(const float *x, size_t n, float *y) {
    for (size_t i = 0; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

void kernel_add(const float *a, const float *b, size_t n, float *y) {
    for (size_t i = 0; i < n; i++) y[i] = a[i] + b[i];
}

void kernel_scale(const float *x, float s, size_t n, float *y) {
    for (size_t i = 0; i < n; i++) y[i] = x[i] * s;
}

void kernel_mul(const float *a, const float *b, size_t n, float *y) {
    for (size_t i = 0; i < n; i++) y[i] = a[i] * b[i];
}

void kernel_softmax(float *x, size_t n) {
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void kernel_l2norm(const float *x, size_t n, float eps, float *y) {
    float ss = 0;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss + eps);
    for (size_t i = 0; i < n; i++) y[i] = x[i] * r;
}

void kernel_rope_partial(const float *x, size_t rows, size_t n, size_t d,
                         const float *cos, const float *sin, float *y) {
    size_t half = d / 2;
    for (size_t r = 0; r < rows; r++) {
        const float *xr = x + r * n;
        float *yr = y + r * n;
        for (size_t i = 0; i < half; i++) {
            float x0 = xr[i], x1 = xr[i + half];
            yr[i] = x0 * cos[i] - x1 * sin[i];
            yr[i + half] = x0 * sin[i] + x1 * cos[i];
        }
        if (d < n) memcpy(yr + d, xr + d, (n - d) * sizeof(float));
    }
}

// quickselect-based partial selection: partition so the k largest values'
// indices occupy out[0..k)
static void kth_largest(const float *x, int *idx, size_t n, size_t k) {
    size_t lo = 0, hi = n;
    for (;;) {
        size_t pivot = lo + (hi - lo) / 2;
        float pv = x[idx[pivot]];
        int tmp = idx[pivot];
        idx[pivot] = idx[hi - 1];
        idx[hi - 1] = tmp;
        size_t store = lo;
        for (size_t i = lo; i < hi - 1; i++) {
            if (x[idx[i]] > pv) {
                tmp = idx[i];
                idx[i] = idx[store];
                idx[store] = tmp;
                store++;
            }
        }
        tmp = idx[store];
        idx[store] = idx[hi - 1];
        idx[hi - 1] = tmp;
        if (store == k - 1) return;
        if (store < k - 1) lo = store + 1;
        else hi = store;
    }
}

void kernel_topk_indices(const float *x, size_t n, size_t k, int *out) {
    if (k >= n) {
        for (size_t i = 0; i < n; i++) out[i] = (int)i;
        return;
    }
    int *idx = xmalloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) idx[i] = (int)i;
    kth_largest(x, idx, n, k);
    memcpy(out, idx, k * sizeof(int));
    free(idx);
}

void kernel_dequant_q4(const uint32_t *q, const float *scales,
                       const float *biases, size_t rows, size_t cols,
                       size_t group_size, float *out) {
    size_t words = cols / 8; // 8 nibbles per u32
    size_t groups = cols / group_size;
    for (size_t r = 0; r < rows; r++) {
        const uint32_t *qr = q + r * words;
        const float *sr = scales + r * groups;
        const float *br = biases + r * groups;
        float *orow = out + r * cols;
        for (size_t g = 0; g < groups; g++) {
            float s = sr[g], b = br[g];
            for (size_t w = 0; w < group_size / 8; w++) {
                uint32_t v = qr[g * (group_size / 8) + w];
                for (int i = 0; i < 8; i++) {
                    orow[g * group_size + w * 8 + i] =
                        (float)((v >> (4 * i)) & 0xF) * s + b;
                }
            }
        }
    }
}

void kernel_dequant_q4_row(const uint32_t *q, const float *scales,
                           const float *biases, size_t cols,
                           size_t group_size, float *out) {
    kernel_dequant_q4(q, scales, biases, 1, cols, group_size, out);
}

void gemm_fp32(int M, int N, int K, const float *A, const float *W,
               float *C) {
    if (M <= 0 || N <= 0) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K,
                W, K, 0.0f, C, N);
}

// bf16 x fp32 matmul via BNNS (AMX-accelerated on apple silicon)
// C[M,N] = A[M,K] @ W^T[N,K]
void gemm_bf16(int M, int N, int K, const float *A, const uint16_t *W,
               float *C) {
    if (M <= 0 || N <= 0) return;
    BNNSNDArrayDescriptor iA = {
        .layout = BNNSDataLayoutRowMajorMatrix,
        .data_type = BNNSDataTypeFloat32,
        .data = (void *)A,
        .size = {K, M, 1, 1},
        .stride = {1, K, K * M, K * M},
    };
    BNNSNDArrayDescriptor iB = {
        .layout = BNNSDataLayoutRowMajorMatrix,
        .data_type = BNNSDataTypeBFloat16,
        .data = (void *)W,
        .size = {K, N, 1, 1},
        .stride = {1, K, K * N, K * N},
    };
    BNNSNDArrayDescriptor o = {
        .layout = BNNSDataLayoutRowMajorMatrix,
        .data_type = BNNSDataTypeFloat32,
        .data = C,
        .size = {N, M, 1, 1},
        .stride = {1, N, N * M, N * M},
    };
    static void *workspace = NULL;
    static size_t workspace_size = 0;
    size_t need = (size_t)BNNSMatMulWorkspaceSize(false, true, 1.0f, &iA, &iB,
                                                  &o, NULL);
    if (need > workspace_size) {
        free(workspace);
        workspace = xmalloc(need ? need : 1);
        workspace_size = need;
    }
    int rc = BNNSMatMul(false, true, 1.0f, &iA, &iB, &o, workspace, NULL);
    if (rc != 0) {
        fprintf(stderr, "gemm_bf16: BNNSMatMul failed rc=%d\n", rc);
        abort();
    }
}

void gemm_q4(int M, int N, int K, const float *A, const uint32_t *W,
             const float *scales, const float *biases, int group_size,
             float *C) {
    if (M <= 0 || N <= 0) return;
    int groups = K / group_size;
    int words_per_group = group_size / 8;
    float *xsum = xmalloc((size_t)groups * sizeof(float));
    // block over outputs for register reuse
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m * K;
        for (int g = 0; g < groups; g++) {
            float s = 0;
            const float *ag = a + (size_t)g * group_size;
            for (int i = 0; i < group_size; i++) s += ag[i];
            xsum[g] = s;
        }
        float *c = C + (size_t)m * N;
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            const uint32_t *w0 = W + (size_t)n * (K / 8);
            const uint32_t *w1 = w0 + (K / 8);
            const uint32_t *w2 = w1 + (K / 8);
            const uint32_t *w3 = w2 + (K / 8);
            const float *s0 = scales + (size_t)n * groups;
            const float *s1 = s0 + groups;
            const float *s2 = s1 + groups;
            const float *s3 = s2 + groups;
            const float *b0 = biases + (size_t)n * groups;
            const float *b1 = b0 + groups;
            const float *b2 = b1 + groups;
            const float *b3 = b2 + groups;
            for (int g = 0; g < groups; g++) {
                float xq0 = 0, xq1 = 0, xq2 = 0, xq3 = 0;
                const float *ag = a + (size_t)g * group_size;
                for (int w = 0; w < words_per_group; w++) {
                    uint32_t v0 = w0[g * words_per_group + w];
                    uint32_t v1 = w1[g * words_per_group + w];
                    uint32_t v2 = w2[g * words_per_group + w];
                    uint32_t v3 = w3[g * words_per_group + w];
                    for (int i = 0; i < 8; i++) {
                        float x = ag[w * 8 + i];
                        xq0 += x * (float)((v0 >> (4 * i)) & 0xF);
                        xq1 += x * (float)((v1 >> (4 * i)) & 0xF);
                        xq2 += x * (float)((v2 >> (4 * i)) & 0xF);
                        xq3 += x * (float)((v3 >> (4 * i)) & 0xF);
                    }
                }
                acc0 += s0[g] * xq0 + b0[g] * xsum[g];
                acc1 += s1[g] * xq1 + b1[g] * xsum[g];
                acc2 += s2[g] * xq2 + b2[g] * xsum[g];
                acc3 += s3[g] * xq3 + b3[g] * xsum[g];
            }
            c[n] = acc0;
            c[n + 1] = acc1;
            c[n + 2] = acc2;
            c[n + 3] = acc3;
        }
        for (; n < N; n++) {
            float acc = 0;
            const uint32_t *w = W + (size_t)n * (K / 8);
            const float *s = scales + (size_t)n * groups;
            const float *b = biases + (size_t)n * groups;
            for (int g = 0; g < groups; g++) {
                float xq = 0;
                const float *ag = a + (size_t)g * group_size;
                for (int wd = 0; wd < words_per_group; wd++) {
                    uint32_t v = w[g * words_per_group + wd];
                    for (int i = 0; i < 8; i++) {
                        xq += ag[wd * 8 + i] * (float)((v >> (4 * i)) & 0xF);
                    }
                }
                acc += s[g] * xq + b[g] * xsum[g];
            }
            c[n] = acc;
        }
    }
    free(xsum);
}

void kernel_conv1d_depthwise(const float *x, size_t seq, size_t ch,
                             const uint16_t *w, size_t k, size_t d, float *y) {
    for (size_t t = 0; t < seq; t++) {
        for (size_t c = 0; c < ch; c++) {
            float acc = 0;
            for (size_t i = 0; i < k; i++) {
                // tap i covers position t - (k-1-i)*d
                size_t src = (size_t)((long long)t - (long long)(k - 1 - i) * (long long)d);
                if (src < seq) acc += x[src * ch + c] * bf16_to_f32(w[c * k + i]);
            }
            y[t * ch + c] = acc;
        }
    }
}
