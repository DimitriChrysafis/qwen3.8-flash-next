#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "test.h"
#include "util.h"

static float frand(uint64_t *state) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*state >> 33) & 0xffff) / 65536.0f - 0.5f;
}

static int close_enough(float a, float b, float tol) {
    return fabsf(a - b) <= tol * (1.0f + fabsf(a) + fabsf(b));
}

static void test_rmsnorm(void) {
    uint64_t st = 7;
    float x[257], w[257], y[257], ref[257];
    for (int i = 0; i < 257; i++) {
        x[i] = frand(&st);
        w[i] = frand(&st) + 1.0f;
    }
    kernel_rmsnorm(x, w, 257, 1e-6f, y);
    float ss = 0;
    for (int i = 0; i < 257; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / 257.0f + 1e-6f);
    for (int i = 0; i < 257; i++) ref[i] = x[i] * r * w[i];
    for (int i = 0; i < 257; i++) CHECK(close_enough(y[i], ref[i], 1e-6f));
    // grouped: 4 rows x 2 groups of 64, weight [128] broadcast over rows
    kernel_rmsnorm_grouped(x, w, 512, 64, 4, 1e-6f, y);
    for (int g = 0; g < 8; g++) {
        float gss = 0;
        for (int i = 0; i < 64; i++) gss += x[g * 64 + i] * x[g * 64 + i];
        float gr = 1.0f / sqrtf(gss / 64.0f + 1e-6f);
        for (int i = 0; i < 64; i++) {
            CHECK(close_enough(y[g * 64 + i], x[g * 64 + i] * gr * w[(g % 2) * 64 + i], 1e-6f));
        }
    }
}

static void test_rope(void) {
    uint64_t st = 11;
    float x[3][8], cosv[4], sinv[4], y[3][8], ref[3][8];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 8; j++) x[i][j] = frand(&st);
    for (int j = 0; j < 4; j++) {
        cosv[j] = cosf((float)j);
        sinv[j] = sinf((float)j);
    }
    // partial rotary: only the first d=4 dims rotate, pairs (0,2) and (1,3)
    kernel_rope_partial(&x[0][0], 3, 8, 4, cosv, sinv, &y[0][0]);
    for (int i = 0; i < 3; i++) {
        ref[i][0] = x[i][0] * cosv[0] - x[i][2] * sinv[0];
        ref[i][1] = x[i][1] * cosv[1] - x[i][3] * sinv[1];
        ref[i][2] = x[i][0] * sinv[0] + x[i][2] * cosv[0];
        ref[i][3] = x[i][1] * sinv[1] + x[i][3] * cosv[1];
        ref[i][4] = x[i][4];
        ref[i][5] = x[i][5];
        ref[i][6] = x[i][6];
        ref[i][7] = x[i][7];
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 8; j++) CHECK(close_enough(y[i][j], ref[i][j], 1e-6f));
    // full rotation with d == n: pairs (0,4),(1,5),(2,6),(3,7)
    kernel_rope_partial(&x[0][0], 1, 8, 8, cosv, sinv, &y[0][0]);
    for (int j = 0; j < 4; j++) {
        CHECK(close_enough(y[0][j], x[0][j] * cosv[j] - x[0][j + 4] * sinv[j], 1e-6f));
        CHECK(close_enough(y[0][j + 4], x[0][j] * sinv[j] + x[0][j + 4] * cosv[j], 1e-6f));
    }
}

static void test_silu_softmax_l2(void) {
    uint64_t st = 13;
    float x[100], y[100];
    for (int i = 0; i < 100; i++) x[i] = frand(&st) * 4;
    kernel_silu(x, 100, y);
    for (int i = 0; i < 100; i++) {
        float v = x[i];
        CHECK(close_enough(y[i], v / (1.0f + expf(-v)), 1e-6f));
    }
    float sm[64];
    for (int i = 0; i < 64; i++) sm[i] = frand(&st) * 3;
    kernel_softmax(sm, 64);
    float s = 0;
    for (int i = 0; i < 64; i++) s += sm[i];
    CHECK(fabsf(s - 1.0f) < 1e-5f);
    kernel_l2norm(x, 100, 1e-6f, y);
    float n2 = 0;
    for (int i = 0; i < 100; i++) n2 += y[i] * y[i];
    CHECK(fabsf(n2 - 1.0f) < 1e-5f);
}

static void test_topk(void) {
    uint64_t st = 17;
    float x[512];
    int idx[512];
    for (int i = 0; i < 512; i++) x[i] = frand(&st) * 10;
    const size_t ks[] = {1, 3, 10, 64, 512};
    for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        size_t k = ks[ki];
        kernel_topk_indices(x, 512, k, idx);
        // verify: min(chosen) >= max(unchosen)
        int used[512] = {0};
        float chosen_min = 1e30f, unchosen_max = -1e30f;
        for (size_t i = 0; i < k; i++) {
            CHECK(idx[i] >= 0 && idx[i] < 512);
            used[idx[i]] = 1;
            if (x[idx[i]] < chosen_min) chosen_min = x[idx[i]];
        }
        for (int j = 0; j < 512; j++) {
            if (!used[j] && x[j] > unchosen_max) unchosen_max = x[j];
        }
        CHECK(chosen_min >= unchosen_max - 1e-6f);
        // uniqueness
        for (size_t i = 0; i < k; i++)
            for (size_t j = i + 1; j < k; j++) CHECK(idx[i] != idx[j]);
    }
}

// packed values from mlx 0.32.2 (verified against mx.quantize/dequantize)
static void test_dequant_mlx_vectors(void) {
    // w = -16..15, group 32, 4 bit: scale 2, bias -16
    uint32_t q[1] = {0x43322110};
    float scales[1] = {2.0f}, biases[1] = {-16.0f};
    float out[32];
    // the first word covers values -16..-9
    kernel_dequant_q4_row(q, scales, biases, 32, 32, out);
    float expect[8] = {-16, -14, -14, -12, -12, -10, -10, -8};
    for (int i = 0; i < 8; i++) CHECK(out[i] == expect[i]);
    // multi-row: q pattern for values 0..7 -> bytes 0x10,0x32,0x54,0x76
    uint32_t q2[4] = {0x76543210, 0x00000000, 0x00000000, 0x00000000};
    float out2[32];
    kernel_dequant_q4_row(q2, scales, biases, 32, 32, out2);
    CHECK(out2[0] == -16); // nibble 0 -> q 0
    CHECK(out2[1] == -14); // nibble 1 -> q 1
    CHECK(out2[2] == -12); // nibble 2 -> q 2
    CHECK(out2[6] == -4);  // nibble 6 -> q 6
    CHECK(out2[7] == -2);  // nibble 7 -> q 7
}

static void test_dequant_random(void) {
    uint64_t st = 23;
    float w[128], out[128], ref[128];
    uint32_t q[16];
    float scales[2], biases[2];
    // simulate mlx quantize
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < 128; i++) {
        w[i] = frand(&st) * 8;
        if (w[i] < mn) mn = w[i];
        if (w[i] > mx) mx = w[i];
    }
    for (int g = 0; g < 2; g++) {
        const float *wg = w + g * 64;
        float gmn = 1e30f, gmx = -1e30f;
        for (int i = 0; i < 64; i++) {
            if (wg[i] < gmn) gmn = wg[i];
            if (wg[i] > gmx) gmx = wg[i];
        }
        float scale = fmaxf((gmx - gmn) / 15.0f, 1e-7f);
        int side = fabsf(gmn) > fabsf(gmx);
        scale = side ? scale : -scale;
        float edge = side ? gmn : gmx;
        float q0 = roundf(edge / scale);
        if (q0 == 0.0f) {
            (void)scale;
        } else {
            scale = edge / q0;
        }
        float bias = (q0 == 0.0f) ? 0.0f : edge;
        scales[g] = scale;
        biases[g] = bias;
        for (int i = 0; i < 64; i++) {
            int v = (int)fminf(roundf((wg[i] - bias) / scale), 15.0f);
            if (v < 0) v = 0;
            int word = i / 8;
            int nib = i % 8;
            if (nib == 0) q[g * 8 + word] = 0;
            q[g * 8 + word] |= (uint32_t)v << (4 * nib);
            ref[g * 64 + i] = (float)v * scale + bias;
        }
    }
    kernel_dequant_q4(q, scales, biases, 2, 64, 64, out);
    for (int i = 0; i < 128; i++) CHECK(out[i] == ref[i]);
}

static void test_gemm_fp32(void) {
    uint64_t st = 29;
    enum { M = 5, N = 7, K = 33 };
    float A[M * K], W[N * K], C[M * N], ref[M * N];
    for (int i = 0; i < M * K; i++) A[i] = frand(&st);
    for (int i = 0; i < N * K; i++) W[i] = frand(&st);
    gemm_fp32(M, N, K, A, W, C);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += A[m * K + k] * W[n * K + k];
            ref[m * N + n] = acc;
        }
    for (int i = 0; i < M * N; i++) CHECK(close_enough(C[i], ref[i], 1e-5f));
}

static void test_gemm_bf16(void) {
    uint64_t st = 31;
    enum { M = 5, N = 7, K = 129 };
    float A[M * K], C[M * N], ref[M * N];
    uint16_t W[N * K];
    for (int i = 0; i < M * K; i++) A[i] = frand(&st);
    for (int i = 0; i < N * K; i++) W[i] = f32_to_bf16(frand(&st));
    gemm_bf16(M, N, K, A, W, C);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += A[m * K + k] * bf16_to_f32(W[n * K + k]);
            ref[m * N + n] = acc;
        }
    for (int i = 0; i < M * N; i++) CHECK(close_enough(C[i], ref[i], 2e-3f));
}

static void test_gemm_q4(void) {
    uint64_t st = 37;
    enum { M = 4, N = 9, K = 256 };
    float A[M * K], C[M * N], ref[M * N];
    uint32_t W[N * K / 8];
    float scales[N * 4], biases[N * 4];
    for (int i = 0; i < M * K; i++) A[i] = frand(&st) * 2;
    for (int n = 0; n < N; n++) {
        float mn = 1e30f, mx = -1e30f;
        for (int g = 0; g < 4; g++) {
            for (int i = 0; i < 64; i++) {
                float v = frand(&st) * 4;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float scale = fmaxf((mx - mn) / 15.0f, 1e-7f);
            int side = fabsf(mn) > fabsf(mx);
            scale = side ? scale : -scale;
            float edge = side ? mn : mx;
            float q0 = roundf(edge / scale);
            if (q0 != 0.0f) scale = edge / q0;
            float bias = (q0 == 0.0f) ? 0.0f : edge;
            scales[n * 4 + g] = scale;
            biases[n * 4 + g] = bias;
        }
    }
    // build quantized weights from a deterministic pattern
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < 4; g++) {
            float s = scales[n * 4 + g], b = biases[n * 4 + g];
            for (int i = 0; i < 64; i++) {
                float v = frand(&st) * 4;
                int qv = (int)fminf(roundf((v - b) / s), 15.0f);
                if (qv < 0) qv = 0;
                int word = g * 8 + i / 8, nib = i % 8;
                if (nib == 0) W[n * (K / 8) + word] = 0;
                W[n * (K / 8) + word] |= (uint32_t)qv << (4 * nib);
            }
        }
    }
    gemm_q4(M, N, K, A, W, scales, biases, 64, C);
    // reference: dequantize and sgemm
    float *Wd = xmalloc(N * K * sizeof(float));
    kernel_dequant_q4(W, scales, biases, N, K, 64, Wd);
    gemm_fp32(M, N, K, A, Wd, ref);
    for (int i = 0; i < M * N; i++) CHECK(close_enough(C[i], ref[i], 1e-4f));
    free(Wd);
}

static void test_conv1d(void) {
    uint64_t st = 41;
    enum { SEQ = 9, CH = 5, K = 4, D = 3 };
    float x[SEQ * CH], wf[CH * K], y[SEQ * CH], ref[SEQ * CH];
    uint16_t w[CH * K];
    for (int i = 0; i < SEQ * CH; i++) x[i] = frand(&st);
    for (int i = 0; i < CH * K; i++) {
        wf[i] = frand(&st);
        w[i] = f32_to_bf16(wf[i]);
    }
    kernel_conv1d_depthwise(x, SEQ, CH, w, K, D, y);
    for (int t = 0; t < SEQ; t++)
        for (int c = 0; c < CH; c++) {
            float acc = 0;
            for (int i = 0; i < K; i++) {
                long long src = (long long)t - (long long)(K - 1 - i) * D;
                if (src >= 0 && src < SEQ)
                    acc += x[src * CH + c] * bf16_to_f32(w[c * K + i]);
            }
            ref[t * CH + c] = acc;
        }
    for (int i = 0; i < SEQ * CH; i++) CHECK(close_enough(y[i], ref[i], 1e-6f));
}

int main(void) {
    RUN_TEST(test_rmsnorm);
    RUN_TEST(test_rope);
    RUN_TEST(test_silu_softmax_l2);
    RUN_TEST(test_topk);
    RUN_TEST(test_dequant_mlx_vectors);
    RUN_TEST(test_dequant_random);
    RUN_TEST(test_gemm_fp32);
    RUN_TEST(test_gemm_bf16);
    RUN_TEST(test_gemm_q4);
    RUN_TEST(test_conv1d);
    return test_summary("kernel");
}
