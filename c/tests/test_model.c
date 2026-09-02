// model e2e test: builds a tiny qwen4_exp checkpoint in c (when no dir is
// given) or loads one, runs prefill + incremental decode, and checks that
// decode logits match the prefill logits per position.
//
// with a directory argument the logits are also dumped to
// /tmp/qwen_logits_c.bin and /tmp/qwen_decode_c.bin for parity checking
// against the colibri reference (see gen_tiny.py).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "model.h"
#include "ngram.h"
#include "test.h"
#include "util.h"

// deterministic pseudo random floats
static uint64_t rng_state = 11;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 33) & 0xffff) / 65536.0f - 0.5f;
}

// build a tiny checkpoint dir with the same geometry as the parity model
typedef struct {
    char name[160];
    int is_i64;
    int ndim;
    int64_t shape[3];
    size_t n; // element count
} tplan;

static void build_tiny_dir(const char *dir) {
    mkdir(dir, 0700);
    enum { D = 32, HC = 4, HCD = 128, LR = 32, VOCAB = 64, LAYERS = 2 };
    enum { CONV_DIM = 48, VALUE_DIM = 32 };
    // geometry
    ngram_geo ng;
    memset(&ng, 0, sizeof(ng));
    ng.ngram_size = 3;
    ng.heads_per_ngram = 1;
    ng.vocab_size = VOCAB;
    ng.eos_token_id = 1;
    ngram_geo_compute(&ng, 31, 1234, 0);
    int row_width = 64 / ((ng.ngram_size - 1) * ng.heads_per_ngram);
    int rps = (int)(ng.total_rows / 2);

    tplan *plan = xcalloc(1 << 12, sizeof(tplan));
    size_t np = 0;
#define TP(nm, is64, ndim_, sh0, sh1, sh2, cnt)                                  \
    do {                                                                       \
        tplan *tp = &plan[np++];                                               \
        snprintf(tp->name, sizeof(tp->name), "%s", nm);                      \
        tp->is_i64 = is64;                                                     \
        tp->ndim = ndim_;                                                      \
        tp->shape[0] = sh0;                                                    \
        tp->shape[1] = sh1;                                                    \
        tp->shape[2] = sh2;                                                    \
        tp->n = cnt;                                                            \
    } while (0)
    TP("model.embed_tokens.weight", 0, 2, VOCAB, D, 0, (size_t)VOCAB * D);
    TP("lm_head.weight", 0, 2, VOCAB, D, 0, (size_t)VOCAB * D);
    for (int layer = 0; layer < LAYERS; layer++) {
        char L[64];
        snprintf(L, sizeof(L), "model.layers.%d", layer);
        char nm[128];
        if (layer == 1) {
            snprintf(nm, sizeof(nm), "%s.self_attn.q_proj.weight", L);
            TP(nm, 0, 2, 64, D, 0, (size_t)64 * D);
            snprintf(nm, sizeof(nm), "%s.self_attn.k_proj.weight", L);
            TP(nm, 0, 2, 16, D, 0, (size_t)16 * D);
            snprintf(nm, sizeof(nm), "%s.self_attn.v_proj.weight", L);
            TP(nm, 0, 2, 16, D, 0, (size_t)16 * D);
            snprintf(nm, sizeof(nm), "%s.self_attn.o_proj.weight", L);
            TP(nm, 0, 2, D, 32, 0, (size_t)D * 32);
            snprintf(nm, sizeof(nm), "%s.self_attn.q_norm.weight", L);
            TP(nm, 0, 1, 16, 0, 0, 16);
            snprintf(nm, sizeof(nm), "%s.self_attn.k_norm.weight", L);
            TP(nm, 0, 1, 16, 0, 0, 16);
            snprintf(nm, sizeof(nm), "%s.self_attn.indexer.index_qk_proj.weight", L);
            TP(nm, 0, 2, 24, D, 0, (size_t)24 * D);
            snprintf(nm, sizeof(nm), "%s.self_attn.indexer.q_layernorm.weight", L);
            TP(nm, 0, 1, 8, 0, 0, 8);
            snprintf(nm, sizeof(nm), "%s.self_attn.indexer.k_layernorm.weight", L);
            TP(nm, 0, 1, 8, 0, 0, 8);
        } else {
            snprintf(nm, sizeof(nm), "%s.linear_attn.in_proj_qkv.weight", L);
            TP(nm, 0, 2, CONV_DIM, D, 0, (size_t)CONV_DIM * D);
            snprintf(nm, sizeof(nm), "%s.linear_attn.in_proj_z.weight", L);
            TP(nm, 0, 2, VALUE_DIM, D, 0, (size_t)VALUE_DIM * D);
            snprintf(nm, sizeof(nm), "%s.linear_attn.in_proj_a.weight", L);
            TP(nm, 0, 2, 2, D, 0, (size_t)2 * D);
            snprintf(nm, sizeof(nm), "%s.linear_attn.in_proj_b.weight", L);
            TP(nm, 0, 2, 2, D, 0, (size_t)2 * D);
            snprintf(nm, sizeof(nm), "%s.linear_attn.out_proj.weight", L);
            TP(nm, 0, 2, D, VALUE_DIM, 0, (size_t)D * VALUE_DIM);
            snprintf(nm, sizeof(nm), "%s.linear_attn.conv1d.weight", L);
            TP(nm, 0, 3, CONV_DIM, 4, 1, (size_t)CONV_DIM * 4);
            snprintf(nm, sizeof(nm), "%s.linear_attn.norm.weight", L);
            TP(nm, 0, 1, 16, 0, 0, 16);
            snprintf(nm, sizeof(nm), "%s.linear_attn.A_log", L);
            TP(nm, 0, 1, 2, 0, 0, 2);
            snprintf(nm, sizeof(nm), "%s.linear_attn.dt_bias", L);
            TP(nm, 0, 1, 2, 0, 0, 2);
        }
        snprintf(nm, sizeof(nm), "%s.mlp.gate.weight", L);
        TP(nm, 0, 2, 4, D, 0, (size_t)4 * D);
        for (int p = 0; p < 3; p++) {
            const char *proj = p == 0 ? "gate_proj" : p == 1 ? "up_proj" : "down_proj";
            snprintf(nm, sizeof(nm), "%s.mlp.shared_expert.%s.weight", L, proj);
            TP(nm, 0, 2, 32, D, 0, (size_t)32 * D);
            snprintf(nm, sizeof(nm), "%s.mlp.switch_mlp.%s.weight", L, proj);
            TP(nm, 0, 3, 4, 32, D, (size_t)4 * 32 * D);
        }
        snprintf(nm, sizeof(nm), "%s.mlp.shared_expert_gate.weight", L);
        TP(nm, 0, 2, 1, D, 0, (size_t)D);
        for (int h = 0; h < 2; h++) {
            const char *hpname = h == 0 ? "attn_hyper_connection" : "mlp_hyper_connection";
            snprintf(nm, sizeof(nm), "%s.%s.hc_norm.weight", L, hpname);
            TP(nm, 0, 1, HCD, 0, 0, HCD);
            snprintf(nm, sizeof(nm), "%s.%s.input_mix_weight_down.weight", L, hpname);
            TP(nm, 0, 2, LR, HCD, 0, (size_t)LR * HCD);
            snprintf(nm, sizeof(nm), "%s.%s.input_mix_weight_up.weight", L, hpname);
            TP(nm, 0, 2, HCD, LR, 0, (size_t)HCD * LR);
            snprintf(nm, sizeof(nm), "%s.%s.block_inject_weight.weight", L, hpname);
            TP(nm, 0, 2, HC, HCD, 0, (size_t)HC * HCD);
        }
        if (layer == 0) {
            for (int shard = 0; shard < 2; shard++) {
                snprintf(nm, sizeof(nm),
                         "%s.ple.ple_embedding.ngram_embedding.shard_%d.weight", L, shard);
                TP(nm, 0, 2, rps, row_width, 0, (size_t)rps * row_width);
            }
            snprintf(nm, sizeof(nm), "%s.ple.ple_embedding.layer_multipliers", L);
            TP(nm, 1, 1, 3, 0, 0, 3);
            snprintf(nm, sizeof(nm), "%s.ple.ple_embedding.ngram_heads_offsets", L);
            TP(nm, 1, 1, 2, 0, 0, 2);
            snprintf(nm, sizeof(nm), "%s.ple.ple_embedding.ngram_heads_vocab_sizes", L);
            TP(nm, 1, 1, 2, 0, 0, 2);
            snprintf(nm, sizeof(nm), "%s.ple.key_proj.weight", L);
            TP(nm, 0, 2, HCD, 64, 0, (size_t)HCD * 64);
            snprintf(nm, sizeof(nm), "%s.ple.value_proj.weight", L);
            TP(nm, 0, 2, D, 64, 0, (size_t)D * 64);
            snprintf(nm, sizeof(nm), "%s.ple.conv1d.weight", L);
            TP(nm, 0, 3, HCD, 2, 1, (size_t)HCD * 2);
            snprintf(nm, sizeof(nm), "%s.ple.norm_key.weight", L);
            TP(nm, 0, 1, HCD, 0, 0, HCD);
            snprintf(nm, sizeof(nm), "%s.ple.norm_query.weight", L);
            TP(nm, 0, 1, HCD, 0, 0, HCD);
            snprintf(nm, sizeof(nm), "%s.ple.norm_conv.weight", L);
            TP(nm, 0, 1, HCD, 0, 0, HCD);
        }
    }
    TP("model.hyper_connection_mixer.hc_norm.weight", 0, 1, HCD, 0, 0, HCD);
    TP("model.hyper_connection_mixer.input_mix_weight_down.weight", 0, 2, LR, HCD, 0, (size_t)LR * HCD);
    TP("model.hyper_connection_mixer.input_mix_weight_up.weight", 0, 2, HCD, LR, 0, (size_t)HCD * LR);
#undef TP

    // compute offsets and write the header
    uint64_t off = 0;
    char *hdr = xmalloc(1 << 20);
    size_t hp = 0;
    hp += (size_t)snprintf(hdr + hp, 1 << 20, "{");
    for (size_t i = 0; i < np; i++) {
        tplan *tp = &plan[i];
        if (i) hdr[hp++] = ',';
        const char *dt = tp->is_i64 ? "I64" : "BF16";
        size_t el = tp->is_i64 ? 8 : 2;
        hp += (size_t)snprintf(hdr + hp, (1 << 20) - hp,
                               "\"%s\":{\"dtype\":\"%s\",\"shape\":[", tp->name, dt);
        for (int d = 0; d < tp->ndim; d++) {
            if (d) hdr[hp++] = ',';
            hp += (size_t)snprintf(hdr + hp, (1 << 20) - hp, "%lld",
                                   (long long)tp->shape[d]);
        }
        hp += (size_t)snprintf(hdr + hp, (1 << 20) - hp, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)off,
                               (unsigned long long)(off + tp->n * el));
        off += (uint64_t)tp->n * el;
    }
    hp += (size_t)snprintf(hdr + hp, 1 << 20, "}");
    uint64_t hlen = hp;

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hlen, f);
    free(hdr);

    // write data
    float *buf = xmalloc(1 << 20);
    for (size_t i = 0; i < np; i++) {
        tplan *tp = &plan[i];
        if (tp->is_i64) {
            int64_t vals[8] = {0};
            if (strstr(tp->name, "layer_multipliers")) {
                vals[0] = ng.multipliers[0];
                vals[1] = ng.multipliers[1];
                vals[2] = ng.multipliers[2];
            } else if (strstr(tp->name, "vocab_sizes")) {
                vals[0] = ng.sizes[0];
                vals[1] = ng.sizes[1];
            } else { // offsets
                vals[0] = ng.offsets[0];
                vals[1] = ng.offsets[1];
            }
            fwrite(vals, 8, tp->n, f);
        } else {
            for (size_t k = 0; k < tp->n; k++) buf[k] = frand() * 0.3f;
            // norms are centered around 1
            if (strstr(tp->name, "norm.weight") || strstr(tp->name, ".A_log") ||
                strstr(tp->name, ".dt_bias")) {
                for (size_t k = 0; k < tp->n; k++) buf[k] += 1.0f;
            }
            for (size_t k = 0; k < tp->n; k++) {
                uint16_t b = f32_to_bf16(buf[k]);
                fwrite(&b, 2, 1, f);
            }
        }
    }
    free(buf);
    free(plan);
    fclose(f);

    // config.json
    snprintf(path, sizeof(path), "%s/config.json", dir);
    f = fopen(path, "w");
    fprintf(f,
            "{\"model_type\":\"qwen4_exp\",\"text_config\":{"
            "\"vocab_size\":64,\"hidden_size\":32,\"num_hidden_layers\":2,"
            "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":16,"
            "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
            "\"linear_num_key_heads\":1,\"linear_num_value_heads\":2,"
            "\"linear_key_head_dim\":8,\"linear_value_head_dim\":16,"
            "\"linear_conv_kernel_dim\":4,\"num_experts\":4,\"num_experts_per_tok\":2,"
            "\"moe_intermediate_size\":32,\"shared_expert_intermediate_size\":32,"
            "\"hc_count\":4,\"hc_lowrank\":32,\"ple_layer_ids\":[1],"
            "\"ple_embed_dim\":64,\"ple_conv_kernel_size\":2,\"ngram_size\":3,"
            "\"heads_per_ngram\":1,\"ngram_vocab_size_base\":31,"
            "\"make_ngram_vocab_size_divisible_by\":8,\"split_ngram_parts\":2,"
            "\"seed\":1234,\"indexer_n_heads\":2,\"indexer_kv_heads\":1,"
            "\"indexer_head_dim\":8,\"indexer_budget\":4,\"indexer_compress_ratio\":2,"
            "\"output_gate_type\":\"sigmoid\",\"eos_token_id\":1,\"bos_token_id\":1,"
            "\"rope_parameters\":{\"rope_theta\":10000.0,"
            "\"partial_rotary_factor\":0.5},\"rms_norm_eps\":1e-6,"
            "\"tie_word_embeddings\":false},\"vision_config\":{},\"quantization\":{}}");
    fclose(f);
}

static int run_forward(qmodel *m, int dump) {
    err_t err = {0};
    int64_t ids[] = {2, 3, 4, 1, 5, 6};
    size_t n = sizeof(ids) / sizeof(ids[0]);
    int64_t vocab = model_vocab(m);
    float *logits = xmalloc((size_t)n * vocab * sizeof(float));
    float *decode = xmalloc((size_t)n * vocab * sizeof(float));
    if (model_forward(m, ids, n, 0, logits, &err) != 0) {
        fprintf(stderr, "prefill failed: %s\n", err.msg);
        return 1;
    }
    // incremental decode: one token at a time
    model_reset_caches(m);
    for (size_t i = 0; i < n; i++) {
        if (model_forward(m, &ids[i], 1, 1, decode + i * vocab, &err) != 0) {
            fprintf(stderr, "decode %zu failed: %s\n", i, err.msg);
            return 1;
        }
    }
    // decode logits must match prefill logits per position
    double worst = 0;
    for (size_t i = 0; i < n * (size_t)vocab; i++) {
        double d = fabs((double)decode[i] - (double)logits[i]);
        if (d > worst) worst = d;
        if (!isfinite(decode[i]) || !isfinite(logits[i])) {
            fprintf(stderr, "non-finite logit at %zu\n", i);
            return 1;
        }
    }
    printf("prefill-vs-decode max diff: %.3e\n", worst);
    if (worst > 1e-3) {
        fprintf(stderr, "prefill/decode mismatch\n");
        return 1;
    }
    if (dump) {
        FILE *f = fopen("/tmp/qwen_logits_c.bin", "wb");
        fwrite(logits, 4, (size_t)n * vocab, f);
        fclose(f);
        f = fopen("/tmp/qwen_decode_c.bin", "wb");
        fwrite(decode, 4, (size_t)n * vocab, f);
        fclose(f);
    }
    free(logits);
    free(decode);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int dump = 0;
    if (argc > 1) {
        dir = argv[1];
        dump = 1;
    }
    if (!dir) {
        dir = "/tmp/qwen_tiny_c";
        build_tiny_dir(dir);
    }
    err_t err = {0};
    qmodel *m = model_load(dir, 4, 1ull << 20, 1ull << 20, &err);
    if (!m) {
        fprintf(stderr, "model_load failed: %s\n", err.msg);
        return 1;
    }
    int rc = run_forward(m, dump);
    const qstats *st = model_stats(m);
    printf("expert loads: %llu, ple loads: %llu\n",
           (unsigned long long)st->expert_loads, (unsigned long long)st->ple_loads);
    if (st->expert_loads == 0 || st->ple_loads == 0) {
        fprintf(stderr, "expected streamed loads from disk\n");
        rc = 1;
    }
    model_free(m);
    return rc;
}
