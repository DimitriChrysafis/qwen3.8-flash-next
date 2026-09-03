#include "model.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "kernel.h"
#include "tokenizer.h"
#include "util.h"

// ---------------------------------------------------------------------------
// config

typedef struct {
    int hidden_size, num_layers, num_attn_heads, num_kv_heads, head_dim;
    int64_t vocab_size;
    int full_attention_interval;
    int num_experts, num_experts_per_tok, moe_intermediate_size;
    int shared_expert_intermediate_size;
    int linear_num_key_heads, linear_num_value_heads;
    int linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim;
    int hc_count, hc_lowrank;
    int indexer_n_heads, indexer_kv_heads, indexer_head_dim, indexer_budget;
    int indexer_compress_ratio;
    int ngram_size, heads_per_ngram;
    int64_t ngram_vocab_size_base;
    int split_ngram_parts, ple_embed_dim, ple_conv_kernel_size;
    int64_t eos_token_id;
    float partial_rotary_factor, rope_theta, rms_norm_eps;
    int layer_types[QMODEL_MAX_LAYERS];
    int n_layers;
    int ple_layer; // 0-based
    int group_size, bits, ple_group_size;
    int64_t seed;
} qcfg;

static int load_cfg(qcfg *c, const char *dir, err_t *err) {
    memset(c, 0, sizeof(*c));
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        err_set(err, "cannot open %s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc((size_t)len + 1);
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = 0;
    char jerr[256];
    json_value *root = json_parse(buf, got, jerr, sizeof(jerr));
    free(buf);
    if (!root) {
        err_set(err, "config.json: %s", jerr);
        return -1;
    }
    json_value *text = json_obj_get(root, "text_config");
    if (!text) text = root;
    if (!text || text->type != JSON_OBJ) {
        err_set(err, "config.json: no text_config");
        json_free(root);
        return -1;
    }
    c->hidden_size = (int)json_num_i64(json_obj_get(text, "hidden_size"));
    c->num_layers = (int)json_num_i64(json_obj_get(text, "num_hidden_layers"));
    c->num_attn_heads = (int)json_num_i64(json_obj_get(text, "num_attention_heads"));
    c->num_kv_heads = (int)json_num_i64(json_obj_get(text, "num_key_value_heads"));
    c->head_dim = (int)json_num_i64(json_obj_get(text, "head_dim"));
    c->vocab_size = json_num_i64(json_obj_get(text, "vocab_size"));
    c->full_attention_interval = (int)json_num_i64(json_obj_get(text, "full_attention_interval"));
    c->num_experts = (int)json_num_i64(json_obj_get(text, "num_experts"));
    c->num_experts_per_tok = (int)json_num_i64(json_obj_get(text, "num_experts_per_tok"));
    c->moe_intermediate_size = (int)json_num_i64(json_obj_get(text, "moe_intermediate_size"));
    c->shared_expert_intermediate_size =
        (int)json_num_i64(json_obj_get(text, "shared_expert_intermediate_size"));
    c->linear_num_key_heads = (int)json_num_i64(json_obj_get(text, "linear_num_key_heads"));
    c->linear_num_value_heads = (int)json_num_i64(json_obj_get(text, "linear_num_value_heads"));
    c->linear_key_head_dim = (int)json_num_i64(json_obj_get(text, "linear_key_head_dim"));
    c->linear_value_head_dim = (int)json_num_i64(json_obj_get(text, "linear_value_head_dim"));
    c->linear_conv_kernel_dim = (int)json_num_i64(json_obj_get(text, "linear_conv_kernel_dim"));
    c->hc_count = (int)json_num_i64(json_obj_get(text, "hc_count"));
    c->hc_lowrank = (int)json_num_i64(json_obj_get(text, "hc_lowrank"));
    c->indexer_n_heads = (int)json_num_i64(json_obj_get(text, "indexer_n_heads"));
    c->indexer_kv_heads = (int)json_num_i64(json_obj_get(text, "indexer_kv_heads"));
    c->indexer_head_dim = (int)json_num_i64(json_obj_get(text, "indexer_head_dim"));
    c->indexer_budget = (int)json_num_i64(json_obj_get(text, "indexer_budget"));
    c->indexer_compress_ratio = (int)json_num_i64(json_obj_get(text, "indexer_compress_ratio"));
    c->ngram_size = (int)json_num_i64(json_obj_get(text, "ngram_size"));
    c->heads_per_ngram = (int)json_num_i64(json_obj_get(text, "heads_per_ngram"));
    c->ngram_vocab_size_base = json_num_i64(json_obj_get(text, "ngram_vocab_size_base"));
    c->split_ngram_parts = (int)json_num_i64(json_obj_get(text, "split_ngram_parts"));
    c->ple_embed_dim = (int)json_num_i64(json_obj_get(text, "ple_embed_dim"));
    c->ple_conv_kernel_size = (int)json_num_i64(json_obj_get(text, "ple_conv_kernel_size"));
    c->eos_token_id = json_num_i64(json_obj_get(text, "eos_token_id"));
    c->partial_rotary_factor = (float)json_num_f64(json_obj_get(text, "partial_rotary_factor"));
    c->seed = json_num_i64(json_obj_get(text, "seed"));
    json_value *rp = json_obj_get(text, "rope_parameters");
    c->rope_theta = rp ? (float)json_num_f64(json_obj_get(rp, "rope_theta")) : 10000000.0f;
    c->rms_norm_eps = 1e-6f;
    json_value *lt = json_obj_get(text, "layer_types");
    if (lt && lt->type == JSON_ARR) {
        c->n_layers = 0;
        for (size_t i = 0; i < lt->u.arr.n && i < QMODEL_MAX_LAYERS; i++) {
            json_value *v = lt->u.arr.items[i];
            if (v->type == JSON_STR && strcmp(v->u.str.s, "full_attention") == 0)
                c->layer_types[c->n_layers++] = 1;
            else
                c->layer_types[c->n_layers++] = 0;
        }
    } else {
        for (int i = 0; i < c->num_layers; i++) {
            c->layer_types[i] = ((i + 1) % c->full_attention_interval == 0) ? 1 : 0;
        }
        c->n_layers = c->num_layers;
    }
    json_value *ple_ids = json_obj_get(text, "ple_layer_ids");
    if (ple_ids && ple_ids->type == JSON_ARR && ple_ids->u.arr.n > 0) {
        c->ple_layer = (int)json_num_i64(ple_ids->u.arr.items[0]) - 1;
    } else {
        c->ple_layer = 0;
    }
    json_value *quant = json_obj_get(root, "quantization");
    c->group_size = quant ? (int)json_num_i64(json_obj_get(quant, "group_size")) : 64;
    c->bits = quant ? (int)json_num_i64(json_obj_get(quant, "bits")) : 4;
    c->ple_group_size = c->group_size;
    if (quant) {
        char pname[512];
        snprintf(pname, sizeof(pname),
                 "model.layers.%d.ple.ple_embedding.ngram_embedding.shard_0", c->ple_layer);
        json_value *pq = json_obj_get(quant, pname);
        if (pq && pq->type == JSON_OBJ) {
            c->ple_group_size = (int)json_num_i64(json_obj_get(pq, "group_size"));
            if (c->ple_group_size <= 0) c->ple_group_size = 32;
        }
    }
    json_free(root);
    if (c->hidden_size < 1 || c->num_layers < 1 || c->vocab_size < 1) {
        err_set(err, "config.json: missing required fields");
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// weights

typedef struct {
    char name[160];
    st_tensor *t;
    int is_q4;
    const uint16_t *bf16;
    const uint32_t *q;
    float *scales, *biases;
    int64_t rows, cols;
} mweight;

typedef struct {
    mweight **items; // stable element pointers; weights are handed out as
                     // mweight * and must not move after mw_find
    size_t n, cap;
} mwset;

static mweight *mw_add(mwset *s, const char *name) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 64;
        s->items = xrealloc(s->items, s->cap * sizeof(mweight *));
    }
    mweight *w = xcalloc(1, sizeof(mweight));
    s->items[s->n++] = w;
    snprintf(w->name, sizeof(w->name), "%s", name);
    return w;
}

static mweight *mw_find(mwset *s, const char *name) {
    for (size_t i = 0; i < s->n; i++) {
        if (strcmp(s->items[i]->name, name) == 0) return s->items[i];
    }
    return NULL;
}

// linear layer dispatch: q4 weights use the packed gemm, bf16 use bnns
static void linear_forward(mweight *w, int M, int N, int K, const float *A,
                           float *C, int group_size) {
    if (w->is_q4) {
        K = (int)w->cols * 8; // packed q4: shape[1] counts u32 words
        gemm_q4(M, N, K, A, w->q, w->scales, w->biases, group_size, C);
    } else {
        gemm_bf16(M, N, K, A, w->bf16, C);
    }
}

// quantized checkpoints keep scales/biases beside the weight without the
// ".weight" suffix ("<base>.weight" -> "<base>.scales")
static void side_name(char *out, size_t cap, const char *weight_name,
                      const char *suffix) {
    size_t n = strlen(weight_name);
    if (n > 7 && strcmp(weight_name + n - 7, ".weight") == 0) n -= 7;
    snprintf(out, cap, "%.*s.%s", (int)n, weight_name, suffix);
}

// load a tensor into a mweight (q4 if scales exist, else bf16). scales and
// biases are converted to fp32.
static int mw_load(mwset *s, st_index *ix, const char *name, err_t *err) {
    if (mw_find(s, name)) return 0;
    st_tensor *t = st_find(ix, name);
    if (!t) {
        err_set(err, "missing tensor %s", name);
        return -1;
    }
    if (t->ndim < 2) {
        err_set(err, "unexpected shape for %s", name);
        return -1;
    }
    mweight *w = mw_add(s, name);
    w->t = t;
    w->rows = t->shape[0];
    w->cols = t->shape[1];
    char sn[512], bn[512];
    side_name(sn, sizeof(sn), name, "scales");
    side_name(bn, sizeof(bn), name, "biases");
    st_tensor *st_scales = st_find(ix, sn);
    st_tensor *st_biases = st_find(ix, bn);
    if (st_scales) {
        w->is_q4 = 1;
        if (t->dtype != ST_U32) {
            err_set(err, "quantized tensor %s must be U32", name);
            return -1;
        }
        size_t nq = (size_t)t->nbytes / 4;
        w->q = xmalloc(t->nbytes);
        if (st_read_tensor(ix, t, (void *)w->q, err) != 0) return -1;
        size_t ngroups = (size_t)st_scales->nbytes / 2;
        w->scales = xmalloc(ngroups * sizeof(float));
        w->biases = xmalloc(ngroups * sizeof(float));
        uint16_t *raws = xmalloc(st_scales->nbytes);
        uint16_t *rawb = xmalloc(st_biases ? st_biases->nbytes : 1);
        if (st_read_tensor(ix, st_scales, raws, err) != 0) return -1;
        if (st_biases) {
            if (st_read_tensor(ix, st_biases, rawb, err) != 0) return -1;
        } else {
            for (size_t i = 0; i < ngroups; i++) rawb[i] = 0;
        }
        for (size_t i = 0; i < ngroups; i++) {
            w->scales[i] = bf16_to_f32(raws[i]);
            w->biases[i] = bf16_to_f32(rawb[i]);
        }
        free(raws);
        free(rawb);
        (void)nq;
    } else {
        w->is_q4 = 0;
        if (t->dtype != ST_BF16) {
            err_set(err, "dense tensor %s must be BF16 (got dtype %d)", name, t->dtype);
            return -1;
        }
        w->bf16 = xmalloc(t->nbytes);
        if (st_read_tensor(ix, t, (void *)w->bf16, err) != 0) return -1;
    }
    return 0;
}

// load a small bf16 tensor as fp32 vector
static int mw_load_f32(const float **out, size_t *n, st_index *ix, const char *name,
                       err_t *err) {
    st_tensor *t = st_find(ix, name);
    if (!t) {
        err_set(err, "missing tensor %s", name);
        return -1;
    }
    size_t cnt = (size_t)(t->nbytes / 2);
    uint16_t *raw = xmalloc(t->nbytes);
    if (st_read_tensor(ix, t, raw, err) != 0) {
        free(raw);
        return -1;
    }
    float *outv = xmalloc(cnt * sizeof(float));
    for (size_t i = 0; i < cnt; i++) outv[i] = bf16_to_f32(raw[i]);
    free(raw);
    *out = outv;
    *n = cnt;
    return 0;
}

static int mw_load_i64(int64_t **out, size_t *n, st_index *ix, const char *name,
                       err_t *err) {
    st_tensor *t = st_find(ix, name);
    if (!t) {
        err_set(err, "missing tensor %s", name);
        return -1;
    }
    size_t cnt = (size_t)(t->nbytes / 8);
    int64_t *outv = xmalloc(cnt * sizeof(int64_t));
    if (st_read_tensor(ix, t, outv, err) != 0) {
        free(outv);
        return -1;
    }
    *out = outv;
    *n = cnt;
    return 0;
}

// ---------------------------------------------------------------------------
// model

typedef struct {
    mweight *w;
    int is_full;
    int is_ple;
    // attention
    mweight *q_proj, *k_proj, *v_proj, *o_proj;
    const float *q_norm, *k_norm;
    mweight *idx_qk;
    const float *idx_q_norm, *idx_k_norm;
    // linear attention
    mweight *in_qkv, *in_z, *in_a, *in_b, *out_proj, *conv1d;
    const float *lin_norm;
    const float *a_log, *dt_bias;
    // moe
    mweight *gate, *sh_g, *sh_u, *sh_d, *sh_gate;
    // hyper connections
    const float *ah_norm, *mh_norm;
    mweight *ah_down, *ah_up, *ah_inject, *mh_down, *mh_up, *mh_inject;
    // ple
    mweight *ple_key, *ple_val, *ple_conv1d;
    const float *ple_norm_key, *ple_norm_query, *ple_norm_conv;
    // caches
    float *k_cache, *v_cache, *idx_keys;
    int kv_len, idx_len;
    float *conv_state, *rec_state, *ple_state;
    int64_t prev_ctx[2];
    int has_prev;
} qlayer;

struct qmodel {
    qcfg cfg;
    st_index ix;
    mwset mw;
    mweight *embed, *lm_head;
    const float *mix_norm;
    mweight *mix_down, *mix_up;
    int64_t *ng_mult, *ng_offsets, *ng_sizes;
    size_t ng_nheads;
    qlayer layers[QMODEL_MAX_LAYERS];
    int n_layers;
    cache_t expert_cache, ple_cache;
    uint64_t expert_budget, ple_budget;
    int64_t ple_rows_per_shard;
    qstats stats;
    tokenizer *tok;
};

static void qlayer_free(qlayer *l) {
    free(l->k_cache);
    free(l->v_cache);
    free(l->idx_keys);
    free(l->conv_state);
    free(l->rec_state);
    free(l->ple_state);
    memset(l, 0, sizeof(*l));
}

void model_free(qmodel *m) {
    if (!m) return;
    for (int i = 0; i < m->n_layers; i++) qlayer_free(&m->layers[i]);
    free(m->ng_mult);
    free(m->ng_offsets);
    free(m->ng_sizes);
    for (size_t i = 0; i < m->mw.n; i++) {
        free((void *)m->mw.items[i]->bf16);
        free((void *)m->mw.items[i]->q);
        free(m->mw.items[i]->scales);
        free(m->mw.items[i]->biases);
        free(m->mw.items[i]);
    }
    free(m->mw.items);
    cache_destroy(&m->expert_cache);
    cache_destroy(&m->ple_cache);
    st_index_close(&m->ix);
    free(m);
}

void model_reset_caches(qmodel *m) {
    for (int i = 0; i < m->n_layers; i++) {
        qlayer *l = &m->layers[i];
        l->kv_len = 0;
        l->idx_len = 0;
        l->has_prev = 0;
        l->prev_ctx[0] = m->cfg.eos_token_id;
        l->prev_ctx[1] = m->cfg.eos_token_id;
        if (l->conv_state) memset(l->conv_state, 0, (size_t)(m->cfg.linear_conv_kernel_dim - 1) * (2 * m->cfg.linear_num_key_heads * m->cfg.linear_key_head_dim + m->cfg.linear_num_value_heads * m->cfg.linear_value_head_dim) * sizeof(float));
        if (l->rec_state) memset(l->rec_state, 0, (size_t)m->cfg.linear_num_value_heads * m->cfg.linear_value_head_dim * m->cfg.linear_key_head_dim * sizeof(float));
        if (l->ple_state) memset(l->ple_state, 0, (size_t)((m->cfg.ple_conv_kernel_size - 1) * m->cfg.ngram_size) * (m->cfg.hc_count * m->cfg.hidden_size) * sizeof(float));
    }
}

int64_t model_vocab(const qmodel *m) { return m->cfg.vocab_size; }
int64_t model_eos(const qmodel *m) { return m->cfg.eos_token_id; }
const qstats *model_stats(const qmodel *m) { return &m->stats; }
void model_set_tokenizer(qmodel *m, tokenizer *tok) { m->tok = tok; }

// ---------------------------------------------------------------------------
// expert + ple stores

typedef struct {
    qmodel *m;
    int64_t rows_per_shard;
    uint64_t total_ple_rows;
} stores;

// dequantize nrows already-read raw rows of a q4 tensor into fp32
// (nrows x cols). group_size applies to the q4 path.
static int dequant_raw(st_tensor *t, st_tensor *st_scales, st_tensor *st_biases,
                       const uint8_t *rawq, const uint8_t *raws,
                       const uint8_t *rawb, size_t nrows, float *out,
                       int group_size) {
    // 2d tensors are [rows, cols]; 3d tensors (expert blocks) are
    // [rows, sub, cols] and one row dequantizes to sub * cols values
    int64_t sub = t->ndim == 3 ? t->shape[1] : 1;
    int64_t cols = t->shape[1] * (t->ndim == 3 ? t->shape[2] : 1);
    cols = cols * 8; // q4: stored cols = cols/8
    size_t row_bytes_q = (size_t)(t->nbytes / (uint64_t)t->shape[0]);
    size_t scale_size = st_scales->dtype == ST_BF16 ? 2 : 4;
    size_t row_bytes_s = (size_t)(st_scales->nbytes / (uint64_t)st_scales->shape[0]);
    size_t groups_per_row = row_bytes_s / scale_size / (size_t)sub;
    size_t vals_per_sub = (size_t)(cols / sub);
    for (size_t r = 0; r < nrows; r++) {
        const uint32_t *qrow = (const uint32_t *)(rawq + r * row_bytes_q);
        float scales[512], biases[512];
        float *sp = scales, *bp = biases;
        if (groups_per_row * (size_t)sub > 512) {
            sp = xmalloc(groups_per_row * (size_t)sub * sizeof(float));
            bp = xmalloc(groups_per_row * (size_t)sub * sizeof(float));
        }
        const uint16_t *srow = (const uint16_t *)(raws + r * row_bytes_s);
        const uint16_t *brow = (const uint16_t *)(rawb + r * row_bytes_s);
        if (scale_size == 2) {
            for (size_t g = 0; g < groups_per_row * (size_t)sub; g++) {
                sp[g] = bf16_to_f32(srow[g]);
                bp[g] = bf16_to_f32(brow[g]);
            }
        } else {
            memcpy(sp, raws + r * row_bytes_s,
                   groups_per_row * (size_t)sub * sizeof(float));
            memcpy(bp, rawb + r * row_bytes_s,
                   groups_per_row * (size_t)sub * sizeof(float));
        }
        for (int64_t s = 0; s < sub; s++) {
            kernel_dequant_q4_row(qrow + (size_t)s * (vals_per_sub / 8),
                                  sp + (size_t)s * groups_per_row,
                                  bp + (size_t)s * groups_per_row,
                                  vals_per_sub, group_size,
                                  out + r * (size_t)cols + (size_t)s * vals_per_sub);
        }
        if (sp != scales) {
            free(sp);
            free(bp);
        }
    }
    return 0;
}

// read rows of a q4 (or plain bf16) tensor and expand into fp32
// (nrows x cols). group_size applies to the q4 path.
static int read_dequant_rows(qmodel *m, mweight *w, const int64_t *rows,
                             size_t nrows, float *out, int group_size,
                             err_t *err) {
    if (nrows == 0) return 0;
    st_tensor *t = w->t;
    // 2d tensors are [rows, cols]; 3d tensors (expert blocks) are
    // [rows, sub, cols] and one row dequantizes to sub * cols values
    int64_t sub = t->ndim == 3 ? t->shape[1] : 1;
    int64_t cols = t->shape[1] * (t->ndim == 3 ? t->shape[2] : 1);
    char sn[512], bn[512];
    side_name(sn, sizeof(sn), t->name, "scales");
    side_name(bn, sizeof(bn), t->name, "biases");
    st_tensor *st_scales = st_find(&m->ix, sn);
    st_tensor *st_biases = st_find(&m->ix, bn);
    if (!st_scales) {
        // plain bf16 rows
        if (t->dtype != ST_BF16) {
            err_set(err, "tensor %s must be BF16 or quantized", t->name);
            return -1;
        }
        size_t row_bytes = (size_t)(t->nbytes / (uint64_t)t->shape[0]);
        uint8_t *raw = xmalloc(row_bytes * nrows);
        if (st_read_rows(&m->ix, t, rows, nrows, raw, err) != 0) {
            free(raw);
            return -1;
        }
        for (size_t r = 0; r < nrows; r++) {
            const uint16_t *rowp = (const uint16_t *)(raw + r * row_bytes);
            float *orow = out + r * (size_t)(sub * cols);
            for (int64_t i = 0; i < sub * cols; i++) orow[i] = bf16_to_f32(rowp[i]);
        }
        free(raw);
        return 0;
    }
    cols = cols * 8; // q4: stored cols = cols/8
    size_t row_bytes_q = (size_t)(t->nbytes / (uint64_t)t->shape[0]);
    size_t row_bytes_s = (size_t)(st_scales->nbytes / (uint64_t)st_scales->shape[0]);
    uint8_t *rawq = xmalloc(row_bytes_q * nrows);
    uint8_t *raws = xmalloc(row_bytes_s * nrows);
    uint8_t *rawb = xmalloc(row_bytes_s * nrows);
    if (st_read_rows(&m->ix, t, rows, nrows, rawq, err) != 0) goto fail;
    if (st_read_rows(&m->ix, st_scales, rows, nrows, raws, err) != 0) goto fail;
    if (st_biases) {
        if (st_read_rows(&m->ix, st_biases, rows, nrows, rawb, err) != 0) goto fail;
    } else {
        memset(rawb, 0, row_bytes_s * nrows);
    }
    int rc = dequant_raw(t, st_scales, st_biases, rawq, raws, rawb, nrows, out,
                         group_size);
    free(rawq);
    free(raws);
    free(rawb);
    return rc;
fail:
    free(rawq);
    free(raws);
    free(rawb);
    return -1;
}

// load a set of experts for one layer into the cache. all raw row reads are
// submitted to the io pool up front so the device sees a deep queue, then
// each expert is dequantized as its reads complete.
static int experts_load(qmodel *m, int layer, const int64_t *ids, size_t n,
                        err_t *err) {
    char gname[256], uname[256], dname[256];
    snprintf(gname, sizeof(gname),
             "model.layers.%d.mlp.switch_mlp.gate_proj.weight", layer);
    snprintf(uname, sizeof(uname),
             "model.layers.%d.mlp.switch_mlp.up_proj.weight", layer);
    snprintf(dname, sizeof(dname),
             "model.layers.%d.mlp.switch_mlp.down_proj.weight", layer);
    st_tensor *tg = st_find(&m->ix, gname);
    st_tensor *tu = st_find(&m->ix, uname);
    st_tensor *td = st_find(&m->ix, dname);
    if (!tg || !tu || !td) {
        err_set(err, "missing expert tensors for layer %d", layer);
        return -1;
    }
    // expert blocks are [n_experts, moe_inter, hidden/8] (down:
    // [n_experts, hidden, moe_inter/8]); one row dequantizes to a
    // [moe_inter, hidden] matrix
    int64_t cols = tg->shape[2] * 8;
    int64_t moe_inter = tg->shape[1];
    int64_t dcols = td->shape[2] * 8;
    int64_t dmoe_inter = td->shape[1];
    size_t qrow_g = (size_t)(tg->nbytes / (uint64_t)tg->shape[0]);
    size_t qrow_d = (size_t)(td->nbytes / (uint64_t)td->shape[0]);
    st_tensor *sg = st_find(&m->ix, gname), *bg = NULL;
    st_tensor *su = st_find(&m->ix, uname), *bu = NULL;
    st_tensor *sd = st_find(&m->ix, dname), *bd = NULL;
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", gname);
        side_name(buf, sizeof(buf), gname, "scales");
        sg = st_find(&m->ix, buf);
        side_name(buf, sizeof(buf), gname, "biases");
        bg = st_find(&m->ix, buf);
        side_name(buf, sizeof(buf), uname, "scales");
        su = st_find(&m->ix, buf);
        side_name(buf, sizeof(buf), uname, "biases");
        bu = st_find(&m->ix, buf);
        side_name(buf, sizeof(buf), dname, "scales");
        sd = st_find(&m->ix, buf);
        side_name(buf, sizeof(buf), dname, "biases");
        bd = st_find(&m->ix, buf);
    }
    if (!sg || !su || !sd || !bg || !bu || !bd) {
        err_set(err, "missing expert scales/biases for layer %d", layer);
        return -1;
    }
    size_t srow_g = (size_t)(sg->nbytes / (uint64_t)sg->shape[0]);
    size_t srow_d = (size_t)(sd->nbytes / (uint64_t)sd->shape[0]);

    // collect the experts that are not cached
    int64_t *pending = xmalloc((size_t)n * sizeof(int64_t));
    size_t npending = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)ids[i];
        if (cache_peek(&m->expert_cache, key)) continue;
        pending[npending++] = ids[i];
    }
    if (!npending) {
        free(pending);
        return 0;
    }

    // per-expert raw block: gate q/s/b, up q/s/b, down q/s/b
    size_t off[9];
    off[0] = 0;               // gate q
    off[1] = off[0] + qrow_g; // gate s
    off[2] = off[1] + srow_g; // gate b
    off[3] = off[2] + srow_g; // up q
    off[4] = off[3] + qrow_g; // up s
    off[5] = off[4] + srow_g; // up b
    off[6] = off[5] + srow_g; // down q
    off[7] = off[6] + qrow_d; // down s
    off[8] = off[7] + srow_d; // down b
    size_t blk = off[8] + srow_d;
    io_job *jobs = xcalloc(npending * 9, sizeof(io_job));
    uint8_t *raw = xmalloc(npending * blk);
    int64_t *row_of = xmalloc(npending * sizeof(int64_t));
    for (size_t i = 0; i < npending; i++) {
        row_of[i] = pending[i];
        uint8_t *b = raw + i * blk;
        st_tensor *targs[9] = {tg, sg, bg, tu, su, bu, td, sd, bd};
        for (int j = 0; j < 9; j++) {
            if (st_read_rows_async(&m->ix, targs[j], &row_of[i], 1, b + off[j],
                                   &jobs[i * 9 + j]) != 0) {
                err_set(err, "expert row submit failed for layer %d", layer);
                goto fail;
            }
        }
    }

    for (size_t i = 0; i < npending; i++) {
        double t0 = now_s();
        for (int j = 0; j < 9; j++) {
            int st = io_job_wait(&jobs[i * 9 + j]);
            if (st) {
                err_set(err, "expert row read failed for layer %d: %s", layer,
                        strerror(st));
                goto fail;
            }
        }
        m->ix.rows_read += 9;
        uint8_t *b = raw + i * blk;
        float *w = xmalloc((size_t)(cols * moe_inter + dcols * dmoe_inter) * 2 *
                           sizeof(float));
        if (dequant_raw(tg, sg, bg, b + off[0], b + off[1], b + off[2], 1, w,
                        m->cfg.group_size) != 0 ||
            dequant_raw(tu, su, bu, b + off[3], b + off[4], b + off[5], 1,
                        w + (size_t)cols * moe_inter, m->cfg.group_size) != 0 ||
            dequant_raw(td, sd, bd, b + off[6], b + off[7], b + off[8], 1,
                        w + (size_t)cols * moe_inter * 2,
                        m->cfg.group_size) != 0) {
            free(w);
            goto fail;
        }
        double dt = now_s() - t0;
        m->stats.expert_loads++;
        m->stats.expert_wait_seconds += dt;
        uint64_t bytes = (uint64_t)(cols * moe_inter + dcols * dmoe_inter) * 2 * 4;
        m->stats.expert_bytes += bytes;
        uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)pending[i];
        if (!cache_put(&m->expert_cache, key, w, bytes, 0, 0)) free(w);
    }
    free(pending);
    free(jobs);
    free(raw);
    free(row_of);
    return 0;
fail:
    free(pending);
    free(jobs);
    free(raw);
    free(row_of);
    return -1;
}

static int ple_rows_load(qmodel *m, const int64_t *gids, size_t n, float *out,
                         err_t *err) {
    int64_t rps = m->ple_rows_per_shard;
    int nhead = (m->cfg.ngram_size - 1) * m->cfg.heads_per_ngram;
    int row_width = m->cfg.ple_embed_dim / nhead;
    for (size_t i = 0; i < n; i++) {
        int64_t gid = gids[i];
        int64_t shard = gid / rps;
        int64_t row = gid % rps;
        uint64_t key = ((uint64_t)(uint32_t)shard << 32) | (uint32_t)row;
        float *v = cache_get(&m->ple_cache, key);
        if (!v) {
            char wname[512];
            snprintf(wname, sizeof(wname),
                     "model.layers.%d.ple.ple_embedding.ngram_embedding.shard_%lld.weight",
                     m->cfg.ple_layer, (long long)shard);
            mweight *wshard = mw_find(&m->mw, wname);
            if (!wshard) {
                // shard tensors are streamed, not in the dense set; build a
                // temporary mweight around the index entry
                wshard = xcalloc(1, sizeof(mweight));
                st_tensor *t = st_find(&m->ix, wname);
                if (!t) {
                    err_set(err, "missing PLE shard tensor %s", wname);
                    free(wshard);
                    return -1;
                }
                wshard->t = t;
                wshard->rows = t->shape[0];
                wshard->cols = t->shape[1];
            }
            v = xmalloc((size_t)row_width * sizeof(float));
            int64_t one = row;
            double t0 = now_s();
            if (read_dequant_rows(m, wshard, &one, 1, v, m->cfg.ple_group_size, err) != 0) { free(v); return -1; }
            m->stats.ple_loads++;
            m->stats.ple_wait_seconds += now_s() - t0;
            m->stats.ple_bytes += (uint64_t)row_width * 4;
            if (!cache_put(&m->ple_cache, key, v, (uint64_t)row_width * 4, 0, 0)) {
                free(v);
                return -1;
            }
        }
        memcpy(out + i * (size_t)row_width, v, (size_t)row_width * sizeof(float));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// layer forward helpers

// x: [n, hc*d] hyper state; returns mixed [n, d] and optionally inject [n, hc]
static void gated_residual(const float *x, const float *hc_norm,
                           mweight *down, mweight *up, mweight *inject,
                           int hc, int d, int lowrank, int n, float *mixed,
                           float *inject_out, float *scratch) {
    // normed = rmsnorm_grouped(x, hc_norm) over groups of d
    kernel_rmsnorm_grouped(x, hc_norm, (size_t)n * hc * d, (size_t)d, (size_t)n, 1e-6f, scratch);
    float *w = scratch + (size_t)n * hc * d; // [n, lowrank]
    float *tmp = w + (size_t)n * lowrank;    // [n, hc*d]
    // w = silu(down(normed) / hc)
    linear_forward(down, n, (int)down->rows, (int)down->cols, scratch, w, 64);
    kernel_scale(w, 1.0f / hc, (size_t)n * lowrank, w);
    kernel_silu(w, (size_t)n * lowrank, w);
    // w = sigmoid(up(w)) -> [n, hc, d]
    linear_forward(up, n, (int)up->rows, (int)up->cols, w, tmp, 64);
    kernel_sigmoid(tmp, (size_t)n * hc * d, tmp);
    
    // mixed = mean over hc of tmp * normed
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < d; i++) {
            float acc = 0;
            for (int h = 0; h < hc; h++) {
                acc += tmp[(size_t)t * hc * d + h * d + i] *
                       scratch[(size_t)t * hc * d + h * d + i];
            }
            mixed[(size_t)t * d + i] = acc / hc;
        }
    }
    if (inject_out) {
        // inject = 2 * sigmoid(block_inject(normed) / hc) -> [n, hc]
        gemm_bf16(n, (int)inject->rows, (int)inject->cols, scratch, inject->bf16,
                  inject_out);
        kernel_scale(inject_out, 1.0f / hc, (size_t)n * hc, inject_out);
        kernel_sigmoid(inject_out, (size_t)n * hc, inject_out);
        kernel_scale(inject_out, 2.0f, (size_t)n * hc, inject_out);
    }
    
}

// full attention with QSA indexer. x: [n, d]; out: [n, d]
static void attention_forward(qmodel *m, qlayer *l, const float *x, int n,
                              int offset, float *out, float *scratch) {
    qcfg *c = &m->cfg;
    int d = c->hidden_size;
    int h = c->num_attn_heads;
    int hd = c->head_dim;
    int kvh = c->num_kv_heads;
    // --- indexer ---
    float *qk = scratch;                       // [n, idx_heads*idx_hd + idx_hd]
    float *q_ = qk + (size_t)n * (c->indexer_n_heads * c->indexer_head_dim + c->indexer_head_dim);
    int idx_out = c->indexer_n_heads * c->indexer_head_dim;
    gemm_bf16(n, idx_out + c->indexer_head_dim, d, x, l->idx_qk->bf16, qk);
    
    // append raw_k to indexer key cache (raw_k is per-token at stride
    // idx_out + indexer_head_dim, not contiguous)
    float *rawk = qk + idx_out;
    if (l->idx_len + n > QMODEL_KV_CAP) {
        // drop oldest
        int drop = l->idx_len + n - QMODEL_KV_CAP;
        memmove(l->idx_keys, l->idx_keys + drop * c->indexer_head_dim,
                (size_t)(l->idx_len - drop) * c->indexer_head_dim * sizeof(float));
        l->idx_len -= drop;
    }
    for (int t = 0; t < n; t++) {
        memcpy(l->idx_keys + (size_t)(l->idx_len + t) * c->indexer_head_dim,
               rawk + (size_t)t * (idx_out + c->indexer_head_dim),
               (size_t)c->indexer_head_dim * sizeof(float));
    }
    int kv_len = l->idx_len + n;
    l->idx_len = kv_len;
    // q/k projections
    float *qbuf = q_ + (size_t)n * (idx_out + c->indexer_head_dim); // [n, h*hd*2]
    float *kbuf = qbuf + (size_t)n * h * hd * 2;                    // [n, kvh*hd]
    float *vbuf = kbuf + (size_t)n * kvh * hd;                      // [n, kvh*hd]
    float *scores = vbuf + (size_t)n * kvh * hd;                    // [n, h, kv_len]
    float *attn_out = scores + (size_t)n * h * kv_len;              // [n, h*hd]
    linear_forward(l->q_proj, n, h * hd * 2, d, x, qbuf, 64);
    linear_forward(l->k_proj, n, kvh * hd, d, x, kbuf, 64);
    linear_forward(l->v_proj, n, kvh * hd, d, x, vbuf, 64);
    // append k/v to cache
    if (l->kv_len + n > QMODEL_KV_CAP) {
        int drop = l->kv_len + n - QMODEL_KV_CAP;
        memmove(l->k_cache, l->k_cache + (size_t)drop * kvh * hd, (size_t)(l->kv_len - drop) * kvh * hd * sizeof(float));
        memmove(l->v_cache, l->v_cache + (size_t)drop * kvh * hd, (size_t)(l->kv_len - drop) * kvh * hd * sizeof(float));
        l->kv_len -= drop;
    }
    memcpy(l->k_cache + (size_t)l->kv_len * kvh * hd, kbuf, (size_t)n * kvh * hd * sizeof(float));
    memcpy(l->v_cache + (size_t)l->kv_len * kvh * hd, vbuf, (size_t)n * kvh * hd * sizeof(float));
    int kv_total = l->kv_len + n;
    l->kv_len = kv_total;
    
    // q/k norms and rope
    float *q_normed = attn_out; // reuse [n, h*hd]
    float *k_normed = q_normed + (size_t)n * h * hd; // [n, kvh*hd]
    float *rope_cos = k_normed + (size_t)n * kvh * hd; // [max(n, kv_len)]
    float *rope_sin = rope_cos + QMODEL_KV_CAP;
    int rot_dim = (int)(hd * c->partial_rotary_factor);
    // build cos/sin for positions offset..offset+n-1
    {
        float *inv = xmalloc((size_t)(rot_dim / 2) * sizeof(float));
        for (int i = 0; i < rot_dim / 2; i++) {
            inv[i] = powf(c->rope_theta, -(2.0f * i) / rot_dim);
        }
        for (int t = 0; t < n; t++) {
            int64_t pos = offset + t;
            for (int i = 0; i < rot_dim / 2; i++) {
                float ang = (float)pos * inv[i];
                rope_cos[t * (rot_dim / 2) + i] = cosf(ang);
                rope_sin[t * (rot_dim / 2) + i] = sinf(ang);
            }
        }
        free(inv);
    }
    for (int t = 0; t < n; t++) {
        const float *qrow = qbuf + (size_t)t * h * hd * 2;
        float *qrow_n = q_normed + (size_t)t * h * hd;
        for (int hh = 0; hh < h; hh++) {
            kernel_rmsnorm(qrow + (size_t)hh * hd, l->q_norm, (size_t)hd, 1e-6f, qrow_n + (size_t)hh * hd);
        }
        kernel_rope_partial(qrow_n, h, hd, (size_t)rot_dim, rope_cos + (size_t)t * (rot_dim / 2), rope_sin + (size_t)t * (rot_dim / 2), qrow_n);
        const float *krow = kbuf + (size_t)t * kvh * hd;
        float *krow_n = k_normed + (size_t)t * kvh * hd;
        for (int hh = 0; hh < kvh; hh++) {
            kernel_rmsnorm(krow + (size_t)hh * hd, l->k_norm, (size_t)hd, 1e-6f, krow_n + (size_t)hh * hd);
        }
        kernel_rope_partial(krow_n, kvh, hd, (size_t)rot_dim, rope_cos + (size_t)t * (rot_dim / 2), rope_sin + (size_t)t * (rot_dim / 2), krow_n);
    }
    
    // scores: [n, h, kv_total] with causal mask and sparse indexer mask
    for (int t = 0; t < n; t++) {
        for (int hh = 0; hh < h; hh++) {
            int kh = hh % kvh;
            const float *qrow = q_normed + ((size_t)t * h + hh) * hd;
            const float *kbase = l->k_cache + (size_t)kh * hd;
            float *srow = scores + ((size_t)t * h + hh) * kv_total;
            for (int kk = 0; kk < kv_total; kk++) {
                const float *krow = kbase + (size_t)kk * kvh * hd;
                float acc = 0;
                for (int i = 0; i < hd; i++) acc += qrow[i] * krow[i];
                srow[kk] = acc * (1.0f / sqrtf((float)hd));
                // causal
                if (kk > offset + t) srow[kk] = -INFINITY;
            }
        }
    }
    // sparse indexer mask (only when kv_total > budget)
    if (kv_total > c->indexer_budget) {
        // compute sparse mask per token and apply
        int compress = c->indexer_compress_ratio;
        int n_blocks = kv_total / compress;
        int block_topk = c->indexer_budget / compress;
        float *pooled = xmalloc((size_t)n_blocks * c->indexer_head_dim * sizeof(float));
        for (int b = 0; b < n_blocks; b++) {
            const float *block = l->idx_keys + (size_t)b * compress * c->indexer_head_dim;
            float *p = pooled + (size_t)b * c->indexer_head_dim;
            for (int i = 0; i < c->indexer_head_dim; i++) {
                float acc = 0;
                for (int j = 0; j < compress; j++) acc += block[(size_t)j * c->indexer_head_dim + i];
                p[i] = acc / compress;
            }
            kernel_rmsnorm(p, l->idx_k_norm, (size_t)c->indexer_head_dim, 1e-6f, p);
        }
        // rope pooled at block starts and q
        float *bcos = xmalloc((size_t)n_blocks * (rot_dim / 2) * sizeof(float));
        float *bsin = xmalloc((size_t)n_blocks * (rot_dim / 2) * sizeof(float));
        {
            float *inv = xmalloc((size_t)(rot_dim / 2) * sizeof(float));
            for (int i = 0; i < rot_dim / 2; i++) inv[i] = powf(c->rope_theta, -(2.0f * i) / rot_dim);
            for (int b = 0; b < n_blocks; b++) {
                for (int i = 0; i < rot_dim / 2; i++) {
                    float a = (float)(b * compress) * inv[i];
                    bcos[b * (rot_dim / 2) + i] = cosf(a);
                    bsin[b * (rot_dim / 2) + i] = sinf(a);
                }
            }
            free(inv);
        }
        kernel_rope_partial(pooled, (size_t)n_blocks, (size_t)c->indexer_head_dim, (size_t)rot_dim, bcos, bsin, pooled);
        float *qscore = xmalloc((size_t)c->indexer_n_heads * c->indexer_head_dim * sizeof(float));
        float *bscores = xmalloc((size_t)n_blocks * sizeof(float));
        int *top = xmalloc((size_t)block_topk * sizeof(int));
        for (int t = 0; t < n; t++) {
            // q indexer: [n, idx_heads, idx_hd] from qk
            for (int hh = 0; hh < c->indexer_n_heads; hh++) {
                kernel_rmsnorm(qk + (size_t)t * (idx_out + c->indexer_head_dim) + (size_t)hh * c->indexer_head_dim,
                               l->idx_q_norm,
                               (size_t)c->indexer_head_dim, 1e-6f,
                               qscore + (size_t)hh * c->indexer_head_dim);
            }
            kernel_rope_partial(qscore, c->indexer_n_heads, (size_t)c->indexer_head_dim, (size_t)rot_dim, rope_cos + (size_t)t * (rot_dim / 2), rope_sin + (size_t)t * (rot_dim / 2), qscore);
            // scores over blocks
            float *bscores = xmalloc((size_t)n_blocks * sizeof(float));
            for (int b = 0; b < n_blocks; b++) {
                float acc = 0;
                const float *pb = pooled + (size_t)b * c->indexer_head_dim;
                for (int hh = 0; hh < c->indexer_n_heads; hh++) {
                    const float *qq = qscore + (size_t)hh * c->indexer_head_dim;
                    float dot = 0;
                    for (int i = 0; i < c->indexer_head_dim; i++) {
                        float v = qq[i] * pb[i];
                        dot += v > 0 ? v : 0;
                    }
                    acc += dot;
                }
                bscores[b] = acc / sqrtf((float)c->indexer_head_dim);
                // visibility: block_end <= q_pos
                if ((int64_t)(b * compress + compress - 1) > offset + t) bscores[b] = -INFINITY;
            }
            
            // topk blocks
            int *top = xmalloc((size_t)block_topk * sizeof(int));
            kernel_topk_indices(bscores, (size_t)n_blocks, (size_t)block_topk, top);
            // apply mask: keep blocks + own tail
            for (int hh = 0; hh < h; hh++) {
                float *srow = scores + ((size_t)t * h + hh) * kv_total;
                for (int kk = 0; kk < kv_total; kk++) {
                    int block = kk / compress;
                    int keep = 0;
                    if (block < n_blocks) {
                        for (int b = 0; b < block_topk; b++) {
                            if (top[b] == block) { keep = 1; break; }
                        }
                    }
                    // own tail: keys in the current block up to q position
                    int64_t own_start = ((offset + t + 1) / compress) * compress;
                    if (kk >= own_start && kk <= offset + t) keep = 1;
                    if (!keep && kk <= offset + t) srow[kk] = -INFINITY;
                }
            }
        }
        free(qscore);
        free(bscores);
        free(top);
        free(pooled);
        free(bcos);
        free(bsin);
    }
    
    // softmax + weighted sum of v
    for (int t = 0; t < n; t++) {
        for (int hh = 0; hh < h; hh++) {
            float *srow = scores + ((size_t)t * h + hh) * kv_total;
            kernel_softmax(srow, (size_t)kv_total);
        }
    }
    for (int t = 0; t < n; t++) {
        for (int hh = 0; hh < h; hh++) {
            int kh = hh % kvh;
            const float *srow = scores + ((size_t)t * h + hh) * kv_total;
            const float *vbase = l->v_cache + (size_t)kh * hd;
            float *orow = attn_out + ((size_t)t * h + hh) * hd;
            for (int i = 0; i < hd; i++) {
                float acc = 0;
                for (int kk = 0; kk < kv_total; kk++) {
                    acc += srow[kk] * vbase[(size_t)kk * kvh * hd + i];
                }
                orow[i] = acc;
            }
        }
    }
    // gate the attention output (before o_proj), then project
    float *gatebuf = qbuf + (size_t)h * hd; // second half of each token's q_proj out
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < h * hd; i++) {
            float g = 1.0f / (1.0f + expf(-gatebuf[(size_t)t * h * hd * 2 + i]));
            attn_out[(size_t)t * h * hd + i] *= g;
        }
    }
    
    linear_forward(l->o_proj, n, d, h * hd, attn_out, out, 64);
}

// gated deltanet. x: [n, d]; out: [n, d]
static void deltanet_forward(qmodel *m, qlayer *l, const float *x, int n,
                             float *out, float *scratch) {
    qcfg *c = &m->cfg;
    int d = c->hidden_size;
    int hk = c->linear_num_key_heads, hv = c->linear_num_value_heads;
    int dk = c->linear_key_head_dim, dv = c->linear_value_head_dim;
    int key_dim = hk * dk, value_dim = hv * dv;
    int conv_dim = key_dim * 2 + value_dim;
    int kconv = c->linear_conv_kernel_dim;
    int repeat = hv / hk;
    float *mixed = scratch;                       // [n, conv_dim]
    float *z = mixed + (size_t)n * conv_dim;      // [n, value_dim]
    float *ab = z + (size_t)n * value_dim;        // [n, 2*hv]
    float *conv_in = ab + (size_t)n * 2 * hv;     // [(kconv-1)+n, conv_dim]
    float *conv_out = conv_in + (size_t)((kconv - 1) + n) * conv_dim; // [n, conv_dim]
    float *qkvbuf = conv_out + (size_t)n * conv_dim;                 // [n, conv_dim]
    float *a = ab, *b = ab + (size_t)n * hv;
    float *out_raw = qkvbuf + (size_t)n * conv_dim;                  // [n, hv*dv]
    float *g = out_raw + (size_t)n * hv * dv;                        // [n, hv]
    

    linear_forward(l->in_qkv, n, conv_dim, d, x, mixed, 64);
    
    linear_forward(l->in_z, n, value_dim, d, x, z, 64);
    
    gemm_bf16(n, hv, d, x, l->in_a->bf16, a);
    
    
    gemm_bf16(n, hv, d, x, l->in_b->bf16, b);
    
    memcpy(conv_in, l->conv_state, (size_t)(kconv - 1) * conv_dim * sizeof(float));
    memcpy(conv_in + (size_t)(kconv - 1) * conv_dim, mixed, (size_t)n * conv_dim * sizeof(float));
    
    memcpy(l->conv_state, conv_in + (size_t)n * conv_dim, (size_t)(kconv - 1) * conv_dim * sizeof(float));
    // conv1d depthwise over conv_in (kconv-1+n rows), take last n rows
    kernel_conv1d_depthwise(conv_in, (size_t)((kconv - 1) + n), (size_t)conv_dim,
                            l->conv1d->bf16, (size_t)kconv, 1, conv_out);
    // shift: conv_out[t] corresponds to input t - (kconv-1); we want output for
    // input positions kconv-1 .. kconv-1+n-1 -> conv_out rows (kconv-1)..(kconv-1+n-1)
    memmove(conv_out, conv_out + (size_t)(kconv - 1) * conv_dim, (size_t)n * conv_dim * sizeof(float));
    kernel_silu(conv_out, (size_t)n * conv_dim, conv_out);
    // split q/k/v
    
    // q/k/v are interleaved per token in the conv output [n, conv_dim]
    float *q = qkvbuf, *k = qkvbuf + (size_t)n * key_dim, *v = qkvbuf + (size_t)n * (key_dim * 2);
    for (int t = 0; t < n; t++) {
        const float *row = conv_out + (size_t)t * conv_dim;
        memcpy(q + (size_t)t * key_dim, row, (size_t)key_dim * sizeof(float));
        memcpy(k + (size_t)t * key_dim, row + key_dim, (size_t)key_dim * sizeof(float));
        memcpy(v + (size_t)t * value_dim, row + 2 * key_dim, (size_t)value_dim * sizeof(float));
    }
    
    // l2norm q/k and scale q
    float scale_q = 1.0f / sqrtf((float)dk);
    for (int t = 0; t < n; t++) {
        for (int hh = 0; hh < hk; hh++) {
            kernel_l2norm(q + ((size_t)t * hk + hh) * dk, (size_t)dk, 1e-6f, q + ((size_t)t * hk + hh) * dk);
            kernel_l2norm(k + ((size_t)t * hk + hh) * dk, (size_t)dk, 1e-6f, k + ((size_t)t * hk + hh) * dk);
            kernel_scale(q + ((size_t)t * hk + hh) * dk, scale_q, (size_t)dk, q + ((size_t)t * hk + hh) * dk);
        }
    }
    
    
    // recurrence
    for (int t = 0; t < n; t++) {
        const float *qt = q + (size_t)t * key_dim;
        const float *kt = k + (size_t)t * key_dim;
        const float *vt = v + (size_t)t * value_dim;
        for (int h = 0; h < hv; h++) {
            int kh = h / repeat;
            float gv = expf(-expf(l->a_log[h]) * log1pf(expf(a[t * hv + h] + l->dt_bias[h])));
            float beta = 1.0f / (1.0f + expf(-b[t * hv + h]));
            float *state = l->rec_state + ((size_t)h * dv) * dk;
            const float *krow = kt + (size_t)kh * dk;
            const float *qrow = qt + (size_t)kh * dk;
            const float *vrow = vt + (size_t)h * dv;
            for (int dd = 0; dd < dv; dd++) {
                float kv_mem = 0;
                float *srow = state + (size_t)dd * dk;
                for (int i = 0; i < dk; i++) {
                    srow[i] *= gv;
                    kv_mem += srow[i] * krow[i];
                }
                float delta = (vrow[dd] - kv_mem) * beta;
                for (int i = 0; i < dk; i++) srow[i] += krow[i] * delta;
                float acc = 0;
                for (int i = 0; i < dk; i++) acc += srow[i] * qrow[i];
                out_raw[(size_t)t * hv * dv + (size_t)h * dv + dd] = acc;
            }
        }
    }
    
    // rmsnorm_gated + out_proj
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < hv; h++) {
            float *orow = out_raw + ((size_t)t * hv + h) * dv;
            float *zrow = z + ((size_t)t * hv + h) * dv;
            kernel_rmsnorm(orow, l->lin_norm, (size_t)dv, 1e-6f, orow);
            for (int i = 0; i < dv; i++) {
                float sg = 1.0f / (1.0f + expf(-zrow[i]));
                orow[i] *= sg;
            }
        }
    }
    
    linear_forward(l->out_proj, n, d, value_dim, out_raw, out, 64);
    (void)g;
}

// moe. x: [n, d]; out: [n, d]
static int moe_forward(qmodel *m, qlayer *l, const float *x, int n, float *out,
                       float *scratch, err_t *err) {
    qcfg *c = &m->cfg;
    int d = c->hidden_size;
    int topk = c->num_experts_per_tok;
    int n_exp = c->num_experts;
    int moe_inter = c->moe_intermediate_size;
    // per-call buffers come from the layer scratch (free after the gated
    // residual that ran just before us)
    int64_t *unique = (int64_t *)scratch;
    int *idx = (int *)(unique + (size_t)n * topk);
    int *tok_ids = idx + (size_t)n * topk;
    float *weights = (float *)(tok_ids + (size_t)n * topk);
    float *logits = weights + (size_t)n * topk;
    float *x_sel = logits + (size_t)n * n_exp;
    float *g_out = x_sel + (size_t)n * d;
    float *u_out = g_out + (size_t)n * moe_inter;
    float *r_out = u_out + (size_t)n * moe_inter;
    float *sh = r_out + (size_t)n * d;
    float *sh_out = sh + (size_t)n * moe_inter;
    gemm_bf16(n, n_exp, d, x, l->gate->bf16, logits);
    for (int t = 0; t < n; t++) {
        kernel_topk_indices(logits + (size_t)t * n_exp, (size_t)n_exp, (size_t)topk,
                            idx + (size_t)t * topk);
        // softmax over selected
        float mx = -1e30f;
        for (int s = 0; s < topk; s++) {
            float v = logits[(size_t)t * n_exp + idx[(size_t)t * topk + s]];
            if (v > mx) mx = v;
        }
        float sum = 0;
        for (int s = 0; s < topk; s++) {
            float v = logits[(size_t)t * n_exp + idx[(size_t)t * topk + s]];
            float e = expf(v - mx);
            weights[(size_t)t * topk + s] = e;
            sum += e;
        }
        for (int s = 0; s < topk; s++) weights[(size_t)t * topk + s] /= sum;
    }
    memset(out, 0, (size_t)n * d * sizeof(float));
    // unique experts
    size_t nunique = 0;
    for (int t = 0; t < n; t++) {
        for (int s = 0; s < topk; s++) {
            int64_t e = idx[(size_t)t * topk + s];
            int seen = 0;
            for (size_t u = 0; u < nunique; u++) {
                if (unique[u] == e) { seen = 1; break; }
            }
            if (!seen) unique[nunique++] = e;
        }
    }
    if (experts_load(m, l - m->layers, unique, nunique, err) != 0) {
        return -1;
    }
    // per expert: gather tokens and compute
    int layer = (int)(l - m->layers);
    for (size_t u = 0; u < nunique; u++) {
        int64_t e = unique[u];
        uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)e;
        float *w = cache_get(&m->expert_cache, key);
        if (!w) {
            // the unique set exceeded the cache budget and this expert was
            // evicted before use; load it on demand
            if (experts_load(m, layer, &e, 1, err) != 0) {
                return -1;
            }
            w = cache_get(&m->expert_cache, key);
        }
        if (!w) {
            err_set(err, "expert %lld vanished from cache", (long long)e);
            return -1;
        }
        // gather tokens routing to this expert
        size_t ntoks = 0;
        for (int t = 0; t < n; t++) {
            for (int s = 0; s < topk; s++) {
                if (idx[(size_t)t * topk + s] == e) {
                    tok_ids[ntoks++] = t;
                    break;
                }
            }
        }
        if (ntoks == 0) continue;
        for (size_t i = 0; i < ntoks; i++) {
            memcpy(x_sel + i * (size_t)d, x + (size_t)tok_ids[i] * d, (size_t)d * sizeof(float));
        }
        const float *wg = w;
        const float *wu = w + (size_t)d * moe_inter;
        const float *wd = w + (size_t)d * moe_inter * 2;
        gemm_fp32((int)ntoks, moe_inter, d, x_sel, wg, g_out);
        gemm_fp32((int)ntoks, moe_inter, d, x_sel, wu, u_out);
        kernel_silu(g_out, ntoks * (size_t)moe_inter, g_out);
        kernel_mul(g_out, u_out, ntoks * (size_t)moe_inter, g_out);
        gemm_fp32((int)ntoks, d, moe_inter, g_out, wd, r_out);
        for (size_t i = 0; i < ntoks; i++) {
            int t = tok_ids[i];
            for (int s = 0; s < topk; s++) {
                if (idx[(size_t)t * topk + s] == e) {
                    float wgt = weights[(size_t)t * topk + s];
                    for (int j = 0; j < d; j++) {
                        out[(size_t)t * d + j] += wgt * r_out[i * (size_t)d + j];
                    }
                }
            }
        }
    }
    // shared expert
    {
        float *gatev = sh_out + (size_t)n * d;
        linear_forward(l->sh_g, n, moe_inter, d, x, sh, 64);
        linear_forward(l->sh_u, n, moe_inter, d, x, sh_out, 64);
        kernel_silu(sh, (size_t)n * moe_inter, sh);
        kernel_mul(sh, sh_out, (size_t)n * moe_inter, sh);
        linear_forward(l->sh_d, n, d, moe_inter, sh, sh_out, 64);
        gemm_bf16(n, 1, d, x, l->sh_gate->bf16, gatev);
        for (int t = 0; t < n; t++) {
            float g = 1.0f / (1.0f + expf(-gatev[t]));
            for (int j = 0; j < d; j++) {
                out[(size_t)t * d + j] += g * sh_out[(size_t)t * d + j];
            }
        }
    }
    return 0;
}

// ple layer: adds to h in place. tokens: current ids; prev_ctx: last 2 tokens
static int ple_forward(qmodel *m, qlayer *l, float *h, const int64_t *tokens,
                       int n, const int64_t *prev_ctx, float *scratch, err_t *err) {
    qcfg *c = &m->cfg;
    int d = c->hidden_size;
    int hc = c->hc_count;
    int hc_dim = hc * d;
    // build history: prev_ctx (2) + tokens
    int ctx_len = c->ngram_size - 1;
    int64_t *hist = xmalloc((size_t)(ctx_len + n) * sizeof(int64_t));
    for (int i = 0; i < ctx_len; i++) hist[i] = prev_ctx[i];
    memcpy(hist + ctx_len, tokens, (size_t)n * sizeof(int64_t));
    // compute gids for the last n positions
    int64_t *gids = xmalloc((size_t)(ctx_len + n) * ((c->ngram_size - 1) * c->heads_per_ngram) * sizeof(int64_t));
    {
        ngram_geo g;
        memset(&g, 0, sizeof(g));
        g.ngram_size = c->ngram_size;
        g.heads_per_ngram = c->heads_per_ngram;
        g.ngram_heads = (c->ngram_size - 1) * c->heads_per_ngram;
        g.vocab_size = c->vocab_size;
        g.eos_token_id = c->eos_token_id;
        memcpy(g.sizes, m->ng_sizes, (size_t)g.ngram_heads * sizeof(int64_t));
        memcpy(g.offsets, m->ng_offsets, (size_t)g.ngram_heads * sizeof(int64_t));
        memcpy(g.multipliers, m->ng_mult, (size_t)g.ngram_size * sizeof(int64_t));
        ngram_indices(&g, hist, (size_t)(ctx_len + n), gids);
        // take last n positions
        memmove(gids, gids + (size_t)ctx_len * g.ngram_heads, (size_t)n * g.ngram_heads * sizeof(int64_t));
    }
    // gather rows -> emb [n, nhead * row_width] = [n, ple_embed_dim]
    int nhead = (c->ngram_size - 1) * c->heads_per_ngram;
    float *emb = scratch;
    float *key = emb + (size_t)n * c->ple_embed_dim;           // [n, hc_dim]
    float *value = key + (size_t)n * hc_dim;                   // [n, d]
    float *query = value + (size_t)n * d;                      // [n, hc_dim]
    float *gated = query + (size_t)n * hc_dim;                 // [n, hc_dim]
    float *conv_in = gated + (size_t)n * hc_dim;               // [ple_state_len + n, hc_dim]
    if (ple_rows_load(m, gids, (size_t)n * nhead, emb, err) != 0) {
        free(hist);
        free(gids);
        return -1;
    }
    
    // key = norm_key(key_proj(emb)) [n, hc, d]
    // the projectors use the global group size; only the ngram embedding
    // shards are quantized at ple_group_size
    linear_forward(l->ple_key, n, hc_dim, c->ple_embed_dim, emb, key,
                   m->cfg.group_size);
    kernel_rmsnorm_grouped(key, l->ple_norm_key, (size_t)n * hc_dim, (size_t)d,
                           (size_t)n, 1e-6f, key);
    // value = value_proj(emb) [n, d]
    linear_forward(l->ple_val, n, d, c->ple_embed_dim, emb, value,
                   m->cfg.group_size);
    // query = norm_query(h) [n, hc, d]
    kernel_rmsnorm_grouped(h, l->ple_norm_query, (size_t)n * hc_dim, (size_t)d, (size_t)n, 1e-6f, query);
    
    // gate = sum(key*query)/sqrt(d) [n, hc]
    for (int t = 0; t < n; t++) {
        for (int hh = 0; hh < hc; hh++) {
            const float *kr = key + ((size_t)t * hc + hh) * d;
            const float *qr = query + ((size_t)t * hc + hh) * d;
            float acc = 0;
            for (int i = 0; i < d; i++) acc += kr[i] * qr[i];
            float gate = acc / sqrtf((float)d);
            float gabs = fabsf(gate);
            float g2 = sqrtf(gabs > 1e-6f ? gabs : 1e-6f) * (gate >= 0 ? 1.0f : -1.0f);
            float sg = 1.0f / (1.0f + expf(-g2));
            for (int i = 0; i < d; i++) {
                gated[(size_t)t * hc_dim + hh * d + i] = sg * value[(size_t)t * d + i];
            }
        }
    }
    // norm_conv + short conv (dilated)
    
    {
        int state_len = (c->ple_conv_kernel_size - 1) * c->ngram_size;
        // the residual adds the un-normed gated; the conv consumes the normed
        // version (colibri: gated + short_conv(norm_conv(gated)))
        float *gated_normed = conv_in + (size_t)(state_len + n) * hc_dim +
                              (size_t)n * hc_dim; // past conv_out's full span
        kernel_rmsnorm_grouped(gated, l->ple_norm_conv, (size_t)n * hc_dim,
                               (size_t)d, (size_t)n, 1e-6f, gated_normed);
        memcpy(conv_in, l->ple_state, (size_t)state_len * hc_dim * sizeof(float));
        memcpy(conv_in + (size_t)state_len * hc_dim, gated_normed, (size_t)n * hc_dim * sizeof(float));
        memcpy(l->ple_state, conv_in + (size_t)n * hc_dim, (size_t)state_len * hc_dim * sizeof(float));
        float *conv_out = conv_in + (size_t)(state_len + n) * hc_dim;
        kernel_conv1d_depthwise(conv_in, (size_t)(state_len + n), (size_t)hc_dim,
                                l->ple_conv1d->bf16,
                                (size_t)c->ple_conv_kernel_size, (size_t)c->ngram_size,
                                conv_out);
        // conv_out[t] = input[t - (k-1)*d]; want outputs for the last n input rows
        memmove(conv_out, conv_out + (size_t)(state_len) * hc_dim, (size_t)n * hc_dim * sizeof(float));
        kernel_silu(conv_out, (size_t)n * hc_dim, conv_out);
        
        float *ple_delta = xmalloc((size_t)n * hc_dim * sizeof(float));
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < hc_dim; i++) {
                float d = gated[(size_t)t * hc_dim + i] + conv_out[(size_t)t * hc_dim + i];
                ple_delta[(size_t)t * hc_dim + i] = d;
                h[(size_t)t * hc_dim + i] += d;
            }
            (void)0;
        }
        
        free(ple_delta);
    }
    free(hist);
    free(gids);
    return 0;
}

// ---------------------------------------------------------------------------
// forward

int model_forward(qmodel *m, const int64_t *tokens, size_t n, int decode,
                  float *logits, err_t *err) {
    qcfg *c = &m->cfg;
    if (n < 1) {
        err_set(err, "model_forward: empty input");
        return -1;
    }
    if (decode && n != 1) {
        err_set(err, "model_forward: decode expects one token");
        return -1;
    }
    int d = c->hidden_size;
    int hc = c->hc_count;
    int hc_dim = hc * d;
    double t0 = now_s();
    // scratch sizing: attention path needs the most
    size_t max_fn_scratch = 0; // per-layer function scratch (attention/deltanet/ple/gr)
    {
        // attention
        size_t need = (size_t)n * (c->indexer_n_heads * c->indexer_head_dim + c->indexer_head_dim) +
                      (size_t)n * c->num_attn_heads * c->head_dim * 2 +
                      (size_t)n * c->num_kv_heads * c->head_dim * 2 +
                      (size_t)n * c->num_attn_heads * QMODEL_KV_CAP +
                      (size_t)n * c->num_attn_heads * c->head_dim +
                      (size_t)n * c->num_kv_heads * c->head_dim +
                      (size_t)n * 2 * c->indexer_head_dim;
        if (need > max_fn_scratch) max_fn_scratch = need;
        // deltanet
        int conv_dim = 2 * c->linear_num_key_heads * c->linear_key_head_dim +
                       c->linear_num_value_heads * c->linear_value_head_dim;
        need = (size_t)n * (3 * conv_dim + c->linear_num_value_heads * c->linear_value_head_dim +
                            2 * c->linear_num_value_heads +
                            c->linear_num_value_heads * c->linear_value_head_dim * c->linear_key_head_dim +
                            c->linear_num_value_heads) +
               (size_t)(c->linear_conv_kernel_dim - 1) * conv_dim +
               (size_t)n * d;
        if (need > max_fn_scratch) max_fn_scratch = need;
        // ple
        int state_len = (c->ple_conv_kernel_size - 1) * c->ngram_size;
        need = (size_t)n * c->ple_embed_dim + (size_t)n * hc_dim * 3 + (size_t)n * d +
               (size_t)(state_len + 3 * n) * hc_dim;
        if (need > max_fn_scratch) max_fn_scratch = need;
        // gated residual: normed + w + tmp + inject
        need = (size_t)n * hc_dim + (size_t)n * c->hc_lowrank + (size_t)n * hc_dim +
               (size_t)n * hc + (size_t)n * d;
        if (need > max_fn_scratch) max_fn_scratch = need;
        // moe: unique/idx/tok_ids/weights + router logits + expert workspace
        need = (size_t)n * (size_t)(4 * c->num_experts_per_tok + c->num_experts +
                                    3 * d + 3 * c->moe_intermediate_size) + 2;
        if (need > max_fn_scratch) max_fn_scratch = need;
    }
    size_t scratch_bytes = (size_t)n * hc_dim * 2 + max_fn_scratch + (size_t)n * 1024;
    float *scratch = xmalloc(scratch_bytes * sizeof(float));
    int status = -1;
    // embed: gather + dequant
    float *h = scratch; // [n, hc_dim]
    float *h2 = h + (size_t)n * hc_dim;
    {
        // rows of embed for each token
        int64_t *rows = xmalloc(n * sizeof(int64_t));
        for (size_t i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= c->vocab_size) {
                err_set(err, "token %lld out of range", (long long)tokens[i]);
                free(rows);
                free(scratch);
                return -1;
            }
            rows[i] = tokens[i];
        }
        float *emb = xmalloc((size_t)n * d * sizeof(float));
        if (read_dequant_rows(m, m->embed, rows, n, emb, m->cfg.group_size, err) != 0) {
            free(rows);
            free(emb);
            free(scratch);
            return -1;
        }
        free(rows);
        // tile hc times
        for (size_t t = 0; t < n; t++) {
            for (int hh = 0; hh < hc; hh++) {
                memcpy(h + ((size_t)t * hc + hh) * d, emb + (size_t)t * d, (size_t)d * sizeof(float));
            }
        }
        free(emb);
    }
    // per layer
    
    float *fn_scratch = h2 + (size_t)n * hc_dim;
    for (int i = 0; i < c->n_layers; i++) {
        qlayer *l = &m->layers[i];
        int offset = l->kv_len;
        // ple first
        if (i == c->ple_layer) {
            int64_t prev[2] = {c->eos_token_id, c->eos_token_id};
            if (l->has_prev) {
                prev[0] = l->prev_ctx[0];
                prev[1] = l->prev_ctx[1];
            }
            if (ple_forward(m, l, h, tokens, (int)n, prev, fn_scratch, err) != 0) goto out;
            
            // update prev ctx
            if (!decode) {
                l->prev_ctx[0] = tokens[n - 2];
                l->prev_ctx[1] = tokens[n - 1];
            } else {
                l->prev_ctx[0] = l->prev_ctx[1];
                l->prev_ctx[1] = tokens[0];
            }
            l->has_prev = 1;
        }
        // attention / deltanet with hyper connections
        float *mixed = h2;               // [n, d]
        float *x = mixed + (size_t)n * d; // [n, d]
        float *inject = x + (size_t)n * d; // [n, hc]
        float *attn_out = inject + (size_t)n * hc; // [n, d]
        gated_residual(h, l->ah_norm, l->ah_down, l->ah_up, l->ah_inject, hc, d,
                       c->hc_lowrank, (int)n, mixed, inject, fn_scratch);
        
        if (c->layer_types[i] == 1) {
            attention_forward(m, l, mixed, (int)n, offset, attn_out, fn_scratch);
        } else {
            
            deltanet_forward(m, l, mixed, (int)n, attn_out, fn_scratch);
        }
        
        // h = hyper + attn_out * inject
        for (int t = 0; t < (int)n; t++) {
            for (int hh = 0; hh < hc; hh++) {
                float inj = inject[(size_t)t * hc + hh];
                for (int j = 0; j < d; j++) {
                    h[((size_t)t * hc + hh) * d + j] += inj * attn_out[(size_t)t * d + j];
                }
            }
        }
        // moe with hyper connections
        
        gated_residual(h, l->mh_norm, l->mh_down, l->mh_up, l->mh_inject, hc, d,
                       c->hc_lowrank, (int)n, mixed, inject, fn_scratch);
        if (moe_forward(m, l, mixed, (int)n, attn_out, fn_scratch, err) != 0) goto out;
        
        
        for (int t = 0; t < (int)n; t++) {
            for (int hh = 0; hh < hc; hh++) {
                float inj = inject[(size_t)t * hc + hh];
                for (int j = 0; j < d; j++) {
                    h[((size_t)t * hc + hh) * d + j] += inj * attn_out[(size_t)t * d + j];
                }
            }
        }
    }
    // final mixer (no inject)
    gated_residual(h, m->mix_norm, m->mix_down, m->mix_up, NULL, hc, d,
                   c->hc_lowrank, (int)n, h2, NULL, fn_scratch);
    // lm_head
    linear_forward(m->lm_head, (int)n, (int)c->vocab_size, d, h2, logits, 64);
    if (decode) m->stats.decode_calls++;
    else m->stats.prefill_calls++;
    status = 0;
out:
    free(scratch);
    m->stats.forward_seconds += now_s() - t0;
    return status;
}

// ---------------------------------------------------------------------------
// load

qmodel *model_load(const char *dir, int io_workers, uint64_t expert_budget,
                   uint64_t ple_budget, err_t *err) {
    qmodel *m = xcalloc(1, sizeof(qmodel));
    m->expert_budget = expert_budget;
    m->ple_budget = ple_budget;
    if (load_cfg(&m->cfg, dir, err) != 0) goto fail;
    qcfg *c = &m->cfg;
    if (st_index_open(&m->ix, dir, io_workers, err) != 0) goto fail;
    // ngram tensors
    {
        char nm[512];
        snprintf(nm, sizeof(nm), "model.layers.%d.ple.ple_embedding.layer_multipliers", c->ple_layer);
        if (mw_load_i64(&m->ng_mult, &m->ng_nheads, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.ple.ple_embedding.ngram_heads_offsets", c->ple_layer);
        if (mw_load_i64(&m->ng_offsets, &m->ng_nheads, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.ple.ple_embedding.ngram_heads_vocab_sizes", c->ple_layer);
        if (mw_load_i64(&m->ng_sizes, &m->ng_nheads, &m->ix, nm, err) != 0) goto fail;
        int64_t total = m->ng_offsets[m->ng_nheads - 1] + m->ng_sizes[m->ng_nheads - 1];
        (void)total;
        // the shard shape is the authority; total/split rounds down and
        // silently misroutes every gid after the first shard
        {
            char pname[512];
            snprintf(pname, sizeof(pname),
                     "model.layers.%d.ple.ple_embedding.ngram_embedding.shard_0.weight",
                     c->ple_layer);
            st_tensor *shard0 = st_find(&m->ix, pname);
            if (!shard0 || shard0->shape[0] < 1) {
                err_set(err, "missing PLE shard_0 tensor");
                goto fail;
            }
            m->ple_rows_per_shard = shard0->shape[0];
        }
        if (m->ple_rows_per_shard < 1) {
            err_set(err, "invalid PLE table geometry");
            goto fail;
        }
    }
    cache_init(&m->expert_cache, expert_budget, CACHE_POLICY_ADAPTIVE, c->num_layers);
    cache_init(&m->ple_cache, ple_budget, CACHE_POLICY_ADAPTIVE, 0);
    m->expert_cache.free_value = free;
    m->ple_cache.free_value = free;
    // dense weights
    {
        char nm[512];
        if (mw_load(&m->mw, &m->ix, "model.embed_tokens.weight", err) != 0) goto fail;
        if (mw_load(&m->mw, &m->ix, "lm_head.weight", err) != 0) goto fail;
        m->embed = mw_find(&m->mw, "model.embed_tokens.weight");
        m->lm_head = mw_find(&m->mw, "lm_head.weight");
        snprintf(nm, sizeof(nm), "model.hyper_connection_mixer.hc_norm.weight");
        if (mw_load_f32(&m->mix_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.hyper_connection_mixer.input_mix_weight_down.weight");
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.hyper_connection_mixer.input_mix_weight_up.weight");
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        m->mix_down = mw_find(&m->mw, "model.hyper_connection_mixer.input_mix_weight_down.weight");
        m->mix_up = mw_find(&m->mw, "model.hyper_connection_mixer.input_mix_weight_up.weight");
    }
    m->n_layers = c->n_layers;
    for (int i = 0; i < c->n_layers; i++) {
        qlayer *l = &m->layers[i];
        l->is_full = c->layer_types[i];
        l->is_ple = (i == c->ple_layer);
        char nm[512];
        if (l->is_full) {
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", i);
            if (mw_load_f32(&l->q_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", i);
            if (mw_load_f32(&l->k_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.indexer.index_qk_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.indexer.q_layernorm.weight", i);
            if (mw_load_f32(&l->idx_q_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.indexer.k_layernorm.weight", i);
            if (mw_load_f32(&l->idx_k_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            char qn[256];
            snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.q_proj.weight", i);
            l->q_proj = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.k_proj.weight", i);
            l->k_proj = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.v_proj.weight", i);
            l->v_proj = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.o_proj.weight", i);
            l->o_proj = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.self_attn.indexer.index_qk_proj.weight", i);
            l->idx_qk = mw_find(&m->mw, qn);
            l->k_cache = xcalloc(QMODEL_KV_CAP * c->num_kv_heads * c->head_dim, sizeof(float));
            l->v_cache = xcalloc(QMODEL_KV_CAP * c->num_kv_heads * c->head_dim, sizeof(float));
            l->idx_keys = xcalloc(QMODEL_KV_CAP * c->indexer_head_dim, sizeof(float));
        } else {
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.out_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.conv1d.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.norm.weight", i);
            if (mw_load_f32(&l->lin_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.A_log", i);
            if (mw_load_f32(&l->a_log, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.dt_bias", i);
            if (mw_load_f32(&l->dt_bias, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            char qn[256];
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            l->in_qkv = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            l->in_z = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            l->in_a = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            l->in_b = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.out_proj.weight", i);
            l->out_proj = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.linear_attn.conv1d.weight", i);
            l->conv1d = mw_find(&m->mw, qn);
            int conv_dim = 2 * c->linear_num_key_heads * c->linear_key_head_dim +
                           c->linear_num_value_heads * c->linear_value_head_dim;
            l->conv_state = xcalloc((size_t)(c->linear_conv_kernel_dim - 1) * conv_dim, sizeof(float));
            l->rec_state = xcalloc((size_t)c->linear_num_value_heads * c->linear_value_head_dim * c->linear_key_head_dim, sizeof(float));
        }
        // moe
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        char qn[256];
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp.gate.weight", i);
        l->gate = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        l->sh_g = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        l->sh_u = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        l->sh_d = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        l->sh_gate = mw_find(&m->mw, qn);
        // hyper connections
        snprintf(nm, sizeof(nm), "model.layers.%d.attn_hyper_connection.hc_norm.weight", i);
        if (mw_load_f32(&l->ah_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.attn_hyper_connection.input_mix_weight_down.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.attn_hyper_connection.input_mix_weight_up.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.attn_hyper_connection.block_inject_weight.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp_hyper_connection.hc_norm.weight", i);
        if (mw_load_f32(&l->mh_norm, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp_hyper_connection.block_inject_weight.weight", i);
        if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
        snprintf(qn, sizeof(qn), "model.layers.%d.attn_hyper_connection.input_mix_weight_down.weight", i);
        l->ah_down = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.attn_hyper_connection.input_mix_weight_up.weight", i);
        l->ah_up = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.attn_hyper_connection.block_inject_weight.weight", i);
        l->ah_inject = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight", i);
        l->mh_down = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight", i);
        l->mh_up = mw_find(&m->mw, qn);
        snprintf(qn, sizeof(qn), "model.layers.%d.mlp_hyper_connection.block_inject_weight.weight", i);
        l->mh_inject = mw_find(&m->mw, qn);
        // ple
        if (l->is_ple) {
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.key_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.value_proj.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.conv1d.weight", i);
            if (mw_load(&m->mw, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.norm_key.weight", i);
            if (mw_load_f32(&l->ple_norm_key, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.norm_query.weight", i);
            if (mw_load_f32(&l->ple_norm_query, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(nm, sizeof(nm), "model.layers.%d.ple.norm_conv.weight", i);
            if (mw_load_f32(&l->ple_norm_conv, &(size_t){0}, &m->ix, nm, err) != 0) goto fail;
            snprintf(qn, sizeof(qn), "model.layers.%d.ple.key_proj.weight", i);
            l->ple_key = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.ple.value_proj.weight", i);
            l->ple_val = mw_find(&m->mw, qn);
            snprintf(qn, sizeof(qn), "model.layers.%d.ple.conv1d.weight", i);
            l->ple_conv1d = mw_find(&m->mw, qn);
            int state_len = (c->ple_conv_kernel_size - 1) * c->ngram_size;
            l->ple_state = xcalloc((size_t)state_len * c->hc_count * c->hidden_size, sizeof(float));
        }
    }
    // cross-check ngram geometry against recomputation
    {
        ngram_geo g;
        memset(&g, 0, sizeof(g));
        g.ngram_size = c->ngram_size;
        g.heads_per_ngram = c->heads_per_ngram;
        g.ngram_heads = (c->ngram_size - 1) * c->heads_per_ngram;
        g.vocab_size = c->vocab_size;
        g.eos_token_id = c->eos_token_id;
        ngram_geo_compute(&g, c->ngram_vocab_size_base, c->seed, 0);
        for (size_t i = 0; i < m->ng_nheads; i++) {
            if (g.sizes[i] != m->ng_sizes[i] || g.offsets[i] != m->ng_offsets[i]) {
                        break;
            }
        }
        for (int i = 0; i < c->ngram_size; i++) {
            if (g.multipliers[i] != m->ng_mult[i]) {
                        break;
            }
        }
    }
    model_reset_caches(m);
    return m;
fail:
    model_free(m);
    return NULL;
}
