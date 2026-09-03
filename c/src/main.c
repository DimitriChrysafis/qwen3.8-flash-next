#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "tokenizer.h"
#include "util.h"

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *command;
    int max_tokens;
    int runs;
    double temperature;
    double top_p;
    uint64_t seed;
    int chat;
    int io_workers;
    double expert_budget_gib;
    double ple_budget_gib;
    int expert_prefetch;
    int ple_prefetch;
} cli_opts;

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s --model DIR <generate|benchmark> [options]\n"
            "\n"
            "options:\n"
            "  --model DIR          model directory (config.json + safetensors)\n"
            "  --prompt TEXT        prompt (generate) or token ids as text (benchmark)\n"
            "  --max-tokens N       tokens to generate (default 64)\n"
            "  --temperature T      sampling temperature (default 0)\n"
            "  --top-p P            nucleus sampling (default 1)\n"
            "  --seed N             sampling seed (default 0)\n"
            "  --runs N             benchmark runs (default 1)\n"
            "  --expert-budget-gib  expert cache budget (default 8)\n"
            "  --ple-budget-gib     ple cache budget (default 1)\n"
            "  --io-workers N       disk read threads (default 6)\n"
            "  --expert-prefetch N  experts to prefetch (default 2)\n"
            "  --ple-prefetch N     ple rows to prefetch (default 8)\n",
            prog);
}

static int parse_cli(int argc, char **argv, cli_opts *o) {
    memset(o, 0, sizeof(*o));
    o->max_tokens = 64;
    o->temperature = 0.0;
    o->top_p = 1.0;
    o->runs = 1;
    o->io_workers = 6;
    o->expert_budget_gib = 8.0;
    o->ple_budget_gib = 1.0;
    o->expert_prefetch = 2;
    o->ple_prefetch = 8;
    static const struct option longopts[] = {
        {"model", required_argument, NULL, 'm'},
        {"prompt", required_argument, NULL, 'p'},
        {"max-tokens", required_argument, NULL, 'n'},
        {"temperature", required_argument, NULL, 't'},
        {"top-p", required_argument, NULL, 'P'},
        {"seed", required_argument, NULL, 's'},
        {"runs", required_argument, NULL, 'r'},
        {"expert-budget-gib", required_argument, NULL, 'e'},
        {"ple-budget-gib", required_argument, NULL, 'l'},
        {"io-workers", required_argument, NULL, 'w'},
        {"expert-prefetch", required_argument, NULL, 'x'},
        {"ple-prefetch", required_argument, NULL, 'y'},
        {"chat", no_argument, NULL, 'c'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int c;
    while ((c = getopt_long(argc, argv, "m:p:n:t:P:s:r:e:l:w:x:y:ch", longopts, NULL)) != -1) {
        switch (c) {
            case 'm': o->model_dir = optarg; break;
            case 'p': o->prompt = optarg; break;
            case 'n': o->max_tokens = atoi(optarg); break;
            case 't': o->temperature = atof(optarg); break;
            case 'P': o->top_p = atof(optarg); break;
            case 's': o->seed = strtoull(optarg, NULL, 10); break;
            case 'r': o->runs = atoi(optarg); break;
            case 'e': o->expert_budget_gib = atof(optarg); break;
            case 'l': o->ple_budget_gib = atof(optarg); break;
            case 'w': o->io_workers = atoi(optarg); break;
            case 'x': o->expert_prefetch = atoi(optarg); break;
            case 'y': o->ple_prefetch = atoi(optarg); break;
            case 'c': o->chat = 1; break;
            case 'h': usage(argv[0]); return -1;
            default: usage(argv[0]); return -1;
        }
    }
    if (optind < argc) o->command = argv[optind];
    if (!o->model_dir || !o->command ||
        (strcmp(o->command, "generate") != 0 && strcmp(o->command, "benchmark") != 0)) {
        usage(argv[0]);
        return -1;
    }
    if (strcmp(o->command, "generate") == 0 && !o->prompt) {
        fprintf(stderr, "generate requires --prompt\n");
        return -1;
    }
    if (strcmp(o->command, "benchmark") == 0 && !o->prompt) {
        fprintf(stderr, "benchmark requires --prompt (comma-separated token ids)\n");
        return -1;
    }
    return 0;
}

// xorshift64* rng
static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

// sample a token from logits (length vocab). returns token id.
static int64_t sample_token(const float *logits, int64_t vocab, double temperature,
                            double top_p, uint64_t *rng) {
    if (temperature <= 0.0) {
        int64_t best = 0;
        for (int64_t i = 1; i < vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }
    // softmax with temperature
    double *probs = xmalloc((size_t)vocab * sizeof(double));
    double max = -1e300;
    for (int64_t i = 0; i < vocab; i++) {
        if (logits[i] > max) max = logits[i];
    }
    double sum = 0;
    for (int64_t i = 0; i < vocab; i++) {
        probs[i] = exp((logits[i] - max) / temperature);
        sum += probs[i];
    }
    for (int64_t i = 0; i < vocab; i++) probs[i] /= sum;
    if (top_p < 1.0) {
        // nucleus: sort descending and keep the smallest set with mass >= top_p
        int64_t *order = xmalloc((size_t)vocab * sizeof(int64_t));
        for (int64_t i = 0; i < vocab; i++) order[i] = i;
        // insertion sort is fine for the top region
        for (int64_t i = 1; i < vocab; i++) {
            int64_t key = order[i];
            int64_t j = i - 1;
            while (j >= 0 && probs[order[j]] < probs[key]) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
        double acc = 0;
        int64_t keep = 0;
        while (keep < vocab && acc + probs[order[keep]] <= top_p) {
            acc += probs[order[keep]];
            keep++;
        }
        if (keep == 0) keep = 1;
        // renormalize and sample within the nucleus
        double nsum = 0;
        for (int64_t i = 0; i < keep; i++) nsum += probs[order[i]];
        double r = (double)(rng_next(rng) >> 11) / (double)(1ull << 53) * nsum;
        double cum = 0;
        int64_t chosen = order[keep - 1];
        for (int64_t i = 0; i < keep; i++) {
            cum += probs[order[i]];
            if (r <= cum) {
                chosen = order[i];
                break;
            }
        }
        free(order);
        free(probs);
        return chosen;
    }
    double r = (double)(rng_next(rng) >> 11) / (double)(1ull << 53) * sum;
    double cum = 0;
    int64_t chosen = 0;
    for (int64_t i = 0; i < vocab; i++) {
        cum += probs[i];
        if (r <= cum) {
            chosen = i;
            break;
        }
    }
    free(probs);
    return chosen;
}

// parse a comma-separated token id list
static int parse_ids(const char *text, int64_t *ids, int cap) {
    int n = 0;
    const char *p = text;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '[' || *p == ']') p++;
        if (!*p) break;
        char *end;
        long long v = strtoll(p, &end, 10);
        if (end == p) return -1;
        if (n >= cap) return -1;
        ids[n++] = v;
        p = end;
    }
    return n;
}

static void print_stats(qmodel *m, double load_seconds, double prefill_seconds,
                        double gen_seconds, int prompt_tokens, int gen_tokens,
                        int runs) {
    const qstats *st = model_stats(m);
    double decode_tokens = gen_tokens - runs > 0 ? gen_tokens - runs : 0;
    printf("{\n");
    printf("  \"prompt_tokens\": %d,\n", prompt_tokens);
    printf("  \"generated_tokens\": %d,\n", gen_tokens);
    printf("  \"load_seconds\": %.3f,\n", load_seconds);
    printf("  \"prefill_seconds\": %.3f,\n", prefill_seconds);
    printf("  \"prefill_tokens_per_second\": %.2f,\n",
           prefill_seconds > 0 ? prompt_tokens / prefill_seconds : 0.0);
    printf("  \"generation_seconds\": %.3f,\n", gen_seconds);
    printf("  \"generation_tokens_per_second\": %.2f,\n",
           gen_seconds > 0 ? decode_tokens / gen_seconds : 0.0);
    printf("  \"expert_loads\": %llu,\n", (unsigned long long)st->expert_loads);
    printf("  \"ple_loads\": %llu,\n", (unsigned long long)st->ple_loads);
    printf("  \"expert_bytes\": %llu,\n", (unsigned long long)st->expert_bytes);
    printf("  \"ple_bytes\": %llu,\n", (unsigned long long)st->ple_bytes);
    printf("  \"expert_wait_seconds\": %.3f,\n", st->expert_wait_seconds);
    printf("  \"ple_wait_seconds\": %.3f,\n", st->ple_wait_seconds);
    printf("  \"forward_seconds\": %.3f\n", st->forward_seconds);
    printf("}\n");
}

int main(int argc, char **argv) {
    cli_opts o;
    if (parse_cli(argc, argv, &o) != 0) return 1;
    err_t err = {0};
    double t0 = now_s();
    qmodel *m = model_load(o.model_dir, o.io_workers,
                           (uint64_t)(o.expert_budget_gib * (1ull << 30)),
                           (uint64_t)(o.ple_budget_gib * (1ull << 30)), &err);
    if (!m) {
        fprintf(stderr, "model_load failed: %s\n", err.msg);
        return 1;
    }
    double load_seconds = now_s() - t0;
    int64_t vocab = model_vocab(m);
    int64_t eos = model_eos(m);

    // tokenize the prompt (raw byte ids when no tokenizer is available)
    int64_t ids[4096];
    int n_ids = 0;
    tokenizer *tok = tokenizer_load(o.model_dir);
    if (strcmp(o.command, "generate") == 0) {
        if (tok) {
            n_ids = tokenizer_encode(tok, o.prompt, ids, 4096);
            if (n_ids < 0) {
                fprintf(stderr, "tokenizer failed\n");
                tokenizer_free(tok);
                model_free(m);
                return 1;
            }
            model_set_tokenizer(m, tok);
        } else {
            fprintf(stderr, "warning: no tokenizer available, using raw byte ids\n");
            n_ids = (int)strlen(o.prompt);
            for (int i = 0; i < n_ids && i < 4096; i++) ids[i] = (unsigned char)o.prompt[i];
        }
    } else {
        n_ids = parse_ids(o.prompt, ids, 4096);
        if (n_ids < 0) {
            fprintf(stderr, "cannot parse prompt as token ids\n");
            model_free(m);
            return 1;
        }
    }
    if (n_ids < 1) {
        fprintf(stderr, "empty prompt\n");
        model_free(m);
        return 1;
    }

    float *logits = xmalloc((size_t)vocab * sizeof(float));
    uint64_t rng = o.seed;
    double prefill_seconds = 0, gen_seconds = 0;
    int gen_tokens = 0;
    int64_t *generated = xmalloc((size_t)(o.max_tokens + 1) * sizeof(int64_t));
    int runs = o.runs > 0 ? o.runs : 1;

    for (int r = 0; r < runs; r++) {
        model_reset_caches(m);
        double t1 = now_s();
        if (model_forward(m, ids, (size_t)n_ids, 0, logits, &err) != 0) {
            fprintf(stderr, "prefill failed: %s\n", err.msg);
            return 1;
        }
        prefill_seconds += now_s() - t1;

        double t2 = now_s();
        float *dlogits = xmalloc((size_t)vocab * sizeof(float));
        for (int step = 0; step < o.max_tokens; step++) {
            int64_t token;
            if (step == 0) {
                token = sample_token(logits + (int64_t)(n_ids - 1) * vocab, vocab,
                                     o.temperature, o.top_p, &rng);
            } else {
                token = sample_token(dlogits, vocab, o.temperature, o.top_p, &rng);
            }
            generated[step] = token;
            gen_tokens++;
            if (token == eos) break;
            if (step + 1 < o.max_tokens) {
                if (model_forward(m, &token, 1, 1, dlogits, &err) != 0) {
                    fprintf(stderr, "decode failed: %s\n", err.msg);
                    return 1;
                }
            }
        }
        free(dlogits);
        gen_seconds += now_s() - t2;
    }

    if (strcmp(o.command, "generate") == 0) {
        if (tok) {
            char *text = tokenizer_decode(tok, generated, (size_t)gen_tokens);
            if (text) {
                printf("%s\n", text);
                free(text);
            }
        } else {
            printf("generated tokens: ");
            for (int i = 0; i < gen_tokens; i++) {
                if (i) printf(" ");
                printf("%lld", (long long)generated[i]);
            }
            printf("\n");
        }
    }
    print_stats(m, load_seconds, prefill_seconds, gen_seconds, n_ids * runs,
                gen_tokens, runs);

    free(logits);
    free(generated);
    tokenizer_free(tok);
    model_free(m);
    return 0;
}
