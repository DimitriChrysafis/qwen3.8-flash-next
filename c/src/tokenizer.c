// byte-level BPE tokenizer for qwen tokenizer.json files.
// parses vocab, merges and added_tokens directly and encodes with the classic
// lowest-rank pair merge after qwen3 pre-tokenization.

#include "tokenizer.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>

#include "util.h"

// ---------------------------------------------------------------------------
// hash maps

static uint64_t fnv1a(const void *data, size_t len) {
    const uint8_t *p = data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

typedef struct {
    uint8_t *bytes;
    size_t len;
    int32_t id;
    int used;
} vslot;

typedef struct {
    vslot *slots;
    size_t nslots;
    size_t count;
} vocab_map;

static size_t next_pow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static void vmap_init(vocab_map *m, size_t cap) {
    m->nslots = next_pow2(cap ? cap : 1);
    m->slots = xcalloc(m->nslots, sizeof(vslot));
    m->count = 0;
}

static void vmap_grow(vocab_map *m) {
    size_t ns = m->nslots ? m->nslots * 2 : 1024;
    vslot *s = xcalloc(ns, sizeof(vslot));
    for (size_t i = 0; i < m->nslots; i++) {
        if (!m->slots[i].used) continue;
        size_t j = (size_t)fnv1a(m->slots[i].bytes, m->slots[i].len) & (ns - 1);
        while (s[j].used) j = (j + 1) & (ns - 1);
        s[j] = m->slots[i];
    }
    free(m->slots);
    m->slots = s;
    m->nslots = ns;
}

static int32_t vmap_get(const vocab_map *m, const uint8_t *bytes, size_t len) {
    size_t i = (size_t)fnv1a(bytes, len) & (m->nslots - 1);
    while (m->slots[i].used) {
        if (m->slots[i].len == len && memcmp(m->slots[i].bytes, bytes, len) == 0) {
            return m->slots[i].id;
        }
        i = (i + 1) & (m->nslots - 1);
    }
    return -1;
}

static void vmap_put(vocab_map *m, const uint8_t *bytes, size_t len, int32_t id) {
    if (m->nslots == 0) vmap_init(m, 1024);
    if (m->count * 4 >= m->nslots * 3) vmap_grow(m);
    size_t i = (size_t)fnv1a(bytes, len) & (m->nslots - 1);
    while (m->slots[i].used) {
        if (m->slots[i].len == len && memcmp(m->slots[i].bytes, bytes, len) == 0) {
            m->slots[i].id = id;
            return;
        }
        i = (i + 1) & (m->nslots - 1);
    }
    m->slots[i].used = 1;
    m->slots[i].bytes = xmalloc(len);
    memcpy(m->slots[i].bytes, bytes, len);
    m->slots[i].len = len;
    m->slots[i].id = id;
    m->count++;
}

static void vmap_free(vocab_map *m) {
    for (size_t i = 0; i < m->nslots; i++) {
        if (m->slots[i].used) free(m->slots[i].bytes);
    }
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

typedef struct {
    uint64_t key;
    int32_t rank;
    int32_t merged;
    int used;
} mslot;

typedef struct {
    mslot *slots;
    size_t nslots;
    size_t count;
} merges_map;

static void mmap_init(merges_map *m, size_t cap) {
    m->nslots = next_pow2(cap ? cap : 1);
    m->slots = xcalloc(m->nslots, sizeof(mslot));
    m->count = 0;
}

static void mmap_grow(merges_map *m) {
    size_t ns = m->nslots ? m->nslots * 2 : 1024;
    mslot *s = xcalloc(ns, sizeof(mslot));
    for (size_t i = 0; i < m->nslots; i++) {
        if (!m->slots[i].used) continue;
        size_t j = (size_t)fnv1a(&m->slots[i].key, 8) & (ns - 1);
        while (s[j].used) j = (j + 1) & (ns - 1);
        s[j] = m->slots[i];
    }
    free(m->slots);
    m->slots = s;
    m->nslots = ns;
}

static mslot *mmap_find(merges_map *m, uint32_t a, uint32_t b) {
    uint64_t key = ((uint64_t)a << 32) | b;
    size_t i = (size_t)fnv1a(&key, 8) & (m->nslots - 1);
    while (m->slots[i].used) {
        if (m->slots[i].key == key) return &m->slots[i];
        i = (i + 1) & (m->nslots - 1);
    }
    return NULL;
}

static void mmap_put(merges_map *m, uint32_t a, uint32_t b, int32_t rank,
                     int32_t merged) {
    if (m->nslots == 0) mmap_init(m, 1024);
    if (m->count * 4 >= m->nslots * 3) mmap_grow(m);
    uint64_t key = ((uint64_t)a << 32) | b;
    size_t i = (size_t)fnv1a(&key, 8) & (m->nslots - 1);
    while (m->slots[i].used) {
        if (m->slots[i].key == key) {
            m->slots[i].rank = rank;
            m->slots[i].merged = merged;
            return;
        }
        i = (i + 1) & (m->nslots - 1);
    }
    m->slots[i].used = 1;
    m->slots[i].key = key;
    m->slots[i].rank = rank;
    m->slots[i].merged = merged;
    m->count++;
}

static void mmap_free(merges_map *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

// ---------------------------------------------------------------------------
// minimal json scanning

typedef struct {
    const char *p;
    const char *end;
    int failed;
} js;

static void js_skip_ws(js *j) {
    while (j->p < j->end && isspace((unsigned char)*j->p)) j->p++;
}

static int js_peek(js *j, char c) {
    js_skip_ws(j);
    return j->p < j->end && *j->p == c;
}

static void js_expect(js *j, char c) {
    js_skip_ws(j);
    if (j->p >= j->end || *j->p != c) {
        j->failed = 1;
        return;
    }
    j->p++;
}

// read a json string, escapes resolved, into out. returns length, -1 on error.
static long js_string_raw(js *j, uint8_t *out, size_t cap) {
    js_skip_ws(j);
    if (j->p >= j->end || *j->p != '"') {
        j->failed = 1;
        return -1;
    }
    j->p++;
    size_t n = 0;
    while (j->p < j->end) {
        unsigned char c = (unsigned char)*j->p;
        if (c == '"') {
            j->p++;
            return (long)n;
        }
        if (c == '\\') {
            j->p++;
            if (j->p >= j->end) break;
            char e = *j->p++;
            if (e == 'u') {
                if (j->end - j->p < 4) break;
                uint32_t cp = 0;
                int ok = 1;
                for (int i = 0; i < 4; i++) {
                    char h = j->p[i];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (uint32_t)(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (uint32_t)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (uint32_t)(h - 'A' + 10);
                    else {
                        ok = 0;
                        break;
                    }
                }
                if (!ok) {
                    j->failed = 1;
                    return -1;
                }
                j->p += 4;
                // surrogate pair
                if (cp >= 0xD800 && cp <= 0xDBFF && j->end - j->p >= 6 &&
                    j->p[0] == '\\' && j->p[1] == 'u') {
                    uint32_t lo = 0;
                    int lok = 1;
                    for (int i = 0; i < 4; i++) {
                        char h = j->p[2 + i];
                        lo <<= 4;
                        if (h >= '0' && h <= '9') lo |= (uint32_t)(h - '0');
                        else if (h >= 'a' && h <= 'f') lo |= (uint32_t)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') lo |= (uint32_t)(h - 'A' + 10);
                        else {
                            lok = 0;
                            break;
                        }
                    }
                    if (lok && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        j->p += 6;
                    }
                }
                if (cp < 0x80) {
                    if (n + 1 > cap) return -1;
                    out[n++] = (uint8_t)cp;
                } else if (cp < 0x800) {
                    if (n + 2 > cap) return -1;
                    out[n++] = (uint8_t)(0xC0 | (cp >> 6));
                    out[n++] = (uint8_t)(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    if (n + 3 > cap) return -1;
                    out[n++] = (uint8_t)(0xE0 | (cp >> 12));
                    out[n++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (uint8_t)(0x80 | (cp & 0x3F));
                } else {
                    if (n + 4 > cap) return -1;
                    out[n++] = (uint8_t)(0xF0 | (cp >> 18));
                    out[n++] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
                    out[n++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (uint8_t)(0x80 | (cp & 0x3F));
                }
                continue;
            }
            switch (e) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default:
                    j->failed = 1;
                    return -1;
            }
            if (n + 1 > cap) return -1;
            out[n++] = c;
            continue;
        }
        if (n + 1 > cap) return -1;
        out[n++] = c;
        j->p++;
    }
    j->failed = 1;
    return -1;
}

static long js_int(js *j) {
    js_skip_ws(j);
    char *end;
    long v = strtol(j->p, &end, 10);
    if (end == j->p) {
        j->failed = 1;
        return 0;
    }
    j->p = end;
    return v;
}

// skip one json value (string, number, literal or nested container)
static int js_skip_value(js *j) {
    js_skip_ws(j);
    if (j->p >= j->end) {
        j->failed = 1;
        return 0;
    }
    if (*j->p == '"') {
        j->p++;
        while (j->p < j->end && *j->p != '"') {
            if (*j->p == '\\' && j->p + 1 < j->end) j->p++;
            j->p++;
        }
        if (j->p >= j->end) {
            j->failed = 1;
            return 0;
        }
        j->p++;
        return 1;
    }
    if (*j->p == '{' || *j->p == '[') {
        int depth = 0;
        while (j->p < j->end) {
            char c = *j->p;
            if (c == '"') {
                j->p++;
                while (j->p < j->end && *j->p != '"') {
                    if (*j->p == '\\' && j->p + 1 < j->end) j->p++;
                    j->p++;
                }
                if (j->p >= j->end) break;
            } else if (c == '{' || c == '[') {
                depth++;
            } else if (c == '}' || c == ']') {
                if (--depth == 0) {
                    j->p++;
                    return 1;
                }
            }
            j->p++;
        }
        j->failed = 1;
        return 0;
    }
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']' &&
           !isspace((unsigned char)*j->p)) {
        j->p++;
    }
    return 1;
}

// scan forward for "key": and leave the cursor on the value. returns 0 when
// not found (cursor state is then undefined).
static int find_key(js *j, const char *key, size_t keylen) {
    while (j->p < j->end) {
        js_skip_ws(j);
        if (j->end - j->p >= (long)keylen + 2 && j->p[0] == '"' &&
            memcmp(j->p + 1, key, keylen) == 0 && j->p[keylen + 1] == '"') {
            j->p += keylen + 2;
            js_expect(j, ':');
            return !j->failed;
        }
        j->p++;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// gpt-2 byte-to-unicode table

// byte -> utf-8 form used in the vocab keys; printable bytes map to
// themselves, the rest to u+0100 + n in byte order
static void byte_to_unicode(uint8_t form[256][3], uint8_t flen[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp;
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            cp = b;
        } else {
            cp = 256 + n;
            n++;
        }
        if (cp < 0x80) {
            form[b][0] = (uint8_t)cp;
            flen[b] = 1;
        } else {
            form[b][0] = (uint8_t)(0xC0 | (cp >> 6));
            form[b][1] = (uint8_t)(0x80 | (cp & 0x3F));
            flen[b] = 2;
        }
    }
}

// ---------------------------------------------------------------------------
// pre-tokenization (qwen3 split regex over icu character classes)

typedef struct {
    int letter, mark, digit, space;
} cclass;

static void cls_at(const uint8_t *s, size_t len, size_t p, cclass *c) {
    // decode one codepoint at p
    uint8_t b = s[p];
    uint32_t cp = b;
    if (b >= 0xC0 && b < 0xF8 && p + 1 < len && (s[p + 1] & 0xC0) == 0x80) {
        size_t n = b >= 0xF0 ? 3 : b >= 0xE0 ? 2 : 1;
        cp = b & (0xFF >> (n + 1));
        for (size_t k = 1; k <= n && p + k < len; k++) {
            cp = (cp << 6) | (uint32_t)(s[p + k] & 0x3F);
        }
    }
    int t = u_charType(cp);
    c->letter = u_isalpha(cp);
    c->mark = (t == U_NON_SPACING_MARK || t == U_COMBINING_SPACING_MARK ||
               t == U_ENCLOSING_MARK);
    c->digit = (t == U_DECIMAL_DIGIT_NUMBER || t == U_LETTER_NUMBER ||
                t == U_OTHER_NUMBER);
    c->space = u_isspace(cp);
}

static size_t next_cp(const uint8_t *s, size_t len, size_t p) {
    uint8_t b = s[p];
    size_t n = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    p += n;
    if (p > len) p = len;
    return p;
}

static size_t letter_run(const uint8_t *s, size_t len, size_t p) {
    while (p < len) {
        cclass c;
        cls_at(s, len, p, &c);
        if (!c.letter && !c.mark) break;
        p = next_cp(s, len, p);
    }
    return p;
}

static size_t punct_run(const uint8_t *s, size_t len, size_t p) {
    while (p < len) {
        cclass c;
        cls_at(s, len, p, &c);
        if (c.space || c.letter || c.mark || c.digit) break;
        p = next_cp(s, len, p);
    }
    return p;
}

static size_t space_run(const uint8_t *s, size_t len, size_t p) {
    while (p < len) {
        cclass c;
        cls_at(s, len, p, &c);
        if (!c.space) break;
        p = next_cp(s, len, p);
    }
    return p;
}

// end of the pre-token starting at i, or 0 when no alternative matches
static size_t pretoken_end(const uint8_t *s, size_t len, size_t i) {
    cclass c0;
    cls_at(s, len, i, &c0);

    // 's | 't | 're | 've | 'm | 'll | 'd (case-insensitive)
    if (s[i] == '\'' && i + 1 < len) {
        char a = (char)tolower(s[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < len) {
            char b = (char)tolower(s[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                (a == 'l' && b == 'l')) {
                return i + 3;
            }
        }
    }

    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (c0.letter || c0.mark) return letter_run(s, len, i);
    if (s[i] != '\r' && s[i] != '\n' && !c0.letter && !c0.digit) {
        size_t next = next_cp(s, len, i);
        if (next < len) {
            cclass c1;
            cls_at(s, len, next, &c1);
            if (c1.letter || c1.mark) return letter_run(s, len, next);
        }
    }

    // \p{N} (single character)
    if (c0.digit) return next_cp(s, len, i);

    // ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    {
        size_t p = i;
        if (s[p] == ' ') p++;
        size_t q = punct_run(s, len, p);
        if (q > p) {
            while (q < len && (s[q] == '\r' || s[q] == '\n')) q++;
            return q;
        }
    }

    // \s*[\r\n]+ : whitespace up through the last newline of the run
    if (c0.space) {
        size_t q = i, last_nl_end = 0;
        while (q < len) {
            cclass c;
            cls_at(s, len, q, &c);
            if (!c.space) break;
            if (s[q] == '\r' || s[q] == '\n') {
                size_t r = q;
                while (r < len && (s[r] == '\r' || s[r] == '\n')) r++;
                last_nl_end = r;
                q = r;
            } else {
                q = next_cp(s, len, q);
            }
        }
        if (last_nl_end) return last_nl_end;

        // \s+(?!\S) : whole run at end of text, else all but last char
        size_t e = space_run(s, len, i);
        if (e >= len) return e;
        if (e - i > 1) {
            size_t k = e - 1;
            while (k > i && (s[k] & 0xC0) == 0x80) k--;
            return k;
        }
        return 0;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// tokenizer

#define MAX_TOK_BYTES 512

struct tokenizer {
    vocab_map vocab;
    merges_map merges;
    vocab_map uni; // unicode form -> raw byte (for decode)
    uint8_t **id_bytes; // id -> vocab-key bytes (unicode form or added token)
    size_t *id_lens;
    size_t nvocab;
    uint8_t byte_form[256][3];
    uint8_t byte_form_len[256];
    int32_t eos_id;
};

typedef struct {
    int32_t *ids;
    uint8_t (*bytes)[MAX_TOK_BYTES];
    size_t *lens;
    size_t n, cap;
    int32_t max_id;
} added_tokens;

static void added_push(added_tokens *a, int32_t id, const uint8_t *bytes,
                       size_t len) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->ids = xrealloc(a->ids, a->cap * sizeof(int32_t));
        a->bytes = xrealloc(a->bytes, a->cap * sizeof(*a->bytes));
        a->lens = xrealloc(a->lens, a->cap * sizeof(size_t));
    }
    a->ids[a->n] = id;
    memcpy(a->bytes[a->n], bytes, len);
    a->lens[a->n] = len;
    a->n++;
    if (id > a->max_id) a->max_id = id;
}

// parse "added_tokens": [ { "id": N, "content": "...", ... }, ... ]
static int parse_added_tokens(js *j, added_tokens *a) {
    js_expect(j, '[');
    if (j->failed) return 0;
    while (!js_peek(j, ']')) {
        js_expect(j, '{');
        if (j->failed) return 0;
        long id = -1;
        uint8_t content[MAX_TOK_BYTES];
        long cl = -1;
        while (!js_peek(j, '}')) {
            uint8_t key[64];
            long kl = js_string_raw(j, key, sizeof(key));
            if (kl < 0) return 0;
            js_expect(j, ':');
            if (j->failed) return 0;
            if (kl == 2 && memcmp(key, "id", 2) == 0) {
                id = js_int(j);
            } else if (kl == 7 && memcmp(key, "content", 7) == 0) {
                cl = js_string_raw(j, content, sizeof(content));
            } else if (!js_skip_value(j)) {
                return 0;
            }
            if (j->failed) return 0;
            if (!js_peek(j, ',')) break;
            j->p++;
        }
        js_expect(j, '}');
        if (j->failed) return 0;
        if (id >= 0 && cl >= 0) added_push(a, (int32_t)id, content, (size_t)cl);
        if (!js_peek(j, ',')) break;
        j->p++;
    }
    js_expect(j, ']');
    return !j->failed;
}

tokenizer *tokenizer_load(const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 16 || len > (1l << 31)) {
        fclose(f);
        return NULL;
    }
    char *buf = xmalloc((size_t)len + 1);
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = 0;
    if (got != (size_t)len) {
        free(buf);
        return NULL;
    }

    tokenizer *t = xcalloc(1, sizeof(tokenizer));
    t->eos_id = -1;
    byte_to_unicode(t->byte_form, t->byte_form_len);
    for (int b = 0; b < 256; b++) {
        vmap_put(&t->uni, t->byte_form[b], t->byte_form_len[b], b);
    }

    js j = {buf, buf + len, 0};

    // added tokens (specials like <|endoftext|>; ids above the base vocab)
    added_tokens added = {0};
    if (find_key(&j, "added_tokens", 12)) {
        parse_added_tokens(&j, &added);
    }
    if (j.failed) goto fail;

    // vocab
    if (!find_key(&j, "vocab", 5)) goto fail;
    js_expect(&j, '{');
    if (j.failed) goto fail;
    {
        // count entries first to size the map
        js c = j;
        size_t n = 0;
        while (!c.failed && !js_peek(&c, '}')) {
            uint8_t key[MAX_TOK_BYTES];
            if (js_string_raw(&c, key, sizeof(key)) < 0) break;
            js_expect(&c, ':');
            js_int(&c);
            if (!js_peek(&c, ',')) break;
            c.p++;
            n++;
        }
        vmap_init(&t->vocab, n * 2 + 1024);
        int32_t max_id = -1;
        while (!js_peek(&j, '}')) {
            uint8_t key[MAX_TOK_BYTES];
            long kl = js_string_raw(&j, key, sizeof(key));
            if (kl < 0) goto fail;
            js_expect(&j, ':');
            long id = js_int(&j);
            if (j.failed) goto fail;
            if (id >= 0) {
                vmap_put(&t->vocab, key, (size_t)kl, (int32_t)id);
                if (id > max_id) max_id = (int32_t)id;
            }
            if (!js_peek(&j, ',')) break;
            j.p++;
        }
        js_expect(&j, '}');
        if (j.failed) goto fail;

        // id -> bytes table (vocab + added tokens)
        t->nvocab = (size_t)(added.max_id > max_id ? added.max_id : max_id) + 1;
        t->id_bytes = xcalloc(t->nvocab, sizeof(uint8_t *));
        t->id_lens = xcalloc(t->nvocab, sizeof(size_t));
        for (size_t i = 0; i < t->vocab.nslots; i++) {
            vslot *s = &t->vocab.slots[i];
            if (!s->used || s->id < 0 || (size_t)s->id >= t->nvocab) continue;
            t->id_bytes[s->id] = xmalloc(s->len);
            memcpy(t->id_bytes[s->id], s->bytes, s->len);
            t->id_lens[s->id] = s->len;
        }
        for (size_t i = 0; i < added.n; i++) {
            int32_t id = added.ids[i];
            if (id < 0 || (size_t)id >= t->nvocab) continue;
            t->id_bytes[id] = xmalloc(added.lens[i]);
            memcpy(t->id_bytes[id], added.bytes[i], added.lens[i]);
            t->id_lens[id] = added.lens[i];
            if (added.lens[i] == 13 &&
                memcmp(added.bytes[i], "<|endoftext|>", 13) == 0) {
                t->eos_id = id;
            }
        }
    }

    // merges ("left right" strings, rank = position)
    if (find_key(&j, "merges", 6)) {
        js_expect(&j, '[');
        if (!j.failed) {
            js c = j;
            size_t n = 0;
            while (!c.failed && !js_peek(&c, ']')) {
                uint8_t pair[2 * MAX_TOK_BYTES];
                if (js_string_raw(&c, pair, sizeof(pair)) < 0) break;
                if (!js_peek(&c, ',')) break;
                c.p++;
                n++;
            }
            mmap_init(&t->merges, n * 2 + 1024);
            int32_t rank = 0;
            while (!js_peek(&j, ']')) {
                uint8_t pair[2 * MAX_TOK_BYTES];
                long pl = js_string_raw(&j, pair, sizeof(pair));
                if (pl < 0) goto fail;
                long sp = -1;
                for (long i = 0; i < pl; i++) {
                    if (pair[i] == ' ') sp = i;
                }
                if (sp > 0 && sp + 1 < pl) {
                    int32_t a = vmap_get(&t->vocab, pair, (size_t)sp);
                    int32_t b = vmap_get(&t->vocab, pair + sp + 1,
                                         (size_t)(pl - sp - 1));
                    if (a >= 0 && b >= 0) {
                        // the merged token is left+right concatenated
                        uint8_t merged_bytes[2 * MAX_TOK_BYTES];
                        memcpy(merged_bytes, pair, (size_t)sp);
                        memcpy(merged_bytes + sp, pair + sp + 1,
                               (size_t)(pl - sp - 1));
                        int32_t merged = vmap_get(&t->vocab, merged_bytes,
                                                  (size_t)(pl - 1));
                        mmap_put(&t->merges, (uint32_t)a, (uint32_t)b, rank,
                                 merged);
                    }
                }
                rank++;
                if (!js_peek(&j, ',')) break;
                j.p++;
            }
            js_expect(&j, ']');
            if (j.failed) goto fail;
        }
    }
    if (j.failed) goto fail;

    free(buf);
    free(added.ids);
    free(added.bytes);
    free(added.lens);
    return t;
fail:
    free(buf);
    free(added.ids);
    free(added.bytes);
    free(added.lens);
    tokenizer_free(t);
    return NULL;
}

void tokenizer_free(tokenizer *t) {
    if (!t) return;
    vmap_free(&t->vocab);
    mmap_free(&t->merges);
    vmap_free(&t->uni);
    if (t->id_bytes) {
        for (size_t i = 0; i < t->nvocab; i++) free(t->id_bytes[i]);
        free(t->id_bytes);
        free(t->id_lens);
    }
    free(t);
}

int64_t tokenizer_eos(const tokenizer *t) {
    return t->eos_id;
}

// bpe one pre-token: bytes in, token ids out. returns count, -1 on error.
static long encode_chunk(const tokenizer *t, const uint8_t *s, size_t len,
                         int64_t *ids, size_t cap) {
    int32_t *cur = xmalloc((len ? len : 1) * sizeof(int32_t));
    int32_t *rank = xmalloc((len ? len : 1) * sizeof(int32_t));
    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        int32_t id = vmap_get(&t->vocab, t->byte_form[s[i]],
                              t->byte_form_len[s[i]]);
        if (id < 0) {
            free(cur);
            free(rank);
            return -1;
        }
        cur[n++] = id;
    }
    for (size_t i = 0; i + 1 < n; i++) {
        const mslot *m = mmap_find((merges_map *)&t->merges,
                                   (uint32_t)cur[i], (uint32_t)cur[i + 1]);
        rank[i] = m ? m->rank : -1;
    }
    while (n > 1) {
        size_t best = n;
        int32_t best_rank = INT32_MAX;
        for (size_t i = 0; i + 1 < n; i++) {
            if (rank[i] >= 0 && rank[i] < best_rank) {
                best_rank = rank[i];
                best = i;
            }
        }
        if (best == n) break;
        const mslot *m = mmap_find((merges_map *)&t->merges,
                                   (uint32_t)cur[best], (uint32_t)cur[best + 1]);
        if (!m || m->merged < 0) {
            rank[best] = -1;
            continue;
        }
        cur[best] = m->merged;
        memmove(cur + best + 1, cur + best + 2, (n - best - 2) * sizeof(int32_t));
        memmove(rank + best, rank + best + 1, (n - best - 2) * sizeof(int32_t));
        n--;
        if (best > 0) {
            const mslot *l = mmap_find((merges_map *)&t->merges,
                                       (uint32_t)cur[best - 1],
                                       (uint32_t)cur[best]);
            rank[best - 1] = l ? l->rank : -1;
        }
        if (best + 1 < n) {
            const mslot *r = mmap_find((merges_map *)&t->merges,
                                       (uint32_t)cur[best],
                                       (uint32_t)cur[best + 1]);
            rank[best] = r ? r->rank : -1;
        }
    }
    if (n > cap) {
        free(cur);
        free(rank);
        return -1;
    }
    for (size_t i = 0; i < n; i++) ids[i] = cur[i];
    free(cur);
    free(rank);
    return (long)n;
}

int tokenizer_encode(tokenizer *t, const char *text, int64_t *ids, size_t cap) {
    const uint8_t *s = (const uint8_t *)text;
    size_t len = strlen(text);
    size_t total = 0;
    size_t i = 0;
    while (i < len) {
        size_t e = pretoken_end(s, len, i);
        if (!e) e = next_cp(s, len, i); // unmatched char stands alone
        long c = encode_chunk(t, s + i, e - i, ids + total, cap - total);
        if (c < 0) return -1;
        total += (size_t)c;
        i = e;
    }
    return (int)total;
}

char *tokenizer_decode(tokenizer *t, const int64_t *ids, size_t n) {
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= 0 && (size_t)ids[i] < t->nvocab && t->id_bytes[ids[i]]) {
            total += t->id_lens[ids[i]];
        }
    }
    char *out = xmalloc(total + 1);
    size_t p = 0;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= t->nvocab || !t->id_bytes[ids[i]]) {
            continue;
        }
        const uint8_t *b = t->id_bytes[ids[i]];
        size_t bl = t->id_lens[ids[i]], j = 0;
        while (j < bl) {
            // vocab keys are 1-2 byte utf-8 forms; map back to raw bytes
            size_t fl = (b[j] & 0x80) == 0 ? 1 : 2;
            if (j + fl > bl) fl = 1;
            int32_t raw = vmap_get(&t->uni, b + j, fl);
            out[p++] = raw >= 0 ? (char)(uint8_t)raw : (char)b[j];
            j += fl;
        }
    }
    out[p] = 0;
    return out;
}
