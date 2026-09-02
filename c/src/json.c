#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef struct {
    const char *p;
    const char *end;
    char *err;
    size_t errlen;
} jp;

static void jfail(jp *j, const char *msg) {
    if (j->errlen) {
        snprintf(j->err, j->errlen, "%s at offset %td", msg, j->p - j->end < 0 ? (long)(j->p - j->end) : 0);
    }
    j->err = NULL; // stop further reporting
}

static void jskipws(jp *j) {
    while (j->p < j->end && isspace((unsigned char)*j->p)) j->p++;
}

static json_value *jnew(jp *j, json_type t) {
    (void)j;
    json_value *v = xcalloc(1, sizeof(json_value));
    v->type = t;
    return v;
}

static json_value *jparse_value(jp *j);

static json_value *jparse_string(jp *j) {
    if (j->p >= j->end || *j->p != '"') {
        jfail(j, "expected string");
        return NULL;
    }
    j->p++;
    size_t cap = 16, n = 0;
    char *buf = xmalloc(cap);
    while (j->p < j->end) {
        unsigned char c = (unsigned char)*j->p;
        if (c == '"') {
            j->p++;
            buf[n] = 0;
            json_value *v = jnew(j, JSON_STR);
            v->u.str.s = buf;
            v->u.str.len = n;
            return v;
        }
        if (c == '\\') {
            j->p++;
            if (j->p >= j->end) break;
            char esc = *j->p++;
            switch (esc) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    // decode \uXXXX as utf-8 (surrogate pairs included)
                    if (j->end - j->p < 4) {
                        jfail(j, "truncated \\u escape");
                        free(buf);
                        return NULL;
                    }
                    uint32_t cp = 0;
                    int ok = 1;
                    for (int i = 0; i < 4; i++) {
                        char h = j->p[i];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (uint32_t)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (uint32_t)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (uint32_t)(h - 'A' + 10);
                        else { ok = 0; break; }
                    }
                    if (!ok) {
                        jfail(j, "bad \\u escape");
                        free(buf);
                        return NULL;
                    }
                    j->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                        uint32_t lo = 0;
                        int lok = 1;
                        for (int i = 0; i < 4; i++) {
                            char h = j->p[2 + i];
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= (uint32_t)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (uint32_t)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (uint32_t)(h - 'A' + 10);
                            else { lok = 0; break; }
                        }
                        if (lok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            j->p += 6;
                        }
                    }
                    // encode utf-8
                    if (cp < 0x80) {
                        if (n + 1 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
                        buf[n++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (n + 2 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
                        buf[n++] = (char)(0xC0 | (cp >> 6));
                        buf[n++] = (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        if (n + 3 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
                        buf[n++] = (char)(0xE0 | (cp >> 12));
                        buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[n++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        if (n + 4 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
                        buf[n++] = (char)(0xF0 | (cp >> 18));
                        buf[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                        buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[n++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue;
                }
                default:
                    jfail(j, "bad escape");
                    free(buf);
                    return NULL;
            }
            if (n + 1 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
            buf[n++] = (char)c;
            continue;
        }
        if (n + 1 >= cap) { cap *= 2; buf = xrealloc(buf, cap); }
        buf[n++] = (char)c;
        j->p++;
    }
    jfail(j, "unterminated string");
    free(buf);
    return NULL;
}

static json_value *jparse_value(jp *j) {
    jskipws(j);
    if (j->p >= j->end) {
        jfail(j, "unexpected end of input");
        return NULL;
    }
    char c = *j->p;
    if (c == '{') {
        j->p++;
        json_value *v = jnew(j, JSON_OBJ);
        jskipws(j);
        if (j->p < j->end && *j->p == '}') {
            j->p++;
            return v;
        }
        for (;;) {
            jskipws(j);
            json_value *key = jparse_string(j);
            if (!key) {
                json_free(v);
                return NULL;
            }
            jskipws(j);
            if (j->p >= j->end || *j->p != ':') {
                jfail(j, "expected ':'");
                json_free(key);
                json_free(v);
                return NULL;
            }
            j->p++;
            json_value *val = jparse_value(j);
            if (!val) {
                json_free(key);
                json_free(v);
                return NULL;
            }
            if (v->u.obj.n == v->u.obj.cap) {
                v->u.obj.cap = v->u.obj.cap ? v->u.obj.cap * 2 : 8;
                v->u.obj.keys = xrealloc(v->u.obj.keys, v->u.obj.cap * sizeof(char *));
                v->u.obj.vals = xrealloc(v->u.obj.vals, v->u.obj.cap * sizeof(json_value *));
            }
            v->u.obj.keys[v->u.obj.n] = key->u.str.s;
            v->u.obj.vals[v->u.obj.n] = val;
            v->u.obj.n++;
            free(key);
            jskipws(j);
            if (j->p >= j->end) {
                jfail(j, "unterminated object");
                json_free(v);
                return NULL;
            }
            if (*j->p == ',') {
                j->p++;
                continue;
            }
            if (*j->p == '}') {
                j->p++;
                return v;
            }
            jfail(j, "expected ',' or '}'");
            json_free(v);
            return NULL;
        }
    }
    if (c == '[') {
        j->p++;
        json_value *v = jnew(j, JSON_ARR);
        jskipws(j);
        if (j->p < j->end && *j->p == ']') {
            j->p++;
            return v;
        }
        for (;;) {
            json_value *item = jparse_value(j);
            if (!item) {
                json_free(v);
                return NULL;
            }
            if (v->u.arr.n == v->u.arr.cap) {
                v->u.arr.cap = v->u.arr.cap ? v->u.arr.cap * 2 : 8;
                v->u.arr.items = xrealloc(v->u.arr.items, v->u.arr.cap * sizeof(json_value *));
            }
            v->u.arr.items[v->u.arr.n++] = item;
            jskipws(j);
            if (j->p >= j->end) {
                jfail(j, "unterminated array");
                json_free(v);
                return NULL;
            }
            if (*j->p == ',') {
                j->p++;
                continue;
            }
            if (*j->p == ']') {
                j->p++;
                return v;
            }
            jfail(j, "expected ',' or ']'");
            json_free(v);
            return NULL;
        }
    }
    if (c == '"') return jparse_string(j);
    if (c == 't') {
        if (j->end - j->p >= 4 && memcmp(j->p, "true", 4) == 0) {
            j->p += 4;
            json_value *v = jnew(j, JSON_BOOL);
            v->u.boolean = 1;
            return v;
        }
        jfail(j, "bad literal");
        return NULL;
    }
    if (c == 'f') {
        if (j->end - j->p >= 5 && memcmp(j->p, "false", 5) == 0) {
            j->p += 5;
            json_value *v = jnew(j, JSON_BOOL);
            v->u.boolean = 0;
            return v;
        }
        jfail(j, "bad literal");
        return NULL;
    }
    if (c == 'n') {
        if (j->end - j->p >= 4 && memcmp(j->p, "null", 4) == 0) {
            j->p += 4;
            return jnew(j, JSON_NULL);
        }
        jfail(j, "bad literal");
        return NULL;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *endp;
        double d = strtod(j->p, &endp);
        if (endp == j->p) {
            jfail(j, "bad number");
            return NULL;
        }
        j->p = endp;
        json_value *v = jnew(j, JSON_NUM);
        v->u.num = d;
        return v;
    }
    jfail(j, "unexpected character");
    return NULL;
}

json_value *json_parse(const char *text, size_t len, char *err, size_t errlen) {
    jp j = {text, text + len, err, errlen};
    if (err && errlen) err[0] = 0;
    json_value *v = jparse_value(&j);
    if (!v) return NULL;
    jskipws(&j);
    if (j.p != j.end) {
        jfail(&j, "trailing content");
        json_free(v);
        return NULL;
    }
    return v;
}

json_value *json_obj_get(const json_value *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJ) return NULL;
    for (size_t i = 0; i < obj->u.obj.n; i++) {
        if (strcmp(obj->u.obj.keys[i], key) == 0) return obj->u.obj.vals[i];
    }
    return NULL;
}

int64_t json_num_i64(const json_value *v) {
    if (!v || v->type != JSON_NUM) return 0;
    return (int64_t)v->u.num;
}

double json_num_f64(const json_value *v) {
    if (!v || v->type != JSON_NUM) return 0.0;
    return v->u.num;
}

void json_free(json_value *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STR:
            free(v->u.str.s);
            break;
        case JSON_ARR:
            for (size_t i = 0; i < v->u.arr.n; i++) json_free(v->u.arr.items[i]);
            free(v->u.arr.items);
            break;
        case JSON_OBJ:
            for (size_t i = 0; i < v->u.obj.n; i++) {
                free(v->u.obj.keys[i]);
                json_free(v->u.obj.vals[i]);
            }
            free(v->u.obj.keys);
            free(v->u.obj.vals);
            break;
        default:
            break;
    }
    free(v);
}
