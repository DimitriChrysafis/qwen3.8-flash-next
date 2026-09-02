#ifndef QWEN_JSON_H
#define QWEN_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUM,
    JSON_STR,
    JSON_ARR,
    JSON_OBJ,
} json_type;

typedef struct json_value json_value;

struct json_value {
    json_type type;
    union {
        int boolean;
        double num;
        struct {
            char *s;
            size_t len;
        } str;
        struct {
            json_value **items;
            size_t n, cap;
        } arr;
        struct {
            char **keys;
            json_value **vals;
            size_t n, cap;
        } obj;
    } u;
};

// parse a json document. returns NULL on error and fills err.
json_value *json_parse(const char *text, size_t len, char *err, size_t errlen);

// linear object lookup; returns NULL when missing.
json_value *json_obj_get(const json_value *obj, const char *key);

// number value as int64; returns 0 when not a number.
int64_t json_num_i64(const json_value *v);
double json_num_f64(const json_value *v);

void json_free(json_value *v);

#endif
