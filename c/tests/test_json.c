#include <math.h>
#include <string.h>

#include "json.h"
#include "test.h"
#include "util.h"

static void test_basic_scalars(void) {
    char err[256];
    json_value *v = json_parse("{\"a\": 1, \"b\": -2.5, \"c\": true, \"d\": null}", strlen("{\"a\": 1, \"b\": -2.5, \"c\": true, \"d\": null}"), err, sizeof(err));
    CHECK_MSG(v != NULL, "%s", err);
    if (!v) return;
    CHECK(v->type == JSON_OBJ);
    CHECK(json_num_i64(json_obj_get(v, "a")) == 1);
    CHECK(fabs(json_num_f64(json_obj_get(v, "b")) + 2.5) < 1e-12);
    CHECK(json_obj_get(v, "c")->type == JSON_BOOL);
    CHECK(json_obj_get(v, "c")->u.boolean == 1);
    CHECK(json_obj_get(v, "d")->type == JSON_NULL);
    CHECK(json_obj_get(v, "missing") == NULL);
    json_free(v);
}

static void test_strings(void) {
    char err[256];
    json_value *v = json_parse("\"he\\nllo\\u0041\\u00e9\\ud83d\\ude00\"", strlen("\"he\\nllo\\u0041\\u00e9\\ud83d\\ude00\""), err, sizeof(err));
    CHECK_MSG(v != NULL, "%s", err);
    if (!v) return;
    CHECK(v->type == JSON_STR);
    CHECK(strcmp(v->u.str.s, "he\nlloA\xc3\xa9\xf0\x9f\x98\x80") == 0);
    json_free(v);
}

static void test_arrays(void) {
    char err[256];
    json_value *v = json_parse("[1, [2, 3], {\"x\": [4]}]", strlen("[1, [2, 3], {\"x\": [4]}]"), err, sizeof(err));
    CHECK_MSG(v != NULL, "%s", err);
    if (!v) return;
    CHECK(v->type == JSON_ARR);
    CHECK(v->u.arr.n == 3);
    CHECK(v->u.arr.items[1]->u.arr.n == 2);
    json_value *x = json_obj_get(v->u.arr.items[2], "x");
    CHECK(x && x->u.arr.items[0]->u.num == 4);
    json_free(v);
}

static void test_empty(void) {
    char err[256];
    json_value *v = json_parse("{}", 2, err, sizeof(err));
    CHECK(v && v->type == JSON_OBJ && v->u.obj.n == 0);
    json_free(v);
    v = json_parse("[]", 2, err, sizeof(err));
    CHECK(v && v->type == JSON_ARR && v->u.arr.n == 0);
    json_free(v);
}

static void test_errors(void) {
    char err[256];
    const char *bad[] = {"{", "[1,]", "{\"a\":}", "\"unterminated", "{\"a\":1,}", "tru", "1 2", ""};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        json_value *v = json_parse(bad[i], strlen(bad[i]), err, sizeof(err));
        CHECK_MSG(v == NULL, "should reject: %s", bad[i]);
        json_free(v);
    }
}

static void test_deep(void) {
    char err[256];
    // nested 64 levels
    char buf[300];
    size_t n = 0;
    for (int i = 0; i < 64; i++) buf[n++] = '[';
    for (int i = 0; i < 64; i++) buf[n++] = ']';
    json_value *v = json_parse(buf, n, err, sizeof(err));
    CHECK_MSG(v != NULL, "%s", err);
    json_value *cur = v;
    for (int i = 0; i < 64 && cur; i++) {
        CHECK(cur->type == JSON_ARR);
        cur = cur->u.arr.n ? cur->u.arr.items[0] : NULL;
    }
    json_free(v);
}

int main(void) {
    RUN_TEST(test_basic_scalars);
    RUN_TEST(test_strings);
    RUN_TEST(test_arrays);
    RUN_TEST(test_empty);
    RUN_TEST(test_errors);
    RUN_TEST(test_deep);
    return test_summary("json");
}
