#include <stdio.h>
#include <string.h>

#include "chat.h"
#include "test.h"

static const char *INSTRUCTIONS =
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer.";

static void test_thinking_default(void) {
    char *s = chat_render(NULL, "hi", 1);
    char want[1024];
    snprintf(want, sizeof(want),
             "<|im_start|>system\n%s<|im_end|>\n"
             "<|im_start|>user\nhi<|im_end|>\n"
             "<|im_start|>assistant\n thinking\n",
             INSTRUCTIONS);
    CHECK_MSG(s && strcmp(s, want) == 0, "got: %s", s ? s : "(null)");
    free(s);
}

static void test_thinking_off(void) {
    char *s = chat_render(NULL, "hi", 0);
    CHECK_MSG(s && strcmp(s,
             "<|im_start|>user\nhi<|im_end|>\n"
             "<|im_start|>assistant\n thinking\n\n response\n\n") == 0,
             "got: %s", s ? s : "(null)");
    free(s);
}

static void test_system(void) {
    char *s = chat_render("be brief", "hi", 1);
    char want[1024];
    snprintf(want, sizeof(want),
             "<|im_start|>system\n%s\n\nbe brief<|im_end|>\n"
             "<|im_start|>user\nhi<|im_end|>\n"
             "<|im_start|>assistant\n thinking\n",
             INSTRUCTIONS);
    CHECK_MSG(s && strcmp(s, want) == 0, "got: %s", s ? s : "(null)");
    free(s);

    s = chat_render("be brief", "hi", 0);
    CHECK_MSG(s && strcmp(s,
             "<|im_start|>system\nbe brief<|im_end|>\n"
             "<|im_start|>user\nhi<|im_end|>\n"
             "<|im_start|>assistant\n thinking\n\n response\n\n") == 0,
             "got: %s", s ? s : "(null)");
    free(s);
}

static void test_long_message(void) {
    char user[4096];
    memset(user, 'a', sizeof(user) - 1);
    user[sizeof(user) - 1] = 0;
    char *s = chat_render(NULL, user, 1);
    CHECK(s != NULL);
    if (s) {
        CHECK(strlen(s) > 4096);
        CHECK(strncmp(s, "<|im_start|>system\n", 19) == 0);
        CHECK(strcmp(s + strlen(s) - 10, " thinking\n") == 0);
        free(s);
    }
}

int main(void) {
    RUN_TEST(test_thinking_default);
    RUN_TEST(test_thinking_off);
    RUN_TEST(test_system);
    RUN_TEST(test_long_message);
    return test_summary("test_chat");
}
