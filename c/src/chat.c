#include <string.h>

#include "chat.h"
#include "util.h"

static const char *INSTRUCTIONS =
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer.";

static void append(char **out, size_t *cap, size_t *len, const char *s) {
    size_t sl = strlen(s);
    if (*len + sl + 1 > *cap) {
        do {
            *cap = *cap ? *cap * 2 : 256;
        } while (*len + sl + 1 > *cap);
        *out = xrealloc(*out, *cap);
    }
    memcpy(*out + *len, s, sl);
    *len += sl;
    (*out)[*len] = 0;
}

char *chat_render(const char *system, const char *user, int thinking) {
    char *out = xmalloc(256);
    size_t cap = 256, len = 0;
    out[0] = 0;
    if (thinking) {
        append(&out, &cap, &len, "<|im_start|>system\n");
        append(&out, &cap, &len, INSTRUCTIONS);
        if (system) {
            append(&out, &cap, &len, "\n\n");
            append(&out, &cap, &len, system);
        }
        append(&out, &cap, &len, "<|im_end|>\n");
    } else if (system) {
        append(&out, &cap, &len, "<|im_start|>system\n");
        append(&out, &cap, &len, system);
        append(&out, &cap, &len, "<|im_end|>\n");
    }
    append(&out, &cap, &len, "<|im_start|>user\n");
    append(&out, &cap, &len, user);
    append(&out, &cap, &len, "<|im_end|>\n");
    append(&out, &cap, &len, "<|im_start|>assistant\n");
    append(&out, &cap, &len, thinking ? " thinking\n" : " thinking\n\n response\n\n");
    return out;
}
