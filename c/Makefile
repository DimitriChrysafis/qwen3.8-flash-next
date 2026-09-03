CC ?= clang
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wno-unused-parameter -Isrc -I.
LDLIBS = -framework Accelerate -licucore

OBJDIR = build
LIBOBJ = $(patsubst src/%.c,$(OBJDIR)/%.o,$(filter-out src/main.c,$(wildcard src/*.c)))

all:

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

BIN = qwen38
SRC = src/json.c src/safetensors.c src/reader.c src/cache.c src/kernel.c \
      src/ngram.c src/model.c src/tokenizer.c src/main.c

all: $(BIN)

$(BIN): $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRC))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

TESTS = test_json test_safetensors test_reader test_cache test_kernel test_model test_tokenizer

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

# tiny-model parity against the colibri reference (needs python mlx + mlx-lm)
parity: test_model
	@python3 tests/gen_tiny.py /tmp/qwen_tiny ./test_model

test_%: tests/test_%.c $(LIBOBJ) | $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBOBJ) $(LDLIBS)

clean:
	rm -rf $(OBJDIR) $(TESTS) $(BIN)

.PHONY: all test clean
