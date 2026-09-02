CC ?= clang
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wno-unused-parameter -Isrc -I.
LDLIBS = -framework Accelerate

OBJDIR = build
ALLOBJ = $(patsubst src/%.c,$(OBJDIR)/%.o,$(wildcard src/*.c))

all:

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

TESTS = test_json test_safetensors test_reader test_cache test_kernel

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

test_%: tests/test_%.c $(ALLOBJ) | $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $< $(ALLOBJ) $(LDLIBS)

clean:
	rm -rf $(OBJDIR) $(TESTS)

.PHONY: all test clean
