CC ?= clang
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wno-unused-parameter -Isrc -I.
LDLIBS = -framework Accelerate

OBJDIR = build

all:

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

test: test_json
	@for t in test_json; do ./$$t || exit 1; done

test_%: tests/test_%.c $(OBJDIR)/%.o | $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $< $(OBJDIR)/$*.o $(LDLIBS)

clean:
	rm -rf $(OBJDIR) test_json

.PHONY: all test clean
