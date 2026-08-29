# qwen3.8-flash-next local runtime

Local Apple Silicon setup for Qwen/Qwen3.8-Flash-Next using `llama.cpp` with
mmap disk-backed weights, CPU-pinned sparse MoE experts (`--cpu-moe`), and a
small Colibri-style cache utility layer for expert and n-gram disk caching.

## Machine + model facts inspected first

- Host: Apple M3 Max (14 CPU cores, 30 GPU cores, Metal 4)
- RAM: 36 GB unified (`hw.memsize=38654705664`)
- Free disk at start: ~375 GiB
- Official checkpoint metadata (source of truth from HF):
  - `model_type=qwen4_exp` / text model `qwen4_exp_text`
  - 48 layers, hidden 2560, 512 experts, top-10 experts/token
  - Original safetensors total size: ~360 GB over 131 shards

Because 360 GB safetensors is not practical on this 36 GB machine, the highest
quality known-compatible GGUF chosen to fit disk headroom was:

- `bartowski/Qwen3.8-Flash-Next-GGUF:Qwen3.8-Flash-Next-IQ1_S`
- Total GGUF size: ~65.29 GiB (2 shards)

## What was implemented

- `scripts/inspect_official.py`
  - Downloads and records official `config.json`, `model.safetensors.index.json`,
    tokenizer + generation config summary to `artifacts/official_summary.json`.
- `scripts/download_model.sh`
  - Resumable downloads into `~/qwen3.8next`:
    - official metadata/tokenizer files
    - IQ1_S GGUF shards
- `src/colibri_runtime/expert_cache.py`
  - Bounded LRU cache with pinning + hit/miss/eviction stats.
- `src/colibri_runtime/ngram_cache.py`
  - Independent disk-backed SQLite n-gram cache.
- `src/colibri_runtime/benchmark.py`, `scripts/run_benchmarks.py`
  - Repeatable benchmark harness with warmup and short/medium/long prompts,
    capturing wall time, TTFT approximation, max RSS, and llama timing output.
- Tests: `tests/test_cache.py` (passes in venv).

## Setup

```bash
brew install llama.cpp
python3 -m venv .venv
.venv/bin/pip install pytest
PYTHONPATH=src .venv/bin/pytest -q tests
python3 scripts/inspect_official.py
./scripts/download_model.sh
```

## Benchmark command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_benchmarks.py
```

Runtime launch configuration used by benchmark harness:

```bash
/usr/bin/time -l llama-cli \
  -m ~/qwen3.8next/models/bartowski-Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00001-of-00002.gguf \
  -c 8192 -t 12 -tb 12 --cpu-moe --gpu-layers 1 --temp 0
```

## Measured results (this run)

- Unit tests: `2 passed`
- Official metadata inspection: success
- Download status during run: IQ1_S shard downloads in progress (resumable),
  partial sizes observed up to ~15.4 GiB and ~12.4 GiB.
- End-to-end generation benchmark: **blocked** until both GGUF shards finish.
- Actual benchmark execution currently returns real loader errors:
  - `failed to open GGUF file ... No such file or directory`
  - no prompt/gen tokens-per-second yet because model file is not complete.

Once download completes, rerun `scripts/run_benchmarks.py` to populate prompt
processing and generation tokens/sec for warmup/short/medium/long cases.
