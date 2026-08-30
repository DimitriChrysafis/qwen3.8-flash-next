# Qwen3.8-Flash-Next Colibri runtime

Disk-streamed Apple Silicon inference for `Qwen/Qwen3.8-Flash-Next`. The runtime executes the complete text architecture with MLX while keeping routed experts and the 51B-parameter n-gram table on SSD.

## Requirements and setup

- Apple Silicon
- Python 3.11+
- About 104 GB for the 4-bit checkpoint

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
HF_BIN=.venv/bin/hf ./scripts/download_model.sh
```

The runtime uses `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, converted from the official BF16 release. It is 103,770,199,081 bytes (96.64 GiB): 4-bit group-64 weights, group-32 n-gram rows, and BF16 routing/selection weights. The official BF16 checkpoint is 360.02 GB and cannot coexist with its conversion workspace on the measured machine, so it is inspected as the architecture source of truth rather than loaded for inference.

## Run

```bash
.venv/bin/colibri-qwen \
  --model ~/qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit \
  --expert-budget-gib 8 --ple-budget-gib 1 \
  --mlx-cache-gib 1 --mlx-memory-gib 28 \
  generate --chat \
  --prompt "In one sentence, explain why the sky appears blue." \
  --max-tokens 64 --temperature 0
```

The measured 24-token run began:

```text
The sky appears blue because molecules in the Earth's atmosphere scatter shorter-wavelength blue light from the sun more strongly than longer
```

It decoded at 1.58 token/s, used 6.81 GiB peak RSS and 11.23 GiB peak MLX memory, read 18.24 GiB, and remained below the 36 GB target.

## Implementation

- Safetensors headers are indexed without loading tensor payloads. `os.pread` fetches only selected expert slices and n-gram rows.
- A small optional CPython C extension performs contiguous row reads without holding the GIL, skips sorting already ordered requests, and reuses its largest range buffer; the validated Python reader remains available as a fallback.
- Every MoE layer runs its BF16 router, loads the selected top-10 experts asynchronously, executes their 4-bit Metal matmuls, and updates a byte-bounded adaptive LFU/LRU cache.
- Route frequencies and transitions drive expert prefetch. Entries support hot and pinned states.
- The 320,001,536-row n-gram table remains on SSD. The exact 16 rows required by known input tokens are prefetched before layer 1 and overlap layer 0 compute.
- The resident path implements embeddings, four-stream gated residuals, 36 Gated DeltaNet layers and recurrent state, 12 QSA layers and indexer/KV state, PLE projection and convolution, shared experts, output head, sampling, and generation.
- Metrics include RSS samples, MLX active/cache/peak memory, exact requested/read bytes, read calls, I/O wait, effective bandwidth, cache hits/misses/evictions, route frequency, and prefetch hits.

Checkpoint accounting from the real loader:

| Component | Bytes | GiB |
|---|---:|---:|
| Streamed experts and n-gram weights | 99,947,878,400 | 93.08 |
| Always-used lazy weights | 2,924,005,400 | 2.72 |
| Indexed tensors | 3,215 | — |

## Validation

```bash
.venv/bin/pip install -e '.[validation]'
PYTHONPATH=src .venv/bin/pytest -q tests
```

Measured result: `29 passed`. A generated tiny Qwen4Exp checkpoint is compared directly with Transformers 5.16.1 across full, incremental, and chunked execution. Maximum absolute logit error was `4.4703484e-08`; incremental versus full error was `2.9802322e-08`. Separate tests cover exact range reads, both native and Python row readers, 4-bit expert matmul parity, adaptive eviction, asynchronous prefetch, generation validation, and 16-row PLE lookup.

## Benchmarks

```bash
.venv/bin/python scripts/run_benchmarks.py --max-tokens 4 --runs 5
```

Measured on 2026-08-30 with an Apple M3 Max (14 CPU cores, 30 GPU cores), 36 GB unified RAM, internal Apple NVMe, macOS 27.0, MLX 0.32.2, and an 8 GiB expert cache. Each prompt case used a fresh process and five runs; the macOS filesystem cache was not forcibly cleared. Values below are medians across the five runs. Decode throughput excludes the first token. Every case generated four tokens.

| Prompt | Tokens | TTFT | Prefill | Decode | Peak RSS | Peak MLX | Disk read (cumulative) | I/O wait (cumulative) | Expert prefetch | PLE prefetch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 11 | 0.382 s | 28.82 tok/s | 15.04 tok/s | 4.24 GiB | 11.21 GiB | 10.56 GiB | 10.32 s | 0.00% | 100% |
| Medium | 122 | 15.818 s | 7.71 tok/s | 1.38 tok/s | 2.81 GiB | 11.64 GiB | 66.80 GiB | 92.58 s | 8.94% | 100% |
| Long | 362 | 20.661 s | 17.52 tok/s | 1.13 tok/s | 3.49 GiB | 12.36 GiB | 91.41 GiB | 126.40 s | 9.99% | 100% |

Per-run disk reads for the first and subsequent runs were 8.01/0.00 GiB for short, 24.56/20.65 GiB for medium, and 32.52/29.11 GiB for long, where the second value is the median of runs two through five.

The full JSON report is written to `artifacts/benchmark_results.json`. The short case reached zero additional disk reads on its final three runs. Medium and long prompts route through more experts than the cache can retain and remain SSD-bound.

A current `llama.cpp` GGUF fallback was also exercised with CPU-mapped experts and n-gram weights. One real IQ1_S token required 90.86 seconds end-to-end, reached 19.57 GiB RSS, and processed the prompt at 3.7 token/s. The streamed MLX path is the practical implementation on this machine.
