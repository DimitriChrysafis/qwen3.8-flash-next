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

Measured result: `8 passed`. A generated tiny Qwen4Exp checkpoint is compared directly with Transformers 5.16.1 across full, incremental, and chunked execution. Maximum absolute logit error was `4.4703484e-08`; incremental versus full error was `2.9802322e-08`. Separate tests cover exact range reads, 4-bit expert matmul parity, adaptive eviction, asynchronous prefetch, and 16-row PLE lookup.

## Benchmarks

```bash
.venv/bin/python scripts/run_benchmarks.py --max-tokens 4 --runs 2
```

Measured on 2026-08-29 with an Apple M3 Max (14 CPU cores, 30 GPU cores), 36 GB unified RAM, internal Apple NVMe, macOS 27.0, MLX 0.32.2, and an 8 GiB expert cache. Each first run used a new process; the macOS filesystem cache was not forcibly cleared. Decode throughput excludes the first token. Every case generated four tokens.

### First run

| Prompt | Tokens | Init | TTFT | Prefill | Decode | Avg RSS | Peak RSS | Peak MLX | Disk read | I/O wait | Expert hit | PLE prefetch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 11 | 0.456 s | 2.884 s | 3.82 tok/s | 1.52 tok/s | 4.25 GiB | 7.62 GiB | 10.93 GiB | 7.79 GiB | 1.95 s | 22.97% | 100% |
| Medium | 122 | 0.483 s | 20.591 s | 5.93 tok/s | 0.76 tok/s | 2.71 GiB | 8.06 GiB | 11.52 GiB | 24.72 GiB | 4.89 s | 5.16% | 100% |
| Long | 362 | 0.522 s | 30.637 s | 11.82 tok/s | 0.72 tok/s | 2.51 GiB | 7.60 GiB | 12.21 GiB | 32.19 GiB | 5.84 s | 3.80% | 100% |

### Repeated same prompt, same process

| Prompt | TTFT | Prefill | Decode | Peak RSS | Additional disk read | Expert hit |
|---|---:|---:|---:|---:|---:|---:|
| Short | 1.256 s | 8.76 tok/s | 6.69 tok/s | 2.63 GiB | 0 GiB | 100% |
| Medium | 26.016 s | 4.69 tok/s | 0.99 tok/s | 4.31 GiB | 21.98 GiB | 14.55% |
| Long | 36.235 s | 9.99 tok/s | 0.92 tok/s | 3.82 GiB | 29.61 GiB | 10.44% |

The full JSON report is written to `artifacts/benchmark_results.json`. The 8 GiB cache retains every expert used by the short prompt, producing a zero-read second run. Medium and long prompts route through more experts than the cache can retain and remain SSD-bound.

A current `llama.cpp` GGUF fallback was also exercised with CPU-mapped experts and n-gram weights. One real IQ1_S token required 90.86 seconds end-to-end, reached 19.57 GiB RSS, and processed the prompt at 3.7 token/s. The streamed MLX path is the practical implementation on this machine.
