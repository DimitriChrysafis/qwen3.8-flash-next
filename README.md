# qwen3.8-flash-next

colibri runs qwen3.8-flash-next on apple silicon with mlx while routed experts and the n-gram table stay on ssd. the hot path combines sparse routing, asynchronous reads, adaptive caching, and metal matmul.

## benchmark

five runs per prompt, four generated tokens, m3 max, 36 gb unified memory, mlx 0.32.2, 8 gib expert cache. medians are shown; each prompt used a fresh process and the filesystem cache was not cleared.

| prompt | input tokens | ttft | prefill | decode | peak rss | peak mlx | disk read (cumulative) | expert prefetch | ple prefetch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 11 | 0.382 s | 28.82 tok/s | 15.04 tok/s | 4.24 gib | 11.21 gib | 10.56 gib | 0.00% | 100% |
| medium | 122 | 15.818 s | 7.71 tok/s | 1.38 tok/s | 2.81 gib | 11.64 gib | 66.80 gib | 8.94% | 100% |
| long | 362 | 20.661 s | 17.52 tok/s | 1.13 tok/s | 3.49 gib | 12.36 gib | 91.41 gib | 9.99% | 100% |

the short case reached zero additional disk reads on its final three runs. medium and long remain ssd-bound because their routed expert sets exceed the 8 gib cache. the raw report is in `artifacts/benchmark_results.json`.

## longer decode

one run on sep 24, 2026: m3 max, 36 gib unified memory, 4-bit qwen3.8-flash-next, mlx 0.32.2, 8 gib expert cache, six i/o workers, greedy decoding, filesystem cache not cleared. a 45-token prompt generated 64 tokens (63 timed decode tokens): 18.35 s prefill, 47.49 s decode, 1.33 tok/s. peak process rss was 4.02 gib and peak mlx allocation was 11.34 gib; the run read 52.74 gib from disk. these memory measures are separate and should not be added together.

set `MODEL_DIR` to the local 4-bit model directory to reproduce the run:

```sh
python -m colibri_runtime.cli --model "$MODEL_DIR" --expert-budget-gib 8 --ple-budget-gib 1 --mlx-cache-gib 1 --mlx-memory-gib 28 --io-workers 6 --expert-prefetch 2 --ple-prefetch 8 benchmark --prompt "Write a detailed explanation of how sparse mixture-of-experts inference streams experts from SSD on Apple Silicon. Cover routing, cache eviction, asynchronous prefetch, and the memory and latency tradeoffs in at least 200 words." --max-tokens 64 --runs 1
```

38 tests pass, including tiny-model parity against transformers, native and python row-reader parity, malformed safetensor validation, cache behavior, async prefetch, special-token encode, chat template rendering, and generation validation.
