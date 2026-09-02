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

29 tests pass, including tiny-model parity against transformers, native and python row-reader parity, malformed safetensor validation, cache behavior, async prefetch, and generation validation.
