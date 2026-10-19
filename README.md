# qwen3.8-flash-next

colibri runs qwen3.8-flash-next on apple silicon. the model is a 180b mixture-of-experts, too big to hold in memory. the routed experts and the n-gram table stay on the ssd and get read in as the router asks for them. everything else runs in mlx with metal matmul.

this is the python runtime. a c port is in progress under `c/`, and it has no published numbers yet.

## benchmark

![benchmark](images/benchmark.png)

m3 max, 36 gb, mlx 0.32.2, 8 gib expert cache. 5 runs per prompt, 4 generated tokens, medians. each run is a fresh process and the filesystem cache is not cleared. raw results are in `artifacts/benchmark_results.json`.

- short prompts run at about 15 tok/s. their experts fit in the cache, and by the last three runs the prompt needed no new disk reads.
- medium and long prompts run at 1.1 to 1.4 tok/s. they route to more experts than the cache holds, so the ssd sets the speed.

one longer run: a 45-token prompt, 64 generated tokens, 1.33 tok/s decode, 52.7 GiB read from disk.

```sh
python -m colibri_runtime.cli --model "$MODEL_DIR" --expert-budget-gib 8 --ple-budget-gib 1 --mlx-cache-gib 1 --mlx-memory-gib 28 --io-workers 6 --expert-prefetch 2 --ple-prefetch 8 benchmark --prompt "Write a detailed explanation of how sparse mixture-of-experts inference streams experts from SSD on Apple Silicon. Cover routing, cache eviction, asynchronous prefetch, and the memory and latency tradeoffs in at least 200 words." --max-tokens 64 --runs 1
```

set `MODEL_DIR` to the local 4-bit model directory first.

## tests

29 python tests (`pytest`). they cover the expert cache, streaming reads, generation, model args, the benchmark, and tiny-model parity against transformers.
