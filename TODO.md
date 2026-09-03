# TODO

## done

- c runtime scaffold (makefile, build, test harness)
- json parser, safetensors index with validation
- threaded row reader (batched pread, run merging, no lost wakeups)
- adaptive byte-budget cache (lru/lfu/adaptive, pin/hot, partition fairness)
- math kernels: rmsnorm (grouped + broadcast weights), rope partial, silu,
  sigmoid, softmax, l2norm, topk (quickselect), q4 dequant, fp32 gemm (cblas),
  bf16 gemm (bnns), q4 gemm, depthwise dilated conv (bf16 taps)
- model: config, dense weights (bf16 + q4), streamed experts + ple rows,
  full attention with qsa indexer, gated deltanet, moe with shared expert,
  gated residuals (hyper connections), ngram embedding with exact mlx
  geometry, prefill + incremental decode
- tiny-model parity vs colibri: 100% argmax agreement, self-consistency 3e-7
- cli: generate + benchmark, greedy/temperature/top-p sampling
- byte-level bpe tokenizer: loads the real 12.8 mb tokenizer.json in ~0.2 s,
  byte-to-unicode table, qwen3 pretokenization (icu classes), added tokens +
  eos, encode matches the hf reference exactly on ascii/unicode/emoji,
  decode wired into the cli
- benchmark --runs loop (reset caches between runs, averaged stats)
- real model end to end (pipenetwork 4bit dir): fixed language_model. prefix
  lookup, bf16 scales in row dequant, 3d expert block dequant, packed q4
  gemm dims, ple shard routing (rows per shard from the shard shape, not
  total/split), ple projector group size, ple scratch overflow, mwset
  pointer stability, cache value leaks. layer 0-2 hidden states match the
  colibri reference to 4 decimals; top-1 logits agree

## next (correctness)

- [ ] real-model numerical parity: c runs f32 activations vs colibri bf16;
      per-layer diffs grow through expert-routing flips (corr 0.999 at
      layer 2, 0.67 on final logits). decide whether to match bf16 rounding
      per op or accept the drift; compare moe routes given identical inputs
- [ ] special-token aware encode (split text on added_tokens before bpe) so
      --prompt can carry <|im_start|> etc.
- [ ] chat template support (--chat flag is a no-op)
- [ ] benchmark the real model and update README numbers
- [ ] remove the python runtime (src/colibri_runtime, tests/, scripts/,
      pyproject.toml, setup.py, native/) once parity is captured in a script
      that does not need it; tests/gen_tiny.py is already gone but the
      makefile parity target still references it

## later (speed)

- [ ] expert prefetch via transition statistics (colibri did this; flags
      --expert-prefetch/--ple-prefetch exist but are unused)
- [ ] overlap io with compute: prefetch next layer's experts while the moe
      matmuls of the current layer run
- [ ] q4 gemm: vectorize the nibble dot product (neon), or dequant-once-at-
      load flag for dense weights
- [ ] attention: fuse the indexer + sdpa loops, avoid per-token mallocs
- [ ] prefill batching: batch expert matmuls across tokens per unique expert
      (already grouped; tune tile sizes for bnns)
- [ ] avoid re-reading dense weights: keep bf16 gemm workspace reused (done),
      consider a dequantized fp32 path for always-hot layers
- [ ] profile with instruments: find where decode time actually goes on the
      real model (ssd vs gemm vs attention)
- [ ] tokenizer encode is o(n) per merge per pre-token; fine for prompts,
      switch to a heap if long-prefill throughput ever matters

## misc

- [ ] fallclasses repo push still fails with http 408 on this network; retry
      later or push from a copy outside icloud drive
- [ ] memory: the real model wants ~11 gib dense + expert cache; on a 36 gb
      machine close apps first or the jetsam killer sigkills the run
