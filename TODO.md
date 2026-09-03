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

## next (correctness)

- [ ] tokenizer: load fails on the real tokenizer.json (12.8 mb, 248k vocab).
      "tok: load start" prints, then load failed. likely the merges pass or a
      key > 512 bytes. debug with prints, then wire encode/decode into the cli
      (generate currently prints raw ids)
- [ ] run the real model end to end (pipenetwork 4bit dir) and sanity check
      generated text
- [ ] benchmark the real model and update README numbers
- [ ] benchmark command ignores --runs; make it loop
- [ ] clean the leftover debug dump code in tests/gen_tiny.py (the per-layer
      bisection dumps) and any stray blank lines in src/model.c
- [ ] remove the python runtime (src/colibri_runtime, tests/, scripts/,
      pyproject.toml, setup.py, native/) once parity is captured in a script
      that does not need it
- [ ] chat template support (--chat flag is a no-op)

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

## misc

- [ ] fallclasses repo push still fails with http 408 on this network; retry
      later or push from a copy outside icloud drive
