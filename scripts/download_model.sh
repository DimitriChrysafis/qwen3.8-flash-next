#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HOME}/qwen3.8next"
MODEL_DIR="${BASE_DIR}/models/bartowski-Qwen3.8-Flash-Next-IQ1_S"
mkdir -p "${MODEL_DIR}"

huggingface-cli download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json --local-dir "${BASE_DIR}/official" --resume-download

huggingface-cli download bartowski/Qwen3.8-Flash-Next-GGUF \
  Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00001-of-00002.gguf \
  Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00002-of-00002.gguf \
  --local-dir "${MODEL_DIR}" --resume-download

echo "Downloaded model shards into ${MODEL_DIR}"