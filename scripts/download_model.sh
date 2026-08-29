#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${COLIBRI_MODEL_HOME:-${HOME}/qwen3.8next}"
MLX_DIR="${BASE_DIR}/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
HF_BIN="${HF_BIN:-hf}"

"${HF_BIN}" download Qwen/Qwen3.8-Flash-Next \
  config.json model.safetensors.index.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir "${BASE_DIR}/official"
"${HF_BIN}" download pipenetwork/Qwen3.8-Flash-Next-MLX-4bit --local-dir "${MLX_DIR}"

if [[ "${COLIBRI_DOWNLOAD_GGUF:-0}" == "1" ]]; then
  GGUF_DIR="${BASE_DIR}/models/bartowski-Qwen3.8-Flash-Next-IQ1_S"
  "${HF_BIN}" download bartowski/Qwen3.8-Flash-Next-GGUF \
    Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00001-of-00002.gguf \
    Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00002-of-00002.gguf \
    --local-dir "${GGUF_DIR}"
fi

printf 'MLX checkpoint: %s\n' "${MLX_DIR}"
