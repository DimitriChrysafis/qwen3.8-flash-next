#!/usr/bin/env python3
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"


def load_json(name: str) -> dict[str, Any]:
    with urllib.request.urlopen(BASE_URL + name, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object for {name}")
    return payload


def main() -> int:
    cfg = load_json("config.json")
    idx = load_json("model.safetensors.index.json")
    tok_cfg = load_json("tokenizer_config.json")
    gen_cfg = load_json("generation_config.json")
    text = cfg["text_config"]
    summary = {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "text_model_type": text.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_experts": text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "layer_types_head": text.get("layer_types", [])[:8],
        "total_size_bytes": idx.get("metadata", {}).get("total_size"),
        "num_weight_shards": len(set(idx.get("weight_map", {}).values())),
        "tokenizer_chat_template": tok_cfg.get("chat_template", "")[:180],
        "default_generation": gen_cfg,
    }
    out = Path(__file__).resolve().parents[1] / "artifacts/official_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
