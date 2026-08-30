from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .model import Model, ModelArgs
from .storage import SafeTensorIndex
from .streaming import ExpertStore, PLEStore


def _is_streamed(name: str) -> bool:
    return ".mlp.switch_mlp." in name or ".ple.ple_embedding.ngram_embedding.shard_" in name


def load_model(
    path: str | Path,
    expert_budget_bytes: int = 8 << 30,
    ple_budget_bytes: int = 1 << 30,
    cache_policy: str = "adaptive",
    io_workers: int = 6,
    prefetch_experts: int = 2,
    prefetch_ple: int = 8,
    evaluate: bool = False,
) -> tuple[Model, dict[str, Any]]:
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    args = ModelArgs.from_dict(config)
    index = SafeTensorIndex(path, workers=io_workers)
    experts = None
    ple = None
    model = None
    try:
        quant = config.get("quantization") or {}
        if not isinstance(quant, dict):
            raise ValueError("quantization must be an object")
        group_size = int(quant.get("group_size", 64))
        bits = int(quant.get("bits", 4))
        ple_layer = args.text.ple_layer_ids[0] - 1
        first_ple = f"model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.shard_0.weight"
        if first_ple not in index.tensors:
            raise KeyError(f"PLE shard tensor not found: {first_ple}")
        ple_shape = index.tensors[first_ple].shape
        if not ple_shape:
            raise ValueError(f"PLE shard tensor must have rows: {first_ple}")
        rows_per_shard = ple_shape[0]
        ple_q = quant.get(first_ple.removesuffix(".weight"), {})
        if not isinstance(ple_q, dict):
            raise ValueError(f"quantization spec must be an object: {first_ple}")
        experts = ExpertStore(
            index,
            expert_budget_bytes,
            args.text.num_hidden_layers,
            args.text.num_experts,
            group_size,
            bits,
            cache_policy,
            prefetch_experts,
        )
        ple = PLEStore(
            index,
            ple_budget_bytes,
            ple_layer,
            args.text.split_ngram_parts,
            rows_per_shard,
            int(ple_q.get("group_size", 32)),
            int(ple_q.get("bits", bits)),
            cache_policy,
            prefetch_ple,
        )
        model = Model(args, experts, ple)
        weights = {}
        lazy_bytes = 0
        for file in sorted(path.glob("*.safetensors")):
            shard = mx.load(str(file))
            for name, value in shard.items():
                if (
                    _is_streamed(name)
                    or name.startswith("vision_tower.")
                    or name.startswith("model.visual.")
                ):
                    continue
                weights[name] = value
                info = index.tensors.get(name.removeprefix("language_model."))
                if info is not None:
                    lazy_bytes += info.nbytes
        weights = model.sanitize(weights)

        def quant_predicate(module_path, _module):
            scales_name = f"{module_path}.scales"
            if scales_name not in weights:
                return False
            spec = quant.get(module_path, {})
            if not isinstance(spec, dict):
                raise ValueError(f"quantization spec must be an object: {module_path}")
            return {
                "group_size": int(spec.get("group_size", group_size)),
                "bits": int(spec.get("bits", bits)),
            }

        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=quant_predicate)
        model.load_weights(list(weights.items()), strict=True)
        if evaluate:
            mx.eval(model.parameters())
        model.eval()
        model.load_stats = {
            "always_used_lazy_bytes": lazy_bytes,
            "streamed_tensor_bytes": sum(
                info.nbytes for name, info in index.tensors.items() if _is_streamed(name)
            ),
            "indexed_tensors": len(index.tensors),
        }
        return model, config
    except BaseException:
        if model is not None:
            model.close()
        else:
            if experts is not None:
                experts.close()
            if ple is not None:
                ple.close()
            index.close()
        raise


def load(path: str | Path, **kwargs: Any) -> tuple[Model, Any]:
    from mlx_lm.utils import load_tokenizer

    model, config = load_model(path, **kwargs)
    try:
        tokenizer_config = config.get("text_config", config)
        if not isinstance(tokenizer_config, dict):
            raise ValueError("text_config must contain an object")
        tokenizer = load_tokenizer(Path(path), eos_token_ids=tokenizer_config.get("eos_token_id"))
    except BaseException:
        model.close()
        raise
    return model, tokenizer
