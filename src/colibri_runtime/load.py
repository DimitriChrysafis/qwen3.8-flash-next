from __future__ import annotations

import glob
import json
from pathlib import Path

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
):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    index = SafeTensorIndex(path, workers=io_workers)
    quant = config.get("quantization") or {}
    group_size = int(quant.get("group_size", 64))
    bits = int(quant.get("bits", 4))
    ple_layer = args.text.ple_layer_ids[0] - 1
    first_ple = (f"model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding."
                 "shard_0.weight")
    if first_ple not in index.tensors:
        index.close()
        raise KeyError(f"PLE shard tensor not found: {first_ple}")
    rows_per_shard = index.tensors[first_ple].shape[0]
    ple_q = quant.get(first_ple.removesuffix(".weight"), {})
    experts = ExpertStore(index, expert_budget_bytes, args.text.num_hidden_layers,
                          args.text.num_experts, group_size, bits, cache_policy,
                          prefetch_experts)
    ple = PLEStore(index, ple_budget_bytes, ple_layer, args.text.split_ngram_parts,
                   rows_per_shard, int(ple_q.get("group_size", 32)),
                   int(ple_q.get("bits", bits)), cache_policy, prefetch_ple)
    model = Model(args, experts, ple)
    weights = {}
    lazy_bytes = 0
    for file in sorted(glob.glob(str(path / "*.safetensors"))):
        shard = mx.load(file)
        for name, value in shard.items():
            if _is_streamed(name) or name.startswith("vision_tower.") or name.startswith("model.visual."):
                continue
            weights[name] = value
            info = index.tensors.get(name)
            if info is not None:
                lazy_bytes += info.nbytes
    weights = model.sanitize(weights)

    def quant_predicate(module_path, module):
        scales_name = f"{module_path}.scales"
        if scales_name not in weights:
            return False
        spec = quant.get(module_path, {})
        return {"group_size": int(spec.get("group_size", group_size)),
                "bits": int(spec.get("bits", bits))}

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=quant_predicate)
    model.load_weights(list(weights.items()), strict=True)
    if evaluate:
        mx.eval(model.parameters())
    model.eval()
    model.load_stats = {
        "always_used_lazy_bytes": lazy_bytes,
        "streamed_tensor_bytes": sum(info.nbytes for name, info in index.tensors.items() if _is_streamed(name)),
        "indexed_tensors": len(index.tensors),
    }
    return model, config


def load(path: str | Path, **kwargs):
    from mlx_lm.utils import load_tokenizer
    model, config = load_model(path, **kwargs)
    tokenizer = load_tokenizer(Path(path), eos_token_ids=config.get("text_config", config).get("eos_token_id"))
    return model, tokenizer
