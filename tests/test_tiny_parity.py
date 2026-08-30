from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from colibri_runtime.generation import generate_tokens, metrics
from colibri_runtime.load import load_model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CFG = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    hc_count=4,
    hc_lowrank=32,
    ple_layer_ids=[1],
    ple_embed_dim=64,
    ple_conv_kernel_size=2,
    ngram_size=3,
    heads_per_ngram=1,
    ngram_vocab_size_base=31,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=2,
    seed=1234,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=4,
    indexer_compress_ratio=2,
    output_gate_type="sigmoid",
    eos_token_id=1,
    bos_token_id=1,
    rope_parameters={
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
        "rope_type": "default",
    },
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)
CENTERED = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
    "ple.norm_conv.weight",
)


def build_checkpoint(path):
    torch.manual_seed(11)
    config = transformers.Qwen4ExpTextConfig(**CFG)
    reference = transformers.Qwen4ExpForCausalLM(config).eval()
    tensors = {}
    for key, value in reference.state_dict().items():
        array = value.detach().numpy().copy()
        if key.endswith(".mlp.experts.gate_up_proj"):
            base = "language_model." + key[: -len("experts.gate_up_proj")]
            tensors[base + "switch_mlp.gate_proj.weight"] = array[
                :, : CFG["moe_intermediate_size"]
            ].copy()
            tensors[base + "switch_mlp.up_proj.weight"] = array[
                :, CFG["moe_intermediate_size"] :
            ].copy()
            continue
        if key.endswith(".mlp.experts.down_proj"):
            base = "language_model." + key[: -len("experts.down_proj")]
            tensors[base + "switch_mlp.down_proj.weight"] = array
            continue
        if key.endswith("ple_embedding.ngram_embedding.weight"):
            rows = array.shape[0] // CFG["split_ngram_parts"]
            for shard in range(CFG["split_ngram_parts"]):
                name = key.replace(
                    "ngram_embedding.weight", f"ngram_embedding.shard_{shard}.weight"
                )
                tensors["language_model." + name] = array[shard * rows : (shard + 1) * rows]
            continue
        if "conv1d.weight" in key and array.ndim == 3:
            array = array.transpose(0, 2, 1)
        if key.endswith(CENTERED):
            array += 1.0
        tensors["language_model." + key] = array
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": CFG,
                "vision_config": {},
            }
        )
    )
    return reference


def as_numpy(value):
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def test_full_streamed_tiny_model_matches_transformers(tmp_path):
    reference = build_checkpoint(tmp_path)
    streamed, _ = load_model(
        tmp_path,
        expert_budget_bytes=1 << 20,
        ple_budget_bytes=1 << 20,
        io_workers=4,
        prefetch_experts=1,
        prefetch_ple=1,
    )
    ids = torch.tensor([[2, 3, 4, 1, 5, 6]])
    with torch.no_grad():
        expected = reference(ids).logits.float().numpy()
    actual = as_numpy(streamed(mx.array(ids.numpy())))
    assert np.max(np.abs(actual - expected)) < 5e-7

    cache = streamed.make_cache()
    pieces = [
        as_numpy(streamed(mx.array(ids[:, i : i + 1].numpy()), cache=cache))
        for i in range(ids.shape[1])
    ]
    assert np.max(np.abs(np.concatenate(pieces, axis=1) - actual)) < 5e-7

    cache = streamed.make_cache()
    chunked = np.concatenate(
        [
            as_numpy(streamed(mx.array(ids[:, :3].numpy()), cache=cache)),
            as_numpy(streamed(mx.array(ids[:, 3:].numpy()), cache=cache)),
        ],
        axis=1,
    )
    np.testing.assert_allclose(chunked, actual, rtol=2e-4, atol=2e-4)
    assert streamed.expert_store.snapshot()["routes"] > 0
    assert streamed.ple_store.snapshot()["rows_requested"] > 0
    generated, timing = generate_tokens(streamed, [2, 3, 4], max_tokens=2)
    report = metrics(streamed, 3, len(generated), timing)
    assert len(generated) == 2
    assert report["disk"]["bytes_read"] > 0
    streamed.expert_store.close()
    streamed.ple_store.close()
    streamed.expert_store.index.close()
