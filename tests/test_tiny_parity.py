from __future__ import annotations

import json
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, "/private/tmp/qwen38-reference")
from qwen38_flash_next_mlx import qwen4_exp as corrected

from colibri_runtime.generation import generate_tokens, metrics
from colibri_runtime.load import load_model


CFG = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1, linear_num_value_heads=2,
    linear_key_head_dim=8, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=32,
    ple_layer_ids=[1], ple_embed_dim=64, ple_conv_kernel_size=2,
    ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=31,
    make_ngram_vocab_size_divisible_by=8, split_ngram_parts=2, seed=1234,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
    indexer_budget=4, indexer_compress_ratio=2, output_gate_type="sigmoid",
    eos_token_id=1,
    rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    rms_norm_eps=1e-6, tie_word_embeddings=False,
)


def build_checkpoint(path):
    mx.random.seed(11)
    args = corrected.ModelArgs.from_dict({"text_config": CFG, "vision_config": {}})
    reference = corrected.Model(args)
    reference.model.layers[0].ple.ple_embedding.layer_multipliers = mx.array([3, 5, 7], dtype=mx.int64)
    nn.quantize(reference, group_size=32, bits=4,
                class_predicate=lambda _, module: hasattr(module, "to_quantized"))
    mx.eval(reference.parameters())
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(reference.parameters())))
    config = {"model_type": "qwen4_exp", "text_config": CFG, "vision_config": {},
              "quantization": {"group_size": 32, "bits": 4}}
    (path / "config.json").write_text(json.dumps(config))
    return reference


def as_numpy(value):
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def test_full_streamed_tiny_model_and_incremental_parity(tmp_path):
    reference = build_checkpoint(tmp_path)
    streamed, _ = load_model(tmp_path, expert_budget_bytes=4096, ple_budget_bytes=2048,
                             io_workers=4, prefetch_experts=1, prefetch_ple=1)
    ids = mx.array([[2, 3, 4, 1, 5, 6]], dtype=mx.int32)
    expected = as_numpy(reference(ids))
    actual = as_numpy(streamed(ids))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        np.array(streamed.model.layers[0].ple.ple_embedding.layer_multipliers),
        np.array([3, 5, 7]),
    )
    raw_name = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"
    shifted = streamed.sanitize({raw_name: mx.zeros(128)})
    np.testing.assert_array_equal(np.array(shifted[raw_name.replace("model.language_model.", "model.")]),
                                  np.ones(128))

    cache = streamed.make_cache()
    pieces = [as_numpy(streamed(ids[:, i:i + 1], cache=cache)) for i in range(ids.shape[1])]
    incremental = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(incremental, actual, rtol=2e-5, atol=2e-5)

    cache = streamed.make_cache()
    chunked = np.concatenate([
        as_numpy(streamed(ids[:, :3], cache=cache)),
        as_numpy(streamed(ids[:, 3:], cache=cache)),
    ], axis=1)
    np.testing.assert_allclose(chunked, actual, rtol=2e-5, atol=2e-5)
    assert streamed.expert_store.snapshot()["cache"]["evictions"] > 0
    assert streamed.expert_store.snapshot()["prefetch_submitted"] > 0
    assert streamed.ple_store.snapshot()["rows_requested"] > 0
    assert streamed.expert_store.index.snapshot()["bytes_read"] > 0
    generated, timing = generate_tokens(streamed, [2, 3, 4], max_tokens=2,
                                         temperature=0.0, eos_token_id=None)
    report = metrics(streamed, 3, len(generated), timing)
    assert len(generated) == 2
    assert report["generated_tokens"] == 2
    assert report["disk"]["bytes_read"] > 0
    streamed.expert_store.close()
    streamed.ple_store.close()
    streamed.expert_store.index.close()
