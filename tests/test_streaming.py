from __future__ import annotations

import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import save_file

from colibri_runtime.storage import SafeTensorIndex
from colibri_runtime.streaming import ExpertStore, PLEStore


def qstack(array, group_size=32, bits=4):
    shape = array.shape
    q, s, b = mx.quantize(mx.array(array.reshape(-1, shape[-1])), group_size=group_size, bits=bits)
    return (np.array(q).reshape(*shape[:-1], q.shape[-1]),
            np.array(s).reshape(*shape[:-1], s.shape[-1]),
            np.array(b).reshape(*shape[:-1], b.shape[-1]))


def tiny_file(path):
    rng = np.random.default_rng(4)
    tensors = {}
    experts = {}
    for proj, shape in {"gate": (4, 32, 32), "up": (4, 32, 32), "down": (4, 32, 32)}.items():
        dense = rng.normal(size=shape).astype(np.float32) / 8
        experts[proj] = dense
        q, s, b = qstack(dense)
        base = f"language_model.model.layers.0.mlp.switch_mlp.{proj}_proj"
        tensors[f"{base}.weight"] = q
        tensors[f"{base}.scales"] = s
        tensors[f"{base}.biases"] = b
    ple = rng.normal(size=(16, 32)).astype(np.float32)
    for shard in range(2):
        q, s, b = qstack(ple[shard * 8:(shard + 1) * 8])
        base = f"language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}"
        tensors[f"{base}.weight"] = q
        tensors[f"{base}.scales"] = s
        tensors[f"{base}.biases"] = b
    save_file(tensors, path / "model.safetensors")
    return experts, ple


def test_header_and_exact_row_ranges(tmp_path):
    _, ple = tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path)
    name = "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    info = index.tensors[name]
    assert len(index._fds) == 1
    rows = index.read_rows(name, [1, 2, 5])
    assert rows.shape == (3, info.shape[1])
    stats = index.snapshot()
    assert stats["bytes_read"] == 3 * info.row_bytes
    assert stats["pread_calls"] == 2
    with open(info.path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        assert info.offset >= 8 + header_len
    index.close()


def test_row_reads_restore_order_and_deduplicate(tmp_path):
    _, _ = tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path)
    name = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    info = index.tensors[name]
    rows = index.read_rows_numpy(name, [3, 1, 3, 2])
    expected = rows[[1, 3, 0]]
    np.testing.assert_array_equal(rows, expected[[2, 0, 2, 1]])
    stats = index.snapshot()
    assert stats["bytes_read"] == 3 * info.row_bytes
    assert stats["pread_calls"] == 1
    assert stats["rows_read"] == 4
    index.close()


def test_expert_slice_dequant_cache_and_prefetch(tmp_path):
    experts, _ = tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path, workers=3)
    one_bytes = sum(index.tensors[f"model.layers.0.mlp.switch_mlp.{p}_proj.{suffix}"].row_bytes
                    for p in ("gate", "up", "down") for suffix in ("weight", "scales", "biases"))
    store = ExpertStore(index, budget_bytes=one_bytes * 2, num_layers=1, num_experts=4,
                        group_size=32, bits=4, prefetch=2)
    future = store.request(0, 2, prefetch=True)
    future.result()
    store.prepare(0, [])
    assert store.snapshot()["ready_prefetch"] == 1
    expert = store.get(0, 2)
    x = mx.array(np.random.default_rng(2).normal(size=(3, 32)).astype(np.float32))
    out = np.array(expert(x))
    qg = mx.quantize(mx.array(experts["gate"][2]), group_size=32, bits=4)
    qu = mx.quantize(mx.array(experts["up"][2]), group_size=32, bits=4)
    qd = mx.quantize(mx.array(experts["down"][2]), group_size=32, bits=4)
    expected = mx.quantized_matmul(
        nn.silu(mx.quantized_matmul(x, *qg, group_size=32, bits=4)) *
        mx.quantized_matmul(x, *qu, group_size=32, bits=4), *qd, group_size=32, bits=4)
    np.testing.assert_allclose(out, np.array(expected), rtol=1e-5, atol=1e-5)
    assert store.snapshot()["prefetch_hits"] == 1
    assert store.snapshot()["loaded_bytes"] == one_bytes
    assert store.snapshot()["wait_seconds"] >= 0
    store.get(0, 0); store.get(0, 1)
    assert store.cache.snapshot()["bytes"] <= one_bytes * 2
    assert store.cache.snapshot()["evictions"] >= 1
    index.close()


def test_expert_batch_coalesces_contiguous_reads(tmp_path):
    tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path, workers=3)
    one_bytes = sum(index.tensors[f"model.layers.0.mlp.switch_mlp.{p}_proj.{suffix}"].row_bytes
                    for p in ("gate", "up", "down") for suffix in ("weight", "scales", "biases"))
    store = ExpertStore(index, budget_bytes=one_bytes * 4, num_layers=1, num_experts=4,
                        group_size=32, bits=4, prefetch=0)
    store.prepare(0, range(4))
    for expert in range(4):
        store.get(0, expert)
    assert index.snapshot()["pread_calls"] == 9
    assert store.snapshot()["loaded_bytes"] == one_bytes * 4
    store.close()
    index.close()


def test_route_frequency_counts_token_assignments(tmp_path):
    tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path)
    store = ExpertStore(index, budget_bytes=4096, num_layers=1, num_experts=4,
                        group_size=32, bits=4, prefetch=0)
    store.cache.put((0, 1), object(), 1)
    store.record_route(0, [1, 1, 1, 1, 2])
    snapshot = store.snapshot()
    assert snapshot["routes"] == 5
    assert snapshot["frequent_experts"][0] == {"layer": 0, "expert": 1, "count": 4}
    assert snapshot["cache"]["hot"] == 1
    store.close()
    index.close()


def test_ple_only_requested_rows_dequant_and_local_prefetch(tmp_path):
    _, ple = tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path, workers=3)
    row_bytes = sum(index.tensors[f"model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.{s}"].row_bytes
                    for s in ("weight", "scales", "biases"))
    store = PLEStore(index, row_bytes * 4, layer=0, n_shards=2, rows_per_shard=8,
                     group_size=32, bits=4, prefetch=8)
    gids = mx.array([[1, 9, 1, 7]])
    store.prefetch_rows(gids)
    out = np.array(store.get_rows(gids))
    expected = []
    for gid in [1, 9, 1, 7]:
        q, s, b = mx.quantize(mx.array(ple[gid:gid + 1]), group_size=32, bits=4)
        expected.append(np.array(mx.dequantize(q, s, b, group_size=32, bits=4))[0])
    np.testing.assert_allclose(out.reshape(4, 32), np.stack(expected), rtol=1e-5, atol=1e-5)
    snap = store.snapshot()
    assert snap["rows_requested"] == 4
    assert snap["unique_rows"] == 3
    assert snap["loaded_bytes"] == 3 * row_bytes
    assert snap["prefetch_submitted"] == 3
    assert snap["prefetch_hits"] == 3
    assert snap["cache"]["bytes"] <= 4 * row_bytes
    index.close()


def test_ple_reads_exactly_sixteen_required_rows(tmp_path):
    tiny_file(tmp_path)
    index = SafeTensorIndex(tmp_path, workers=2)
    row_bytes = sum(index.tensors[f"model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.{s}"].row_bytes
                    for s in ("weight", "scales", "biases"))
    store = PLEStore(index, row_bytes * 8, layer=0, n_shards=2, rows_per_shard=8,
                     group_size=32, bits=4, prefetch=0)
    before = index.snapshot()["bytes_read"]
    out = store.get_rows(mx.arange(16).reshape(1, 1, 16))
    mx.eval(out)
    assert out.shape == (1, 1, 16, 32)
    assert index.snapshot()["bytes_read"] - before == 16 * row_bytes
    assert store.snapshot()["unique_rows"] == 16
    assert store.snapshot()["cache"]["bytes"] <= 8 * row_bytes
    store.close()
    index.close()
