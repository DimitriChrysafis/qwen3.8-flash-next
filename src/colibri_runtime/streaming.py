# Copyright 2026 The Qwen Team, The HuggingFace Inc. team, and PipeNetwork contributors.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .expert_cache import ExpertCache
from .storage import SafeTensorIndex


def _to_mlx(value, dtype_name: str):
    if value is None or isinstance(value, mx.array):
        return value
    out = mx.array(value)
    return out.view(mx.bfloat16) if dtype_name == "BF16" else out


@dataclass
class StreamStats:
    loads: int = 0
    load_seconds: float = 0.0
    wait_seconds: float = 0.0
    loaded_bytes: int = 0
    prefetch_submitted: int = 0
    prefetch_hits: int = 0
    prefetch_wasted: int = 0
    routes: int = 0
    rows_requested: int = 0
    unique_rows: int = 0


@dataclass
class LinearSlice:
    weight: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None
    group_size: int = 64
    bits: int = 4

    def __call__(self, x: mx.array) -> mx.array:
        if self.scales is None:
            return x @ self.weight.T
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                   group_size=self.group_size, bits=self.bits)


@dataclass
class ExpertSlice:
    gate: LinearSlice
    up: LinearSlice
    down: LinearSlice
    nbytes: int

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class ExpertStore:
    def __init__(
        self,
        index: SafeTensorIndex,
        budget_bytes: int,
        num_layers: int,
        num_experts: int,
        group_size: int = 64,
        bits: int = 4,
        policy: str = "adaptive",
        prefetch: int = 2,
        pinned: Iterable[tuple[int, int]] = (),
    ):
        self.index = index
        self.cache = ExpertCache(max_bytes=budget_bytes, policy=policy)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.group_size = group_size
        self.bits = bits
        self.prefetch_count = max(0, prefetch)
        self.stats = StreamStats()
        self._futures: dict[tuple[int, int], Future] = {}
        self._future_prefetch: set[tuple[int, int]] = set()
        self._lock = threading.RLock()
        self._freq: defaultdict[int, Counter] = defaultdict(Counter)
        self._recent: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=8))
        self._transitions: defaultdict[tuple[int, int], Counter] = defaultdict(Counter)
        self._pinned = set(pinned)
        self._detect_layout()
        self._expert_bytes = sum(
            self.index.tensors[self._name(0, proj, suffix)].row_bytes
            for proj in ("gate", "up", "down")
            for suffix in ("weight", "scales", "biases")
            if self._name(0, proj, suffix) in self.index.tensors
        )
        self._max_speculative = max(1, budget_bytes // max(1, self._expert_bytes))
        self.io_window = max(1, min(getattr(index.executor, "_max_workers", 1),
                                    self._max_speculative))

    def _base(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.switch_mlp"

    def _name(self, layer: int, proj: str, suffix: str) -> str:
        return f"{self._base(layer)}.{proj}_proj.{suffix}"

    def _detect_layout(self) -> None:
        required = [self._name(0, p, "weight") for p in ("gate", "up", "down")]
        missing = [x for x in required if x not in self.index.tensors]
        if missing:
            raise KeyError(f"streamed expert tensors missing: {missing}")
        shape = self.index.tensors[required[0]].shape
        if shape[0] != self.num_experts:
            raise ValueError(f"expert count {shape[0]} != config {self.num_experts}")

    def _read_linear_many(self, layer: int, experts: tuple[int, ...], proj: str):
        weight_name = self._name(layer, proj, "weight")
        weights = self.index.read_rows_numpy(weight_name, experts)
        row_bytes = self.index.tensors[weight_name].row_bytes
        scales_name = self._name(layer, proj, "scales")
        biases_name = self._name(layer, proj, "biases")
        if scales_name not in self.index.tensors:
            return [LinearSlice(weight) for weight in weights], row_bytes
        scales = self.index.read_rows_numpy(scales_name, experts)
        biases = self.index.read_rows_numpy(biases_name, experts) if biases_name in self.index.tensors else None
        row_bytes += self.index.tensors[scales_name].row_bytes
        if biases is not None:
            row_bytes += self.index.tensors[biases_name].row_bytes
        return [
            LinearSlice(weights[i], scales[i], None if biases is None else biases[i],
                        self.group_size, self.bits)
            for i in range(len(experts))
        ], row_bytes

    def _load_many(self, layer: int, experts: tuple[int, ...]):
        t0 = time.perf_counter()
        gates, ng = self._read_linear_many(layer, experts, "gate")
        ups, nu = self._read_linear_many(layer, experts, "up")
        downs, nd = self._read_linear_many(layer, experts, "down")
        nbytes = ng + nu + nd
        values = {
            (layer, expert): ExpertSlice(gates[i], ups[i], downs[i], nbytes)
            for i, expert in enumerate(experts)
        }
        with self._lock:
            self.stats.loads += len(experts)
            self.stats.loaded_bytes += len(experts) * nbytes
            self.stats.load_seconds += time.perf_counter() - t0
        return values

    def request_many(self, layer: int, experts: Iterable[int], prefetch: bool = False):
        layer = int(layer)
        keys = []
        with self._lock:
            available = self._max_speculative - len(self._future_prefetch) if prefetch else None
            for expert in dict.fromkeys(map(int, experts)):
                key = (layer, expert)
                if self.cache.peek(key) is None and key not in self._futures:
                    if available is not None and len(keys) >= max(0, available):
                        break
                    keys.append(key)
            if not keys:
                return {}
            ids = tuple(expert for _, expert in keys)
            group_future = self.index.executor.submit(self._load_many, layer, ids)
            children = {}
            for key in keys:
                child = Future()
                self._futures[key] = child
                children[key] = child
            if prefetch:
                self._future_prefetch.update(keys)
                self.stats.prefetch_submitted += len(keys)

        def finish(future):
            try:
                values = future.result()
                for key, value in values.items():
                    children[key].set_result(value)
            except BaseException as exc:
                for child in children.values():
                    child.set_exception(exc)

        group_future.add_done_callback(finish)
        return children

    def request(self, layer: int, expert: int, prefetch: bool = False) -> Future | None:
        key = (int(layer), int(expert))
        if self.cache.peek(key) is not None:
            return None
        with self._lock:
            existing = self._futures.get(key)
        if existing is not None:
            return existing
        child = self.request_many(layer, [expert], prefetch).get(key)
        if child is not None:
            return child
        with self._lock:
            return self._futures.get(key)

    def prepare(self, layer: int, experts: Iterable[int]) -> None:
        experts = list(dict.fromkeys(map(int, experts)))
        for start in range(0, len(experts), 32):
            self.request_many(layer, experts[start:start + 32])

    def get(self, layer: int, expert: int) -> ExpertSlice:
        key = (int(layer), int(expert))
        value = self.cache.get(key)
        if value is not None:
            return value
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            future = self.request(*key)
        assert future is not None
        wait_started = time.perf_counter()
        value = future.result()
        waited = time.perf_counter() - wait_started
        with self._lock:
            self.stats.wait_seconds += waited
        for proj, name in ((value.gate, "gate"), (value.up, "up"), (value.down, "down")):
            proj.weight = _to_mlx(proj.weight, self.index.tensors[self._name(layer, name, "weight")].dtype)
            if proj.scales is not None:
                proj.scales = _to_mlx(proj.scales, self.index.tensors[self._name(layer, name, "scales")].dtype)
            if proj.biases is not None:
                proj.biases = _to_mlx(proj.biases, self.index.tensors[self._name(layer, name, "biases")].dtype)
        with self._lock:
            self._futures.pop(key, None)
            was_prefetch = key in self._future_prefetch
            self._future_prefetch.discard(key)
            if was_prefetch:
                self.stats.prefetch_hits += 1
        count = self._freq[layer][expert]
        self.cache.put(key, value, value.nbytes, pin=key in self._pinned, hot=count >= 4)
        return value

    def record_route(self, layer: int, experts: Iterable[int]) -> None:
        routed = tuple(map(int, experts))
        selected = tuple(sorted(set(routed)))
        with self._lock:
            previous = self._recent[layer][-1] if self._recent[layer] else ()
            for old in previous:
                self._transitions[(layer, old)].update(selected)
            self._freq[layer].update(routed)
            self._recent[layer].append(selected)
            self.stats.routes += len(routed)
            for expert in selected:
                self.cache.mark_hot((layer, expert), self._freq[layer][expert] >= 4)

    def prefetch_predictions(self, layer: int, selected: Iterable[int]) -> None:
        if not self.prefetch_count:
            return
        candidates = Counter()
        with self._lock:
            for expert in selected:
                candidates.update(self._transitions[(layer, int(expert))])
            if not candidates:
                candidates.update(self._freq[layer])
            if layer + 1 < self.num_layers:
                candidates.update({e: n * 2 for e, n in self._freq[layer + 1].items()})
        for expert, _ in candidates.most_common(self.prefetch_count):
            self.request(layer, expert, prefetch=True)
        if layer + 1 < self.num_layers:
            with self._lock:
                next_ids = self._freq[layer + 1].most_common(self.prefetch_count)
            for expert, _ in next_ids:
                self.request(layer + 1, expert, prefetch=True)

    def snapshot(self) -> dict:
        with self._lock:
            stale = sum(f.done() for f in self._futures.values())
            out = asdict(self.stats)
            out["inflight"] = len(self._futures)
            out["completed_prefetch"] = stale
            frequent = Counter({(layer, expert): count for layer, counts in self._freq.items()
                                for expert, count in counts.items()})
            out["frequent_experts"] = [
                {"layer": layer, "expert": expert, "count": count}
                for (layer, expert), count in frequent.most_common(20)
            ]
        out["cache"] = self.cache.snapshot()
        return out

    def close(self) -> None:
        with self._lock:
            futures = list(self._futures.values())
            self.stats.prefetch_wasted += len(self._future_prefetch)
        for future in futures:
            future.cancel()


@dataclass
class PLERow:
    weight: mx.array
    scales: mx.array | None
    biases: mx.array | None
    nbytes: int


class PLEStore:
    def __init__(
        self,
        index: SafeTensorIndex,
        budget_bytes: int,
        layer: int,
        n_shards: int,
        rows_per_shard: int,
        group_size: int = 32,
        bits: int = 4,
        policy: str = "adaptive",
        prefetch: int = 8,
    ):
        self.index = index
        self.cache = ExpertCache(max_bytes=budget_bytes, policy=policy)
        self.layer = layer
        self.n_shards = n_shards
        self.rows_per_shard = rows_per_shard
        self.group_size = group_size
        self.bits = bits
        self.prefetch_count = max(0, prefetch)
        self.stats = StreamStats()
        self._futures: dict[tuple[int, int], Future] = {}
        self._lock = threading.RLock()
        self._recent: deque[int] = deque(maxlen=256)
        if self._name(0, "weight") not in index.tensors:
            raise KeyError(f"streamed PLE tensor missing: {self._name(0, 'weight')}")
        self._row_bytes = sum(
            index.tensors[self._name(0, suffix)].row_bytes
            for suffix in ("weight", "scales", "biases")
            if self._name(0, suffix) in index.tensors
        )
        self._max_speculative = max(1, budget_bytes // max(1, self._row_bytes))

    def _name(self, shard: int, suffix: str) -> str:
        return (f"model.layers.{self.layer}.ple.ple_embedding.ngram_embedding."
                f"shard_{shard}.{suffix}")

    def _load_rows(self, shard: int, rows: tuple[int, ...]) -> dict[tuple[int, int], PLERow]:
        t0 = time.perf_counter()
        wn = self._name(shard, "weight")
        sn = self._name(shard, "scales")
        bn = self._name(shard, "biases")
        weights = self.index.read_rows_numpy(wn, rows)
        scales = self.index.read_rows_numpy(sn, rows) if sn in self.index.tensors else None
        biases = self.index.read_rows_numpy(bn, rows) if bn in self.index.tensors else None
        row_bytes = self.index.tensors[wn].row_bytes
        if scales is not None:
            row_bytes += self.index.tensors[sn].row_bytes
        if biases is not None:
            row_bytes += self.index.tensors[bn].row_bytes
        out = {}
        for i, row in enumerate(rows):
            out[(shard, row)] = PLERow(weights[i], None if scales is None else scales[i],
                                        None if biases is None else biases[i], row_bytes)
        with self._lock:
            self.stats.loads += len(rows)
            self.stats.loaded_bytes += len(rows) * row_bytes
            self.stats.load_seconds += time.perf_counter() - t0
        return out

    def _submit_group(self, shard: int, rows: Iterable[int], prefetch: bool) -> None:
        missing = tuple(sorted(set(int(r) for r in rows if self.cache.peek((shard, int(r))) is None
                                   and (shard, int(r)) not in self._futures)))
        if prefetch:
            slots = self._max_speculative - len(self._futures)
            missing = missing[:max(0, slots)]
        if not missing:
            return
        group_future = self.index.executor.submit(self._load_rows, shard, missing)
        for row in missing:
            child: Future = Future()
            self._futures[(shard, row)] = child
        if prefetch:
            self.stats.prefetch_submitted += len(missing)

        def finish(future):
            try:
                values = future.result()
                for key, value in values.items():
                    self._futures[key].set_result((value, prefetch))
            except BaseException as exc:
                for row in missing:
                    self._futures[(shard, row)].set_exception(exc)
        group_future.add_done_callback(finish)

    def prefetch_rows(self, gids: mx.array | np.ndarray) -> None:
        if not self.prefetch_count:
            return
        if isinstance(gids, mx.array):
            mx.eval(gids)
            ids = np.asarray(gids).astype(np.int64, copy=False)
        else:
            ids = np.asarray(gids, dtype=np.int64)
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for gid in set(ids.reshape(-1).tolist()):
            shard, row = divmod(int(gid), self.rows_per_shard)
            if shard < 0 or shard >= self.n_shards:
                raise IndexError("PLE global row outside sharded table")
            groups[shard].append(row)
        with self._lock:
            for shard, rows in groups.items():
                self._submit_group(shard, rows, True)

    def get_rows(self, gids: mx.array | np.ndarray) -> mx.array:
        if isinstance(gids, mx.array):
            mx.eval(gids)
            ids = np.asarray(gids).astype(np.int64, copy=False)
        else:
            ids = np.asarray(gids, dtype=np.int64)
        flat = ids.reshape(-1)
        keys = [(int(g // self.rows_per_shard), int(g % self.rows_per_shard)) for g in flat]
        if keys and (min(k[0] for k in keys) < 0 or max(k[0] for k in keys) >= self.n_shards):
            raise IndexError("PLE global row outside sharded table")
        self.stats.rows_requested += len(keys)
        self.stats.unique_rows += len(set(keys))
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for shard, row in set(keys):
            if self.cache.peek((shard, row)) is None:
                groups[shard].append(row)
        with self._lock:
            for shard, rows in groups.items():
                self._submit_group(shard, rows, False)
        values: dict[tuple[int, int], PLERow] = {}
        for key in set(keys):
            value = self.cache.get(key)
            if value is None:
                with self._lock:
                    future = self._futures[key]
                wait_started = time.perf_counter()
                value, prefetched = future.result()
                waited = time.perf_counter() - wait_started
                with self._lock:
                    self.stats.wait_seconds += waited
                shard, _ = key
                value.weight = _to_mlx(value.weight, self.index.tensors[self._name(shard, "weight")].dtype)
                if value.scales is not None:
                    value.scales = _to_mlx(value.scales, self.index.tensors[self._name(shard, "scales")].dtype)
                if value.biases is not None:
                    value.biases = _to_mlx(value.biases, self.index.tensors[self._name(shard, "biases")].dtype)
                with self._lock:
                    self._futures.pop(key, None)
                    if prefetched:
                        self.stats.prefetch_hits += 1
                self.cache.put(key, value, value.nbytes)
            values[key] = value
        dense = []
        for key in keys:
            row = values[key]
            if row.scales is None:
                dense.append(row.weight)
            else:
                dense.append(mx.dequantize(row.weight[None], row.scales[None],
                                           None if row.biases is None else row.biases[None],
                                           group_size=self.group_size, bits=self.bits)[0])
        if not dense:
            scales_name = self._name(0, "scales")
            width = (self.index.tensors[scales_name].shape[-1] * self.group_size
                     if scales_name in self.index.tensors
                     else self.index.tensors[self._name(0, "weight")].shape[-1])
            return mx.zeros((*ids.shape, width), dtype=mx.float32)
        return mx.stack(dense).reshape(*ids.shape, -1)

    def snapshot(self) -> dict:
        out = asdict(self.stats)
        out["inflight"] = len(self._futures)
        out["cache"] = self.cache.snapshot()
        return out

    def close(self) -> None:
        with self._lock:
            self.stats.prefetch_wasted += len(self._futures)
            futures = list(self._futures.values())
        for future in futures:
            future.cancel()
