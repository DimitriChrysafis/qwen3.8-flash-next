# Copyright 2026 The Qwen Team, The HuggingFace Inc. team, and PipeNetwork contributors.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from operator import index as to_index
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .expert_cache import ExpertCache
from .storage import SafeTensorIndex

ExpertKey = tuple[int, int]
RowKey = tuple[int, int]


def _to_mlx(value: Any, dtype_name: str) -> Any:
    if value is None or isinstance(value, mx.array):
        return value
    out = mx.array(value)
    return out.view(mx.bfloat16) if dtype_name == "BF16" else out


@dataclass(slots=True)
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


@dataclass(slots=True)
class LinearSlice:
    weight: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None
    group_size: int = 64
    bits: int = 4

    def __call__(self, x: mx.array) -> mx.array:
        if self.scales is None:
            return x @ self.weight.T
        return mx.quantized_matmul(
            x, self.weight, self.scales, self.biases, group_size=self.group_size, bits=self.bits
        )


@dataclass(slots=True)
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
    ) -> None:
        if budget_bytes < 1:
            raise ValueError("budget_bytes must be positive")
        if num_layers < 1 or num_experts < 1:
            raise ValueError("num_layers and num_experts must be positive")
        if group_size < 1 or bits < 1:
            raise ValueError("group_size and bits must be positive")
        self.index = index
        self.cache = ExpertCache(max_bytes=budget_bytes, policy=policy, partitions=num_layers)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.group_size = group_size
        self.bits = bits
        self.prefetch_count = max(0, prefetch)
        self.stats = StreamStats()
        self._futures: dict[tuple[int, int], Future] = {}
        self._future_prefetch: set[tuple[int, int]] = set()
        self._prefetched_ready: set[tuple[int, int]] = set()
        self._lock = threading.RLock()
        self._freq: defaultdict[int, Counter] = defaultdict(Counter)
        self._recent: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=8))
        self._transitions: defaultdict[tuple[int, int], Counter] = defaultdict(Counter)
        self._pinned = {self._validate_key(layer, expert) for layer, expert in pinned}
        self._closed = False
        self._detect_layout()
        self._expert_bytes = max(self._layer_expert_bytes(layer) for layer in range(num_layers))
        if self._expert_bytes < 1:
            raise ValueError("expert tensors must contain at least one byte per expert")
        if budget_bytes < self._expert_bytes:
            raise ValueError("budget_bytes must hold at least one expert")
        cache_slots = budget_bytes // self._expert_bytes
        workers = getattr(index.executor, "_max_workers", 1)
        self._max_speculative = min(cache_slots, max(workers * 2, self.prefetch_count * num_layers))
        self.io_window = max(1, min(workers, cache_slots))

    def _base(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.switch_mlp"

    def _name(self, layer: int, proj: str, suffix: str) -> str:
        return f"{self._base(layer)}.{proj}_proj.{suffix}"

    def _detect_layout(self) -> None:
        for layer in range(self.num_layers):
            required = [
                self._name(layer, projection, "weight") for projection in ("gate", "up", "down")
            ]
            missing = [name for name in required if name not in self.index.tensors]
            if missing:
                raise KeyError(f"streamed expert tensors missing: {missing}")
            for name in required:
                shape = self.index.tensors[name].shape
                if not shape or shape[0] != self.num_experts:
                    count = shape[0] if shape else 0
                    raise ValueError(
                        f"expert count {count} != config {self.num_experts} for {name}"
                    )
            for projection in ("gate", "up", "down"):
                scales = self.index.tensors.get(self._name(layer, projection, "scales"))
                biases = self.index.tensors.get(self._name(layer, projection, "biases"))
                if biases is not None and scales is None:
                    raise ValueError(
                        f"expert biases require scales for layer {layer}, {projection}"
                    )
                for suffix, tensor in (("scales", scales), ("biases", biases)):
                    if tensor is not None and (
                        not tensor.shape or tensor.shape[0] != self.num_experts
                    ):
                        raise ValueError(
                            f"expert {suffix} row count does not match config for "
                            f"layer {layer}, {projection}"
                        )

    def _layer_expert_bytes(self, layer: int) -> int:
        return sum(
            self.index.tensors[self._name(layer, projection, suffix)].row_bytes
            for projection in ("gate", "up", "down")
            for suffix in ("weight", "scales", "biases")
            if self._name(layer, projection, suffix) in self.index.tensors
        )

    def _validate_key(self, layer: int, expert: int) -> ExpertKey:
        if isinstance(layer, bool) or isinstance(expert, bool):
            raise TypeError("layer and expert must be integers")
        try:
            layer, expert = to_index(layer), to_index(expert)
        except TypeError as error:
            raise TypeError("layer and expert must be integers") from error
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer outside [0, {self.num_layers}): {layer}")
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert outside [0, {self.num_experts}): {expert}")
        return layer, expert

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("expert store is closed")

    def _read_linear_many(
        self, layer: int, experts: tuple[int, ...], proj: str
    ) -> tuple[list[LinearSlice], int]:
        weight_name = self._name(layer, proj, "weight")
        weights = self.index.read_rows_numpy(weight_name, experts)
        row_bytes = self.index.tensors[weight_name].row_bytes
        scales_name = self._name(layer, proj, "scales")
        biases_name = self._name(layer, proj, "biases")
        if scales_name not in self.index.tensors:
            return [LinearSlice(weight) for weight in weights], row_bytes
        scales = self.index.read_rows_numpy(scales_name, experts)
        biases = (
            self.index.read_rows_numpy(biases_name, experts)
            if biases_name in self.index.tensors
            else None
        )
        row_bytes += self.index.tensors[scales_name].row_bytes
        if biases is not None:
            row_bytes += self.index.tensors[biases_name].row_bytes
        return [
            LinearSlice(
                weights[i],
                scales[i],
                None if biases is None else biases[i],
                self.group_size,
                self.bits,
            )
            for i in range(len(experts))
        ], row_bytes

    def _load_many(self, layer: int, experts: tuple[int, ...]) -> dict[ExpertKey, ExpertSlice]:
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

    def request_many(
        self, layer: int, experts: Iterable[int], prefetch: bool = False
    ) -> dict[ExpertKey, Future]:
        self._ensure_open()
        layer = self._validate_key(layer, 0)[0]
        expert_ids = tuple(
            dict.fromkeys(self._validate_key(layer, expert)[1] for expert in experts)
        )
        keys = [(layer, expert) for expert in expert_ids]
        with self._lock:
            available = self._max_speculative - len(self._future_prefetch) if prefetch else None
            missing = []
            for key in keys:
                if self.cache.peek(key) is None and key not in self._futures:
                    if available is not None and len(missing) >= max(0, available):
                        break
                    missing.append(key)
            if not missing:
                return {}
            ids = tuple(expert for _, expert in missing)
            group_future = self.index.executor.submit(self._load_many, layer, ids)
            children = {}
            for key in missing:
                child = Future()
                self._futures[key] = child
                children[key] = child
            if prefetch:
                self._future_prefetch.update(missing)
                self.stats.prefetch_submitted += len(missing)

        def finish(future):
            try:
                values = future.result()
                for key, value in values.items():
                    child = children[key]
                    if not child.done():
                        child.set_result(value)
            except Exception as error:
                for child in children.values():
                    if not child.done():
                        child.set_exception(error)

        group_future.add_done_callback(finish)
        return children

    def request(self, layer: int, expert: int, prefetch: bool = False) -> Future | None:
        self._ensure_open()
        key = self._validate_key(layer, expert)
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
        self._ensure_open()
        layer = self._validate_key(layer, 0)[0]
        self._harvest_prefetches()
        experts = list(dict.fromkeys(self._validate_key(layer, expert)[1] for expert in experts))
        for start in range(0, len(experts), 32):
            self.request_many(layer, experts[start : start + 32])

    def _materialize(self, layer: int, value: ExpertSlice) -> ExpertSlice:
        for proj, name in ((value.gate, "gate"), (value.up, "up"), (value.down, "down")):
            proj.weight = _to_mlx(
                proj.weight, self.index.tensors[self._name(layer, name, "weight")].dtype
            )
            if proj.scales is not None:
                proj.scales = _to_mlx(
                    proj.scales, self.index.tensors[self._name(layer, name, "scales")].dtype
                )
            if proj.biases is not None:
                proj.biases = _to_mlx(
                    proj.biases, self.index.tensors[self._name(layer, name, "biases")].dtype
                )
        return value

    def _harvest_prefetches(self) -> None:
        with self._lock:
            completed = [
                (key, self._futures[key])
                for key in self._future_prefetch
                if self._futures[key].done()
            ]
        for key, future in completed:
            layer, expert = key
            try:
                value = self._materialize(layer, future.result())
            except Exception:
                with self._lock:
                    if self._futures.get(key) is future:
                        self._futures.pop(key, None)
                        self._future_prefetch.discard(key)
                    self.stats.prefetch_wasted += 1
                continue
            with self._lock:
                if self._closed or self._futures.get(key) is not future:
                    continue
                hot = self._freq[layer][expert] >= 4
            inserted = self.cache.put(key, value, value.nbytes, pin=key in self._pinned, hot=hot)
            with self._lock:
                if self._futures.get(key) is not future:
                    continue
                self._futures.pop(key, None)
                self._future_prefetch.discard(key)
                if inserted:
                    self._prefetched_ready.add(key)
                else:
                    self.stats.prefetch_wasted += 1
        with self._lock:
            stale = [key for key in self._prefetched_ready if self.cache.peek(key) is None]
            self._prefetched_ready.difference_update(stale)
            self.stats.prefetch_wasted += len(stale)

    def get(self, layer: int, expert: int) -> ExpertSlice:
        self._ensure_open()
        key = self._validate_key(layer, expert)
        layer, expert = key
        value = self.cache.get(key)
        if value is not None:
            with self._lock:
                if key in self._prefetched_ready:
                    self._prefetched_ready.discard(key)
                    self.stats.prefetch_hits += 1
            return value
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            future = self.request(*key)
        if future is None:
            value = self.cache.get(key)
            if value is None:
                raise RuntimeError(f"expert request disappeared for layer={layer}, expert={expert}")
            return value
        wait_started = time.perf_counter()
        try:
            value = future.result()
        except Exception:
            with self._lock:
                if self._futures.get(key) is future:
                    self._futures.pop(key, None)
                    self._future_prefetch.discard(key)
            raise
        waited = time.perf_counter() - wait_started
        with self._lock:
            self.stats.wait_seconds += waited
        try:
            value = self._materialize(layer, value)
        except Exception:
            with self._lock:
                if self._futures.get(key) is future:
                    self._futures.pop(key, None)
                    self._future_prefetch.discard(key)
            raise
        with self._lock:
            if self._futures.get(key) is not future:
                cached = self.cache.get(key)
                return value if cached is None else cached
            self._futures.pop(key, None)
            was_prefetch = key in self._future_prefetch
            self._future_prefetch.discard(key)
            if was_prefetch:
                self.stats.prefetch_hits += 1
        with self._lock:
            count = self._freq[layer][expert]
        self.cache.put(key, value, value.nbytes, pin=key in self._pinned, hot=count >= 4)
        return value

    def record_route(self, layer: int, experts: Iterable[int]) -> None:
        self._ensure_open()
        layer = self._validate_key(layer, 0)[0]
        routed = tuple(self._validate_key(layer, expert)[1] for expert in experts)
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
        self._ensure_open()
        layer = self._validate_key(layer, 0)[0]
        selected = tuple(self._validate_key(layer, expert)[1] for expert in selected)
        if not self.prefetch_count:
            return
        candidates = Counter()
        with self._lock:
            for expert in selected:
                candidates.update(self._transitions[(layer, expert)])
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stale = sum(f.done() for f in self._futures.values())
            out = asdict(self.stats)
            out["inflight"] = len(self._futures)
            out["completed_prefetch"] = stale
            out["ready_prefetch"] = len(self._prefetched_ready)
            out["prefetch_limit"] = self._max_speculative
            frequent = Counter(
                {
                    (layer, expert): count
                    for layer, counts in self._freq.items()
                    for expert, count in counts.items()
                }
            )
            out["frequent_experts"] = [
                {"layer": layer, "expert": expert, "count": count}
                for (layer, expert), count in frequent.most_common(20)
            ]
        out["cache"] = self.cache.snapshot()
        return out

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._futures.values())
            self.stats.prefetch_wasted += len(self._future_prefetch) + len(self._prefetched_ready)
            self._futures.clear()
            self._future_prefetch.clear()
            self._prefetched_ready.clear()
        for future in futures:
            future.cancel()

    def __enter__(self) -> ExpertStore:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(slots=True)
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
    ) -> None:
        if budget_bytes < 1:
            raise ValueError("budget_bytes must be positive")
        if layer < 0:
            raise ValueError("layer must be nonnegative")
        if n_shards < 1 or rows_per_shard < 1:
            raise ValueError("n_shards and rows_per_shard must be positive")
        if group_size < 1 or bits < 1:
            raise ValueError("group_size and bits must be positive")
        self.index = index
        self.cache = ExpertCache(max_bytes=budget_bytes, policy=policy)
        self.layer = layer
        self.n_shards = n_shards
        self.rows_per_shard = rows_per_shard
        self.group_size = group_size
        self.bits = bits
        self.prefetch_count = max(0, prefetch)
        self.stats = StreamStats()
        self._futures: dict[RowKey, Future] = {}
        self._lock = threading.RLock()
        self._closed = False
        if self._name(0, "weight") not in index.tensors:
            raise KeyError(f"streamed PLE tensor missing: {self._name(0, 'weight')}")
        for shard in range(n_shards):
            weight = index.tensors.get(self._name(shard, "weight"))
            if weight is None:
                raise KeyError(f"streamed PLE tensor missing: {self._name(shard, 'weight')}")
            if not weight.shape or weight.shape[0] != rows_per_shard:
                raise ValueError(f"PLE shard row count mismatch: {self._name(shard, 'weight')}")
            for suffix in ("scales", "biases"):
                tensor = index.tensors.get(self._name(shard, suffix))
                if tensor is not None and (not tensor.shape or tensor.shape[0] != rows_per_shard):
                    raise ValueError(f"PLE shard row count mismatch: {self._name(shard, suffix)}")
            scales = index.tensors.get(self._name(shard, "scales"))
            biases = index.tensors.get(self._name(shard, "biases"))
            if biases is not None and scales is None:
                raise ValueError(f"PLE biases require scales for shard {shard}")
        self._row_bytes = sum(
            index.tensors[self._name(0, suffix)].row_bytes
            for suffix in ("weight", "scales", "biases")
            if self._name(0, suffix) in index.tensors
        )
        if self._row_bytes < 1:
            raise ValueError("PLE tensors must contain at least one byte per row")
        if budget_bytes < self._row_bytes:
            raise ValueError("budget_bytes must hold at least one PLE row")
        for shard in range(1, n_shards):
            shard_bytes = sum(
                index.tensors[self._name(shard, suffix)].row_bytes
                for suffix in ("weight", "scales", "biases")
                if self._name(shard, suffix) in index.tensors
            )
            if shard_bytes != self._row_bytes:
                raise ValueError("PLE shards must have equal row sizes")
        self._max_speculative = max(1, budget_bytes // max(1, self._row_bytes))

    def _name(self, shard: int, suffix: str) -> str:
        return f"model.layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.{suffix}"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PLE store is closed")

    @staticmethod
    def _as_ids(gids: mx.array | np.ndarray) -> np.ndarray:
        if isinstance(gids, mx.array):
            mx.eval(gids)
        ids = np.asarray(gids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("PLE row ids must be integers")
        return ids.astype(np.int64, copy=False)

    def _keys(self, ids: np.ndarray) -> list[RowKey]:
        flat = ids.reshape(-1)
        if flat.size:
            minimum = int(flat.min())
            maximum = int(flat.max())
            total_rows = self.n_shards * self.rows_per_shard
            if minimum < 0 or maximum >= total_rows:
                raise IndexError("PLE global row outside sharded table")
        return [divmod(int(gid), self.rows_per_shard) for gid in flat]

    def _materialize(self, shard: int, value: PLERow) -> PLERow:
        value.weight = _to_mlx(value.weight, self.index.tensors[self._name(shard, "weight")].dtype)
        if value.scales is not None:
            value.scales = _to_mlx(
                value.scales, self.index.tensors[self._name(shard, "scales")].dtype
            )
        if value.biases is not None:
            value.biases = _to_mlx(
                value.biases, self.index.tensors[self._name(shard, "biases")].dtype
            )
        return value

    def _load_rows(self, shard: int, rows: tuple[int, ...]) -> dict[RowKey, PLERow]:
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
            out[(shard, row)] = PLERow(
                weights[i],
                None if scales is None else scales[i],
                None if biases is None else biases[i],
                row_bytes,
            )
        with self._lock:
            self.stats.loads += len(rows)
            self.stats.loaded_bytes += len(rows) * row_bytes
            self.stats.load_seconds += time.perf_counter() - t0
        return out

    def _submit_group(self, shard: int, rows: Iterable[int], prefetch: bool) -> None:
        missing = tuple(
            sorted(
                set(
                    int(r)
                    for r in rows
                    if self.cache.peek((shard, int(r))) is None
                    and (shard, int(r)) not in self._futures
                )
            )
        )
        if prefetch:
            slots = self._max_speculative - len(self._futures)
            missing = missing[: max(0, slots)]
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
                    with self._lock:
                        child = self._futures.get(key)
                        if child is not None and not child.done():
                            child.set_result((value, prefetch))
            except Exception as error:
                with self._lock:
                    for row in missing:
                        child = self._futures.get((shard, row))
                        if child is not None and not child.done():
                            child.set_exception(error)

        group_future.add_done_callback(finish)

    def prefetch_rows(self, gids: mx.array | np.ndarray) -> None:
        self._ensure_open()
        if not self.prefetch_count:
            return
        ids = self._as_ids(gids)
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for shard, row in dict.fromkeys(self._keys(ids)):
            groups[shard].append(row)
        with self._lock:
            for shard, rows in groups.items():
                self._submit_group(shard, rows, True)

    def get_rows(self, gids: mx.array | np.ndarray) -> mx.array:
        self._ensure_open()
        ids = self._as_ids(gids)
        keys = self._keys(ids)
        unique_keys = tuple(dict.fromkeys(keys))
        with self._lock:
            self.stats.rows_requested += len(keys)
            self.stats.unique_rows += len(unique_keys)
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for shard, row in unique_keys:
            if self.cache.peek((shard, row)) is None:
                groups[shard].append(row)
        with self._lock:
            for shard, rows in groups.items():
                self._submit_group(shard, rows, False)
        values: dict[RowKey, PLERow] = {}
        for key in unique_keys:
            value = self.cache.get(key)
            if value is None:
                with self._lock:
                    future = self._futures[key]
                wait_started = time.perf_counter()
                try:
                    value, prefetched = future.result()
                except Exception:
                    with self._lock:
                        if self._futures.get(key) is future:
                            self._futures.pop(key, None)
                    raise
                waited = time.perf_counter() - wait_started
                with self._lock:
                    self.stats.wait_seconds += waited
                shard, _ = key
                try:
                    value = self._materialize(shard, value)
                except Exception:
                    with self._lock:
                        if self._futures.get(key) is future:
                            self._futures.pop(key, None)
                    raise
                with self._lock:
                    if self._futures.get(key) is not future:
                        cached = self.cache.get(key)
                        values[key] = value if cached is None else cached
                        continue
                    self._futures.pop(key, None)
                    if prefetched:
                        self.stats.prefetch_hits += 1
                self.cache.put(key, value, value.nbytes)
            values[key] = value
        if not keys:
            width = self._embedding_width()
            return mx.zeros((*ids.shape, width), dtype=mx.float32)
        unique = list(values)
        rows = [values[key] for key in unique]
        weights = mx.stack([row.weight for row in rows])
        if rows[0].scales is not None:
            scales = mx.stack([row.scales for row in rows])
            biases = None if rows[0].biases is None else mx.stack([row.biases for row in rows])
            weights = mx.dequantize(
                weights, scales, biases, group_size=self.group_size, bits=self.bits
            )
        lookup = {key: i for i, key in enumerate(unique)}
        positions = mx.array([lookup[key] for key in keys])
        return mx.take(weights, positions, axis=0).reshape(*ids.shape, -1)

    def _embedding_width(self) -> int:
        scales_name = self._name(0, "scales")
        return (
            self.index.tensors[scales_name].shape[-1] * self.group_size
            if scales_name in self.index.tensors
            else self.index.tensors[self._name(0, "weight")].shape[-1]
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = asdict(self.stats)
            out["inflight"] = len(self._futures)
        out["cache"] = self.cache.snapshot()
        return out

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.stats.prefetch_wasted += len(self._futures)
            futures = list(self._futures.values())
            self._futures.clear()
        for future in futures:
            future.cancel()

    def __enter__(self) -> PLEStore:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
