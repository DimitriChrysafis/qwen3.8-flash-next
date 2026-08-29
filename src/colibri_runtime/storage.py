# Copyright 2026 The Qwen Team, The HuggingFace Inc. team, and PipeNetwork contributors.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import json
import os
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np


_DTYPES = {
    "BOOL": (np.bool_, 1), "U8": (np.uint8, 1), "I8": (np.int8, 1),
    "U16": (np.uint16, 2), "I16": (np.int16, 2), "U32": (np.uint32, 4),
    "I32": (np.int32, 4), "U64": (np.uint64, 8), "I64": (np.int64, 8),
    "F16": (np.float16, 2), "BF16": (np.uint16, 2), "F32": (np.float32, 4),
    "F64": (np.float64, 8),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def row_bytes(self) -> int:
        if not self.shape:
            return self.nbytes
        return self.nbytes // self.shape[0]


@dataclass
class IOStats:
    pread_calls: int = 0
    bytes_read: int = 0
    requested_bytes: int = 0
    read_seconds: float = 0.0
    tensors_read: int = 0
    rows_read: int = 0


class SafeTensorIndex:
    def __init__(self, model_dir: str | Path, workers: int = 4):
        self.model_dir = Path(model_dir)
        self.tensors: dict[str, TensorInfo] = {}
        self.stats = IOStats()
        self._lock = threading.Lock()
        self._fds: dict[Path, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="colibri-io")
        files = sorted(self.model_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors in {self.model_dir}")
        for path in files:
            self._parse(path)

    def _parse(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, 8, 0)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_len = struct.unpack("<Q", raw)[0]
            header_raw = os.pread(fd, header_len, 8)
            if len(header_raw) != header_len:
                raise ValueError(f"truncated safetensors JSON: {path}")
            header = json.loads(header_raw)
            base = 8 + header_len
            file_size = os.fstat(fd).st_size
            for stored_name, meta in header.items():
                if stored_name == "__metadata__":
                    continue
                name = stored_name.removeprefix("language_model.")
                dtype = meta["dtype"]
                if dtype not in _DTYPES:
                    raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
                lo, hi = map(int, meta["data_offsets"])
                shape = tuple(map(int, meta["shape"]))
                expected = int(np.prod(shape, dtype=np.int64)) * _DTYPES[dtype][1]
                if hi - lo != expected or base + hi > file_size:
                    raise ValueError(f"invalid safetensors range for {name}")
                if name in self.tensors:
                    raise ValueError(f"duplicate tensor {name}")
                self.tensors[name] = TensorInfo(name, path, dtype, shape, base + lo, hi - lo)
        finally:
            os.close(fd)

    def _fd(self, path: Path) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            return fd

    def _pread(self, info: TensorInfo, offset: int, nbytes: int) -> bytes:
        t0 = time.perf_counter()
        data = os.pread(self._fd(info.path), nbytes, info.offset + offset)
        dt = time.perf_counter() - t0
        if len(data) != nbytes:
            raise IOError(f"short pread for {info.name}: {len(data)} != {nbytes}")
        with self._lock:
            self.stats.pread_calls += 1
            self.stats.bytes_read += len(data)
            self.stats.requested_bytes += nbytes
            self.stats.read_seconds += dt
        return data

    @staticmethod
    def _numpy(data: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(data, dtype=_DTYPES[dtype][0]).reshape(shape)

    @classmethod
    def _array(cls, data: bytes, dtype: str, shape: tuple[int, ...]) -> mx.array:
        out = mx.array(cls._numpy(data, dtype, shape))
        return out.view(mx.bfloat16) if dtype == "BF16" else out

    def read_tensor(self, name: str) -> mx.array:
        info = self.tensors[name]
        data = self._pread(info, 0, info.nbytes)
        with self._lock:
            self.stats.tensors_read += 1
        return self._array(data, info.dtype, info.shape)

    def read_row(self, name: str, row: int) -> mx.array:
        return self.read_rows(name, [row])[0]

    def read_row_numpy(self, name: str, row: int) -> np.ndarray:
        return self.read_rows_numpy(name, [row])[0]

    def read_rows_numpy(self, name: str, rows: Iterable[int]) -> np.ndarray:
        info = self.tensors[name]
        if not info.shape:
            raise ValueError(f"scalar tensor {name} has no rows")
        requested = [int(x) for x in rows]
        if not requested:
            return np.empty((0, *info.shape[1:]), dtype=_DTYPES[info.dtype][0])
        if min(requested) < 0 or max(requested) >= info.shape[0]:
            raise IndexError(f"row outside {name} shape {info.shape}")
        unique = sorted(set(requested))
        raw: dict[int, bytes] = {}
        start = prev = unique[0]
        runs: list[tuple[int, int]] = []
        for row in unique[1:]:
            if row != prev + 1:
                runs.append((start, prev + 1))
                start = row
            prev = row
        runs.append((start, prev + 1))
        rb = info.row_bytes
        for lo, hi in runs:
            block = self._pread(info, lo * rb, (hi - lo) * rb)
            for row in range(lo, hi):
                begin = (row - lo) * rb
                raw[row] = block[begin:begin + rb]
        data = b"".join(raw[row] for row in requested)
        with self._lock:
            self.stats.rows_read += len(requested)
        return self._numpy(data, info.dtype, (len(requested), *info.shape[1:]))

    def read_rows(self, name: str, rows: Iterable[int]) -> mx.array:
        info = self.tensors[name]
        array = self.read_rows_numpy(name, rows)
        out = mx.array(array)
        return out.view(mx.bfloat16) if info.dtype == "BF16" else out

    def read_rows_async(self, name: str, rows: Iterable[int]) -> Future:
        rows = tuple(rows)
        return self.executor.submit(self.read_rows_numpy, name, rows)

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self.stats)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
