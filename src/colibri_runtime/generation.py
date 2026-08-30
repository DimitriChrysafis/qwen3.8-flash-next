from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np
import psutil


@dataclass(slots=True)
class GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    decode_tokens: int
    load_seconds: float
    prefill_seconds: float
    generation_seconds: float
    ttft_seconds: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    rss_bytes: int
    average_rss_bytes: int
    peak_rss_bytes: int
    peak_mlx_bytes: int
    active_mlx_bytes: int
    cache_mlx_bytes: int


class MemorySampler:
    def __init__(self, interval: float = 0.05) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.process = psutil.Process()
        self.samples: list[int] = []
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._samples_lock:
                self.samples.append(self.process.memory_info().rss)

    def reset(self) -> None:
        sample = self.process.memory_info().rss
        with self._samples_lock:
            self.samples[:] = [sample]

    def snapshot(self) -> dict[str, int]:
        with self._samples_lock:
            samples = tuple(self.samples)
        samples = samples or (self.process.memory_info().rss,)
        return {
            "average_rss_bytes": int(sum(samples) / len(samples)),
            "peak_rss_bytes": max(samples),
            "rss_samples": len(samples),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join()


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_p < 1.0:
        order = mx.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        remove = mx.cumsum(probs, axis=-1) - probs > top_p
        sorted_logits = mx.where(remove, -mx.inf, sorted_logits)
        choice = mx.random.categorical(sorted_logits)
        return mx.take_along_axis(order, choice[:, None], axis=-1)[:, 0]
    return mx.random.categorical(logits)


def generate_tokens(
    model,
    input_ids,
    max_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | Iterable[int] | None = None,
    seed: int = 0,
    on_token: Callable[[int], None] | None = None,
) -> tuple[list[int], dict[str, float]]:
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    mx.random.seed(seed)
    ids = mx.array(input_ids, dtype=mx.int32)
    if ids.ndim == 1:
        ids = ids[None]
    elif ids.ndim != 2:
        raise ValueError("input_ids must be one- or two-dimensional")
    if ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")
    eos_ids = (
        frozenset()
        if eos_token_id is None
        else frozenset({eos_token_id})
        if isinstance(eos_token_id, int)
        else frozenset(eos_token_id)
    )
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    mx.eval(logits)
    prefill_done = time.perf_counter()
    generated: list[int] = []
    first_at: float | None = None
    for step in range(max_tokens):
        token = _sample(logits[:, -1].astype(mx.float32), temperature, top_p)
        mx.eval(token)
        if first_at is None:
            first_at = time.perf_counter()
        value = int(np.asarray(token)[0])
        generated.append(value)
        if on_token is not None:
            on_token(value)
        if eos_ids and value in eos_ids:
            break
        if step + 1 < max_tokens:
            logits = model(token[:, None], cache=cache)
            mx.eval(logits)
    done = time.perf_counter()
    return generated, {
        "started": t0,
        "prefill_done": prefill_done,
        "first_token": first_at if first_at is not None else done,
        "done": done,
    }


def metrics(
    model,
    prompt_tokens: int,
    generated_tokens: int,
    timing: dict[str, float],
    load_seconds: float = 0.0,
    memory: dict[str, int] | None = None,
) -> dict[str, object]:
    if prompt_tokens < 0 or generated_tokens < 0:
        raise ValueError("token counts must be nonnegative")
    prefill = timing["prefill_done"] - timing["started"]
    generation = timing["done"] - timing["first_token"]
    decode_tokens = max(0, generated_tokens - 1)
    process = psutil.Process()
    memory = memory or {
        "average_rss_bytes": process.memory_info().rss,
        "peak_rss_bytes": process.memory_info().rss,
    }
    result = GenerationMetrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        decode_tokens=decode_tokens,
        load_seconds=load_seconds,
        prefill_seconds=prefill,
        generation_seconds=generation,
        ttft_seconds=timing["first_token"] - timing["started"],
        prompt_tokens_per_second=prompt_tokens / prefill if prefill else 0.0,
        generation_tokens_per_second=decode_tokens / generation if generation else 0.0,
        rss_bytes=process.memory_info().rss,
        average_rss_bytes=memory["average_rss_bytes"],
        peak_rss_bytes=memory["peak_rss_bytes"],
        peak_mlx_bytes=mx.get_peak_memory(),
        active_mlx_bytes=mx.get_active_memory(),
        cache_mlx_bytes=mx.get_cache_memory(),
    )
    out = asdict(result)
    out["expert"] = model.expert_store.snapshot()
    out["ple"] = model.ple_store.snapshot()
    out["disk"] = model.expert_store.index.snapshot()
    out["load"] = model.load_stats
    read_seconds = out["disk"]["read_seconds"]
    out["disk"]["effective_bytes_per_second"] = (
        out["disk"]["bytes_read"] / read_seconds if read_seconds else 0.0
    )
    for stream in (out["expert"], out["ple"]):
        submitted = stream["prefetch_submitted"]
        stream["prefetch_hit_rate"] = stream["prefetch_hits"] / submitted if submitted else 0.0
    return out
