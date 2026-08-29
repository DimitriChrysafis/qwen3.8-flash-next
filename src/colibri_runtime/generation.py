from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np
import psutil


@dataclass
class GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    load_seconds: float
    prefill_seconds: float
    generation_seconds: float
    ttft_seconds: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    rss_bytes: int
    peak_mlx_bytes: int
    active_mlx_bytes: int
    cache_mlx_bytes: int


def _sample(logits, temperature: float, top_p: float):
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


def generate_tokens(model, input_ids, max_tokens=64, temperature=0.0, top_p=1.0,
                    eos_token_id=None, seed=0, on_token=None):
    mx.random.seed(seed)
    ids = mx.array(input_ids, dtype=mx.int32)
    if ids.ndim == 1:
        ids = ids[None]
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    mx.eval(logits)
    prefill_done = time.perf_counter()
    generated = []
    first_at = None
    for _ in range(max_tokens):
        token = _sample(logits[:, -1].astype(mx.float32), temperature, top_p)
        mx.eval(token)
        if first_at is None:
            first_at = time.perf_counter()
        value = int(np.asarray(token)[0])
        generated.append(value)
        if on_token is not None:
            on_token(value)
        if eos_token_id is not None:
            eos = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
            if value in eos:
                break
        logits = model(token[:, None], cache=cache)
        mx.eval(logits)
    done = time.perf_counter()
    return generated, {
        "started": t0,
        "prefill_done": prefill_done,
        "first_token": first_at or done,
        "done": done,
    }


def metrics(model, prompt_tokens, generated_tokens, timing, load_seconds=0.0):
    prefill = timing["prefill_done"] - timing["started"]
    generation = timing["done"] - timing["prefill_done"]
    process = psutil.Process()
    result = GenerationMetrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        load_seconds=load_seconds,
        prefill_seconds=prefill,
        generation_seconds=generation,
        ttft_seconds=timing["first_token"] - timing["started"],
        prompt_tokens_per_second=prompt_tokens / prefill if prefill else 0.0,
        generation_tokens_per_second=generated_tokens / generation if generation else 0.0,
        rss_bytes=process.memory_info().rss,
        peak_mlx_bytes=mx.get_peak_memory(),
        active_mlx_bytes=mx.get_active_memory(),
        cache_mlx_bytes=mx.get_cache_memory(),
    )
    out = asdict(result)
    out["expert"] = model.expert_store.snapshot()
    out["ple"] = model.ple_store.snapshot()
    out["disk"] = model.expert_store.index.snapshot()
    out["load"] = model.load_stats
    return out
