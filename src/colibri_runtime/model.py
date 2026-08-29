# Copyright 2026 The Qwen Team, The HuggingFace Inc. team, and PipeNetwork contributors.
# Adapted and modified for disk-streamed Colibri execution.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache
from mlx_lm.models.gated_delta import gated_delta_update

from .streaming import ExpertStore, PLEStore


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(rp.get("partial_rotary_factor", self.text.partial_rotary_factor))
        if not self.text.layer_types:
            self.text.layer_types = [
                "full_attention" if (i + 1) % self.text.full_attention_interval == 0 else "linear_attention"
                for i in range(self.text.num_hidden_layers)
            ]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError("RMSNorm group size does not divide dimension")

    def __call__(self, x):
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        return mx.fast.rms_norm(x, None, self.eps).reshape(shape) * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x, gate=None):
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        return (act(gate.astype(mx.float32)) * out.astype(mx.float32)).astype(x.dtype)


def _rope_partial(x, cos, sin):
    d = cos.shape[-1]
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    rot = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


def _l2norm(x, eps=1e-6):
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + eps)).astype(x.dtype)


class RotaryEmbedding:
    def __init__(self, dim, base):
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, positions):
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class QSAIndexer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(args.hidden_size,
            (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x, rope, cache, offset):
        bsz, seq, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(bsz, seq, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(bsz, seq, self.head_dim)
        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = raw_k.shape[1]
        if kv_len <= self.token_budget:
            return None
        n_blocks = kv_len // self.compress_ratio
        pooled = raw_k[:, :n_blocks * self.compress_ratio].reshape(
            bsz, n_blocks, self.compress_ratio, self.head_dim)
        pooled = self.k_layernorm(pooled.astype(mx.float32).mean(2).astype(raw_k.dtype))
        starts = mx.arange(n_blocks) * self.compress_ratio
        cos, sin = rope(starts[None])
        pooled = _rope_partial(pooled, cos, sin)
        q_pos = mx.arange(offset, offset + seq)
        cos, sin = rope(q_pos[None])
        q = _rope_partial(self.q_layernorm(q), cos[:, :, None], sin[:, :, None])
        scores = mx.einsum("bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32))
        scores = mx.maximum(scores, 0).sum(-1) / math.sqrt(self.head_dim)
        block_end = starts + self.compress_ratio - 1
        visible = block_end[None, None] <= q_pos[None, :, None]
        scores = mx.where(visible, scores, -mx.inf)
        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
        keep_block = mx.zeros((bsz, seq, n_blocks + 1), dtype=mx.bool_)
        top = mx.where(mx.take_along_axis(visible, top, -1), top, n_blocks)
        keep_block = mx.put_along_axis(keep_block, top, mx.array(True), -1)[..., :n_blocks]
        keep = mx.repeat(keep_block, self.compress_ratio, -1)
        tail = kv_len - n_blocks * self.compress_ratio
        if tail:
            keep = mx.concatenate([keep, mx.zeros((bsz, seq, tail), dtype=mx.bool_)], -1)
        key_pos = mx.arange(kv_len)[None, None]
        qp = q_pos[None, :, None]
        own_start = ((qp + 1) // self.compress_ratio) * self.compress_ratio
        own_tail = (key_pos >= own_start) & (key_pos <= qp)
        return ((keep | own_tail) & (key_pos <= qp))[:, None]


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)

    def __call__(self, x, rope, mask, cache, idx_cache):
        bsz, seq, _ = x.shape
        offset = cache.offset if cache is not None else 0
        sparse = self.indexer(x, rope, idx_cache, offset)
        q, gate = mx.split(self.q_proj(x).reshape(bsz, seq, self.n_heads, -1), 2, -1)
        gate = gate.reshape(bsz, seq, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(bsz, seq, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cos, sin = rope(mx.arange(offset, offset + seq)[None])
        q = _rope_partial(q, cos[:, None], sin[:, None])
        k = _rope_partial(k, cos[:, None], sin[:, None])
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if sparse is not None:
            if mask is None or isinstance(mask, str):
                mask = sparse
            elif mask.dtype == mx.bool_:
                mask = mask & sparse
            else:
                neg = mx.finfo(q.dtype).min
                mask = mask + mx.where(sparse, mx.array(0, q.dtype), mx.array(neg, q.dtype))
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class GatedDeltaNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_v, self.n_k = args.linear_num_value_heads, args.linear_num_key_heads
        self.dk, self.dv = args.linear_key_head_dim, args.linear_value_head_dim
        self.key_dim, self.value_dim = self.dk * self.n_k, self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False,
                                kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=0)
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(self.dv, args.rms_norm_eps, args.output_gate_type)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x, mask, cache):
        bsz, seq, _ = x.shape
        mixed = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(bsz, seq, self.n_v, self.dv)
        b, a = self.in_proj_b(x), self.in_proj_a(x)
        state = cache[0] if cache is not None and cache[0] is not None else mx.zeros(
            (bsz, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        if mask is not None:
            mixed = mx.where(mask[..., None], mixed, 0)
        conv_input = mx.concatenate([state, mixed], 1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):])
        conv = nn.silu(self.conv1d(conv_input))
        q, k, v = mx.split(conv, [self.key_dim, 2 * self.key_dim], -1)
        q = _l2norm(q.reshape(bsz, seq, self.n_k, self.dk)) * self.dk ** -0.5
        k = _l2norm(k.reshape(bsz, seq, self.n_k, self.dk))
        v = v.reshape(bsz, seq, self.n_v, self.dv)
        recurrent = cache[1] if cache is not None else None
        out, recurrent = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias,
                                             recurrent, mask, use_kernel=not self.training and self.dk >= 32)
        if cache is not None:
            cache[1] = recurrent
            cache.advance(seq)
        return self.out_proj(self.norm(out, z).reshape(bsz, seq, -1))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class StreamedSparseMoeBlock(nn.Module):
    def __init__(self, args, layer_idx, store):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.layer_idx = layer_idx
        self.store = store
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, -1)[..., :self.top_k]
        weights = mx.softmax(mx.take_along_axis(logits, idx, -1), -1, precise=True)
        mx.eval(idx, weights)
        idx_np = np.asarray(idx).reshape(-1, self.top_k)
        flat_x = x.reshape(-1, x.shape[-1])
        unique = np.unique(idx_np)
        unique_ids = unique.tolist()
        self.store.prepare(self.layer_idx, unique_ids)
        slots = mx.zeros((flat_x.shape[0] * self.top_k, x.shape[-1]), dtype=x.dtype)
        for position, expert_id in enumerate(unique_ids):
            token, slot = np.nonzero(idx_np == expert_id)
            positions = token * self.top_k + slot
            selected = mx.take(flat_x, mx.array(token), axis=0)
            result = self.store.get(self.layer_idx, expert_id)(selected).astype(x.dtype)
            ahead = position + self.store.io_window
            if ahead < len(unique_ids):
                self.store.request(self.layer_idx, unique_ids[ahead])
            slots = mx.put_along_axis(slots, mx.array(positions)[:, None], result, axis=0)
        routed = slots.reshape(flat_x.shape[0], self.top_k, -1)
        routed = (routed * weights.reshape(-1, self.top_k)[..., None]).sum(1).reshape(x.shape)
        self.store.record_route(self.layer_idx, idx_np.reshape(-1).tolist())
        self.store.prefetch_predictions(self.layer_idx, unique.tolist())
        return routed + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class GatedResidual(nn.Module):
    def __init__(self, args, use_combine=True):
        super().__init__()
        self.hc, self.d = args.hc_count, args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, self.d, args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None

    def __call__(self, hyper):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w)).reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _splitmix64(v):
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v):
    if v < 2: return False
    if v % 2 == 0: return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p): p += 1
    return p


class StreamedNGramEmbedding(nn.Module):
    def __init__(self, args, embed_dim, ple_layer_index, store):
        super().__init__()
        self.store = store
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.eos_token_id = args.eos_token_id[0] if isinstance(args.eos_token_id, list) else args.eos_token_id
        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            size = _nth_prime_after(args.ngram_vocab_size_base - 1, ple_layer_index * self.ngram_heads + h + 1)
            sizes.append(size); offsets.append(total); total += size
        mults = []
        half = max(1, (((1 << 63) - 1) // max(args.vocab_size, 1)) // 2)
        seed = args.seed + 10007 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(2 * (_splitmix64((seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1)
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)

    def _shift_right(self, ids, shift):
        if shift == 0: return ids
        bsz, seq = ids.shape
        pos = mx.arange(seq)
        prev_incl = mx.cummax(mx.where(ids == self.eos_token_id, pos, -1), axis=1)
        prev = mx.concatenate([mx.full((bsz, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], 1)
        src = pos - shift
        gathered = mx.take_along_axis(ids, mx.broadcast_to(mx.maximum(src, 0)[None], (bsz, seq)), 1)
        return mx.where((pos[None] - (prev + 1) >= shift) & (src[None] >= 0), gathered, self.eos_token_id)

    def indices(self, ids, prev_context):
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], 1).astype(mx.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            gid = mixed[..., None] % self.ngram_heads_vocab_sizes[lo:hi].reshape(1, 1, -1)
            blocks.append(gid + self.ngram_heads_offsets[lo:hi].reshape(1, 1, -1))
        return mx.concatenate(blocks, -1)[:, -n_new:]

    def prefetch(self, gid):
        self.store.prefetch_rows(gid)

    def __call__(self, ids, prev_context, gid=None):
        gid = self.indices(ids, prev_context) if gid is None else gid
        return self.store.get_rows(gid).reshape(*gid.shape[:2], -1)


class PLELayer(nn.Module):
    def __init__(self, args, ple_index, store):
        super().__init__()
        self.d, self.hc = args.hidden_size, args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = StreamedNGramEmbedding(args, args.ple_embed_dim, ple_index, store)
        self.dilation = args.ngram_size
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, self.d, args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, self.d, args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, self.d, args.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc_dim, hc_dim, kernel_size=args.ple_conv_kernel_size,
                                groups=hc_dim, dilation=self.dilation, bias=False)

    def _short_conv(self, x, cache):
        seq, n = x.shape[1], self.short_conv_state_len
        state = cache[2] if cache is not None and cache[2] is not None else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        full = mx.concatenate([state, x], 1)
        if cache is not None: cache[2] = mx.contiguous(full[:, -n:])
        return nn.silu(self.conv1d(full[:, -(n + seq):]))

    def __call__(self, hidden, ids, prev_ctx, cache, gid=None):
        emb = self.ple_embedding(ids, prev_ctx, gid).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb)).reshape(*emb.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden).reshape(*hidden.shape[:-1], self.hc, self.d)
        gate = (key * query).sum(-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(*value.shape[:-1], -1)
        return gated + self._short_conv(self.norm_conv(gated), cache)


class DecoderLayer(nn.Module):
    def __init__(self, args, index, experts, ple_store):
        super().__init__()
        self.layer_type = args.layer_types[index]
        if self.layer_type == "linear_attention": self.linear_attn = GatedDeltaNet(args)
        else: self.self_attn = Attention(args)
        self.mlp = StreamedSparseMoeBlock(args, index, experts)
        ple_idx = args.ple_layer_ids.index(index + 1) if index + 1 in args.ple_layer_ids else None
        self.ple = PLELayer(args, ple_idx, ple_store) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx, ple_gid=None):
        if self.ple is not None: h = h + self.ple(h, ids, prev_ctx, cache, ple_gid)
        x, hyper, inject = self.attn_hyper_connection(h)
        x = self.linear_attn(x, conv_mask, cache) if self.layer_type == "linear_attention" else self.self_attn(x, rope, mask, cache, idx_cache)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args, experts, ple_store):
        super().__init__()
        self.args, self.hc = args, args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i, experts, ple_store) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, False)
        self.rope = RotaryEmbedding(int(args.head_dim * args.partial_rotary_factor), args.rope_theta)
        self.ple_layers = [i for i in range(args.num_hidden_layers) if i + 1 in args.ple_layer_ids]

    def __call__(self, ids, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None: cache = [None] * len(self.layers)
        full = [i for i, layer in enumerate(self.layers) if layer.layer_type != "linear_attention"]
        attn_cache = cache[full[0]] if full else None
        mask = create_attention_mask(h, [attn_cache] if attn_cache is not None else None)
        prev_ctx = None
        ple_gid = None
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id[0] if isinstance(self.args.eos_token_id, list) else self.args.eos_token_id
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = prev if prev is not None else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            if pc is not None: pc[3] = mx.concatenate([prev_ctx, ids], 1)[:, -ctx_len:]
            ple_embedding = self.layers[self.ple_layers[0]].ple.ple_embedding
            ple_gid = ple_embedding.indices(ids, prev_ctx)
            ple_embedding.prefetch(ple_gid)
        h = mx.tile(h, (1, 1, self.hc))
        for layer, state in zip(self.layers, cache):
            idx_cache = state.indexer if state is not None and hasattr(state, "indexer") else None
            h = layer(h, self.rope, mask, None, state, idx_cache, ids, prev_ctx, ple_gid)
        return self.hyper_connection_mixer(h)


class _IndexerCache(_BaseCache):
    def __init__(self): self.keys = None
    def update(self, k): self.keys = k if self.keys is None else mx.concatenate([self.keys, k], 1); return self.keys
    @property
    def state(self): return self.keys
    @state.setter
    def state(self, value): self.keys = value


class _AttnCache(KVCache):
    def __init__(self): super().__init__(); self.indexer = _IndexerCache()


class Model(nn.Module):
    CENTERED_NORMS = ("hc_norm.weight", "q_norm.weight", "k_norm.weight",
        "indexer.q_layernorm.weight", "indexer.k_layernorm.weight",
        "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight")

    def __init__(self, args, expert_store, ple_store):
        super().__init__()
        self.args, self.model_type = args, args.model_type
        self.expert_store, self.ple_store = expert_store, ple_store
        self.model = Qwen4ExpModel(args.text, expert_store, ple_store)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(args.text.hidden_size, args.text.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(out) if self.args.text.tie_word_embeddings else self.lm_head(out)

    @property
    def layers(self): return self.model.layers

    def make_cache(self):
        return [_AttnCache() if kind != "linear_attention" else ArraysCache(4) for kind in self.args.text.layer_types]

    def sanitize(self, weights):
        raw = any(k.startswith("model.language_model.") for k in weights) or any(k.endswith("mlp.experts.gate_up_proj") for k in weights)
        out = {}
        for key, value in weights.items():
            if key.startswith("mtp.") or key.startswith("vision_tower.") or key.startswith("model.visual."): continue
            if key.startswith("language_model."): key = key[len("language_model."):]
            if key.startswith("model.language_model."): key = "model." + key[len("model.language_model."):]
            if key.endswith(".mlp.experts.gate_up_proj") or key.endswith(".mlp.experts.down_proj"): continue
            if "conv1d.weight" in key and value.ndim == 3 and value.shape[1] == 1 and value.shape[-1] != 1:
                value = value.transpose(0, 2, 1)
            if raw and key.endswith(self.CENTERED_NORMS): value = value + 1.0
            out[key] = value
        return out
