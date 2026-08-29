from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from .generation import MemorySampler, generate_tokens, metrics
from .load import load


def _bytes_gib(value):
    return int(float(value) * (1 << 30))


def _counters(model):
    return {
        "expert": model.expert_store.snapshot(),
        "ple": model.ple_store.snapshot(),
        "disk": model.expert_store.index.snapshot(),
    }


def _delta(after, before):
    out = {}
    for group in after:
        out[group] = {
            key: value - before[group].get(key, 0)
            for key, value in after[group].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and key not in {"inflight", "completed_prefetch", "ready_prefetch", "prefetch_limit"}
        }
        if "cache" in after[group]:
            out[group]["cache"] = {
                key: value - before[group]["cache"].get(key, 0)
                for key, value in after[group]["cache"].items()
                if key in {"hits", "misses", "evictions", "rejected"}
            }
    return out


def _parser():
    parser = argparse.ArgumentParser(prog="colibri-qwen")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expert-budget-gib", type=float, default=8.0)
    parser.add_argument("--ple-budget-gib", type=float, default=1.0)
    parser.add_argument("--mlx-cache-gib", type=float, default=1.0)
    parser.add_argument("--mlx-memory-gib", type=float, default=28.0)
    parser.add_argument("--cache-policy", choices=["lru", "lfu", "adaptive"], default="adaptive")
    parser.add_argument("--io-workers", type=int, default=6)
    parser.add_argument("--expert-prefetch", type=int, default=2)
    parser.add_argument("--ple-prefetch", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--chat", action="store_true")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--prompt", required=True)
    bench.add_argument("--max-tokens", type=int, default=32)
    bench.add_argument("--runs", type=int, default=1)
    bench.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    mx.set_cache_limit(_bytes_gib(args.mlx_cache_gib))
    mx.set_memory_limit(_bytes_gib(args.mlx_memory_gib))
    mx.reset_peak_memory()
    sampler = MemorySampler()
    model = None
    try:
        t0 = time.perf_counter()
        model, tokenizer = load(
            args.model,
            expert_budget_bytes=_bytes_gib(args.expert_budget_gib),
            ple_budget_bytes=_bytes_gib(args.ple_budget_gib),
            cache_policy=args.cache_policy,
            io_workers=args.io_workers,
            prefetch_experts=args.expert_prefetch,
            prefetch_ple=args.ple_prefetch,
        )
        loaded = time.perf_counter() - t0
        load_memory = sampler.snapshot()
        ids = (tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ) if args.command == "generate" and args.chat else tokenizer.encode(args.prompt))
        eos = model.args.text.eos_token_id
        if args.command == "generate":
            sampler.reset()
            counters_before = _counters(model)
            decoder = tokenizer.decode
            generated, timing = generate_tokens(
                model, ids, args.max_tokens, args.temperature, args.top_p, eos, args.seed,
                on_token=lambda token: print(decoder([token]), end="", flush=True),
            )
            print()
            report = metrics(model, len(ids), len(generated), timing, loaded, sampler.snapshot())
            report["run_counters"] = _delta(_counters(model), counters_before)
            report["load_memory"] = load_memory
            print(json.dumps(report, indent=2))
        else:
            reports = []
            for run in range(args.runs):
                sampler.reset()
                counters_before = _counters(model)
                generated, timing = generate_tokens(model, ids, args.max_tokens, 0.0, 1.0, eos, run)
                report = metrics(model, len(ids), len(generated), timing,
                                 loaded if run == 0 else 0.0, sampler.snapshot())
                report["run_counters"] = _delta(_counters(model), counters_before)
                if run == 0:
                    report["load_memory"] = load_memory
                reports.append(report)
            print(json.dumps(reports if args.runs > 1 else reports[0], indent=2))
        return 0
    finally:
        sampler.close()
        if model is not None:
            model.expert_store.close()
            model.ple_store.close()
            model.expert_store.index.close()


if __name__ == "__main__":
    raise SystemExit(main())
