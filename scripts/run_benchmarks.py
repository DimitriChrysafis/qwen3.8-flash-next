#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = Path.home() / "qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
BASE = (
    "Explain how a disk-backed sparse mixture-of-experts runtime routes tokens, "
    "loads selected experts, caches hot weights, and overlaps SSD reads with Metal compute. "
)
PROMPTS = {
    "short": "Define sparse mixture-of-experts routing in one sentence.",
    "medium": BASE * 4,
    "long": BASE * 12,
}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except FileNotFoundError as error:
        raise RuntimeError("system benchmarks require macOS sysctl") from error


def system_snapshot() -> dict[str, object]:
    memory = _sysctl("hw.memsize")
    chip = _sysctl("machdep.cpu.brand_string")
    disk = shutil.disk_usage(Path.home())
    return {
        "platform": platform.platform(),
        "chip": chip,
        "memory_bytes": int(memory),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def run_case(args: argparse.Namespace, name: str, prompt: str) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "colibri_runtime.cli",
        "--model",
        str(args.model),
        "--expert-budget-gib",
        str(args.expert_budget_gib),
        "--ple-budget-gib",
        str(args.ple_budget_gib),
        "--mlx-cache-gib",
        str(args.mlx_cache_gib),
        "--mlx-memory-gib",
        str(args.mlx_memory_gib),
        "--io-workers",
        str(args.io_workers),
        "--expert-prefetch",
        str(args.expert_prefetch),
        "--ple-prefetch",
        str(args.ple_prefetch),
        "benchmark",
        "--prompt",
        prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--runs",
        str(args.runs),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        return {
            "name": name,
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
        }
    return {
        "name": name,
        "command": command,
        "returncode": 0,
        "metrics": json.loads(process.stdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_results.json"))
    parser.add_argument("--max-tokens", type=_nonnegative_int, default=4)
    parser.add_argument("--runs", type=_positive_int, default=2)
    parser.add_argument("--expert-budget-gib", type=_positive_float, default=8.0)
    parser.add_argument("--ple-budget-gib", type=_positive_float, default=1.0)
    parser.add_argument("--mlx-cache-gib", type=_positive_float, default=1.0)
    parser.add_argument("--mlx-memory-gib", type=_positive_float, default=28.0)
    parser.add_argument("--io-workers", type=_positive_int, default=6)
    parser.add_argument("--expert-prefetch", type=_nonnegative_int, default=2)
    parser.add_argument("--ple-prefetch", type=_nonnegative_int, default=8)
    args = parser.parse_args()
    result = {
        "system": system_snapshot(),
        "model": str(args.model),
        "cases": [run_case(args, name, prompt) for name, prompt in PROMPTS.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all(case["returncode"] == 0 for case in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
