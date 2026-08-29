#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from colibri_runtime.benchmark import benchmark, write_json

MODEL = str(Path.home() / "qwen3.8next/models/bartowski-Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S/Qwen3.8-Flash-Next-IQ1_S-00001-of-00002.gguf")


def sys_snapshot() -> dict:
    mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    disk = subprocess.check_output(["df", "-h", str(Path.home())], text=True)
    return {"platform": platform.platform(), "mem_bytes": int(mem), "disk": disk}


def main() -> None:
    out = {
        "system": sys_snapshot(),
        "model": MODEL,
        "runs": {},
    }
    generated = int(os.environ.get("COLIBRI_BENCH_TOKENS", "8"))
    warmup = benchmark(MODEL, "Reply with the single word: warm", 1, 512, 12, "0")
    out["runs"]["warmup"] = warmup

    prompts = {
        "short": "Give one sentence defining sparse Mixture-of-Experts routing.",
        "medium": "Explain in 8 bullet points how to tune context length, batch size, and GPU offload for an Apple Silicon unified-memory laptop running a sparse MoE model.",
        "long": "Write a detailed, structured plan for benchmarking local LLM inference. Include warmup, TTFT, prompt processing, generation speed, memory pressure checks, disk I/O checks, and reproducibility controls."
                " Also include caveats for quantized sparse MoE models and fallback strategies when a full checkpoint cannot fit.",
    }
    for k, p in prompts.items():
        out["runs"][k] = benchmark(MODEL, p, generated, 512, 12, "0")

    write_json("/Users/dofa/Documents/github/qwen3.8-flash-next/artifacts/benchmark_results.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
