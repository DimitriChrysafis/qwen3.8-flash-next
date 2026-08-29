from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path


def run_streaming(cmd: list[str], prompt: str) -> tuple[int, str, str, float, float | None]:
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout is not None and p.stderr is not None
    chunks: list[str] = []
    errors: list[str] = []
    drain = threading.Thread(target=lambda: errors.extend(iter(p.stderr.readline, "")), daemon=True)
    drain.start()
    first_token_at: float | None = None
    marker = f"> {prompt}"
    armed = False
    while True:
        ch = p.stdout.read(1)
        if ch == "" and p.poll() is not None:
            break
        if not ch:
            continue
        chunks.append(ch)
        if not armed and marker in "".join(chunks[-len(marker) - 2:]):
            armed = True
        elif armed and first_token_at is None and not ch.isspace():
            first_token_at = time.time()
    rc = p.wait()
    drain.join()
    return rc, "".join(chunks), "".join(errors), time.time() - t0, (first_token_at - t0) if first_token_at else None


def parse_llama_timings(text: str) -> dict[str, float]:
    patterns = {
        "load_s": r"load time =\s*([\d.]+) ms",
        "prompt_tps": r"prompt eval time =\s*[\d.]+ ms /\s*\d+ tokens .*?,\s*([\d.]+) tokens per second",
        "gen_tps": r"eval time =\s*[\d.]+ ms /\s*\d+ runs .*?,\s*([\d.]+) tokens per second",
    }
    out: dict[str, float] = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if k == "load_s":
                v = v / 1000.0
            out[k] = v
    summary = re.search(r"\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]", text)
    if summary:
        out["prompt_tps"] = float(summary.group(1))
        out["gen_tps"] = float(summary.group(2))
    return out


def benchmark(model: str, prompt: str, n_predict: int, ctx: int, threads: int, ngl: str) -> dict:
    executable = os.environ.get(
        "COLIBRI_LLAMA_CLI",
        str(Path.home() / "qwen3.8next/llama.cpp/build/bin/llama-cli"),
    )
    cmd = [
        executable,
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx),
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "-b",
        "32",
        "-ub",
        "32",
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
        "--single-turn",
        "--no-warmup",
        "--perf",
        "--cpu-moe",
        "--gpu-layers",
        ngl,
        "-ot",
        "per_layer_token_embd=CPU",
        "--fit",
        "off",
    ]
    timed_cmd = ["/usr/bin/time", "-l", *cmd]
    rc, so, se, wall, ttft = run_streaming(timed_cmd, prompt)
    parsed = parse_llama_timings(so + "\n" + se)
    rss = None
    m = re.search(r"(\d+)\s+maximum resident set size", se)
    if m:
        rss = int(m.group(1))
    return {
        "cmd": timed_cmd,
        "returncode": rc,
        "wall_s": wall,
        "ttft_s": ttft,
        "max_rss_bytes": rss,
        "timings": parsed,
        "stderr_tail": se[-4000:],
        "stdout_tail": so[-2000:],
    }


def write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
