from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path


def run_streaming(cmd: Sequence[str], prompt: str) -> tuple[int, str, str, float, float | None]:
    started = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("benchmark process did not expose output pipes")
    chunks: list[str] = []
    errors: list[str] = []
    drain = threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True)
    drain.start()
    first_token_at: float | None = None
    marker = f"> {prompt}"
    armed = False
    marker_window = ""
    while True:
        ch = process.stdout.read(1)
        if ch == "" and process.poll() is not None:
            break
        if not ch:
            continue
        chunks.append(ch)
        if not armed:
            marker_window = (marker_window + ch)[-len(marker) :]
            armed = marker_window == marker
        elif first_token_at is None and not ch.isspace():
            first_token_at = time.perf_counter()
    rc = process.wait()
    drain.join()
    finished = time.perf_counter()
    return (
        rc,
        "".join(chunks),
        "".join(errors),
        finished - started,
        first_token_at - started if first_token_at is not None else None,
    )


def parse_llama_timings(text: str) -> dict[str, float]:
    patterns = {
        "load_s": r"load time =\s*([\d.]+) ms",
        "prompt_tps": (
            r"prompt eval time =\s*[\d.]+ ms /\s*\d+ tokens .*?,\s*"
            r"([\d.]+) tokens per second"
        ),
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


def benchmark(
    model: str, prompt: str, n_predict: int, ctx: int, threads: int, ngl: str
) -> dict[str, object]:
    if n_predict < 0 or ctx < 1 or threads < 1:
        raise ValueError("n_predict must be nonnegative, ctx and threads must be positive")
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
    timed_cmd = ["/usr/bin/time", "-l", *cmd] if Path("/usr/bin/time").is_file() else cmd
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


def write_json(path: str | Path, obj: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2), encoding="utf-8")
