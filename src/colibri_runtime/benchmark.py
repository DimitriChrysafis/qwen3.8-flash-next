from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


def run_streaming(cmd: list[str]) -> tuple[int, str, str, float, float | None]:
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout is not None
    chunks: list[str] = []
    first_token_at: float | None = None
    while True:
        ch = p.stdout.read(1)
        if ch == "" and p.poll() is not None:
            break
        if ch:
            chunks.append(ch)
            if first_token_at is None and not ch.isspace():
                first_token_at = time.time()
    se = p.stderr.read() if p.stderr is not None else ""
    rc = p.wait()
    return rc, "".join(chunks), se, time.time() - t0, (first_token_at - t0) if first_token_at else None


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
    return out


def benchmark(model: str, prompt: str, n_predict: int, ctx: int, threads: int, ngl: str) -> dict:
    cmd = [
        "llama-cli",
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
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
        "--cpu-moe",
        "--gpu-layers",
        ngl,
    ]
    timed_cmd = ["/usr/bin/time", "-l", *cmd]
    rc, so, se, wall, ttft = run_streaming(timed_cmd)
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
