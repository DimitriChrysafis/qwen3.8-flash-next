from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


def run_cmd(cmd: list[str]) -> tuple[int, str, str, float]:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


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
    rc, so, se, wall = run_cmd(cmd)
    parsed = parse_llama_timings(so + "\n" + se)
    return {
        "cmd": cmd,
        "returncode": rc,
        "wall_s": wall,
        "timings": parsed,
        "stderr_tail": se[-2000:],
        "stdout_tail": so[-2000:],
    }


def write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
