import sys

from colibri_runtime.benchmark import parse_llama_timings, run_streaming


def test_parse_current_llama_summary() -> None:
    assert parse_llama_timings("[ Prompt: 3.7 t/s | Generation: 0.4 t/s ]") == {
        "prompt_tps": 3.7,
        "gen_tps": 0.4,
    }


def test_parse_detailed_llama_timings() -> None:
    output = """
    load time = 125.0 ms
    prompt eval time = 250.0 ms / 10 tokens (25.0 ms per token, 40.0 tokens per second)
    eval time = 500.0 ms / 5 runs (100.0 ms per token, 10.0 tokens per second)
    """
    assert parse_llama_timings(output) == {"load_s": 0.125, "prompt_tps": 40.0, "gen_tps": 10.0}


def test_run_streaming_captures_first_token_and_stderr() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; print('> hello world', flush=True); print('diagnostic', file=sys.stderr)",
    ]
    returncode, stdout, stderr, wall_seconds, ttft_seconds = run_streaming(command, "hello")
    assert returncode == 0
    assert stdout == "> hello world\n"
    assert stderr == "diagnostic\n"
    assert wall_seconds >= 0
    assert ttft_seconds is not None
