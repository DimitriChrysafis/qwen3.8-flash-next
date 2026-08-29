from colibri_runtime.benchmark import parse_llama_timings


def test_parse_current_llama_summary() -> None:
    assert parse_llama_timings("[ Prompt: 3.7 t/s | Generation: 0.4 t/s ]") == {
        "prompt_tps": 3.7,
        "gen_tps": 0.4,
    }
