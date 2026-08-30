import mlx.core as mx
import pytest

from colibri_runtime.generation import _sample, generate_tokens


class FixedModel:
    def make_cache(self):
        return []

    def __call__(self, ids, cache=None):
        return mx.zeros((ids.shape[0], ids.shape[1], 4)) + mx.array([0.0, 0.0, 1.0, 0.0])


def test_generation_stops_at_eos_and_reports_timing() -> None:
    seen = []
    generated, timing = generate_tokens(
        FixedModel(), [1, 2], max_tokens=4, eos_token_id=2, on_token=seen.append
    )
    assert generated == [2]
    assert seen == [2]
    assert timing["started"] <= timing["prefill_done"] <= timing["first_token"] <= timing["done"]


@pytest.mark.parametrize(
    ("temperature", "top_p"),
    [(float("nan"), 1.0), (-1.0, 1.0), (0.0, 0.0), (0.0, 1.1)],
)
def test_sampling_rejects_invalid_parameters(temperature: float, top_p: float) -> None:
    with pytest.raises(ValueError):
        _sample(mx.array([[1.0, 0.0]]), temperature, top_p)
