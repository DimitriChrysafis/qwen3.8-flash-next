import pytest

from colibri_runtime.model import ModelArgs


def test_model_args_derives_attention_layout() -> None:
    args = ModelArgs(text_config={"num_hidden_layers": 3, "full_attention_interval": 2})
    assert args.text.layer_types == ["linear_attention", "full_attention", "linear_attention"]


@pytest.mark.parametrize(
    "config",
    [
        {"ple_layer_ids": []},
        {"layer_types": ["unknown"]},
        {"partial_rotary_factor": 0},
    ],
)
def test_model_args_rejects_invalid_architecture(config) -> None:
    with pytest.raises(ValueError):
        ModelArgs(text_config=config)
