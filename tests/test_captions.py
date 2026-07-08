from src.captions import CAPTIONS, get_caption


def test_required_caption_keys_exist() -> None:
    required = {
        "Input",
        "Conv 1",
        "Pool 1",
        "Conv 2",
        "Pool 2",
        "Conv 3",
        "Conv 4",
        "Conv 5",
        "Pool 5",
        "Avg pool",
        "Classifier",
        "Prediction",
    }
    assert required.issubset(CAPTIONS)


def test_unknown_caption_uses_safe_wording() -> None:
    assert "trained vision model" in get_caption("Unknown stage")
