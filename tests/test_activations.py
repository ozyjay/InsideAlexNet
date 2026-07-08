import torch

from src.activations import ActivationCapture, LayerSpec, SELECTED_LAYER_NAMES


def test_activation_capture_records_named_layer() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, kernel_size=1), torch.nn.ReLU())
    spec = LayerSpec(
        key="early",
        label="Early layer",
        module_path="1",
        caption_key="Early layer",
        public_note="test",
    )

    with ActivationCapture(model, specs=(spec,)) as capture:
        model(torch.rand(1, 3, 8, 8))

    assert tuple(capture.activations["early"].shape) == (1, 4, 8, 8)


def test_selectable_layer_names_cover_alexnet_path() -> None:
    assert SELECTED_LAYER_NAMES == [
        "Conv 1",
        "Pool 1",
        "Conv 2",
        "Pool 2",
        "Conv 3",
        "Conv 4",
        "Conv 5",
        "Pool 5",
        "Avg pool",
    ]
