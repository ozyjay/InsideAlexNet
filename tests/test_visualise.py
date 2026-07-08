import torch

from src.visualise import activation_grid_png_base64


def test_activation_grid_png_base64_returns_png_data_uri() -> None:
    activation = torch.rand(1, 12, 9, 9)

    data_uri = activation_grid_png_base64(activation, max_channels=8, columns=4, tile_size=12, gap=2)

    assert data_uri.startswith("data:image/png;base64,")
