import base64
from io import BytesIO

from PIL import Image
import torch

from src.visualise import activation_grid_png_base64, fixed_channel_indices, select_fixed_channels


def test_activation_grid_png_base64_returns_png_data_uri() -> None:
    activation = torch.rand(1, 12, 9, 9)

    data_uri = activation_grid_png_base64(activation, max_channels=8, columns=4, tile_size=12, gap=2)

    assert data_uri.startswith("data:image/png;base64,")


def test_fixed_channel_indices_are_stable_and_ordered() -> None:
    assert fixed_channel_indices(100, max_channels=8) == list(range(8))
    assert fixed_channel_indices(4, max_channels=8) == [0, 1, 2, 3]


def test_select_fixed_channels_does_not_reorder_by_activation_strength() -> None:
    maps = torch.zeros(5, 2, 2)
    maps[4] = 100
    maps[0] = 1

    selected = select_fixed_channels(maps, max_channels=3)

    assert selected.shape[0] == 3
    assert torch.equal(selected[0], maps[0])
    assert torch.equal(selected[1], maps[1])
    assert torch.equal(selected[2], maps[2])


def test_compact_average_pool_summary_uses_channel_relative_intensity() -> None:
    activation = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]])

    data_uri = activation_grid_png_base64(
        activation,
        max_channels=4,
        columns=4,
        tile_size=8,
        gap=1,
    )
    image = _decode_png_data_uri(data_uri)

    first_tile = image.getpixel((1 + 4, 1 + 4))
    last_tile = image.getpixel((1 + 3 * (8 + 1) + 4, 1 + 4))

    assert sum(first_tile) < sum(last_tile)


def test_non_zero_constant_compact_average_pool_summary_is_visible() -> None:
    activation = torch.ones(1, 4, 1, 1)

    data_uri = activation_grid_png_base64(
        activation,
        max_channels=4,
        columns=4,
        tile_size=8,
        gap=1,
    )
    image = _decode_png_data_uri(data_uri)
    tile_centre = image.getpixel((1 + 4, 1 + 4))

    assert sum(tile_centre) > 60


def _decode_png_data_uri(data_uri: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert data_uri.startswith(prefix)
    return Image.open(BytesIO(base64.b64decode(data_uri.removeprefix(prefix))))
