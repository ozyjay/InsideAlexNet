from PIL import Image

from src.preprocess import preprocess_for_alexnet


def test_preprocess_output_shape() -> None:
    image = Image.new("RGB", (320, 240), color=(120, 80, 40))

    tensor = preprocess_for_alexnet(image)

    assert tuple(tensor.shape) == (1, 3, 224, 224)
