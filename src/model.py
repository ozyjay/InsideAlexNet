"""AlexNet model loading, prediction, and activation analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image
import torch

from src.activations import ActivationCapture, SELECTED_LAYER_SPECS
from src.preprocess import preprocess_for_alexnet
from src.visualise import activation_grid_png_base64


class ModelUnavailableError(RuntimeError):
    """Raised when live AlexNet inference is not available locally."""


@dataclass(frozen=True)
class Prediction:
    """A public top-k prediction result."""

    label: str
    probability: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-safe representation."""
        return {"label": self.label, "probability": self.probability}


@dataclass(frozen=True)
class ActivationVisualisation:
    """A rendered public visualisation for one selected AlexNet layer."""

    key: str
    label: str
    caption_key: str
    note: str
    tensor_shape: tuple[int, ...]
    image_data: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "key": self.key,
            "label": self.label,
            "caption_key": self.caption_key,
            "note": self.note,
            "tensor_shape": self.tensor_shape,
            "image_data": self.image_data,
        }


@dataclass(frozen=True)
class AlexNetAnalysis:
    """Live AlexNet predictions and selected layer visualisations."""

    predictions: list[Prediction]
    visualisations: list[ActivationVisualisation]


@dataclass(frozen=True)
class AlexNetBundle:
    """Loaded AlexNet model and label metadata."""

    model: torch.nn.Module
    categories: tuple[str, ...]


@lru_cache(maxsize=1)
def load_alexnet() -> AlexNetBundle:
    """Load pretrained torchvision AlexNet in eval mode.

    If the weights are not already available and cannot be downloaded, this
    raises `ModelUnavailableError` so the UI can display setup instructions
    instead of crashing.
    """
    try:
        from torchvision.models import AlexNet_Weights, alexnet

        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)
        model.eval()
        categories = tuple(weights.meta.get("categories", ()))
    except Exception as exc:  # pragma: no cover - exact failures depend on local cache/network state.
        raise ModelUnavailableError(
            "Pretrained AlexNet weights are unavailable locally. Run the setup on a networked machine once, "
            "or use precomputed fallback replay assets when they are available."
        ) from exc

    if not categories:
        raise ModelUnavailableError("AlexNet loaded, but ImageNet label metadata was unavailable.")

    return AlexNetBundle(model=model, categories=categories)


def run_alexnet_top5(image: Image.Image, top_k: int = 5) -> list[Prediction]:
    """Run live AlexNet inference and return top-k ImageNet predictions."""
    analysis = run_alexnet_analysis(image=image, top_k=top_k, include_visualisations=False)
    return analysis.predictions


def run_alexnet_analysis(
    image: Image.Image,
    *,
    top_k: int = 5,
    include_visualisations: bool = True,
) -> AlexNetAnalysis:
    """Run AlexNet once and return predictions plus selected activation grids."""
    bundle = load_alexnet()
    input_tensor = preprocess_for_alexnet(image)

    with torch.inference_mode(), ActivationCapture(bundle.model) as capture:
        logits = bundle.model(input_tensor)

    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
    predictions = _format_predictions(bundle.categories, top_probabilities, top_indices)

    visualisations: list[ActivationVisualisation] = []
    if include_visualisations:
        for spec in SELECTED_LAYER_SPECS:
            activation = capture.activations.get(spec.key)
            if activation is None:
                continue
            visualisations.append(
                ActivationVisualisation(
                    key=spec.key,
                    label=spec.label,
                    caption_key=spec.caption_key,
                    note=spec.public_note,
                    tensor_shape=tuple(int(dim) for dim in activation.shape),
                    image_data=activation_grid_png_base64(activation),
                )
            )

    return AlexNetAnalysis(predictions=predictions, visualisations=visualisations)


def _format_predictions(
    categories: tuple[str, ...],
    top_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist(), strict=True):
        label = categories[index] if index < len(categories) else f"class {index}"
        predictions.append(Prediction(label=label, probability=float(probability)))
    return predictions
