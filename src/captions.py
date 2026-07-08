"""Editable public captions for demo stages and AlexNet layers."""

from __future__ import annotations

CAPTIONS: dict[str, str] = {
    "Input": "The image is resized and normalised before entering the network.",
    "Early layer": "Early layers often respond to simple patterns such as edges, colour changes, and corners.",
    "Middle layer": "Middle layers combine simpler patterns into textures, curves, and repeated shapes.",
    "Deep layer": "Deeper layers respond to combinations of features that may be useful for recognising objects.",
    "Prediction": "The final prediction is a likely class from the model’s training labels. It can be wrong.",
}


def get_caption(key: str) -> str:
    """Return a public caption for a stage, falling back to a safe default."""
    return CAPTIONS.get(key, "This shows how a trained vision model responds at this stage.")
