"""Activation capture helpers for the selectable AlexNet layers."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import torch


@dataclass(frozen=True)
class LayerSpec:
    """A public demo layer to capture and explain."""

    key: str
    label: str
    module_path: str
    caption_key: str
    public_note: str


SELECTED_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="conv1",
        label="Conv 1",
        module_path="features.1",
        caption_key="Conv 1",
        public_note="First convolution response after ReLU, shown as fixed feature-map channels so grid positions stay stable.",
    ),
    LayerSpec(
        key="pool1",
        label="Pool 1",
        module_path="features.2",
        caption_key="Pool 1",
        public_note="First max-pooling output, where nearby strong responses are kept and the map becomes smaller.",
    ),
    LayerSpec(
        key="conv2",
        label="Conv 2",
        module_path="features.4",
        caption_key="Conv 2",
        public_note="Second convolution response after ReLU, where simple patterns are combined into richer local features.",
    ),
    LayerSpec(
        key="pool2",
        label="Pool 2",
        module_path="features.5",
        caption_key="Pool 2",
        public_note="Second max-pooling output, preserving strong responses in a smaller spatial grid.",
    ),
    LayerSpec(
        key="conv3",
        label="Conv 3",
        module_path="features.7",
        caption_key="Conv 3",
        public_note="Third convolution response after ReLU, combining earlier patterns into more complex textures and parts.",
    ),
    LayerSpec(
        key="conv4",
        label="Conv 4",
        module_path="features.9",
        caption_key="Conv 4",
        public_note="Fourth convolution response after ReLU, continuing to combine useful visual patterns.",
    ),
    LayerSpec(
        key="conv5",
        label="Conv 5",
        module_path="features.11",
        caption_key="Conv 5",
        public_note="Final convolution response after ReLU, before the classifier layers.",
    ),
    LayerSpec(
        key="pool5",
        label="Pool 5",
        module_path="features.12",
        caption_key="Pool 5",
        public_note="Final max-pooling output, a compact spatial summary passed toward the classifier.",
    ),
    LayerSpec(
        key="avgpool",
        label="Avg pool",
        module_path="avgpool",
        caption_key="Avg pool",
        public_note="Adaptive average-pooling output, shaped into the fixed grid expected by AlexNet’s classifier.",
    ),
)

SELECTED_LAYER_NAMES = [spec.label for spec in SELECTED_LAYER_SPECS]


def get_module_by_path(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """Return a nested module using a dotted path such as `features.4`."""
    module: torch.nn.Module = model
    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class ActivationCapture:
    """Context manager that captures selected layer outputs during one forward pass."""

    def __init__(self, model: torch.nn.Module, specs: tuple[LayerSpec, ...] = SELECTED_LAYER_SPECS) -> None:
        self.model = model
        self.specs = specs
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationCapture":
        for spec in self.specs:
            module = get_module_by_path(self.model, spec.module_path)
            self._handles.append(module.register_forward_hook(self._make_hook(spec.key)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, key: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[key] = output.detach().cpu()

        return hook
