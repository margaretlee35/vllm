# SPDX-License-Identifier: Apache-2.0
"""Registry for visual-token pruning strategies.

Each strategy implements :class:`VisualPruningStrategy` and is registered
at import time via :func:`register_visual_pruning_strategy`.  Model
wrappers retrieve a strategy by name with :func:`get_visual_pruning_strategy`.

Example usage (inside a strategy module like ``vscan.py``)::

    from vllm.multimodal.visual_token_pruning.registry import (
        VisualPruningStrategy,
        register_visual_pruning_strategy,
    )

    class MyStrategy(VisualPruningStrategy):
        name = "my_strategy"
        ...

    register_visual_pruning_strategy(MyStrategy())

Example usage (inside a model wrapper)::

    from vllm.multimodal.visual_token_pruning.registry import (
        get_visual_pruning_strategy,
    )

    strategy = get_visual_pruning_strategy("my_strategy")
    if strategy is not None and strategy.is_applicable(self):
        pruned = strategy.prune(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from vllm.config.multimodal import MultiModalConfig

# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class VisualPruningStrategy(ABC):
    """Base class for visual-token pruning strategies.

    Subclasses must set a ``name`` class attribute and implement
    :meth:`is_applicable` and :meth:`prune`.
    """

    name: str

    @abstractmethod
    def is_applicable(self, model: nn.Module) -> bool:
        """Return True if this strategy should activate for *model*."""
        ...

    @abstractmethod
    def prune(
        self,
        *,
        model: nn.Module,
        modality: str,
        embeds_split: tuple[Tensor, ...],
        aux: dict[str, Any],
        config: MultiModalConfig,
    ) -> tuple[Tensor, ...]:
        """Prune visual-token embeddings at embed time.

        Args:
            model: The outer multimodal model instance.
            modality: ``"image"`` or ``"video"``.
            embeds_split: Per-item embedding tensors (one per image/video).
            aux: Auxiliary data (e.g. pre-computed scores).
            config: The multimodal configuration.

        Returns:
            Pruned embedding tensors, same length as *embeds_split*.
        """
        ...


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, VisualPruningStrategy] = {}


def register_visual_pruning_strategy(
    strategy: VisualPruningStrategy,
) -> None:
    """Register a strategy instance under ``strategy.name``."""
    _REGISTRY[strategy.name] = strategy


def get_visual_pruning_strategy(
    name: str,
) -> VisualPruningStrategy | None:
    """Look up a registered strategy by name, or return ``None``."""
    return _REGISTRY.get(name)
