"""Ball detection wrapper used by the main tracker viewer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from detection_strategies import BallDetection, CombinedStrategy, StrategyConfig

__all__ = ["BallDetection", "BallTracker"]


class BallTracker:
    """Combined MOG2 + brightness strategy with warmup support."""

    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 2500,
        min_circularity: float = 0.55,
        brightness_threshold: int = 175,
        warmup_frames: int = 45,
        max_jump_px: int = 120,
    ) -> None:
        self.warmup_frames = warmup_frames
        self._strategy = CombinedStrategy(
            StrategyConfig(
                min_area=min_area,
                max_area=max_area,
                min_circularity=min_circularity,
                brightness_threshold=brightness_threshold,
                warmup_frames=warmup_frames,
                max_jump_px=max_jump_px,
            )
        )

    def reset(self) -> None:
        self._strategy.reset()

    @property
    def is_warming_up(self) -> bool:
        return self._strategy.is_warming_up

    @property
    def warmup_progress(self) -> tuple[int, int]:
        return self._strategy.warmup_progress

    def detect(self, frame: np.ndarray) -> Optional[BallDetection]:
        result = self._strategy.detect(frame)
        return result.detection
