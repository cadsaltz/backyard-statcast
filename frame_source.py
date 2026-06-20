from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None when the stream has ended."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying capture resources."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        ...


class LiveFrameSource(FrameSource):
    def __init__(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open: {source}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def is_live(self) -> bool:
        return True

    def read(self) -> np.ndarray | None:
        while True:
            ok, frame = self._cap.read()
            if ok:
                return frame
            if not self._cap.isOpened():
                return None
            time.sleep(0.01)

    def release(self) -> None:
        self._cap.release()


class FileFrameSource(FrameSource):
    def __init__(self, path: str) -> None:
        if not Path(path).is_file():
            raise SystemExit(f"File not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def is_live(self) -> bool:
        return False

    @property
    def frame_count(self) -> int | None:
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None

    @property
    def frame_index(self) -> int:
        return max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if ok:
            return frame
        return None

    def seek(self, index: int) -> np.ndarray | None:
        """Read frame at index (0-based). Returns None if out of range."""
        index = max(0, index)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if ok:
            return frame
        return None

    def release(self) -> None:
        self._cap.release()


def _resolve_live_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    return source


def open_frame_source(
    source: str,
    width: int | None = None,
    height: int | None = None,
) -> FrameSource:
    if Path(source).is_file():
        return FileFrameSource(source)
    live_source = _resolve_live_source(source)
    return LiveFrameSource(live_source, width, height)
