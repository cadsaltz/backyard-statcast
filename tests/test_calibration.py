import json
from pathlib import Path

import numpy as np
import pytest

from calibration import FieldCalibration, load_calibration, save_calibration


def test_field_calibration_round_trip(tmp_path: Path):
    ignore = np.zeros((720, 1280), dtype=np.uint8)
    ignore[100:200, 300:400] = 255

    cal = FieldCalibration(
        frame_width=1280,
        frame_height=720,
        roi=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)],
        strike_zone=[(0.4, 0.5), (0.6, 0.5), (0.6, 0.8), (0.4, 0.8)],
        release_center=(0.5, 0.15),
        release_radius=0.08,
        ignore_mask=ignore,
    )

    json_path = tmp_path / "field.json"
    save_calibration(cal, json_path)
    loaded = load_calibration(json_path)

    assert loaded.frame_width == 1280
    assert loaded.frame_height == 720
    assert loaded.roi == cal.roi
    assert loaded.strike_zone == cal.strike_zone
    assert loaded.release_center == cal.release_center
    assert loaded.release_radius == pytest.approx(0.08)
    assert loaded.ignore_mask.shape == (720, 1280)
    assert loaded.ignore_mask[150, 350] == 255


def test_load_missing_ignore_png(tmp_path: Path):
    json_path = tmp_path / "field.json"
    json_path.write_text(
        json.dumps(
            {
                "frame_width": 640,
                "frame_height": 480,
                "roi": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "strike_zone": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
                "release_center": [0.5, 0.1],
                "release_radius": 0.05,
            }
        )
    )
    loaded = load_calibration(json_path)
    assert loaded.ignore_mask.shape == (480, 640)
    assert loaded.ignore_mask.max() == 0
