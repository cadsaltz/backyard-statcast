import numpy as np

from calibration import FieldCalibration
from spatial_filter import (
    classify_point,
    ignore_mask_for_process,
    is_ignored,
    point_in_circle,
    point_in_polygon,
    resize_ignore_mask,
    scaled_polygons,
)


def _sample_cal() -> FieldCalibration:
    ignore = np.zeros((100, 200), dtype=np.uint8)
    ignore[40:60, 80:120] = 255
    return FieldCalibration(
        frame_width=200,
        frame_height=100,
        roi=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        strike_zone=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        release_center=(0.5, 0.1),
        release_radius=0.1,
        ignore_mask=ignore,
    )


def test_point_in_polygon():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((0.5, 0.5), square) is True
    assert point_in_polygon((1.5, 0.5), square) is False


def test_point_in_circle():
    assert point_in_circle((0.5, 0.1), (0.5, 0.1), 0.1, frame_width=200, frame_height=100)
    assert not point_in_circle((0.9, 0.9), (0.5, 0.1), 0.05, frame_width=200, frame_height=100)


def test_is_ignored():
    cal = _sample_cal()
    assert is_ignored((100, 50), cal) is True
    assert is_ignored((10, 10), cal) is False


def test_classify_point():
    cal = _sample_cal()
    cls = classify_point((100, 50), cal)
    assert cls.ignored is True
    assert cls.in_roi is False

    cls2 = classify_point((100, 50), cal, ignore_check=False)
    assert cls2.ignored is False
    assert cls2.in_roi is True


def test_resize_ignore_mask():
    cal = _sample_cal()
    small = resize_ignore_mask(cal.ignore_mask, (100, 50))
    assert small.shape == (50, 100)


def test_scaled_polygons():
    cal = _sample_cal()
    roi_px = scaled_polygons(cal.roi, frame_width=200, frame_height=100)
    assert roi_px[0] == (0, 0)
    assert roi_px[2] == (200, 100)


def test_classify_point_runtime_frame_size():
    cal = _sample_cal()
    cls = classify_point((100, 50), cal, frame_width=400, frame_height=200, ignore_check=False)
    assert cls.in_roi is True


def test_ignore_mask_for_process_scales_to_runtime():
    cal = _sample_cal()
    proc = ignore_mask_for_process(cal, frame_width=400, frame_height=200, process_scale=0.5)
    assert proc.shape == (100, 200)
