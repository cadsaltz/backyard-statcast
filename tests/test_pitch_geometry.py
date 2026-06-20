import pytest

from calibration import FieldCalibration
from pitch_geometry import FieldGeometry, build_field_geometry


@pytest.fixture
def sample_cal() -> FieldCalibration:
    import numpy as np

    return FieldCalibration(
        frame_width=1920,
        frame_height=1080,
        roi=[(0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)],
        strike_zone=[(0.55, 0.51), (0.58, 0.51), (0.58, 0.57), (0.55, 0.57)],
        release_center=(0.45, 0.33),
        release_radius=0.05,
        ignore_mask=np.zeros((1080, 1920), dtype=np.uint8),
    )


def test_build_field_geometry_process_coords(sample_cal: FieldCalibration):
    geom = build_field_geometry(sample_cal, frame_width=1920, frame_height=1080, process_scale=0.5)
    assert geom.proc_w == 960
    assert geom.proc_h == 540
    assert geom.release_cx_proc < geom.strike_leading_x_proc < geom.strike_trailing_x_proc
    assert geom.strike_right_x_proc == geom.strike_trailing_x_proc
    assert geom.strike_right_pad_proc >= 4
    assert geom.strike_bottom_pad_proc >= 4
    assert geom.strike_right_line_proc > geom.strike_right_x_proc
    assert geom.strike_bottom_line_proc > geom.strike_bottom_y_proc
    assert geom.strike_bottom_y_proc > 0
    assert geom.release_r_proc > 0


def test_in_release_zone(sample_cal: FieldCalibration):
    geom = build_field_geometry(sample_cal, frame_width=1920, frame_height=1080, process_scale=0.5)
    assert geom.in_release_zone(geom.release_cx_proc, geom.release_cy_proc)
    assert not geom.in_release_zone(0, 0)
