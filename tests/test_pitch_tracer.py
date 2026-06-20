import numpy as np

from pitch_tracer import PitchTracer, tracer_alpha


def test_tracer_alpha_fades_to_zero():
    assert tracer_alpha(created_at=0.0, now=4.0, fade_sec=4.0) == 0.0
    assert tracer_alpha(created_at=0.0, now=0.0, fade_sec=4.0) == 1.0
    assert 0.0 < tracer_alpha(created_at=0.0, now=2.0, fade_sec=4.0) < 1.0


def test_live_points_cleared_on_finalize():
    tracer = PitchTracer(fade_sec=4.0)
    tracer.add_point(10, 20)
    tracer.add_point(30, 40)
    assert len(tracer.live_points_full) == 2
    tracer.finalize_path(now=1.0, valid=True)
    assert tracer.live_points_full == []
    assert len(tracer.fading_paths) == 1
