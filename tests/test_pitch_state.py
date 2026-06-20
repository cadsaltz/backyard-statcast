from pitch_geometry import build_field_geometry
from pitch_state import (
    PitchConfig,
    PitchDiscardReason,
    PitchEndReason,
    PitchFrameInput,
    PitchMode,
    PitchStartReason,
    PitchStateMachine,
)


def test_pitch_config_defaults():
    cfg = PitchConfig()
    assert cfg.ring_buffer_frames == 4
    assert cfg.backfill_frames == 3
    assert cfg.max_pitch_sec == 20.0
    assert cfg.max_pitch_frames == 120
    assert cfg.tracer_fade_sec == 4.0
    assert cfg.dominance_ratio == 2.0


def test_pitch_modes_exist():
    assert PitchMode.IDLE.value == "idle"
    assert PitchMode.RECORDING.value == "recording"


def _geom():
    import numpy as np
    from calibration import FieldCalibration

    cal = FieldCalibration(
        frame_width=1920,
        frame_height=1080,
        roi=[(0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)],
        strike_zone=[(0.55, 0.51), (0.58, 0.51), (0.58, 0.57), (0.55, 0.57)],
        release_center=(0.45, 0.33),
        release_radius=0.05,
        ignore_mask=np.zeros((1080, 1920), dtype=np.uint8),
    )
    return build_field_geometry(cal, frame_width=1920, frame_height=1080, process_scale=0.5)


def test_start_recording_on_release_motion():
    geom = _geom()
    sm = PitchStateMachine(geom)
    cx, cy = geom.release_cx_proc, geom.release_cy_proc

    # Frame 1: inside release, no prior — buffer only
    sm.update(PitchFrameInput(frame_index=1, timestamp_sec=0.0, x_proc=cx - 5, y_proc=cy, detected=True))
    assert sm.mode == PitchMode.IDLE

    # Frame 2: crossed center with strong rightward motion
    sm.update(
        PitchFrameInput(
            frame_index=2,
            timestamp_sec=0.033,
            x_proc=cx + 8,
            y_proc=cy + 1,
            detected=True,
        )
    )
    assert sm.mode == PitchMode.RECORDING
    assert len(sm.active_samples) >= 2
    assert sm.active_samples[0].frame_index == 1  # backfill


def test_backfill_capped_to_config():
    geom = _geom()
    sm = PitchStateMachine(geom, PitchConfig(backfill_frames=2, ring_buffer_frames=4))
    cx, cy = geom.release_cx_proc, geom.release_cy_proc

    sm.update(PitchFrameInput(1, 0.0, cx - 15, cy, detected=True))
    sm.update(PitchFrameInput(2, 0.03, cx - 10, cy, detected=True))
    sm.update(PitchFrameInput(3, 0.06, cx - 5, cy, detected=True))
    sm.update(
        PitchFrameInput(4, 0.09, cx + 8, cy + 1, detected=True)
    )
    assert sm.mode == PitchMode.RECORDING
    assert [s.frame_index for s in sm.active_samples] == [2, 3, 4]


def _recording_sm():
    geom = _geom()
    sm = PitchStateMachine(geom)
    cx, cy = geom.release_cx_proc, geom.release_cy_proc
    sm.update(PitchFrameInput(1, 0.0, cx - 5, cy, detected=True))
    sm.update(PitchFrameInput(2, 0.03, cx + 8, cy + 1, detected=True))
    assert sm.mode == PitchMode.RECORDING
    return sm, geom


def test_end_on_strike_zone_right():
    sm, geom = _recording_sm()
    right = geom.strike_right_x_proc
    y = (geom.release_cy_proc + geom.strike_bottom_y_proc) // 2
    x = geom.release_cx_proc + 20
    result = None
    for i, fi in enumerate(range(3, 45)):
        x += 4
        result = sm.update(PitchFrameInput(fi, 0.05 * i, x, y, detected=True))
        if result.pitch_completed is not None or result.pitch_discarded:
            break
    assert result is not None
    assert result.pitch_completed is not None
    assert result.pitch_completed.end_reason == PitchEndReason.STRIKE_ZONE_RIGHT
    assert sm.mode == PitchMode.IDLE


def test_end_on_strike_zone_bottom():
    sm, geom = _recording_sm()
    bottom = geom.strike_bottom_y_proc
    x = geom.strike_leading_x_proc + 10
    y = geom.release_cy_proc
    result = None
    for i, fi in enumerate(range(3, 40)):
        y += 4
        result = sm.update(PitchFrameInput(fi, 0.05 * i, x, y, detected=True))
        if result.pitch_completed is not None or result.pitch_discarded:
            break
    assert result is not None
    assert result.pitch_completed is not None
    assert result.pitch_completed.end_reason == PitchEndReason.STRIKE_ZONE_BOTTOM
    assert sm.mode == PitchMode.IDLE


def test_crossed_strike_zone_edges():
    geom = _geom()
    right_line = geom.strike_right_line_proc
    bottom_line = geom.strike_bottom_line_proc
    assert geom.crossed_strike_right_edge(right_line - 5, 100, right_line + 1, 100)
    assert not geom.crossed_strike_right_edge(right_line + 1, 100, right_line + 5, 100)
    assert geom.crossed_strike_bottom_edge(100, bottom_line - 5, 100, bottom_line + 1)
    assert not geom.crossed_strike_bottom_edge(100, bottom_line + 1, 100, bottom_line + 5)


def test_gap_tolerance_does_not_end_pitch():
    sm, geom = _recording_sm()
    cx = geom.release_cx_proc + 20
    sm.update(PitchFrameInput(3, 0.1, cx, geom.release_cy_proc, detected=True))
    sm.update(PitchFrameInput(4, 0.13, detected=False))
    sm.update(PitchFrameInput(5, 0.16, detected=False))
    assert sm.mode == PitchMode.RECORDING
    sm.update(PitchFrameInput(6, 0.19, cx + 10, geom.release_cy_proc, detected=True))
    assert sm.mode == PitchMode.RECORDING


def test_discard_false_start_with_no_forward_progress():
    sm, geom = _recording_sm()
    # Only samples near release — plate pass never reached; force timeout via max frames
    sm.config.max_pitch_frames = 3
    sm.update(PitchFrameInput(3, 0.1, geom.release_cx_proc + 2, geom.release_cy_proc, detected=True))
    result = sm.update(PitchFrameInput(4, 0.13, geom.release_cx_proc + 3, geom.release_cy_proc, detected=True))
    assert result.pitch_discarded is True
    assert result.pitch_completed is None
    assert result.discard_reason == PitchDiscardReason.TIMEOUT


def test_track_lost_terminal_saves_incomplete():
    sm, geom = _recording_sm()
    x = geom.strike_leading_x_proc
    for fi in range(3, 10):
        sm.update(PitchFrameInput(fi, 0.05 * fi, x + fi * 3, geom.release_cy_proc, detected=True))
    sm.config.terminal_lost_frames = 2
    sm.update(PitchFrameInput(20, 1.0, detected=False))
    result = sm.update(PitchFrameInput(21, 1.1, detected=False))
    assert result.pitch_completed is not None
    assert result.pitch_completed.end_reason == PitchEndReason.TRACK_LOST_TERMINAL
    assert result.pitch_completed.complete is False
