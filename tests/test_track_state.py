from track_state import (
    TrackConfig,
    TrackMode,
    TrackState,
    padded_point_rect,
    search_window_rect,
)


def test_velocity_prediction():
    ts = TrackState()
    ts.accept_search_peak(10, 20, 100.0)
    ts.accept_search_peak(20, 30, 100.0)
    assert ts.velocity() == (10, 10)
    assert ts.predicted() == (30, 40)


def test_slow_motion_prediction_stays_on_last():
    ts = TrackState(TrackConfig(slow_speed_px=5.0))
    ts.accept_search_peak(10, 10, 50.0)
    ts.accept_search_peak(12, 10, 50.0)
    assert ts.is_slow_motion()
    assert ts.predicted() == (12, 10)


def test_search_window_rect_contains_last():
    rect = search_window_rect(
        (50, 50),
        (80, 50),
        min_pad=32,
        margin=24,
        slow=False,
        extra_pad=0,
        frame_w=200,
        frame_h=200,
    )
    x0, y0, x1, y1 = rect
    assert x0 <= 50 <= x1
    assert y0 <= 50 <= y1
    assert x0 <= 80 <= x1


def test_slow_search_window_centered_on_last():
    ts = TrackState()
    ts.accept_search_peak(40, 60, 50.0)
    ts.accept_search_peak(42, 60, 50.0)
    x0, y0, x1, y1 = ts.search_window(200, 200)
    assert x0 <= 42 <= x1
    assert y0 <= 60 <= y1
    assert x1 - x0 == y1 - y0 == 64


def test_search_window_clamped_to_frame():
    ts = TrackState()
    ts.accept_search_peak(5, 5, 50.0)
    ts.accept_search_peak(10, 10, 50.0)
    x0, y0, x1, y1 = ts.search_window(40, 30)
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 40 and y1 <= 30


def test_search_to_track_requires_two_detections():
    ts = TrackState()
    ts.accept_search_peak(10, 10, 80.0)
    assert ts.mode == TrackMode.SEARCH
    ts.accept_search_peak(15, 15, 90.0)
    assert ts.mode == TrackMode.TRACK
    assert ts._search_lock_density == 90.0


def test_search_streak_resets_on_miss():
    ts = TrackState()
    ts.accept_search_peak(10, 10, 80.0)
    ts.search_miss()
    ts.accept_search_peak(20, 20, 70.0)
    assert ts.mode == TrackMode.SEARCH


def test_track_to_search_after_misses():
    ts = TrackState(TrackConfig(coast_frames=2, track_lost_frames=3))
    ts.accept_search_peak(10, 10, 80.0)
    ts.accept_search_peak(25, 10, 90.0)
    assert ts.track_miss() == (25, 10)
    assert ts.is_coasting
    assert ts.track_miss() == (25, 10)
    assert ts.track_miss() is None
    assert ts.mode == TrackMode.SEARCH
    assert ts.search_prior == (25, 10)


def test_slow_track_requires_more_misses():
    ts = TrackState(TrackConfig(slow_track_lost_frames=5, coast_frames=2))
    ts.accept_search_peak(10, 10, 80.0)
    ts.accept_search_peak(12, 10, 90.0)
    assert ts.is_slow_motion()
    assert ts.track_miss() == (12, 10)
    assert ts.track_miss() == (12, 10)
    assert ts.track_miss() is None
    assert ts.mode == TrackMode.TRACK
    assert ts.track_miss() is None
    assert ts.track_miss() is None
    assert ts.mode == TrackMode.SEARCH


def test_displacement_gate_accepts_near_last_not_pred():
    ts = TrackState(TrackConfig(max_jump_px=20))
    ts.accept_search_peak(0, 0, 100.0)
    ts.accept_search_peak(50, 0, 100.0)
    ts._frames_in_track = 1
    assert ts.check_track_peak(55, 0, 100.0) is True
    assert ts.check_track_peak(200, 0, 100.0) is False


def test_density_ratio_skipped_when_slow():
    ts = TrackState(TrackConfig(track_min_density_ratio=0.4, slow_speed_px=5.0))
    ts.accept_search_peak(0, 0, 100.0)
    ts.accept_search_peak(2, 0, 100.0)
    ts._frames_in_track = 1
    assert ts.is_slow_motion()
    assert ts.check_track_peak(3, 0, 10.0) is True


def test_density_ratio_rejects_weak_peak_when_fast():
    ts = TrackState(TrackConfig(track_min_density_ratio=0.4, slow_speed_px=5.0))
    ts.accept_search_peak(0, 0, 100.0)
    ts.accept_search_peak(20, 0, 100.0)
    ts._frames_in_track = 1
    assert not ts.is_slow_motion()
    assert ts.check_track_peak(22, 0, 30.0) is False
    assert ts.check_track_peak(22, 0, 40.0) is True


def test_coast_emits_last_position():
    ts = TrackState(TrackConfig(coast_frames=2, track_lost_frames=3))
    ts.accept_search_peak(5, 5, 50.0)
    ts.accept_search_peak(20, 5, 50.0)
    pos = ts.track_miss()
    assert pos == (20, 5)
    assert ts.is_coasting


def test_coast_widens_window_pad():
    ts = TrackState(TrackConfig(coast_pad_boost=8))
    ts.accept_search_peak(50, 50, 50.0)
    ts.accept_search_peak(80, 50, 50.0)
    base = ts.extra_window_pad()
    ts.track_miss()
    assert ts.extra_window_pad() == base + 8


def test_reset_clears_history_and_prior():
    ts = TrackState()
    ts.accept_search_peak(1, 2, 50.0)
    ts.accept_search_peak(3, 4, 60.0)
    ts.force_search(clear_prior=False)
    assert ts.search_prior == (3, 4)
    ts.reset()
    assert ts.mode == TrackMode.SEARCH
    assert ts.last_position() is None
    assert ts.search_prior is None


def test_relock_with_prior_needs_one_detection():
    ts = TrackState(TrackConfig(search_relock_streak=1))
    ts.accept_search_peak(10, 10, 80.0)
    ts.accept_search_peak(20, 10, 90.0)
    ts.force_search(clear_prior=False)
    ts.accept_search_peak(21, 10, 85.0)
    assert ts.mode == TrackMode.TRACK


def test_prior_search_window():
    ts = TrackState(TrackConfig(search_prior_pad=40))
    ts._search_prior = (100, 100)
    x0, y0, x1, y1 = ts.prior_search_window(300, 300)
    assert x0 == 60 and y0 == 60
    assert x1 == 140 and y1 == 140


def test_fast_motion_rect_elongates_along_velocity():
    ts = TrackState()
    ts.accept_search_peak(10, 50, 50.0)
    ts.accept_search_peak(60, 50, 50.0)
    x0, y0, x1, y1 = ts.search_window(200, 200)
    width = x1 - x0
    height = y1 - y0
    assert width > height


def test_padded_point_rect():
    rect = padded_point_rect(10, 20, 5, 100, 100)
    assert rect == (5, 15, 15, 25)


def test_qualify_search_lock_rejects_small_blob():
    ts = TrackState()
    ts.set_roi_gating(True)
    assert not ts.qualify_search_lock(
        density_score=50.0,
        blob_area=5.0,
        in_roi=True,
        has_motion=True,
    )


def test_qualify_search_lock_rejects_outside_roi():
    ts = TrackState()
    ts.set_roi_gating(True)
    assert not ts.qualify_search_lock(
        density_score=50.0,
        blob_area=20.0,
        in_roi=False,
        has_motion=True,
    )


def test_register_search_hit_outside_roi_does_not_enter_track():
    ts = TrackState()
    ts.set_roi_gating(True)
    assert not ts.register_search_hit(
        10, 10, 50.0, 20.0, in_roi=False, has_motion=True
    )
    assert ts.mode == TrackMode.SEARCH
    assert ts.register_search_hit(
        10, 10, 50.0, 20.0, in_roi=True, has_motion=True
    )
    assert ts.mode == TrackMode.SEARCH
    assert ts.register_search_hit(
        12, 12, 55.0, 22.0, in_roi=True, has_motion=True
    )
    assert ts.mode == TrackMode.TRACK


def test_roi_exit_forces_search():
    ts = TrackState(TrackConfig(roi_exit_frames=3))
    ts.set_roi_gating(True)
    ts.accept_search_peak(10, 10, 80.0)
    ts.accept_search_peak(15, 15, 90.0)
    assert ts.update_track_health(in_roi=False, blob_area=20.0) is True
    assert ts.update_track_health(in_roi=False, blob_area=20.0) is True
    assert ts.update_track_health(in_roi=False, blob_area=20.0) is False
    assert ts.mode == TrackMode.SEARCH
    assert ts.search_prior is None


def test_stuck_lock_forces_search():
    ts = TrackState(TrackConfig(stuck_frames=3, stuck_max_area=8.0))
    ts.accept_search_peak(10, 10, 80.0)
    ts.accept_search_peak(11, 10, 90.0)
    assert ts.is_slow_motion()
    assert ts.update_track_health(in_roi=True, blob_area=5.0) is True
    assert ts.update_track_health(in_roi=True, blob_area=5.0) is True
    assert ts.update_track_health(in_roi=True, blob_area=5.0) is False
    assert ts.mode == TrackMode.SEARCH
    assert ts.search_prior is None


def test_qualify_track_rejects_small_blob():
    ts = TrackState()
    ts.accept_search_peak(0, 0, 100.0)
    ts.accept_search_peak(10, 0, 100.0)
    assert not ts.qualify_track_peak(
        11, 0, 80.0, 4.0, in_roi=True
    )
    assert ts.qualify_track_peak(
        11, 0, 80.0, 12.0, in_roi=True
    )
