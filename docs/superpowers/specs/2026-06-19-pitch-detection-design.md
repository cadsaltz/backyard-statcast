# Pitch Detection — Design Spec

**Date:** 2026-06-19  
**Status:** Approved (design conversation)  
**Scope:** Pitch event state machine, trajectory recording, live tracer overlay for `track_ball.py`

## Problem

The tracker finds the ball frame-to-frame (`TrackState`: SEARCH/TRACK) but does not define pitch boundaries. The existing `pitch_active` flag only latches when the ball enters the ROI — too coarse for trajectory capture, analytics, or reliable pitch segmentation.

With only ~10–15 visible frames per pitch, the system must enter recording within 1–2 frames of release and tolerate brief tracking dropouts without truncating or invalidating the pitch prematurely.

## Goal

Introduce a pitch state machine **orthogonal** to `TrackState` that:

1. Enters **RECORDING** quickly using release-area motion heuristics
2. Appends timestamped samples with tracking metadata during flight
3. Exits on impact, plate crossing, or terminal track loss
4. Validates trajectories in **FINALIZING** (accept, save incomplete, or discard)
5. Draws a **live flight-path tracer** during RECORDING and a **fading tracer** after return to IDLE

No polynomial fitting, velocity estimation, or pitch classification runs during live tracking.

## Non-Goals

- Homography / 3D reconstruction
- Live pitch analytics (fit, velocity, break, classification)
- Persisting pitches to disk (optional follow-up; in-memory + console log in v1)
- Changing detection thresholds or `TrackState` behavior
- Gating detection by release or strike zones

## Architecture

Three layers, same pattern as search/track:

| Module | Responsibility |
|--------|----------------|
| `pitch_geometry.py` | Strike/release bounds in pixel coords from calibration |
| `pitch_state.py` | `PitchStateMachine`, `Pitch`, `PitchSample`, transitions, validation |
| `pitch_tracer.py` | Live polyline during RECORDING; fading polylines after finalize |
| `track_ball.py` | Wire tracker output → pitch machine → tracer overlay |

```
TrackResult + Classification
        │
        ▼
PitchStateMachine.update()
        │
        ├── RECORDING → append PitchSample; tracer.add_point()
        └── FINALIZING → validate Pitch; tracer.finalize_path()
                │
                ▼
        pitch_tracer.draw(frame, now)   # live + fading (visual only)
```

## State Machine

```
IDLE ──► RECORDING ──► FINALIZING ──► IDLE
  ▲          │               │
  └──────────┴───────────────┘  (discard → IDLE)
```

| State | Behavior |
|-------|----------|
| **IDLE** | Ring-buffer valid detections; evaluate start trigger; draw fading tracers only |
| **RECORDING** | Append samples; evaluate end triggers; draw **live** tracer on every frame |
| **FINALIZING** | Run validation; finalize or discard `Pitch`; hand path to fade buffer; enter cooldown |

`FINALIZING` completes in the same frame (no multi-frame finalize state required).

## Enter RECORDING (fast start)

On each frame with a valid detection in process coords, compute:

```text
dx = x_t - x_{t-1}
dy = y_t - y_{t-1}
```

**Release crossed:** `x >= release_cx`

**Rightward motion:** `dx >= min_dx`

**Horizontal dominance:** `abs(dx) >= dominance_ratio * max(abs(dy), min_dy)`

**In or exiting release:** center inside release circle × `release_expand`, OR was inside previous frame and now outside with `dx > 0`

**Start when:**

```text
(rightward_motion AND horizontal_dominance AND release_crossed)
AND (in_release OR just_exited OR x < strike_leading_x)
AND NOT in_cooldown
```

Enter on **first qualifying frame**. Prepend ring buffer (`ring_buffer_frames`, default 10).

## Exit RECORDING (priority order)

1. **Impact / deflection** (hit or strike contact): angle between velocity vectors ≥ 45° for 2 frames, OR ≥ 60° for 1 frame; OR speed ratio `< 0.5`. End reasons: `STRIKE_CONTACT`, `HIT_OR_DEFLECTION`.
2. **Plate passed:** `x >= strike_trailing_x + pass_margin_px` → `PLATE_PASSED`
3. **Terminal track loss:** in approach zone (`x >= strike_leading_x - approach_margin`) and no detection for `terminal_lost_frames` → `TRACK_LOST_TERMINAL`
4. **Timeout:** `max_pitch_frames` or `max_pitch_sec` → `TIMEOUT`

## Gap tolerance (during RECORDING)

- Allow up to `max_gap_frames` (3) consecutive non-detections without ending
- On reacquire: accept if distance to predicted position ≤ `max_jump_px + speed × gap × slack`
- Sustained implausible reacquire (`implausible_streak >= 5`) → stop and discard

## Validation (FINALIZING)

**Reject** if: too few samples, insufficient forward progress, no release association, extended reverse motion, excessive gaps.

**Accept** (`complete=True`): impact or plate-pass end with valid trajectory.

**Accept incomplete** (`complete=False`): `TRACK_LOST_TERMINAL` after approach zone with valid partial path.

**Discard:** false starts, timeout, implausible reacquire streak.

## Tracer (visual only)

| Phase | Drawing |
|-------|---------|
| RECORDING | Cyan polyline through all recorded full-frame points; small dots at each sample; does not affect pitch logic |
| IDLE after pitch | Path moved to fade list; alpha decays linearly over `tracer_fade_sec` (4 s); removed when alpha ≤ 0 |

Multiple faded paths may coexist. Valid pitches use cyan→transparent; discarded pitches may use gray (optional).

## Replace `pitch_active`

- Red dot + `PITCH` status when `PitchMode.RECORDING`
- Yellow dot otherwise (Choice B unchanged)
- Remove `PITCH_IDLE_FRAMES` / ROI latch for pitch semantics; ROI remains in classification metadata only

## Default parameters

See `PitchConfig` in implementation plan (`pitch_state.py`).

## Success Criteria

- [ ] Recording starts within 1–2 frames of release heuristic
- [ ] Live tracer visible during entire RECORDING state
- [ ] Tracer fades over ~4 s after pitch ends
- [ ] Brief 1–3 frame dropouts do not end or invalidate pitch
- [ ] Balls missing strike zone vertically end via plate-pass or terminal loss
- [ ] Impact ends recording for hit / strike-contact cases
- [ ] Invalid pitches discarded without saving
- [ ] Unit tests cover start, end, gaps, validation without video
- [ ] `TrackState` and detection unchanged
