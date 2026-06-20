# Spatial Calibration — Design Spec

**Date:** 2026-06-19  
**Status:** Implemented (2026-06-19)  
**Scope:** Pre-tracking calibration UI and spatial filtering for `track_ball.py`

## Problem

The density-based ball tracker runs on the full frame and is affected by background motion (trees, fences, glare). There is no way to define spatial regions before tracking starts.

## Goal

Before tracking begins, freeze on the first frame and let the user define spatial regions. After pressing Start, tracking uses those regions to filter false positives while still running detection on the full frame.

## Calibration Regions

| Region | Required | Shape | Detection use |
|--------|----------|-------|---------------|
| ROI | Yes | 4-point polygon | Pitch-active gating; prefer candidates when multiple exist |
| Ignore zones | No (multiple) | Painted bitmap | Hard reject — never accept detections inside |
| Strike zone | Yes | 4-point polygon | Overlay only (future classification) |
| Release zone | Yes | Circle (center + radius) | Overlay + future trajectory origin; no gating in v1 |

## Tracking Pipeline

```
full-frame detection (unchanged)
  → reject if center inside ignore mask
  → if multiple candidates: prefer ROI (future / strategy path)
  → always accept remaining detection for display (Choice B)
  → set pitch_active when detection center enters ROI
```

### Choice B — Validation tracking

Always show the tracked dot when a valid detection exists (after ignore filtering), even outside ROI. This acts as a live “is the tracker working?” indicator without pitching.

`pitch_active` is a separate boolean used for future pitch logic. It turns **True** when the detection center enters the ROI polygon. It **stays** True while tracking continues—even if the ball later leaves ROI mid-flight. It turns **False** only after 30 consecutive frames with no valid detection, or when the user presses `b` (background relearn / reset).

### Visual encoding

| State | Dot color | Meaning |
|-------|-----------|---------|
| Valid detection, pitch inactive | Yellow | Tracker sees something; not a pitch event |
| Valid detection, pitch active | Red (current) | Detection inside ROI — pitch event active |
| No detection | None | No candidate after ignore filter |

## Non-Goals

- Perspective transforms / homography
- Formal pitch state machine
- ROI cropping of detection input
- Strike-zone or release-zone detection gating
- Refactoring `detection_strategies.py` / `test_detection_strategies.py` (optional follow-up)

## Coordinate Storage

All geometry stored in **normalized coordinates** (0.0–1.0 relative to calibration frame width/height). Ignore mask stored as a PNG sidecar at calibration frame resolution. At runtime, scale masks and test points against process-resolution frames inside `BallTracker.process()`.

## Module Layout

```
calibration.py       FieldCalibration dataclass, load/save JSON + ignore PNG
spatial_filter.py    point-in-polygon, circle test, mask scaling, candidate filter
calibration_ui.py    First-frame freeze, mouse drawing, Start key
track_ball.py        Calibration phase → tracking loop with overlays
tests/               Unit tests for geometry and serialization
```

## Success Criteria

- [ ] First frame freezes; user can define all four region types before tracking
- [ ] Ignore zones eliminate detections in painted areas
- [ ] Dot visible anywhere (outside ROI) when tracker finds a valid point — Choice B
- [ ] `pitch_active` only when detection center is inside ROI
- [ ] Strike zone and release zone drawn as overlays
- [ ] Calibration persists to disk and reloads via `--calibration` flag
- [ ] Live and file sources both supported (via existing `FrameSource`)
