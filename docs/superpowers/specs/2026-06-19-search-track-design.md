# Search / Track Mode — Design Spec

**Date:** 2026-06-19  
**Status:** Implemented (2026-06-19)  
**Scope:** Frame-to-frame tracking state for `track_ball.py` density pipeline

## Problem

The live tracker runs **global** detection every frame: full-frame background subtraction, color filtering, then a density peak over the entire `color_mask`. There is no memory of the previous ball position.

When the ball is far from the camera it produces only a few foreground pixels (often 3–8px at process resolution). The fixed 25×25 density kernel (`density_radius=12`) accumulates neighbors globally, so fence glare, tree motion, or other noise frequently outscores the distant ball. The tracker drops the ball mid-pitch even though it is still visible in frame.

Spatial calibration (ignore zones, ROI gating) filters false positives **after** the peak is chosen. It does not change where detection runs or reduce global competition for the density maximum.

## Goal

Introduce a two-state tracking model that prioritizes **continuity over per-frame perfection**:

- Maintain the ball position frame-to-frame while it remains in frame during a pitch
- Reacquire quickly when tracking is briefly lost
- Leverage smooth, bounded ball motion (no teleporting)
- Keep the existing detection stack (bg subtract → color mask → density peak) — add state and local search, do not replace it

## Non-Goals

- Replacing background subtraction or switching to contour / ML strategies
- Kalman filters, homography, or 3D trajectory reconstruction
- Cropping preprocessing (absdiff, HSV) before the color mask is built **globally**
- Loosening color/motion thresholds on the **full frame** in Track Mode (relax is window-local only)
- Using the calibration ROI polygon as the Track Mode search window (ROI remains pitch gating only)
- Refactoring `detection_strategies.py` / `ball_tracker.py` (those paths keep `max_jump_px` for comparison tooling)
- FPS optimization as a primary objective (local crop may help marginally; robustness is the goal)

### Distinction from spatial-calibration non-goals

The spatial-calibration spec lists "ROI cropping of detection input" as a non-goal. Search/Track **local search** is not that: it crops a small window around a **predicted ball position** for density peak only, not the user-drawn ROI polygon. ROI continues to gate `pitch_active`; Track Mode search window is driven by motion history.

## Recommended Architecture

### Two detection modes

| Mode | When active | Search region | Density params |
|------|-------------|---------------|----------------|
| **Search** | No valid track, or Track lost for ≥ `track_lost_frames` | Full `color_mask` | Strict color/motion thresholds; `density_radius=12`; global max |
| **Track** | Valid detection within last `track_lost_frames` | Crop around predicted position | Relaxed color/motion thresholds **inside window only**; smaller `density_radius`; lower density floor |

### Pipeline (unchanged through color mask)

```
frame
  → downscale (process_scale)
  → absdiff vs learned background
  → threshold + morph open → fg_mask
  → HSV color filter
  → color_mask = fg_mask & color_filter
```

### Pipeline (new after color mask)

```
color_mask (strict, full frame)     ← used for Search Mode
  → [TrackState] choose Search or Track
  → if Track:
       crop small + fg_mask to search window
       rebuild color_mask with relaxed thresholds on crop only
       track_mask = relaxed_color & cropped_fg
  → density_peak(search_mask or full strict mask) with mode-appropriate radius
  → if Track: map peak to full process coords; gate by max displacement
  → if Track fail: increment miss counter; coast or fall back to Search
  → ignore mask filter (existing)
  → return TrackResult + tracking metadata
```

Full-frame strict preprocessing always runs (Search path and debug panels). Track Mode adds a **second, relaxed mask pass on the window crop only** before local density peak.

### State machine

```
                    ┌─────────────┐
         reset/b    │   SEARCH    │◄────────────────────┐
         startup    │  full frame │                       │
                    └──────┬──────┘                       │
                           │ valid detection              │
                           │ (after ignore filter)        │
                           ▼                              │
                    ┌─────────────┐                       │
                    │    TRACK    │                       │
                    │ local window│─── miss ≥ N ──────────┘
                    └─────────────┘     (no valid peak
                           │           in window)
                           │ valid peak each frame
                           └──────────► stay in TRACK
```

**Enter Track:** Search produces a valid peak (post-ignore) → store position, switch to Track.

**Stay in Track:** Local peak found within displacement gate → update history, reset miss counter.

**Coast (optional sub-state within Track):** Peak not found for 1–2 frames → emit last known position, widen search window, do not switch to Search yet.

**Exit to Search:** Miss counter reaches `track_lost_frames` (recommended 3) → clear velocity history, run full-frame Search next frame.

**Reset to Search:** User presses `b` (background relearn) → same as today plus clear TrackState.

`pitch_active` lifecycle is **unchanged**: set on ROI entry, cleared after 30 idle frames or `b`. Track Mode and pitch_active are orthogonal.

## Motion Prediction

Use the last two accepted positions to predict the next search center.

```python
velocity = last_pos - prev_pos          # process-resolution pixels
predicted = last_pos + velocity         # constant-velocity extrapolation
```

Search window is a square (or axis-aligned rect) centered on `predicted`:

```
half_extent = max(min_search_half, |velocity| + margin)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `margin` | 24 px (process coords) | Base padding when ball is slow |
| `min_search_half` | 32 px | Minimum half-width of search window |
| `max_search_half` | 96 px | Cap for fast pitches / coast widening |

On first frame after entering Track (only one history point), center the window on `last_pos` with `min_search_half`.

During coast, multiply `half_extent` by 1.5× per missed frame (up to `max_search_half`) before falling back to Search.

## Track Mode Detection

Track Mode improves continuity through **six** mechanisms. Local window reduction is only one of them.

| Mechanism | What it fixes |
|-----------|----------------|
| **Local search window** | Distant ball no longer competes with global noise for the density max |
| **Smaller density radius** | Large kernel dilutes tiny balls; tight local kernel resolves 3–5px blobs |
| **Relaxed local color/motion** | Distant/dim ball pixels that barely fail strict Search thresholds still contribute inside the window |
| **Displacement gate** | Wrong peak inside the window (edge flutter, shadow) rejected if too far from prediction |
| **Lower density floor** | Accept a weaker peak for a known object (`track_min_density_ratio`) |
| **Coasting + velocity prediction** | Window follows expected motion; hold last position through 1–2 blank frames |

### Strict Search, relaxed Track (color / motion)

**Recommendation: yes — strict in Search, relaxed in Track, but only inside the track window.**

Distant-ball dropouts are often **signal starvation**, not just wrong global max. A ball at 4px may produce 2–3 pixels above `white_threshold` and `diff_threshold`. Strict Search correctly avoids locking onto random bright patches frame-wide; once Track has a lock, the prior is “the ball is here” and the question becomes “recover any ball-like pixels nearby.”

Apply relaxation **only when rebuilding the mask on the window crop** — never on the full frame:

| Channel | Search (strict) | Track relax (added to strict base) |
|---------|-----------------|-------------------------------------|
| White value | `white_threshold` (default 140) | −20 → effective 120 in window |
| White saturation cap | `max_saturation` (default 100) | +25 → effective 125 in window |
| Motion diff | `diff_threshold` (default 25) | −5 → effective 20 on cropped fg |
| Yellow value | `yellow_min_val` (default 140) | −20 in window |
| Yellow saturation | `yellow_min_sat` (default 80) | −15 in window |

Implementation: `_color_mask(hsv_crop, *, relax: bool)` and re-threshold cropped `fg_mask` with relaxed diff when `relax=True`.

**False-positive risk:** Low for global acquisition (Search unchanged). Inside the small window, extra pixels usually come from motion near the ball path — acceptable because displacement gate + prediction constrain which peak wins. Main failure mode: window expands during coast over a busy background patch; mitigated by `max_search_half` cap and short coast before Search fallback. Do **not** relax thresholds in Search Mode.

Optional fallback if relaxed mask adds too much local clutter: morphological **open** on track crop only (keep strict Search morph unchanged).

### Smaller density radius

In Track Mode use `track_density_radius` (default **6**, kernel 13×13) instead of the global `density_radius` (default 12). A large kernel dilutes peaks from tiny balls; locally a tight kernel resolves small blobs better.

Search Mode keeps the user-tunable `density_radius` (keyboard `,` / `.`).

### Displacement gate

Reject a local peak if its distance from `predicted` exceeds `max_jump_px` (default **80** at process scale, ~160 full-frame at 0.5 scale). Mirrors the proven pattern in `detection_strategies._pick_best`.

If the peak fails the gate, treat as a miss (increment counter, coast or fall back).

### Minimum density score (Track only)

Record the density value at the global peak during Search lock-on. In Track Mode, accept a local peak if its density ≥ `track_min_density_ratio × last_search_density` (default ratio **0.4**). Prevents locking onto weak noise while allowing weaker signal for a known object.

If no prior search density exists (entered Track from first detection), skip this check for the first Track frame.

## Parameters Summary

| Parameter | Default | Mode | Purpose |
|-----------|---------|------|---------|
| `density_radius` | 12 | Search | Existing global kernel (runtime tunable) |
| `track_density_radius` | 6 | Track | Tighter local kernel |
| `track_lost_frames` | 3 | — | Misses before Search fallback |
| `coast_frames` | 2 | Track | Frames to hold last position before Search |
| `max_jump_px` | 80 | Track | Max displacement from predicted (process px) |
| `margin` | 24 | Track | Search window padding |
| `min_search_half` | 32 | Track | Minimum half-extent |
| `max_search_half` | 96 | Track | Maximum half-extent |
| `track_min_density_ratio` | 0.4 | Track | Min peak strength vs Search lock-on |
| `track_white_relax` | 20 | Track | Lower white V threshold in window |
| `track_sat_relax` | 25 | Track | Raise white S cap in window |
| `track_diff_relax` | 5 | Track | Lower motion threshold on cropped fg |
| `track_yellow_val_relax` | 20 | Track | Lower yellow min V in window |
| `track_yellow_sat_relax` | 15 | Track | Lower yellow min S in window |

All distances are in **process-resolution** pixels unless noted. Relax offsets are subtracted/added to the current strict values (respect runtime keyboard tuning). CLI flags are optional follow-up; defaults live in `TrackState` / `BallTracker` initially.

## Module Layout

```
track_state.py       TrackMode enum, position history, predict(), search window, state transitions
track_ball.py        BallTracker.process() calls TrackState after color_mask; main loop unchanged except reset
tests/test_track_state.py   Unit tests for prediction, window bounds, state transitions (no OpenCV video needed)
```

Keep `TrackState` separate from `BallTracker` so detection math stays in `BallTracker` and temporal logic stays testable without frames.

### BallTracker changes

`process()` gains optional integration:

1. Build `color_mask` (existing)
2. Delegate peak finding to `TrackState.find_peak(color_mask, density_peak_fn)` where `density_peak_fn` is the existing `_density_peak(mask, radius)` 
3. Apply ignore mask to result (existing)
4. Return extended `TrackResult` with `mode: Search | Track` for debug overlay (optional)

### TrackResult extension

```python
@dataclass
class TrackResult:
    density_peak: tuple[int, int] | None
    density_peak_proc: tuple[int, int] | None = None
    foreground: np.ndarray | None = None
    color_fg: np.ndarray | None = None
    rejected_by_ignore: bool = False
    raw_peak_proc: tuple[int, int] | None = None
    track_mode: Literal["search", "track"] = "search"   # new, debug/status
    coasting: bool = False                               # new, optional overlay
    search_window_proc: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 when Track
    predicted_proc: tuple[int, int] | None = None
```

Status line addition: `mode=SEARCH` / `mode=TRACK` (and `COAST` when coasting).

### Debug windows

When debug panels are enabled (`d`, default on), add a **fourth window** dedicated to Track Mode geometry:

| Window | Name constant | Content |
|--------|---------------|---------|
| Live | `WIN_LIVE` | Capture resolution + calibration overlay + dot (unchanged) |
| Foreground | `WIN_FG` | Motion mask (unchanged) |
| Color Filter | `WIN_COLOR` | Strict full-frame color mask + dot (unchanged) |
| **Track Window** | `WIN_TRACK` | **New** — process-resolution view of tracking state |

**`WIN_TRACK` contents (process resolution, scaled to same debug panel size as FG/Color):**

- Background: strict `color_mask` (dim) or cropped `small` grayscale
- **Cyan rectangle:** current search window (`search_window_proc`)
- **Yellow dot:** last accepted position
- **Magenta dot:** predicted center (`predicted_proc`)
- **White arrow:** velocity vector (last − prev)
- **Red dot:** current density peak (if any)
- Label: `SEARCH` / `TRACK` / `COAST` + window half-extent in px

In Search Mode, show full process frame with label `SEARCH` and no rectangle (or full-frame border). Window opens/closes with existing `d` toggle alongside FG and Color panels.

Layout: place `WIN_TRACK` adjacent to Color panel (same row or below FG) so all diagnostic views are visible together.

## Interaction with Spatial Calibration

| Feature | Behavior with Search/Track |
|---------|---------------------------|
| Ignore mask | Applied after peak selection (unchanged). Rejected peaks do not enter Track history. |
| ROI | Still sets `pitch_active` on entry (unchanged). Does not constrain Track search window. |
| Choice B | Yellow/red dot still shown for any valid post-ignore detection (unchanged). |
| `b` reset | Clears background **and** `TrackState` → Search Mode. |

Future: when Search finds multiple comparable peaks, prefer the one inside ROI. Out of scope for v1 (current pipeline returns a single global max).

## Alternatives Considered

| Approach | Verdict |
|----------|---------|
| **Search/Track with local density crop** | **Recommended** — minimal change, targets root cause |
| Port `max_jump_px` only (no local crop) | Insufficient — still global max; distant ball loses to noise |
| Smaller global `density_radius` always | Hurts Search acquisition; noise peaks get noisier globally |
| Kalman + full tracker rewrite | Over-engineered for current goal |
| ROI-restricted detection | Wrong window — ball path crosses ROI boundary; ROI is event gating not motion prior |

## Testing Strategy

**Unit tests (`test_track_state.py`):**

- Velocity prediction from two points
- Search window clamped to frame bounds
- Search → Track on valid detection
- Track → Search after N misses
- Displacement gate rejects far peaks
- Coast emits last position for ≤ `coast_frames`
- Reset clears history

**Manual validation:**

- Live feed: track ball from release through plate at full distance without dropout inside ROI
- Recorded clip replay (`--source file.mp4`) for repeatable before/after comparison
- Debug overlay: `WIN_TRACK` shows search window, prediction, velocity, and peak every frame when `d` enabled

## Success Criteria

- [ ] Ball tracked continuously through a full pitch when ball remains visible in frame
- [ ] Noticeable improvement tracking distant (small) ball vs current global-only peak
- [ ] Search reacquires ball within 3 frames after brief occlusion or Track loss
- [ ] No regression on near-field tracking (release zone / close shots)
- [ ] `pitch_active`, ignore mask, and calibration overlays behave identically to today
- [ ] `b` reset returns to Search Mode with cleared history
- [ ] Unit tests cover state machine and prediction without video fixtures
- [ ] `WIN_TRACK` debug window shows search rectangle, prediction, and peak in Track Mode

## Open Questions (resolve during implementation)

1. **Density value exposure:** `_density_peak` currently returns only location. Extend to return `(x, y, score)` for min-density ratio check.
2. **File vs live defaults:** Same parameters at `process_scale=1.0` (720p) vs `0.5` (1080p)? Start with shared process-pixel defaults; tune if 720p clips fast pitches.
3. **Relax magnitude:** If local false peaks appear in `WIN_TRACK`, reduce relax offsets before touching Search strictness.
