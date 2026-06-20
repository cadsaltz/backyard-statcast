"""Pixel geometry derived from field calibration for pitch detection."""

from __future__ import annotations

from dataclasses import dataclass

from calibration import FieldCalibration
from spatial_filter import scale_point_to_pixels, scaled_polygons


@dataclass(frozen=True)
class FieldGeometry:
    proc_w: int
    proc_h: int
    full_w: int
    full_h: int
    release_cx_proc: int
    release_cy_proc: int
    release_r_proc: float
    release_cx_full: int
    release_cy_full: int
    release_r_full: float
    strike_leading_x_proc: int
    strike_trailing_x_proc: int
    strike_right_x_proc: int
    strike_bottom_y_proc: int
    strike_right_pad_proc: int
    strike_bottom_pad_proc: int
    strike_leading_x_full: int
    strike_trailing_x_full: int
    pass_margin_proc: int
    approach_margin_proc: int

    @property
    def strike_right_line_proc(self) -> int:
        """Right boundary trigger line (zone edge + padding)."""
        return self.strike_right_x_proc + self.strike_right_pad_proc

    @property
    def strike_bottom_line_proc(self) -> int:
        """Bottom boundary trigger line (zone edge + padding)."""
        return self.strike_bottom_y_proc + self.strike_bottom_pad_proc

    def crossed_strike_right_edge(
        self, prev_x: int, prev_y: int, x: int, y: int
    ) -> bool:
        """True when path crosses the padded right edge of the strike zone."""
        line = self.strike_right_line_proc
        return prev_x < line and x >= line

    def crossed_strike_bottom_edge(
        self, prev_x: int, prev_y: int, x: int, y: int
    ) -> bool:
        """True when path crosses the padded bottom edge of the strike zone."""
        line = self.strike_bottom_line_proc
        return prev_y < line and y >= line

    def in_release_zone(self, x_proc: int, y_proc: int, *, expand: float = 1.0) -> bool:
        r = self.release_r_proc * expand
        dx = x_proc - self.release_cx_proc
        dy = y_proc - self.release_cy_proc
        return dx * dx + dy * dy <= r * r

    def proc_to_full(self, x_proc: int, y_proc: int) -> tuple[int, int]:
        sx = self.full_w / self.proc_w
        sy = self.full_h / self.proc_h
        return int(x_proc * sx), int(y_proc * sy)


def _strike_x_bounds(strike_pts: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [p[0] for p in strike_pts]
    return min(xs), max(xs)


def _strike_bottom_y(strike_pts: list[tuple[int, int]]) -> int:
    return max(p[1] for p in strike_pts)


def build_field_geometry(
    cal: FieldCalibration,
    *,
    frame_width: int,
    frame_height: int,
    process_scale: float,
    pass_margin_frac: float = 0.03,
    approach_margin_frac: float = 0.04,
    strike_boundary_pad_frac: float = 0.012,
) -> FieldGeometry:
    proc_w = max(1, int(frame_width * process_scale))
    proc_h = max(1, int(frame_height * process_scale))
    scale = min(frame_width, frame_height)

    rcx, rcy = scale_point_to_pixels(cal.release_center, frame_width=proc_w, frame_height=proc_h)
    r_proc = cal.release_radius * min(proc_w, proc_h)
    rcx_f, rcy_f = scale_point_to_pixels(cal.release_center, frame_width=frame_width, frame_height=frame_height)
    r_full = cal.release_radius * scale

    strike_proc = scaled_polygons(cal.strike_zone, frame_width=proc_w, frame_height=proc_h)
    strike_full = scaled_polygons(cal.strike_zone, frame_width=frame_width, frame_height=frame_height)
    leading_p, trailing_p = _strike_x_bounds(strike_proc)
    leading_f, trailing_f = _strike_x_bounds(strike_full)
    bottom_y_proc = _strike_bottom_y(strike_proc)
    right_pad = max(4, int(proc_w * strike_boundary_pad_frac))
    bottom_pad = max(4, int(proc_h * strike_boundary_pad_frac))

    return FieldGeometry(
        proc_w=proc_w,
        proc_h=proc_h,
        full_w=frame_width,
        full_h=frame_height,
        release_cx_proc=rcx,
        release_cy_proc=rcy,
        release_r_proc=r_proc,
        release_cx_full=rcx_f,
        release_cy_full=rcy_f,
        release_r_full=r_full,
        strike_leading_x_proc=leading_p,
        strike_trailing_x_proc=trailing_p,
        strike_right_x_proc=trailing_p,
        strike_bottom_y_proc=bottom_y_proc,
        strike_right_pad_proc=right_pad,
        strike_bottom_pad_proc=bottom_pad,
        strike_leading_x_full=leading_f,
        strike_trailing_x_full=trailing_f,
        pass_margin_proc=int(frame_width * process_scale * pass_margin_frac),
        approach_margin_proc=int(frame_width * process_scale * approach_margin_frac),
    )
