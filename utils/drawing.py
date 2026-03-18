# utils/drawing.py — OpenCV Annotation & Overlay Utilities
# Everything that draws ON the camera frame lives here.
# No business logic — pure rendering. Called by the detection thread
# after inference so the annotated frame can be shown in Pygame + Streamlit.

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import (
    ARM_NAMES,
    ROI_COLORS,
    ROI_ALPHA,
    ROIS,
    HAZARD_CLASSES,
    EMERGENCY_CLASSES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette  (BGR — OpenCV convention)
# ---------------------------------------------------------------------------

CLR_WHITE    = (255, 255, 255)
CLR_BLACK    = (0,   0,   0)
CLR_GREEN    = (0,   220,  80)
CLR_YELLOW   = (0,   200, 255)
CLR_RED      = (30,   30, 220)
CLR_ORANGE   = (0,   140, 255)
CLR_CYAN     = (255, 255,   0)
CLR_PURPLE   = (220,   0, 180)
CLR_GRAY     = (160, 160, 160)

# Bbox colours by detection type
BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    'default':   (60,  220,  60),   # normal vehicle — green
    'emergency': (0,    0,  255),   # ambulance/fire truck — red
    'hazard':    (0,  140,  255),   # animal — orange
    'person':    (255, 220,   0),   # pedestrian — cyan
}

# Signal phase → colour (BGR)
SIGNAL_COLORS: dict[str, tuple[int, int, int]] = {
    'GREEN':  CLR_GREEN,
    'YELLOW': CLR_YELLOW,
    'RED':    CLR_RED,
    'WALK':   CLR_CYAN,
    'ALL_RED': CLR_RED,
    'UNKNOWN': CLR_GRAY,
}


# ---------------------------------------------------------------------------
# ROI rendering
# ---------------------------------------------------------------------------

def draw_rois(
    frame: np.ndarray,
    rois: dict[str, np.ndarray] = ROIS,
    alpha: float = ROI_ALPHA,
    colors: dict[str, tuple[int, int, int]] = ROI_COLORS,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw semi-transparent filled ROI polygons + outlines onto frame.

    Args:
        frame:       BGR frame to annotate (not modified in place — copy made).
        rois:        Dict of arm_name → numpy polygon array.
        alpha:       Transparency of filled polygon [0=invisible, 1=solid].
        colors:      Per-arm BGR colour dict.
        show_labels: If True, draw arm name at polygon centroid.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    overlay = frame.copy()

    for arm, polygon in rois.items():
        color = colors.get(arm, CLR_GRAY)

        # Filled semi-transparent polygon on overlay
        cv2.fillPoly(overlay, [polygon], color)

        # Hard outline on out (not blended — always fully visible)
        cv2.polylines(out, [polygon], isClosed=True, color=color, thickness=2)

        if show_labels:
            cx, cy = _polygon_centroid(polygon)
            _draw_label(out, arm, cx, cy, color, font_scale=0.55, bg=True)

    # Blend overlay (filled areas) with original frame
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    return out


def draw_roi_single(
    frame: np.ndarray,
    arm: str,
    polygon: Optional[np.ndarray] = None,
    alpha: float = ROI_ALPHA,
) -> np.ndarray:
    """Draw a single ROI — useful during ROI calibration."""
    polygon = polygon if polygon is not None else ROIS.get(arm)
    if polygon is None:
        return frame
    return draw_rois(frame, rois={arm: polygon}, alpha=alpha)


# ---------------------------------------------------------------------------
# Bounding box rendering
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: list,          # list[Detection] — avoid circular import
    show_confidence: bool = True,
    show_pcu: bool = True,
) -> np.ndarray:
    """
    Draw styled bounding boxes for all detections on the frame.

    Box colour:
      - Emergency vehicles → red
      - Hazard animals     → orange
      - Pedestrians        → cyan
      - Everything else    → green

    Args:
        frame:            BGR frame.
        detections:       List of Detection objects from detector.py.
        show_confidence:  Append confidence score to label.
        show_pcu:         Append PCU weight to label.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()

    for det in detections:
        color = _bbox_color(det)
        x1, y1, x2, y2 = map(int, det.xyxy)

        # Main bounding box (2px) + subtle inner highlight (1px, lighter)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lighter = _lighten(color, 80)
        cv2.rectangle(out, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), lighter, 1)

        # Centroid dot
        cv2.circle(out, (int(det.cx), int(det.cy)), 3, color, -1)

        # Label text
        parts = [det.cls_name]
        if show_confidence:
            parts.append(f"{det.conf:.2f}")
        pcu = getattr(det, "pcu", 0.0)
        if show_pcu and pcu > 0:
            parts.append(f"PCU={pcu}")
        label = " ".join(parts)

        _draw_label(out, label, x1, y1 - 6, color, font_scale=0.42, bg=True)

    return out


def draw_detection_count(
    frame: np.ndarray,
    detections: list,
    position: tuple[int, int] = (10, 55),
) -> np.ndarray:
    """
    Draw a compact detection summary in the top-left corner.
    E.g.: "14 det | car×4 moto×6 bus×1 person×3"
    """
    out = frame.copy()
    if not detections:
        return out

    from collections import Counter
    counts = Counter(d.cls_name for d in detections)
    summary = "  ".join(f"{cls}×{n}" for cls, n in counts.most_common(5))
    text = f"{len(detections)} det | {summary}"

    _draw_label(out, text, position[0], position[1], CLR_WHITE,
                font_scale=0.48, bg=True, bg_color=CLR_BLACK)
    return out


# ---------------------------------------------------------------------------
# Signal / arm status HUD
# ---------------------------------------------------------------------------

def draw_signal_hud(
    frame: np.ndarray,
    arm_states: dict,           # dict[str, ArmState] — avoid circular import
    current_green: Optional[str] = None,
    phase: str = 'normal',
    position: tuple[int, int] = (10, 90),
    compact: bool = False,
) -> np.ndarray:
    """
    Draw a heads-up display showing signal state for all arms.

    Each row: [●] ARM  density=X  wait=Ys  score=Z
    The active green arm row is highlighted.

    Args:
        frame:         BGR frame.
        arm_states:    Dict of arm_name → ArmState from controller/state.py.
        current_green: Name of arm currently holding green.
        phase:         Current intersection phase string.
        position:      Top-left pixel of the HUD box.
        compact:       If True, render a minimal one-liner per arm.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    x0, y0 = position
    line_h = 22 if not compact else 18
    pad = 6

    # Phase banner at top
    phase_color = CLR_GREEN if phase == 'normal' else CLR_YELLOW
    if phase == 'emergency':
        phase_color = CLR_RED
    elif phase == 'pedestrian':
        phase_color = CLR_CYAN

    phase_label = f"PHASE: {phase.upper()}"
    _draw_label(out, phase_label, x0, y0, phase_color,
                font_scale=0.55, bg=True, bg_color=(20, 20, 20))

    for i, arm in enumerate(ARM_NAMES):
        state = arm_states.get(arm)
        if state is None:
            continue

        y = y0 + (i + 1) * line_h + pad

        is_green = (arm == current_green)
        dot_color = CLR_GREEN if is_green else CLR_RED
        row_color = CLR_GREEN if is_green else CLR_WHITE

        # Coloured status dot
        cv2.circle(out, (x0 + 6, y - 4), 5, dot_color, -1)

        d_val = state.get('density', 0.0) if isinstance(state, dict) else state.density
        w_val = state.get('wait_time', 0.0) if isinstance(state, dict) else state.wait_time
        f_val = state.get('flow_rate', 0.0) if isinstance(state, dict) else state.flow_rate
        e_val = state.get('emergency', False) if isinstance(state, dict) else getattr(state, 'emergency', False)
        h_val = state.get('hazard', False) if isinstance(state, dict) else getattr(state, 'hazard', False)

        if compact:
            text = f"{arm[:1]}  d={d_val:.1f}  w={int(w_val)}s"
        else:
            text = (
                f"{arm:<6} "
                f"density={d_val:5.1f}  "
                f"wait={int(w_val):3d}s  "
                f"flow={f_val:4.1f}"
            )
            if e_val:
                text += "  🚨 EMRG"
            if h_val:
                text += "  ⚠ HZRD"

        _draw_label(out, text, x0 + 16, y, row_color,
                    font_scale=0.44, bg=True,
                    bg_color=(0, 60, 0) if is_green else (20, 20, 20))

    return out


def draw_density_bars(
    frame: np.ndarray,
    arm_states: dict,
    max_density: float = 50.0,
    position: tuple[int, int] = (10, 200),
    bar_width: int = 120,
    bar_height: int = 12,
) -> np.ndarray:
    """
    Draw horizontal density bars for each arm — quick visual at a glance.

    Args:
        frame:       BGR frame.
        arm_states:  Dict of arm_name → ArmState.
        max_density: Value at which bar is 100% full.
        position:    Top-left pixel of bar group.
        bar_width:   Pixel width of full bar.
        bar_height:  Pixel height of each bar.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    x0, y0 = position
    spacing = bar_height + 8

    for i, arm in enumerate(ARM_NAMES):
        state = arm_states.get(arm)
        if state is None:
            continue

        d_val = state.get('density', 0.0) if isinstance(state, dict) else state.density

        y = y0 + i * spacing
        fill = int(min(1.0, d_val / max_density) * bar_width)

        # Background track
        cv2.rectangle(out, (x0, y), (x0 + bar_width, y + bar_height),
                      (50, 50, 50), -1)

        # Filled portion — green → yellow → red based on density
        ratio = d_val / max_density
        bar_color = _density_color(ratio)
        if fill > 0:
            cv2.rectangle(out, (x0, y), (x0 + fill, y + bar_height),
                          bar_color, -1)

        # Border
        cv2.rectangle(out, (x0, y), (x0 + bar_width, y + bar_height),
                      CLR_GRAY, 1)

        # Label
        label = f"{arm[:1]} {d_val:.1f}"
        cv2.putText(out, label, (x0 + bar_width + 6, y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_WHITE, 1)

    return out


# ---------------------------------------------------------------------------
# Alert banners
# ---------------------------------------------------------------------------

def draw_emergency_banner(frame: np.ndarray, arm: str) -> np.ndarray:
    """Full-width red banner: EMERGENCY — {ARM} ARM CLEARED"""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = 36
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 180), -1)
    text = f"  EMERGENCY OVERRIDE — {arm.upper()} ARM PRIORITY"
    cv2.putText(out, text, (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_WHITE, 1)
    return out


def draw_hazard_banner(frame: np.ndarray, arm: str, cls: str) -> np.ndarray:
    """Full-width orange banner for animal on road."""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = 36
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 100, 220), -1)
    text = f"  ⚠ HAZARD — {cls.upper()} ON {arm.upper()} ARM  (signal extended +5s)"
    cv2.putText(out, text, (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, CLR_WHITE, 1)
    return out


def draw_pedestrian_banner(frame: np.ndarray, ped_count: float) -> np.ndarray:
    """Full-width cyan banner when pedestrian phase is active."""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = 36
    cv2.rectangle(out, (0, 0), (w, banner_h), (160, 160, 0), -1)
    text = f"  PEDESTRIAN PHASE ACTIVE  ({ped_count:.0f} persons detected)"
    cv2.putText(out, text, (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, CLR_BLACK, 1)
    return out


# ---------------------------------------------------------------------------
# Frame metadata overlay
# ---------------------------------------------------------------------------

def draw_frame_info(
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    inference_ms: float = 0.0,
) -> np.ndarray:
    """
    Draw frame number, FPS, and inference time in the top-right corner.

    Args:
        frame:        BGR frame.
        frame_number: Current frame index.
        fps:          Measured frames per second.
        inference_ms: Last YOLO inference latency in milliseconds.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [
        f"Frame: {frame_number}",
        f"FPS:   {fps:.1f}",
    ]
    if inference_ms > 0:
        lines.append(f"Infer: {inference_ms:.0f}ms")

    for i, line in enumerate(lines):
        y = 18 + i * 18
        x = w - 130
        cv2.putText(out, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BLACK, 3)
        cv2.putText(out, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1)

    return out


def draw_timestamp(frame: np.ndarray, timestamp: str) -> np.ndarray:
    """Stamp a timestamp string in the bottom-right corner."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.putText(out, timestamp, (w - 200, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BLACK, 3)
    cv2.putText(out, timestamp, (w - 200, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GRAY, 1)
    return out


# ---------------------------------------------------------------------------
# Debug overlay (toggled by D key in Pygame)
# ---------------------------------------------------------------------------

def draw_debug_overlay(
    frame: np.ndarray,
    detections: list,
    arm_states: dict,
    show_centroids: bool = True,
    show_roi_labels: bool = True,
) -> np.ndarray:
    """
    Draw full debug overlay: ROIs + bboxes + HUD + density bars.
    Composites all drawing functions in the correct Z-order.

    Args:
        frame:           BGR frame (already preprocessed).
        detections:      List of Detection objects.
        arm_states:      Dict of arm_name → ArmState.
        show_centroids:  Draw centroid dots for each detection.
        show_roi_labels: Show arm name at each ROI centre.

    Returns:
        Fully annotated BGR frame.
    """
    out = draw_rois(frame, show_labels=show_roi_labels)
    out = draw_detections(out, detections)
    out = draw_detection_count(out, detections)

    if arm_states:
        out = draw_signal_hud(out, arm_states, position=(10, 90))
        out = draw_density_bars(out, arm_states, position=(10, 310))

    if show_centroids:
        for det in detections:
            cv2.drawMarker(
                out, (int(det.cx), int(det.cy)),
                _bbox_color(det),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
            )

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_color(det) -> tuple[int, int, int]:
    """Return the correct BGR colour for a Detection object."""
    if det.cls_name in EMERGENCY_CLASSES:
        return BBOX_COLORS['emergency']
    if det.cls_name in HAZARD_CLASSES:
        return BBOX_COLORS['hazard']
    if det.cls_name == 'person':
        return BBOX_COLORS['person']
    return BBOX_COLORS['default']


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    bg: bool = False,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draw text with optional filled background rectangle.
    Modifies frame in place.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if bg:
        pad = 3
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            bg_color,
            -1,
        )

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                lineType=cv2.LINE_AA)


def _polygon_centroid(polygon: np.ndarray) -> tuple[int, int]:
    """Return the integer centroid of a polygon."""
    cx = int(np.mean(polygon[:, 0]))
    cy = int(np.mean(polygon[:, 1]))
    return cx, cy


def _lighten(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    """Add amount to each BGR channel, clamped to 255."""
    return tuple(min(255, c + amount) for c in color)  # type: ignore[return-value]


def _density_color(ratio: float) -> tuple[int, int, int]:
    """
    Map a [0, 1] density ratio to a BGR colour gradient:
      0.0 → green   (low density — all clear)
      0.5 → yellow  (medium density)
      1.0 → red     (high density — congested)
    """
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.5:
        t = ratio * 2.0
        # green (0,220,80) → yellow (0,200,255)
        b = 0
        g = int(220 - t * 20)
        r = int(t * 255)
    else:
        t = (ratio - 0.5) * 2.0
        # yellow (0,200,255) → red (30,30,220)
        b = int(t * 30)
        g = int(200 - t * 170)
        r = int(255 - t * 35)
    return (b, g, r)


# ---------------------------------------------------------------------------
# Standalone test  (python -m utils.drawing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import time
    from config import VIDEO_SOURCE
    from utils.preprocessing import preprocess

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print("Press Q to quit | Shows ROI overlays + drawing utilities\n")

    frame_n = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed = preprocess(frame)

        # Draw ROIs
        out = draw_rois(processed)

        # Frame counter + FPS
        frame_n += 1
        if frame_n % 15 == 0:
            fps = 15.0 / (time.time() - t0)
            t0 = time.time()
        out = draw_frame_info(out, frame_n, fps)

        cv2.imshow('Drawing Test — ROI Overlays', out)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")