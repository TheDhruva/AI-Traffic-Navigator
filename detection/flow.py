# detection/flow.py — Optical Flow Analyser (Lucas-Kanade Sparse)
# Measures traffic motion magnitude per ROI arm.
# Low flow = vehicles are stopped (congestion signal → raises priority score).
# High flow = arm is clearing (reduces urgency).
#
# Why sparse LK over dense Farneback?
#   Dense: every pixel, ~15ms/frame on CPU → too slow at 20 FPS
#   Sparse LK: tracks 50–200 feature points per ROI, ~2ms/frame → viable
#   Accuracy is sufficient: we only need a scalar "is traffic moving?" per arm.

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    ARM_NAMES,
    ROIS,
    FLOW_MIN_MAGNITUDE,
    FLOW_MAX_FEATURES,
    FLOW_QUALITY_LEVEL,
    FLOW_MIN_DISTANCE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lucas-Kanade parameters (tuned for traffic scenes)
# ---------------------------------------------------------------------------

# Shi-Tomasi corner detector params (feature seeding)
_SHITOMASI_PARAMS = dict(
    maxCorners=FLOW_MAX_FEATURES,
    qualityLevel=FLOW_QUALITY_LEVEL,   # 0.3 — accepts weaker corners in low contrast
    minDistance=FLOW_MIN_DISTANCE,     # px — prevents feature clustering on one vehicle
    blockSize=7,
)

# Lucas-Kanade optical flow params
_LK_PARAMS = dict(
    winSize=(15, 15),       # search window per feature — 15px handles ~30km/h at 20FPS
    maxLevel=2,             # pyramid levels — handles faster motion without large winSize
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,     # max iterations
        0.03,   # epsilon — convergence threshold
    ),
)


# ---------------------------------------------------------------------------
# Per-arm flow result
# ---------------------------------------------------------------------------

@dataclass
class FlowResult:
    """Optical flow output for all arms after one frame pair."""

    # Mean motion magnitude per arm (pixels/frame).
    # 0.0 means no features tracked or no motion.
    magnitudes: dict[str, float] = field(default_factory=dict)

    # Smoothed (rolling average) magnitudes — used by the controller
    smoothed: dict[str, float] = field(default_factory=dict)

    # Per-arm boolean: True when smoothed magnitude < FLOW_MIN_MAGNITUDE
    is_stopped: dict[str, bool] = field(default_factory=dict)

    # Number of tracked feature points per arm (useful for debug)
    tracked_points: dict[str, int] = field(default_factory=dict)

    # Raw flow vectors for debug drawing: arm → list of (prev_pt, curr_pt)
    vectors: dict[str, list[tuple]] = field(default_factory=dict)

    def __repr__(self) -> str:
        mag = {k: f"{v:.2f}" for k, v in self.smoothed.items()}
        stopped = [k for k, v in self.is_stopped.items() if v]
        return f"FlowResult(smoothed={mag} stopped={stopped})"


# ---------------------------------------------------------------------------
# Flow Analyser — instantiate once, call update() each frame
# ---------------------------------------------------------------------------

class FlowAnalyser:
    """
    Sparse Lucas-Kanade optical flow tracker per intersection arm.

    Lifecycle:
        analyser = FlowAnalyser()
        # first frame:
        result = analyser.update(gray_frame)   # returns zeros (no prev frame yet)
        # subsequent frames:
        result = analyser.update(gray_frame)   # returns live flow data

    The analyser re-seeds features every RESEED_INTERVAL frames or when
    tracked point count for an arm drops below MIN_POINTS_THRESHOLD.
    This prevents drift as vehicles enter/leave the scene.
    """

    RESEED_INTERVAL      = 10   # frames between forced feature re-seeding
    MIN_POINTS_THRESHOLD = 5    # if fewer points tracked, re-seed immediately
    ROLLING_WINDOW       = 5    # frames for smoothing magnitude per arm

    def __init__(
        self,
        rois: dict[str, np.ndarray] = ROIS,
        arm_names: list[str] = ARM_NAMES,
        flow_min_magnitude: float = FLOW_MIN_MAGNITUDE,
    ) -> None:
        self._rois = rois
        self._arm_names = arm_names
        self._flow_min = flow_min_magnitude

        # Previous grayscale frame
        self._prev_gray: Optional[np.ndarray] = None

        # Previous feature points per arm: arm → (N,1,2) float32 or None
        self._prev_pts: dict[str, Optional[np.ndarray]] = {
            arm: None for arm in arm_names
        }

        # Rolling magnitude buffers for smoothing
        self._mag_buffers: dict[str, deque[float]] = {
            arm: deque(maxlen=self.ROLLING_WINDOW) for arm in arm_names
        }

        # Frame counter — drives periodic re-seeding
        self._frame_count: int = 0

        logger.info(
            "FlowAnalyser ready — %d arms, min_magnitude=%.1f",
            len(arm_names), flow_min_magnitude,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> FlowResult:
        """
        Compute per-arm optical flow for one new frame.

        Args:
            frame: BGR or grayscale uint8 frame (same resolution every call).

        Returns:
            FlowResult. On first call (no previous frame), all magnitudes = 0.
        """
        gray = self._to_gray(frame)
        result = FlowResult(
            magnitudes    = {arm: 0.0   for arm in self._arm_names},
            smoothed      = {arm: 0.0   for arm in self._arm_names},
            is_stopped    = {arm: True  for arm in self._arm_names},
            tracked_points= {arm: 0     for arm in self._arm_names},
            vectors       = {arm: []    for arm in self._arm_names},
        )

        self._frame_count += 1
        force_reseed = (self._frame_count % self.RESEED_INTERVAL == 0)

        if self._prev_gray is None:
            # First frame — seed features, nothing to track yet
            self._seed_all(gray)
            self._prev_gray = gray
            return result

        # ── Track existing features with LK ──────────────────────────────
        for arm in self._arm_names:
            prev_pts = self._prev_pts.get(arm)

            needs_reseed = (
                force_reseed
                or prev_pts is None
                or len(prev_pts) < self.MIN_POINTS_THRESHOLD
            )

            if needs_reseed:
                self._seed_arm(arm, self._prev_gray)
                prev_pts = self._prev_pts[arm]

            if prev_pts is None or len(prev_pts) == 0:
                # No features in this ROI (empty road / very dark patch)
                self._mag_buffers[arm].append(0.0)
                result.magnitudes[arm]     = 0.0
                result.tracked_points[arm] = 0
            else:
                mag, tracked, vectors, next_pts = self._track_arm(
                    self._prev_gray, gray, prev_pts
                )
                # Save surviving points for next frame
                self._prev_pts[arm] = next_pts

                self._mag_buffers[arm].append(mag)
                result.magnitudes[arm]     = mag
                result.tracked_points[arm] = tracked
                result.vectors[arm]        = vectors

        # ── Smoothed magnitudes ───────────────────────────────────────────
        for arm in self._arm_names:
            buf = self._mag_buffers[arm]
            smoothed = float(np.mean(buf)) if buf else 0.0
            result.smoothed[arm]   = smoothed
            result.is_stopped[arm] = smoothed < self._flow_min

        self._prev_gray = gray
        return result

    def reset(self) -> None:
        """Clear all state — call on video source change or manual reset."""
        self._prev_gray = None
        self._prev_pts  = {arm: None for arm in self._arm_names}
        for buf in self._mag_buffers.values():
            buf.clear()
        self._frame_count = 0
        logger.info("FlowAnalyser state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR → grayscale if needed. Already gray → pass through."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _seed_all(self, gray: np.ndarray) -> None:
        """Seed feature points for all arms from the given grayscale frame."""
        for arm in self._arm_names:
            self._seed_arm(arm, gray)

    def _seed_arm(self, arm: str, gray: np.ndarray) -> None:
        """
        Detect Shi-Tomasi corners inside the arm's ROI polygon.
        Stores result in self._prev_pts[arm].

        Steps:
          1. Create a binary mask from the ROI polygon.
          2. Run goodFeaturesToTrack constrained to that mask.
          3. Store as (N,1,2) float32 for LK input format.
        """
        polygon = self._rois.get(arm)
        if polygon is None:
            self._prev_pts[arm] = None
            return

        # Build ROI mask
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **_SHITOMASI_PARAMS)

        if pts is None or len(pts) == 0:
            self._prev_pts[arm] = None
        else:
            self._prev_pts[arm] = pts.astype(np.float32)

    def _track_arm(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> tuple[float, int, list[tuple], Optional[np.ndarray]]:
        """
        Run LK optical flow on prev_pts from prev_gray to curr_gray.

        Returns:
            (mean_magnitude, num_tracked_points, vectors, surviving_pts)
            vectors: list of ((px, py), (cx, cy)) tuples for debug drawing.
        """
        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
            )
        except cv2.error as exc:
            logger.debug("LK flow error (non-fatal): %s", exc)
            return 0.0, 0, [], None

        if next_pts is None or status is None:
            return 0.0, 0, [], None

        # Keep only points where LK converged (status == 1)
        good_mask   = status.ravel() == 1
        good_prev   = prev_pts[good_mask]
        good_next   = next_pts[good_mask]

        tracked = int(np.sum(good_mask))

        if tracked == 0:
            return 0.0, 0, [], None

        # Motion vectors and magnitudes
        deltas     = good_next - good_prev                        # (N,1,2)
        magnitudes = np.linalg.norm(deltas.reshape(-1, 2), axis=1)  # (N,)
        mean_mag   = float(np.mean(magnitudes))

        # Build vector list for debug drawing
        vectors: list[tuple] = []
        for p, n in zip(good_prev.reshape(-1, 2), good_next.reshape(-1, 2)):
            vectors.append(((int(p[0]), int(p[1])), (int(n[0]), int(n[1]))))

        # Return surviving points for next iteration
        surviving = good_next.reshape(-1, 1, 2)
        return mean_mag, tracked, vectors, surviving


# ---------------------------------------------------------------------------
# Debug drawing helper (called from drawing.py or main.py if D key active)
# ---------------------------------------------------------------------------

def draw_flow_vectors(
    frame: np.ndarray,
    flow_result: FlowResult,
    arm_colors: Optional[dict[str, tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw optical flow motion arrows on the frame.

    Each arrow goes from previous feature point to current feature point.
    Arrow colour matches the arm's ROI colour.

    Args:
        frame:       BGR frame to annotate.
        flow_result: Output of FlowAnalyser.update().
        arm_colors:  Per-arm BGR colour override. Defaults to ROI_COLORS.

    Returns:
        Annotated BGR frame.
    """
    from config import ROI_COLORS

    out = frame.copy()
    colors = arm_colors or ROI_COLORS

    for arm, vectors in flow_result.vectors.items():
        color = colors.get(arm, (200, 200, 200))

        for (px, py), (cx, cy) in vectors:
            cv2.arrowedLine(
                out,
                (px, py), (cx, cy),
                color,
                thickness=1,
                tipLength=0.4,
            )

        # Magnitude label at top-right of ROI
        polygon = ROIS.get(arm)
        if polygon is not None:
            rx = int(np.max(polygon[:, 0])) - 60
            ry = int(np.min(polygon[:, 1])) + 14
            mag = flow_result.smoothed.get(arm, 0.0)
            stopped = flow_result.is_stopped.get(arm, True)
            label = f"{arm[:1]} {mag:.1f}px {'STOP' if stopped else 'FLOW'}"
            cv2.putText(out, label, (rx, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 180) if stopped else color, 1)

    return out


# ---------------------------------------------------------------------------
# Standalone test  (python -m detection.flow)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import time

    from config import VIDEO_SOURCE
    from utils.preprocessing import preprocess
    from utils.drawing import draw_rois, draw_frame_info

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    analyser = FlowAnalyser()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print(
        "Press Q to quit\n"
        "Shows: optical flow arrows per arm + STOP/FLOW labels\n"
    )

    frame_n = 0
    t0      = time.time()
    fps     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            analyser.reset()
            continue

        processed = preprocess(frame)
        result    = analyser.update(processed)

        frame_n += 1
        if frame_n % 30 == 0:
            fps = 30.0 / (time.time() - t0)
            t0  = time.time()
            for arm in ARM_NAMES:
                mag     = result.smoothed.get(arm, 0.0)
                stopped = result.is_stopped.get(arm, True)
                pts     = result.tracked_points.get(arm, 0)
                print(
                    f"  {arm:<6}  mag={mag:5.2f} px/frame  "
                    f"tracked={pts:3d}  {'STOPPED' if stopped else 'FLOWING'}"
                )
            print()

        out = draw_rois(processed)
        out = draw_flow_vectors(out, result)
        out = draw_frame_info(out, frame_n, fps)

        cv2.imshow('Flow Test — LK Optical Flow', out)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")