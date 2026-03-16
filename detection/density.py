# detection/density.py — PCU-Weighted Density Estimation Per ROI Arm
# For each arm, counts detections whose centroid falls inside the ROI polygon,
# weights them by PCU, and returns per-arm density scores.
# Also flags emergency and hazard conditions discovered during the sweep.

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
    PCU_WEIGHTS,
    HAZARD_CLASSES,
    EMERGENCY_CLASSES,
    PED_THRESHOLD,
    PED_ROLLING_FRAMES,
    HAZARD_CLEAR_FRAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container returned after every frame sweep
# ---------------------------------------------------------------------------

@dataclass
class DensityResult:
    """Per-frame density estimation output for all arms."""

    # PCU-weighted vehicle density per arm  e.g. {'North': 12.4, 'South': 3.0, ...}
    densities: dict[str, float] = field(default_factory=dict)

    # Arms where an emergency vehicle was detected this frame
    emergency_arms: list[str] = field(default_factory=list)

    # Arms where a hazard (animal) was detected this frame
    hazard_arms: list[str] = field(default_factory=list)

    # Raw person count inside the PED ROI this frame
    ped_count: int = 0

    # Rolling average ped count (smoothed over PED_ROLLING_FRAMES)
    ped_rolling_avg: float = 0.0

    # True when rolling average >= PED_THRESHOLD
    ped_phase_triggered: bool = False

    # Detections that fell inside at least one ROI — for drawing / debugging
    matched_detections: list = field(default_factory=list)

    def __repr__(self) -> str:
        dens = {k: f"{v:.1f}" for k, v in self.densities.items()}
        return (
            f"DensityResult(densities={dens} "
            f"ped={self.ped_count} avg={self.ped_rolling_avg:.1f} "
            f"emrg={self.emergency_arms} hazard={self.hazard_arms})"
        )


# ---------------------------------------------------------------------------
# Density Estimator — instantiate once, call update() every frame
# ---------------------------------------------------------------------------

class DensityEstimator:
    """
    Computes PCU-weighted vehicle density for each intersection arm.

    Design decisions:
      - Uses cv2.pointPolygonTest (exact, handles concave polygons, fast in C).
      - Centroid-based assignment: a vehicle belongs to the arm whose ROI
        contains its bounding-box centre. This handles vehicles that straddle
        multiple ROIs at the stop-line.
      - Animals and emergency vehicles are flagged separately from PCU density
        so the controller can handle them independently.
      - Pedestrians in the VEHICLE ROIs contribute 0 PCU (config) but are
        counted in the PED ROI for the pedestrian phase trigger.
      - Rolling average over PED_ROLLING_FRAMES prevents a single frame with
        spurious detections from triggering a ped phase.

    Usage:
        estimator = DensityEstimator()
        result = estimator.update(detections)
    """

    def __init__(
        self,
        rois: dict[str, np.ndarray] = ROIS,
        arm_names: list[str] = ARM_NAMES,
        ped_threshold: int = PED_THRESHOLD,
        ped_rolling_frames: int = PED_ROLLING_FRAMES,
        hazard_clear_frames: int = HAZARD_CLEAR_FRAMES,
    ) -> None:
        self._rois = rois
        self._arm_names = arm_names
        self._ped_threshold = ped_threshold

        # Rolling buffer for pedestrian count smoothing
        self._ped_buffer: deque[int] = deque(maxlen=ped_rolling_frames)

        # Per-arm hazard clear countdown: tracks consecutive hazard-free frames
        self._hazard_clear_countdown: dict[str, int] = {
            arm: hazard_clear_frames for arm in arm_names
        }
        self._hazard_clear_frames = hazard_clear_frames

        # Per-arm active hazard state (survives until HAZARD_CLEAR_FRAMES pass)
        self._active_hazards: dict[str, bool] = {arm: False for arm in arm_names}

        logger.info(
            "DensityEstimator ready — %d arms, ped_threshold=%d, rolling=%d frames",
            len(arm_names), ped_threshold, ped_rolling_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list) -> DensityResult:
        """
        Process one frame's detections and return density estimates.

        Args:
            detections: List of Detection objects from TrafficDetector.detect().

        Returns:
            DensityResult with per-arm densities and event flags.
        """
        result = DensityResult()

        # Initialise density to 0 for every arm
        result.densities = {arm: 0.0 for arm in self._arm_names}

        ped_count_this_frame: int = 0
        matched: list = []

        for det in detections:
            # ── Pedestrian ROI check ──────────────────────────────────────
            if det.class_name == 'person':
                if self._point_in_roi('PED', det.cx, det.cy):
                    ped_count_this_frame += 1
                # Persons are 0 PCU in vehicle ROIs — skip density accumulation
                continue

            # ── Emergency vehicle check ───────────────────────────────────
            if det.is_emergency:
                arm = self._centroid_to_arm(det.cx, det.cy)
                if arm and arm not in result.emergency_arms:
                    result.emergency_arms.append(arm)
                    logger.warning(
                        "EMERGENCY: %s detected in %s arm (conf=%.2f)",
                        det.class_name, arm, det.confidence,
                    )
                matched.append(det)
                continue  # Emergency vehicles don't add PCU density

            # ── Hazard (animal) check ────────────────────────────────────
            if det.is_hazard:
                arm = self._centroid_to_arm(det.cx, det.cy)
                if arm:
                    self._active_hazards[arm] = True
                    self._hazard_clear_countdown[arm] = self._hazard_clear_frames
                    if arm not in result.hazard_arms:
                        result.hazard_arms.append(arm)
                        logger.warning(
                            "HAZARD: %s on %s arm (conf=%.2f)",
                            det.class_name, arm, det.confidence,
                        )
                matched.append(det)
                continue  # Animals don't add PCU density

            # ── Normal vehicle — accumulate PCU density ──────────────────
            arm = self._centroid_to_arm(det.cx, det.cy)
            if arm is not None:
                pcu = PCU_WEIGHTS.get(det.class_name, PCU_WEIGHTS['unknown'])
                result.densities[arm] = result.densities.get(arm, 0.0) + pcu
                matched.append(det)

        # ── Hazard persistence: keep hazard flagged until clear countdown ──
        for arm in self._arm_names:
            if self._active_hazards[arm]:
                # Tick down the clear counter
                if arm not in result.hazard_arms:
                    # No hazard this frame — count down
                    self._hazard_clear_countdown[arm] -= 1
                    if self._hazard_clear_countdown[arm] <= 0:
                        self._active_hazards[arm] = False
                        logger.info("HAZARD cleared on %s arm", arm)
                    else:
                        # Still within clear window — keep hazard active
                        result.hazard_arms.append(arm)
                else:
                    # Hazard still present this frame — reset countdown
                    self._hazard_clear_countdown[arm] = self._hazard_clear_frames

        # ── Pedestrian rolling average ────────────────────────────────────
        self._ped_buffer.append(ped_count_this_frame)
        rolling_avg = float(np.mean(self._ped_buffer)) if self._ped_buffer else 0.0

        result.ped_count = ped_count_this_frame
        result.ped_rolling_avg = rolling_avg
        result.ped_phase_triggered = rolling_avg >= self._ped_threshold
        result.matched_detections = matched

        if result.ped_phase_triggered:
            logger.info(
                "Pedestrian phase triggered: rolling_avg=%.1f >= threshold=%d",
                rolling_avg, self._ped_threshold,
            )

        return result

    def reset(self) -> None:
        """Clear all rolling state — call on video source change or restart."""
        self._ped_buffer.clear()
        self._active_hazards = {arm: False for arm in self._arm_names}
        self._hazard_clear_countdown = {
            arm: self._hazard_clear_frames for arm in self._arm_names
        }
        logger.info("DensityEstimator state reset")

    def get_ped_rolling_avg(self) -> float:
        """Return the current pedestrian rolling average (thread-safe read)."""
        return float(np.mean(self._ped_buffer)) if self._ped_buffer else 0.0

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _point_in_roi(self, roi_name: str, cx: float, cy: float) -> bool:
        """
        Return True if (cx, cy) is inside the named ROI polygon.

        Uses cv2.pointPolygonTest with measureDist=False for maximum speed.
        Return value: +1.0 inside, -1.0 outside, 0.0 on edge — we treat
        0 (on edge) as inside to avoid missing stop-line vehicles.
        """
        polygon = self._rois.get(roi_name)
        if polygon is None:
            return False
        result = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False)
        return result >= 0.0  # 0 = on edge → count as inside

    def _centroid_to_arm(self, cx: float, cy: float) -> Optional[str]:
        """
        Return the arm name whose ROI contains (cx, cy), or None.

        Checks arms in ARM_NAMES order. If a centroid falls in two
        overlapping ROIs (e.g. near intersection box), the first match wins.
        PED ROI is excluded — handled separately.
        """
        for arm in self._arm_names:
            if self._point_in_roi(arm, cx, cy):
                return arm
        return None  # centroid outside all arm ROIs (e.g. in intersection box)

    def arm_densities_snapshot(self) -> dict[str, float]:
        """
        Return a copy of last-computed densities.
        Useful for the controller thread to read without re-running update().
        Note: update() should be the primary call; this is for diagnostics.
        """
        # Returns empty dict if called before first update() — safe
        return {}


# ---------------------------------------------------------------------------
# Module-level convenience function (used by main.py detection thread)
# ---------------------------------------------------------------------------

def compute_density(
    detections: list,
    estimator: DensityEstimator,
) -> DensityResult:
    """
    Thin wrapper so the detection thread can call a single function
    without holding an estimator reference itself.

    Args:
        detections: Output of TrafficDetector.detect().
        estimator:  Shared DensityEstimator instance (created in main.py).

    Returns:
        DensityResult for this frame.
    """
    return estimator.update(detections)


# ---------------------------------------------------------------------------
# Standalone test  (python -m detection.density)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import time
    import cv2 as _cv2

    from config import VIDEO_SOURCE
    from utils.preprocessing import preprocess
    from utils.drawing import draw_rois, draw_detections, draw_frame_info
    from detection.detector import TrafficDetector

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    detector  = TrafficDetector()
    estimator = DensityEstimator()

    cap = _cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print(
        "Press Q to quit\n"
        "Shows: ROI overlays + bboxes + per-arm PCU density printed to console\n"
    )

    frame_n = 0
    t0      = time.time()
    fps     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed  = preprocess(frame)
        detections = detector.detect(processed)
        result     = estimator.update(detections)

        # ── Console output every 30 frames ───────────────────────────────
        frame_n += 1
        if frame_n % 30 == 0:
            fps = 30.0 / (time.time() - t0)
            t0  = time.time()
            dens_str = "  ".join(
                f"{arm}: {result.densities.get(arm, 0):.1f} PCU"
                for arm in ARM_NAMES
            )
            print(
                f"[{frame_n:5d}]  {dens_str}  "
                f"| ped={result.ped_count} avg={result.ped_rolling_avg:.1f}"
                f"{'  🚨 EMERGENCY: ' + str(result.emergency_arms) if result.emergency_arms else ''}"
                f"{'  ⚠ HAZARD: '    + str(result.hazard_arms)    if result.hazard_arms    else ''}"
                f"{'  🚶 PED PHASE'                                if result.ped_phase_triggered else ''}"
            )

        # ── Visual output ────────────────────────────────────────────────
        out = draw_rois(processed)
        out = draw_detections(out, detections)

        # Density text per arm at ROI centroid
        for arm in ARM_NAMES:
            polygon = ROIS.get(arm)
            if polygon is None:
                continue
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1])) + 18   # below ROI label
            density = result.densities.get(arm, 0.0)
            label   = f"{density:.1f} PCU"
            _cv2.putText(out, label, (cx - 28, cy),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                         (255, 255, 255), 2)
            _cv2.putText(out, label, (cx - 28, cy),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                         (0, 0, 0), 1)

        # Ped count in PED ROI
        ped_roi = ROIS.get('PED')
        if ped_roi is not None:
            pcx = int(np.mean(ped_roi[:, 0]))
            pcy = int(np.mean(ped_roi[:, 1]))
            ped_label = f"PED {result.ped_count} (avg {result.ped_rolling_avg:.1f})"
            _cv2.putText(out, ped_label, (pcx - 50, pcy),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                         (0, 255, 255), 2)

        out = draw_frame_info(out, frame_n, fps)
        _cv2.imshow('Density Test — PCU per arm', out)

        if _cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    _cv2.destroyAllWindows()
    print("Done.")