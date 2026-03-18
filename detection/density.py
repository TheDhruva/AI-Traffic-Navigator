"""
detection/density.py — PCU-Weighted Density + Queue Length Estimator
=====================================================================
Improvements over original:
  1. queue_length() — converts PCU density to estimated vehicle count
     using class-specific PCU weights (not a fixed avg)
  2. Direction-aware stopped detection — classifies flow as
     'toward'|'away'|'stopped'|'unknown' for better scoring
  3. Closest emergency vehicle — returns which arm the nearest
     ambulance is in (by centroid distance to ROI center)
  4. Pedestrian stability — 15-frame rolling average + hysteresis
     (requires avg >= PED_THRESHOLD to trigger, avg < PED_CLEAR to cancel)
  5. Arrival rate estimation — delta between consecutive density readings
     (used by algorithm.py for predictive scoring)

Kept intact from original:
  • Polygon ROI centroid test (cv2.pointPolygonTest)
  • PCU weights (bus/truck=3.0, car=1.0, auto=0.8, moto=0.4, bike=0.3)
  • CLAHE mandatory upstream
  • Low conf threshold (0.35) — handled in detector.py
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

# Pedestrian clear threshold (hysteresis: trigger at >=8, clear at <4)
PED_CLEAR_THRESHOLD = 4

# Minimum flow magnitude to classify as "moving" (pixels/frame)
FLOW_MOVING_THRESHOLD = 1.5
# Flow direction angle band for "toward" vs "away" (degrees)
# In a top-down view: toward camera = generally downward (90°±60°)
TOWARD_ANGLE_CENTER = 90.0
TOWARD_ANGLE_BAND = 70.0


# ─────────────────────────────────────────────────────────────────────────────
# Detection dataclass (matches output from detector.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single YOLO detection. Must match detector.py output."""
    xyxy: Tuple[float, float, float, float]   # x1,y1,x2,y2 in pixels
    conf: float
    cls_id: int
    cls_name: str

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2.0

    @property
    def area(self) -> float:
        w = self.xyxy[2] - self.xyxy[0]
        h = self.xyxy[3] - self.xyxy[1]
        return max(0.0, w * h)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArmDensityResult:
    """Per-arm density result for one frame."""
    arm: str
    density: float = 0.0              # PCU-weighted sum
    queue_length: int = 0             # estimated vehicle count
    vehicle_count: int = 0            # raw count (unweighted)
    flow_direction: str = 'unknown'   # 'toward'|'away'|'stopped'|'unknown'
    arrival_rate_delta: float = 0.0   # density change since last frame (PCU/frame)
    has_emergency: bool = False
    has_hazard: bool = False
    matched_detections: List[Detection] = field(default_factory=list)


@dataclass
class DensityResult:
    """Full frame density result across all arms."""
    arm_results: Dict[str, ArmDensityResult] = field(default_factory=dict)

    # Backward-compat properties (used by existing controller + drawing code)
    @property
    def densities(self) -> Dict[str, float]:
        return {arm: r.density for arm, r in self.arm_results.items()}

    @property
    def emergency_arms(self) -> List[str]:
        return [arm for arm, r in self.arm_results.items() if r.has_emergency]

    @property
    def hazard_arms(self) -> List[str]:
        return [arm for arm, r in self.arm_results.items() if r.has_hazard]

    @property
    def ped_count(self) -> int:
        return self._ped_count

    @property
    def ped_rolling_avg(self) -> float:
        return self._ped_rolling_avg

    @property
    def ped_phase_triggered(self) -> bool:
        return self._ped_phase_triggered

    @property
    def matched_detections(self) -> list:
        all_dets = []
        for r in self.arm_results.values():
            all_dets.extend(r.matched_detections)
        return all_dets

    # Set by DensityEstimator.update()
    _ped_count: int = 0
    _ped_rolling_avg: float = 0.0
    _ped_phase_triggered: bool = False

    # Closest emergency arm (new field)
    closest_emergency_arm: Optional[str] = None

    def __repr__(self) -> str:
        dens = {k: f"{v:.1f}" for k, v in self.densities.items()}
        return (
            f"DensityResult(densities={dens} "
            f"ped={self._ped_count:.0f} avg={self._ped_rolling_avg:.1f} "
            f"emrg={self.emergency_arms} hazard={self.hazard_arms})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core estimator
# ─────────────────────────────────────────────────────────────────────────────

class DensityEstimator:
    """
    PCU-weighted density + queue estimator for all intersection arms.

    Usage:
        estimator = DensityEstimator()
        result = estimator.update(detections, flow_vectors=flow_result)
    """

    def __init__(self) -> None:
        # Previous density per arm for arrival rate delta
        self._prev_density: Dict[str, float] = {arm: 0.0 for arm in ARM_NAMES}

        # Pedestrian rolling buffer
        self._ped_buffer: deque = deque(maxlen=PED_ROLLING_FRAMES)
        self._ped_triggered: bool = False   # hysteresis state

        # Hazard clear frame counter per arm
        self._hazard_clear_count: Dict[str, int] = {arm: 0 for arm in ARM_NAMES}

        # ROI polygon arrays (pre-built for speed)
        self._roi_polygons: Dict[str, np.ndarray] = {}
        self._roi_centers: Dict[str, Tuple[float, float]] = {}
        self._build_roi_data()

        logger.info("DensityEstimator initialized for arms: %s", ARM_NAMES)

    def _build_roi_data(self) -> None:
        """Pre-compute ROI polygon arrays and centroid coordinates."""
        for arm in ARM_NAMES:
            if arm not in ROIS:
                logger.warning("No ROI defined for arm: %s", arm)
                continue
            pts = np.array(ROIS[arm], dtype=np.float32)
            self._roi_polygons[arm] = pts
            # Centroid = mean of polygon vertices
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            self._roi_centers[arm] = (cx, cy)

        # Pedestrian ROI
        if 'PED' in ROIS:
            self._ped_roi = np.array(ROIS['PED'], dtype=np.float32)
        else:
            self._ped_roi = None
            logger.warning("No PED ROI defined — pedestrian phase disabled")

    def update(
        self,
        detections: list,
        flow_vectors: Optional[dict] = None,
    ) -> DensityResult:
        """
        Process one frame of detections.

        Args:
            detections: list of Detection objects from detector.py
            flow_vectors: optional dict from flow.py with per-arm flow info

        Returns:
            DensityResult with per-arm density, queue, emergency, hazard, ped data
        """
        result = DensityResult()
        arm_results: Dict[str, ArmDensityResult] = {
            arm: ArmDensityResult(arm=arm) for arm in ARM_NAMES
        }

        # ── Step 1: classify each detection into an arm ───────────────────
        ped_count_frame = 0
        emergency_candidates: List[Tuple[str, float]] = []  # (arm, dist_to_center)

        for det in detections:
            if not hasattr(det, 'cx'):
                # Support both Detection dataclass and dict/object from detector.py
                try:
                    cx = (det.xyxy[0] + det.xyxy[2]) / 2.0
                    cy = (det.xyxy[1] + det.xyxy[3]) / 2.0
                    cls_name = det.cls_name
                except AttributeError:
                    continue
            else:
                cx, cy = det.cx, det.cy
                cls_name = det.cls_name

            # ── Pedestrian ROI check ───────────────────────────────────────
            if cls_name == 'person' and self._ped_roi is not None:
                pt_test = cv2.pointPolygonTest(
                    self._ped_roi, (float(cx), float(cy)), False
                )
                if pt_test >= 0:
                    ped_count_frame += 1
                continue   # persons not counted in PCU density

            # ── Arm ROI assignment ─────────────────────────────────────────
            assigned_arm: Optional[str] = None
            for arm, polygon in self._roi_polygons.items():
                test = cv2.pointPolygonTest(
                    polygon, (float(cx), float(cy)), False
                )
                if test >= 0:
                    assigned_arm = arm
                    break

            if assigned_arm is None:
                continue

            ar = arm_results[assigned_arm]
            ar.matched_detections.append(det)
            ar.vehicle_count += 1

            # ── Emergency vehicle ──────────────────────────────────────────
            if cls_name in EMERGENCY_CLASSES:
                ar.has_emergency = True
                # Distance from vehicle to ROI center (for closest-arm logic)
                roi_cx, roi_cy = self._roi_centers[assigned_arm]
                dist = math.hypot(cx - roi_cx, cy - roi_cy)
                emergency_candidates.append((assigned_arm, dist))
                continue   # emergency vehicles not counted in density

            # ── Hazard (animal) ────────────────────────────────────────────
            if cls_name in HAZARD_CLASSES:
                ar.has_hazard = True
                self._hazard_clear_count[assigned_arm] = 0
                continue

            # ── PCU density ───────────────────────────────────────────────
            pcu = PCU_WEIGHTS.get(cls_name, 0.5)
            ar.density += pcu

        # ── Step 2: finalize per-arm results ──────────────────────────────
        for arm, ar in arm_results.items():
            # Hazard clear countdown
            if not ar.has_hazard:
                self._hazard_clear_count[arm] = min(
                    self._hazard_clear_count[arm] + 1, HAZARD_CLEAR_FRAMES + 1
                )
                if self._hazard_clear_count[arm] < HAZARD_CLEAR_FRAMES:
                    ar.has_hazard = True   # hold hazard flag until truly cleared

            # Queue length (PCU → vehicle count)
            ar.queue_length = self._density_to_queue(ar.density, ar.matched_detections)

            # Arrival rate delta
            prev = self._prev_density[arm]
            ar.arrival_rate_delta = ar.density - prev
            self._prev_density[arm] = ar.density

            # Flow direction from flow_vectors
            if flow_vectors and arm in flow_vectors:
                ar.flow_direction = self._classify_flow_direction(
                    flow_vectors[arm]
                )

        # ── Step 3: closest emergency arm ─────────────────────────────────
        closest_emrg_arm = None
        if emergency_candidates:
            closest_emrg_arm = min(emergency_candidates, key=lambda x: x[1])[0]

        # ── Step 4: pedestrian rolling average + hysteresis ───────────────
        self._ped_buffer.append(ped_count_frame)
        ped_avg = float(sum(self._ped_buffer) / max(1, len(self._ped_buffer)))

        # Hysteresis: trigger at >=PED_THRESHOLD, clear at <PED_CLEAR_THRESHOLD
        if not self._ped_triggered and ped_avg >= PED_THRESHOLD:
            self._ped_triggered = True
            logger.info("Pedestrian phase triggered (avg=%.1f >= %d)", ped_avg, PED_THRESHOLD)
        elif self._ped_triggered and ped_avg < PED_CLEAR_THRESHOLD:
            self._ped_triggered = False

        # ── Assemble result ────────────────────────────────────────────────
        result.arm_results = arm_results
        result._ped_count = ped_count_frame
        result._ped_rolling_avg = ped_avg
        result._ped_phase_triggered = self._ped_triggered
        result.closest_emergency_arm = closest_emrg_arm

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _density_to_queue(
        self,
        density: float,
        detections: List,
    ) -> int:
        """
        Convert PCU density to estimated vehicle count.
        Uses actual class composition if detections are available,
        otherwise uses a default average PCU of 0.7.
        """
        if not detections:
            avg_pcu = 0.7
            return max(0, int(round(density / max(avg_pcu, 0.01))))

        total_pcu = 0.0
        count = 0
        for det in detections:
            cls_name = getattr(det, 'cls_name', 'unknown')
            if cls_name == 'person':
                continue
            pcu = PCU_WEIGHTS.get(cls_name, 0.5)
            if pcu > 0:
                total_pcu += pcu
                count += 1

        if count == 0:
            return 0
        avg_pcu = total_pcu / count
        return max(0, int(round(density / max(avg_pcu, 0.01))))

    def _classify_flow_direction(self, flow_info: dict) -> str:
        """
        Classify optical flow direction for one arm.

        flow_info expected keys:
          'magnitude': float       — mean flow magnitude (px/frame)
          'angle_deg': float       — mean flow angle (0–360°, 0=right, 90=down)

        Returns: 'toward' | 'away' | 'stopped' | 'unknown'
        """
        if not isinstance(flow_info, dict):
            return 'unknown'

        magnitude = flow_info.get('magnitude', 0.0)
        angle = flow_info.get('angle_deg', None)

        if magnitude < FLOW_MOVING_THRESHOLD:
            return 'stopped'

        if angle is None:
            return 'unknown'

        # Check if angle is within TOWARD_ANGLE_BAND of TOWARD_ANGLE_CENTER
        # Default: toward = roughly "downward" in image = vehicles approaching
        angle_diff = abs(angle - TOWARD_ANGLE_CENTER) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff <= TOWARD_ANGLE_BAND:
            return 'toward'
        else:
            return 'away'

    def reset(self) -> None:
        """Reset all internal state (useful for tests or source switches)."""
        self._prev_density = {arm: 0.0 for arm in ARM_NAMES}
        self._ped_buffer.clear()
        self._ped_triggered = False
        self._hazard_clear_count = {arm: 0 for arm in ARM_NAMES}
        logger.info("DensityEstimator reset")