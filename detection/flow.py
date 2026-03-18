"""
detection/flow.py — Direction-Aware Optical Flow Analyser (Upgraded)
======================================================================
Improvements over original:
  1. Direction-aware flow — computes mean angle of motion vectors per arm
     → 'toward' | 'away' | 'stopped' | 'unknown' (used by density.py + algorithm)
  2. Per-arm result includes angle_deg for direction classification
  3. Separate toward/away magnitude — discharge_rate estimation uses
     only "away" vectors (vehicles actually leaving the intersection)
  4. Discharge rate output — PCU/s estimate from flow magnitude × saturation
  5. Flow quality metric — filters bad optical flow estimates (< min_features)

Kept intact from original:
  • Lucas-Kanade sparse flow (not Farneback — too slow on CPU)
  • Shi-Tomasi feature detection
  • Per-arm ROI masking
  • ~2ms/frame target on CPU
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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

# ── Optical flow parameters ───────────────────────────────────────────────────
_SHITOMASI_PARAMS = dict(
    maxCorners=FLOW_MAX_FEATURES,
    qualityLevel=FLOW_QUALITY_LEVEL,
    minDistance=FLOW_MIN_DISTANCE,
    blockSize=7,
)

_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Minimum features to trust flow result
_MIN_VALID_FEATURES = 3
# Saturated flow rate (PCU/s) for discharge estimation
_SATURATION_FLOW = 0.35
# Smoothing factor for per-arm flow history
_SMOOTH = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm flow result (new fields added)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArmFlowResult:
    arm: str
    magnitude: float = 0.0          # mean flow magnitude (px/frame)
    angle_deg: float = 0.0          # mean flow angle (0=right, 90=down, 180=left, 270=up)
    toward_magnitude: float = 0.0   # magnitude of vectors pointing "toward" camera
    away_magnitude: float = 0.0     # magnitude of vectors pointing "away" from camera
    is_moving: bool = False
    feature_count: int = 0
    discharge_rate: float = 0.0     # estimated PCU/s leaving intersection
    flow_quality: str = 'unknown'   # 'good' | 'low_features' | 'stopped' | 'unknown'


@dataclass
class FlowResult:
    """Full frame flow result. Backward-compat with old .flow_rates dict."""
    arm_results: Dict[str, ArmFlowResult] = field(default_factory=dict)

    @property
    def flow_rates(self) -> Dict[str, float]:
        """Backward-compatible property for existing controller code."""
        return {arm: r.magnitude for arm, r in self.arm_results.items()}

    def to_dict(self) -> Dict[str, dict]:
        """Full per-arm flow data for density.py direction classification."""
        return {
            arm: {
                'magnitude':        r.magnitude,
                'angle_deg':        r.angle_deg,
                'toward_magnitude': r.toward_magnitude,
                'away_magnitude':   r.away_magnitude,
                'is_moving':        r.is_moving,
                'discharge_rate':   r.discharge_rate,
                'flow_quality':     r.flow_quality,
            }
            for arm, r in self.arm_results.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Flow Analyser
# ─────────────────────────────────────────────────────────────────────────────

class FlowAnalyser:
    """
    Sparse Lucas-Kanade optical flow with direction analysis.

    Direction reference (image coordinates, top-left origin):
      0°   = right
      90°  = down (toward bottom of frame = toward camera in typical mount)
      180° = left
      270° = up (toward top of frame = away from camera)

    For a typical intersection camera mounted overhead looking down the road:
      "toward" vectors point roughly downward (90° ± 60°) → vehicles approaching
      "away" vectors point roughly upward (270° ± 60°) → vehicles departing
    """

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

        # Per-arm masks (pre-built polygon masks)
        self._arm_masks: Dict[str, np.ndarray] = {}
        self._masks_built = False

        # Per-arm smoothed magnitude history
        self._smooth_magnitude: Dict[str, float] = {arm: 0.0 for arm in ARM_NAMES}
        self._smooth_angle: Dict[str, float] = {arm: 90.0 for arm in ARM_NAMES}

        self._frame_count = 0

        logger.info("FlowAnalyser initialized")

    def _build_masks(self, h: int, w: int) -> None:
        """Build binary polygon masks for each arm ROI (done once)."""
        for arm in ARM_NAMES:
            if arm not in ROIS:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(ROIS[arm], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self._arm_masks[arm] = mask
        self._masks_built = True

    def update(self, frame: np.ndarray) -> FlowResult:
        """
        Compute per-arm optical flow from consecutive frames.

        Args:
            frame: BGR frame (preprocessed with CLAHE)

        Returns:
            FlowResult with per-arm magnitude, angle, direction
        """
        result = FlowResult()
        result.arm_results = {arm: ArmFlowResult(arm=arm) for arm in ARM_NAMES}

        if frame is None:
            return result

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Build masks on first frame
        if not self._masks_built:
            self._build_masks(h, w)

        self._frame_count += 1

        # Need at least 2 frames
        if self._prev_gray is None:
            self._prev_gray = gray
            return result

        # ── Per-arm flow computation ───────────────────────────────────────
        for arm in ARM_NAMES:
            if arm not in self._arm_masks:
                continue

            mask = self._arm_masks[arm]
            ar = result.arm_results[arm]

            # Detect features in previous frame within this arm's ROI
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                mask=mask,
                **_SHITOMASI_PARAMS
            )

            if prev_pts is None or len(prev_pts) < _MIN_VALID_FEATURES:
                ar.flow_quality = 'low_features'
                ar.feature_count = 0 if prev_pts is None else len(prev_pts)
                # Apply smoothing (decay toward zero)
                self._smooth_magnitude[arm] *= (1 - _SMOOTH)
                ar.magnitude = self._smooth_magnitude[arm]
                continue

            ar.feature_count = len(prev_pts)

            # Compute Lucas-Kanade flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev_pts, None, **_LK_PARAMS
            )

            if next_pts is None or status is None:
                ar.flow_quality = 'error'
                continue

            # Filter to good points only
            status = status.flatten()
            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]

            if len(good_prev) < _MIN_VALID_FEATURES:
                ar.flow_quality = 'low_features'
                self._smooth_magnitude[arm] *= (1 - _SMOOTH)
                ar.magnitude = self._smooth_magnitude[arm]
                continue

            # ── Compute motion vectors ────────────────────────────────────
            vectors = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)

            # Magnitudes
            magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
            mean_mag = float(np.mean(magnitudes))

            # Filter outlier magnitudes (vehicles that jumped due to ID switch)
            valid_mask = magnitudes < np.percentile(magnitudes, 90)
            if valid_mask.sum() < _MIN_VALID_FEATURES:
                valid_mask = np.ones(len(magnitudes), dtype=bool)

            filtered_vecs = vectors[valid_mask]
            filtered_mags = magnitudes[valid_mask]

            mean_mag = float(np.mean(filtered_mags))

            # ── Mean angle (circular mean via unit vector sum) ────────────
            angles_rad = np.arctan2(filtered_vecs[:, 1], filtered_vecs[:, 0])
            mean_sin = float(np.mean(np.sin(angles_rad)))
            mean_cos = float(np.mean(np.cos(angles_rad)))
            mean_angle_rad = np.arctan2(mean_sin, mean_cos)
            mean_angle_deg = float(np.degrees(mean_angle_rad)) % 360

            # ── Direction split (toward vs away) ─────────────────────────
            # toward = 90° ± 70° (roughly downward)
            # away   = 270° ± 70° (roughly upward)
            toward_mask = self._angle_in_band(angles_rad, target_deg=90, band_deg=70)
            away_mask   = self._angle_in_band(angles_rad, target_deg=270, band_deg=70)

            toward_mag = float(np.mean(filtered_mags[toward_mask])) if toward_mask.any() else 0.0
            away_mag   = float(np.mean(filtered_mags[away_mask]))   if away_mask.any()   else 0.0

            # ── Smooth magnitude and angle ────────────────────────────────
            self._smooth_magnitude[arm] = (
                _SMOOTH * mean_mag + (1 - _SMOOTH) * self._smooth_magnitude[arm]
            )
            # Smooth angle using circular mean
            self._smooth_angle[arm] = (
                _SMOOTH * mean_angle_deg + (1 - _SMOOTH) * self._smooth_angle[arm]
            )

            smoothed_mag = self._smooth_magnitude[arm]

            # ── Discharge rate estimate ───────────────────────────────────
            # Vehicles leaving = away flow normalized to PCU/s estimate
            # Only meaningful if significant away flow exists
            if away_mag > FLOW_MIN_MAGNITUDE:
                # Normalize: 5 px/frame ≈ 1 car passing per second at typical scale
                discharge = min(1.0, away_mag / 5.0) * _SATURATION_FLOW
            else:
                discharge = 0.0

            # ── Populate result ───────────────────────────────────────────
            ar.magnitude = round(smoothed_mag, 3)
            ar.angle_deg = round(self._smooth_angle[arm], 1)
            ar.toward_magnitude = round(toward_mag, 3)
            ar.away_magnitude = round(away_mag, 3)
            ar.is_moving = smoothed_mag > FLOW_MIN_MAGNITUDE
            ar.discharge_rate = round(discharge, 4)
            ar.flow_quality = 'good' if smoothed_mag > FLOW_MIN_MAGNITUDE else 'stopped'

        self._prev_gray = gray
        return result

    @staticmethod
    def _angle_in_band(
        angles_rad: np.ndarray,
        target_deg: float,
        band_deg: float,
    ) -> np.ndarray:
        """
        Return boolean mask for angles within ±band_deg of target_deg.
        Handles circular wrap-around.
        """
        target_rad = np.radians(target_deg)
        diff = np.abs(angles_rad - target_rad)
        diff = np.minimum(diff, 2 * np.pi - diff)   # circular wrap
        band_rad = np.radians(band_deg)
        return diff <= band_rad

    def reset(self) -> None:
        """Reset flow state (on video source switch)."""
        self._prev_gray = None
        self._smooth_magnitude = {arm: 0.0 for arm in ARM_NAMES}
        self._smooth_angle = {arm: 90.0 for arm in ARM_NAMES}
        logger.info("FlowAnalyser reset")