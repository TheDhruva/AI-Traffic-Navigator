# detection/emergency.py — Emergency Vehicle & Pedestrian Phase Detector
# Consolidates all event-driven detection logic that sits above normal density scoring.
#
# Two responsibilities:
#   1. Emergency override  — detect ambulance / fire truck in any ROI arm,
#                            identify WHICH arm, signal the controller immediately.
#   2. Pedestrian phase    — maintain a confidence tracker with cooldown so a
#                            ped phase isn't re-triggered the moment it ends.
#
# This module does NOT change signals. It only sets flags on EmergencyResult.
# The controller (controller/algorithm.py) reads those flags and acts.

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    ARM_NAMES,
    ROIS,
    EMERGENCY_CLASSES,
    HAZARD_CLASSES,
    PED_THRESHOLD,
    PED_ROLLING_FRAMES,
    HAZARD_CLEAR_FRAMES,
    EMERGENCY_HOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EmergencyResult:
    """
    Output of EmergencyDetector.update() for one frame.

    The controller reads this every cycle and acts on any True flags.
    """

    # ── Emergency vehicle ────────────────────────────────────────────────
    emergency_detected: bool = False

    # Which arm the emergency vehicle centroid falls in (None if not detected)
    emergency_arm: Optional[str] = None

    # Class name that triggered the override e.g. 'ambulance'
    emergency_class: Optional[str] = None

    # Confidence of the triggering detection
    emergency_conf: float = 0.0

    # ── Pedestrian phase ─────────────────────────────────────────────────
    ped_phase_triggered: bool = False

    # Raw count this frame
    ped_count: int = 0

    # Smoothed rolling average
    ped_rolling_avg: float = 0.0

    # ── Hazard (animal) ──────────────────────────────────────────────────
    # Dict of arm_name → class_name for any active hazard
    hazard_arms: dict[str, str] = field(default_factory=dict)

    # ── Metadata ─────────────────────────────────────────────────────────
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        parts = []
        if self.emergency_detected:
            parts.append(
                f"EMERGENCY({self.emergency_class}@{self.emergency_arm} "
                f"conf={self.emergency_conf:.2f})"
            )
        if self.ped_phase_triggered:
            parts.append(f"PED(avg={self.ped_rolling_avg:.1f})")
        if self.hazard_arms:
            parts.append(f"HAZARD({self.hazard_arms})")
        return f"EmergencyResult({', '.join(parts) or 'clear'})"


# ---------------------------------------------------------------------------
# Emergency Detector
# ---------------------------------------------------------------------------

class EmergencyDetector:
    """
    Detects emergency vehicles and pedestrian rush conditions from
    a list of Detection objects produced by TrafficDetector.

    Emergency vehicle logic:
      - Any detection whose class_name is in EMERGENCY_CLASSES triggers an override.
      - The arm is identified by which ROI the detection centroid falls in.
      - A HOLD_FRAMES lockout prevents re-triggering until the current
        emergency green cycle ends (avoids rapid oscillation if two
        ambulances appear simultaneously on different arms).

    Pedestrian phase logic:
      - Person detections inside the PED ROI are counted each frame.
      - A rolling average over PED_ROLLING_FRAMES is maintained.
      - Phase is triggered when avg >= PED_THRESHOLD.
      - A cooldown of PED_COOLDOWN_SECONDS prevents immediate re-trigger
        after the ped WALK phase completes.

    Hazard logic:
      - Mirrors density.py hazard tracking but at the event level.
      - Hazard state persists for HAZARD_CLEAR_FRAMES after last detection.
    """

    # Frames to ignore new emergency detections after one is already active.
    # Prevents flicker if the same ambulance is detected on consecutive frames.
    HOLD_FRAMES = 3

    # Seconds to suppress ped phase re-trigger after it has fired.
    # Should be >= PED_WALK_DURATION (15s) + ALL_RED_BUFFER (1s).
    PED_COOLDOWN_SECONDS = 20.0

    def __init__(
        self,
        rois: dict[str, np.ndarray] = ROIS,
        arm_names: list[str] = ARM_NAMES,
        ped_threshold: int = PED_THRESHOLD,
        ped_rolling_frames: int = PED_ROLLING_FRAMES,
        hazard_clear_frames: int = HAZARD_CLEAR_FRAMES,
        emergency_classes: list[str] = EMERGENCY_CLASSES,
        hazard_classes: list[str] = HAZARD_CLASSES,
    ) -> None:
        self._rois              = rois
        self._arm_names         = arm_names
        self._ped_threshold     = ped_threshold
        self._emergency_classes = set(emergency_classes)
        self._hazard_classes    = set(hazard_classes)
        self._hazard_clear_n    = hazard_clear_frames

        # Emergency hold-off counter (counts down in frames)
        self._emergency_holdoff: int = 0

        # Most recent confirmed emergency arm — held for controller to read
        # across multiple frames without re-triggering.
        self._active_emergency_arm:   Optional[str] = None
        self._active_emergency_class: Optional[str] = None

        # Pedestrian rolling buffer
        self._ped_buffer: deque[int] = deque(maxlen=ped_rolling_frames)

        # Timestamp of last ped phase trigger (for cooldown)
        self._last_ped_trigger: float = 0.0

        # Hazard persistence per arm: arm → frames_until_clear
        self._hazard_countdown: dict[str, int] = {
            arm: 0 for arm in arm_names
        }
        # arm → last detected hazard class name
        self._hazard_class: dict[str, str] = {}

        logger.info(
            "EmergencyDetector ready — emergency_classes=%s  ped_threshold=%d",
            emergency_classes,
            ped_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list) -> EmergencyResult:
        """
        Scan one frame's detections for emergency events.

        Args:
            detections: List of Detection objects from TrafficDetector.detect().

        Returns:
            EmergencyResult with all active flags set.
        """
        result = EmergencyResult()

        ped_count_this_frame: int = 0

        for det in detections:

            # ── Emergency vehicle ────────────────────────────────────────
            if det.cls_name in self._emergency_classes:
                self._handle_emergency(det, result)
                continue

            # ── Pedestrian (PED ROI only) ────────────────────────────────
            if det.cls_name == 'person':
                if self._point_in_roi('PED', det.cx, det.cy):
                    ped_count_this_frame = ped_count_this_frame + 1  # pyre-ignore
                continue

            # ── Hazard animal ────────────────────────────────────────────
            if det.cls_name in self._hazard_classes:
                self._handle_hazard(det)

        # ── Decay emergency hold-off ─────────────────────────────────────
        if self._emergency_holdoff > 0:
            self._emergency_holdoff -= 1
            if self._emergency_holdoff == 0:
                logger.info(
                    "Emergency hold-off expired — arm %s cleared",
                    self._active_emergency_arm,
                )
                self._active_emergency_arm   = None
                self._active_emergency_class = None

        # Propagate persisted emergency to result if still active
        if self._active_emergency_arm is not None:
            result.emergency_detected = True
            result.emergency_arm      = self._active_emergency_arm
            result.emergency_class    = self._active_emergency_class

        # ── Pedestrian rolling average ───────────────────────────────────
        self._ped_buffer.append(ped_count_this_frame)
        rolling_avg = (
            float(np.mean(self._ped_buffer)) if self._ped_buffer else 0.0
        )
        result.ped_count       = ped_count_this_frame
        result.ped_rolling_avg = rolling_avg

        now = time.time()
        cooldown_elapsed = (now - self._last_ped_trigger) >= self.PED_COOLDOWN_SECONDS
        if rolling_avg >= self._ped_threshold and cooldown_elapsed:
            result.ped_phase_triggered = True
            self._last_ped_trigger     = now
            logger.info(
                "Pedestrian phase triggered: rolling_avg=%.1f  "
                "(cooldown will suppress for %.0fs)",
                rolling_avg,
                self.PED_COOLDOWN_SECONDS,
            )

        # ── Hazard active arms ───────────────────────────────────────────
        for arm in self._arm_names:
            if self._hazard_countdown.get(arm, 0) > 0:
                cls = self._hazard_class.get(arm, 'unknown')
                result.hazard_arms[arm] = cls
                # Tick down frames where no new hazard was seen
                if arm not in [
                    a for a in result.hazard_arms
                    if self._hazard_countdown.get(a, 0) == self._hazard_clear_n
                ]:
                    self._hazard_countdown[arm] -= 1

        return result

    def clear_emergency(self) -> None:
        """
        Explicitly release the emergency lock.
        Called by the controller after the emergency green cycle finishes.
        """
        logger.info(
            "Emergency cleared by controller (was arm=%s class=%s)",
            self._active_emergency_arm,
            self._active_emergency_class,
        )
        self._active_emergency_arm   = None
        self._active_emergency_class = None
        self._emergency_holdoff      = 0

    def simulate_emergency(self, arm: str, cls: str = 'ambulance') -> None:
        """
        Inject a simulated emergency for testing (keyboard shortcut E in Pygame).

        Args:
            arm: Arm name e.g. 'North'.
            cls: Emergency class label e.g. 'ambulance'.
        """
        logger.warning("SIMULATED EMERGENCY: %s on %s arm", cls, arm)
        self._active_emergency_arm   = arm
        self._active_emergency_class = cls
        self._emergency_holdoff      = EMERGENCY_HOLD * 20  # ~30s at 20 FPS

    def simulate_ped_rush(self) -> None:
        """
        Inject a pedestrian rush for testing (keyboard shortcut P in Pygame).
        Fills the entire rolling buffer above threshold.
        """
        logger.info("SIMULATED PEDESTRIAN RUSH")
        for _ in range(self._ped_buffer.maxlen or PED_ROLLING_FRAMES):
            self._ped_buffer.append(self._ped_threshold + 5)
        self._last_ped_trigger = 0.0   # reset cooldown so it fires immediately

    def reset(self) -> None:
        """Clear all state — call on video source change."""
        self._emergency_holdoff      = 0
        self._active_emergency_arm   = None
        self._active_emergency_class = None
        self._ped_buffer.clear()
        self._last_ped_trigger       = 0.0
        self._hazard_countdown       = {arm: 0 for arm in self._arm_names}
        self._hazard_class.clear()
        logger.info("EmergencyDetector state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_emergency(self, det, result: EmergencyResult) -> None:
        """Process a single emergency vehicle detection."""
        # Already holding — don't re-trigger from a different frame
        if self._emergency_holdoff > 0:
            return

        arm = self._centroid_to_arm(det.cx, det.cy)

        # Detection centroid outside all ROIs (e.g. in intersection box)
        # Fall back to nearest arm by distance
        if arm is None:
            arm = self._nearest_arm(det.cx, det.cy)

        if arm is None:
            logger.warning(
                "Emergency %s detected but couldn't assign to any arm "
                "(cx=%.0f cy=%.0f)",
                det.cls_name, det.cx, det.cy,
            )
            return

        logger.warning(
            "EMERGENCY OVERRIDE: %s detected on %s arm (conf=%.2f)",
            det.cls_name, arm, det.confidence,
        )

        self._active_emergency_arm   = arm
        self._active_emergency_class = det.cls_name
        self._emergency_holdoff      = self.HOLD_FRAMES

        result.emergency_detected = True
        result.emergency_arm      = arm
        result.emergency_class    = det.cls_name
        result.emergency_conf     = det.confidence

    def _handle_hazard(self, det) -> None:
        """Process a single hazard (animal) detection."""
        arm = self._centroid_to_arm(det.cx, det.cy)
        if arm is None:
            return

        prev_count = self._hazard_countdown.get(arm, 0)
        self._hazard_countdown[arm] = self._hazard_clear_n
        self._hazard_class[arm]     = det.cls_name

        if prev_count == 0:
            logger.warning(
                "HAZARD: %s detected on %s arm (conf=%.2f)",
                det.cls_name, arm, det.confidence,
            )

    def _point_in_roi(self, roi_name: str, cx: float, cy: float) -> bool:
        """Return True if (cx, cy) is inside the named ROI polygon."""
        polygon = self._rois.get(roi_name)
        if polygon is None:
            return False
        return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0.0

    def _centroid_to_arm(self, cx: float, cy: float) -> Optional[str]:
        """Return the first arm ROI that contains (cx, cy), or None."""
        for arm in self._arm_names:
            if self._point_in_roi(arm, cx, cy):
                return arm
        return None

    def _nearest_arm(self, cx: float, cy: float) -> Optional[str]:
        """
        Fallback: return the arm whose ROI centroid is closest to (cx, cy).
        Used when an emergency vehicle is detected in the intersection box
        rather than in any approach ROI.
        """
        best_arm:  Optional[str]  = None
        best_dist: float          = float('inf')

        for arm in self._arm_names:
            polygon = self._rois.get(arm)
            if polygon is None:
                continue
            rcx = float(np.mean(polygon[:, 0]))
            rcy = float(np.mean(polygon[:, 1]))
            dist = (cx - rcx) ** 2 + (cy - rcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_arm  = arm

        return best_arm


# ---------------------------------------------------------------------------
# Standalone test  (python -m detection.emergency)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    from config import VIDEO_SOURCE
    from utils.preprocessing import preprocess
    from utils.drawing import (
        draw_rois, draw_detections, draw_frame_info,
        draw_emergency_banner, draw_pedestrian_banner, draw_hazard_banner,
    )
    from detection.detector import TrafficDetector

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    detector   = TrafficDetector()
    emerg_det  = EmergencyDetector()
    cap        = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print(
        "Press Q to quit\n"
        "Press E to simulate emergency (North arm)\n"
        "Press P to simulate pedestrian rush\n"
    )

    import time
    frame_n = 0
    t0      = time.time()
    fps     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            emerg_det.reset()
            continue

        processed  = preprocess(frame)
        detections = detector.detect(processed)
        result     = emerg_det.update(detections)

        frame_n += 1  # pyre-ignore
        if frame_n % 30 == 0:
            fps = 30.0 / (time.time() - t0)  # pyre-ignore
            t0  = time.time()

        # Console log for any events
        if result.emergency_detected:
            print(
                f"[{frame_n}] 🚨 EMERGENCY: {result.emergency_class} "
                f"on {result.emergency_arm} arm  conf={result.emergency_conf:.2f}"
            )
        if result.ped_phase_triggered:
            print(
                f"[{frame_n}] 🚶 PED PHASE: avg={result.ped_rolling_avg:.1f}"
            )
        if result.hazard_arms:
            print(f"[{frame_n}] ⚠ HAZARD: {result.hazard_arms}")

        # Visual
        out = draw_rois(processed)
        out = draw_detections(out, detections)

        if result.emergency_detected and result.emergency_arm:
            out = draw_emergency_banner(out, result.emergency_arm)
        elif result.ped_phase_triggered:
            out = draw_pedestrian_banner(out, result.ped_rolling_avg)
        elif result.hazard_arms:
            arm, cls = next(iter(result.hazard_arms.items()))
            out = draw_hazard_banner(out, arm, cls)

        # Ped counter overlay
        ped_label = (
            f"PED ROI: {result.ped_count}  "
            f"avg={result.ped_rolling_avg:.1f}/{emerg_det._ped_threshold}"
        )
        cv2.putText(out, ped_label, (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out = draw_frame_info(out, frame_n, fps)
        cv2.imshow('Emergency Detector Test', out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('e'):
            emerg_det.simulate_emergency('North')
            print("[KEY] Simulated emergency on North arm")
        elif key == ord('p'):
            emerg_det.simulate_ped_rush()
            print("[KEY] Simulated pedestrian rush")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")