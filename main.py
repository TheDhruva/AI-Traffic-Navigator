#!/usr/bin/env python3
# main.py — AI Smart Traffic System: Entry Point (Upgraded with CameraManager)
# ==============================================================================
# Launches detection + controller + Pygame threads with CameraManager:
#
#   CameraManager (internal)       — 1 daemon thread per arm (N/S/E/W)
#                                   Manages per-arm VideoCapture + CLAHE + health
#   Thread 1 (detection)           — Read from CameraManager → detect → density →
#                                    flow → emergency → IntersectionState
#   Thread 2 (signal controller)   — Read state → score → send Arduino commands
#   Main thread (Pygame)           — Display sim + camera feeds + state
#
# Multi-arm camera features:
#   • Each arm: independent source (webcam, file, RTSP, IP camera)
#   • Per-arm health metrics: FPS, brightness, sharpness, status
#   • Automatic retry on failure (non-blocking)
#   • Optional CLAHE preprocessing
#
# Usage:
#   python main.py --demo                      # single file → 4 arms
#   python main.py --webcam                    # webcam 0 → 4 arms
#   python main.py --source path.mp4           # custom file → 4 arms
#   python main.py --north n.mp4 --south s.mp4 --east 1 --west rtsp://...
#   python main.py --port /dev/ttyACM0         # Arduino port
#   python main.py --no-sim                    # headless
#   python main.py --no-clahe                  # disable CLAHE in camera mgr
#   python main.py --debug                     # verbose logging

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
from config import (
    VIDEO_SOURCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SERIAL_PORT,
    LOG_LEVEL,
    LOG_FILE,
    ARM_NAMES,
)
from utils.camera_manager import CameraManager, FeedStatus
from controller.algorithm import SignalController
from controller.state import IntersectionState, create_state
from detection.density import DensityEstimator
from detection.detector import TrafficDetector
from detection.emergency import EmergencyDetector
from detection.flow import FlowAnalyser
from hardware.arduino import ArduinoController, create_arduino
from utils.drawing import (
    draw_debug_overlay,
    draw_emergency_banner,
    draw_pedestrian_banner,
    draw_hazard_banner,
    draw_frame_info,
)
from utils.preprocessing import preprocess, check_frame_quality
from dashboard.app import DashboardBridge

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(level: str = LOG_LEVEL, log_file: str = LOG_FILE) -> None:
    """Configure logging to console and file."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    except OSError:
        pass
    logging.basicConfig(
        level=numeric,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Thread 1 — Detection Pipeline
# ════════════════════════════════════════════════════════════════════════════

class DetectionThread(threading.Thread):
    """
    Reads frames from CameraManager (all 4 arms), runs full detection pipeline,
    writes results to shared IntersectionState every frame.

    Pipeline per frame (per arm):
        CameraManager.get_frame(arm)
        → quality check
        → TrafficDetector.detect() [YOLOv8]
        → DensityEstimator.update() [PCU per arm]
        → FlowAnalyser.update() [optical flow per arm]
        → EmergencyDetector.update() [emergency + ped + hazard]
        → IntersectionState write (with lock)
        → annotated frame → state.set_annotated_frame()
    """

    def __init__(
        self,
        state: IntersectionState,
        camera_mgr: CameraManager,
        show_overlay: bool = True,
    ) -> None:
        super().__init__(name="DetectionThread", daemon=True)
        self._state        = state
        self._camera_mgr   = camera_mgr
        self._show_overlay = show_overlay

        # Pipeline components — built lazily in run()
        self._detector:  Optional[TrafficDetector]  = None
        self._density:   Optional[DensityEstimator] = None
        self._flow:      Optional[FlowAnalyser]     = None
        self._emergency: Optional[EmergencyDetector] = None

        # Stats
        self._frame_count:   int   = 0
        self._inference_ms:  float = 0.0
        self._fps:           float = 0.0
        self._fps_timer:     float = time.time()

    @property
    def emergency_detector(self) -> Optional[EmergencyDetector]:
        return self._emergency

    def run(self) -> None:
        logger.info("Detection thread starting (reading from CameraManager)")

        # ── Initialise pipeline ───────────────────────────────────────────
        try:
            self._detector  = TrafficDetector()
            self._density   = DensityEstimator()
            self._flow      = FlowAnalyser()
            self._emergency = EmergencyDetector()
        except Exception as exc:
            logger.critical("Failed to initialise detection pipeline: %s", exc)
            with self._state.lock:
                self._state.running = False
            return

        logger.info("Detection pipeline initialized")

        # ── Wait for at least one camera to deliver a frame ────────────────
        ready = any(
                self._camera_mgr.wait_for_first_frame(arm, timeout=15.0)
                for arm in ["North","South","East","West"]
            )
        if not ready:
            logger.critical("No frames from cameras after 15s — aborting")
            with self._state.lock:
                self._state.running = False
            return

        logger.info("Camera feeds ready — starting detection loop")

        # ── Main detection loop ───────────────────────────────────────────
        while True:
            with self._state.lock:
                if not self._state.running:
                    break

            self._frame_count += 1

            # Get the latest composite frame from CameraManager
            # (CameraManager.get_all_frames() returns a dict of arm → frame)
            frames = self._camera_mgr.get_all_frames()

            # Check health of at least one arm — if all are unusable, pause
            healths = self._camera_mgr.get_all_health()
            any_usable = any(h.is_usable for h in healths.values())
            if not any_usable:
                logger.debug("All camera feeds degraded — skipping frame %d", self._frame_count)
                time.sleep(0.1)
                continue

            # For now, use North arm as the primary frame for display.
            # In a full system, you could stitch all 4 or process each independently.
            for arm, frame in frames.items():

                if frame is None:
                    continue

                quality = check_frame_quality(frame)
                if not quality['is_usable']:
                    continue

                processed = frame

                t0 = time.perf_counter()
                detections = self._detector.detect(processed)
                self._inference_ms = (time.perf_counter() - t0) * 1000

                density_result = self._density.update(detections)
                flow_result = self._flow.update(processed)
                emrg_result = self._emergency.update(detections)

                with self._state.lock:
                    self._state.update_from_density(density_result)
                    self._state.update_from_flow(flow_result)
                    self._state.update_from_emergency(emrg_result)

            # ── Quality gate ───────────────────────────────────────────────
            quality = check_frame_quality(frame)
            if not quality['is_usable']:
                logger.debug(
                    "Frame %d degraded (bright=%.0f sharp=%.0f) — skipping",
                    self._frame_count, quality['brightness'], quality['sharpness'],
                )
                time.sleep(0.01)
                continue

            # ── Preprocessing (may be redundant if CameraManager applies CLAHE) ───
            # Only apply if not already done in camera_manager
            processed = preprocess(frame)

            # ── YOLO inference ────────────────────────────────────────────
            t0 = time.perf_counter()
            detections = self._detector.detect(processed)
            self._inference_ms = (time.perf_counter() - t0) * 1000

            # ── Density estimation ────────────────────────────────────────
            density_result = self._density.update(detections)

            # ── Optical flow ──────────────────────────────────────────────
            flow_result = self._flow.update(processed)

            # ── Emergency / pedestrian detection ──────────────────────────
            emrg_result = self._emergency.update(detections)

            # ── Write to shared state ─────────────────────────────────────
            with self._state.lock:
                self._state.update_from_density(density_result)
                self._state.update_from_flow(flow_result)
                self._state.update_from_emergency(emrg_result)

            # ── Annotate frame ────────────────────────────────────────────
            if self._show_overlay:
                annotated = self._annotate(processed, detections, emrg_result)
                with self._state.lock:
                    self._state.set_annotated_frame(annotated)

            # ── FPS tracking ──────────────────────────────────────────────
            if self._frame_count % 30 == 0:
                elapsed = time.time() - self._fps_timer
                self._fps = 30.0 / max(elapsed, 0.001)
                self._fps_timer = time.time()
                logger.debug(
                    "Detection: frame=%d fps=%.1f infer=%.0fms dets=%d",
                    self._frame_count, self._fps,
                    self._inference_ms, len(detections),
                )

        logger.info("Detection thread stopped (frames=%d)", self._frame_count)

    def _annotate(self, frame: np.ndarray, detections, emrg_result) -> np.ndarray:
        """Build the annotated frame shown in Pygame camera panel."""
        arm_snap_raw = self._state.snapshot_arms()

        arm_snap = {
            arm: {
                "signal": s.signal,
                "density": s.density,
                "wait_time": s.wait_time,
                "emergency": s.emergency,
                "hazard": s.hazard,
            }
            for arm, s in arm_snap_raw.items()
        }

        out = draw_debug_overlay(
            frame, detections, arm_snap,
            show_centroids=True,
            show_roi_labels=True,
        )

        # Alert banners (top strip)
        if emrg_result.emergency_detected and emrg_result.emergency_arm:
            out = draw_emergency_banner(out, emrg_result.emergency_arm)
        elif emrg_result.ped_phase_triggered:
            out = draw_pedestrian_banner(out, emrg_result.ped_rolling_avg)
        elif emrg_result.hazard_arms:
            arm, cls = next(iter(emrg_result.hazard_arms.items()))
            out = draw_hazard_banner(out, arm, cls)

        out = draw_frame_info(out, self._frame_count, self._fps, self._inference_ms)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Smart Traffic Optimization System — Indian Cities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo              # use test_video.mp4 for all 4 arms
  python main.py --webcam            # use webcam index 0 for all 4 arms
  python main.py --source vid.mp4    # custom video file for all 4 arms
  python main.py --north n.mp4 --south s.mp4 --east 1 --west rtsp://...
  python main.py --port /dev/ttyACM0 # Linux Arduino port
  python main.py --no-sim            # headless / server mode
  python main.py --no-clahe          # disable CLAHE in camera manager
  python main.py --debug             # verbose logging + debug overlay
        """,
    )
    p.add_argument('--source', type=str, default=None,
                   help='Video source (file/webcam/RTSP) — replicate to all 4 arms')
    p.add_argument('--webcam', action='store_true',
                   help='Use webcam index 0 for all 4 arms')
    p.add_argument('--demo', action='store_true',
                   help='Use default test video for all 4 arms')
    p.add_argument('--north', type=str, default=None,
                   help='North arm source (file/int/RTSP)')
    p.add_argument('--south', type=str, default=None,
                   help='South arm source')
    p.add_argument('--east', type=str, default=None,
                   help='East arm source')
    p.add_argument('--west', type=str, default=None,
                   help='West arm source')
    p.add_argument('--port', type=str, default=SERIAL_PORT,
                   help=f'Serial port for Arduino (default: {SERIAL_PORT})')
    p.add_argument('--no-arduino', action='store_true',
                   help='Skip Arduino connection (simulation mode)')
    p.add_argument('--no-sim', action='store_true',
                   help='Headless mode — no Pygame window')
    p.add_argument('--no-clahe', action='store_true',
                   help='Disable CLAHE preprocessing in CameraManager')
    p.add_argument('--debug', action='store_true',
                   help='Enable DEBUG logging + debug overlay')
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Signal handlers
# ════════════════════════════════════════════════════════════════════════════

def _setup_signal_handlers(state: IntersectionState) -> None:
    """Register SIGINT / SIGTERM handlers for clean shutdown."""
    def _handler(signum, frame):
        logger.info("Shutdown signal received (%s) — stopping...", signum)
        with state.lock:
            state.running = False

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# ════════════════════════════════════════════════════════════════════════════
# Status printer
# ════════════════════════════════════════════════════════════════════════════

def _status_printer(state: IntersectionState, camera_mgr: Optional[CameraManager] = None) -> None:
    """Print one-line status every 5s with camera health info."""
    while state.running:
        with state.lock:
            running = state.running

        if not running:
            break

        status_line = state.summary_string()

        # Append camera health summary if available
        if camera_mgr:
            health_summary = camera_mgr.health_summary()
            status_line = f"{status_line} | {health_summary}"

        print(f"\r{status_line}", end='', flush=True)
        time.sleep(5.0)

    print()  # final newline


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── Logging ──────────────────────────────────────────────────────────
    level = 'DEBUG' if args.debug else LOG_LEVEL
    _setup_logging(level)
    logger.info("=" * 60)
    logger.info("AI Smart Traffic System — Starting up")
    logger.info("=" * 60)

    # ── Resolve camera sources ───────────────────────────────────────────
    # Priority: explicit per-arm > single source replicated to all 4
    if args.north or args.south or args.east or args.west:
        # Per-arm source specified
        north_src = args.north or VIDEO_SOURCE
        south_src = args.south or VIDEO_SOURCE
        east_src  = args.east  or VIDEO_SOURCE
        west_src  = args.west  or VIDEO_SOURCE

        logger.info("Using per-arm camera sources")
        camera_mgr = CameraManager.from_config(
            north=_parse_source(north_src),
            south=_parse_source(south_src),
            east=_parse_source(east_src),
            west=_parse_source(west_src),
            apply_clahe=not args.no_clahe,
        )
    else:
        # Single source → replicate to all 4 arms
        if args.webcam:
            source = 0
        elif args.source:
            source = args.source
        elif args.demo:
            source = VIDEO_SOURCE
        else:
            source = VIDEO_SOURCE

        # Validate file source
        if isinstance(source, str) and not source.isdigit() and not source.startswith("rtsp://"):
            if not os.path.exists(source):
                logger.warning("Source not found: %s — falling back to webcam 0", source)
                source = 0

        logger.info("Using single source for all 4 arms: %s", source)
        camera_mgr = CameraManager.from_single_source(
            source=_parse_source(source),
            apply_clahe=not args.no_clahe,
        )

    # Start CameraManager (spawns daemon threads for each arm)
    camera_mgr.start()
    logger.info("CameraManager started (4 reader threads)")

    # ── Shared state ──────────────────────────────────────────────────────
    state = create_state()
    _setup_signal_handlers(state)

    # ── Arduino ───────────────────────────────────────────────────────────
    if args.no_arduino or args.demo:
        logger.info("Arduino: skipped (simulation mode)")
        arduino = create_arduino(auto_connect=False)
    else:
        arduino = create_arduino(port=args.port)
        if arduino.is_simulation:
            logger.info("Arduino: not found — simulation mode")
        else:
            logger.info("Arduino: connected on %s", arduino._port)

    # ── Signal controller (Thread 2) ───────────────────────────────────────
    controller = SignalController(
        state=state,
        send_command=arduino.get_send_callback(),
    )
    controller.start()
    logger.info("Signal controller started")

    # ── Detection thread (Thread 1) ───────────────────────────────────────
    detection_thread = DetectionThread(
        state=state,
        camera_mgr=camera_mgr,
        show_overlay=not args.no_sim,
    )
    detection_thread.start()
    logger.info("Detection thread started")

    # ── Status printer (daemon) ───────────────────────────────────────────
    status_thread = threading.Thread(
        target=_status_printer,
        args=(state, camera_mgr),
        name="StatusPrinter",
        daemon=True,
    )
    status_thread.start()

    # ── Dashboard bridge ──────────────────────────────────────────────────
    while detection_thread.emergency_detector is None:
        time.sleep(0.1)

    dashboard_bridge = DashboardBridge(
        state,
        emergency_detector=detection_thread.emergency_detector
    )
    dashboard_bridge.start()
    logger.info("DashboardBridge started")

    # ── Main thread: Pygame simulation ────────────────────────────────────
    if args.no_sim:
        logger.info("Headless mode — no Pygame window. Press Ctrl+C to stop.")
        _headless_wait(state)
    else:
        try:
            from simulation.pygame_sim import create_simulation
        except ImportError as exc:
            logger.error("Pygame unavailable (%s) — headless mode", exc)
            _headless_wait(state)
        else:
            sim = create_simulation(
                state=state,
                emergency_detector=detection_thread.emergency_detector,
            )
            logger.info("Pygame simulation starting (main thread)")
            try:
                sim.run()
            except Exception as exc:
                logger.error("Pygame crashed: %s", exc, exc_info=True)

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Initiating shutdown...")

    with state.lock:
        state.running = False

    time.sleep(0.5)

    controller.stop()
    controller.join(timeout=5.0)
    logger.info("Controller stopped")

    arduino.send_all_red()
    time.sleep(0.2)
    arduino.disconnect()
    logger.info("Arduino disconnected")

    detection_thread.join(timeout=5.0)
    logger.info("Detection thread stopped")

    camera_mgr.stop()
    logger.info("CameraManager stopped")

    # Final stats
    phase_snap = state.snapshot_phase()
    logger.info(
        "Session summary: cycles=%d cleared=%d uptime=%.0fs",
        phase_snap.get('total_cycles', 0),
        phase_snap.get('vehicles_cleared', 0),
        phase_snap.get('uptime_s', 0),
    )
    logger.info("Shutdown complete.")


def _headless_wait(state: IntersectionState) -> None:
    """Block main thread in headless mode until shutdown."""
    try:
        while True:
            with state.lock:
                if not state.running:
                    break
            time.sleep(0.5)
    except KeyboardInterrupt:
        with state.lock:
            state.running = False


def _parse_source(src) -> int | str:
    """
    Convert string representation to int (webcam index) or str (file/RTSP).
    E.g. "0" → 0, "path.mp4" → "path.mp4", "rtsp://..." → "rtsp://..."
    """
    if isinstance(src, int):
        return src
    if isinstance(src, str):
        try:
            return int(src)
        except ValueError:
            return src
    return src


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()