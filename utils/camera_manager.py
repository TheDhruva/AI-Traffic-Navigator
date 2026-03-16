# utils/camera_manager.py — Multi-Arm Camera Feed Manager
# ==========================================================
# Manages one VideoCapture per intersection arm (N/S/E/W).
# Each arm runs in its own daemon thread so a stalled camera
# never blocks the detection pipeline.
#
# Supported source types per arm:
#   int            → USB webcam index          e.g. 0, 1, 2, 3
#   str (file)     → pre-recorded video file   e.g. "assets/north.mp4"
#   str (RTSP/HTTP)→ IP camera stream          e.g. "rtsp://192.168.1.10:554/stream"
#
# Design rules:
#   • CAP_PROP_BUFFERSIZE = 1  → always serve the newest frame, never queue
#   • Video files loop automatically (seek to frame 0 on EOF)
#   • Failed cameras retry after RETRY_DELAY_S seconds, not immediately
#   • CLAHE preprocessing applied before frames are stored (optional, default ON)
#   • One RLock guards the shared frame dict — reads never block each other for long
#   • Health metrics (FPS, brightness, sharpness, status) exposed for the dashboard

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union

import cv2
import numpy as np

from config import ARM_NAMES, FRAME_WIDTH, FRAME_HEIGHT
from utils.preprocessing import preprocess, check_frame_quality

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ArmSource = Union[int, str]   # webcam index OR file/URL path


class FeedStatus(Enum):
    """Lifecycle state of a single camera feed."""
    STARTING   = auto()   # thread launched, not yet reading
    ACTIVE     = auto()   # frames arriving normally
    DEGRADED   = auto()   # frames arriving but quality check failing
    RETRYING   = auto()   # VideoCapture failed; waiting to retry
    STOPPED    = auto()   # CameraManager.stop() called


# ---------------------------------------------------------------------------
# Per-arm health snapshot (dashboard-safe — no locks needed after copy)
# ---------------------------------------------------------------------------

@dataclass
class FeedHealth:
    """Point-in-time health metrics for one camera arm."""

    arm:          str
    status:       FeedStatus = FeedStatus.STARTING
    fps:          float      = 0.0          # measured frames per second
    brightness:   float      = 0.0          # mean luminance [0, 255]
    sharpness:    float      = 0.0          # Laplacian variance
    is_usable:    bool       = False        # True when frame passes quality gate
    frames_read:  int        = 0            # total frames successfully decoded
    errors:       int        = 0            # consecutive read errors
    source:       str        = ""           # human-readable source description

    def as_dict(self) -> dict:
        return {
            "arm":        self.arm,
            "status":     self.status.name,
            "fps":        round(self.fps, 1),
            "brightness": round(self.brightness, 1),
            "sharpness":  round(self.sharpness, 1),
            "is_usable":  self.is_usable,
            "frames_read":self.frames_read,
            "errors":     self.errors,
            "source":     self.source,
        }


# ---------------------------------------------------------------------------
# Internal per-arm reader — runs in a daemon thread
# ---------------------------------------------------------------------------

# How long to wait before retrying a failed VideoCapture (seconds)
_RETRY_DELAY_S: float = 3.0

# Max consecutive read errors before treating stream as failed
_MAX_ERRORS: int = 30

# How often (in frames) to recompute measured FPS
_FPS_WINDOW_FRAMES: int = 30


class _ArmReader:
    """
    One thread per arm.  Opens VideoCapture, reads frames in a tight loop,
    applies optional CLAHE preprocessing, and stores the latest frame in
    a shared dict guarded by the parent CameraManager's lock.
    """

    def __init__(
        self,
        arm:       str,
        source:    ArmSource,
        frames:    dict[str, Optional[np.ndarray]],
        health:    dict[str, FeedHealth],
        lock:      threading.RLock,
        running:   threading.Event,
        apply_clahe: bool,
    ) -> None:
        self._arm         = arm
        self._source      = source
        self._frames      = frames
        self._health      = health
        self._lock        = lock
        self._running     = running
        self._apply_clahe = apply_clahe

        # Human-readable source label for logs / dashboard
        self._src_label = str(source) if isinstance(source, str) else f"webcam:{source}"

        self._thread = threading.Thread(
            target=self._loop,
            name=f"CamReader-{arm}",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        logger.debug("[%s] reader thread started ← %s", self._arm, self._src_label)

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main reader loop.  Runs until CameraManager.stop() sets _running."""

        self._update_health(status=FeedStatus.STARTING, source=self._src_label)

        while self._running.is_set():
            cap = self._open_capture()

            if cap is None:
                # _open_capture already logged; wait before retry
                self._update_health(status=FeedStatus.RETRYING)
                self._sleep_interruptible(_RETRY_DELAY_S)
                continue

            logger.info("[%s] VideoCapture open ← %s", self._arm, self._src_label)
            self._update_health(status=FeedStatus.ACTIVE)

            consecutive_errors = 0
            fps_frame_count    = 0
            fps_t0             = time.perf_counter()

            while self._running.is_set():
                ret, raw = cap.read()

                if not ret:
                    consecutive_errors += 1

                    # Video file EOF → loop from the beginning
                    if self._is_file_source():
                        logger.debug("[%s] EOF — looping video", self._arm)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_errors = 0
                        continue

                    # Stream / webcam read failure
                    if consecutive_errors >= _MAX_ERRORS:
                        logger.warning(
                            "[%s] %d consecutive read errors — reopening capture",
                            self._arm, consecutive_errors,
                        )
                        self._update_health(status=FeedStatus.RETRYING,
                                            errors=consecutive_errors)
                        break   # exit inner loop → outer loop reopens cap

                    time.sleep(0.01)
                    continue

                # ── Successful read ───────────────────────────────────────
                consecutive_errors = 0

                # Resize first (cheaper than CLAHE on full resolution)
                frame = cv2.resize(
                    raw, (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Optional CLAHE — same pipeline as preprocessing.py
                if self._apply_clahe:
                    frame = preprocess(frame)   # includes CLAHE + resize

                # Quality check
                quality = check_frame_quality(frame)

                # Store frame + update metrics under lock
                fps_frame_count += 1
                measured_fps = 0.0
                if fps_frame_count >= _FPS_WINDOW_FRAMES:
                    elapsed     = time.perf_counter() - fps_t0
                    measured_fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_frame_count = 0
                    fps_t0 = time.perf_counter()

                with self._lock:
                    self._frames[self._arm] = frame
                    h = self._health[self._arm]
                    h.frames_read  += 1
                    h.errors        = 0
                    h.brightness    = quality["brightness"]
                    h.sharpness     = quality["sharpness"]
                    h.is_usable     = quality["is_usable"]
                    h.status        = (
                        FeedStatus.ACTIVE if quality["is_usable"]
                        else FeedStatus.DEGRADED
                    )
                    if measured_fps > 0:
                        h.fps = measured_fps

            cap.release()
            logger.debug("[%s] VideoCapture released", self._arm)

        self._update_health(status=FeedStatus.STOPPED)
        logger.debug("[%s] reader thread exiting", self._arm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open VideoCapture for this arm's source.  Returns None on failure."""
        try:
            cap = cv2.VideoCapture(self._source)
        except Exception as exc:
            logger.error("[%s] VideoCapture() raised: %s", self._arm, exc)
            return None

        if not cap.isOpened():
            logger.warning("[%s] Cannot open source: %s", self._arm, self._src_label)
            cap.release()
            return None

        # Keep only the single newest frame in the OS buffer.
        # Critical for live cameras: prevents queue buildup that causes latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Request target resolution (VideoCapture may ignore this for video files)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        return cap

    def _is_file_source(self) -> bool:
        """True when source is a local file path (not webcam int or RTSP URL)."""
        if isinstance(self._source, int):
            return False
        s = str(self._source).lower()
        return not (s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://"))

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in small increments so _running.is_set() can cut it short."""
        deadline = time.perf_counter() + seconds
        while self._running.is_set() and time.perf_counter() < deadline:
            time.sleep(0.1)

    def _update_health(self, **kwargs) -> None:
        """Thread-safe health field update."""
        with self._lock:
            h = self._health[self._arm]
            for k, v in kwargs.items():
                setattr(h, k, v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Default video source used when the caller does not specify a per-arm source.
# Falls back to webcam 0 if the file doesn't exist, matching main.py behaviour.
_DEFAULT_SOURCES: dict[str, ArmSource] = {
    "North": "assets/north.mp4",
    "South": "assets/south.mp4",
    "East":  "assets/east.mp4",
    "West":  "assets/west.mp4",
}


class CameraManager:
    """
    Manages four simultaneous camera feeds — one per intersection arm.

    Each arm runs in a background daemon thread.  The main thread (or
    detection thread) calls get_frame(arm) which returns a preprocessed
    BGR frame instantly with no blocking.

    Typical usage::

        # Four separate video files (demo mode)
        cam = CameraManager({
            "North": "assets/north.mp4",
            "South": "assets/south.mp4",
            "East":  "assets/east.mp4",
            "West":  "assets/west.mp4",
        })
        cam.start()

        # Or: single source replicated to all arms (existing main.py style)
        cam = CameraManager.from_single_source("assets/test_video.mp4")
        cam.start()

        frame = cam.get_frame("North")   # None until first frame decoded
        health = cam.get_health("North") # FeedHealth dataclass

        cam.stop()

    Args:
        sources:     Dict mapping ARM_NAME → ArmSource (int / str).
                     Missing arms fall back to the default video files.
        apply_clahe: Apply CLAHE preprocessing before storing frames.
                     Set False if the detection thread calls preprocess()
                     itself to avoid double-processing.
    """

    def __init__(
        self,
        sources:     Optional[dict[str, ArmSource]] = None,
        apply_clahe: bool = True,
    ) -> None:
        # Resolve sources: explicit dict overrides defaults
        self._sources: dict[str, ArmSource] = {}
        for arm in ARM_NAMES:
            if sources and arm in sources:
                self._sources[arm] = sources[arm]
            else:
                self._sources[arm] = _DEFAULT_SOURCES.get(arm, 0)

        self._apply_clahe = apply_clahe

        # Shared state — accessed by all reader threads
        self._lock:   threading.RLock                    = threading.RLock()
        self._frames: dict[str, Optional[np.ndarray]]   = {a: None for a in ARM_NAMES}
        self._health: dict[str, FeedHealth]              = {
            a: FeedHealth(arm=a, source=str(self._sources[a]))
            for a in ARM_NAMES
        }

        # Signals all reader threads to keep running
        self._running: threading.Event = threading.Event()

        # One reader per arm
        self._readers: dict[str, _ArmReader] = {
            arm: _ArmReader(
                arm=arm,
                source=self._sources[arm],
                frames=self._frames,
                health=self._health,
                lock=self._lock,
                running=self._running,
                apply_clahe=apply_clahe,
            )
            for arm in ARM_NAMES
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "CameraManager":
        """
        Start all reader threads.  Returns self so you can chain::

            cam = CameraManager(...).start()
        """
        if self._running.is_set():
            logger.warning("CameraManager.start() called when already running — ignored")
            return self

        logger.info("CameraManager starting — %d arms", len(ARM_NAMES))
        self._running.set()

        for arm, reader in self._readers.items():
            logger.info("  %-6s ← %s", arm, self._sources[arm])
            reader.start()

        return self

    def stop(self) -> None:
        """
        Signal all reader threads to exit and wait up to 3 s for them to finish.
        Releases all VideoCapture resources.
        """
        if not self._running.is_set():
            return

        logger.info("CameraManager stopping...")
        self._running.clear()   # signals every _ArmReader._loop to exit

        for arm, reader in self._readers.items():
            reader.join(timeout=3.0)
            logger.debug("[%s] reader joined", arm)

        logger.info("CameraManager stopped")

    def __enter__(self) -> "CameraManager":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, arm: str) -> Optional[np.ndarray]:
        """
        Return the latest preprocessed BGR frame for *arm*.

        Returns None if no frame has arrived yet (first ~0.1 s after start).
        The returned array is a copy — safe to modify without lock.

        Args:
            arm: One of ARM_NAMES e.g. "North", "South", "East", "West".
        """
        if arm not in self._frames:
            logger.warning("get_frame() called with unknown arm: %s", arm)
            return None

        with self._lock:
            f = self._frames[arm]
            return f.copy() if f is not None else None

    def get_all_frames(self) -> dict[str, Optional[np.ndarray]]:
        """
        Return copies of all four arm frames in one lock acquisition.
        More efficient than calling get_frame() four times in a tight loop.

        Returns:
            Dict mapping arm name → BGR frame (or None if not yet available).
        """
        with self._lock:
            return {
                arm: (f.copy() if f is not None else None)
                for arm, f in self._frames.items()
            }

    def wait_for_first_frame(self, arm: str, timeout: float = 10.0) -> bool:
        """
        Block until the first frame for *arm* is available or timeout expires.

        Useful in main.py after cam.start() to avoid the 'no frame yet' flash
        in the Pygame window.

        Returns:
            True if frame arrived within timeout, False otherwise.
        """
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frames[arm] is not None:
                    return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # Health & diagnostics
    # ------------------------------------------------------------------

    def get_health(self, arm: str) -> FeedHealth:
        """
        Return a point-in-time copy of health metrics for *arm*.
        Safe to read from any thread.
        """
        with self._lock:
            h = self._health[arm]
            # Return a shallow copy so caller can't mutate shared state
            return FeedHealth(
                arm=h.arm, status=h.status, fps=h.fps,
                brightness=h.brightness, sharpness=h.sharpness,
                is_usable=h.is_usable, frames_read=h.frames_read,
                errors=h.errors, source=h.source,
            )

    def get_all_health(self) -> dict[str, FeedHealth]:
        """Return health snapshots for all arms in one lock acquisition."""
        with self._lock:
            return {
                arm: FeedHealth(
                    arm=h.arm, status=h.status, fps=h.fps,
                    brightness=h.brightness, sharpness=h.sharpness,
                    is_usable=h.is_usable, frames_read=h.frames_read,
                    errors=h.errors, source=h.source,
                )
                for arm, h in self._health.items()
            }

    def all_active(self) -> bool:
        """True when all four arms are in ACTIVE or DEGRADED state."""
        with self._lock:
            return all(
                h.status in (FeedStatus.ACTIVE, FeedStatus.DEGRADED)
                for h in self._health.values()
            )

    def health_summary(self) -> str:
        """One-line string for status printer in main.py."""
        with self._lock:
            parts = []
            for arm in ARM_NAMES:
                h  = self._health[arm]
                ok = "✓" if h.is_usable else "✗"
                parts.append(f"{arm[0]}:{ok}{h.fps:.0f}fps")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_single_source(
        cls,
        source:      ArmSource,
        apply_clahe: bool = True,
    ) -> "CameraManager":
        """
        Replicate one source across all four arms.

        Matches the existing single-camera workflow in main.py::

            cam = CameraManager.from_single_source("assets/test_video.mp4")
            cam = CameraManager.from_single_source(0)   # webcam

        Each arm gets its own independent VideoCapture so they can be at
        different playback positions (useful for a 4-arm demo using one file).
        """
        return cls(
            sources={arm: source for arm in ARM_NAMES},
            apply_clahe=apply_clahe,
        )

    @classmethod
    def from_config(
        cls,
        north: ArmSource = "assets/north.mp4",
        south: ArmSource = "assets/south.mp4",
        east:  ArmSource = "assets/east.mp4",
        west:  ArmSource = "assets/west.mp4",
        apply_clahe: bool = True,
    ) -> "CameraManager":
        """
        Explicit per-arm keyword constructor — easiest for main.py argparse::

            cam = CameraManager.from_config(
                north="assets/north.mp4",
                south=1,                          # USB webcam index 1
                east="rtsp://192.168.1.12/live",  # IP camera
                west="assets/west.mp4",
            )
        """
        return cls(
            sources={
                "North": north,
                "South": south,
                "East":  east,
                "West":  west,
            },
            apply_clahe=apply_clahe,
        )


# ---------------------------------------------------------------------------
# Standalone test  (python -m utils.camera_manager)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Build source map from command-line args or fall back to defaults ──
    # Usage:  python -m utils.camera_manager [source_for_all_arms]
    #   e.g.  python -m utils.camera_manager assets/test_video.mp4
    #   e.g.  python -m utils.camera_manager 0

    if len(sys.argv) >= 2:
        raw_src = sys.argv[1]
        try:
            src: ArmSource = int(raw_src)
        except ValueError:
            src = raw_src
        cam = CameraManager.from_single_source(src)
    else:
        cam = CameraManager()   # uses default file paths

    cam.start()

    # ── Wait for at least one arm to deliver a frame ──────────────────────
    print("\nWaiting for first frames (up to 10 s)...")
    ready = cam.wait_for_first_frame("North", timeout=10.0)
    if not ready:
        print("No frame from North arm — check source path. Exiting.")
        cam.stop()
        sys.exit(1)

    print("Frames arriving. Press Q to quit.\n")

    # ── Display 2×2 grid of all four arm feeds ────────────────────────────
    TILE_W, TILE_H = 640, 360
    BORDER = 4
    LABEL_H = 28

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_sc   = 0.55
    thickness = 1

    try:
        while True:
            frames = cam.get_all_frames()
            healths = cam.get_all_health()

            tiles = []
            for arm in ARM_NAMES:
                f = frames.get(arm)
                h = healths.get(arm)

                if f is None:
                    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                    cv2.putText(tile, f"{arm}: no frame yet", (10, TILE_H // 2),
                                font, 0.8, (100, 100, 100), 2)
                else:
                    tile = cv2.resize(f, (TILE_W, TILE_H))

                # Status bar at top of tile
                bar_colour = (0, 180, 0) if h and h.is_usable else (0, 60, 180)
                cv2.rectangle(tile, (0, 0), (TILE_W, LABEL_H), bar_colour, -1)

                status_txt = (
                    f"{arm}  FPS:{h.fps:.0f}  "
                    f"Bright:{h.brightness:.0f}  "
                    f"Sharp:{h.sharpness:.0f}  "
                    f"{h.status.name}"
                ) if h else f"{arm}: initialising"

                cv2.putText(tile, status_txt, (6, 18),
                            font, font_sc, (255, 255, 255), thickness)

                # Border
                cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1),
                              (60, 60, 60), BORDER)
                tiles.append(tile)

            # Stack 2 rows × 2 columns
            top_row    = np.hstack(tiles[:2])
            bottom_row = np.hstack(tiles[2:])
            grid       = np.vstack([top_row, bottom_row])

            # Health summary at bottom
            summary_bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(summary_bar, cam.health_summary(), (10, 20),
                        font, 0.55, (200, 200, 200), 1)
            display = np.vstack([grid, summary_bar])

            cv2.imshow("CameraManager — 4-Arm Feed Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("\nStopped.")