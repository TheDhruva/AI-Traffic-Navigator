# utils/preprocessing.py — Frame Preprocessing Pipeline
# Applied to every frame BEFORE YOLO inference.
# CLAHE is mandatory for Indian road conditions: handles harsh noon glare,
# underpass shadows, dawn/dusk low-light, and oncoming headlight flare.

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    INPUT_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level CLAHE object — create once, reuse every frame (cv2 recommends this)
# ---------------------------------------------------------------------------

_clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID,
)


# ---------------------------------------------------------------------------
# Core pipeline — called by the detection thread on every raw frame
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for a single BGR frame.

    Steps:
        1. Validate input
        2. Resize to FRAME_WIDTH × FRAME_HEIGHT (keeps aspect if square)
        3. Apply CLAHE on the L channel (LAB colour space)
        4. Return BGR frame ready for YOLO

    Args:
        frame: Raw BGR uint8 frame from OpenCV VideoCapture.

    Returns:
        Preprocessed BGR uint8 frame.  Same HWC shape as after resize.
        Returns the original frame unchanged on any error (safe fallback).
    """
    if frame is None or frame.size == 0:
        logger.warning("preprocess() received empty frame — returning as-is")
        return frame

    try:
        out = resize_frame(frame)
        out = apply_clahe(out)
        return out
    except Exception as exc:
        logger.error("Preprocessing failed (%s) — using raw frame", exc)
        return frame


# ---------------------------------------------------------------------------
# Individual steps (exported so other modules can call them independently)
# ---------------------------------------------------------------------------

def resize_frame(
    frame: np.ndarray,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize frame to (width, height).

    Uses INTER_LINEAR by default — fastest interpolation with acceptable quality.
    Switch to INTER_AREA when downscaling significantly (avoids moiré).

    Args:
        frame:         Input BGR frame.
        width:         Target width in pixels.
        height:        Target height in pixels.
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized BGR frame.
    """
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame  # already correct size — skip copy

    # Auto-select better interpolation when downscaling more than 2×
    if interpolation == cv2.INTER_LINEAR and (w > 2 * width or h > 2 * height):
        interpolation = cv2.INTER_AREA

    return cv2.resize(frame, (width, height), interpolation=interpolation)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    L channel of the LAB colour space.

    Why LAB and not HSV?
        LAB's L channel is a pure perceptual luminance measure.
        Equalising only L preserves hue and saturation, so vehicle colours
        remain correct for any future colour-based classification.
        HSV's V channel mixes luminance with saturation artefacts.

    Args:
        frame: BGR uint8 frame.

    Returns:
        Contrast-enhanced BGR uint8 frame.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_eq = _clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def prepare_for_yolo(frame: np.ndarray) -> np.ndarray:
    """
    Resize a frame to INPUT_SIZE × INPUT_SIZE specifically for YOLO.
    Separate from resize_frame so ROI coordinates stay at FRAME resolution,
    while YOLO always gets its fixed square input.

    Note: ultralytics handles this internally when you pass the full frame,
    so you only need this if you are manually cropping before inference.

    Args:
        frame: BGR frame at any resolution.

    Returns:
        BGR frame resized to (INPUT_SIZE, INPUT_SIZE).
    """
    return resize_frame(frame, width=INPUT_SIZE, height=INPUT_SIZE)


def denoise(
    frame: np.ndarray,
    h: float = 6.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Optional fast non-local means denoising.

    Useful for night footage with high ISO noise.
    Not called by default (too slow for 20 FPS) — enable in main.py
    if running on a GPU or if frame rate isn't critical.

    Args:
        frame:           BGR uint8 frame.
        h:               Filter strength (higher = more blurring).
        template_window: Odd number, size of template patch.
        search_window:   Odd number, size of search area.

    Returns:
        Denoised BGR uint8 frame.
    """
    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


def adjust_brightness_contrast(
    frame: np.ndarray,
    alpha: float = 1.0,
    beta: int = 0,
) -> np.ndarray:
    """
    Linear brightness/contrast adjustment: output = alpha * frame + beta.

    Useful for extreme underexposure where CLAHE alone isn't enough.
    alpha > 1 increases contrast; beta > 0 increases brightness.

    Args:
        frame: BGR uint8 frame.
        alpha: Contrast multiplier [0.5–3.0 typical].
        beta:  Brightness addend  [-100–100 typical].

    Returns:
        Adjusted BGR uint8 frame (values clamped to [0, 255]).
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def check_frame_quality(frame: np.ndarray) -> dict[str, float]:
    """
    Compute basic frame quality metrics.

    Used to detect camera occlusion, lens cover, or hardware failure.
    If brightness and sharpness both drop near zero, fall back to fixed-timer mode.

    Returns:
        Dict with keys:
          'brightness' — mean luminance [0, 255]
          'sharpness'  — Laplacian variance (higher = sharper, >100 = usable)
          'is_usable'  — True if frame appears valid for inference
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness: float = float(np.mean(gray))
    sharpness: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_usable: bool = brightness > 10.0 and sharpness > 20.0

    return {
        'brightness': brightness,
        'sharpness': sharpness,
        'is_usable': is_usable,
    }


def stack_debug_frames(
    raw: np.ndarray,
    processed: np.ndarray,
    label_raw: str = "RAW",
    label_proc: str = "CLAHE",
) -> np.ndarray:
    """
    Horizontally stack raw and preprocessed frames with labels.
    Useful during calibration to visually verify CLAHE is helping.

    Args:
        raw:        Original BGR frame.
        processed:  Preprocessed BGR frame (same resolution as raw).
        label_raw:  Text drawn on left panel.
        label_proc: Text drawn on right panel.

    Returns:
        Single wide BGR frame: [raw | processed].
    """
    # Ensure both frames are the same height for hstack
    h = max(raw.shape[0], processed.shape[0])
    w = raw.shape[1]

    def _pad(f: np.ndarray) -> np.ndarray:
        if f.shape[0] < h:
            pad = np.zeros((h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            return np.vstack([f, pad])
        return f

    left  = _pad(raw.copy())
    right = _pad(processed.copy())

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    colour     = (255, 255, 255)

    cv2.putText(left,  label_raw,  (10, 28), font, font_scale, (0, 0, 0),   thickness + 2)
    cv2.putText(left,  label_raw,  (10, 28), font, font_scale, colour,      thickness)
    cv2.putText(right, label_proc, (10, 28), font, font_scale, (0, 0, 0),   thickness + 2)
    cv2.putText(right, label_proc, (10, 28), font, font_scale, colour,      thickness)

    return np.hstack([left, right])


# ---------------------------------------------------------------------------
# Standalone test  (python -m utils.preprocessing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from config import VIDEO_SOURCE

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print("Press Q to quit | Side-by-side: RAW (left) vs CLAHE (right)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        raw_resized = resize_frame(frame)
        processed   = preprocess(frame)
        quality     = check_frame_quality(processed)

        comparison = stack_debug_frames(raw_resized, processed)

        status = (
            f"Brightness: {quality['brightness']:.1f}  "
            f"Sharpness: {quality['sharpness']:.1f}  "
            f"Usable: {quality['is_usable']}"
        )
        cv2.putText(comparison, status, (10, comparison.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow('Preprocessing Test', comparison)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")