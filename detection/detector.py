# detection/detector.py — YOLOv8 Inference Wrapper
# Loads the model once, exposes a clean detect() interface.
# Returns typed Detection objects so every downstream module is decoupled from ultralytics internals.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    INPUT_SIZE,
    PCU_WEIGHTS,
    EMERGENCY_CLASSES,
    HAZARD_CLASSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract — every module downstream uses this, never raw ultralytics
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection result from one YOLO inference pass."""

    x1: float                       # bounding box left   (pixels, original frame coords)
    y1: float                       # bounding box top
    x2: float                       # bounding box right
    y2: float                       # bounding box bottom
    confidence: float               # model confidence [0, 1]
    class_id: int                   # COCO numeric class id
    class_name: str                 # human-readable label e.g. 'car', 'person'

    # Derived fields — computed on construction
    cx: float = field(init=False)   # centroid x
    cy: float = field(init=False)   # centroid y
    pcu: float = field(init=False)  # PCU weight for this class
    is_emergency: bool = field(init=False)
    is_hazard: bool = field(init=False)

    def __post_init__(self) -> None:
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        self.pcu = PCU_WEIGHTS.get(self.class_name, PCU_WEIGHTS['unknown'])
        self.is_emergency = self.class_name in EMERGENCY_CLASSES
        self.is_hazard = self.class_name in HAZARD_CLASSES

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name} conf={self.confidence:.2f} "
            f"cx={self.cx:.0f} cy={self.cy:.0f} pcu={self.pcu})"
        )


# ---------------------------------------------------------------------------
# Detector — singleton-style; instantiate once in main.py and share the ref
# ---------------------------------------------------------------------------

class TrafficDetector:
    """
    Wraps YOLOv8 inference for traffic scene analysis.

    Usage:
        detector = TrafficDetector()
        detections = detector.detect(frame)   # list[Detection]
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        input_size: int = INPUT_SIZE,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path or name of YOLO weights ('yolov8n.pt' auto-downloads).
            conf:       Confidence threshold. 0.35 keeps occluded two-wheelers.
            iou:        NMS IoU threshold. 0.4 allows dense cluster detections.
            input_size: Inference resolution. Must be multiple of 32.
            device:     'cpu', 'cuda', 'mps', or None (auto-detect).
        """
        self.conf = conf
        self.iou = iou
        self.input_size = input_size

        logger.info("Loading YOLO model: %s", model_path)
        self._model = YOLO(model_path)

        # Auto-select device: CUDA > MPS > CPU
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        logger.info("Using device: %s", self.device)

        # Warm up the model with a blank frame so first real frame isn't slow
        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame (as returned by OpenCV).

        Args:
            frame: H×W×3 uint8 BGR numpy array.

        Returns:
            List of Detection objects. Empty list if no detections or on error.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame — skipping")
            return []

        try:
            results = self._model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.input_size,
                device=self.device,
                verbose=False,       # silence per-frame ultralytics logs
                stream=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        return self._parse_results(results, frame.shape)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.
        Useful if you want to pre-buffer frames and infer together.

        Returns:
            List of detection lists, one per input frame.
        """
        if not frames:
            return []

        try:
            results = self._model(
                frames,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.input_size,
                device=self.device,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            logger.error("YOLO batch inference failed: %s", exc)
            return [[] for _ in frames]

        return [self._parse_results([r], frames[i].shape) for i, r in enumerate(results)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        """Return the model's full COCO class id → name mapping."""
        return self._model.names  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(
        self, results: list, frame_shape: tuple[int, ...]
    ) -> list[Detection]:
        """Convert ultralytics Results objects to our Detection dataclass list."""
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            # All tensors → numpy for consistent downstream handling
            xyxy_all   = boxes.xyxy.cpu().numpy()    # (N, 4) float32
            conf_all   = boxes.conf.cpu().numpy()    # (N,)   float32
            cls_all    = boxes.cls.cpu().numpy().astype(int)  # (N,) int

            for xyxy, conf, cls_id in zip(xyxy_all, conf_all, cls_all):
                class_name = self._model.names.get(cls_id, 'unknown')

                # Clamp coordinates to frame bounds
                h, w = frame_shape[:2]
                x1 = float(np.clip(xyxy[0], 0, w))
                y1 = float(np.clip(xyxy[1], 0, h))
                x2 = float(np.clip(xyxy[2], 0, w))
                y2 = float(np.clip(xyxy[3], 0, h))

                # Skip degenerate boxes (can appear at frame edges)
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                )

        return detections

    def _warmup(self) -> None:
        """Run a single blank-frame inference to initialise CUDA/model graph."""
        try:
            blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._model(blank, imgsz=self.input_size, device=self.device, verbose=False)
            logger.info("YOLO warmup complete")
        except Exception as exc:
            # Non-fatal — warmup is a nice-to-have
            logger.warning("YOLO warmup failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Quick standalone test  (python -m detection.detector)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import cv2
    from config import VIDEO_SOURCE

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    detector = TrafficDetector()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", VIDEO_SOURCE)
        sys.exit(1)

    print("\nPress Q to quit | Running YOLO inference...\n")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
            continue

        detections = detector.detect(frame)
        frame_count += 1

        # Draw bounding boxes manually (drawing.py does this properly later)
        for d in detections:
            colour = (0, 255, 0) if not d.is_emergency else (0, 0, 255)
            if d.is_hazard:
                colour = (0, 165, 255)
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), colour, 2)
            label = f"{d.class_name} {d.confidence:.2f} PCU={d.pcu}"
            cv2.putText(frame, label, (int(d.x1), int(d.y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        info = f"Frame {frame_count} | Detections: {len(detections)}"
        cv2.putText(frame, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow('Detector Test', frame)

        if frame_count % 30 == 0:
            classes = [d.class_name for d in detections]
            print(f"[Frame {frame_count}] {len(detections)} detections: {classes}")

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")