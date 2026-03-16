# config.py — Smart Traffic System Configuration
# All constants, thresholds, ROI coordinates, and hardware settings live here.
# Change values here; never hardcode them elsewhere.

import numpy as np

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------
MODEL_PATH       = 'yolov8n.pt'   # auto-downloads ~6MB on first run
CONF_THRESHOLD   = 0.35           # low — needed for occluded two-wheelers
IOU_THRESHOLD    = 0.40           # low NMS — allows dense cluster detections
INPUT_SIZE       = 640            # YOLO inference resolution (square)

# ---------------------------------------------------------------------------
# SIGNAL TIMING (seconds)
# ---------------------------------------------------------------------------
MIN_GREEN          = 10    # minimum green duration
MAX_GREEN          = 60    # maximum green duration
YELLOW_DURATION    = 3     # always fixed
ALL_RED_BUFFER     = 1     # safety buffer between every phase change
PED_WALK_DURATION  = 15    # pedestrian WALK phase
EMERGENCY_HOLD     = 30    # emergency vehicle green hold
HAZARD_EXTENSION   = 5     # extend green when animal detected on road

# ---------------------------------------------------------------------------
# DETECTION THRESHOLDS
# ---------------------------------------------------------------------------
PED_THRESHOLD       = 8    # persons in crosswalk ROI to trigger ped phase
PED_ROLLING_FRAMES  = 15   # rolling average window (frames)
HAZARD_CLEAR_FRAMES = 5    # consecutive frames with no hazard = cleared
STARVATION_THRESHOLD = 120 # seconds before a starved arm gets a hard boost

# ---------------------------------------------------------------------------
# SCORING WEIGHTS
# ---------------------------------------------------------------------------
DENSITY_WEIGHT   = 2.0
WAIT_WEIGHT      = 10.0
FLOW_WEIGHT      = 5.0
STARVATION_BOOST = 50.0           # score bonus after starvation threshold
EMERGENCY_SCORE  = float('inf')   # always wins

# ---------------------------------------------------------------------------
# PCU WEIGHTS  (Passenger Car Unit — vehicle space equivalence)
# ---------------------------------------------------------------------------
PCU_WEIGHTS: dict[str, float] = {
    'car':        1.0,
    'truck':      3.0,
    'bus':        3.0,
    'motorcycle': 0.4,
    'bicycle':    0.3,
    'auto':       0.8,   # autorickshaw (not a COCO class — mapped via alias)
    'person':     0.0,   # counted separately for pedestrian phase
    'unknown':    0.5,   # fallback for unrecognised detections
}

# Classes that map to 'auto' (autorickshaw) in Indian traffic
# YOLOv8 COCO doesn't have 'auto' — nearest visual matches listed here
AUTO_ALIAS_CLASSES: list[str] = []  # extend if you fine-tune the model

# COCO class names that trigger emergency override
EMERGENCY_CLASSES: list[str] = ['ambulance', 'fire truck', 'fire_truck']

# COCO class names that trigger hazard (animal on road)
HAZARD_CLASSES: list[str] = ['dog', 'cow', 'horse', 'elephant', 'cat']

# ---------------------------------------------------------------------------
# VIDEO INPUT
# ---------------------------------------------------------------------------
VIDEO_SOURCE  = 'assets/test_video.mp4'  # 0 = webcam, or path to .mp4
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT   = 2.0          # CLAHE contrast clip limit
CLAHE_TILE_GRID    = (8, 8)       # CLAHE tile grid size

# ---------------------------------------------------------------------------
# OPTICAL FLOW
# ---------------------------------------------------------------------------
FLOW_MIN_MAGNITUDE   = 1.5        # pixels/frame below = traffic stopped
FLOW_MAX_FEATURES    = 200        # max Shi-Tomasi features per ROI
FLOW_QUALITY_LEVEL   = 0.3
FLOW_MIN_DISTANCE    = 7

# ---------------------------------------------------------------------------
# SERIAL / ARDUINO
# ---------------------------------------------------------------------------
SERIAL_PORT    = 'COM3'           # Windows. Linux: '/dev/ttyACM0'
SERIAL_BAUD    = 9600
SERIAL_TIMEOUT = 1                # seconds
ARDUINO_BOOT_DELAY = 2            # seconds — wait for Arduino reset on connect

# ---------------------------------------------------------------------------
# ROI COORDINATES  (for 640×480 input frame, 4-way intersection)
# Polygons: clockwise from top-left corner.
# Adjust with utils/roi_calibrator.py before a live demo.
# ---------------------------------------------------------------------------
ROIS: dict[str, np.ndarray] = {
    'North': np.array([[160,  20], [480,  20], [480, 180], [160, 180]], dtype=np.int32),
    'South': np.array([[160, 300], [480, 300], [480, 460], [160, 460]], dtype=np.int32),
    'East':  np.array([[460, 160], [630, 160], [630, 320], [460, 320]], dtype=np.int32),
    'West':  np.array([[10,  160], [180, 160], [180, 320], [10,  320]], dtype=np.int32),
    'PED':   np.array([[160, 180], [480, 180], [480, 300], [160, 300]], dtype=np.int32),
}

# Arm order for score iteration (controls tie-breaking by index)
ARM_NAMES: list[str] = ['North', 'South', 'East', 'West']

# ---------------------------------------------------------------------------
# ROI DISPLAY COLOURS  (BGR for OpenCV)
# ---------------------------------------------------------------------------
ROI_COLORS: dict[str, tuple[int, int, int]] = {
    'North': (0,   255, 100),   # green
    'South': (0,   180, 255),   # orange
    'East':  (255, 100,   0),   # blue
    'West':  (180,   0, 255),   # purple
    'PED':   (255, 255,   0),   # cyan
}

ROI_ALPHA = 0.25   # overlay transparency (0 = invisible, 1 = solid)

# ---------------------------------------------------------------------------
# PYGAME SIMULATION
# ---------------------------------------------------------------------------
SIM_WIDTH  = 800
SIM_HEIGHT = 800
SIM_FPS    = 30

# Signal light colours (RGB)
SIM_COLOR_GREEN  = (0,   220,  80)
SIM_COLOR_YELLOW = (255, 200,   0)
SIM_COLOR_RED    = (220,  30,  30)
SIM_COLOR_OFF    = (40,   40,  40)

# Vehicle colours by class (RGB)
VEHICLE_COLORS: dict[str, tuple[int, int, int]] = {
    'car':        (100, 160, 220),
    'bus':        (220, 120,  30),
    'motorcycle': (160, 220, 100),
    'truck':      (180,  80,  80),
    'auto':       (220, 180,  60),
    'bicycle':    (200, 200,  80),
}

# Vehicle spawn weights per arm (fraction, must sum ≤ 1.0)
VEHICLE_SPAWN_WEIGHTS: dict[str, float] = {
    'motorcycle': 0.40,
    'car':        0.30,
    'auto':       0.15,
    'bus':        0.10,
    'truck':      0.05,
}

# ---------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------
DASHBOARD_REFRESH_MS  = 500    # milliseconds between dashboard refreshes
DASHBOARD_MAX_HISTORY = 100    # data points kept for time-series charts

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = 'INFO'   # DEBUG | INFO | WARNING | ERROR
LOG_FILE  = 'traffic_system.log'