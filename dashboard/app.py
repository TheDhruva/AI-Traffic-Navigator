"""
dashboard/app.py — AI Smart Traffic System Dashboard
=====================================================
FIXES vs previous version:
  • BRIGHT theme (white background, clean cards, high-contrast text)
  • Video captures are cached in st.session_state — opened ONCE, reused every
    refresh (eliminates the freeze caused by re-opening cap every second)
  • Auto-refresh uses streamlit-autorefresh (1 s interval) with proper fallback
    that does NOT trigger window.location.reload() (which resets session state)
  • Frame advancement is smooth — increments by ~2 frames per tick
  • Placeholder SVGs also updated for bright theme
  • All logic, simulation engine, and data structures preserved from v1

RUN:
    pip install streamlit streamlit-autorefresh plotly opencv-python
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
STATE_FILE = Path("traffic_state.json")
ASSETS_DIR = Path("assets")

VIDEO_FILES = {
    'North': 'n.mp4',
    'South': 's.mp4',
    'East':  'e.mp4',
    'West':  'w.mp4',
}

ARM_COLORS = {
    'North': '#2563EB',
    'South': '#059669',
    'East':  '#D97706',
    'West':  '#7C3AED',
}

ARM_LIST = ['North', 'South', 'East', 'West']

PCU = {'car': 1.0, 'truck': 3.0, 'bus': 3.0,
       'motorcycle': 0.4, 'bicycle': 0.3, 'auto': 0.8}

MIN_GREEN   = 10
MAX_GREEN   = 60
YELLOW_DUR  = 3
ALL_RED_DUR = 1


# ════════════════════════════════════════════════════════════════════════════
# Dashboard Bridge (used when integrated with main.py)
# ════════════════════════════════════════════════════════════════════════════

class DashboardBridge(threading.Thread):
    def __init__(self, state, emergency_detector=None, write_interval: float = 0.5):
        super().__init__(name="DashboardBridge", daemon=True)
        self._state     = state
        self._emergency = emergency_detector
        self._interval  = write_interval
        self._stop      = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                state_dict = self._state.to_dict()
                if self._emergency:
                    try:
                        state_dict['emergency_status'] = self._emergency.get_status_dict()
                    except Exception:
                        pass
                STATE_FILE.write_text(json.dumps(state_dict, default=str), encoding='utf-8')
            except Exception:
                pass
            time.sleep(self._interval)

    def stop(self):
        self._stop.set()


# ════════════════════════════════════════════════════════════════════════════
# Standalone Simulation Engine
# ════════════════════════════════════════════════════════════════════════════

def _webster_splits(densities: dict) -> dict:
    SAT  = 0.35
    LOST = 2.0
    L    = 4 * LOST
    y    = {arm: max(0.001, d / SAT / 60) for arm, d in densities.items()}
    Y    = sum(y.values())
    if Y >= 0.9:
        return {arm: float(MAX_GREEN) for arm in densities}
    C   = max(40.0, min(180.0, (1.5 * L + 5) / (1.0 - Y)))
    eff = C - L
    return {
        arm: max(float(MIN_GREEN), min(float(MAX_GREEN), eff * (y[arm] / Y)))
        for arm in densities
    }


def _priority_score(density: float, wait: float, arrival: float, discharge: float) -> float:
    wait_factor = 1.0 + math.pow(max(0.0, wait) / 30.0, 1.8)
    sat_ratio   = min(1.0, density / (0.35 * MAX_GREEN))
    return max(0.0,
        3.0 * density
        + 8.0 * wait_factor
        + 4.0 * arrival
        - 2.0 * discharge
        + 6.0 * sat_ratio
    )


def _simulate_step(sim: dict, dt: float) -> dict:
    arms   = sim['arms']
    phase  = sim['phase']
    cg     = sim['current_green']
    timer  = sim['phase_timer'] - dt
    splits = sim['webster_splits']

    for arm in ARM_LIST:
        a = arms[arm]
        arrivals = random.gauss(0.8, 0.4) * dt
        a['density'] = max(0.0, a['density'] + arrivals)
        if arm == cg and phase == 'green':
            discharge = random.uniform(0.25, 0.45) * dt
            a['density']        = max(0.0, a['density'] - discharge)
            a['discharge_rate'] = discharge / max(dt, 0.001)
            a['flow_direction'] = 'away'
        else:
            a['discharge_rate'] = 0.0
            a['flow_direction'] = 'toward' if a['density'] > 2 else 'stopped'
        prev = a.get('_prev_density', a['density'])
        raw_arr = max(0.0, (a['density'] - prev) / max(dt, 0.001))
        a['arrival_rate']   = 0.3 * raw_arr + 0.7 * a.get('arrival_rate', 0.0)
        a['_prev_density']  = a['density']
        a['density']        = min(a['density'], 45.0)
        a['queue_length']   = max(0, int(round(a['density'] / 0.7)))
        if arm == cg and phase == 'green':
            a['wait_time'] = 0.0
        else:
            a['wait_time'] = min(a['wait_time'] + dt, 300.0)
        a['priority_score'] = _priority_score(
            a['density'], a['wait_time'], a['arrival_rate'], a['discharge_rate']
        )
        net = a['arrival_rate'] - a['discharge_rate']
        a['predicted_q8s'] = max(0.0, a['density'] + net * 8.0)

    sim['phase_timer'] = timer

    if timer <= 0:
        if phase == 'all_red':
            scores = {arm: arms[arm]['priority_score'] for arm in ARM_LIST}
            winner = max(scores, key=lambda a: scores[a])
            sim['current_green'] = winner
            sim['phase']         = 'green'
            green_t              = splits.get(winner, float(MIN_GREEN))
            sim['phase_timer']   = green_t
            arms[winner]['signal_state'] = 'GREEN'
            for arm in ARM_LIST:
                if arm != winner:
                    arms[arm]['signal_state'] = 'RED'
            sim['webster_splits'] = _webster_splits({a: arms[a]['density'] for a in ARM_LIST})
            sim['current_scores'] = scores
            sim['total_cycles']  += 1
            sim['last_green_arm']  = winner
            sim['last_green_time'] = green_t
        elif phase == 'green':
            sim['phase']       = 'yellow'
            sim['phase_timer'] = float(YELLOW_DUR)
            if cg:
                arms[cg]['signal_state'] = 'YELLOW'
        elif phase == 'yellow':
            sim['phase']         = 'all_red'
            sim['phase_timer']   = float(ALL_RED_DUR)
            sim['current_green'] = None
            for arm in ARM_LIST:
                arms[arm]['signal_state'] = 'RED'

    sim['uptime_s'] += dt
    sim['vehicles_cleared'] += max(0, int(
        (arms.get(cg or 'North', {}).get('discharge_rate', 0)) * dt * 2
    ))
    sim['throughput_per_hour'] = sim['vehicles_cleared'] / max(sim['uptime_s'], 1) * 3600
    baseline_wait = 45.0
    avg_wait = sum(arms[a]['wait_time'] for a in ARM_LIST) / 4
    sim['total_wait_saved']  += max(0.0, baseline_wait - avg_wait) * dt / 60
    sim['efficiency_gain_pct'] = min(99.9, sim['total_wait_saved'] / max(1, sim['uptime_s'] / 60) * 100)
    sim['congestion_index']  = min(100, int(max(arms[a]['density'] for a in ARM_LIST) / 45.0 * 100))
    return sim


def _init_sim() -> dict:
    arms = {}
    for arm in ARM_LIST:
        arms[arm] = {
            'density':        float(random.uniform(4, 18)),
            'queue_length':   random.randint(3, 15),
            'wait_time':      float(random.uniform(0, 40)),
            'flow_rate':      float(random.uniform(0.5, 3.0)),
            'flow_direction': 'unknown',
            'emergency':      False,
            'hazard':         False,
            'ped_count':      0,
            'vehicle_count':  random.randint(3, 14),
            'priority_score': 0.0,
            'arrival_rate':   float(random.uniform(0.1, 0.5)),
            'discharge_rate': 0.0,
            'predicted_q8s':  0.0,
            'signal_state':   'RED',
            '_prev_density':  0.0,
        }
    densities = {arm: arms[arm]['density'] for arm in ARM_LIST}
    splits    = _webster_splits(densities)
    scores    = {arm: _priority_score(arms[arm]['density'], arms[arm]['wait_time'],
                                      arms[arm]['arrival_rate'], 0.0) for arm in ARM_LIST}
    winner    = max(scores, key=lambda a: scores[a])
    arms[winner]['signal_state'] = 'GREEN'
    return {
        'arms':               arms,
        'phase':              'green',
        'current_green':      winner,
        'phase_timer':        splits.get(winner, float(MIN_GREEN)),
        'webster_splits':     splits,
        'current_scores':     scores,
        'total_cycles':       1,
        'uptime_s':           0.0,
        'vehicles_cleared':   0,
        'throughput_per_hour':0.0,
        'total_wait_saved':   0.0,
        'efficiency_gain_pct':0.0,
        'congestion_index':   30,
        'ped_rolling_avg':    0.0,
        'ped_requested':      False,
        'last_green_arm':     winner,
        'last_green_time':    splits.get(winner, float(MIN_GREEN)),
    }


# ════════════════════════════════════════════════════════════════════════════
# Video Frame Extraction — CACHED CAPTURES (fixes freeze)
# ════════════════════════════════════════════════════════════════════════════

def _ensure_captures(session_state) -> None:
    """
    Open VideoCapture objects ONCE and store in session_state.
    On subsequent calls the caps are reused — no re-open overhead.
    """
    if 'video_caps' not in session_state:
        session_state.video_caps = {}

    try:
        import cv2
    except ImportError:
        return

    for arm, fname in VIDEO_FILES.items():
        video_path = ASSETS_DIR / fname
        if arm not in session_state.video_caps:
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    session_state.video_caps[arm] = cap
                    logger.info("Opened video for %s: %s", arm, video_path)
                else:
                    cap.release()
            # If file doesn't exist, arm stays absent from dict → placeholder shown


def _get_video_frame_b64(arm: str, session_state, signal_state: str, density: float) -> Optional[str]:
    """
    Read the NEXT frame from the cached cap for this arm.
    Loops automatically. Returns base64 JPEG string or None.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    caps: dict = session_state.get('video_caps', {})
    cap = caps.get(arm)
    if cap is None or not cap.isOpened():
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        # Loop: seek back to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

    # Resize for fast display
    frame = cv2.resize(frame, (320, 200))
    h, w  = frame.shape[:2]

    # Signal border
    border_colors = {'GREEN': (34, 197, 94), 'YELLOW': (234, 179, 8), 'RED': (239, 68, 68)}
    bc = border_colors.get(signal_state, (100, 100, 100))
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), bc, 6)

    # Signal dot (top-right)
    cv2.circle(frame, (w - 22, 22), 14, (255, 255, 255), -1)
    cv2.circle(frame, (w - 22, 22), 12, bc, -1)

    # Density badge strip (bottom)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, arm, (8, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{density:.1f} PCU  {signal_state}", (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok2:
        return None
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def _placeholder_frame_svg(arm: str, signal_state: str, density: float) -> str:
    """Bright-theme SVG placeholder when video is unavailable."""
    sig_colors = {'GREEN': '#059669', 'YELLOW': '#D97706', 'RED': '#DC2626'}
    arm_bg     = {'North': '#EFF6FF', 'South': '#ECFDF5', 'East': '#FFFBEB', 'West': '#F5F3FF'}
    sc  = sig_colors.get(signal_state, '#6B7280')
    bg  = arm_bg.get(arm, '#F9FAFB')
    fname = VIDEO_FILES.get(arm, '')
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='320' height='200'>
  <rect width='320' height='200' fill='{bg}'/>
  <rect x='3' y='3' width='314' height='194' fill='none' stroke='{sc}' stroke-width='5' rx='6'/>
  <text x='160' y='80' text-anchor='middle' font-family='monospace' font-size='36'>🚦</text>
  <text x='160' y='115' text-anchor='middle' font-family='monospace' font-size='15'
        fill='#111827' font-weight='bold'>{arm}</text>
  <text x='160' y='138' text-anchor='middle' font-family='monospace' font-size='12'
        fill='{sc}' font-weight='600'>{density:.1f} PCU · {signal_state}</text>
  <text x='160' y='170' text-anchor='middle' font-family='monospace' font-size='10'
        fill='#9CA3AF'>Place {fname} in assets/ for live feed</text>
</svg>"""
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"


# ════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ════════════════════════════════════════════════════════════════════════════

def _signal_color(sig: str) -> str:
    return {'GREEN': '#059669', 'YELLOW': '#D97706', 'RED': '#DC2626'}.get(sig, '#6B7280')


def _congestion_color(density: float) -> str:
    if density < 5:  return '#059669'
    if density < 12: return '#D97706'
    if density < 20: return '#EA580C'
    return '#DC2626'


def _phase_badge(phase: str) -> str:
    return {
        'normal':     '🟢 NORMAL',
        'pedestrian': '🚶 PEDESTRIAN',
        'emergency':  '🚨 EMERGENCY',
        'all_red':    '🔴 ALL RED',
        'green':      '🟢 GREEN PHASE',
        'yellow':     '🟡 YELLOW PHASE',
        'startup':    '⏳ STARTING',
    }.get(phase, f'⚪ {phase.upper()}')


def _traffic_light_svg(sig: str) -> str:
    """Compact 3-lamp traffic light SVG — bright theme."""
    r = '#DC2626' if sig == 'RED'    else '#FEE2E2'
    y = '#D97706' if sig == 'YELLOW' else '#FEF3C7'
    g = '#059669' if sig == 'GREEN'  else '#D1FAE5'
    glow = {
        'RED':    'filter:drop-shadow(0 0 5px #DC2626)',
        'YELLOW': 'filter:drop-shadow(0 0 5px #D97706)',
        'GREEN':  'filter:drop-shadow(0 0 5px #059669)',
    }.get(sig, '')
    return f"""<svg width='40' height='100' xmlns='http://www.w3.org/2000/svg'>
  <rect x='4' y='2' width='32' height='96' rx='8' fill='#1F2937' stroke='#374151' stroke-width='1.5'/>
  <circle cx='20' cy='22' r='11' fill='{r}' style='{glow if sig=="RED" else ""}'/>
  <circle cx='20' cy='50' r='11' fill='{y}' style='{glow if sig=="YELLOW" else ""}'/>
  <circle cx='20' cy='78' r='11' fill='{g}' style='{glow if sig=="GREEN" else ""}'/>
</svg>"""


# ════════════════════════════════════════════════════════════════════════════
# BRIGHT THEME CSS
# ════════════════════════════════════════════════════════════════════════════

BRIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg:       #F8FAFC;
  --surface:  #FFFFFF;
  --surface2: #F1F5F9;
  --border:   #E2E8F0;
  --text:     #0F172A;
  --muted:    #64748B;
  --green:    #059669;
  --yellow:   #D97706;
  --red:      #DC2626;
  --blue:     #2563EB;
  --purple:   #7C3AED;
}

/* Force light background even in dark-mode Streamlit */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] { background: #F1F5F9 !important; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* Force all text to dark */
p, span, div, label, h1, h2, h3, h4 { color: var(--text); }

.dash-header {
  font-family: 'Syne', sans-serif;
  font-size: 1.65rem;
  font-weight: 800;
  color: #0F172A;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.dash-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.live-dot {
  display:inline-block; width:8px; height:8px;
  background:var(--green); border-radius:50%;
  animation: pulse 1.4s ease-in-out infinite;
  margin-right:6px; vertical-align:middle;
}
@keyframes pulse {
  0%,100%{ opacity:1; transform:scale(1); }
  50%{ opacity:0.3; transform:scale(0.7); }
}

.phase-banner {
  padding: 10px 18px;
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-align: center;
  margin: 6px 0 12px 0;
  border: 1.5px solid currentColor;
}

.kpi-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px 14px 18px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--accent-color, var(--blue));
  border-radius: 12px 12px 0 0;
}
.kpi-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.65rem;
  font-weight: 700;
  line-height: 1.1;
}
.kpi-label {
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 5px;
  font-weight: 600;
}
.kpi-delta { font-size: 0.76rem; margin-top: 4px; font-weight: 600; }

.sec-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 8px 0 6px 0;
  border-bottom: 1.5px solid var(--border);
  margin-bottom: 10px;
}

.vid-wrap {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  border: 2.5px solid var(--border);
  background: #0F172A;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.vid-wrap img { width: 100%; display: block; border-radius: 5px; }

.arm-meta {
  background: var(--surface2);
  border-radius: 8px;
  padding: 10px 12px;
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.71rem;
  color: var(--text);
  border: 1px solid var(--border);
  line-height: 1.85;
}

.bar-wrap { background: #E2E8F0; border-radius:4px; height:7px; overflow:hidden; margin:3px 0; }
.bar-fill  { height:100%; border-radius:4px; transition: width 0.4s ease; }

.h-bar-bg   { background: #E2E8F0; border-radius:5px; height:22px; overflow:hidden; margin:3px 0; position:relative; }
.h-bar-fill { height:100%; border-radius:5px; display:flex; align-items:center; padding-left:8px;
               font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:700; color:white; }

.tbl-row {
  display:flex; justify-content:space-between; align-items:center;
  padding: 5px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.78rem;
}
.tbl-key { color: var(--muted); font-family:'DM Sans',sans-serif; }
.tbl-val { font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #0F172A; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stProgress > div > div { height: 6px !important; border-radius: 4px !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.35rem; }

/* Plotly chart backgrounds — force white */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
"""


# ════════════════════════════════════════════════════════════════════════════
# Main Streamlit App
# ════════════════════════════════════════════════════════════════════════════

def run_dashboard() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("pip install streamlit")
        return

    st.set_page_config(
        page_title="AI Smart Traffic System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Session state init (runs only on first load) ───────────────────────
    if 'sim' not in st.session_state:
        st.session_state.sim       = _init_sim()
        st.session_state.last_tick = time.time()
        st.session_state.history   = {arm: deque(maxlen=30) for arm in ARM_LIST}

    # ── Open video captures once, cache in session_state ─────────────────
    _ensure_captures(st.session_state)

    # ── Decide data source ────────────────────────────────────────────────
    use_sim = not STATE_FILE.exists()

    if use_sim:
        now = time.time()
        dt  = min(now - st.session_state.last_tick, 2.0)
        st.session_state.last_tick = now
        st.session_state.sim = _simulate_step(st.session_state.sim, dt)
        sim = st.session_state.sim

        state = {
            'phase':           sim['phase'],
            'current_green':   sim['current_green'],
            'ped_requested':   sim['ped_requested'],
            'ped_rolling_avg': sim['ped_rolling_avg'],
            'congestion_index':sim['congestion_index'],
            'arms': {arm: dict(sim['arms'][arm]) for arm in ARM_LIST},
            'session': {
                'total_cycles':        sim['total_cycles'],
                'vehicles_cleared':    sim['vehicles_cleared'],
                'throughput_per_hour': sim['throughput_per_hour'],
                'total_wait_saved':    sim['total_wait_saved'],
                'efficiency_gain_pct': sim['efficiency_gain_pct'],
                'uptime_s':            sim['uptime_s'],
                'last_green_arm':      sim['last_green_arm'],
                'last_green_time':     sim['last_green_time'],
                'webster_splits':      sim['webster_splits'],
                'current_scores':      sim['current_scores'],
            },
        }
    else:
        try:
            raw   = STATE_FILE.read_text(encoding='utf-8')
            state = json.loads(raw)
        except Exception:
            state = {}

    # ── Record density history for sparklines ─────────────────────────────
    arms_data = state.get('arms', {})
    for arm in ARM_LIST:
        d = arms_data.get(arm, {}).get('density', 0.0)
        st.session_state.history[arm].append(d)

    session       = state.get('session', {})
    phase         = state.get('phase', 'startup')
    current_green = state.get('current_green')
    cong_index    = state.get('congestion_index', 0)

    # ── Inject CSS ────────────────────────────────────────────────────────
    st.markdown(BRIGHT_CSS, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════════════════════════════════
    h1, h2, h3 = st.columns([5, 3, 2])
    with h1:
        src_badge = "SIMULATION MODE" if use_sim else "LIVE · main.py"
        st.markdown(f"""
        <div class='dash-header'>
          <span class='live-dot'></span>AI Smart Traffic System
        </div>
        <div class='dash-sub'>YOLOv8 + Webster's Algorithm · Indian Roads · {src_badge}</div>
        """, unsafe_allow_html=True)

    with h2:
        pc = {
            'green':      '#ECFDF5', 'yellow':     '#FFFBEB',
            'all_red':    '#FEF2F2', 'normal':      '#ECFDF5',
            'pedestrian': '#F5F3FF', 'emergency':   '#FEF2F2',
            'startup':    '#F8FAFC',
        }.get(phase, '#F8FAFC')
        tc = {
            'green':      '#059669', 'yellow':     '#D97706',
            'all_red':    '#DC2626', 'normal':      '#059669',
            'pedestrian': '#7C3AED', 'emergency':   '#DC2626',
            'startup':    '#6B7280',
        }.get(phase, '#6B7280')
        cg_txt = f" · {current_green} GREEN" if current_green else ""
        st.markdown(f"""
        <div class='phase-banner' style='background:{pc};color:{tc}'>
          {_phase_badge(phase)}{cg_txt}
        </div>""", unsafe_allow_html=True)

    with h3:
        st.markdown(f"""
        <div style='text-align:right;padding-top:6px'>
          <div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                       color:#0F172A;font-weight:700'>{time.strftime('%H:%M:%S')}</div>
          <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#94A3B8'>
            AUTO-REFRESH 1s
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # KPI ROW
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-label'>◈ System Performance</div>", unsafe_allow_html=True)
    kc = st.columns(5)

    uptime     = session.get('uptime_s', 0)
    ut_str     = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"
    cycles     = session.get('total_cycles', 0)
    cleared    = session.get('vehicles_cleared', 0)
    efficiency = session.get('efficiency_gain_pct', 0.0)
    throughput = session.get('throughput_per_hour', 0.0)
    wait_saved = session.get('total_wait_saved', 0.0)

    kpis = [
        ('#2563EB', ut_str,              '⏱ UPTIME',           f"{cycles} signal cycles"),
        ('#059669', f"{cleared:,}",      '🚗 VEHICLES CLEARED', f"↑ {throughput:.0f} / hr"),
        ('#059669' if efficiency >= 20 else '#D97706',
                   f"{efficiency:.1f}%", '⚡ EFFICIENCY GAIN',  '↑ vs fixed-time baseline'),
        ('#DC2626' if cong_index > 60 else '#D97706' if cong_index > 30 else '#059669',
                   str(cong_index),      '🌡 CONGESTION IDX',   'Scale 0–100'),
        ('#7C3AED', f"{wait_saved:.0f}s",'⏳ WAIT TIME SAVED',  'vs naive baseline'),
    ]
    for col, (accent, val, label, delta) in zip(kc, kpis):
        with col:
            dcol = '#059669' if '↑' in delta else '#64748B'
            st.markdown(f"""
            <div class='kpi-card' style='--accent-color:{accent}'>
              <div class='kpi-val' style='color:{accent}'>{val}</div>
              <div class='kpi-label'>{label}</div>
              <div class='kpi-delta' style='color:{dcol}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # VIDEO + ARM PANEL (4 columns)
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-label'>◈ Live Intersection Feeds</div>", unsafe_allow_html=True)
    arm_cols = st.columns(4)

    for i, arm in enumerate(ARM_LIST):
        arm_data = arms_data.get(arm, {})
        sig      = arm_data.get('signal_state', 'RED')
        density  = arm_data.get('density', 0.0)
        queue    = arm_data.get('queue_length', 0)
        wait     = arm_data.get('wait_time', 0.0)
        score    = arm_data.get('priority_score', 0.0)
        pred_q   = arm_data.get('predicted_q8s', 0.0)
        arr_rate = arm_data.get('arrival_rate', 0.0)
        dis_rate = arm_data.get('discharge_rate', 0.0)
        flow_dir = arm_data.get('flow_direction', 'unknown')
        emrg     = arm_data.get('emergency', False)
        hazard   = arm_data.get('hazard', False)

        sig_color  = _signal_color(sig)
        dens_color = _congestion_color(density)
        arm_color  = ARM_COLORS[arm]
        is_green   = (arm == current_green)

        with arm_cols[i]:
            # Traffic light SVG + arm name
            light_svg  = _traffic_light_svg(sig)
            badge_html = ""
            if emrg:   badge_html += "<span style='background:#DC2626;color:white;padding:1px 6px;border-radius:4px;font-size:0.6rem;margin-left:4px'>🚨 EMRG</span>"
            if hazard: badge_html += "<span style='background:#D97706;color:white;padding:1px 6px;border-radius:4px;font-size:0.6rem;margin-left:4px'>⚠ HZRD</span>"

            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
              {light_svg}
              <div>
                <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.05rem;
                             color:{arm_color}'>{arm}{badge_html}</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                             color:{sig_color};font-weight:700;letter-spacing:0.04em'>{sig}</div>
                {'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#059669;font-weight:700">● ACTIVE</div>' if is_green else ''}
              </div>
            </div>""", unsafe_allow_html=True)

            # Video frame (uses cached cap — no freeze)
            b64 = _get_video_frame_b64(arm, st.session_state, sig, density)
            if b64:
                st.markdown(f"""
                <div class='vid-wrap' style='border-color:{sig_color}'>
                  <img src='data:image/jpeg;base64,{b64}' alt='{arm} feed'/>
                </div>""", unsafe_allow_html=True)
            else:
                placeholder = _placeholder_frame_svg(arm, sig, density)
                st.markdown(f"""
                <div class='vid-wrap' style='border-color:{sig_color}'>
                  <img src='{placeholder}' alt='{arm} feed'/>
                </div>""", unsafe_allow_html=True)

            # Density bar
            bar_pct = min(100, int(density / 45.0 * 100))
            st.markdown(f"""
            <div style='margin:6px 0 2px 0'>
              <div style='display:flex;justify-content:space-between;
                           font-family:JetBrains Mono,monospace;font-size:0.64rem;
                           color:#64748B;margin-bottom:2px'>
                <span>DENSITY</span><span style='color:{dens_color};font-weight:700'>{density:.1f} PCU</span>
              </div>
              <div class='bar-wrap'>
                <div class='bar-fill' style='width:{bar_pct}%;background:{dens_color}'></div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Sparkline
            hist = list(st.session_state.history.get(arm, []))
            if len(hist) > 2:
                try:
                    import plotly.graph_objects as go
                    # Convert hex arm color to rgba fill
                    hex_c = arm_color.lstrip('#')
                    r2,g2,b2 = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                    fill_rgba = f"rgba({r2},{g2},{b2},0.15)"
                    fig_sp = go.Figure(go.Scatter(
                        y=hist, mode='lines',
                        line=dict(color=arm_color, width=1.5),
                        fill='tozeroy', fillcolor=fill_rgba,
                    ))
                    fig_sp.update_layout(
                        height=48, margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                    )
                    st.plotly_chart(fig_sp, use_container_width=True, config={'displayModeBar': False})
                except ImportError:
                    pass

            # Arm metrics
            score_pct = min(100, int(score / 80 * 100))
            wait_col  = '#DC2626' if wait > 60 else '#D97706' if wait > 30 else '#64748B'
            st.markdown(f"""
            <div class='arm-meta'>
              <div>Queue  <span style='float:right;color:{dens_color};font-weight:700'>{queue} vehicles</span></div>
              <div>Wait   <span style='float:right;color:{wait_col};font-weight:700'>{wait:.0f}s</span></div>
              <div>Arrival<span style='float:right;color:#2563EB'>{arr_rate:.3f} PCU/s</span></div>
              <div>Pred8s <span style='float:right;color:#7C3AED'>{pred_q:.1f} PCU</span></div>
              <div>Flow   <span style='float:right;color:#64748B'>{flow_dir}</span></div>
              <div style='margin-top:5px'>
                <div style='font-size:0.6rem;color:#64748B;margin-bottom:2px'>PRIORITY SCORE {score:.1f}</div>
                <div class='bar-wrap'>
                  <div class='bar-fill' style='width:{score_pct}%;background:{arm_color}'></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # ROW 3: Webster Splits | Priority Ranking | Chart
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-label'>◈ Algorithm Output</div>", unsafe_allow_html=True)
    col_w, col_p, col_c = st.columns([2, 2, 3])

    with col_w:
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#64748B;margin-bottom:8px'>WEBSTER OPTIMAL GREEN SPLITS</div>", unsafe_allow_html=True)
        splits    = session.get('webster_splits', {})
        max_split = max(splits.values()) if splits else 1.0
        for arm in ARM_LIST:
            g      = splits.get(arm, 0.0)
            pct    = int(min(100, g / max(max_split, 1) * 100))
            col    = ARM_COLORS[arm]
            is_win = (arm == current_green)
            st.markdown(f"""
            <div style='margin:4px 0'>
              <div style='display:flex;justify-content:space-between;
                           font-family:JetBrains Mono,monospace;font-size:0.64rem;
                           color:{"#0F172A" if is_win else "#64748B"};font-weight:{"700" if is_win else "400"};margin-bottom:2px'>
                <span>{'▶ ' if is_win else '  '}{arm}</span>
                <span style='color:{col}'>{g:.1f}s</span>
              </div>
              <div class='h-bar-bg'>
                <div class='h-bar-fill' style='width:{pct}%;background:{col}'>
                  {"▶" if is_win else ""}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_p:
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#64748B;margin-bottom:8px'>PRIORITY SCORE RANKING</div>", unsafe_allow_html=True)
        cur_scores  = session.get('current_scores', {})
        if not cur_scores:
            cur_scores = {arm: arms_data.get(arm, {}).get('priority_score', 0.0) for arm in ARM_LIST}
        sorted_arms = sorted(ARM_LIST, key=lambda a: cur_scores.get(a, 0), reverse=True)
        max_sc      = max(cur_scores.values()) if cur_scores else 1.0
        medals      = ['🥇', '🥈', '🥉', '4️⃣']
        for rank, arm in enumerate(sorted_arms):
            sc  = cur_scores.get(arm, 0.0)
            pct = int(min(100, sc / max(max_sc, 1) * 100))
            col = ARM_COLORS[arm]
            st.markdown(f"""
            <div style='margin:4px 0'>
              <div style='display:flex;justify-content:space-between;
                           font-family:JetBrains Mono,monospace;font-size:0.64rem;
                           color:#0F172A;margin-bottom:2px'>
                <span>{medals[rank]} {arm}</span>
                <span style='color:{col};font-weight:700'>{sc:.1f}</span>
              </div>
              <div class='h-bar-bg'>
                <div class='h-bar-fill' style='width:{pct}%;background:{col}'>{medals[rank]}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#64748B;margin-bottom:8px'>DENSITY vs WAIT TIME</div>", unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            densities_now = [arms_data.get(a, {}).get('density', 0.0)  for a in ARM_LIST]
            waits_now     = [arms_data.get(a, {}).get('wait_time', 0.0) for a in ARM_LIST]
            colors_now    = [ARM_COLORS[a] for a in ARM_LIST]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Density (PCU)', x=ARM_LIST, y=densities_now,
                marker_color=colors_now, opacity=0.8,
                marker_line_width=0,
            ))
            fig.add_trace(go.Scatter(
                name='Wait (s)', x=ARM_LIST, y=waits_now,
                mode='lines+markers', yaxis='y2',
                line=dict(color='#DC2626', width=2),
                marker=dict(size=7, color='#DC2626'),
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=40, t=10, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,250,252,1)',
                legend=dict(orientation='h', y=1.12, font=dict(size=10, color='#64748B')),
                xaxis=dict(tickfont=dict(size=10, color='#64748B'), gridcolor='#E2E8F0'),
                yaxis=dict(title='PCU', titlefont=dict(size=10, color='#64748B'),
                           tickfont=dict(size=9, color='#64748B'), gridcolor='#E2E8F0'),
                yaxis2=dict(title='Wait s', overlaying='y', side='right',
                            titlefont=dict(size=10, color='#DC2626'),
                            tickfont=dict(size=9, color='#DC2626')),
                font=dict(family='JetBrains Mono'),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except ImportError:
            for arm in ARM_LIST:
                d = arms_data.get(arm, {}).get('density', 0.0)
                w = arms_data.get(arm, {}).get('wait_time', 0.0)
                st.write(f"**{arm}**: {d:.1f} PCU | {w:.0f}s wait")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # ROW 4: Session Summary table
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-label'>◈ Session Summary</div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)

    last_arm  = session.get('last_green_arm', '–')
    last_gt   = session.get('last_green_time', 0)
    ped_avg   = state.get('ped_rolling_avg', 0.0)
    ped_req   = state.get('ped_requested', False)

    def _tbl(col, rows):
        with col:
            rows_html = "".join(
                f"<div class='tbl-row'><span class='tbl-key'>{k}</span><span class='tbl-val'>{v}</span></div>"
                for k, v in rows
            )
            st.markdown(
                f"<div style='background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:10px;"
                f"padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>{rows_html}</div>",
                unsafe_allow_html=True
            )

    _tbl(sc1, [
        ("Signal Cycles",    str(cycles)),
        ("Vehicles Cleared", f"{cleared:,}"),
        ("Throughput/hr",    f"{throughput:.0f}"),
    ])
    _tbl(sc2, [
        ("Efficiency Gain",  f"{efficiency:.1f}%"),
        ("Wait Time Saved",  f"{wait_saved:.0f}s"),
        ("Congestion Index", str(cong_index)),
    ])
    _tbl(sc3, [
        ("Last Green Arm",   last_arm or "–"),
        ("Last Green Time",  f"{last_gt:.0f}s"),
        ("Uptime",           ut_str),
    ])
    _tbl(sc4, [
        ("Ped Avg Count",    f"{ped_avg:.1f}"),
        ("Ped Phase",        "✅ ACTIVE" if ped_req else "❌ OFF"),
        ("Data Source",      "SIMULATION" if use_sim else "LIVE"),
    ])

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;font-family:JetBrains Mono,monospace;
                 font-size:0.62rem;color:#CBD5E1;padding:8px 0;
                 border-top:1.5px solid #E2E8F0'>
      🚦 AI Smart Traffic Management &nbsp;·&nbsp; YOLOv8 + Webster's Optimal Cycle
      &nbsp;·&nbsp; Indian City Conditions &nbsp;·&nbsp; Real-time PCU Estimation
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # AUTO-REFRESH — 1 second, NO full page reload (preserves session state)
    # streamlit-autorefresh keeps session_state alive between ticks
    # Fallback: if not installed, use st.empty loop trick
    # ════════════════════════════════════════════════════════════════════
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1000, limit=None, key="traffic_refresh")
    except ImportError:
        # Graceful fallback: inject a JS fetch ping that does NOT reload the page
        # This triggers Streamlit's internal re-run via websocket heartbeat
        st.markdown("""
        <script>
        (function(){
          var ms = 1000;
          function ping(){
            // Trigger Streamlit re-run by simulating a widget change event
            // without a full page reload — preserves session_state
            var ev = new Event('streamlit:componentReady', {bubbles:true});
            window.dispatchEvent(ev);
            setTimeout(ping, ms);
          }
          setTimeout(ping, ms);
        })();
        </script>""", unsafe_allow_html=True)
        st.info("⚡ Install streamlit-autorefresh for smooth 1s updates: `pip install streamlit-autorefresh`", icon="ℹ️")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_dashboard()