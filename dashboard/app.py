# dashboard/app.py — AI Smart Traffic Command Center
# =====================================================
# This file serves two roles in one module:
#
#   Role 1 — DashboardBridge (imported by main.py)
#     Runs as a daemon thread inside the main process.
#     Reads IntersectionState + annotated frames every 500ms and writes them
#     to /tmp/ as JSON + JPEGs so the Streamlit process can read them without
#     sharing memory across processes.
#
#   Role 2 — Streamlit App (run separately)
#     Launched with:  streamlit run dashboard/app.py
#     Reads from /tmp/, renders the full command center UI, and writes back
#     control params + spawn commands that main.py picks up next tick.
#
# IPC contract (/tmp/ files):
#   /tmp/traffic_state.json   — full state snapshot (written by DashboardBridge)
#   /tmp/frame_{arm}.jpg      — latest annotated JPEG per arm (written by Bridge)
#   /tmp/control.json         — live param overrides (written by Streamlit sidebar)
#   /tmp/spawn.json           — one-shot spawn command (written by Streamlit sidebar)
#
# Why /tmp/ files instead of multiprocessing.Manager or Redis?
#   Zero extra dependencies, survives Python subprocess boundaries, works on
#   every OS the judges' laptop might run, and is fast enough at 500ms cadence.
#
# Run order:
#   Terminal 1:  python main.py --demo
#   Terminal 2:  streamlit run dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    # Avoid importing heavy CV deps at Streamlit startup time
    from controller.state import IntersectionState
    from detection.emergency import EmergencyDetector

logger = logging.getLogger(__name__)

# ─── IPC paths ────────────────────────────────────────────────────────────────
_TMP           = Path("/tmp")
_STATE_FILE    = _TMP / "traffic_state.json"
_CONTROL_FILE  = _TMP / "control.json"
_SPAWN_FILE    = _TMP / "spawn.json"
_FRAME_PATTERN = str(_TMP / "frame_{arm}.jpg")   # .format(arm="North")

_ARM_NAMES = ["North", "South", "East", "West"]

# JPEG compression quality for IPC frames — 70 is indistinguishable on screen
# but ~3× smaller than 95, which matters when writing 4 files every 500ms.
_JPEG_QUALITY = 70


# ═════════════════════════════════════════════════════════════════════════════
# Role 1 — DashboardBridge  (runs inside main.py process)
# ═════════════════════════════════════════════════════════════════════════════

class DashboardBridge(threading.Thread):
    """
    Daemon thread that serialises IntersectionState → /tmp/ at a fixed cadence.

    Instantiated and started in main.py::

        bridge = DashboardBridge(state, emergency_detector=det)
        bridge.start()

    The Streamlit process reads those files independently — no shared memory,
    no sockets, no extra dependencies.

    Args:
        state:              Shared IntersectionState from main.py.
        emergency_detector: Optional EmergencyDetector; used to expose
                            simulate_emergency() / simulate_ped_rush() via
                            the /tmp/spawn.json command channel.
        interval_s:         How often to write state (default 0.5 s = 2 Hz).
                            Faster = smoother dashboard, higher I/O cost.
    """

    def __init__(
        self,
        state: "IntersectionState",
        emergency_detector: Optional["EmergencyDetector"] = None,
        interval_s: float = 0.5,
    ) -> None:
        super().__init__(name="DashboardBridge", daemon=True)
        self._state      = state
        self._emrg_det   = emergency_detector
        self._interval   = interval_s
        self._running    = True

        # Write a sane default control file so Streamlit can read params
        # even before the user opens the sidebar.
        self._write_default_control()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        logger.info("DashboardBridge started (interval=%.1fs)", self._interval)
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                # Never crash main.py over a dashboard write failure
                logger.debug("DashboardBridge tick error (non-fatal): %s", exc)
            time.sleep(self._interval)
        logger.info("DashboardBridge stopped")

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """One write cycle: snapshot state → JSON, annotated frames → JPEGs."""

        # ── 1. Serialize state ────────────────────────────────────────────────
        arm_snap   = self._state.snapshot_arms()
        phase_snap = self._state.snapshot_phase()

        # Compute congestion level from total PCU density
        total_density = sum(getattr(s, "density", 0.0) for s in arm_snap.values())
        if total_density < 12:
            congestion = "LOW"
        elif total_density < 28:
            congestion = "MEDIUM"
        elif total_density < 45:
            congestion = "HIGH"
        else:
            congestion = "CRITICAL"

        state_doc = {
            "timestamp": time.time(),
            "arms": arm_snap,
            "phase": phase_snap,
            "congestion": congestion,
        }
        _atomic_write_json(_STATE_FILE, state_doc)

        # ── 2. Write annotated frames ─────────────────────────────────────────
        # IntersectionState stores one shared annotated frame (single-camera mode).
        # We write it to all four arm slots so the dashboard always shows something.
        annotated = self._state.get_annotated_frame()
        if annotated is not None:
            ret, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
            )
            if ret:
                raw_bytes = buf.tobytes()
                for arm in _ARM_NAMES:
                    _atomic_write_bytes(
                        Path(_FRAME_PATTERN.format(arm=arm)), raw_bytes
                    )

        # ── 3. Process spawn commands from Streamlit sidebar ─────────────────
        self._process_spawn_command()

    # ── Spawn command handler ─────────────────────────────────────────────────

    def _process_spawn_command(self) -> None:
        """
        Read /tmp/spawn.json (one-shot command from Streamlit).
        Unlinks the file immediately after reading so it isn't processed twice.

        Supported commands::
            {"type": "ambulance", "arm": "North"}   → simulate_emergency()
            {"type": "ped_rush"}                     → simulate_ped_rush()
        """
        if not _SPAWN_FILE.exists():
            return
        try:
            cmd = json.loads(_SPAWN_FILE.read_text())
            _SPAWN_FILE.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Failed to read spawn command: %s", exc)
            return

        cmd_type = cmd.get("type", "")
        arm      = cmd.get("arm", "North")

        if cmd_type == "ambulance" and self._emrg_det is not None:
            self._emrg_det.simulate_emergency(arm)
            logger.info("[DashboardBridge] Spawned emergency on %s", arm)

        elif cmd_type == "ped_rush" and self._emrg_det is not None:
            self._emrg_det.simulate_ped_rush()
            logger.info("[DashboardBridge] Spawned pedestrian rush")

        elif cmd_type == "reset_waits":
            # Zero out all arm wait timers — useful during demo resets
            with self._state.lock:
                for arm_obj in self._state.arms.values():
                    arm_obj.wait_time = 0.0
            logger.info("[DashboardBridge] Wait timers reset")

        else:
            logger.debug("[DashboardBridge] Unknown spawn command: %s", cmd)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _write_default_control() -> None:
        if not _CONTROL_FILE.exists():
            _atomic_write_json(_CONTROL_FILE, _default_control())


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers used by both Bridge and the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def _default_control() -> dict:
    return {
        "spawn_rate":  1.0,
        "ped_rate":    0.3,
        "sim_speed":   1.0,
        "min_green":   10,
        "max_green":   60,
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to a temp file then rename — prevents Streamlit reading a partial write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=_json_default))
    tmp.replace(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _json_default(obj):
    """Fallback serializer for numpy types that appear in state dicts."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_state() -> dict:
    """Read /tmp/traffic_state.json. Returns empty dict on any failure."""
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _load_frame(arm: str) -> Optional[np.ndarray]:
    """Read the latest JPEG for *arm*. Returns None if unavailable."""
    p = Path(_FRAME_PATTERN.format(arm=arm))
    try:
        raw = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _load_control() -> dict:
    try:
        return json.loads(_CONTROL_FILE.read_text())
    except Exception:
        return _default_control()


# ═════════════════════════════════════════════════════════════════════════════
# Role 2 — Streamlit App
#
# Everything below this guard is only executed when Streamlit imports this file
# as its entry point.  The DashboardBridge class above is still importable by
# main.py without triggering any Streamlit calls.
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or os.environ.get("STREAMLIT_RUNTIME"):
    # ── Guard: only import streamlit when running as dashboard ───────────────
    import streamlit as st
    _STREAMLIT_RUNNING = True
else:
    _STREAMLIT_RUNNING = False


def _run_streamlit_app() -> None:
    """
    Full Streamlit dashboard.  Called at module level when Streamlit imports
    this file.  Streamlit re-runs this function top-to-bottom on every
    interaction and every st.rerun() call.
    """
    import streamlit as st

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="AI Traffic Command Center",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS — clean, bright, judge-readable ────────────────────────────
    st.markdown("""
    <style>
        /* Tighten top padding */
        .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

        /* Metric card styling */
        [data-testid="stMetric"] {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px 14px;
        }
        [data-testid="stMetricLabel"]  { font-size: 0.72rem; color: #555; }
        [data-testid="stMetricValue"]  { font-size: 1.4rem;  font-weight: 700; }

        /* Camera feed caption */
        .cam-caption {
            font-size: 0.78rem; color: #444;
            margin-top: 2px; line-height: 1.4;
        }

        /* Alert badge */
        .badge-red   { background:#ff4b4b; color:#fff; padding:2px 8px;
                       border-radius:12px; font-size:0.75rem; font-weight:600; }
        .badge-green { background:#00c851; color:#fff; padding:2px 8px;
                       border-radius:12px; font-size:0.75rem; font-weight:600; }
        .badge-amber { background:#ffbb33; color:#222; padding:2px 8px;
                       border-radius:12px; font-size:0.75rem; font-weight:600; }
        .badge-blue  { background:#33b5e5; color:#fff; padding:2px 8px;
                       border-radius:12px; font-size:0.75rem; font-weight:600; }

        /* Section headers */
        h3 { margin-bottom: 0.4rem !important; }

        /* Sidebar slider labels */
        .stSlider > label { font-size: 0.82rem; }

        /* Reduce sidebar top whitespace */
        section[data-testid="stSidebar"] > div { padding-top: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Load current state ────────────────────────────────────────────────────
    state      = _load_state()
    arm_data   = state.get("arms",       {})
    phase_data = state.get("phase",      {})
    congestion = state.get("congestion", "—")
    ts         = state.get("timestamp",  0)
    data_age_s = time.time() - ts if ts else 999

    # ── Sidebar — Control Panel ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        _sidebar_status_badge(data_age_s)
        st.divider()

        st.markdown("### ⚙️ Simulation Parameters")
        spawn_rate = st.slider("Vehicle Spawn Rate",  0.1, 3.0, 1.0, 0.1,
                               help="Vehicles per second per arm (Pygame sim)")
        ped_rate   = st.slider("Pedestrian Rate",     0.0, 2.0, 0.3, 0.1,
                               help="Pedestrians per second in crosswalk ROI")
        sim_speed  = st.slider("Simulation Speed",    0.5, 3.0, 1.0, 0.5,
                               help="Time multiplier for Pygame animation")
        min_green  = st.slider("Min Green Time (s)",  5,   20,  10,
                               help="Minimum green phase duration per arm")
        max_green  = st.slider("Max Green Time (s)",  30,  90,  60,
                               help="Maximum green phase duration per arm")

        # Persist control params so Pygame and controller can read them
        _atomic_write_json(_CONTROL_FILE, {
            "spawn_rate": spawn_rate, "ped_rate": ped_rate,
            "sim_speed":  sim_speed,  "min_green": min_green,
            "max_green":  max_green,
        })

        st.divider()
        st.markdown("### 🕹️ Manual Spawn")

        arm_sel = st.selectbox("Target Arm", _ARM_NAMES, index=0)

        col_v, col_p = st.columns(2)
        if col_v.button("🚗 Car",       use_container_width=True):
            _send_spawn("car", arm_sel)
            st.toast(f"🚗 Car spawned on {arm_sel}", icon="🚗")
        if col_p.button("🚌 Bus",       use_container_width=True):
            _send_spawn("bus", arm_sel)
            st.toast(f"🚌 Bus spawned on {arm_sel}", icon="🚌")

        col_a, col_b = st.columns(2)
        if col_a.button("🛺 Auto",      use_container_width=True):
            _send_spawn("auto", arm_sel)
            st.toast(f"🛺 Auto spawned on {arm_sel}", icon="🛺")
        if col_b.button("🏍️ Bike",     use_container_width=True):
            _send_spawn("motorcycle", arm_sel)
            st.toast(f"🏍️ Bike spawned on {arm_sel}", icon="🏍️")

        col_c, col_d = st.columns(2)
        if col_c.button("🚶 Pedestrian",use_container_width=True):
            _send_spawn("ped_rush")
            st.toast("🚶 Pedestrian rush triggered", icon="🚶")
        if col_d.button("🐄 Animal",    use_container_width=True):
            _send_spawn("animal", arm_sel)
            st.toast(f"🐄 Animal on {arm_sel}", icon="🐄")

        st.divider()
        st.markdown("### 🚨 Override Controls")
        if st.button("🚑 Ambulance Override",
                     type="primary", use_container_width=True,
                     help="Triggers ALL RED → Emergency arm GREEN"):
            _send_spawn("ambulance", arm_sel)
            st.toast(f"🚨 Emergency override — {arm_sel} arm!", icon="🚨")

        if st.button("🔄 Reset Wait Timers",
                     use_container_width=True,
                     help="Zero all arm wait counters — use after demo scenario"):
            _send_spawn("reset_waits")
            st.toast("Wait timers reset", icon="🔄")

        st.divider()
        st.caption(
            "**Terminal 1:** `python main.py --demo`\n\n"
            "**Terminal 2:** `streamlit run dashboard/app.py`"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main content
    # ─────────────────────────────────────────────────────────────────────────

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_col, badge_col = st.columns([3, 1])
    with title_col:
        st.markdown("# 🚦 AI Traffic Command Center")
        st.caption("Indian Cities Smart Signal Optimization · YOLOv8 + Arduino")
    with badge_col:
        _congestion_badge(congestion)

    # ── Signal status row (4 metrics) ─────────────────────────────────────────
    st.markdown("### 🟢 Live Signal Status")
    _render_signal_status(phase_data, congestion)

    # Emergency / hazard / pedestrian alerts
    _render_alert_banners(arm_data, phase_data)

    st.divider()

    # ── Camera feeds (2 × 2 grid) ─────────────────────────────────────────────
    st.markdown("### 📹 Camera Feeds — YOLO Detection")
    _render_camera_grid(arm_data, phase_data)

    st.divider()

    # ── Analytics row ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Traffic Analytics")
    _render_analytics(arm_data)

    st.divider()

    # ── Session stats ─────────────────────────────────────────────────────────
    st.markdown("### 📈 Session Statistics")
    _render_session_stats(phase_data)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    # Streamlit >= 1.27 supports st.rerun(). Older versions need experimental.
    time.sleep(0.6)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_signal_status(phase_data: dict, congestion: str) -> None:
    import streamlit as st

    phase        = phase_data.get("phase",         "—")
    current_green= phase_data.get("current_green", "—")
    total_cycles = phase_data.get("total_cycles",  0)
    uptime_s     = phase_data.get("uptime_s",      0.0)

    # Determine phase colour label
    phase_icons = {
        "normal":      "🟢 NORMAL",
        "emergency":   "🔴 EMERGENCY",
        "pedestrian":  "🔵 WALK",
        "all_red":     "🔴 ALL RED",
        "yellow":      "🟡 YELLOW",
    }
    phase_label = phase_icons.get(phase.lower(), f"⚪ {phase.upper()}")

    # Congestion colour
    cong_icons = {
        "LOW":      "🟢 LOW",
        "MEDIUM":   "🟡 MEDIUM",
        "HIGH":     "🟠 HIGH",
        "CRITICAL": "🔴 CRITICAL",
    }
    cong_label = cong_icons.get(congestion, f"— {congestion}")

    uptime_str = _fmt_uptime(uptime_s)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Green Arm", f"ARM {current_green}" if current_green != "—" else "—")
    c2.metric("Signal Phase",     phase_label)
    c3.metric("Congestion Level", cong_label)
    c4.metric("Signal Cycles",    f"{total_cycles}  ·  {uptime_str}")


def _render_alert_banners(arm_data: dict, phase_data: dict) -> None:
    import streamlit as st

    phase = phase_data.get("phase", "normal").lower()

    if phase == "emergency":
        emrg_arm = next(
            (arm for arm, s in arm_data.items() if getattr(s, "emergency")), "Unknown"
        )
        st.error(f"🚨 **EMERGENCY OVERRIDE ACTIVE** — {emrg_arm} arm has priority. "
                 "All other arms are RED.")

    elif phase == "pedestrian":
        avg = phase_data.get("ped_rolling_avg", 0.0)
        st.info(f"🚶 **PEDESTRIAN PHASE** — {avg:.1f} pedestrians detected. "
                "WALK signal active. All vehicle arms RED.")

    hazard_arms = {
        arm: getattr(s, "hazard")
        for arm, s in arm_data.items()
        if getattr(s, "hazard")
    }
    if hazard_arms:
        hazard_str = ", ".join(
            f"{arm} ({cls})" for arm, cls in hazard_arms.items()
        )
        st.warning(f"⚠️ **ANIMAL HAZARD** detected on: {hazard_str}. "
                   "Signal extension active.")


def _render_camera_grid(arm_data: dict, phase_data: dict) -> None:
    import streamlit as st

    current_green = phase_data.get("current_green", "")
    cols = st.columns(4)

    for i, arm in enumerate(_ARM_NAMES):
        s         = arm_data.get(arm, {})
        density   = getattr(s, "density",   0.0)
        wait      = getattr(s, "wait_time", 0.0)
        emergency = getattr(s, "emergency", False)
        hazard    = getattr(s, "hazard",    False)
        is_green  = (arm == current_green)

        with cols[i]:
            # Status badge in the column header
            if emergency:
                st.markdown(
                    f'<span class="badge-red">🚨 EMERGENCY</span> **{arm}**',
                    unsafe_allow_html=True,
                )
            elif hazard:
                st.markdown(
                    f'<span class="badge-amber">⚠️ HAZARD</span> **{arm}**',
                    unsafe_allow_html=True,
                )
            elif is_green:
                st.markdown(
                    f'<span class="badge-green">🟢 GREEN</span> **{arm}**',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge-red">🔴 RED</span> **{arm}**',
                    unsafe_allow_html=True,
                )

            # Camera frame
            frame = _load_frame(arm)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_column_width=True)
            else:
                # Placeholder while main.py starts up
                placeholder = np.full((180, 320, 3), 30, dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"{arm}: awaiting feed",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                )
                st.image(placeholder, use_column_width=True,
                         channels="BGR")

            # Metrics beneath the frame
            st.markdown(
                f'<div class="cam-caption">'
                f'PCU density: <b>{density:.1f}</b> &nbsp;|&nbsp; '
                f'Wait: <b>{wait:.0f}s</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Density bar (visual only — no extra widget needed)
            bar_pct = min(density / 50.0, 1.0)
            bar_color = _density_bar_color(bar_pct)
            st.markdown(
                f'<div style="height:6px;border-radius:3px;'
                f'background:linear-gradient(to right,{bar_color} {bar_pct*100:.0f}%,'
                f'#e0e0e0 {bar_pct*100:.0f}%);margin-top:4px;"></div>',
                unsafe_allow_html=True,
            )


def _render_analytics(arm_data: dict) -> None:
    import streamlit as st

    cols = st.columns(4)
    for i, arm in enumerate(_ARM_NAMES):
        s         = arm_data.get(arm, {})
        density   = getattr(s, "density",   0.0)
        wait      = getattr(s, "wait_time", 0.0)

        with cols[i]:
            st.markdown(f"**{arm} Arm**")
            st.metric("PCU Density",  f"{density:.1f}")
            st.metric("Wait Time",    f"{wait:.0f} s")

            # Classify density level
            if density < 10:
                level_html = '<span class="badge-green">LOW</span>'
            elif density < 25:
                level_html = '<span class="badge-amber">MEDIUM</span>'
            elif density < 40:
                level_html = '<span style="background:#ff8800;color:#fff;'  \
                             'padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600">HIGH</span>'
            else:
                level_html = '<span class="badge-red">CRITICAL</span>'

            st.markdown(f"Density level: {level_html}", unsafe_allow_html=True)
            st.markdown("")   # breathing room


def _render_session_stats(phase_data: dict) -> None:
    import streamlit as st

    total_cycles     = phase_data.get("total_cycles",    0)
    vehicles_cleared = phase_data.get("vehicles_cleared", 0)
    uptime_s         = phase_data.get("uptime_s",         0.0)
    ped_avg          = phase_data.get("ped_rolling_avg",   0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal Cycles Completed", str(total_cycles))
    c2.metric("Vehicles Cleared",        str(vehicles_cleared))
    c3.metric("System Uptime",           _fmt_uptime(uptime_s))
    c4.metric("Ped Rolling Avg",         f"{ped_avg:.1f} / 8.0",
              help="Pedestrian phase triggers at rolling average ≥ 8")


# ─────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar_status_badge(data_age_s: float) -> None:
    import streamlit as st
    if data_age_s < 2.0:
        st.markdown(
            '<span class="badge-green">● LIVE</span> main.py connected',
            unsafe_allow_html=True,
        )
    elif data_age_s < 10.0:
        st.markdown(
            f'<span class="badge-amber">● DELAYED</span> {data_age_s:.0f}s ago',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge-red">● OFFLINE</span> Run: `python main.py --demo`',
            unsafe_allow_html=True,
        )


def _congestion_badge(congestion: str) -> None:
    import streamlit as st
    color_map = {
        "LOW":      "#00c851",
        "MEDIUM":   "#ffbb33",
        "HIGH":     "#ff8800",
        "CRITICAL": "#ff4b4b",
    }
    color = color_map.get(congestion, "#999")
    st.markdown(
        f'<div style="text-align:right;margin-top:16px;">'
        f'<span style="background:{color};color:{"#222" if congestion=="MEDIUM" else "#fff"};'
        f'padding:6px 16px;border-radius:16px;font-size:1rem;font-weight:700;">'
        f'Congestion: {congestion}</span></div>',
        unsafe_allow_html=True,
    )


def _send_spawn(cmd_type: str, arm: str = "North") -> None:
    """Write a one-shot spawn command for DashboardBridge to consume."""
    _atomic_write_json(_SPAWN_FILE, {"type": cmd_type, "arm": arm})


def _density_bar_color(ratio: float) -> str:
    """Return a CSS hex colour: green → amber → red based on 0–1 ratio."""
    if ratio < 0.4:
        return "#00c851"
    if ratio < 0.7:
        return "#ffbb33"
    return "#ff4b4b"


def _fmt_uptime(seconds: float) -> str:
    """Format seconds as H:MM:SS string."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ═════════════════════════════════════════════════════════════════════════════
# Entry point — Streamlit calls this module directly, so we detect that and
# run the app.  When imported by main.py only DashboardBridge is used.
# ═════════════════════════════════════════════════════════════════════════════

# Streamlit re-imports the module on every rerun, so we just call the function
# at module level and let Streamlit's script runner handle the loop.
try:
    import streamlit as _st_probe  # noqa: F401
    _RUNNING_IN_STREAMLIT = True
except ImportError:
    _RUNNING_IN_STREAMLIT = False

if _RUNNING_IN_STREAMLIT and os.environ.get("STREAMLIT_SERVER_PORT"):
    _run_streamlit_app()