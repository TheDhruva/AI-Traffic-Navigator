"""
dashboard/app.py — Smart City Traffic Dashboard (Production-Grade)
===================================================================
Upgrade from functional → impressive:
  • Bright modern theme (white bg, color-coded signals)
  • Clean metric cards with delta indicators
  • Real-time analytics: throughput, wait time, efficiency gain
  • Congestion heat indicators per arm
  • Animated signal state display
  • Webster splits visualization
  • Predicted queue 8s ahead (predictive AI badge)
  • Uses state.to_dict() (canonical contract) — no more ad-hoc field access

Run standalone:
    streamlit run dashboard/app.py

Or integrated via DashboardBridge (started in main.py):
    bridge = DashboardBridge(state, emergency_detector)
    bridge.start()   # writes /tmp/traffic_state.json every 500ms
    # Then: streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Dashboard state file (IPC between main.py and Streamlit process) ─────────
STATE_FILE = Path("/tmp/traffic_state.json")
ANALYTICS_FILE = Path("/tmp/traffic_analytics.json")
REFRESH_INTERVAL_MS = 500   # Streamlit auto-refresh period

# ═══════════════════════════════════════════════════════════════════════════
# Dashboard Bridge — runs in main.py, writes state file for Streamlit
# ═══════════════════════════════════════════════════════════════════════════

class DashboardBridge(threading.Thread):
    """
    Runs as daemon thread in main.py.
    Periodically serializes IntersectionState → JSON → writes to /tmp/traffic_state.json.
    Streamlit dashboard reads this file to avoid cross-process import issues.

    Write interval: 500ms (2 FPS for dashboard is sufficient)
    """

    def __init__(
        self,
        state,                    # IntersectionState
        emergency_detector=None,  # EmergencyDetector (optional)
        write_interval: float = 0.5,
    ) -> None:
        super().__init__(name="DashboardBridge", daemon=True)
        self._state = state
        self._emergency = emergency_detector
        self._interval = write_interval
        self._stop = threading.Event()

    def run(self) -> None:
        logger.info("DashboardBridge: starting (writes every %.1fs)", self._interval)
        while not self._stop.is_set():
            try:
                self._write_state()
            except Exception as exc:
                logger.debug("DashboardBridge write error: %s", exc)
            time.sleep(self._interval)

    def stop(self) -> None:
        self._stop.set()

    def _write_state(self) -> None:
        """Serialize state to canonical JSON using state.to_dict()."""
        state_dict = self._state.to_dict()

        # Enrich with emergency detector data if available
        if self._emergency:
            try:
                emrg_data = self._emergency.get_status_dict()
                state_dict['emergency_status'] = emrg_data
            except Exception:
                pass

        try:
            STATE_FILE.write_text(json.dumps(state_dict, default=str), encoding='utf-8')
        except OSError as exc:
            logger.debug("Cannot write state file: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit Dashboard App
# ═══════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    """Load state from JSON file. Returns empty dict on failure."""
    try:
        if STATE_FILE.exists():
            raw = STATE_FILE.read_text(encoding='utf-8')
            return json.loads(raw)
    except Exception:
        pass
    return {}


def _signal_color(sig: str) -> str:
    """Return hex color for signal state."""
    return {'GREEN': '#00C851', 'YELLOW': '#FFB300', 'RED': '#FF3547'}.get(sig, '#888888')


def _congestion_color(density: float) -> str:
    """Traffic density → color: green/yellow/orange/red."""
    if density < 5:    return '#00C851'   # free flow
    if density < 12:   return '#FFB300'   # moderate
    if density < 20:   return '#FF6D00'   # heavy
    return '#FF3547'                       # critical


def _phase_badge(phase: str) -> str:
    badges = {
        'normal':     '🟢 NORMAL',
        'pedestrian': '🚶 PEDESTRIAN',
        'emergency':  '🚨 EMERGENCY',
        'all_red':    '🔴 ALL RED',
        'startup':    '⏳ STARTING',
    }
    return badges.get(phase, f'⚪ {phase.upper()}')


def run_dashboard() -> None:
    """Main Streamlit dashboard entry point."""
    try:
        import streamlit as st
    except ImportError:
        print("streamlit not installed: pip install streamlit")
        return

    # ── Page config ───────────────────────────────────────────────────────
    st.set_page_config(
        page_title="AI Smart Traffic System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Custom CSS (bright, modern, clean) ────────────────────────────────
    st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #F8FAFC; }
    .block-container { padding: 1rem 2rem 1rem 2rem; max-width: 100%; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2563EB;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .metric-delta-pos { color: #10B981; font-size: 0.9rem; font-weight: 600; }
    .metric-delta-neg { color: #EF4444; font-size: 0.9rem; font-weight: 600; }

    /* Signal indicator */
    .signal-box {
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Arm card */
    .arm-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 10px;
        border: 1px solid #E2E8F0;
    }
    .arm-name {
        font-size: 1rem;
        font-weight: 700;
        color: #1E293B;
    }
    .arm-stats {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Status banner */
    .status-banner {
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* Progress bar override */
    .stProgress > div > div { height: 8px; border-radius: 4px; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Webster split bars */
    .split-bar {
        background: #EEF2FF;
        border-radius: 4px;
        height: 24px;
        position: relative;
        margin: 4px 0;
        overflow: hidden;
    }
    .split-fill {
        height: 100%;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    col_logo, col_title, col_time = st.columns([1, 6, 2])
    with col_logo:
        st.markdown("# 🚦")
    with col_title:
        st.markdown("""
        <div style='padding-top:8px'>
            <span style='font-size:1.6rem;font-weight:800;color:#1E293B'>
                AI Smart Traffic Management
            </span>
            <span style='font-size:0.9rem;color:#64748B;margin-left:12px'>
                Powered by YOLOv8 + Webster's Algorithm
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_time:
        st.markdown(f"""
        <div style='text-align:right;padding-top:12px;color:#64748B;font-size:0.85rem'>
            Last refresh: {time.strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Load state ─────────────────────────────────────────────────────────
    state = _load_state()
    arms = state.get('arms', {})
    session = state.get('session', {})
    phase = state.get('phase', 'startup')
    current_green = state.get('current_green')
    congestion_index = state.get('congestion_index', 0)

    arm_list = ['North', 'South', 'East', 'West']

    # ── Phase status banner ────────────────────────────────────────────────
    phase_colors = {
        'normal':     '#2563EB',
        'pedestrian': '#7C3AED',
        'emergency':  '#DC2626',
        'all_red':    '#374151',
        'startup':    '#64748B',
    }
    banner_color = phase_colors.get(phase, '#64748B')
    st.markdown(f"""
    <div class='status-banner' style='background:{banner_color};color:white'>
        {_phase_badge(phase)}
        {'  ·  🟢 ' + current_green + ' is GREEN' if current_green else ''}
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Row 1: KPI Metric Cards
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>📊 System Performance</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    uptime = session.get('uptime_s', 0)
    uptime_str = f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}"

    cycles = session.get('total_cycles', 0)
    cleared = session.get('vehicles_cleared', 0)
    efficiency = session.get('efficiency_gain_pct', 0.0)
    throughput = session.get('throughput_per_hour', 0.0)
    wait_saved = session.get('total_wait_saved', 0.0)

    with k1:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#2563EB'>
            <div class='metric-value'>{uptime_str}</div>
            <div class='metric-label'>⏱ System Uptime</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#10B981'>
            <div class='metric-value'>{cleared:,}</div>
            <div class='metric-label'>🚗 Vehicles Cleared</div>
            <div class='metric-delta-pos'>↑ {throughput:.0f}/hr throughput</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        eff_color = '#10B981' if efficiency > 0 else '#EF4444'
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:{eff_color}'>
            <div class='metric-value' style='color:{eff_color}'>{efficiency:.1f}%</div>
            <div class='metric-label'>⚡ Efficiency vs Fixed-Time</div>
            <div class='metric-delta-pos'>↑ Better than naive baseline</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        cong_color = _congestion_color(congestion_index / 5.0)
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:{cong_color}'>
            <div class='metric-value' style='color:{cong_color}'>{congestion_index}</div>
            <div class='metric-label'>🌡 Congestion Index (0–100)</div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:#F59E0B'>
            <div class='metric-value'>{wait_saved:.0f}s</div>
            <div class='metric-label'>⏳ Total Wait Time Saved</div>
            <div class='metric-delta-pos'>↑ vs fixed-time baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Row 2: Arm Signal States + Per-Arm Details
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>🚦 Intersection Arms</div>", unsafe_allow_html=True)
    arm_cols = st.columns(4)

    for i, arm in enumerate(arm_list):
        arm_data = arms.get(arm, {})
        sig = arm_data.get('signal_state', 'RED')
        density = arm_data.get('density', 0.0)
        queue = arm_data.get('queue_length', 0)
        wait = arm_data.get('wait_time', 0.0)
        score = arm_data.get('priority_score', 0.0)
        pred_q = arm_data.get('predicted_q8s', 0.0)
        arr_rate = arm_data.get('arrival_rate', 0.0)
        flow_dir = arm_data.get('flow_direction', 'unknown')
        emergency = arm_data.get('emergency', False)
        hazard = arm_data.get('hazard', False)

        sig_color = _signal_color(sig)
        dens_color = _congestion_color(density)

        # Emergency/hazard badges
        badges = ''
        if emergency: badges += ' 🚨'
        if hazard:    badges += ' ⚠️'

        with arm_cols[i]:
            # Signal box
            st.markdown(f"""
            <div class='signal-box' style='background:{sig_color}'>
                {arm[0]} · {sig}{badges}
            </div>
            """, unsafe_allow_html=True)

            # Stats
            st.markdown(f"""
            <div class='arm-card'>
                <div class='arm-name'>{arm}</div>
                <div class='arm-stats'>
                    📦 Density: <b style='color:{dens_color}'>{density:.1f} PCU</b><br>
                    🚗 Queue: <b>{queue} vehicles</b><br>
                    ⏱ Wait: <b>{wait:.0f}s</b><br>
                    📈 Score: <b>{score:.1f}</b><br>
                    🔮 Pred (8s): <b>{pred_q:.1f} PCU</b><br>
                    ➡ Flow: <b>{flow_dir}</b><br>
                    📡 Arrival: <b>{arr_rate:.3f} PCU/s</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Density progress bar
            bar_pct = min(1.0, density / 21.0)
            st.progress(bar_pct, text=f"Capacity: {bar_pct*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Row 3: Webster Splits + Priority Scores side by side
    # ══════════════════════════════════════════════════════════════════════
    col_splits, col_scores = st.columns(2)

    with col_splits:
        st.markdown("<div class='section-header'>📐 Webster Optimal Green Splits</div>", unsafe_allow_html=True)
        splits = session.get('webster_splits', {})
        arm_colors = {
            'North': '#2563EB',
            'South': '#10B981',
            'East':  '#F59E0B',
            'West':  '#7C3AED',
        }
        if splits:
            max_split = max(splits.values()) if splits else 1.0
            for arm in arm_list:
                g = splits.get(arm, 0.0)
                pct = int(min(100, g / max_split * 100))
                color = arm_colors.get(arm, '#888')
                st.markdown(f"""
                <div style='margin:4px 0'>
                    <span style='font-size:0.82rem;font-weight:600;color:#475569;
                                 display:inline-block;width:60px'>{arm}</span>
                    <div class='split-bar' style='display:inline-block;width:calc(100% - 110px);
                                                   vertical-align:middle;'>
                        <div class='split-fill' style='width:{pct}%;background:{color}'>
                            {g:.0f}s
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for Webster calculation...")

    with col_scores:
        st.markdown("<div class='section-header'>🏆 Priority Scores (Current Cycle)</div>", unsafe_allow_html=True)
        current_scores = session.get('current_scores', {})
        if current_scores:
            # Sort by score descending
            sorted_arms = sorted(arm_list, key=lambda a: current_scores.get(a, 0), reverse=True)
            max_score = max(current_scores.values()) if current_scores else 1.0
            for rank, arm in enumerate(sorted_arms):
                sc = current_scores.get(arm, 0.0)
                pct = int(min(100, sc / max(max_score, 1.0) * 100))
                rank_emoji = ['🥇', '🥈', '🥉', '4️⃣'][rank]
                color = arm_colors.get(arm, '#888')
                st.markdown(f"""
                <div style='margin:4px 0'>
                    <span style='font-size:0.82rem;font-weight:600;color:#475569;
                                 display:inline-block;width:80px'>{rank_emoji} {arm}</span>
                    <div class='split-bar' style='display:inline-block;width:calc(100% - 130px);
                                                   vertical-align:middle;'>
                        <div class='split-fill' style='width:{pct}%;background:{color}'>
                            {sc:.1f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for scoring cycle...")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # Row 4: Efficiency metrics + session table
    # ══════════════════════════════════════════════════════════════════════
    col_eff, col_sess = st.columns([3, 2])

    with col_eff:
        st.markdown("<div class='section-header'>📉 Traffic Efficiency vs Fixed-Time Baseline</div>", unsafe_allow_html=True)

        try:
            import plotly.graph_objects as go

            # Generate efficiency timeline data
            # (In real deployment, this comes from a time-series buffer)
            # Here we compute instantaneous metrics for display
            arm_densities = {arm: arms.get(arm, {}).get('density', 0.0) for arm in arm_list}
            arm_waits = {arm: arms.get(arm, {}).get('wait_time', 0.0) for arm in arm_list}

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Density (PCU)',
                x=arm_list,
                y=[arm_densities[a] for a in arm_list],
                marker_color=[arm_colors[a] for a in arm_list],
                opacity=0.85,
            ))
            fig.add_trace(go.Scatter(
                name='Wait Time (s)',
                x=arm_list,
                y=[arm_waits[a] for a in arm_list],
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='#EF4444', width=2),
                marker=dict(size=8),
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=20, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(orientation='h', y=1.1),
                yaxis=dict(title='PCU Density', gridcolor='#F1F5F9'),
                yaxis2=dict(title='Wait (s)', overlaying='y', side='right'),
                font=dict(size=11),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback if plotly not installed
            for arm in arm_list:
                d = arms.get(arm, {}).get('density', 0.0)
                w = arms.get(arm, {}).get('wait_time', 0.0)
                st.write(f"**{arm}**: {d:.1f} PCU | Wait: {w:.0f}s")

    with col_sess:
        st.markdown("<div class='section-header'>📋 Session Summary</div>", unsafe_allow_html=True)
        last_arm = session.get('last_green_arm', '–')
        last_gt = session.get('last_green_time', 0)
        ped_avg = state.get('ped_rolling_avg', 0.0)
        ped_req = state.get('ped_requested', False)

        summary_data = {
            "Signal Cycles": cycles,
            "Vehicles Cleared": cleared,
            "Throughput/hr": f"{throughput:.0f}",
            "Efficiency Gain": f"{efficiency:.1f}%",
            "Wait Time Saved": f"{wait_saved:.0f}s",
            "Last Green Arm": last_arm,
            "Last Green Time": f"{last_gt:.0f}s",
            "Pedestrian Avg": f"{ped_avg:.1f}",
            "Ped Phase Active": "✅" if ped_req else "❌",
        }
        for label, value in summary_data.items():
            col_a, col_b = st.columns([2, 1])
            col_a.caption(label)
            col_b.markdown(f"**{value}**")

    # ── Auto-refresh ───────────────────────────────────────────────────────
    st.markdown("""
    <script>
    setTimeout(function() { window.location.reload(); }, 1000);
    </script>
    """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94A3B8;font-size:0.8rem'>"
        "🚦 AI Smart Traffic Management · YOLOv8 + Webster's Optimal Cycle · "
        "Indian City Conditions · Real-time PCU Estimation"
        "</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_dashboard()