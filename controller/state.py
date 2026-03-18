"""
controller/state.py — Unified Intersection State with Clean Data Contract
=========================================================================
Single source of truth for all threads.

Problems with old state:
  • Backend objects had no serialization method → UI used ad-hoc dict building
  • Inconsistent field names between state.arms[arm].density vs JSON 'density_pcu'
  • Missing fields: priority_score, arrival_rate, predicted_queue, efficiency metrics
  • session_metrics didn't exist as a first-class field

This version:
  • ArmState.to_dict() → canonical dict used by ALL consumers (UI, dashboard, logging)
  • IntersectionState.to_json() → full snapshot, thread-safe, one call
  • session_metrics is a first-class field (dict), updated by controller
  • ped_phase_requested is a first-class bool field (not ad-hoc attribute)
  • All fields documented with units

Thread safety:
  Use state.lock (threading.Lock) for ALL reads and writes.
  Copy values out of the lock before using them outside.
  Never hold state.lock during sleep() or render.

Canonical field contract (ArmState.to_dict()):
  {
    "arm":            str,       # "North" | "South" | "East" | "West"
    "density":        float,     # PCU-weighted vehicle count in ROI
    "queue_length":   int,       # estimated vehicle count (density / avg PCU)
    "wait_time":      float,     # seconds since arm last had green
    "flow_rate":      float,     # optical flow magnitude (px/frame, 0–10)
    "flow_direction": str,       # "toward" | "away" | "stopped" | "unknown"
    "emergency":      bool,      # ambulance/fire truck detected
    "hazard":         bool,      # animal on road
    "priority_score": float,     # latest score from algorithm
    "arrival_rate":   float,     # smoothed PCU/s arriving
    "predicted_q8s":  float,     # predicted queue 8s from now
    "last_green_ago": float,     # seconds since last green (same as wait_time for non-green)
    "signal_state":   str,       # "GREEN" | "YELLOW" | "RED"
  }
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Per-arm state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArmState:
    arm_name: str

    # Detection fields
    density: float = 0.0
    wait_time: float = 0.0
    flow_rate: float = 0.0
    flow_direction: str = 'unknown'
    emergency: bool = False
    hazard: bool = False
    ped_count: int = 0
    last_green: float = 0.0
    vehicle_count: int = 0

    # Control fields
    priority_score: float = 0.0
    signal_state: str = 'RED'          # "RED" | "YELLOW" | "GREEN"
    green_time_allocated: float = 0.0

    # Analytics
    arrival_rate: float = 0.0
    discharge_rate: float = 0.0
    predicted_q8s: float = 0.0

    # ── BUG 2 FIX: .signal property alias ────────────────────────────────
    # pygame_sim.py uses getattr(s, 'signal', 'RED').
    # ArmState has signal_state, not signal. Add property alias.
    @property
    def signal(self) -> str:
        """Alias for signal_state — backward compat with pygame_sim.py."""
        return self.signal_state

    @signal.setter
    def signal(self, value: str) -> None:
        self.signal_state = value

    def queue_length(self) -> int:
        avg_pcu = 0.7
        return max(0, int(round(self.density / max(avg_pcu, 0.1))))

    def to_dict(self) -> dict:
        now = time.time()
        last_green_ago = now - self.last_green if self.last_green > 0 else self.wait_time
        return {
            "arm":              self.arm_name,
            "density":          round(self.density, 2),
            "queue_length":     self.queue_length(),
            "wait_time":        round(self.wait_time, 1),
            "flow_rate":        round(self.flow_rate, 3),
            "flow_direction":   self.flow_direction,
            "emergency":        self.emergency,
            "hazard":           self.hazard,
            "ped_count":        self.ped_count,
            "vehicle_count":    self.vehicle_count,
            "priority_score":   round(self.priority_score, 2),
            "arrival_rate":     round(self.arrival_rate, 3),
            "discharge_rate":   round(self.discharge_rate, 3),
            "predicted_q8s":    round(self.predicted_q8s, 2),
            "last_green_ago":   round(last_green_ago, 1),
            "signal_state":     self.signal_state,
            # Include signal alias for any consumers using old key
            "signal":           self.signal_state,
            "green_time_allocated": round(self.green_time_allocated, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Intersection state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IntersectionState:
    arms: Dict[str, ArmState]

    current_green: Optional[str] = None
    phase: str = 'startup'
    running: bool = True

    ped_phase_requested: bool = False
    ped_rolling_avg: float = 0.0

    _annotated_frame = None
    _frame_lock: threading.Lock = field(default_factory=threading.Lock)

    session_metrics: dict = field(default_factory=lambda: {
        'total_cycles':         0,
        'vehicles_cleared':     0,
        'total_wait_saved':     0.0,
        'uptime_s':             0.0,
        'last_green_arm':       None,
        'last_green_time':      0.0,
        'webster_splits':       {},
        'current_scores':       {},
        'efficiency_gain_pct':  0.0,
        'throughput_per_hour':  0.0,
    })

    lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.time)

    # ── BUG 3 FIX: total_cycles property ─────────────────────────────────
    @property
    def total_cycles(self) -> int:
        """Backward compat — pygame_sim references state.total_cycles."""
        return self.session_metrics.get('total_cycles', 0)

    @total_cycles.setter
    def total_cycles(self, value: int) -> None:
        self.session_metrics['total_cycles'] = value

    # ── BUG 3 FIX: set_signal() method ───────────────────────────────────
    def set_signal(self, arm: Optional[str], sig: str) -> None:
        """
        Set signal state for one arm (or all arms if arm=None).
        Called by pygame_sim.py _force_green() and standalone mock.
        MUST be called with state.lock already held (called inside with block).
        """
        if arm is None:
            # Set all arms to sig
            for a in self.arms.values():
                a.signal_state = sig
        else:
            if arm in self.arms:
                self.arms[arm].signal_state = sig

    # ── BUG 1 FIX: _update_arm_signals() ─────────────────────────────────
    def _update_arm_signals_locked(self, current_green: Optional[str], phase: str) -> None:
        """
        Update per-arm signal_state based on current_green + phase.
        MUST be called with state.lock already held.

        This is the missing link — the controller sets current_green and phase
        but never wrote to arm.signal_state. Pygame reads arm.signal_state.
        Without this, all arms stay RED forever regardless of what the controller does.
        """
        if phase in ('all_red', 'pedestrian', 'startup'):
            for arm in self.arms.values():
                arm.signal_state = 'RED'
        elif phase == 'green' and current_green:
            for arm_name, arm in self.arms.items():
                arm.signal_state = 'GREEN' if arm_name == current_green else 'RED'
            if current_green in self.arms:
                self.arms[current_green].wait_time = 0.0
                self.arms[current_green].last_green = time.time()
        elif phase == 'yellow' and current_green:
            for arm_name, arm in self.arms.items():
                arm.signal_state = 'YELLOW' if arm_name == current_green else 'RED'
        elif phase == 'emergency' and current_green:
            for arm_name, arm in self.arms.items():
                arm.signal_state = 'GREEN' if arm_name == current_green else 'RED'

    # ── BUG 4 FIX: update_from_* methods ─────────────────────────────────
    def update_from_density(self, density_result) -> None:
        """
        Write DensityResult into arm states.
        Called by DetectionThread with state.lock already held.
        Handles both old DensityResult (has .densities dict) and new ArmDensityResult.
        """
        # Handle new DensityResult (has arm_results dict)
        if hasattr(density_result, 'arm_results'):
            for arm_name, ar in density_result.arm_results.items():
                if arm_name in self.arms:
                    a = self.arms[arm_name]
                    a.density = ar.density
                    a.vehicle_count = ar.vehicle_count
                    a.emergency = ar.has_emergency
                    a.hazard = ar.has_hazard
                    a.arrival_rate = ar.arrival_rate_delta

        # Handle old DensityResult (has .densities dict) — backward compat
        elif hasattr(density_result, 'densities'):
            for arm_name, density_val in density_result.densities.items():
                if arm_name in self.arms:
                    self.arms[arm_name].density = density_val
            for arm_name in density_result.emergency_arms:
                if arm_name in self.arms:
                    self.arms[arm_name].emergency = True
            for arm_name in density_result.hazard_arms:
                if arm_name in self.arms:
                    self.arms[arm_name].hazard = True

        # Pedestrian
        if hasattr(density_result, '_ped_rolling_avg'):
            self.ped_rolling_avg = density_result._ped_rolling_avg
            if density_result._ped_phase_triggered:
                self.ped_phase_requested = True
        elif hasattr(density_result, 'ped_rolling_avg'):
            self.ped_rolling_avg = density_result.ped_rolling_avg
            if getattr(density_result, 'ped_phase_triggered', False):
                self.ped_phase_requested = True

    def update_from_flow(self, flow_result) -> None:
        """
        Write FlowResult into arm states.
        Handles both new FlowResult (has .arm_results) and old (has .flow_rates).
        """
        if flow_result is None:
            return

        # New FlowResult
        if hasattr(flow_result, 'arm_results'):
            for arm_name, ar in flow_result.arm_results.items():
                if arm_name in self.arms:
                    self.arms[arm_name].flow_rate = ar.magnitude
                    self.arms[arm_name].flow_direction = ar.flow_quality
                    self.arms[arm_name].discharge_rate = ar.discharge_rate

        # Old FlowResult (has .flow_rates dict)
        elif hasattr(flow_result, 'flow_rates'):
            for arm_name, rate in flow_result.flow_rates.items():
                if arm_name in self.arms:
                    self.arms[arm_name].flow_rate = rate

    def update_from_emergency(self, emrg_result) -> None:
        """
        Write EmergencyResult into arm states.
        Handles EmergencyDetector output — different from density emergency flags.
        """
        if emrg_result is None:
            return

        # New EmergencyResult format
        if hasattr(emrg_result, 'emergency_arm') and emrg_result.emergency_arm:
            arm = emrg_result.emergency_arm
            if arm in self.arms:
                self.arms[arm].emergency = True

        # Ped phase from emergency detector
        if getattr(emrg_result, 'ped_phase_triggered', False):
            self.ped_phase_requested = True
        if hasattr(emrg_result, 'ped_rolling_avg'):
            self.ped_rolling_avg = emrg_result.ped_rolling_avg

        # Hazard arms
        if hasattr(emrg_result, 'hazard_arms'):
            for arm_name in (emrg_result.hazard_arms or {}):
                if arm_name in self.arms:
                    self.arms[arm_name].hazard = True

    # ── Annotated frame ───────────────────────────────────────────────────

    def set_annotated_frame(self, frame) -> None:
        with self._frame_lock:
            self._annotated_frame = frame

    def get_annotated_frame(self):
        with self._frame_lock:
            return self._annotated_frame

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self.lock:
            arms_data = {arm: self.arms[arm].to_dict() for arm in self.arms}
            metrics = dict(self.session_metrics)
            phase = self.phase
            current_green = self.current_green
            ped_req = self.ped_phase_requested
            ped_avg = self.ped_rolling_avg
            uptime = time.time() - self._start_time

        metrics['uptime_s'] = round(uptime, 1)
        if metrics['uptime_s'] > 0:
            metrics['throughput_per_hour'] = round(
                metrics['vehicles_cleared'] / metrics['uptime_s'] * 3600, 1
            )
        max_density = max((a['density'] for a in arms_data.values()), default=0.0)
        congestion_index = min(100, int(max_density / 21.0 * 100))

        return {
            "timestamp":        time.time(),
            "phase":            phase,
            "current_green":    current_green,
            "ped_requested":    ped_req,
            "ped_rolling_avg":  round(ped_avg, 1),
            "congestion_index": congestion_index,
            "arms":             arms_data,
            "session":          metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    # ── Snapshot helpers ──────────────────────────────────────────────────

    def snapshot_arms(self) -> Dict[str, dict]:
        """
        Returns per-arm as dicts (canonical contract).
        NOTE: pygame_sim uses getattr(s, 'signal', 'RED') on these dicts.
        Dict key access is used instead in the patched pygame_sim below.
        """
        with self.lock:
            return {arm: self.arms[arm].to_dict() for arm in self.arms}

    def snapshot_arms_obj(self) -> Dict[str, ArmState]:
        """
        Returns live ArmState objects (no copy).
        Used by pygame_sim when it needs object attribute access.
        WARNING: do not hold state.lock while rendering.
        """
        with self.lock:
            return dict(self.arms)

    def snapshot_phase(self) -> dict:
        with self.lock:
            return {
                'phase':            self.phase,
                'current_green':    self.current_green,
                'total_cycles':     self.session_metrics.get('total_cycles', 0),
                'vehicles_cleared': self.session_metrics.get('vehicles_cleared', 0),
                'uptime_s':         round(time.time() - self._start_time, 1),
                'ped_rolling_avg':  self.ped_rolling_avg,
            }

    def summary_string(self) -> str:
        with self.lock:
            parts = []
            for arm in self.arms:
                a = self.arms[arm]
                sig = a.signal_state[0]
                emrg = '!' if a.emergency else ''
                parts.append(f"{arm[0]}:{sig}({a.density:.0f}){emrg}")
            phase = self.phase
            cg = self.current_green or '-'
            cyc = self.session_metrics.get('total_cycles', 0)
        return f"Phase={phase} Green={cg} | {' | '.join(parts)} | cycles={cyc}"

    def update_wait_times(self) -> None:
        """Increment wait for non-green arms. Call once per second."""
        with self.lock:
            for arm_name, arm in self.arms.items():
                if arm_name == self.current_green:
                    arm.wait_time = 0.0
                    arm.last_green = time.time()
                else:
                    arm.wait_time += 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_state(arm_names: list[str] | None = None) -> IntersectionState:
    if arm_names is None:
        from config import ARM_NAMES
        arm_names = ARM_NAMES

    arms = {name: ArmState(arm_name=name) for name in arm_names}
    return IntersectionState(arms=arms, phase='startup', running=True)