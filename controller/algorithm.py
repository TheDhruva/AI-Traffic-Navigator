"""
BUGFIX: controller/algorithm.py — _write_current_phase fix
===========================================================
BUG 1 ROOT CAUSE:
  _write_current_phase() only wrote state.current_green and state.phase.
  It NEVER called state._update_arm_signals_locked().
  Result: arm.signal_state stayed 'RED' forever.
  Pygame reads arm.signal_state to color the traffic lights → always RED.

FIX:
  _write_current_phase() now calls state._update_arm_signals_locked()
  after writing current_green and phase, while still holding the lock.

This is a surgical fix — only _write_current_phase() changes.
Everything else in algorithm.py stays identical.

HOW TO APPLY:
  Find _write_current_phase() in your controller/algorithm.py and replace it
  with the version below. Or drop this whole file in as controller/algorithm.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from config import (
    ARM_NAMES,
    MIN_GREEN,
    MAX_GREEN,
    YELLOW_DURATION,
    ALL_RED_BUFFER,
    PED_WALK_DURATION,
    EMERGENCY_HOLD,
    HAZARD_EXTENSION,
    STARVATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Scoring weights
ALPHA: float = 3.0
BETA: float = 8.0
GAMMA: float = 4.0
DELTA: float = 2.0
EPSILON: float = 6.0
SATURATION_FLOW: float = 0.35
LOST_TIME_PER_PHASE: float = 2.0
PREDICTION_HORIZON: float = 8.0
ALPHA_SMOOTH: float = 0.3
ABSOLUTE_STARVATION_CAP: float = 150.0
MIN_DISCHARGE_THRESHOLD: float = 0.05


@dataclass
class ArmAnalytics:
    arm: str
    density_history: deque = field(default_factory=lambda: deque(maxlen=30))
    smoothed_arrival_rate: float = 0.0
    smoothed_discharge_rate: float = 0.0
    vehicles_cleared: int = 0
    peak_density: float = 0.0
    green_phases: int = 0
    total_green_time: float = 0.0

    def update_arrival_rate(self, current_density: float, dt: float) -> float:
        self.density_history.append(current_density)
        if len(self.density_history) < 3:
            return self.smoothed_arrival_rate
        recent = list(self.density_history)[-3:]
        raw_arrival = max(0.0, (recent[-1] - recent[0]) / (dt * 2)) if dt > 0 else 0.0
        self.smoothed_arrival_rate = (
            ALPHA_SMOOTH * raw_arrival + (1 - ALPHA_SMOOTH) * self.smoothed_arrival_rate
        )
        return self.smoothed_arrival_rate

    def update_discharge_rate(self, flow_rate: float, is_green: bool) -> float:
        raw_discharge = flow_rate * SATURATION_FLOW if (is_green and flow_rate > MIN_DISCHARGE_THRESHOLD) else 0.0
        self.smoothed_discharge_rate = (
            ALPHA_SMOOTH * raw_discharge + (1 - ALPHA_SMOOTH) * self.smoothed_discharge_rate
        )
        return self.smoothed_discharge_rate

    def predict_queue_in(self, seconds: float) -> float:
        if not self.density_history:
            return 0.0
        current = self.density_history[-1]
        net_rate = self.smoothed_arrival_rate - self.smoothed_discharge_rate
        return max(0.0, current + net_rate * seconds)


def webster_optimal_cycle(
    volume_flows: Dict[str, float],
    saturation_flow: float = SATURATION_FLOW,
    lost_time_per_phase: float = LOST_TIME_PER_PHASE,
    n_phases: int = 4,
) -> Dict[str, float]:
    L = n_phases * lost_time_per_phase
    total_flow = sum(volume_flows.values())
    if total_flow < 0.01:
        return {arm: float(MIN_GREEN) for arm in volume_flows}
    y = {arm: v / saturation_flow for arm, v in volume_flows.items()}
    Y = sum(y.values())
    if Y >= 0.9:
        splits = {}
        for arm, yi in y.items():
            proportion = yi / Y if Y > 0 else 1.0 / len(y)
            splits[arm] = float(min(MAX_GREEN, max(MIN_GREEN, proportion * MAX_GREEN * 4)))
        return splits
    C_opt = max(40.0, min(180.0, (1.5 * L + 5.0) / (1.0 - Y)))
    effective_green_total = C_opt - L
    splits = {}
    for arm, yi in y.items():
        proportion = yi / Y if Y > 0 else 1.0 / len(y)
        splits[arm] = float(max(MIN_GREEN, min(MAX_GREEN, effective_green_total * proportion)))
    logger.debug("Webster: C_opt=%.1fs Y=%.3f splits=%s", C_opt, Y, {k: f"{v:.1f}s" for k, v in splits.items()})
    return splits


def compute_priority_score(
    arm_name: str,
    density: float,
    wait_time: float,
    flow_rate: float,
    analytics: ArmAnalytics,
    is_current_green: bool,
    dt: float,
) -> float:
    if density < 0:
        density = 0.0
    arrival_rate = analytics.update_arrival_rate(density, dt)
    discharge_rate = analytics.update_discharge_rate(flow_rate, is_current_green)
    wait_factor = 1.0 + math.pow(max(0.0, wait_time) / 30.0, 1.8)
    saturation_capacity = SATURATION_FLOW * MAX_GREEN
    saturation_ratio = min(1.0, density / max(saturation_capacity, 1.0))
    saturation_bonus = saturation_ratio * EPSILON
    score = (
        ALPHA * density
        + BETA * wait_factor
        + GAMMA * arrival_rate
        - DELTA * discharge_rate
        + saturation_bonus
    )
    predicted_q = analytics.predict_queue_in(PREDICTION_HORIZON)
    if predicted_q > density * 1.3:
        growth_factor = min(2.0, predicted_q / max(density, 1.0))
        score *= growth_factor
    return max(0.0, score)


class SignalController(threading.Thread):
    def __init__(
        self,
        state,
        send_command: Callable[[str], None],
        use_webster: bool = True,
    ) -> None:
        super().__init__(name="SignalController", daemon=True)
        self._state = state
        self._send = send_command
        self._use_webster = use_webster
        self._stop_event = threading.Event()
        self._analytics: Dict[str, ArmAnalytics] = {
            arm: ArmAnalytics(arm=arm) for arm in ARM_NAMES
        }
        self._session_start = time.time()
        self._total_cycles = 0
        self._vehicles_cleared = 0
        self._webster_splits: Dict[str, float] = {arm: float(MIN_GREEN) for arm in ARM_NAMES}
        self._last_cycle_time = time.time()
        self._total_wait_saved = 0.0
        self._baseline_wait = 45.0

    def run(self) -> None:
        logger.info("SignalController starting (webster=%s)", self._use_webster)
        time.sleep(2.0)
        while not self._stop_event.is_set():
            try:
                self._control_cycle()
            except Exception as exc:
                logger.error("Controller error: %s", exc, exc_info=True)
                time.sleep(1.0)

    def stop(self) -> None:
        self._stop_event.set()

    def _control_cycle(self) -> None:
        snap = self._snapshot_state()
        if not snap['running']:
            self._stop_event.set()
            return

        dt = time.time() - self._last_cycle_time
        self._last_cycle_time = time.time()

        emergency_arm = self._find_emergency_arm(snap)
        if emergency_arm:
            logger.warning("EMERGENCY: arm=%s", emergency_arm)
            self._execute_emergency(emergency_arm)
            self._clear_emergency(emergency_arm)
            return

        if snap['ped_phase_requested']:
            logger.info("PEDESTRIAN PHASE")
            self._execute_pedestrian_phase()
            self._clear_ped_request()
            return

        if snap.get('hazard_active') and snap.get('current_green'):
            ext = HAZARD_EXTENSION
            logger.warning("HAZARD — extending green %ds", ext)
            self._send(f"{snap['current_green'][0]}:GREEN:{ext}\n")
            time.sleep(ext)
            return

        scores: Dict[str, float] = {}
        for arm in ARM_NAMES:
            arm_data = snap['arms'][arm]
            is_green = (snap['current_green'] == arm)
            scores[arm] = compute_priority_score(
                arm_name=arm,
                density=arm_data['density'],
                wait_time=arm_data['wait_time'],
                flow_rate=arm_data['flow_rate'],
                analytics=self._analytics[arm],
                is_current_green=is_green,
                dt=dt,
            )

        for arm in ARM_NAMES:
            if snap['arms'][arm]['wait_time'] >= ABSOLUTE_STARVATION_CAP:
                logger.warning("STARVATION: %s", arm)
                scores[arm] = float('inf') - 1.0

        winner = max(scores, key=lambda a: scores[a])
        logger.info("Scores: %s → winner=%s (%.1f)",
                    {k: f"{v:.1f}" for k, v in scores.items()}, winner, scores[winner])

        if self._use_webster:
            volume_flows = {arm: max(0.0, snap['arms'][arm]['density'] / MAX_GREEN) for arm in ARM_NAMES}
            self._webster_splits = webster_optimal_cycle(volume_flows)
            green_time = self._webster_splits.get(winner, float(MIN_GREEN))
        else:
            density = snap['arms'][winner]['density']
            arrival = self._analytics[winner].smoothed_arrival_rate
            green_time = max(MIN_GREEN, min(MAX_GREEN, density * 1.5 + arrival * 3.0))

        green_time = max(MIN_GREEN, min(MAX_GREEN, green_time))
        self._execute_phase(winner=winner, green_time=green_time, all_scores=scores)

        self._total_cycles += 1
        self._vehicles_cleared += int(snap['arms'][winner]['density'])
        self._analytics[winner].green_phases += 1
        self._analytics[winner].total_green_time += green_time
        wait_this_cycle = snap['arms'][winner]['wait_time']
        self._total_wait_saved += max(0.0, self._baseline_wait - wait_this_cycle)
        self._write_metrics(winner, green_time, scores)

    def _execute_phase(self, winner: str, green_time: float, all_scores: Dict[str, float]) -> None:
        initial = winner[0].upper()
        # ALL RED
        self._write_current_phase(winner, 'all_red')
        self._send("A:RED:0\n")
        self._sleep(ALL_RED_BUFFER)
        # GREEN
        self._write_current_phase(winner, 'green')
        self._send(f"{initial}:GREEN:{int(green_time)}\n")
        logger.info("GREEN → %s for %.0fs", winner, green_time)
        self._sleep(green_time)
        # YELLOW
        self._write_current_phase(winner, 'yellow')
        self._send(f"{initial}:YELLOW:{YELLOW_DURATION}\n")
        self._sleep(YELLOW_DURATION)
        # ALL RED
        self._write_current_phase(None, 'all_red')
        self._send("A:RED:0\n")

    def _execute_emergency(self, arm: str) -> None:
        initial = arm[0].upper()
        self._write_current_phase(arm, 'emergency')
        self._send("A:RED:0\n")
        self._sleep(ALL_RED_BUFFER)
        self._send(f"{initial}:GREEN:{EMERGENCY_HOLD}\n")
        self._sleep(EMERGENCY_HOLD)
        self._send(f"{initial}:YELLOW:{YELLOW_DURATION}\n")
        self._sleep(YELLOW_DURATION)
        self._write_current_phase(None, 'all_red')
        self._send("A:RED:0\n")

    def _execute_pedestrian_phase(self) -> None:
        self._write_current_phase(None, 'pedestrian')
        self._send("A:RED:0\n")
        self._sleep(ALL_RED_BUFFER)
        self._send(f"P:WALK:{PED_WALK_DURATION}\n")
        self._sleep(PED_WALK_DURATION)
        self._write_current_phase(None, 'all_red')
        self._send("A:RED:0\n")

    def _snapshot_state(self) -> dict:
        with self._state.lock:
            running = self._state.running
            current_green = self._state.current_green
            arms = {}
            for arm in ARM_NAMES:
                a = self._state.arms[arm]
                arms[arm] = {
                    'density':   a.density,
                    'wait_time': a.wait_time,
                    'flow_rate': a.flow_rate,
                    'emergency': a.emergency,
                    'hazard':    a.hazard,
                }
            ped_req = getattr(self._state, 'ped_phase_requested', False)
            hazard_active = any(a['hazard'] for a in arms.values())
        return {
            'running':             running,
            'current_green':       current_green,
            'arms':                arms,
            'ped_phase_requested': ped_req,
            'hazard_active':       hazard_active,
        }

    def _find_emergency_arm(self, snap: dict) -> Optional[str]:
        for arm in ARM_NAMES:
            if snap['arms'][arm]['emergency']:
                return arm
        return None

    def _write_current_phase(self, current_green: Optional[str], phase: str) -> None:
        """
        ── BUG 1 FIX ──
        Previously only wrote state.current_green and state.phase.
        Now also calls state._update_arm_signals_locked() which writes
        per-arm signal_state (RED/YELLOW/GREEN) that pygame renders.
        Without this call, all lights stayed RED forever.
        """
        with self._state.lock:
            self._state.current_green = current_green
            self._state.phase = phase
            # ← THE FIX: update each arm's signal_state
            self._state._update_arm_signals_locked(current_green, phase)

    def _clear_emergency(self, arm: str) -> None:
        with self._state.lock:
            if arm in self._state.arms:
                self._state.arms[arm].emergency = False

    def _clear_ped_request(self) -> None:
        with self._state.lock:
            self._state.ped_phase_requested = False

    def _write_metrics(self, winner: str, green_time: float, scores: Dict[str, float]) -> None:
        with self._state.lock:
            for arm in ARM_NAMES:
                if hasattr(self._state.arms[arm], 'priority_score'):
                    self._state.arms[arm].priority_score = scores.get(arm, 0.0)
            if hasattr(self._state, 'session_metrics'):
                self._state.session_metrics.update({
                    'total_cycles':       self._total_cycles,
                    'vehicles_cleared':   self._vehicles_cleared,
                    'total_wait_saved':   self._total_wait_saved,
                    'last_green_arm':     winner,
                    'last_green_time':    green_time,
                    'webster_splits':     dict(self._webster_splits),
                    'current_scores':     dict(scores),
                    'efficiency_gain_pct': (
                        self._total_wait_saved / max(1.0, self._total_cycles * self._baseline_wait) * 100
                    ),
                })

    def _sleep(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_event.is_set():
                return
            time.sleep(min(0.1, deadline - time.time()))

    def get_analytics_snapshot(self) -> dict:
        return {
            arm: {
                'smoothed_arrival_rate':   a.smoothed_arrival_rate,
                'smoothed_discharge_rate': a.smoothed_discharge_rate,
                'predicted_queue_8s':      a.predict_queue_in(8.0),
                'green_phases':            a.green_phases,
                'total_green_time':        a.total_green_time,
            }
            for arm, a in self._analytics.items()
        }

    @property
    def efficiency_gain_pct(self) -> float:
        total_baseline = self._total_cycles * self._baseline_wait
        if total_baseline < 1.0:
            return 0.0
        return self._total_wait_saved / total_baseline * 100