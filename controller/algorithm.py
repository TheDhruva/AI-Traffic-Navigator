# controller/algorithm.py — Traffic Signal Decision Engine
# Runs in Thread 2 (daemon). Reads IntersectionState, computes priority
# scores, sends serial commands, sleeps for signal durations.
#
# Control loop order every cycle:
#   1. Check emergency override  → immediate preemption
#   2. Check pedestrian phase    → execute after current MIN_GREEN
#   3. Check hazard extension    → extend current green
#   4. Score all arms            → select best arm
#   5. Execute green cycle       → ALL_RED → GREEN → YELLOW
#   6. Update wait times         → tick all arms

from __future__ import annotations

import logging
import time
import threading
from typing import Optional, Callable

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
    STARVATION_BOOST,
    DENSITY_WEIGHT,
    WAIT_WEIGHT,
    FLOW_WEIGHT,
    EMERGENCY_SCORE,
)
from controller.state import IntersectionState, ArmState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure scoring functions (no I/O — easy to unit-test)
# ---------------------------------------------------------------------------

def priority_score(arm: ArmState) -> float:
    """
    Compute the priority score for a single arm.

    Formula (from spec):
        score = (density × DENSITY_WEIGHT)
              + (wait_factor × WAIT_WEIGHT)
              + (flow_penalty × FLOW_WEIGHT)

        wait_factor  = 1.0 + (wait_seconds / 30.0) ^ 1.5
        flow_penalty = max(0.0, 1.0 - (flow_rate / 5.0))

    Emergency override returns +∞ (always selected first).
    Starvation boost adds STARVATION_BOOST when wait > STARVATION_THRESHOLD.

    Args:
        arm: ArmState snapshot (can be a real ArmState or a plain object
             with .density, .wait_time, .flow_rate, .emergency attributes).

    Returns:
        Float score. Higher = more urgent.
    """
    if arm.emergency:
        return EMERGENCY_SCORE

    density = max(0.0, arm.density)
    wait    = max(0.0, arm.wait_time)
    flow    = max(0.0, arm.flow_rate)

    wait_factor  = 1.0 + (wait / 30.0) ** 1.5
    flow_penalty = max(0.0, 1.0 - (flow / 5.0))

    score = (
        (density      * DENSITY_WEIGHT)
        + (wait_factor  * WAIT_WEIGHT)
        + (flow_penalty * FLOW_WEIGHT)
    )

    # Hard starvation boost — ensures no arm is perpetually starved
    if wait >= STARVATION_THRESHOLD:
        score += STARVATION_BOOST
        logger.warning(
            "Starvation boost applied to %s (wait=%.0fs)",
            arm.arm_name, wait,
        )

    return score


def compute_green_time(arm: ArmState) -> int:
    """
    Calculate dynamic green duration for the chosen arm.

    Formula:
        green_time = clamp(density × 1.5, MIN_GREEN, MAX_GREEN)

    Args:
        arm: ArmState for the arm receiving green.

    Returns:
        Integer seconds in [MIN_GREEN, MAX_GREEN].
    """
    raw = arm.density * 1.5
    return int(min(MAX_GREEN, max(MIN_GREEN, raw)))


def select_best_arm(arms: dict[str, ArmState]) -> tuple[str, float]:
    """
    Score all arms and return the arm with the highest priority.

    Args:
        arms: Dict of arm_name → ArmState.

    Returns:
        (best_arm_name, best_score) tuple.
    """
    scores = {name: priority_score(arm) for name, arm in arms.items()}
    best   = max(scores, key=lambda k: scores[k])
    return best, scores[best]


# ---------------------------------------------------------------------------
# Signal Controller — runs the full control loop in its own thread
# ---------------------------------------------------------------------------

class SignalController:
    """
    Event-driven traffic signal controller.

    Runs as a daemon thread. On each iteration:
      - Reads shared IntersectionState (with lock).
      - Decides next phase (emergency / pedestrian / normal).
      - Sends serial commands via a send_command callback.
      - Sleeps for the appropriate duration.
      - Writes results back to IntersectionState.

    The send_command callback signature:
        send_command(arm: str, phase: str, duration: int) -> None
    e.g. "N", "GREEN", 30  →  sends "N:GREEN:30\n" to Arduino.

    If no Arduino is connected, pass a no-op lambda and the controller
    runs in simulation mode — Pygame still reads the state correctly.
    """

    def __init__(
        self,
        state: IntersectionState,
        send_command: Optional[Callable[[str, str, int], None]] = None,
    ) -> None:
        """
        Args:
            state:        Shared IntersectionState instance.
            send_command: Callback to send serial command to Arduino.
                          Signature: (arm_initial, phase_str, duration_int).
                          Pass None for simulation-only mode.
        """
        self._state        = state
        self._send         = send_command or self._noop_send
        self._thread: Optional[threading.Thread] = None

        # Hazard extension bookkeeping — track remaining extension seconds
        self._hazard_extra: float = 0.0

        logger.info(
            "SignalController initialised (send_command=%s)",
            "hardware" if send_command else "simulation",
        )

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the controller loop in a daemon thread."""
        self._thread = threading.Thread(
            target=self._loop,
            name="SignalController",
            daemon=True,
        )
        self._thread.start()
        logger.info("SignalController thread started")

    def stop(self) -> None:
        """Signal the controller to stop cleanly."""
        with self._state.lock:
            self._state.running = False
        logger.info("SignalController stop requested")

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the controller thread to finish."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """
        Infinite control loop. Runs until state.running = False.
        Each iteration is one complete signal cycle for one arm.
        """
        logger.info("Control loop started")

        with self._state.lock:
            self._state.phase = 'normal'

        while True:
            # Check shutdown flag
            with self._state.lock:
                running = self._state.running
            if not running:
                logger.info("Control loop exiting (running=False)")
                self._execute_all_red(reason="shutdown")
                break

            try:
                self._cycle()
            except Exception as exc:
                # Never let an unhandled exception kill the controller thread.
                # Log it and continue — worst case is one missed cycle.
                logger.error(
                    "Unhandled exception in control cycle: %s", exc,
                    exc_info=True,
                )
                time.sleep(ALL_RED_BUFFER)

    def _cycle(self) -> None:
        """Execute one complete signal cycle."""

        # ── 1. Read current state snapshot ───────────────────────────────
        with self._state.lock:
            arms_snapshot = {
                name: _arm_copy(arm)
                for name, arm in self._state.arms.items()
            }
            ped_triggered  = (
                self._state.latest_emergency_result is not None
                and self._state.latest_emergency_result.ped_phase_triggered
            )
            # Use ped_rolling_avg as a fallback if no emergency result yet
            ped_avg        = self._state.ped_rolling_avg
            current_green  = self._state.current_green

        # ── 2. Emergency override — highest priority ──────────────────────
        emrg_arm = self._find_emergency_arm(arms_snapshot)
        if emrg_arm:
            self._execute_emergency(emrg_arm)
            return

        # ── 3. Pedestrian phase ───────────────────────────────────────────
        if ped_triggered:
            self._execute_pedestrian()
            return

        # ── 4. Hazard extension on current green arm ──────────────────────
        if current_green and self._is_hazard_active(arms_snapshot, current_green):
            self._extend_for_hazard(current_green)
            # After extension, fall through to normal scoring

        # ── 5. Normal priority scoring ────────────────────────────────────
        best_arm, best_score = select_best_arm(arms_snapshot)
        green_duration = compute_green_time(arms_snapshot[best_arm])

        logger.info(
            "Selected arm: %s  score=%.1f  green=%ds",
            best_arm, best_score, green_duration,
        )

        # ── 6. Execute green cycle ────────────────────────────────────────
        self._execute_green_cycle(best_arm, green_duration)

    # ------------------------------------------------------------------
    # Phase executors
    # ------------------------------------------------------------------

    def _execute_green_cycle(self, arm: str, green_duration: int) -> None:
        """
        ALL_RED(1s) → arm GREEN(green_duration) → arm YELLOW(3s)
        Updates state and wait times throughout.
        """
        cycle_start = time.time()

        # ALL RED buffer
        self._execute_all_red(reason=f"before {arm} green")

        # GREEN
        with self._state.lock:
            self._state.phase         = 'normal'
            self._state.current_green = arm
            self._state.set_signal(None, 'RED')
            self._state.set_signal(arm, 'GREEN')
            self._state.arms[arm].green_count    += 1
            self._state.arms[arm].last_green_start = time.time()

        arm_initial = arm[0].upper()
        self._send(arm_initial, 'GREEN', green_duration)
        logger.debug("GREEN: %s for %ds", arm, green_duration)

        self._sleep_interruptible(green_duration, check_emergency=True)

        # YELLOW
        with self._state.lock:
            self._state.set_signal(arm, 'YELLOW')

        self._send(arm_initial, 'YELLOW', YELLOW_DURATION)
        logger.debug("YELLOW: %s for %ds", arm, YELLOW_DURATION)
        time.sleep(YELLOW_DURATION)

        # Update timing metrics
        elapsed = time.time() - cycle_start
        with self._state.lock:
            self._state.arms[arm].total_green_s += green_duration
            self._state.tick_wait_times(green_duration + YELLOW_DURATION)
            self._state.total_cycles += 1
            self._state.current_green = None
            self._state.cycle_complete.set()
            self._state.cycle_complete.clear()

        logger.info(
            "Cycle complete: %s  green=%ds  elapsed=%.1fs",
            arm, green_duration, elapsed,
        )

    def _execute_all_red(self, reason: str = "") -> None:
        """Set all arms RED for ALL_RED_BUFFER seconds."""
        with self._state.lock:
            self._state.set_signal(None, 'RED')
            self._state.current_green = None
            if reason != "shutdown":
                self._state.phase = 'all_red'

        self._send('A', 'RED', 0)
        time.sleep(ALL_RED_BUFFER)

    def _execute_emergency(self, arm: str) -> None:
        """
        Emergency override:
            ALL_RED(1s) → emergency arm GREEN(EMERGENCY_HOLD s) → ALL_RED(1s)
        """
        logger.warning("EMERGENCY PHASE: arm=%s hold=%ds", arm, EMERGENCY_HOLD)

        with self._state.lock:
            self._state.phase = 'emergency'
            self._state.set_signal(None, 'RED')
            self._state.current_green = None

        self._send('A', 'RED', 0)
        time.sleep(ALL_RED_BUFFER)

        with self._state.lock:
            self._state.set_signal(arm, 'GREEN')
            self._state.current_green = arm

        arm_initial = arm[0].upper()
        self._send(arm_initial, 'GREEN', EMERGENCY_HOLD)
        time.sleep(EMERGENCY_HOLD)

        # Clear emergency flag
        with self._state.lock:
            if arm in self._state.arms:
                self._state.arms[arm].emergency = False
            self._state.current_green = None
            self._state.phase = 'normal'

        self._execute_all_red(reason="after emergency")

        logger.info("Emergency phase complete — arm %s cleared", arm)

    def _execute_pedestrian(self) -> None:
        """
        Pedestrian phase:
            ALL_RED(1s) → PED WALK(PED_WALK_DURATION s) → ALL_RED(1s)
        """
        logger.info("PEDESTRIAN PHASE: walk=%ds", PED_WALK_DURATION)

        with self._state.lock:
            self._state.phase         = 'pedestrian'
            self._state.ped_phase_active = True
            self._state.set_signal(None, 'RED')
            self._state.current_green = None

        self._send('A', 'RED', 0)
        time.sleep(ALL_RED_BUFFER)

        self._send('P', 'WALK', PED_WALK_DURATION)
        time.sleep(PED_WALK_DURATION)

        with self._state.lock:
            self._state.ped_phase_active = False
            self._state.phase = 'normal'

            # Clear ped trigger in emergency result so it doesn't re-fire
            if self._state.latest_emergency_result is not None:
                self._state.latest_emergency_result.ped_phase_triggered = False

        self._execute_all_red(reason="after pedestrian")

        logger.info("Pedestrian phase complete")

    def _extend_for_hazard(self, arm: str) -> None:
        """
        Extend current green by HAZARD_EXTENSION seconds to allow road clearing.
        Only extends once per hazard event — tracks via self._hazard_extra.
        """
        if self._hazard_extra > 0:
            return   # already extended this event

        logger.warning(
            "HAZARD EXTENSION: extending %s green by +%ds",
            arm, HAZARD_EXTENSION,
        )
        self._hazard_extra = HAZARD_EXTENSION
        arm_initial = arm[0].upper()
        self._send(arm_initial, 'GREEN', HAZARD_EXTENSION)
        time.sleep(HAZARD_EXTENSION)

        with self._state.lock:
            self._state.arms[arm].total_green_s += HAZARD_EXTENSION

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_emergency_arm(
        self, arms_snapshot: dict[str, ArmState]
    ) -> Optional[str]:
        """Return the first arm with emergency=True, or None."""
        for name, arm in arms_snapshot.items():
            if arm.emergency:
                return name
        return None

    def _is_hazard_active(
        self, arms_snapshot: dict[str, ArmState], arm: str
    ) -> bool:
        """Return True if the named arm has an active hazard."""
        a = arms_snapshot.get(arm)
        return a is not None and a.hazard

    def _sleep_interruptible(
        self,
        duration: float,
        check_emergency: bool = False,
        interval: float = 0.5,
    ) -> None:
        """
        Sleep for `duration` seconds, waking every `interval` seconds.
        If check_emergency is True and an emergency is detected mid-sleep,
        return early so the next cycle handles it immediately.

        Args:
            duration:        Total sleep duration in seconds.
            check_emergency: Whether to poll for emergency preemption.
            interval:        Poll interval in seconds.
        """
        elapsed = 0.0
        while elapsed < duration:
            sleep_chunk = min(interval, duration - elapsed)
            time.sleep(sleep_chunk)
            elapsed += sleep_chunk

            if not check_emergency:
                continue

            # Check if an emergency appeared mid-green
            with self._state.lock:
                running = self._state.running
                emrg = any(
                    arm.emergency
                    for arm in self._state.arms.values()
                )

            if not running:
                return

            if emrg:
                logger.warning(
                    "Emergency detected mid-green — interrupting after %.1fs",
                    elapsed,
                )
                return

    @staticmethod
    def _noop_send(arm: str, phase: str, duration: int) -> None:
        """No-op send function used when Arduino is not connected."""
        logger.debug("SIM send: %s:%s:%d", arm, phase, duration)


# ---------------------------------------------------------------------------
# Arm snapshot helper — avoids holding the lock while scoring
# ---------------------------------------------------------------------------

def _arm_copy(arm: ArmState) -> ArmState:
    """
    Return a lightweight copy of an ArmState for use outside the lock.
    Uses object.__new__ + manual field copy to avoid dataclass overhead.
    """
    copy = ArmState.__new__(ArmState)
    copy.arm_name        = arm.arm_name
    copy.density         = arm.density
    copy.flow_rate       = arm.flow_rate
    copy.emergency       = arm.emergency
    copy.hazard          = arm.hazard
    copy.wait_time       = arm.wait_time
    copy.last_green_start = arm.last_green_start
    copy.signal          = arm.signal
    copy.green_count     = arm.green_count
    copy.total_green_s   = arm.total_green_s
    return copy


# ---------------------------------------------------------------------------
# Standalone test  (python -m controller.algorithm)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    from controller.state import create_state

    state = create_state()

    # Inject realistic arm conditions
    with state.lock:
        state.arms['North'].density   = 28.0
        state.arms['North'].flow_rate = 0.5   # stopped
        state.arms['North'].wait_time = 45.0

        state.arms['South'].density   = 6.0
        state.arms['South'].flow_rate = 3.2
        state.arms['South'].wait_time = 12.0

        state.arms['East'].density    = 14.0
        state.arms['East'].flow_rate  = 1.2
        state.arms['East'].wait_time  = 30.0

        state.arms['West'].density    = 2.0
        state.arms['West'].flow_rate  = 4.5
        state.arms['West'].wait_time  = 0.0

    # Print scores
    print("\n── Priority Scores ──")
    with state.lock:
        for name, arm in state.arms.items():
            score = priority_score(arm)
            gtime = compute_green_time(arm)
            print(
                f"  {name:<6}  density={arm.density:5.1f}  "
                f"wait={arm.wait_time:5.1f}s  "
                f"flow={arm.flow_rate:4.1f}  "
                f"score={score:7.2f}  green={gtime}s"
            )
        arms_copy = {n: _arm_copy(a) for n, a in state.arms.items()}

    best, score = select_best_arm(arms_copy)
    print(f"\n  → Best arm: {best}  (score={score:.2f})")
    print(f"  → Green time: {compute_green_time(arms_copy[best])}s")

    # Run controller for a few cycles in simulation mode
    print("\n── Running controller (simulation, 3 cycles) ──\n")

    cycle_count = [0]

    original_execute = SignalController._execute_green_cycle

    def patched_execute(self, arm, green_duration):
        cycle_count[0] += 1
        original_execute(self, arm, green_duration)
        if cycle_count[0] >= 3:
            with self._state.lock:
                self._state.running = False

    SignalController._execute_green_cycle = patched_execute

    controller = SignalController(state)
    controller.start()

    # Poll and print status until stopped
    for _ in range(60):
        time.sleep(0.5)
        print(state.summary_string())
        with state.lock:
            if not state.running:
                break

    controller.join(timeout=5.0)
    print("\nController test complete.")