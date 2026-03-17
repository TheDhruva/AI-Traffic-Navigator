# controller/state.py
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np

# Use ARM_NAMES if it exists in config, otherwise default
try:
    from config import ARM_NAMES
except ImportError:
    ARM_NAMES = ['North', 'South', 'East', 'West']

@dataclass
class ArmState:
    arm_name: str
    density: float = 0.0          # PCU density
    flow_rate: float = 0.0        # optical flow magnitude
    emergency: bool = False
    hazard: bool = False
    wait_time: float = 0.0
    last_green_start: float = 0.0
    signal: str = 'RED'           # 'GREEN', 'YELLOW', 'RED'
    green_count: int = 0
    total_green_s: int = 0

class IntersectionState:
    def __init__(self):
        self.lock = threading.RLock()
        self.running: bool = True
        self.phase: str = 'startup'
        self.current_green: Optional[str] = None
        self.arms: Dict[str, ArmState] = {
            arm: ArmState(arm_name=arm) for arm in ARM_NAMES
        }
        self.ped_rolling_avg: float = 0.0
        self.ped_phase_active: bool = False
        self.latest_emergency_result = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.start_time = time.time()
        
        self.total_cycles: int = 0
        self.cycle_complete = threading.Event()
        
    def update_from_density(self, density_result):
        with self.lock:
            for arm, dens in density_result.densities.items():
                if arm in self.arms:
                    self.arms[arm].density = dens
            self.ped_rolling_avg = density_result.ped_rolling_avg

    def update_from_flow(self, flow_result):
        with self.lock:
            for arm, mag in flow_result.smoothed.items():
                if arm in self.arms:
                    self.arms[arm].flow_rate = mag

    def update_from_emergency(self, emrg_result):
        with self.lock:
            self.latest_emergency_result = emrg_result
            
            # Update hazard flags based on emergency result
            for arm in self.arms.values():
                arm.hazard = arm.arm_name in emrg_result.hazard_arms
                arm.emergency = (
                        emrg_result.emergency_detected
                        and emrg_result.emergency_arm == arm.arm_name
                )
                if emrg_result.emergency_detected and emrg_result.emergency_arm == arm.arm_name:
                    arm.emergency = True

    def set_annotated_frame(self, frame):
        with self.lock:
            if frame is not None:
                self.annotated_frame = frame.copy()

    def get_annotated_frame(self):
        with self.lock:
            if self.annotated_frame is None:
                return None
            return self.annotated_frame.copy()

    def set_signal(self, arm: Optional[str], signal_color: str):
        # if arm is None, set all arms
        with self.lock:
            if arm is None:
                for a in self.arms.values():
                    a.signal = signal_color
            else:
                if arm in self.arms:
                    self.arms[arm].signal = signal_color

    def tick_wait_times(self, elapsed_s: float):
        with self.lock:
            for arm in self.arms.values():
                if arm.signal == 'RED':
                    arm.wait_time += elapsed_s
                else:
                    arm.wait_time = 0.0

    def snapshot_arms(self) -> Dict[str, ArmState]:
        # Avoid circular imports by reconstructing ArmState directly
        with self.lock:
            out = {}
            for name, arm in self.arms.items():
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
                out[name] = copy
            return out

    def snapshot_phase(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "current_green": self.current_green,
                "ped_rolling_avg": self.ped_rolling_avg,
                "total_cycles": self.total_cycles,
                "vehicles_cleared": self.total_cycles * 5,
                "uptime_s": time.time() - self.start_time,
            }

    def summary_string(self) -> str:
        with self.lock:
            return f"State: phase={self.phase} green={self.current_green}"

def create_state() -> IntersectionState:
    return IntersectionState()