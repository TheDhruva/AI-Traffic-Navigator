# simulation/vehicles.py — Vehicle Spawner, Mover & Queue Manager
# Drives the top-down intersection simulation seen in the Pygame window.
# Vehicles spawn at the edge of each arm at a rate proportional to density,
# queue behind the stop line when RED, and clear when GREEN.
#
# Coordinate system:
#   (0, 0) = top-left of Pygame canvas
#   Intersection box = 300×300 centred at (400, 400)
#   North arm:  vehicles travel  ↓  (y increasing)
#   South arm:  vehicles travel  ↑  (y decreasing)
#   East arm:   vehicles travel  ←  (x decreasing)
#   West arm:   vehicles travel  →  (x increasing)

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from config import (
    ARM_NAMES,
    SIM_WIDTH,
    SIM_HEIGHT,
    SIM_FPS,
    VEHICLE_COLORS,
    VEHICLE_SPAWN_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Geometry constants for the 800×800 canvas
# ---------------------------------------------------------------------------

# Intersection box boundaries
ISECT_LEFT   = 300
ISECT_RIGHT  = 500
ISECT_TOP    = 300
ISECT_BOTTOM = 500
ISECT_CX     = 400
ISECT_CY     = 400

# Stop-line positions — vehicles halt here on RED
STOP_LINES: dict[str, int] = {
    'North': ISECT_TOP    - 4,   # y value — vehicle stops when its front reaches here
    'South': ISECT_BOTTOM + 4,   # y value
    'East':  ISECT_RIGHT  + 4,   # x value
    'West':  ISECT_LEFT   - 4,   # x value
}

# Spawn positions — where new vehicles appear (off-screen edge of each arm)
SPAWN_POSITIONS: dict[str, tuple[int, int]] = {
    'North': (ISECT_CX, -20),       # top centre, above canvas
    'South': (ISECT_CX, SIM_HEIGHT + 20),  # bottom centre
    'East':  (SIM_WIDTH + 20, ISECT_CY),   # right edge
    'West':  (-20,            ISECT_CY),   # left edge
}

# Exit positions — vehicle is removed when it passes this point
EXIT_POSITIONS: dict[str, tuple[str, int]] = {
    'North': ('y', SIM_HEIGHT + 40),
    'South': ('y', -40),
    'East':  ('x', -40),
    'West':  ('x', SIM_WIDTH + 40),
}

# Movement direction vectors (dx, dy) per arm
DIRECTIONS: dict[str, tuple[float, float]] = {
    'North': ( 0.0,  1.0),   # moving downward
    'South': ( 0.0, -1.0),   # moving upward
    'East':  (-1.0,  0.0),   # moving leftward
    'West':  ( 1.0,  0.0),   # moving rightward
}

# Vehicle dimensions (width along road, height = perpendicular)
VEHICLE_DIMS: dict[str, tuple[int, int]] = {
    'car':        (14, 22),
    'bus':        (16, 36),
    'truck':      (16, 32),
    'motorcycle': ( 8, 16),
    'auto':       (12, 18),
    'bicycle':    ( 6, 14),
}

# Speed constants (pixels per frame at SIM_FPS)
SPEED_GREEN   = 2.5     # cruising speed when arm is green
SPEED_CREEP   = 0.3     # slow creep at back of queue (red, not at stop line)
SPEED_STOPPED = 0.0     # fully stopped at stop line or behind another vehicle
DECEL_DIST    = 60.0    # pixels ahead to start decelerating for stop line / leader

# Spawn rate table: density bucket → vehicles per second per arm
_SPAWN_RATE_TABLE = [
    (5,  0.10),
    (15, 0.30),
    (30, 0.60),
    (50, 1.00),
]

# Maximum vehicles per arm before spawning is suppressed (performance guard)
MAX_VEHICLES_PER_ARM = 18


# ---------------------------------------------------------------------------
# Vehicle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    """A single vehicle in the simulation."""

    arm:       str            # which arm this vehicle belongs to
    cls:       str            # vehicle class ('car', 'bus', etc.)
    x:         float          # current x position (canvas coords)
    y:         float          # current y position
    speed:     float = SPEED_GREEN
    cleared:   bool  = False  # True when vehicle has exited the canvas

    # Unique ID for stable ordering
    vid: int = field(default_factory=lambda: Vehicle._next_id())

    # Visual extras
    color:  tuple = field(init=False)
    width:  int   = field(init=False)
    height: int   = field(init=False)

    _id_counter: int = 0  # class-level counter (see _next_id)

    def __post_init__(self) -> None:
        self.color  = VEHICLE_COLORS.get(self.cls, (180, 180, 180))
        dims = VEHICLE_DIMS.get(self.cls, (12, 20))
        # For N/S arms width is perpendicular to travel; for E/W it's swapped
        if self.arm in ('North', 'South'):
            self.width, self.height = dims
        else:
            self.width, self.height = dims[1], dims[0]

    @staticmethod
    def _next_id() -> int:
        Vehicle._id_counter += 1
        return Vehicle._id_counter

    @property
    def front(self) -> float:
        """Leading edge coordinate in the direction of travel."""
        if self.arm == 'North':  return self.y + self.height / 2
        if self.arm == 'South':  return self.y - self.height / 2
        if self.arm == 'East':   return self.x - self.width  / 2
        if self.arm == 'West':   return self.x + self.width  / 2
        return 0.0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """(x, y, width, height) suitable for pygame.draw.rect."""
        return (
            int(self.x - self.width  / 2),
            int(self.y - self.height / 2),
            self.width,
            self.height,
        )

    def distance_to_stop(self, arm: str) -> float:
        """Signed distance from vehicle front to the stop line. Positive = before line."""
        stop = STOP_LINES[arm]
        if arm == 'North':  return stop - self.front
        if arm == 'South':  return self.front - stop
        if arm == 'East':   return self.front - stop
        if arm == 'West':   return stop - self.front
        return float('inf')

    def __repr__(self) -> str:
        return f"Vehicle({self.cls} {self.arm} x={self.x:.0f} y={self.y:.0f})"


# ---------------------------------------------------------------------------
# Arm vehicle queue
# ---------------------------------------------------------------------------

class ArmQueue:
    """
    Manages all vehicles on one arm: spawning, movement, and clearing.

    Queue ordering: vehicles are ordered by their position along the arm.
    The 'head' vehicle is the one closest to the stop line.
    """

    # Gap to maintain between consecutive vehicles (pixels)
    FOLLOWING_GAP = 6

    def __init__(self, arm: str) -> None:
        self.arm      = arm
        self.vehicles: list[Vehicle] = []
        self._spawn_accumulator: float = 0.0   # fractional vehicle accumulator

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def update_spawn(self, density: float, dt: float) -> None:
        """
        Possibly spawn a new vehicle based on density and elapsed time.

        Args:
            density: PCU-weighted density for this arm (from detection).
            dt:      Seconds elapsed since last update (1 / SIM_FPS typical).
        """
        if len(self.vehicles) >= MAX_VEHICLES_PER_ARM:
            return

        rate = _density_to_spawn_rate(density)
        self._spawn_accumulator += rate * dt

        while self._spawn_accumulator >= 1.0:
            self._spawn_accumulator -= 1.0
            self._spawn_vehicle()

    def _spawn_vehicle(self) -> None:
        """Create one new vehicle at the spawn point for this arm."""
        cls    = _random_vehicle_class()
        sx, sy = SPAWN_POSITIONS[self.arm]

        # Jitter position slightly so vehicles don't stack exactly
        jitter = random.uniform(-6, 6)
        if self.arm in ('North', 'South'):
            sx += jitter
        else:
            sy += jitter

        v = Vehicle(arm=self.arm, cls=cls, x=float(sx), y=float(sy))
        self.vehicles.append(v)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def update_movement(self, is_green: bool, dt: float) -> None:
        """
        Move all vehicles on this arm forward one simulation step.

        Args:
            is_green: True when this arm is currently GREEN.
            dt:       Seconds elapsed since last update.
        """
        if not self.vehicles:
            return

        # Sort vehicles so head (closest to stop line) is first in list
        self._sort_queue()

        # Update each vehicle from head to tail so following logic works
        for i, vehicle in enumerate(self.vehicles):
            leader = self.vehicles[i - 1] if i > 0 else None
            self._move_vehicle(vehicle, leader, is_green, dt)

        # Remove vehicles that have exited the canvas
        self.vehicles = [v for v in self.vehicles if not v.cleared]

    def _move_vehicle(
        self,
        vehicle: Vehicle,
        leader: Optional[Vehicle],
        is_green: bool,
        dt: float,
    ) -> None:
        """
        Compute target speed and advance one vehicle by dt seconds.

        Speed rules (in priority order):
          1. Gap to leader vehicle         → slow/stop to maintain following gap
          2. Stop line on RED              → decelerate and stop
          3. Stop line cleared on GREEN    → continue at cruising speed
        """
        dx, dy = DIRECTIONS[vehicle.arm]
        pixels_per_frame = vehicle.speed

        # ── Determine target speed ────────────────────────────────────────
        dist_to_stop = vehicle.distance_to_stop(vehicle.arm)

        # Case 1: following gap to leader
        if leader is not None:
            gap = self._gap_to_leader(vehicle, leader)
            if gap < self.FOLLOWING_GAP:
                target_speed = SPEED_STOPPED
            elif gap < DECEL_DIST:
                target_speed = SPEED_CREEP + (gap / DECEL_DIST) * (SPEED_GREEN - SPEED_CREEP)
            else:
                target_speed = SPEED_GREEN
        else:
            target_speed = SPEED_GREEN

        # Case 2: RED stop line
        if not is_green and dist_to_stop > 0:
            if dist_to_stop < self.FOLLOWING_GAP + 2:
                target_speed = SPEED_STOPPED
            elif dist_to_stop < DECEL_DIST:
                stop_speed = SPEED_CREEP + (dist_to_stop / DECEL_DIST) * (SPEED_GREEN - SPEED_CREEP)
                target_speed = min(target_speed, stop_speed)

        # Case 3: past stop line on GREEN — clear normally
        if is_green and dist_to_stop <= 0:
            # Don't re-apply stop line; follow leader only
            pass

        # Smooth speed change (simple lerp — avoids abrupt acceleration)
        vehicle.speed = _lerp(vehicle.speed, target_speed, alpha=0.25)
        pixels_per_frame = vehicle.speed

        # ── Move ──────────────────────────────────────────────────────────
        vehicle.x += dx * pixels_per_frame
        vehicle.y += dy * pixels_per_frame

        # ── Mark cleared ──────────────────────────────────────────────────
        axis, threshold = EXIT_POSITIONS[vehicle.arm]
        pos = vehicle.x if axis == 'x' else vehicle.y
        direction_sign = dx if axis == 'x' else dy

        if direction_sign > 0 and pos > threshold:
            vehicle.cleared = True
        elif direction_sign < 0 and pos < threshold:
            vehicle.cleared = True

    def _gap_to_leader(self, follower: Vehicle, leader: Vehicle) -> float:
        """
        Return the pixel gap between follower's front and leader's rear.
        Positive = space available. Negative = overlapping (shouldn't happen).
        """
        arm = follower.arm
        if arm == 'North':
            leader_rear = leader.y - leader.height / 2
            follower_front = follower.front
            return leader_rear - follower_front
        if arm == 'South':
            leader_rear = leader.y + leader.height / 2
            follower_front = follower.front
            return follower_front - leader_rear
        if arm == 'East':
            leader_rear = leader.x + leader.width / 2
            follower_front = follower.front
            return follower_front - leader_rear
        if arm == 'West':
            leader_rear = leader.x - leader.width / 2
            follower_front = follower.front
            return leader_rear - follower_front
        return float('inf')

    def _sort_queue(self) -> None:
        """Sort vehicles so index 0 is the head (closest to stop line)."""
        arm = self.arm
        if arm == 'North':
            self.vehicles.sort(key=lambda v: v.y,  reverse=False)  # smallest y = furthest up = tail
            # Actually head = largest y (furthest down = closest to stop)
            self.vehicles.sort(key=lambda v: v.y, reverse=True)
        elif arm == 'South':
            self.vehicles.sort(key=lambda v: v.y, reverse=False)   # smallest y = head
        elif arm == 'East':
            self.vehicles.sort(key=lambda v: v.x, reverse=False)   # smallest x = head
        elif arm == 'West':
            self.vehicles.sort(key=lambda v: v.x, reverse=True)    # largest x = head

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def queue_length(self) -> int:
        """Number of vehicles currently stopped at or near the stop line."""
        stopped = [
            v for v in self.vehicles
            if v.speed <= SPEED_CREEP + 0.1
        ]
        return len(stopped)

    def __len__(self) -> int:
        return len(self.vehicles)

    def __repr__(self) -> str:
        return f"ArmQueue({self.arm} n={len(self.vehicles)} queued={self.queue_length()})"


# ---------------------------------------------------------------------------
# Intersection vehicle manager — coordinates all four arms
# ---------------------------------------------------------------------------

class VehicleManager:
    """
    Top-level manager: owns one ArmQueue per arm, drives updates.

    Called once per Pygame frame from the simulation thread.

    Usage:
        manager = VehicleManager()
        # each frame:
        manager.update(arm_states, dt=1/SIM_FPS)
        # draw:
        for arm, queue in manager.queues.items():
            for vehicle in queue.vehicles:
                draw_vehicle(vehicle)
    """

    def __init__(self) -> None:
        self.queues: dict[str, ArmQueue] = {
            arm: ArmQueue(arm) for arm in ARM_NAMES
        }
        self._last_update = time.time()
        self._cleared_total: int = 0

    def update(self, arm_states: dict, dt: Optional[float] = None) -> int:
        """
        Advance all vehicles by one simulation step.

        Args:
            arm_states: Dict of arm_name → ArmState (or plain dict with
                        'density' and 'signal' keys). Reads:
                          .density  → spawn rate
                          .signal   → 'GREEN' / other (for movement)
            dt:         Elapsed seconds. None = auto-compute from wall clock.

        Returns:
            Number of vehicles cleared this frame (for metrics counter).
        """
        if dt is None:
            now = time.time()
            dt  = min(now - self._last_update, 0.1)   # cap at 100ms (pause recovery)
            self._last_update = now

        cleared_this_frame = 0

        for arm in ARM_NAMES:
            queue     = self.queues[arm]
            arm_state = arm_states.get(arm)
            if arm_state is None:
                continue

            # Read density and signal — support both ArmState objects and dicts
            density  = _read_attr(arm_state, 'density',  0.0)
            signal   = _read_attr(arm_state, 'signal',   'RED')
            is_green = (signal == 'GREEN')

            before = len(queue)
            queue.update_spawn(density, dt)
            queue.update_movement(is_green, dt)
            after = len(queue)

            cleared = max(0, before - after)
            cleared_this_frame += cleared

        self._cleared_total += cleared_this_frame
        return cleared_this_frame

    def all_vehicles(self) -> list[Vehicle]:
        """Return a flat list of all vehicles across all arms."""
        vehicles: list[Vehicle] = []
        for q in self.queues.values():
            vehicles.extend(q.vehicles)
        return vehicles

    def vehicle_count_by_arm(self) -> dict[str, int]:
        """Return vehicle count per arm."""
        return {arm: len(q) for arm, q in self.queues.items()}

    def queued_count_by_arm(self) -> dict[str, int]:
        """Return stopped/queued vehicle count per arm."""
        return {arm: q.queue_length() for arm, q in self.queues.items()}

    def total_cleared(self) -> int:
        """Cumulative vehicles cleared since creation."""
        return self._cleared_total

    def reset(self) -> None:
        """Remove all vehicles from all arms."""
        for q in self.queues.values():
            q.vehicles.clear()
            q._spawn_accumulator = 0.0
        self._cleared_total = 0

    def __repr__(self) -> str:
        counts = {arm: len(q) for arm, q in self.queues.items()}
        return f"VehicleManager(vehicles={counts} cleared={self._cleared_total})"


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def _density_to_spawn_rate(density: float) -> float:
    """
    Map PCU density to vehicles-per-second spawn rate.
    Uses the table from the spec; linear interpolation between buckets.
    """
    if density <= 0:
        return 0.0
    for threshold, rate in _SPAWN_RATE_TABLE:
        if density <= threshold:
            return rate
    return _SPAWN_RATE_TABLE[-1][1]   # cap at max rate


def _random_vehicle_class() -> str:
    """
    Sample a vehicle class according to VEHICLE_SPAWN_WEIGHTS.
    Returns 'car' as a safe fallback if weights are misconfigured.
    """
    classes = list(VEHICLE_SPAWN_WEIGHTS.keys())
    weights = list(VEHICLE_SPAWN_WEIGHTS.values())
    try:
        return random.choices(classes, weights=weights, k=1)[0]
    except (ValueError, IndexError):
        return 'car'


def _lerp(current: float, target: float, alpha: float) -> float:
    """Linear interpolation for smooth speed transitions."""
    return current + (target - current) * alpha


def _read_attr(obj, attr: str, default):
    """Read attribute from object or key from dict — supports both."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ---------------------------------------------------------------------------
# Standalone test  (python -m simulation.vehicles)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    try:
        import pygame
    except ImportError:
        print("pygame not installed: pip install pygame")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
    pygame.display.set_caption("Vehicle Manager Test")
    clock = pygame.time.Clock()

    manager = VehicleManager()

    # Mock arm states: North heavy traffic GREEN, others RED with various densities
    mock_states = {
        'North': {'density': 25.0, 'signal': 'GREEN'},
        'South': {'density': 8.0,  'signal': 'RED'},
        'East':  {'density': 14.0, 'signal': 'RED'},
        'West':  {'density': 3.0,  'signal': 'RED'},
    }

    # Road colours
    C_BG     = (40,  44,  52)
    C_ROAD   = (70,  75,  85)
    C_LINE   = (200, 200, 60)
    C_ISECT  = (55,  60,  70)
    C_STOP_R = (220, 30,  30)
    C_STOP_G = (30,  220, 80)
    FONT     = pygame.font.SysFont('monospace', 13)

    frame  = 0
    cycle  = 0

    print("Vehicle manager test running. Close window to quit.\n")

    while True:
        dt = clock.tick(SIM_FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit(0)

        # Cycle green arm every 5 seconds for demo
        frame += 1
        if frame % (SIM_FPS * 5) == 0:
            arms = ARM_NAMES
            cycle = (cycle + 1) % len(arms)
            for arm in ARM_NAMES:
                mock_states[arm]['signal'] = 'GREEN' if arm == arms[cycle] else 'RED'

        manager.update(mock_states, dt)

        # ── Draw ────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Road arms
        road_w = 80
        cx, cy = ISECT_CX, ISECT_CY
        pygame.draw.rect(screen, C_ROAD, (cx - road_w//2, 0,        road_w, SIM_HEIGHT))
        pygame.draw.rect(screen, C_ROAD, (0,              cy - road_w//2, SIM_WIDTH, road_w))

        # Intersection box
        pygame.draw.rect(screen, C_ISECT,
                         (ISECT_LEFT, ISECT_TOP,
                          ISECT_RIGHT - ISECT_LEFT, ISECT_BOTTOM - ISECT_TOP))

        # Stop lines
        for arm in ARM_NAMES:
            sig   = mock_states[arm]['signal']
            color = C_STOP_G if sig == 'GREEN' else C_STOP_R
            stop  = STOP_LINES[arm]
            if arm in ('North', 'South'):
                pygame.draw.line(screen, color,
                                 (cx - road_w//2, stop), (cx + road_w//2, stop), 3)
            else:
                pygame.draw.line(screen, color,
                                 (stop, cy - road_w//2), (stop, cy + road_w//2), 3)

        # Vehicles
        for v in manager.all_vehicles():
            r = v.rect
            pygame.draw.rect(screen, v.color, r, border_radius=2)
            # Dark outline
            pygame.draw.rect(screen, (0, 0, 0), r, 1, border_radius=2)

        # HUD
        counts = manager.vehicle_count_by_arm()
        queued = manager.queued_count_by_arm()
        y_off  = 8
        for arm in ARM_NAMES:
            sig = mock_states[arm]['signal']
            sc  = C_STOP_G if sig == 'GREEN' else C_STOP_R
            txt = FONT.render(
                f"{arm:<6} sig={sig:<6} total={counts[arm]} queued={queued[arm]}",
                True, sc,
            )
            screen.blit(txt, (8, y_off))
            y_off += 16

        cleared_txt = FONT.render(
            f"Cleared: {manager.total_cleared()}  Frame: {frame}", True, (200, 200, 200)
        )
        screen.blit(cleared_txt, (8, SIM_HEIGHT - 20))

        pygame.display.flip()

    pygame.quit()