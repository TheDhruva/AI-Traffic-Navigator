# simulation/pygame_sim.py — Upgraded Interactive Intersection Simulation
# =========================================================================
# Upgrades over base version:
#   • Interactive control panel strip (bottom of sim panel) — clickable spawn buttons
#   • Manual spawn: Car / Bus / Auto / Bike / Pedestrian / Animal / Ambulance
#   • Arm selector (N/S/E/W) — spawn targets the chosen arm
#   • Auto-spawn mode — vehicles appear automatically at configurable rate
#   • Zoom (mouse scroll + Z/X keys) with pan (middle-drag)
#   • /tmp/control.json polling — live spawn_rate / sim_speed from Streamlit sidebar
#   • /tmp/spawn.json polling  — one-shot spawns triggered from Streamlit sidebar
#   • Zebra crossings, direction arrows, improved road markings
#   • Vehicle count badges per arm entrance
#   • New keyboard shortcuts: C/B/A/M/G/V/U + Z/X zoom + TAB cycle arm

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("pygame not installed: pip install pygame")
    sys.exit(1)

from config import (
    ARM_NAMES,
    SIM_FPS,
    SIM_COLOR_GREEN,
    SIM_COLOR_YELLOW,
    SIM_COLOR_RED,
    VEHICLE_COLORS,
)
from controller.state import IntersectionState
from simulation.vehicles import (
    Vehicle,
    VehicleManager,
    ISECT_LEFT, ISECT_RIGHT, ISECT_TOP, ISECT_BOTTOM,
    ISECT_CX, ISECT_CY,
    STOP_LINES, SPAWN_POSITIONS,
    MAX_VEHICLES_PER_ARM,
    _random_vehicle_class,
)

logger = logging.getLogger(__name__)

# ── IPC paths ─────────────────────────────────────────────────────────────────
_CONTROL_FILE = Path("/tmp/control.json")
_SPAWN_FILE   = Path("/tmp/spawn.json")

# ── Window layout ─────────────────────────────────────────────────────────────
WIN_W          = 1280
WIN_H          = 800          # +80px vs base — space for control panel
SIM_PANEL_W    = 720
SIM_PANEL_H    = WIN_H        # full height
CTRL_PANEL_H   = 90           # control panel strip height (inside sim panel, bottom)
SIM_DRAW_H     = WIN_H - CTRL_PANEL_H   # drawable intersection area
CAM_PANEL_X    = SIM_PANEL_W
CAM_PANEL_W    = WIN_W - SIM_PANEL_W
CAM_PANEL_H    = WIN_H

ROAD_WIDTH     = 80
SIM_CX         = SIM_PANEL_W  // 2      # 360
SIM_CY         = SIM_DRAW_H   // 2      # 355

# Coordinate scalers (intersection → screen)
def _sx(x: float) -> int: return int((x / ISECT_CX) * SIM_CX)
def _sy(y: float) -> int: return int((y / ISECT_CY) * SIM_CY)
def _sv(v: float) -> int: return int(v * (SIM_CX / ISECT_CX))

# Signal light positions
_SIG_R = 9
_SIG_POS: dict[str, tuple[int, int]] = {
    'North': (SIM_CX - ROAD_WIDTH // 2 - 18, _sy(ISECT_TOP)    - 22),
    'South': (SIM_CX + ROAD_WIDTH // 2 + 18, _sy(ISECT_BOTTOM) + 22),
    'East':  (_sx(ISECT_RIGHT)  + 22, SIM_CY - ROAD_WIDTH // 2 - 18),
    'West':  (_sx(ISECT_LEFT)   - 22, SIM_CY + ROAD_WIDTH // 2 + 18),
}

# Arm label positions (for vehicle count badge)
_ARM_LABEL_POS: dict[str, tuple[int, int]] = {
    'North': (SIM_CX - ROAD_WIDTH // 2 - 38, _sy(ISECT_TOP)    - 40),
    'South': (SIM_CX - ROAD_WIDTH // 2 - 38, _sy(ISECT_BOTTOM) + 26),
    'East':  (_sx(ISECT_RIGHT)  + 10, SIM_CY + ROAD_WIDTH // 2 + 8),
    'West':  (_sx(ISECT_LEFT)   - 90, SIM_CY + ROAD_WIDTH // 2 + 8),
}

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG         = ( 28,  32,  38)
C_ROAD       = ( 55,  60,  70)
C_ROAD_EDGE  = ( 42,  47,  56)
C_ISECT      = ( 48,  53,  63)
C_LANE_MARK  = (170, 160,  50)
C_ZEBRA_W    = (210, 210, 210)
C_ZEBRA_G    = ( 60,  65,  75)
C_KERB       = ( 88,  94, 106)
C_PAVEMENT   = ( 72,  78,  88)
C_GRASS      = ( 34,  52,  34)
C_PANEL_DIV  = ( 45,  50,  60)
C_TEXT_MAIN  = (230, 235, 245)
C_TEXT_DIM   = (120, 130, 148)
C_TEXT_GREEN = ( 55, 215,  85)
C_TEXT_YEL   = (240, 200,  40)
C_TEXT_RED   = (215,  55,  55)
C_TEXT_CYAN  = ( 55, 215, 215)
C_HBAR_BG    = ( 40,  45,  55)
C_CTRL_BG    = ( 18,  21,  28)
C_BTN_IDLE   = ( 42,  48,  62)
C_BTN_HOVER  = ( 62,  70,  90)
C_BTN_ACTIVE = ( 30, 130,  70)
C_BTN_EMRG   = (140,  25,  25)
C_BTN_PED    = ( 20,  80, 140)
C_BTN_ANIMAL = (110,  75,  20)
C_BTN_SEL    = ( 30, 100, 170)
C_EMRG_BG   = (150,  18,  18)
C_PED_BG    = ( 18,  80, 140)
C_HZRD_BG   = (140,  90,  16)

_PHASE_COLORS = {
    'normal':     C_TEXT_GREEN,
    'emergency':  C_TEXT_RED,
    'pedestrian': C_TEXT_CYAN,
    'all_red':    C_TEXT_RED,
    'yellow':     C_TEXT_YEL,
    'startup':    C_TEXT_DIM,
}

# ── Spawn button definitions ───────────────────────────────────────────────────
# Each dict: label, vehicle class (or special action), key shortcut, colour
_SPAWN_BTNS = [
    {'label': 'CAR',    'cls': 'car',        'key': 'C', 'color': C_BTN_IDLE},
    {'label': 'BUS',    'cls': 'bus',        'key': 'B', 'color': C_BTN_IDLE},
    {'label': 'AUTO',   'cls': 'auto',       'key': 'A', 'color': C_BTN_IDLE},
    {'label': 'BIKE',   'cls': 'motorcycle', 'key': 'M', 'color': C_BTN_IDLE},
    {'label': 'PED',    'cls': 'ped',        'key': 'G', 'color': C_BTN_PED},
    {'label': 'ANIMAL', 'cls': 'animal',     'key': 'V', 'color': C_BTN_ANIMAL},
    {'label': 'AMBUL',  'cls': 'ambulance',  'key': 'U', 'color': C_BTN_EMRG},
]

_ARM_BTNS = ['North', 'South', 'East', 'West']

# Button geometry (within the control panel strip)
_BTN_W    = 76
_BTN_H    = 30
_BTN_GAP  = 6
_ARM_BTN_W = 52
_ARM_BTN_H = 24


# ═════════════════════════════════════════════════════════════════════════════
class PygameSimulation:
    """
    Upgraded Pygame simulation with interactive control panel and zoom.

    Keyboard shortcuts:
        Q / ESC   → quit
        E         → emergency override (North)
        P         → pedestrian rush
        C/B/A/M   → spawn car / bus / auto / bike on selected arm
        G         → spawn pedestrian rush
        V         → animal hazard on selected arm
        U         → ambulance override on selected arm
        TAB       → cycle selected arm (N → S → E → W)
        1/2/3/4   → force green arm (debug)
        R         → reset wait times
        D         → toggle debug overlay
        S         → save screenshot
        SPACE     → pause / resume
        Z / +     → zoom in
        X / -     → zoom out
        0         → reset zoom & pan
    """

    def __init__(
        self,
        state: IntersectionState,
        emergency_detector=None,
        title: str = "AI Smart Traffic System — Indian Cities",
    ) -> None:
        self._state    = state
        self._emrg_det = emergency_detector
        self._title    = title

        self._screen:  Optional[pygame.Surface] = None
        self._clock:   Optional[pygame.time.Clock] = None
        self._fonts:   dict[str, pygame.font.Font] = {}

        self._vehicles = VehicleManager()

        # ── Interaction state ──────────────────────────────────────────────
        self._selected_arm   = 'North'          # current spawn target arm
        self._auto_spawn     = False            # auto-spawn toggle
        self._auto_timer     = 0.0              # accumulator for auto-spawn
        self._spawn_rate     = 1.0              # vehicles/sec (overridden by control.json)
        self._sim_speed      = 1.0              # dt multiplier (overridden by control.json)
        self._debug_overlay  = False
        self._paused         = False
        self._running        = True
        self._screenshot_n   = 0
        self._frame_count    = 0

        # ── Zoom / pan ────────────────────────────────────────────────────
        self._zoom            = 1.0             # 0.5 → 2.0
        self._pan_x           = 0.0             # pixel offset of sim surface
        self._pan_y           = 0.0
        self._drag_active     = False
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_pan_start: Optional[tuple[float, float]] = None

        # ── Control panel button rects (computed in _init_pygame) ─────────
        self._spawn_btn_rects: list[pygame.Rect] = []
        self._arm_btn_rects:   list[pygame.Rect] = []
        self._auto_btn_rect:   Optional[pygame.Rect] = None
        self._zoom_in_rect:    Optional[pygame.Rect] = None
        self._zoom_out_rect:   Optional[pygame.Rect] = None
        self._reset_zoom_rect: Optional[pygame.Rect] = None

        # ── FPS tracking ──────────────────────────────────────────────────
        self._fps_samples:   list[float] = []
        self._last_fps_time: float = time.time()
        self._display_fps:   float = 0.0

        # ── Hover state ───────────────────────────────────────────────────
        self._hover_btn: Optional[int]  = None   # index into _spawn_btn_rects
        self._hover_arm: Optional[int]  = None   # index into _arm_btn_rects

        # ── Sim surface (zoom target) ─────────────────────────────────────
        self._sim_surf: Optional[pygame.Surface] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._init_pygame()
        logger.info("Pygame simulation started — %dx%d @ %d FPS", WIN_W, WIN_H, SIM_FPS)

        while self._running:
            dt = self._clock.tick(SIM_FPS) / 1000.0
            dt = min(dt, 0.1) * self._sim_speed

            self._handle_events()

            if not self._paused:
                self._update(dt)

            self._draw()
            pygame.display.flip()
            self._frame_count += 1

        pygame.quit()
        logger.info("Pygame simulation stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_caption(self._title)
        self._screen   = pygame.display.set_mode((WIN_W, WIN_H))
        self._sim_surf = pygame.Surface((SIM_PANEL_W, SIM_DRAW_H))
        self._clock    = pygame.time.Clock()

        self._fonts = {
            'sm':    pygame.font.SysFont('monospace', 11),
            'md':    pygame.font.SysFont('monospace', 13),
            'lg':    pygame.font.SysFont('monospace', 16),
            'xl':    pygame.font.SysFont('monospace', 21, bold=True),
            'title': pygame.font.SysFont('monospace', 10),
            'btn':   pygame.font.SysFont('monospace', 11, bold=True),
        }

        self._build_button_rects()

    def _build_button_rects(self) -> None:
        """Compute all clickable rects for the control panel."""
        panel_y = SIM_DRAW_H    # y of control panel top inside window

        # ── Row 1: Arm selector (N S E W) at top of panel ─────────────────
        arm_row_y = panel_y + 8
        arm_total_w = len(_ARM_BTNS) * _ARM_BTN_W + (len(_ARM_BTNS) - 1) * 4
        arm_start_x = 8

        self._arm_btn_rects = []
        for i in range(len(_ARM_BTNS)):
            x = arm_start_x + i * (_ARM_BTN_W + 4)
            self._arm_btn_rects.append(
                pygame.Rect(x, arm_row_y, _ARM_BTN_W, _ARM_BTN_H)
            )

        # ── Row 1 right side: AUTO + ZOOM controls ─────────────────────────
        ctrl_x = arm_start_x + arm_total_w + 16

        self._auto_btn_rect = pygame.Rect(ctrl_x, arm_row_y, 80, _ARM_BTN_H)
        ctrl_x += 86
        self._zoom_in_rect  = pygame.Rect(ctrl_x,      arm_row_y, 34, _ARM_BTN_H)
        self._zoom_out_rect = pygame.Rect(ctrl_x + 38, arm_row_y, 34, _ARM_BTN_H)
        self._reset_zoom_rect = pygame.Rect(ctrl_x + 76, arm_row_y, 34, _ARM_BTN_H)

        # ── Row 2: Spawn buttons ───────────────────────────────────────────
        btn_row_y = panel_y + _ARM_BTN_H + 14
        self._spawn_btn_rects = []
        for i in range(len(_SPAWN_BTNS)):
            x = 8 + i * (_BTN_W + _BTN_GAP)
            self._spawn_btn_rects.append(
                pygame.Rect(x, btn_row_y, _BTN_W, _BTN_H)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Event handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self._hover_btn = None
        self._hover_arm = None

        # Update hover state
        for i, r in enumerate(self._spawn_btn_rects):
            if r.collidepoint(mouse_pos):
                self._hover_btn = i
        for i, r in enumerate(self._arm_btn_rects):
            if r.collidepoint(mouse_pos):
                self._hover_arm = i

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event.button, event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:   # middle click release
                    self._drag_active = False

            elif event.type == pygame.MOUSEMOTION:
                if self._drag_active and self._drag_start:
                    dx = event.pos[0] - self._drag_start[0]
                    dy = event.pos[1] - self._drag_start[1]
                    self._pan_x = self._drag_pan_start[0] + dx
                    self._pan_y = self._drag_pan_start[1] + dy
                    self._clamp_pan()

    def _handle_mouse_down(self, button: int, pos: tuple[int, int]) -> None:
        # ── Scroll wheel zoom ─────────────────────────────────────────────
        if button == 4:   # scroll up → zoom in
            self._adjust_zoom(+0.1, pos)
            return
        if button == 5:   # scroll down → zoom out
            self._adjust_zoom(-0.1, pos)
            return

        # ── Middle drag (pan) ─────────────────────────────────────────────
        if button == 2:
            self._drag_active    = True
            self._drag_start     = pos
            self._drag_pan_start = (self._pan_x, self._pan_y)
            return

        if button != 1:   # only left click from here
            return

        # ── Spawn buttons ─────────────────────────────────────────────────
        for i, r in enumerate(self._spawn_btn_rects):
            if r.collidepoint(pos):
                self._trigger_spawn(_SPAWN_BTNS[i]['cls'])
                return

        # ── Arm selector ──────────────────────────────────────────────────
        for i, r in enumerate(self._arm_btn_rects):
            if r.collidepoint(pos):
                self._selected_arm = _ARM_BTNS[i]
                return

        # ── Auto spawn toggle ─────────────────────────────────────────────
        if self._auto_btn_rect and self._auto_btn_rect.collidepoint(pos):
            self._auto_spawn = not self._auto_spawn
            logger.info("[CTRL] Auto-spawn: %s", "ON" if self._auto_spawn else "OFF")
            return

        # ── Zoom buttons ──────────────────────────────────────────────────
        if self._zoom_in_rect and self._zoom_in_rect.collidepoint(pos):
            self._adjust_zoom(+0.2, (SIM_PANEL_W // 2, SIM_DRAW_H // 2))
        elif self._zoom_out_rect and self._zoom_out_rect.collidepoint(pos):
            self._adjust_zoom(-0.2, (SIM_PANEL_W // 2, SIM_DRAW_H // 2))
        elif self._reset_zoom_rect and self._reset_zoom_rect.collidepoint(pos):
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self._quit()
        elif key == pygame.K_e:
            self._trigger_spawn('ambulance', arm='North')
        elif key == pygame.K_p:
            self._trigger_spawn('ped')
        elif key == pygame.K_c:
            self._trigger_spawn('car')
        elif key == pygame.K_b:
            self._trigger_spawn('bus')
        elif key == pygame.K_a:
            self._trigger_spawn('auto')
        elif key == pygame.K_m:
            self._trigger_spawn('motorcycle')
        elif key == pygame.K_g:
            self._trigger_spawn('ped')
        elif key == pygame.K_v:
            self._trigger_spawn('animal')
        elif key == pygame.K_u:
            self._trigger_spawn('ambulance')
        elif key == pygame.K_TAB:
            idx = _ARM_BTNS.index(self._selected_arm)
            self._selected_arm = _ARM_BTNS[(idx + 1) % len(_ARM_BTNS)]
        elif key == pygame.K_d:
            self._debug_overlay = not self._debug_overlay
        elif key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_r:
            with self._state.lock:
                for arm in self._state.arms.values():
                    arm.wait_time = 0.0
        elif key == pygame.K_s:
            self._save_screenshot()
        elif key in (pygame.K_z, pygame.K_PLUS, pygame.K_EQUALS):
            self._adjust_zoom(+0.15, (SIM_PANEL_W // 2, SIM_DRAW_H // 2))
        elif key in (pygame.K_x, pygame.K_MINUS):
            self._adjust_zoom(-0.15, (SIM_PANEL_W // 2, SIM_DRAW_H // 2))
        elif key == pygame.K_0:
            self._zoom = 1.0; self._pan_x = 0.0; self._pan_y = 0.0
        elif key == pygame.K_1:
            self._force_green('North')
        elif key == pygame.K_2:
            self._force_green('South')
        elif key == pygame.K_3:
            self._force_green('East')
        elif key == pygame.K_4:
            self._force_green('West')

    def _trigger_spawn(self, cls: str, arm: Optional[str] = None) -> None:
        """Route a spawn action to the right handler."""
        target = arm or self._selected_arm

        if cls == 'ped':
            if self._emrg_det:
                self._emrg_det.simulate_ped_rush()
            logger.info("[SPAWN] Pedestrian rush triggered")

        elif cls == 'ambulance':
            if self._emrg_det:
                self._emrg_det.simulate_emergency(target)
            logger.info("[SPAWN] Emergency on %s", target)

        elif cls == 'animal':
            # Set hazard flag on arm state directly (cleared by EmergencyDetector countdown)
            try:
                with self._state.lock:
                    self._state.arms[target].hazard = True
            except (AttributeError, KeyError):
                pass
            logger.info("[SPAWN] Animal hazard on %s", target)

        else:
            # Normal vehicle — inject directly into queue
            self._manual_spawn(cls, target)

    def _manual_spawn(self, cls: str, arm: str) -> None:
        """Create one vehicle of a specific class at the arm spawn point."""
        queue = self._vehicles.queues.get(arm)
        if queue is None or len(queue.vehicles) >= MAX_VEHICLES_PER_ARM:
            return

        sx, sy = SPAWN_POSITIONS[arm]
        jitter = random.uniform(-5, 5)
        if arm in ('North', 'South'):
            sx += jitter
        else:
            sy += jitter

        v = Vehicle(arm=arm, cls=cls, x=float(sx), y=float(sy))
        queue.vehicles.append(v)
        logger.debug("[SPAWN] %s on %s arm", cls, arm)

    def _force_green(self, arm: str) -> None:
        with self._state.lock:
            self._state.set_signal(None, 'RED')
            self._state.set_signal(arm, 'GREEN')
            self._state.current_green = arm
            self._state.phase = 'normal'

    def _quit(self) -> None:
        self._running = False
        with self._state.lock:
            self._state.running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Zoom helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _adjust_zoom(self, delta: float, anchor: tuple[int, int]) -> None:
        """Zoom toward/away from anchor point (screen coords in sim panel)."""
        old_zoom = self._zoom
        self._zoom = max(0.5, min(2.5, self._zoom + delta))

        # Adjust pan so anchor pixel stays fixed
        ax, ay = anchor
        ratio   = self._zoom / old_zoom
        self._pan_x = ax - ratio * (ax - self._pan_x)
        self._pan_y = ay - ratio * (ay - self._pan_y)
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        """Prevent panning the sim surface entirely off-screen."""
        scaled_w = SIM_PANEL_W * self._zoom
        scaled_h = SIM_DRAW_H  * self._zoom
        self._pan_x = max(min(self._pan_x, SIM_PANEL_W * 0.5),
                          SIM_PANEL_W - scaled_w - SIM_PANEL_W * 0.5)
        self._pan_y = max(min(self._pan_y, SIM_DRAW_H  * 0.5),
                          SIM_DRAW_H  - scaled_h - SIM_DRAW_H  * 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        arm_snapshot = self._state.snapshot_arms()
        self._vehicles.update(arm_snapshot, dt)

        # Auto-spawn
        if self._auto_spawn:
            self._auto_timer += dt
            interval = 1.0 / max(self._spawn_rate, 0.1)
            while self._auto_timer >= interval:
                self._auto_timer -= interval
                arm = random.choice(ARM_NAMES)
                cls = _random_vehicle_class()
                self._manual_spawn(cls, arm)

        # File polling (every 60 frames ≈ 2s)
        if self._frame_count % 60 == 0:
            self._poll_control_file()

        # Spawn command from Streamlit sidebar (every frame — one-shot file)
        self._poll_spawn_file()

        # FPS
        now = time.time()
        self._fps_samples.append(1.0 / max(dt / max(self._sim_speed, 0.01), 0.001))
        if now - self._last_fps_time >= 1.0:
            self._display_fps = (
                sum(self._fps_samples) / max(len(self._fps_samples), 1)
            )
            self._fps_samples.clear()
            self._last_fps_time = now

    def _poll_control_file(self) -> None:
        try:
            ctrl = json.loads(_CONTROL_FILE.read_text())
            self._spawn_rate = float(ctrl.get('spawn_rate', 1.0))
            self._sim_speed  = float(ctrl.get('sim_speed',  1.0))
        except Exception:
            pass

    def _poll_spawn_file(self) -> None:
        if not _SPAWN_FILE.exists():
            return
        try:
            cmd = json.loads(_SPAWN_FILE.read_text())
            _SPAWN_FILE.unlink(missing_ok=True)
            cls = cmd.get('type', 'car')
            arm = cmd.get('arm', self._selected_arm)
            self._trigger_spawn(cls, arm)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Draw — top level
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._screen.fill(C_BG)

        # Render intersection to _sim_surf, then zoom+blit to screen
        self._sim_surf.fill(C_BG)
        self._draw_intersection(self._sim_surf)

        if self._zoom == 1.0 and self._pan_x == 0.0 and self._pan_y == 0.0:
            self._screen.blit(self._sim_surf, (0, 0))
        else:
            scaled_w = int(SIM_PANEL_W * self._zoom)
            scaled_h = int(SIM_DRAW_H  * self._zoom)
            scaled   = pygame.transform.scale(self._sim_surf, (scaled_w, scaled_h))
            # Clip to sim panel area
            clip_rect = pygame.Rect(0, 0, SIM_PANEL_W, SIM_DRAW_H)
            self._screen.set_clip(clip_rect)
            self._screen.blit(scaled, (int(self._pan_x), int(self._pan_y)))
            self._screen.set_clip(None)

        # Control panel (never zoomed — stays anchored to bottom)
        self._draw_control_panel()

        # Camera panel (right side)
        self._draw_camera_panel()

        # Divider
        pygame.draw.line(
            self._screen, C_PANEL_DIV,
            (SIM_PANEL_W, 0), (SIM_PANEL_W, WIN_H), 2
        )

        if self._paused:
            self._draw_pause_overlay()

    # ─────────────────────────────────────────────────────────────────────────
    # Intersection drawing  (renders into _sim_surf at 1:1 scale)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_intersection(self, surf: pygame.Surface) -> None:
        # ── Grass / pavement corners ──────────────────────────────────────
        road_l = SIM_CX - ROAD_WIDTH // 2
        road_r = SIM_CX + ROAD_WIDTH // 2
        road_t = SIM_CY - ROAD_WIDTH // 2
        road_b = SIM_CY + ROAD_WIDTH // 2

        corners = [
            (0,      0,      road_l,      road_t),    # top-left
            (road_r, 0,      SIM_PANEL_W - road_r, road_t),  # top-right
            (0,      road_b, road_l,      SIM_DRAW_H - road_b),  # bot-left
            (road_r, road_b, SIM_PANEL_W - road_r, SIM_DRAW_H - road_b),  # bot-right
        ]
        for rx, ry, rw, rh in corners:
            pygame.draw.rect(surf, C_GRASS, (rx, ry, rw, rh))

        # ── Road arms ─────────────────────────────────────────────────────
        pygame.draw.rect(surf, C_ROAD, (road_l, 0, ROAD_WIDTH, SIM_DRAW_H))
        pygame.draw.rect(surf, C_ROAD, (0, road_t, SIM_PANEL_W, ROAD_WIDTH))

        # Kerb edges
        for x in (road_l, road_r):
            pygame.draw.line(surf, C_KERB, (x, 0), (x, SIM_DRAW_H), 2)
        for y in (road_t, road_b):
            pygame.draw.line(surf, C_KERB, (0, y), (SIM_PANEL_W, y), 2)

        # ── Intersection box ──────────────────────────────────────────────
        isect_x = _sx(ISECT_LEFT)
        isect_y = _sy(ISECT_TOP)
        isect_w = _sx(ISECT_RIGHT)  - isect_x
        isect_h = _sy(ISECT_BOTTOM) - isect_y
        pygame.draw.rect(surf, C_ISECT, (isect_x, isect_y, isect_w, isect_h))
        pygame.draw.rect(surf, C_KERB,  (isect_x, isect_y, isect_w, isect_h), 1)

        # ── Dashed centre lines (outside intersection) ────────────────────
        _draw_dashed_line(surf, C_LANE_MARK, (SIM_CX, 0), (SIM_CX, isect_y - 1), 12, 7)
        _draw_dashed_line(surf, C_LANE_MARK, (SIM_CX, isect_y + isect_h + 1), (SIM_CX, SIM_DRAW_H), 12, 7)
        _draw_dashed_line(surf, C_LANE_MARK, (0, SIM_CY), (isect_x - 1, SIM_CY), 12, 7)
        _draw_dashed_line(surf, C_LANE_MARK, (isect_x + isect_w + 1, SIM_CY), (SIM_PANEL_W, SIM_CY), 12, 7)

        # ── Zebra crossings ───────────────────────────────────────────────
        self._draw_zebra_crossings(surf)

        # ── Direction arrows ──────────────────────────────────────────────
        self._draw_direction_arrows(surf)

        # ── Stop lines + signal colours ───────────────────────────────────
        with self._state.lock:
            signals = {name: arm.signal for name, arm in self._state.arms.items()}

        for arm, sig in signals.items():
            color = (SIM_COLOR_GREEN if sig == 'GREEN' else
                     SIM_COLOR_YELLOW if sig == 'YELLOW' else SIM_COLOR_RED)
            stop  = STOP_LINES[arm]
            if arm in ('North', 'South'):
                sy_ = _sy(stop)
                pygame.draw.line(surf, color,
                                 (road_l, sy_), (road_r, sy_), 3)
            else:
                sx_ = _sx(stop)
                pygame.draw.line(surf, color,
                                 (sx_, road_t), (sx_, road_b), 3)

        # ── Vehicles ──────────────────────────────────────────────────────
        for v in self._vehicles.all_vehicles():
            rx, ry, rw, rh = v.rect
            vr = pygame.Rect(_sx(rx), _sy(ry), max(4, _sv(rw)), max(6, _sv(rh)))
            pygame.draw.rect(surf, v.color, vr, border_radius=2)
            pygame.draw.rect(surf, (0, 0, 0), vr, 1, border_radius=2)

        # ── Signal lights ─────────────────────────────────────────────────
        self._draw_signal_lights(surf, signals)

        # ── Vehicle count badges ──────────────────────────────────────────
        self._draw_arm_badges(surf)

        # ── HUD ───────────────────────────────────────────────────────────
        self._draw_sim_hud(surf)

        # ── Alert banner (top strip of sim panel) ─────────────────────────
        self._draw_alert_banner(surf)

        if self._debug_overlay:
            self._draw_debug_info(surf)

    def _draw_zebra_crossings(self, surf: pygame.Surface) -> None:
        """Striped pedestrian crossings at each arm entrance."""
        road_l = SIM_CX - ROAD_WIDTH // 2
        road_r = SIM_CX + ROAD_WIDTH // 2
        road_t = SIM_CY - ROAD_WIDTH // 2
        road_b = SIM_CY + ROAD_WIDTH // 2
        isect_y_top = _sy(ISECT_TOP)
        isect_y_bot = _sy(ISECT_BOTTOM)
        isect_x_l   = _sx(ISECT_LEFT)
        isect_x_r   = _sx(ISECT_RIGHT)

        STRIPE = 5
        GAP    = 5
        DEPTH  = 18   # crossing depth perpendicular to road

        # North crossing (horizontal stripes just above intersection)
        y0 = isect_y_top - DEPTH
        x  = road_l
        while x < road_r:
            pygame.draw.rect(surf, C_ZEBRA_W, (x, y0, STRIPE, DEPTH))
            x += STRIPE + GAP

        # South crossing
        y0 = isect_y_bot
        x  = road_l
        while x < road_r:
            pygame.draw.rect(surf, C_ZEBRA_W, (x, y0, STRIPE, DEPTH))
            x += STRIPE + GAP

        # East crossing (vertical stripes just right of intersection)
        x0 = isect_x_r
        y  = road_t
        while y < road_b:
            pygame.draw.rect(surf, C_ZEBRA_W, (x0, y, DEPTH, STRIPE))
            y += STRIPE + GAP

        # West crossing
        x0 = isect_x_l - DEPTH
        y  = road_t
        while y < road_b:
            pygame.draw.rect(surf, C_ZEBRA_W, (x0, y, DEPTH, STRIPE))
            y += STRIPE + GAP

    def _draw_direction_arrows(self, surf: pygame.Surface) -> None:
        """Small direction arrows on each arm approach road."""
        road_l = SIM_CX - ROAD_WIDTH // 2
        road_r = SIM_CX + ROAD_WIDTH // 2
        road_t = SIM_CY - ROAD_WIDTH // 2
        road_b = SIM_CY + ROAD_WIDTH // 2
        isect_y_top = _sy(ISECT_TOP)
        isect_y_bot = _sy(ISECT_BOTTOM)
        isect_x_l   = _sx(ISECT_LEFT)
        isect_x_r   = _sx(ISECT_RIGHT)

        C_ARROW = (100, 105, 120)

        # North arm — arrow pointing down ↓
        ax, ay = SIM_CX, isect_y_top - 50
        _draw_arrow(surf, C_ARROW, (ax, ay - 12), (ax, ay + 12), 8)

        # South arm — arrow pointing up ↑
        ax, ay = SIM_CX, isect_y_bot + 50
        _draw_arrow(surf, C_ARROW, (ax, ay + 12), (ax, ay - 12), 8)

        # East arm — arrow pointing left ←
        ax, ay = isect_x_r + 50, SIM_CY
        _draw_arrow(surf, C_ARROW, (ax + 12, ay), (ax - 12, ay), 8)

        # West arm — arrow pointing right →
        ax, ay = isect_x_l - 50, SIM_CY
        _draw_arrow(surf, C_ARROW, (ax - 12, ay), (ax + 12, ay), 8)

    def _draw_signal_lights(
        self, surf: pygame.Surface, signals: dict[str, str]
    ) -> None:
        for arm in ARM_NAMES:
            sig     = signals.get(arm, 'RED')
            cx, cy  = _SIG_POS[arm]
            is_ns   = arm in ('North', 'South')

            hw = _SIG_R * 2 + 8
            hh = _SIG_R * 6 + 12
            if not is_ns:
                hw, hh = hh, hw

            pygame.draw.rect(surf, (18, 20, 25),
                             (cx - hw // 2, cy - hh // 2, hw, hh), border_radius=4)
            pygame.draw.rect(surf, (40, 44, 52),
                             (cx - hw // 2, cy - hh // 2, hw, hh), 1, border_radius=4)

            if is_ns:
                lamp_pos = [
                    (cx, cy - _SIG_R * 2),
                    (cx, cy),
                    (cx, cy + _SIG_R * 2),
                ]
            else:
                lamp_pos = [
                    (cx - _SIG_R * 2, cy),
                    (cx, cy),
                    (cx + _SIG_R * 2, cy),
                ]

            lamp_defs = [('RED', SIM_COLOR_RED), ('YELLOW', SIM_COLOR_YELLOW), ('GREEN', SIM_COLOR_GREEN)]
            for (lx, ly), (phase, on_color) in zip(lamp_pos, lamp_defs):
                active = (sig == phase) or (sig == 'WALK' and phase == 'RED')
                color  = on_color if active else (30, 30, 30)
                pygame.draw.circle(surf, color, (lx, ly), _SIG_R)
                pygame.draw.circle(surf, (0, 0, 0), (lx, ly), _SIG_R, 1)

    def _draw_arm_badges(self, surf: pygame.Surface) -> None:
        """Vehicle count badge at each arm entrance."""
        counts = self._vehicles.vehicle_count_by_arm()
        for arm in ARM_NAMES:
            n   = counts.get(arm, 0)
            sel = (arm == self._selected_arm)
            lx, ly = _ARM_LABEL_POS[arm]

            bg = C_BTN_SEL if sel else (30, 35, 45)
            text = f"{arm[0]}:{n:2d}"
            lbl  = self._fonts['sm'].render(text, True, C_TEXT_MAIN if sel else C_TEXT_DIM)
            pad  = 4
            r    = pygame.Rect(lx - pad, ly - 2, lbl.get_width() + pad * 2, 16)
            pygame.draw.rect(surf, bg, r, border_radius=3)
            surf.blit(lbl, (lx, ly))

    def _draw_sim_hud(self, surf: pygame.Surface) -> None:
        """Bottom-left HUD: phase + per-arm data table."""
        phase_snap   = self._state.snapshot_phase()
        arm_snap     = self._state.snapshot_arms()
        phase        = phase_snap.get('phase', 'normal')
        current_green= phase_snap.get('current_green')
        phase_color  = _PHASE_COLORS.get(phase, C_TEXT_DIM)

        hud_x  = 6
        hud_y  = SIM_DRAW_H - 120
        hud_w  = 400
        hud_h  = 114

        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((12, 15, 20, 190))
        surf.blit(hud_surf, (hud_x, hud_y))
        pygame.draw.rect(surf, C_PANEL_DIV, (hud_x, hud_y, hud_w, hud_h), 1)

        phase_lbl = self._fonts['md'].render(
            f"PHASE: {phase.upper():<12}  cycles={phase_snap.get('total_cycles', 0)}",
            True, phase_color,
        )
        surf.blit(phase_lbl, (hud_x + 5, hud_y + 4))

        for i, arm in enumerate(ARM_NAMES):
            s      = arm_snap.get(arm, {})
            sig    = s.get('signal', 'RED')
            dens   = s.get('density', 0.0)
            wait   = s.get('wait_time', 0.0)
            emrg   = s.get('emergency', False)
            hzrd   = s.get('hazard', False)
            is_grn = (arm == current_green)

            sig_color = (C_TEXT_GREEN if sig == 'GREEN' else
                         C_TEXT_YEL   if sig == 'YELLOW' else C_TEXT_RED)
            row_color  = C_TEXT_GREEN if is_grn else C_TEXT_MAIN
            row_y      = hud_y + 22 + i * 20

            dot_col = SIM_COLOR_GREEN if is_grn else SIM_COLOR_RED
            pygame.draw.circle(surf, dot_col, (hud_x + 12, row_y + 6), 4)

            badge = self._fonts['sm'].render(f"{sig[0]}", True, sig_color)
            surf.blit(badge, (hud_x + 22, row_y))

            flags    = (" EMRG" if emrg else "") + (" HZRD" if hzrd else "")
            row_text = f"{arm:<6} d={dens:5.1f} w={int(wait):3d}s{flags}"
            row_lbl  = self._fonts['sm'].render(row_text, True, row_color)
            surf.blit(row_lbl, (hud_x + 34, row_y))

            bar_x  = hud_x + 290
            bar_w  = 35
            bar_h  = 9
            fill   = int(min(1.0, dens / 50.0) * bar_w)
            bar_c  = _density_color(dens / 50.0)
            pygame.draw.rect(surf, C_HBAR_BG, (bar_x, row_y + 1, bar_w, bar_h))
            if fill > 0:
                pygame.draw.rect(surf, bar_c, (bar_x, row_y + 1, fill, bar_h))
            pygame.draw.rect(surf, C_TEXT_DIM, (bar_x, row_y + 1, bar_w, bar_h), 1)

        fps_lbl = self._fonts['title'].render(
            f"SIM {self._display_fps:.0f}fps  zoom={self._zoom:.1f}x  "
            f"vehicles={len(self._vehicles.all_vehicles())}",
            True, C_TEXT_DIM,
        )
        surf.blit(fps_lbl, (hud_x + 5, hud_y + hud_h - 12))

    def _draw_alert_banner(self, surf: pygame.Surface) -> None:
        phase_snap   = self._state.snapshot_phase()
        arm_snap     = self._state.snapshot_arms()
        phase        = phase_snap.get('phase', 'normal')
        emrg_arm     = next((a for a, s in arm_snap.items() if s.get('emergency')), None)
        hzrd_info    = next(((a, s.get('hazard')) for a, s in arm_snap.items()
                             if s.get('hazard')), None)

        banner_h = 30
        banner   = pygame.Surface((SIM_PANEL_W, banner_h), pygame.SRCALPHA)

        if phase == 'emergency' and emrg_arm:
            banner.fill((*C_EMRG_BG, 225))
            surf.blit(banner, (0, 0))
            txt = self._fonts['lg'].render(
                f"  🚨  EMERGENCY — {emrg_arm.upper()} ARM PRIORITY",
                True, (255, 255, 255),
            )
            surf.blit(txt, (8, 6))

        elif phase == 'pedestrian':
            banner.fill((*C_PED_BG, 215))
            surf.blit(banner, (0, 0))
            avg = phase_snap.get('ped_rolling_avg', 0.0)
            txt = self._fonts['lg'].render(
                f"  🚶  PEDESTRIAN PHASE  ({avg:.0f} persons)",
                True, (255, 255, 255),
            )
            surf.blit(txt, (8, 6))

        elif hzrd_info:
            a, _ = hzrd_info
            banner.fill((*C_HZRD_BG, 215))
            surf.blit(banner, (0, 0))
            txt = self._fonts['lg'].render(
                f"  ⚠  ANIMAL ON ROAD — {a.upper()} ARM  (+5s extension)",
                True, (255, 255, 255),
            )
            surf.blit(txt, (8, 6))

    def _draw_debug_info(self, surf: pygame.Surface) -> None:
        counts = self._vehicles.vehicle_count_by_arm()
        queued = self._vehicles.queued_count_by_arm()
        y = 36
        for arm in ARM_NAMES:
            txt = self._fonts['sm'].render(
                f"[D] {arm:<6} total={counts[arm]:2d} queued={queued[arm]:2d}",
                True, (100, 200, 255),
            )
            surf.blit(txt, (SIM_PANEL_W - 200, y))
            y += 14

    # ─────────────────────────────────────────────────────────────────────────
    # Control panel (never zoomed)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_control_panel(self) -> None:
        surf  = self._screen
        py    = SIM_DRAW_H    # panel top y in window

        # Background
        pygame.draw.rect(surf, C_CTRL_BG, (0, py, SIM_PANEL_W, CTRL_PANEL_H))
        pygame.draw.line(surf, C_PANEL_DIV, (0, py), (SIM_PANEL_W, py), 1)

        # ── Row 1: Arm selector ───────────────────────────────────────────
        for i, (arm, rect) in enumerate(zip(_ARM_BTNS, self._arm_btn_rects)):
            selected = (arm == self._selected_arm)
            hovered  = (self._hover_arm == i)
            bg = C_BTN_SEL if selected else (C_BTN_HOVER if hovered else C_BTN_IDLE)
            pygame.draw.rect(surf, bg, rect, border_radius=4)
            lbl = self._fonts['btn'].render(arm[0], True,
                                            C_TEXT_MAIN if selected else C_TEXT_DIM)
            surf.blit(lbl, (rect.x + rect.w // 2 - lbl.get_width() // 2,
                            rect.y + rect.h // 2 - lbl.get_height() // 2))

        # Arm label
        lbl = self._fonts['title'].render(
            f"TARGET ARM: {self._selected_arm}  (TAB to cycle)",
            True, C_TEXT_DIM,
        )
        ax   = self._arm_btn_rects[-1].right + 8
        surf.blit(lbl, (ax, self._arm_btn_rects[0].y + 6))

        # ── Auto-spawn button ─────────────────────────────────────────────
        r  = self._auto_btn_rect
        bg = C_BTN_ACTIVE if self._auto_spawn else C_BTN_IDLE
        pygame.draw.rect(surf, bg, r, border_radius=4)
        auto_lbl = self._fonts['btn'].render(
            "AUTO ▶" if self._auto_spawn else "AUTO ■", True,
            C_TEXT_GREEN if self._auto_spawn else C_TEXT_DIM,
        )
        surf.blit(auto_lbl, (r.x + r.w // 2 - auto_lbl.get_width() // 2,
                              r.y + r.h // 2 - auto_lbl.get_height() // 2))

        # ── Zoom buttons ──────────────────────────────────────────────────
        for rect, label in (
            (self._zoom_in_rect,    "Z+"),
            (self._zoom_out_rect,   "Z-"),
            (self._reset_zoom_rect, " 1:1"),
        ):
            if rect:
                pygame.draw.rect(surf, C_BTN_IDLE, rect, border_radius=4)
                zl = self._fonts['btn'].render(label, True, C_TEXT_DIM)
                surf.blit(zl, (rect.x + rect.w // 2 - zl.get_width() // 2,
                               rect.y + rect.h // 2 - zl.get_height() // 2))

        # ── Row 2: Spawn buttons ──────────────────────────────────────────
        _BTN_COLORS = {
            'car':        C_BTN_IDLE,
            'bus':        C_BTN_IDLE,
            'auto':       C_BTN_IDLE,
            'motorcycle': C_BTN_IDLE,
            'ped':        C_BTN_PED,
            'animal':     C_BTN_ANIMAL,
            'ambulance':  C_BTN_EMRG,
        }

        for i, (btn, rect) in enumerate(zip(_SPAWN_BTNS, self._spawn_btn_rects)):
            hovered = (self._hover_btn == i)
            base    = _BTN_COLORS.get(btn['cls'], C_BTN_IDLE)
            bg      = tuple(min(255, c + 30) for c in base) if hovered else base
            pygame.draw.rect(surf, bg, rect, border_radius=5)
            pygame.draw.rect(surf, C_PANEL_DIV, rect, 1, border_radius=5)

            lbl = self._fonts['btn'].render(btn['label'], True, C_TEXT_MAIN)
            surf.blit(lbl, (rect.x + rect.w // 2 - lbl.get_width() // 2,
                            rect.y + rect.h // 2 - lbl.get_height() // 2))

            # Key hint below
            key_lbl = self._fonts['title'].render(
                f"[{btn['key']}]", True, C_TEXT_DIM
            )
            surf.blit(key_lbl, (rect.x + rect.w // 2 - key_lbl.get_width() // 2,
                                 rect.bottom + 1))

        # Spawn rate display
        rate_lbl = self._fonts['title'].render(
            f"spawn_rate={self._spawn_rate:.1f}/s  sim_speed={self._sim_speed:.1f}x  "
            f"[scroll=zoom  mid-drag=pan  0=reset]",
            True, C_TEXT_DIM,
        )
        surf.blit(rate_lbl, (8, py + CTRL_PANEL_H - 12))

    # ─────────────────────────────────────────────────────────────────────────
    # Camera panel (right side)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_camera_panel(self) -> None:
        surf  = self._screen
        frame = self._state.get_annotated_frame()

        # Title bar
        title_surf = pygame.Surface((CAM_PANEL_W, 26), pygame.SRCALPHA)
        title_surf.fill((16, 18, 24, 240))
        surf.blit(title_surf, (CAM_PANEL_X, 0))
        title_lbl = self._fonts['md'].render(
            "Live YOLO Detection Feed", True, C_TEXT_MAIN
        )
        surf.blit(title_lbl, (CAM_PANEL_X + 8, 5))
        fps_lbl = self._fonts['sm'].render(f"{self._display_fps:.0f}fps", True, C_TEXT_DIM)
        surf.blit(fps_lbl, (CAM_PANEL_X + CAM_PANEL_W - fps_lbl.get_width() - 6, 7))

        # Camera feed
        cam_h    = 400
        cam_rect = pygame.Rect(CAM_PANEL_X, 28, CAM_PANEL_W, cam_h)

        if frame is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (CAM_PANEL_W, cam_h))
                cam_surf  = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                surf.blit(cam_surf, cam_rect.topleft)
            except Exception:
                self._draw_no_camera(surf, cam_rect)
        else:
            self._draw_no_camera(surf, cam_rect)

        pygame.draw.rect(surf, C_PANEL_DIV, cam_rect, 1)
        self._draw_alert_banner_cam(surf, cam_rect)
        self._draw_metrics_sidebar(surf, cam_rect.bottom + 6)

    def _draw_no_camera(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surf, (18, 20, 26), rect)
        msg = self._fonts['md'].render("Waiting for detection feed...", True, C_TEXT_DIM)
        surf.blit(msg, (
            rect.x + rect.w // 2 - msg.get_width() // 2,
            rect.y + rect.h // 2,
        ))

    def _draw_alert_banner_cam(self, surf: pygame.Surface, cam_rect: pygame.Rect) -> None:
        phase_snap = self._state.snapshot_phase()
        arm_snap   = self._state.snapshot_arms()
        phase      = phase_snap.get('phase', 'normal')
        emrg_arm   = next((a for a, s in arm_snap.items() if s.get('emergency')), None)
        hzrd_info  = next(((a, s.get('hazard')) for a, s in arm_snap.items() if s.get('hazard')), None)

        bh      = 32
        b_rect  = pygame.Rect(cam_rect.x, cam_rect.y, cam_rect.w, bh)
        b_surf  = pygame.Surface((cam_rect.w, bh), pygame.SRCALPHA)

        if phase == 'emergency' and emrg_arm:
            b_surf.fill((*C_EMRG_BG, 228))
            surf.blit(b_surf, b_rect.topleft)
            txt = self._fonts['lg'].render(
                f"  🚨  EMERGENCY — {emrg_arm.upper()} ARM", True, (255, 255, 255)
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 7))
        elif phase == 'pedestrian':
            b_surf.fill((*C_PED_BG, 215))
            surf.blit(b_surf, b_rect.topleft)
            avg = phase_snap.get('ped_rolling_avg', 0.0)
            txt = self._fonts['lg'].render(
                f"  🚶  PEDESTRIAN PHASE  ({avg:.0f} persons)", True, (255, 255, 255)
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 7))
        elif hzrd_info:
            a, _ = hzrd_info
            b_surf.fill((*C_HZRD_BG, 215))
            surf.blit(b_surf, b_rect.topleft)
            txt = self._fonts['lg'].render(
                f"  ⚠  ANIMAL — {a.upper()} ARM", True, (255, 255, 255)
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 7))

    def _draw_metrics_sidebar(self, surf: pygame.Surface, y_start: int) -> None:
        arm_snap   = self._state.snapshot_arms()
        phase_snap = self._state.snapshot_phase()

        x0    = CAM_PANEL_X + 6
        y     = y_start
        col_w = (CAM_PANEL_W - 12) // 4

        headers = ['ARM', 'DENSITY', 'WAIT', 'SIGNAL']
        for i, h in enumerate(headers):
            lbl = self._fonts['title'].render(h, True, C_TEXT_DIM)
            surf.blit(lbl, (x0 + i * col_w, y))
        y += 14
        pygame.draw.line(surf, C_PANEL_DIV, (x0, y), (x0 + CAM_PANEL_W - 12, y), 1)
        y += 4

        for arm in ARM_NAMES:
            s      = arm_snap.get(arm, {})
            sig    = s.get('signal', 'RED')
            dens   = s.get('density', 0.0)
            wait   = s.get('wait_time', 0.0)
            is_grn = (arm == phase_snap.get('current_green'))

            sig_color = (C_TEXT_GREEN if sig == 'GREEN' else
                         C_TEXT_YEL  if sig == 'YELLOW' else C_TEXT_RED)
            row_color = C_TEXT_GREEN if is_grn else C_TEXT_MAIN

            cols   = [arm[:5], f"{dens:5.1f}", f"{int(wait):3d}s", sig]
            colors = [row_color, row_color, row_color, sig_color]

            for i, (col_txt, col_color) in enumerate(zip(cols, colors)):
                lbl = self._fonts['sm'].render(col_txt, True, col_color)
                surf.blit(lbl, (x0 + i * col_w, y))

            bar_x = x0 + col_w + 52
            bar_w = 46
            bar_h = 8
            fill  = int(min(1.0, dens / 50.0) * bar_w)
            pygame.draw.rect(surf, C_HBAR_BG, (bar_x, y + 2, bar_w, bar_h))
            if fill > 0:
                pygame.draw.rect(surf, _density_color(dens / 50.0),
                                 (bar_x, y + 2, fill, bar_h))
            pygame.draw.rect(surf, C_TEXT_DIM, (bar_x, y + 2, bar_w, bar_h), 1)
            y += 18

        y += 4
        pygame.draw.line(surf, C_PANEL_DIV, (x0, y), (x0 + CAM_PANEL_W - 12, y), 1)
        y += 6

        uptime  = phase_snap.get('uptime_s', 0.0)
        cleared = self._vehicles.total_cleared()
        cycles  = phase_snap.get('total_cycles', 0)

        for stat in [
            f"Uptime:   {int(uptime // 60):02d}:{int(uptime % 60):02d}",
            f"Cycles:   {cycles}",
            f"Cleared:  {cleared} vehicles",
            f"Sim FPS:  {self._display_fps:.0f}",
        ]:
            lbl = self._fonts['sm'].render(stat, True, C_TEXT_DIM)
            surf.blit(lbl, (x0, y))
            y += 15

        # Key hints
        hints = "E:emrg  P:ped  C/B/A/M:spawn  TAB:arm  Z/X:zoom  SPACE:pause  Q:quit"
        surf.blit(
            self._fonts['title'].render(hints, True, C_TEXT_DIM),
            (x0, WIN_H - 13),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Pause / screenshot
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_pause_overlay(self) -> None:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (0, 0))
        msg = self._fonts['xl'].render("PAUSED — SPACE to resume", True, C_TEXT_YEL)
        self._screen.blit(msg, (
            WIN_W // 2 - msg.get_width()  // 2,
            WIN_H // 2 - msg.get_height() // 2,
        ))

    def _save_screenshot(self) -> None:
        self._screenshot_n += 1
        fname = f"screenshot_{self._screenshot_n:03d}.png"
        pygame.image.save(self._screen, fname)
        logger.info("Screenshot saved: %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Pure utilities
# ─────────────────────────────────────────────────────────────────────────────

def _density_color(ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.5:
        t = ratio * 2.0
        return (int(t * 220), 220, int((1 - t) * 60))
    else:
        t = (ratio - 0.5) * 2.0
        return (220, int((1 - t) * 220), 0)


def _draw_dashed_line(
    surf: pygame.Surface,
    color: tuple,
    start: tuple[int, int],
    end: tuple[int, int],
    dash: int = 12,
    gap:  int = 7,
    width: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx     = x1 - x0
    dy     = y1 - y0
    length = max(1, int(math.hypot(dx, dy)))
    nx, ny = dx / length, dy / length
    pos    = 0
    while pos < length:
        seg_end = min(pos + dash, length)
        pygame.draw.line(
            surf, color,
            (int(x0 + nx * pos), int(y0 + ny * pos)),
            (int(x0 + nx * seg_end), int(y0 + ny * seg_end)),
            width,
        )
        pos += dash + gap


def _draw_arrow(
    surf: pygame.Surface,
    color: tuple,
    start: tuple[int, int],
    end: tuple[int, int],
    head_size: int = 8,
) -> None:
    """Draw a filled arrow from start → end."""
    pygame.draw.line(surf, color, start, end, 2)
    dx  = end[0] - start[0]
    dy  = end[1] - start[1]
    ang = math.atan2(dy, dx)
    for side in (+0.5, -0.5):
        px = int(end[0] - head_size * math.cos(ang + side * math.pi * 0.6))
        py = int(end[1] - head_size * math.sin(ang + side * math.pi * 0.6))
        pygame.draw.line(surf, color, end, (px, py), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_simulation(
    state: IntersectionState,
    emergency_detector=None,
) -> PygameSimulation:
    return PygameSimulation(state=state, emergency_detector=emergency_detector)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test  (python -m simulation.pygame_sim)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import threading
    from controller.state import create_state
    from detection.emergency import EmergencyDetector

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    state    = create_state()
    emrg_det = EmergencyDetector()

    def mock_cycle():
        arms = ARM_NAMES
        idx  = 0
        while True:
            arm = arms[idx % len(arms)]
            with state.lock:
                state.set_signal(None, 'RED')
                state.set_signal(arm, 'GREEN')
                state.current_green = arm
                state.phase         = 'normal'
                for i, a in enumerate(arms):
                    state.arms[a].density   = (10 + i * 7 + (idx % 5) * 3) % 40
                    state.arms[a].flow_rate = 2.0 if a == arm else 0.4
                    state.arms[a].wait_time = (
                        0.0 if a == arm else state.arms[a].wait_time + 8
                    )
                state.total_cycles += 1
            time.sleep(8)
            with state.lock:
                state.set_signal(arm, 'YELLOW')
            time.sleep(3)
            idx += 1

    threading.Thread(target=mock_cycle, daemon=True).start()
    create_simulation(state, emergency_detector=emrg_det).run()