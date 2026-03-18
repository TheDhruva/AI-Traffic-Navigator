# simulation/pygame_sim.py — AI Smart Traffic Simulation (Bright Theme, 16:9)
# =============================================================================
# Full rewrite: 1600×900 true 16:9, bright high-contrast theme, larger fonts.
# Layout:
#   Left  panel (980px): top-down intersection + control strip
#   Right panel (620px): camera feed (top) + metrics table (bottom)
#
# Keyboard shortcuts:
#   Q/ESC   quit            E   emergency (North)     P   pedestrian rush
#   C       spawn car       B   spawn bus             A   spawn auto
#   M       spawn bike      V   animal hazard         U   ambulance
#   TAB     cycle arm       1-4 force green arm       R   reset waits
#   D       debug overlay   S   screenshot            SPACE pause
#   Z/+     zoom in         X/- zoom out              0   reset zoom

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

# ── IPC ───────────────────────────────────────────────────────────────────────
_CONTROL_FILE = Path("/tmp/control.json")
_SPAWN_FILE   = Path("/tmp/spawn.json")

# ── Window — true 16:9 ────────────────────────────────────────────────────────
WIN_W        = 1600
WIN_H        = 900
SIM_PANEL_W  = 980          # left  — intersection + control strip
CTRL_H       = 110          # control strip height at bottom of sim panel
SIM_DRAW_H   = WIN_H - CTRL_H   # 790px — drawable intersection area
CAM_PANEL_X  = SIM_PANEL_W
CAM_PANEL_W  = WIN_W - SIM_PANEL_W   # 620px

# Road geometry (within sim panel)
ROAD_W       = 90
SIM_CX       = SIM_PANEL_W // 2     # 490
SIM_CY       = SIM_DRAW_H  // 2     # 395

# Coordinate scalers (800×800 vehicle-space → screen)
def _sx(x: float) -> int: return int((x / ISECT_CX) * SIM_CX)
def _sy(y: float) -> int: return int((y / ISECT_CY) * SIM_CY)
def _sv(v: float) -> int: return int(v * (SIM_CX / ISECT_CX))

# Signal light positions (cx, cy on sim surface)
_SIG_R = 11
_SIG_POS: dict[str, tuple[int, int]] = {
    'North': (SIM_CX - ROAD_W // 2 - 22, _sy(ISECT_TOP)    - 28),
    'South': (SIM_CX + ROAD_W // 2 + 22, _sy(ISECT_BOTTOM) + 28),
    'East':  (_sx(ISECT_RIGHT)  + 28, SIM_CY - ROAD_W // 2 - 22),
    'West':  (_sx(ISECT_LEFT)   - 28, SIM_CY + ROAD_W // 2 + 22),
}

# Vehicle count badge positions
_BADGE_POS: dict[str, tuple[int, int]] = {
    'North': (SIM_CX - ROAD_W // 2 - 54, _sy(ISECT_TOP)    - 54),
    'South': (SIM_CX - ROAD_W // 2 - 54, _sy(ISECT_BOTTOM) + 34),
    'East':  (_sx(ISECT_RIGHT)  + 14,    SIM_CY + ROAD_W // 2 + 12),
    'West':  (_sx(ISECT_LEFT)   - 110,   SIM_CY + ROAD_W // 2 + 12),
}

# ── Colour palette — BRIGHT THEME ────────────────────────────────────────────
C_BG           = (232, 236, 245)
C_PANEL_WHITE  = (255, 255, 255)
C_PANEL_LIGHT  = (245, 247, 252)
C_DIVIDER      = (200, 205, 218)
C_ROAD         = ( 80,  88, 102)
C_ROAD_KERB    = ( 55,  62,  76)
C_ISECT        = ( 68,  76,  90)
C_LANE_MARK    = (220, 205,  60)
C_ZEBRA        = (225, 228, 235)
C_GRASS        = ( 82, 148,  66)
C_PAVEMENT     = (175, 182, 195)
C_TEXT_DARK    = ( 18,  22,  40)
C_TEXT_MID     = ( 70,  82, 110)
C_TEXT_DIM     = (140, 152, 175)
C_TEXT_WHITE   = (255, 255, 255)
C_GREEN_TEXT   = ( 18, 158,  65)
C_YELLOW_TEXT  = (185, 135,   0)
C_RED_TEXT     = (200,  35,  35)
C_CYAN_TEXT    = (  0, 148, 190)
_SIG_GREEN     = ( 34, 197,  94)
_SIG_YELLOW    = (251, 191,  36)
_SIG_RED       = (239,  68,  68)
C_BTN_DEFAULT  = (210, 215, 228)
C_BTN_HOVER    = (188, 196, 218)
C_BTN_SEL      = ( 37, 120, 220)
C_BTN_AUTO_ON  = ( 30, 165,  68)
C_BTN_EMRG     = (205,  38,  38)
C_BTN_PED      = ( 37, 100, 200)
C_BTN_ANIMAL   = (175, 108,  18)
C_BTN_ZOOM     = (178, 185, 202)
C_CTRL_BG      = (218, 222, 234)
C_CTRL_BORDER  = (175, 182, 204)
C_BANNER_EMRG  = (210,  38,  38)
C_BANNER_PED   = ( 35,  98, 198)
C_BANNER_HZRD  = (185, 116,  10)
C_HUD_BG       = (255, 255, 255)
C_HUD_BORDER   = (198, 207, 225)

_PHASE_COL = {
    'normal':     C_GREEN_TEXT,
    'emergency':  C_RED_TEXT,
    'pedestrian': C_CYAN_TEXT,
    'all_red':    C_RED_TEXT,
    'yellow':     C_YELLOW_TEXT,
    'startup':    C_TEXT_DIM,
}

# ── Spawn buttons ─────────────────────────────────────────────────────────────
_SPAWN_BTNS = [
    {'label': 'CAR',       'cls': 'car',        'key': 'C'},
    {'label': 'BUS',       'cls': 'bus',        'key': 'B'},
    {'label': 'AUTO',      'cls': 'auto',       'key': 'A'},
    {'label': 'BIKE',      'cls': 'motorcycle', 'key': 'M'},
    {'label': 'PED RUSH',  'cls': 'ped',        'key': 'G'},
    {'label': 'ANIMAL',    'cls': 'animal',     'key': 'V'},
    {'label': 'AMBULANCE', 'cls': 'ambulance',  'key': 'U'},
]
_ARM_BTNS  = ['North', 'South', 'East', 'West']
_BTN_W, _BTN_H, _BTN_GAP = 96, 36, 8
_ARM_W, _ARM_H             = 64, 28


# ═════════════════════════════════════════════════════════════════════════════
class PygameSimulation:
    """Bright-theme 1600×900 interactive traffic simulation."""

    def __init__(
        self,
        state: IntersectionState,
        emergency_detector=None,
        title: str = "AI Smart Traffic System — Indian Cities",
        camera_manager=None,
    ) -> None:
        self._state    = state
        self._emrg_det = emergency_detector
        self._cam_mgr  = camera_manager
        self._title    = title
        self._screen:   Optional[pygame.Surface] = None
        self._clock:    Optional[pygame.time.Clock] = None
        self._fonts:    dict[str, pygame.font.Font] = {}
        self._sim_surf: Optional[pygame.Surface] = None
        self._vehicles  = VehicleManager()

        self._selected_arm = 'North'
        self._auto_spawn   = False
        self._auto_timer   = 0.0
        self._spawn_rate   = 1.0
        self._sim_speed    = 1.0
        self._debug        = False
        self._paused       = False
        self._running      = True
        self._frame_n      = 0
        self._screenshot_n = 0

        self._zoom   = 1.0
        self._pan_x  = 0.0
        self._pan_y  = 0.0
        self._dragging     = False
        self._drag_origin: Optional[tuple[int, int]]    = None
        self._drag_pan0:   Optional[tuple[float, float]] = None

        self._arm_rects:     list[pygame.Rect] = []
        self._spawn_rects:   list[pygame.Rect] = []
        self._auto_rect:     Optional[pygame.Rect] = None
        self._zoom_in_rect:  Optional[pygame.Rect] = None
        self._zoom_out_rect: Optional[pygame.Rect] = None
        self._zoom_rst_rect: Optional[pygame.Rect] = None
        self._hover_spawn: Optional[int] = None
        self._hover_arm:   Optional[int] = None

        self._fps_buf:  list[float] = []
        self._fps_t:    float = time.time()
        self._fps_disp: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._init_pygame()
        logger.info("Pygame simulation started %dx%d @%dfps", WIN_W, WIN_H, SIM_FPS)
        while self._running:
            raw_dt = self._clock.tick(SIM_FPS) / 1000.0
            dt     = min(raw_dt, 0.1) * self._sim_speed
            self._handle_events()
            if not self._paused:
                self._update(dt)
            self._draw()
            pygame.display.flip()
            self._frame_n += 1
        pygame.quit()
        logger.info("Pygame simulation stopped")

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        """
        Fixed _init_pygame that auto-scales window to display resolution.
        1600×900 is ideal but many laptops are 1366×768 or 1280×720.
        Scales down to 90% of available display if needed.
        """
        pygame.init()
        pygame.display.set_caption(self._title)

        # Auto-detect display size
        info = pygame.display.Info()
        screen_w = info.current_w
        screen_h = info.current_h

        # Target 1600×900 but cap at 90% of screen
        target_w, target_h = 1600, 900
        max_w = int(screen_w * 0.92)
        max_h = int(screen_h * 0.92)

        actual_w = min(target_w, max_w)
        actual_h = min(target_h, max_h)

        # Maintain 16:9 ratio — scale down proportionally if needed
        if actual_w / actual_h > 16 / 9:
            actual_w = int(actual_h * 16 / 9)
        else:
            actual_h = int(actual_w * 9 / 16)

        # Update module-level globals — these are used by all draw functions
        import simulation.pygame_sim as _mod
        _mod.WIN_W       = actual_w
        _mod.WIN_H       = actual_h
        _mod.SIM_PANEL_W = int(actual_w * 0.6125)   # ~980/1600
        _mod.CTRL_H      = int(actual_h * 0.122)    # ~110/900
        _mod.SIM_DRAW_H  = actual_h - _mod.CTRL_H
        _mod.CAM_PANEL_X = _mod.SIM_PANEL_W
        _mod.CAM_PANEL_W = actual_w - _mod.SIM_PANEL_W
        _mod.SIM_CX      = _mod.SIM_PANEL_W // 2
        _mod.SIM_CY      = _mod.SIM_DRAW_H  // 2

        self._screen   = pygame.display.set_mode((actual_w, actual_h))
        self._sim_surf = pygame.Surface((_mod.SIM_PANEL_W, _mod.SIM_DRAW_H))
        self._clock    = pygame.time.Clock()

        # Font sizes scaled to window
        scale = actual_h / 900.0
        def _fs(n): return max(9, int(n * scale))

        self._fonts = {
            'xs':      pygame.font.SysFont('segoeui',   _fs(11)),
            'sm':      pygame.font.SysFont('segoeui',   _fs(13)),
            'md':      pygame.font.SysFont('segoeui',   _fs(15)),
            'lg':      pygame.font.SysFont('segoeui',   _fs(18), bold=True),
            'xl':      pygame.font.SysFont('segoeui',   _fs(24), bold=True),
            'xxl':     pygame.font.SysFont('segoeui',   _fs(32), bold=True),
            'btn':     pygame.font.SysFont('segoeui',   _fs(13), bold=True),
            'mono':    pygame.font.SysFont('monospace', _fs(13)),
            'mono_sm': pygame.font.SysFont('monospace', _fs(11)),
        }
        self._build_rects()

    def _build_rects(self) -> None:
        ctrl_top = SIM_DRAW_H
        r1_y  = ctrl_top + 10
        arm_x = 10
        self._arm_rects = [
            pygame.Rect(arm_x + i * (_ARM_W + 6), r1_y, _ARM_W, _ARM_H)
            for i in range(len(_ARM_BTNS))
        ]
        x = arm_x + len(_ARM_BTNS) * (_ARM_W + 6) + 18
        self._auto_rect     = pygame.Rect(x,       r1_y, 110, _ARM_H)
        x += 118
        self._zoom_in_rect  = pygame.Rect(x,       r1_y, 44, _ARM_H)
        self._zoom_out_rect = pygame.Rect(x + 48,  r1_y, 44, _ARM_H)
        self._zoom_rst_rect = pygame.Rect(x + 96,  r1_y, 44, _ARM_H)
        r2_y = ctrl_top + _ARM_H + 18
        self._spawn_rects = [
            pygame.Rect(10 + i * (_BTN_W + _BTN_GAP), r2_y, _BTN_W, _BTN_H)
            for i in range(len(_SPAWN_BTNS))
        ]

    # ── Events ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        mp = pygame.mouse.get_pos()
        self._hover_spawn = next((i for i, r in enumerate(self._spawn_rects) if r.collidepoint(mp)), None)
        self._hover_arm   = next((i for i, r in enumerate(self._arm_rects)   if r.collidepoint(mp)), None)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:                 self._quit()
            elif ev.type == pygame.KEYDOWN:            self._on_key(ev.key)
            elif ev.type == pygame.MOUSEBUTTONDOWN:    self._on_click(ev.button, ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 2: self._dragging = False
            elif ev.type == pygame.MOUSEMOTION:
                if self._dragging and self._drag_origin:
                    dx = ev.pos[0] - self._drag_origin[0]
                    dy = ev.pos[1] - self._drag_origin[1]
                    self._pan_x = self._drag_pan0[0] + dx
                    self._pan_y = self._drag_pan0[1] + dy
                    self._clamp_pan()

    def _on_click(self, btn: int, pos: tuple[int, int]) -> None:
        mid = (SIM_PANEL_W // 2, SIM_DRAW_H // 2)
        if btn == 4:   self._zoom_by(+0.12, pos);  return
        if btn == 5:   self._zoom_by(-0.12, pos);  return
        if btn == 2:
            self._dragging = True
            self._drag_origin = pos
            self._drag_pan0   = (self._pan_x, self._pan_y)
            return
        if btn != 1: return
        for i, r in enumerate(self._spawn_rects):
            if r.collidepoint(pos): self._do_spawn(_SPAWN_BTNS[i]['cls']); return
        for i, r in enumerate(self._arm_rects):
            if r.collidepoint(pos): self._selected_arm = _ARM_BTNS[i]; return
        if self._auto_rect     and self._auto_rect.collidepoint(pos):
            self._auto_spawn = not self._auto_spawn; return
        if self._zoom_in_rect  and self._zoom_in_rect.collidepoint(pos):
            self._zoom_by(+0.2, mid); return
        if self._zoom_out_rect and self._zoom_out_rect.collidepoint(pos):
            self._zoom_by(-0.2, mid); return
        if self._zoom_rst_rect and self._zoom_rst_rect.collidepoint(pos):
            self._zoom = 1.0; self._pan_x = self._pan_y = 0.0

    def _on_key(self, key: int) -> None:
        mid = (SIM_PANEL_W // 2, SIM_DRAW_H // 2)
        binds = {
            pygame.K_q:      lambda: self._quit(),
            pygame.K_ESCAPE: lambda: self._quit(),
            pygame.K_e:      lambda: self._do_spawn('ambulance', 'North'),
            pygame.K_p:      lambda: self._do_spawn('ped'),
            pygame.K_c:      lambda: self._do_spawn('car'),
            pygame.K_b:      lambda: self._do_spawn('bus'),
            pygame.K_a:      lambda: self._do_spawn('auto'),
            pygame.K_m:      lambda: self._do_spawn('motorcycle'),
            pygame.K_g:      lambda: self._do_spawn('ped'),
            pygame.K_v:      lambda: self._do_spawn('animal'),
            pygame.K_u:      lambda: self._do_spawn('ambulance'),
            pygame.K_TAB:    lambda: self._cycle_arm(),
            pygame.K_d:      lambda: setattr(self, '_debug', not self._debug),
            pygame.K_SPACE:  lambda: setattr(self, '_paused', not self._paused),
            pygame.K_r:      lambda: self._reset_waits(),
            pygame.K_s:      lambda: self._screenshot(),
            pygame.K_1:      lambda: self._force_green('North'),
            pygame.K_2:      lambda: self._force_green('South'),
            pygame.K_3:      lambda: self._force_green('East'),
            pygame.K_4:      lambda: self._force_green('West'),
            pygame.K_0:      lambda: self._reset_zoom(),
            pygame.K_z:      lambda: self._zoom_by(+0.15, mid),
            pygame.K_PLUS:   lambda: self._zoom_by(+0.15, mid),
            pygame.K_EQUALS: lambda: self._zoom_by(+0.15, mid),
            pygame.K_x:      lambda: self._zoom_by(-0.15, mid),
            pygame.K_MINUS:  lambda: self._zoom_by(-0.15, mid),
        }
        fn = binds.get(key)
        if fn: fn()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _do_spawn(self, cls: str, arm: Optional[str] = None) -> None:
        target = arm or self._selected_arm
        if cls == 'ped':
            if self._emrg_det: self._emrg_det.simulate_ped_rush()
        elif cls == 'ambulance':
            if self._emrg_det: self._emrg_det.simulate_emergency(target)
        elif cls == 'animal':
            try:
                with self._state.lock:
                    self._state.arms[target].hazard = True
            except Exception: pass
        else:
            q = self._vehicles.queues.get(target)
            if q and len(q.vehicles) < MAX_VEHICLES_PER_ARM:
                sx, sy = SPAWN_POSITIONS[target]
                jit = random.uniform(-6, 6)
                if target in ('North', 'South'): sx += jit
                else:                             sy += jit
                q.vehicles.append(Vehicle(arm=target, cls=cls, x=float(sx), y=float(sy)))

    def _force_green(self, arm: str) -> None:
        with self._state.lock:
            for a in self._state.arms.values():
                a.signal_state = 'RED'
            if arm in self._state.arms:
                self._state.arms[arm].signal_state = 'GREEN'
            self._state.current_green = arm
            self._state.phase = 'normal'
            self._state._update_arm_signals_locked(arm, 'green')

    def _cycle_arm(self) -> None:
        idx = _ARM_BTNS.index(self._selected_arm)
        self._selected_arm = _ARM_BTNS[(idx + 1) % len(_ARM_BTNS)]

    def _reset_waits(self) -> None:
        with self._state.lock:
            for arm in self._state.arms.values():
                arm.wait_time = 0.0

    def _reset_zoom(self) -> None:
        self._zoom = 1.0; self._pan_x = self._pan_y = 0.0

    def _zoom_by(self, delta: float, anchor: tuple[int, int]) -> None:
        old = self._zoom
        self._zoom = max(0.4, min(3.0, self._zoom + delta))
        ax, ay = anchor
        r = self._zoom / old
        self._pan_x = ax - r * (ax - self._pan_x)
        self._pan_y = ay - r * (ay - self._pan_y)
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        sw = SIM_PANEL_W * self._zoom;  sh = SIM_DRAW_H * self._zoom
        self._pan_x = max(min(self._pan_x, SIM_PANEL_W * 0.5), SIM_PANEL_W - sw - SIM_PANEL_W * 0.5)
        self._pan_y = max(min(self._pan_y, SIM_DRAW_H  * 0.5), SIM_DRAW_H  - sh - SIM_DRAW_H  * 0.5)

    def _quit(self) -> None:
        self._running = False
        with self._state.lock: self._state.running = False

    def _screenshot(self) -> None:
        self._screenshot_n += 1
        fname = f"screenshot_{self._screenshot_n:03d}.png"
        pygame.image.save(self._screen, fname)
        logger.info("Screenshot: %s", fname)

    # ── Update ────────────────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        self._vehicles.update(self._state.snapshot_arms(), dt)
        if self._auto_spawn:
            self._auto_timer += dt
            iv = 1.0 / max(self._spawn_rate, 0.1)
            while self._auto_timer >= iv:
                self._auto_timer -= iv
                self._do_spawn(_random_vehicle_class(), random.choice(ARM_NAMES))
        if self._frame_n % 60 == 0: self._poll_control()
        self._poll_spawn()
        now = time.time()
        self._fps_buf.append(1.0 / max(dt / max(self._sim_speed, 0.01), 0.001))
        if now - self._fps_t >= 1.0:
            self._fps_disp = sum(self._fps_buf) / max(len(self._fps_buf), 1)
            self._fps_buf.clear(); self._fps_t = now

    def _poll_control(self) -> None:
        try:
            c = json.loads(_CONTROL_FILE.read_text())
            self._spawn_rate = float(c.get('spawn_rate', 1.0))
            self._sim_speed  = float(c.get('sim_speed',  1.0))
        except Exception: pass

    def _poll_spawn(self) -> None:
        if not _SPAWN_FILE.exists(): return
        try:
            c = json.loads(_SPAWN_FILE.read_text())
            _SPAWN_FILE.unlink(missing_ok=True)
            self._do_spawn(c.get('type', 'car'), c.get('arm', self._selected_arm))
        except Exception: pass

    # ── Draw top-level ────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._screen.fill(C_BG)

        self._sim_surf.fill(C_BG)
        self._draw_intersection(self._sim_surf)

        if self._zoom == 1.0 and self._pan_x == 0.0 and self._pan_y == 0.0:
            self._screen.blit(self._sim_surf, (0, 0))
        else:
            sw = int(SIM_PANEL_W * self._zoom)
            sh = int(SIM_DRAW_H  * self._zoom)
            scaled = pygame.transform.scale(self._sim_surf, (sw, sh))
            self._screen.set_clip(pygame.Rect(0, 0, SIM_PANEL_W, SIM_DRAW_H))
            self._screen.blit(scaled, (int(self._pan_x), int(self._pan_y)))
            self._screen.set_clip(None)

        self._draw_control_strip()
        self._draw_camera_panel()
        pygame.draw.line(self._screen, C_DIVIDER, (SIM_PANEL_W, 0), (SIM_PANEL_W, WIN_H), 2)
        if self._paused: self._draw_pause_overlay()

    # ── Intersection (onto _sim_surf) ─────────────────────────────────────────

    def _draw_intersection(self, surf: pygame.Surface) -> None:
        rl = SIM_CX - ROAD_W // 2;  rr = SIM_CX + ROAD_W // 2
        rt = SIM_CY - ROAD_W // 2;  rb = SIM_CY + ROAD_W // 2
        ix = _sx(ISECT_LEFT);  iy = _sy(ISECT_TOP)
        iw = _sx(ISECT_RIGHT) - ix;  ih = _sy(ISECT_BOTTOM) - iy

        # Grass corners
        for gx, gy, gw, gh in [(0, 0, rl, rt), (rr, 0, SIM_PANEL_W - rr, rt),
                                 (0, rb, rl, SIM_DRAW_H - rb), (rr, rb, SIM_PANEL_W - rr, SIM_DRAW_H - rb)]:
            pygame.draw.rect(surf, C_GRASS, (gx, gy, gw, gh))
        # Pavement strips beside road
        for px, py, pw, ph in [(rl - 9, 0, 9, SIM_DRAW_H), (rr, 0, 9, SIM_DRAW_H),
                                 (0, rt - 9, SIM_PANEL_W, 9), (0, rb, SIM_PANEL_W, 9)]:
            pygame.draw.rect(surf, C_PAVEMENT, (px, py, pw, ph))

        # Road
        pygame.draw.rect(surf, C_ROAD, (rl, 0,  ROAD_W, SIM_DRAW_H))
        pygame.draw.rect(surf, C_ROAD, (0,  rt, SIM_PANEL_W, ROAD_W))
        for x in (rl, rr): pygame.draw.line(surf, C_ROAD_KERB, (x, 0), (x, SIM_DRAW_H), 3)
        for y in (rt, rb): pygame.draw.line(surf, C_ROAD_KERB, (0, y), (SIM_PANEL_W, y), 3)

        # Intersection box
        pygame.draw.rect(surf, C_ISECT,    (ix, iy, iw, ih))
        pygame.draw.rect(surf, C_ROAD_KERB,(ix, iy, iw, ih), 2)

        # Dashed centre lines
        _dash(surf, C_LANE_MARK, (SIM_CX, 0),        (SIM_CX, iy - 1),         14, 8, 2)
        _dash(surf, C_LANE_MARK, (SIM_CX, iy + ih + 1), (SIM_CX, SIM_DRAW_H),  14, 8, 2)
        _dash(surf, C_LANE_MARK, (0, SIM_CY),        (ix - 1, SIM_CY),         14, 8, 2)
        _dash(surf, C_LANE_MARK, (ix + iw + 1, SIM_CY), (SIM_PANEL_W, SIM_CY), 14, 8, 2)

        # Zebra crossings
        S, G, D = 6, 5, 22
        for y0 in (iy - D, iy + ih):
            x = rl
            while x < rr:
                pygame.draw.rect(surf, C_ZEBRA, (x, y0, S, D)); x += S + G
        for x0 in (ix - D, ix + iw):
            y = rt
            while y < rb:
                pygame.draw.rect(surf, C_ZEBRA, (x0, y, D, S)); y += S + G

        # Direction arrows
        C_ARR = (100, 108, 125)
        _arrow(surf, C_ARR, (SIM_CX, iy - 70),     (SIM_CX, iy - 38),     12)
        _arrow(surf, C_ARR, (SIM_CX, iy + ih + 70),(SIM_CX, iy + ih + 38),12)
        _arrow(surf, C_ARR, (ix + iw + 70, SIM_CY),(ix + iw + 38, SIM_CY),12)
        _arrow(surf, C_ARR, (ix - 70, SIM_CY),     (ix - 38, SIM_CY),     12)

        # Stop lines
        with self._state.lock:
            sigs = {n: a.signal_state for n, a in self._state.arms.items()}
        for arm, sig in sigs.items():
            col  = _SIG_GREEN if sig == 'GREEN' else (_SIG_YELLOW if sig == 'YELLOW' else _SIG_RED)
            stop = STOP_LINES[arm]
            if arm in ('North', 'South'):
                sy_ = _sy(stop)
                pygame.draw.line(surf, col, (rl, sy_), (rr, sy_), 4)
            else:
                sx_ = _sx(stop)
                pygame.draw.line(surf, col, (sx_, rt), (sx_, rb), 4)

        # Vehicles
        for v in self._vehicles.all_vehicles():
            rx, ry, rw, rh = v.rect
            vr = pygame.Rect(_sx(rx), _sy(ry), max(5, _sv(rw)), max(7, _sv(rh)))
            pygame.draw.rect(surf, v.color, vr, border_radius=3)
            pygame.draw.rect(surf, (0, 0, 0), vr, 1, border_radius=3)

        # Signal lights
        for arm in ARM_NAMES:
            sig = sigs.get(arm, 'RED')
            cx, cy = _SIG_POS[arm]
            ns = arm in ('North', 'South')
            hw = _SIG_R * 2 + 10;  hh = _SIG_R * 6 + 16
            if not ns: hw, hh = hh, hw
            pygame.draw.rect(surf, (48, 54, 65), (cx - hw // 2, cy - hh // 2, hw, hh), border_radius=5)
            pygame.draw.rect(surf, (28, 32, 40), (cx - hw // 2, cy - hh // 2, hw, hh), 1, border_radius=5)
            lamps = [(cx, cy - _SIG_R * 2), (cx, cy), (cx + _SIG_R * 2 if not ns else 0,
                      cy + _SIG_R * 2 if ns else cy)] if ns else \
                    [(cx - _SIG_R * 2, cy), (cx, cy), (cx + _SIG_R * 2, cy)]
            # Re-derive cleanly
            if ns:
                lamps = [(cx, cy - _SIG_R * 2), (cx, cy), (cx, cy + _SIG_R * 2)]
            else:
                lamps = [(cx - _SIG_R * 2, cy), (cx, cy), (cx + _SIG_R * 2, cy)]
            defs = [('RED', _SIG_RED), ('YELLOW', _SIG_YELLOW), ('GREEN', _SIG_GREEN)]
            for (lx, ly), (phase, on) in zip(lamps, defs):
                active = (sig == phase) or (sig == 'WALK' and phase == 'RED')
                col    = on if active else (42, 44, 52)
                pygame.draw.circle(surf, col, (lx, ly), _SIG_R)
                pygame.draw.circle(surf, (20, 22, 28), (lx, ly), _SIG_R, 1)
                if active:
                    gsurf = pygame.Surface((_SIG_R * 4, _SIG_R * 4), pygame.SRCALPHA)
                    pygame.draw.circle(gsurf, (*on, 55), (_SIG_R * 2, _SIG_R * 2), _SIG_R * 2)
                    surf.blit(gsurf, (lx - _SIG_R * 2, ly - _SIG_R * 2))

        # Arm badges
        counts = self._vehicles.vehicle_count_by_arm()
        for arm in ARM_NAMES:
            n = counts.get(arm, 0); sel = (arm == self._selected_arm)
            bx, by = _BADGE_POS[arm]
            bg = C_BTN_SEL if sel else C_PANEL_WHITE
            tc = C_TEXT_WHITE if sel else C_TEXT_DARK
            lbl = self._fonts['sm'].render(f"{arm[0]}  {n:2d}", True, tc)
            pad = 6; r = pygame.Rect(bx - pad, by - 3, lbl.get_width() + pad * 2, 20)
            pygame.draw.rect(surf, bg, r, border_radius=4)
            pygame.draw.rect(surf, C_DIVIDER, r, 1, border_radius=4)
            surf.blit(lbl, (bx, by))

        # HUD card
        self._draw_hud(surf)

        # Alert banner
        self._draw_banner(surf)

        if self._debug:
            q = self._vehicles.vehicle_count_by_arm()
            qq = self._vehicles.queued_count_by_arm()
            yy = 46
            for arm in ARM_NAMES:
                t = self._fonts['mono_sm'].render(
                    f"[D] {arm:<6} total={q[arm]:2d} queued={qq[arm]:2d}", True, C_BTN_PED)
                surf.blit(t, (SIM_PANEL_W - 220, yy)); yy += 15

    def _draw_hud(self, surf: pygame.Surface) -> None:
        ph   = self._state.snapshot_phase()
        armp = self._state.snapshot_arms()
        phase   = ph.get('phase', 'normal')
        cur_grn = ph.get('current_green')
        p_col   = _PHASE_COL.get(phase, C_TEXT_DIM)
        hx, hy, hw, hh = 10, SIM_DRAW_H - 142, 490, 136

        # Shadow
        sh = pygame.Surface((hw + 4, hh + 4), pygame.SRCALPHA)
        sh.fill((0, 0, 0, 28)); surf.blit(sh, (hx + 2, hy + 2))
        pygame.draw.rect(surf, C_HUD_BG,     (hx, hy, hw, hh), border_radius=10)
        pygame.draw.rect(surf, C_HUD_BORDER, (hx, hy, hw, hh), 1, border_radius=10)

        # Phase header
        ph_lbl = self._fonts['lg'].render(
            f"PHASE: {phase.upper()}    cycles = {ph.get('total_cycles', 0)}", True, p_col)
        surf.blit(ph_lbl, (hx + 12, hy + 10))
        pygame.draw.line(surf, C_HUD_BORDER, (hx + 12, hy + 34), (hx + hw - 12, hy + 34), 1)

        for i, arm in enumerate(ARM_NAMES):
            s    = armp.get(arm, {})
            sig  = s.get('signal_state', 'RED')
            dens = s.get('density',   0.0)
            wait = s.get('wait_time', 0.0)
            emrg = s.get('emergency', False)
            hzrd = s.get('hazard',    False)
            grn  = (arm == cur_grn)

            sc = C_GREEN_TEXT if sig == 'GREEN' else (C_YELLOW_TEXT if sig == 'YELLOW' else C_RED_TEXT)
            rc = C_GREEN_TEXT if grn else C_TEXT_DARK
            ry = hy + 40 + i * 22

            dot = _SIG_GREEN if grn else _SIG_RED
            pygame.draw.circle(surf, dot, (hx + 16, ry + 8), 6)

            surf.blit(self._fonts['mono_sm'].render(sig[0], True, sc), (hx + 30, ry + 2))
            flags   = (" ⚠EMRG" if emrg else "") + (" ⚠HZRD" if hzrd else "")
            row_txt = f"{arm:<6}  dens={dens:5.1f}  wait={int(wait):3d}s{flags}"
            surf.blit(self._fonts['mono_sm'].render(row_txt, True, rc), (hx + 44, ry + 2))

            # Density bar
            bx  = hx + 365; bw = 110; bht = 12
            fill = int(min(1.0, dens / 50.0) * bw)
            pygame.draw.rect(surf, C_BG,       (bx, ry + 3, bw, bht), border_radius=4)
            if fill > 0:
                pygame.draw.rect(surf, _density_col(dens / 50.0), (bx, ry + 3, fill, bht), border_radius=4)
            pygame.draw.rect(surf, C_DIVIDER,  (bx, ry + 3, bw, bht), 1, border_radius=4)

        fps_txt = self._fonts['xs'].render(
            f"zoom={self._zoom:.1f}×   vehicles={len(self._vehicles.all_vehicles())}   {self._fps_disp:.0f} fps",
            True, C_TEXT_DIM)
        surf.blit(fps_txt, (hx + 12, hy + hh - 14))

    def _draw_banner(self, surf: pygame.Surface) -> None:
        ph   = self._state.snapshot_phase()
        armp = self._state.snapshot_arms()
        phase = ph.get('phase', 'normal')
        emrg  = next((a for a, s in armp.items() if s.get('emergency', False)), None)
        hzrd  = next(((a, True) for a, s in armp.items() if s.get('hazard', False)), None)
        bh = 40; bs = pygame.Surface((SIM_PANEL_W, bh), pygame.SRCALPHA)
        if phase == 'emergency' and emrg:
            bs.fill((*C_BANNER_EMRG, 238)); surf.blit(bs, (0, 0))
            surf.blit(self._fonts['lg'].render(
                f"  🚨  EMERGENCY OVERRIDE — {emrg.upper()} ARM PRIORITY", True, C_TEXT_WHITE), (12, 10))
        elif phase == 'pedestrian':
            bs.fill((*C_BANNER_PED, 228)); surf.blit(bs, (0, 0))
            avg = ph.get('ped_rolling_avg', 0.0)
            surf.blit(self._fonts['lg'].render(
                f"  🚶  PEDESTRIAN PHASE ACTIVE — {avg:.0f} persons detected", True, C_TEXT_WHITE), (12, 10))
        elif hzrd:
            a, _ = hzrd; bs.fill((*C_BANNER_HZRD, 228)); surf.blit(bs, (0, 0))
            surf.blit(self._fonts['lg'].render(
                f"  ⚠  ANIMAL ON ROAD — {a.upper()} ARM  (+5s extension)", True, C_TEXT_WHITE), (12, 10))

    # ── Control strip ─────────────────────────────────────────────────────────

    def _draw_control_strip(self) -> None:
        surf = self._screen
        py   = SIM_DRAW_H
        pygame.draw.rect(surf, C_CTRL_BG, (0, py, SIM_PANEL_W, CTRL_H))
        pygame.draw.line(surf, C_CTRL_BORDER, (0, py), (SIM_PANEL_W, py), 2)

        # Arm selector
        for i, (arm, r) in enumerate(zip(_ARM_BTNS, self._arm_rects)):
            sel = (arm == self._selected_arm); hov = (self._hover_arm == i)
            bg  = C_BTN_SEL if sel else (C_BTN_HOVER if hov else C_BTN_DEFAULT)
            tc  = C_TEXT_WHITE if sel else C_TEXT_DARK
            pygame.draw.rect(surf, bg, r, border_radius=6)
            lbl = self._fonts['btn'].render(arm[0], True, tc)
            surf.blit(lbl, _ctr(lbl, r))

        # Arm name label
        ai = self._fonts['sm'].render(f"Target arm: {self._selected_arm}  [TAB]", True, C_TEXT_MID)
        surf.blit(ai, (self._arm_rects[-1].right + 10, self._arm_rects[0].centery - ai.get_height() // 2))

        # Auto-spawn
        r  = self._auto_rect
        bg = C_BTN_AUTO_ON if self._auto_spawn else C_BTN_DEFAULT
        tc = C_TEXT_WHITE  if self._auto_spawn else C_TEXT_DARK
        pygame.draw.rect(surf, bg, r, border_radius=6)
        surf.blit(self._fonts['btn'].render("▶ AUTO" if self._auto_spawn else "■ AUTO", True, tc), _ctr(self._fonts['btn'].render("▶ AUTO" if self._auto_spawn else "■ AUTO", True, tc), r))

        # Zoom buttons
        for rect, label in [(self._zoom_in_rect, "Z+"), (self._zoom_out_rect, "Z−"), (self._zoom_rst_rect, "1:1")]:
            if rect:
                pygame.draw.rect(surf, C_BTN_ZOOM, rect, border_radius=6)
                lbl = self._fonts['btn'].render(label, True, C_TEXT_DARK)
                surf.blit(lbl, _ctr(lbl, rect))

        # Spawn buttons
        _BG = {'ped': C_BTN_PED, 'animal': C_BTN_ANIMAL, 'ambulance': C_BTN_EMRG}
        _TC = {'ped': C_TEXT_WHITE, 'animal': C_TEXT_WHITE, 'ambulance': C_TEXT_WHITE}
        for i, (btn, r) in enumerate(zip(_SPAWN_BTNS, self._spawn_rects)):
            hov  = (self._hover_spawn == i)
            base = _BG.get(btn['cls'], C_BTN_DEFAULT)
            bg   = tuple(min(255, c + 22) for c in base) if hov else base
            tc   = _TC.get(btn['cls'], C_TEXT_DARK)
            pygame.draw.rect(surf, bg, r, border_radius=7)
            pygame.draw.rect(surf, C_CTRL_BORDER, r, 1, border_radius=7)
            lbl = self._fonts['btn'].render(btn['label'], True, tc)
            surf.blit(lbl, _ctr(lbl, r))
            kl = self._fonts['xs'].render(f"[{btn['key']}]", True, C_TEXT_DIM)
            surf.blit(kl, (r.centerx - kl.get_width() // 2, r.bottom + 2))

        st = self._fonts['xs'].render(
            f"rate={self._spawn_rate:.1f}/s   speed={self._sim_speed:.1f}×   "
            f"scroll=zoom   mid-drag=pan   [0]=reset view   [S]=screenshot",
            True, C_TEXT_DIM)
        surf.blit(st, (10, SIM_DRAW_H + CTRL_H - 14))

    # ── Camera panel ──────────────────────────────────────────────────────────

    def _draw_camera_panel(self) -> None:
        surf = self._screen; px = CAM_PANEL_X
        pygame.draw.rect(surf, C_PANEL_WHITE, (px, 0, CAM_PANEL_W, WIN_H))

        # Header bar
        pygame.draw.rect(surf, C_BTN_SEL, (px, 0, CAM_PANEL_W, 44))
        surf.blit(self._fonts['lg'].render("Live YOLO Detection Feed", True, C_TEXT_WHITE),
                  (px + 14, 11))
        fps_l = self._fonts['sm'].render(f"{self._fps_disp:.0f} fps", True, C_TEXT_WHITE)
        surf.blit(fps_l, (px + CAM_PANEL_W - fps_l.get_width() - 14, 13))

        # Camera feed
        cam_h    = 385
        cam_rect = pygame.Rect(px + 8, 52, CAM_PANEL_W - 16, cam_h)
        frame    = self._state.get_annotated_frame()
        if frame is not None:
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb  = cv2.resize(rgb, (cam_rect.w, cam_rect.h))
                csf  = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                surf.blit(csf, cam_rect.topleft)
            except Exception:
                self._no_camera(surf, cam_rect)
        else:
            self._no_camera(surf, cam_rect)
        pygame.draw.rect(surf, C_DIVIDER, cam_rect, 1)
        self._draw_cam_banner(surf, cam_rect)

        # Metrics below camera
        self._draw_metrics(surf, cam_rect.bottom + 14)

    def _no_camera(self, surf, rect) -> None:
        pygame.draw.rect(surf, (226, 230, 240), rect)
        msg = self._fonts['md'].render("Waiting for detection feed...", True, C_TEXT_DIM)
        surf.blit(msg, (rect.centerx - msg.get_width() // 2, rect.centery - 10))

    def _draw_cam_banner(self, surf, cam_rect) -> None:
        ph   = self._state.snapshot_phase()
        armp = self._state.snapshot_arms()
        phase = ph.get('phase', 'normal')
        emrg  = next((a for a, s in armp.items() if s.get('emergency', False)), None)
        hzrd  = next(((a, True) for a, s in armp.items() if s.get('hazard', False)), None)
        bh = 36; bs = pygame.Surface((cam_rect.w, bh), pygame.SRCALPHA)
        if phase == 'emergency' and emrg:
            bs.fill((*C_BANNER_EMRG, 235)); surf.blit(bs, cam_rect.topleft)
            surf.blit(self._fonts['md'].render(f"  🚨  EMERGENCY — {emrg.upper()} ARM", True, C_TEXT_WHITE),
                      (cam_rect.x + 8, cam_rect.y + 9))
        elif phase == 'pedestrian':
            bs.fill((*C_BANNER_PED, 225)); surf.blit(bs, cam_rect.topleft)
            avg = ph.get('ped_rolling_avg', 0.0)
            surf.blit(self._fonts['md'].render(f"  🚶  PEDESTRIAN PHASE  ({avg:.0f} persons)", True, C_TEXT_WHITE),
                      (cam_rect.x + 8, cam_rect.y + 9))
        elif hzrd:
            a, _ = hzrd; bs.fill((*C_BANNER_HZRD, 225)); surf.blit(bs, cam_rect.topleft)
            surf.blit(self._fonts['md'].render(f"  ⚠  ANIMAL — {a.upper()} ARM", True, C_TEXT_WHITE),
                      (cam_rect.x + 8, cam_rect.y + 9))

    def _draw_metrics(self, surf, y0: int) -> None:
        ph   = self._state.snapshot_phase()
        armp = self._state.snapshot_arms()
        cur  = ph.get('current_green')
        px   = CAM_PANEL_X + 14

        sec = self._fonts['lg'].render("Traffic Analytics", True, C_TEXT_DARK)
        surf.blit(sec, (px, y0))
        pygame.draw.line(surf, C_DIVIDER, (px, y0 + 24), (px + CAM_PANEL_W - 28, y0 + 24), 1)
        y0 += 30

        # Column headers
        cx_list = [px, px + 78, px + 175, px + 260, px + 340]
        for hdr, cx in zip(['ARM', 'SIGNAL', 'DENSITY', 'WAIT', 'BAR'], cx_list):
            surf.blit(self._fonts['xs'].render(hdr, True, C_TEXT_DIM), (cx, y0))
        y0 += 18

        for arm in ARM_NAMES:
            s    = armp.get(arm, {})
            sig  = s.get('signal_state', 'RED')
            dens = s.get('density',   0.0)
            wait = s.get('wait_time', 0.0)
            grn  = (arm == cur)

            if grn:
                pygame.draw.rect(surf, (228, 248, 232),
                                 pygame.Rect(px - 6, y0 - 2, CAM_PANEL_W - 20, 22), border_radius=4)

            sc = C_GREEN_TEXT if sig == 'GREEN' else (C_YELLOW_TEXT if sig == 'YELLOW' else C_RED_TEXT)
            rc = C_GREEN_TEXT if grn else C_TEXT_DARK

            surf.blit(self._fonts['md'].render(arm,           True, rc), (cx_list[0], y0))
            surf.blit(self._fonts['md'].render(sig,           True, sc), (cx_list[1], y0))
            surf.blit(self._fonts['md'].render(f"{dens:.1f}", True, C_TEXT_MID), (cx_list[2], y0))
            surf.blit(self._fonts['md'].render(f"{int(wait)}s", True, C_TEXT_MID), (cx_list[3], y0))

            bx = cx_list[4]; bw = CAM_PANEL_W - bx + CAM_PANEL_X - 22; bht = 13
            fill = int(min(1.0, dens / 50.0) * bw)
            pygame.draw.rect(surf, C_BG,      (bx, y0 + 3, bw, bht), border_radius=4)
            if fill > 0:
                pygame.draw.rect(surf, _density_col(dens / 50.0), (bx, y0 + 3, fill, bht), border_radius=4)
            pygame.draw.rect(surf, C_DIVIDER, (bx, y0 + 3, bw, bht), 1, border_radius=4)
            y0 += 24

        pygame.draw.line(surf, C_DIVIDER, (px, y0 + 4), (px + CAM_PANEL_W - 28, y0 + 4), 1)
        y0 += 12

        uptime  = ph.get('uptime_s',        0.0)
        cleared = self._vehicles.total_cleared()
        cycles  = ph.get('total_cycles',     0)
        ped_avg = ph.get('ped_rolling_avg',  0.0)

        for label, val in [("Uptime",          f"{int(uptime // 60):02d}:{int(uptime % 60):02d}"),
                           ("Signal cycles",   str(cycles)),
                           ("Vehicles cleared",str(cleared)),
                           ("Ped avg",         f"{ped_avg:.1f} / 8.0")]:
            surf.blit(self._fonts['sm'].render(label, True, C_TEXT_DIM),  (px, y0))
            surf.blit(self._fonts['md'].render(val,   True, C_TEXT_DARK), (px + 148, y0))
            y0 += 20

        hints = self._fonts['xs'].render(
            "E:emrg  P:ped  C/B/A/M:spawn  V:animal  TAB:arm  Z/X:zoom  SPACE:pause  Q:quit",
            True, C_TEXT_DIM)
        surf.blit(hints, (px, WIN_H - 14))

    def _draw_pause_overlay(self) -> None:
        ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        ov.fill((255, 255, 255, 130))
        self._screen.blit(ov, (0, 0))
        msg = self._fonts['xxl'].render("PAUSED — press SPACE to resume", True, C_TEXT_DARK)
        self._screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2 - msg.get_height() // 2))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _density_col(r: float) -> tuple[int, int, int]:
    r = max(0.0, min(1.0, r))
    if r < 0.5:  return (int(r * 2 * 230), 200, 40)
    else:        return (230, int((1 - (r - 0.5) * 2) * 200), 0)


def _dash(surf, color, start, end, dash=12, gap=7, width=1) -> None:
    x0, y0 = start;  x1, y1 = end
    dx, dy  = x1 - x0, y1 - y0
    ln = max(1, int(math.hypot(dx, dy)));  nx, ny = dx / ln, dy / ln
    p  = 0
    while p < ln:
        e = min(p + dash, ln)
        pygame.draw.line(surf, color,
                         (int(x0 + nx * p), int(y0 + ny * p)),
                         (int(x0 + nx * e), int(y0 + ny * e)), width)
        p += dash + gap


def _arrow(surf, color, start, end, hs=10) -> None:
    pygame.draw.line(surf, color, start, end, 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    a = math.atan2(dy, dx)
    for s in (+0.5, -0.5):
        pygame.draw.line(surf, color, end,
                         (int(end[0] - hs * math.cos(a + s * math.pi * 0.65)),
                          int(end[1] - hs * math.sin(a + s * math.pi * 0.65))), 2)


def _ctr(lbl, rect: pygame.Rect) -> tuple[int, int]:
    return (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_simulation(
    state: IntersectionState,
    emergency_detector=None,
    camera_manager=None,
) -> PygameSimulation:
    return PygameSimulation(state=state, emergency_detector=emergency_detector,
                            camera_manager=camera_manager)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import threading
    from controller.state import create_state
    from detection.emergency import EmergencyDetector

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    state    = create_state()
    emrg_det = EmergencyDetector()

    def _mock():
        idx: int = 0
        while True:
            arm = ARM_NAMES[idx % len(ARM_NAMES)]
            with state.lock:
                state.set_signal(None, 'RED')
                state.set_signal(arm, 'GREEN')
                state.current_green = arm
                state.phase         = 'normal'
                idx_val = int(idx)
                for i_val, a in enumerate(ARM_NAMES):
                    val = (int(i_val) * 6 + (idx_val % 6) * 4)
                    state.arms[a].density = (8 + val) % 45
                    state.arms[a].flow_rate = 2.2 if a == arm else 0.3
                    state.arms[a].wait_time = 0.0 if a == arm else state.arms[a].wait_time + 7
                state.session_metrics['total_cycles'] = state.session_metrics.get('total_cycles', 0) + 1
            time.sleep(8)
            with state.lock: state.set_signal(arm, 'YELLOW')
            time.sleep(3)
            idx = idx + 1  # pyre-ignore

    threading.Thread(target=_mock, daemon=True).start()
    create_simulation(state, emergency_detector=emrg_det).run()