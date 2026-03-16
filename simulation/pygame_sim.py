# simulation/pygame_sim.py — Pygame Top-Down Intersection Simulation
# Runs on the MAIN thread (Thread 3). Reads IntersectionState and renders:
#   - Top-down 2D intersection with animated vehicles
#   - Signal lights (R/Y/G) at each arm entrance
#   - Live annotated camera feed (from detection thread)
#   - Signal HUD: arm name, density, wait time, score
#   - Alert banners: EMERGENCY / PEDESTRIAN / HAZARD
#   - Keyboard shortcuts for demo control
#
# Layout (1280 × 720):
#   Left  panel (720×720): top-down intersection simulation
#   Right panel (560×720): annotated camera feed + metrics sidebar

from __future__ import annotations

import logging
import sys
import time
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
    SIM_WIDTH,
    SIM_HEIGHT,
    SIM_FPS,
    SIM_COLOR_GREEN,
    SIM_COLOR_YELLOW,
    SIM_COLOR_RED,
    SIM_COLOR_OFF,
    VEHICLE_COLORS,
    MIN_GREEN,
    MAX_GREEN,
)
from controller.state import IntersectionState
from simulation.vehicles import (
    VehicleManager,
    ISECT_LEFT, ISECT_RIGHT, ISECT_TOP, ISECT_BOTTOM,
    ISECT_CX, ISECT_CY,
    STOP_LINES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window / layout constants
# ---------------------------------------------------------------------------

WIN_W         = 1280
WIN_H         = 720
SIM_PANEL_W   = 720          # left panel — intersection simulation
CAM_PANEL_X   = SIM_PANEL_W  # right panel start x
CAM_PANEL_W   = WIN_W - SIM_PANEL_W
CAM_PANEL_H   = WIN_H

# Road geometry (within the 720×720 sim panel)
ROAD_WIDTH    = 80
SIM_CX        = SIM_PANEL_W // 2   # 360
SIM_CY        = WIN_H       // 2   # 360

# Scaled intersection box (centred in sim panel)
_SCALE        = SIM_CY / ISECT_CY  # ≈ 0.9
def _sx(x):   return int((x / ISECT_CX) * SIM_CX)
def _sy(y):   return int((y / ISECT_CY) * SIM_CY)
def _sv(v):   return int(v * (_sx(ISECT_CX) / ISECT_CX))

# Signal light positions (cx, cy) for each arm — placed at stop line entrance
_SIG_RADIUS   = 9
_SIG_POSITIONS: dict[str, tuple[int, int]] = {
    'North': (SIM_CX - ROAD_WIDTH // 2 - 18, _sy(ISECT_TOP)   - 18),
    'South': (SIM_CX + ROAD_WIDTH // 2 + 18, _sy(ISECT_BOTTOM) + 18),
    'East':  (_sx(ISECT_RIGHT)  + 18, SIM_CY - ROAD_WIDTH // 2 - 18),
    'West':  (_sx(ISECT_LEFT)   - 18, SIM_CY + ROAD_WIDTH // 2 + 18),
}

# Colours (RGB)
C_BG          = (28,  32,  38)
C_ROAD        = (62,  68,  78)
C_ROAD_EDGE   = (50,  55,  64)
C_ISECT       = (50,  55,  65)
C_LANE_MARK   = (180, 170,  60)
C_LANE_MARK2  = (80,   80,  80)
C_KERB        = (90,   95, 105)
C_PANEL_DIV   = (45,   50,  60)
C_TEXT_MAIN   = (230, 235, 245)
C_TEXT_DIM    = (130, 140, 155)
C_TEXT_GREEN  = ( 60, 220,  90)
C_TEXT_YELLOW = (240, 200,  40)
C_TEXT_RED    = (220,  60,  60)
C_TEXT_CYAN   = ( 60, 220, 220)
C_HBAR_BG     = ( 45,  50,  60)
C_EMRG_BG     = (160,  20,  20)
C_PED_BG      = (20,   90, 150)
C_HZRD_BG     = (160, 100,  20)

# Phase → text colour
_PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    'normal':      C_TEXT_GREEN,
    'emergency':   C_TEXT_RED,
    'pedestrian':  C_TEXT_CYAN,
    'all_red':     C_TEXT_RED,
    'yellow':      C_TEXT_YELLOW,
    'startup':     C_TEXT_DIM,
}


# ---------------------------------------------------------------------------
# Pygame Simulation
# ---------------------------------------------------------------------------

class PygameSimulation:
    """
    Main Pygame window and rendering engine.

    Run on the MAIN thread by calling run(). Reads IntersectionState
    continuously; never writes to it (except for keyboard-triggered
    simulate_* calls forwarded to EmergencyDetector).

    Keyboard controls:
        Q / ESC   → quit
        E         → simulate emergency (North arm)
        P         → simulate pedestrian rush
        1/2/3/4   → force green on N/S/E/W (debug)
        R         → reset all wait times
        D         → toggle debug overlay
        S         → save screenshot
        SPACE     → pause/resume detection thread flag
    """

    def __init__(
        self,
        state: IntersectionState,
        emergency_detector=None,   # EmergencyDetector | None
        title: str = "AI Smart Traffic System — Indian Cities",
    ) -> None:
        self._state    = state
        self._emrg_det = emergency_detector
        self._title    = title

        # Pygame surfaces / objects (initialised in _init_pygame)
        self._screen:  Optional[pygame.Surface] = None
        self._clock:   Optional[pygame.time.Clock] = None
        self._fonts:   dict[str, pygame.font.Font] = {}

        # Vehicle manager
        self._vehicles = VehicleManager()

        # State flags
        self._debug_overlay  = False
        self._paused         = False
        self._running        = True
        self._screenshot_n   = 0

        # FPS tracking
        self._fps_samples:   list[float] = []
        self._last_fps_time: float = time.time()
        self._display_fps:   float = 0.0

        # Scroll offset for camera panel label
        self._cam_label_x: float = CAM_PANEL_W

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialise Pygame and enter the render loop. Blocks until quit."""
        self._init_pygame()
        logger.info("Pygame simulation started — %dx%d @ %d FPS", WIN_W, WIN_H, SIM_FPS)

        while self._running:
            dt = self._clock.tick(SIM_FPS) / 1000.0
            dt = min(dt, 0.1)   # cap at 100ms to avoid physics explosion on resume

            self._handle_events()

            if not self._paused:
                self._update(dt)

            self._draw()
            pygame.display.flip()

        pygame.quit()
        logger.info("Pygame simulation stopped")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_caption(self._title)
        self._screen = pygame.display.set_mode((WIN_W, WIN_H))
        self._clock  = pygame.time.Clock()

        # Load fonts — use monospace for data readability
        self._fonts = {
            'sm':    pygame.font.SysFont('monospace', 12),
            'md':    pygame.font.SysFont('monospace', 14),
            'lg':    pygame.font.SysFont('monospace', 17),
            'xl':    pygame.font.SysFont('monospace', 22, bold=True),
            'title': pygame.font.SysFont('monospace', 11),
        }

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self._quit()

        elif key == pygame.K_e:
            # Simulate emergency on North arm
            if self._emrg_det:
                self._emrg_det.simulate_emergency('North')
            logger.info("[KEY] Simulated emergency — North arm")

        elif key == pygame.K_p:
            # Simulate pedestrian rush
            if self._emrg_det:
                self._emrg_det.simulate_ped_rush()
            logger.info("[KEY] Simulated pedestrian rush")

        elif key == pygame.K_d:
            self._debug_overlay = not self._debug_overlay
            logger.info("[KEY] Debug overlay: %s", self._debug_overlay)

        elif key == pygame.K_SPACE:
            self._paused = not self._paused
            logger.info("[KEY] Simulation %s", "paused" if self._paused else "resumed")

        elif key == pygame.K_r:
            with self._state.lock:
                for arm in self._state.arms.values():
                    arm.wait_time = 0.0
            logger.info("[KEY] Wait times reset")

        elif key == pygame.K_s:
            self._save_screenshot()

        elif key == pygame.K_1:
            self._force_green('North')
        elif key == pygame.K_2:
            self._force_green('South')
        elif key == pygame.K_3:
            self._force_green('East')
        elif key == pygame.K_4:
            self._force_green('West')

    def _force_green(self, arm: str) -> None:
        """Debug: manually force a specific arm green."""
        with self._state.lock:
            self._state.set_signal(None, 'RED')
            self._state.set_signal(arm, 'GREEN')
            self._state.current_green = arm
            self._state.phase = 'normal'
        logger.info("[KEY] Forced green: %s", arm)

    def _quit(self) -> None:
        self._running = False
        with self._state.lock:
            self._state.running = False

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        """Advance vehicle simulation by dt seconds."""
        arm_snapshot = self._state.snapshot_arms()
        self._vehicles.update(arm_snapshot, dt)

        # FPS tracking
        now = time.time()
        self._fps_samples.append(1.0 / max(dt, 0.001))
        if now - self._last_fps_time >= 1.0:
            self._display_fps = sum(self._fps_samples) / max(len(self._fps_samples), 1)
            self._fps_samples.clear()
            self._last_fps_time = now

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._screen.fill(C_BG)

        self._draw_sim_panel()
        self._draw_camera_panel()
        self._draw_divider()
        self._draw_title_bar()

        if self._paused:
            self._draw_pause_overlay()

    # ── Sim panel ────────────────────────────────────────────────────────

    def _draw_sim_panel(self) -> None:
        """Draw the left 720×720 top-down intersection."""
        surf = self._screen

        # Road arms
        road_rect_v = pygame.Rect(SIM_CX - ROAD_WIDTH // 2, 0, ROAD_WIDTH, WIN_H)
        road_rect_h = pygame.Rect(0, SIM_CY - ROAD_WIDTH // 2, SIM_PANEL_W, ROAD_WIDTH)
        pygame.draw.rect(surf, C_ROAD, road_rect_v)
        pygame.draw.rect(surf, C_ROAD, road_rect_h)

        # Kerb edges
        for offset in (ROAD_WIDTH // 2, -ROAD_WIDTH // 2):
            pygame.draw.line(surf, C_KERB,
                             (SIM_CX + offset, 0), (SIM_CX + offset, WIN_H), 2)
            pygame.draw.line(surf, C_KERB,
                             (0, SIM_CY + offset), (SIM_PANEL_W, SIM_CY + offset), 2)

        # Dashed centre lines
        self._draw_dashed_line(surf, C_LANE_MARK,
                               (SIM_CX, 0), (SIM_CX, _sy(ISECT_TOP) - 2),
                               dash=14, gap=8)
        self._draw_dashed_line(surf, C_LANE_MARK,
                               (SIM_CX, _sy(ISECT_BOTTOM) + 2), (SIM_CX, WIN_H),
                               dash=14, gap=8)
        self._draw_dashed_line(surf, C_LANE_MARK,
                               (0, SIM_CY), (_sx(ISECT_LEFT) - 2, SIM_CY),
                               dash=14, gap=8)
        self._draw_dashed_line(surf, C_LANE_MARK,
                               (_sx(ISECT_RIGHT) + 2, SIM_CY), (SIM_PANEL_W, SIM_CY),
                               dash=14, gap=8)

        # Intersection box
        isect_rect = pygame.Rect(
            _sx(ISECT_LEFT), _sy(ISECT_TOP),
            _sx(ISECT_RIGHT) - _sx(ISECT_LEFT),
            _sy(ISECT_BOTTOM) - _sy(ISECT_TOP),
        )
        pygame.draw.rect(surf, C_ISECT, isect_rect)
        pygame.draw.rect(surf, C_KERB, isect_rect, 1)

        # Stop lines
        with self._state.lock:
            signals = {name: arm.signal for name, arm in self._state.arms.items()}

        for arm, sig in signals.items():
            color = SIM_COLOR_GREEN if sig == 'GREEN' else (
                    SIM_COLOR_YELLOW if sig == 'YELLOW' else SIM_COLOR_RED)
            stop = STOP_LINES[arm]
            if arm in ('North', 'South'):
                sy_ = _sy(stop)
                pygame.draw.line(surf, color,
                                 (SIM_CX - ROAD_WIDTH // 2, sy_),
                                 (SIM_CX + ROAD_WIDTH // 2, sy_), 3)
            else:
                sx_ = _sx(stop)
                pygame.draw.line(surf, color,
                                 (sx_, SIM_CY - ROAD_WIDTH // 2),
                                 (sx_, SIM_CY + ROAD_WIDTH // 2), 3)

        # Vehicles
        for v in self._vehicles.all_vehicles():
            rx, ry, rw, rh = v.rect
            # Scale vehicle rect to sim panel
            srx = _sx(rx)
            sry = _sy(ry)
            srw = max(4, _sv(rw))
            srh = max(6, _sv(rh))
            vr  = pygame.Rect(srx, sry, srw, srh)
            pygame.draw.rect(surf, v.color, vr, border_radius=2)
            pygame.draw.rect(surf, (0, 0, 0), vr, 1, border_radius=2)

        # Signal lights
        self._draw_signal_lights(surf, signals)

        # Phase + arm info HUD (bottom-left of sim panel)
        self._draw_sim_hud(surf)

        # Debug overlay
        if self._debug_overlay:
            self._draw_debug_info(surf)

    def _draw_signal_lights(
        self, surf: pygame.Surface, signals: dict[str, str]
    ) -> None:
        """Draw 3-lamp traffic light at each arm entrance."""
        for arm in ARM_NAMES:
            sig = signals.get(arm, 'RED')
            cx, cy = _SIG_POSITIONS[arm]

            # Housing background
            housing_w = _SIG_RADIUS * 2 + 8
            housing_h = _SIG_RADIUS * 6 + 12
            # Orient housing vertically for N/S arms, horizontally for E/W
            if arm in ('North', 'South'):
                hw, hh = housing_w, housing_h
            else:
                hw, hh = housing_h, housing_w

            hx = cx - hw // 2
            hy = cy - hh // 2
            pygame.draw.rect(surf, (20, 22, 26), (hx, hy, hw, hh), border_radius=4)
            pygame.draw.rect(surf, (45, 48, 55), (hx, hy, hw, hh), 1, border_radius=4)

            # Lamp positions — R, Y, G
            if arm in ('North', 'South'):
                lamp_positions = [
                    (cx, cy - _SIG_RADIUS * 2),   # Red (top)
                    (cx, cy),                       # Yellow (mid)
                    (cx, cy + _SIG_RADIUS * 2),    # Green (bottom)
                ]
            else:
                lamp_positions = [
                    (cx - _SIG_RADIUS * 2, cy),   # Red (left)
                    (cx, cy),                       # Yellow (mid)
                    (cx + _SIG_RADIUS * 2, cy),    # Green (right)
                ]

            lamp_states = [
                ('RED',    SIM_COLOR_RED,    (60, 15, 15)),
                ('YELLOW', SIM_COLOR_YELLOW, (55, 45, 10)),
                ('GREEN',  SIM_COLOR_GREEN,  (10, 55, 20)),
            ]

            for (lx, ly), (phase, on_color, off_color) in zip(lamp_positions, lamp_states):
                active = (sig == phase) or (sig == 'WALK' and phase == 'RED')
                color  = on_color if active else off_color
                pygame.draw.circle(surf, color, (lx, ly), _SIG_RADIUS)
                if active:
                    # Glow effect
                    glow_surf = pygame.Surface(
                        (_SIG_RADIUS * 4, _SIG_RADIUS * 4), pygame.SRCALPHA
                    )
                    glow_color = (*on_color, 60)
                    pygame.draw.circle(
                        glow_surf, glow_color,
                        (_SIG_RADIUS * 2, _SIG_RADIUS * 2), _SIG_RADIUS * 2
                    )
                    surf.blit(glow_surf, (lx - _SIG_RADIUS * 2, ly - _SIG_RADIUS * 2))

            # Arm label
            label = self._fonts['title'].render(arm[:1], True, C_TEXT_DIM)
            surf.blit(label, (cx - 4, cy + _SIG_RADIUS * 3 + 4))

    def _draw_sim_hud(self, surf: pygame.Surface) -> None:
        """Draw arm metrics at the bottom-left of the simulation panel."""
        phase_snap = self._state.snapshot_phase()
        arm_snap   = self._state.snapshot_arms()

        phase        = phase_snap.get('phase', 'normal')
        current_green = phase_snap.get('current_green')
        phase_color  = _PHASE_COLORS.get(phase, C_TEXT_DIM)

        # Background panel
        hud_x, hud_y = 8, WIN_H - 130
        hud_w, hud_h = 380, 120
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((15, 18, 24, 200))
        surf.blit(hud_surf, (hud_x, hud_y))
        pygame.draw.rect(surf, C_PANEL_DIV, (hud_x, hud_y, hud_w, hud_h), 1)

        # Phase label
        phase_label = self._fonts['md'].render(
            f"PHASE: {phase.upper():<12}  cycles={phase_snap.get('total_cycles', 0)}",
            True, phase_color,
        )
        surf.blit(phase_label, (hud_x + 6, hud_y + 5))

        # Per-arm rows
        for i, arm in enumerate(ARM_NAMES):
            s      = arm_snap.get(arm, {})
            sig    = s.get('signal', 'RED')
            dens   = s.get('density', 0.0)
            wait   = s.get('wait_time', 0.0)
            flow   = s.get('flow_rate', 0.0)
            emrg   = s.get('emergency', False)
            hzrd   = s.get('hazard', False)
            is_grn = (arm == current_green)

            sig_color = (
                C_TEXT_GREEN  if sig == 'GREEN'  else
                C_TEXT_YELLOW if sig == 'YELLOW' else
                C_TEXT_RED
            )
            row_y  = hud_y + 26 + i * 22

            # Dot
            dot_color = SIM_COLOR_GREEN if is_grn else SIM_COLOR_RED
            pygame.draw.circle(surf, dot_color, (hud_x + 14, row_y + 7), 5)

            # Signal badge
            badge = self._fonts['sm'].render(f"{sig[0]}", True, sig_color)
            surf.blit(badge, (hud_x + 26, row_y))

            # Arm data
            flags = (" 🚨" if emrg else "") + (" ⚠" if hzrd else "")
            row_text = (
                f"{arm:<6}  d={dens:5.1f}  w={int(wait):3d}s  "
                f"flow={flow:4.1f}{flags}"
            )
            col = C_TEXT_GREEN if is_grn else C_TEXT_MAIN
            row_label = self._fonts['sm'].render(row_text, True, col)
            surf.blit(row_label, (hud_x + 38, row_y))

            # Density bar (mini)
            bar_x   = hud_x + 340
            bar_w   = 28
            bar_h   = 10
            fill    = int(min(1.0, dens / 50.0) * bar_w)
            bar_col = _density_color(dens / 50.0)
            pygame.draw.rect(surf, C_HBAR_BG, (bar_x, row_y + 2, bar_w, bar_h))
            if fill > 0:
                pygame.draw.rect(surf, bar_col, (bar_x, row_y + 2, fill, bar_h))
            pygame.draw.rect(surf, C_TEXT_DIM, (bar_x, row_y + 2, bar_w, bar_h), 1)

        # FPS
        fps_label = self._fonts['title'].render(
            f"SIM {self._display_fps:.0f} FPS  vehicles={len(self._vehicles.all_vehicles())}",
            True, C_TEXT_DIM,
        )
        surf.blit(fps_label, (hud_x + 6, hud_y + hud_h - 14))

    def _draw_debug_info(self, surf: pygame.Surface) -> None:
        """Debug overlay: queue lengths, vehicle IDs, spawn accumulators."""
        counts = self._vehicles.vehicle_count_by_arm()
        queued = self._vehicles.queued_count_by_arm()
        y = 8
        for arm in ARM_NAMES:
            txt = self._fonts['sm'].render(
                f"[D] {arm:<6} total={counts[arm]:2d}  queued={queued[arm]:2d}",
                True, (100, 200, 255),
            )
            surf.blit(txt, (8, y))
            y += 15

        cleared_txt = self._fonts['sm'].render(
            f"[D] cleared={self._vehicles.total_cleared()}",
            True, (100, 200, 255),
        )
        surf.blit(cleared_txt, (8, y))

    # ── Camera panel ─────────────────────────────────────────────────────

    def _draw_camera_panel(self) -> None:
        """Draw the right panel: annotated camera feed + metrics."""
        surf  = self._screen
        frame = self._state.get_annotated_frame()

        # Camera feed area: 560 wide × 480 tall (16:9-ish)
        cam_h    = 420
        cam_rect = pygame.Rect(CAM_PANEL_X, 30, CAM_PANEL_W, cam_h)

        if frame is not None:
            try:
                # OpenCV BGR → RGB → Pygame surface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (CAM_PANEL_W, cam_h))
                cam_surf  = pygame.surfarray.make_surface(
                    frame_rgb.swapaxes(0, 1)
                )
                surf.blit(cam_surf, cam_rect.topleft)
            except Exception as exc:
                logger.debug("Camera frame render error: %s", exc)
                self._draw_no_camera(surf, cam_rect)
        else:
            self._draw_no_camera(surf, cam_rect)

        pygame.draw.rect(surf, C_PANEL_DIV, cam_rect, 1)

        # Alert banner (overlaid on camera feed)
        self._draw_alert_banner(surf, cam_rect)

        # Metrics sidebar below camera
        self._draw_metrics_sidebar(surf, cam_rect.bottom + 8)

    def _draw_no_camera(
        self, surf: pygame.Surface, rect: pygame.Rect
    ) -> None:
        """Placeholder when no camera frame is available yet."""
        pygame.draw.rect(surf, (20, 22, 28), rect)
        msg = self._fonts['md'].render(
            "Waiting for camera feed...", True, C_TEXT_DIM
        )
        surf.blit(msg, (
            rect.x + rect.w // 2 - msg.get_width() // 2,
            rect.y + rect.h // 2,
        ))

    def _draw_alert_banner(
        self, surf: pygame.Surface, cam_rect: pygame.Rect
    ) -> None:
        """Overlay a full-width alert banner on the camera feed when active."""
        phase_snap = self._state.snapshot_phase()
        arm_snap   = self._state.snapshot_arms()

        phase         = phase_snap.get('phase', 'normal')
        current_green = phase_snap.get('current_green')

        # Check emergency
        emrg_arm = next(
            (arm for arm, s in arm_snap.items() if s.get('emergency')), None
        )
        # Check hazard
        hzrd_info = next(
            ((arm, s.get('hazard')) for arm, s in arm_snap.items() if s.get('hazard')),
            None,
        )

        banner_h = 34
        banner_rect = pygame.Rect(cam_rect.x, cam_rect.y, cam_rect.w, banner_h)

        if phase == 'emergency' and emrg_arm:
            banner_surf = pygame.Surface((cam_rect.w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((*C_EMRG_BG, 230))
            surf.blit(banner_surf, banner_rect.topleft)
            txt = self._fonts['lg'].render(
                f"  🚨  EMERGENCY OVERRIDE — {emrg_arm.upper()} ARM PRIORITY",
                True, (255, 255, 255),
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 8))

        elif phase == 'pedestrian':
            banner_surf = pygame.Surface((cam_rect.w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((*C_PED_BG, 220))
            surf.blit(banner_surf, banner_rect.topleft)
            ped_avg = phase_snap.get('ped_rolling_avg', 0.0)
            txt = self._fonts['lg'].render(
                f"  🚶  PEDESTRIAN PHASE ACTIVE  ({ped_avg:.0f} persons detected)",
                True, (255, 255, 255),
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 8))

        elif hzrd_info:
            arm, _ = hzrd_info
            banner_surf = pygame.Surface((cam_rect.w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((*C_HZRD_BG, 220))
            surf.blit(banner_surf, banner_rect.topleft)
            txt = self._fonts['lg'].render(
                f"  ⚠  ANIMAL ON ROAD — {arm.upper()} ARM  (+5s extension)",
                True, (255, 255, 255),
            )
            surf.blit(txt, (cam_rect.x + 6, cam_rect.y + 8))

    def _draw_metrics_sidebar(self, surf: pygame.Surface, y_start: int) -> None:
        """Draw arm metrics and system stats below the camera feed."""
        arm_snap   = self._state.snapshot_arms()
        phase_snap = self._state.snapshot_phase()

        x0    = CAM_PANEL_X + 6
        y     = y_start
        col_w = (CAM_PANEL_W - 12) // 4

        # Column headers
        headers = ['ARM', 'DENSITY', 'WAIT', 'SIGNAL']
        for i, h in enumerate(headers):
            lbl = self._fonts['title'].render(h, True, C_TEXT_DIM)
            surf.blit(lbl, (x0 + i * col_w, y))
        y += 14
        pygame.draw.line(surf, C_PANEL_DIV,
                         (x0, y), (x0 + CAM_PANEL_W - 12, y), 1)
        y += 4

        for arm in ARM_NAMES:
            s      = arm_snap.get(arm, {})
            sig    = s.get('signal', 'RED')
            dens   = s.get('density', 0.0)
            wait   = s.get('wait_time', 0.0)
            is_grn = (arm == phase_snap.get('current_green'))

            sig_color = (
                C_TEXT_GREEN  if sig == 'GREEN'  else
                C_TEXT_YELLOW if sig == 'YELLOW' else
                C_TEXT_RED
            )
            row_color = C_TEXT_GREEN if is_grn else C_TEXT_MAIN

            cols = [arm[:5], f"{dens:5.1f}", f"{int(wait):3d}s", sig]
            colors = [row_color, row_color, row_color, sig_color]

            for i, (col_txt, col_color) in enumerate(zip(cols, colors)):
                lbl = self._fonts['sm'].render(col_txt, True, col_color)
                surf.blit(lbl, (x0 + i * col_w, y))

            # Mini density bar
            bar_x  = x0 + col_w + 50
            bar_w  = 50
            bar_h  = 8
            fill   = int(min(1.0, dens / 50.0) * bar_w)
            pygame.draw.rect(surf, C_HBAR_BG, (bar_x, y + 2, bar_w, bar_h))
            if fill > 0:
                pygame.draw.rect(surf, _density_color(dens / 50.0),
                                 (bar_x, y + 2, fill, bar_h))
            pygame.draw.rect(surf, C_TEXT_DIM, (bar_x, y + 2, bar_w, bar_h), 1)

            y += 18

        y += 4
        pygame.draw.line(surf, C_PANEL_DIV,
                         (x0, y), (x0 + CAM_PANEL_W - 12, y), 1)
        y += 6

        # System stats
        uptime  = phase_snap.get('uptime_s', 0.0)
        cleared = self._vehicles.total_cleared()
        cycles  = phase_snap.get('total_cycles', 0)

        stats = [
            f"Uptime:   {int(uptime // 60):02d}:{int(uptime % 60):02d}",
            f"Cycles:   {cycles}",
            f"Cleared:  {cleared} vehicles",
            f"Sim FPS:  {self._display_fps:.0f}",
        ]
        for stat in stats:
            lbl = self._fonts['sm'].render(stat, True, C_TEXT_DIM)
            surf.blit(lbl, (x0, y))
            y += 15

        # Key bindings hint
        y += 4
        hints = "E:emergency  P:ped  D:debug  SPACE:pause  S:screenshot  Q:quit"
        hint_lbl = self._fonts['title'].render(hints, True, C_TEXT_DIM)
        surf.blit(hint_lbl, (x0, WIN_H - 14))

    # ── Shared decorations ───────────────────────────────────────────────

    def _draw_divider(self) -> None:
        """Vertical divider between sim panel and camera panel."""
        pygame.draw.line(
            self._screen, C_PANEL_DIV,
            (SIM_PANEL_W, 0), (SIM_PANEL_W, WIN_H), 2
        )

    def _draw_title_bar(self) -> None:
        """Slim title bar at the top of the camera panel."""
        title_surf = pygame.Surface((CAM_PANEL_W, 28), pygame.SRCALPHA)
        title_surf.fill((18, 20, 26, 240))
        self._screen.blit(title_surf, (CAM_PANEL_X, 0))

        title_lbl = self._fonts['md'].render(
            "AI Traffic System — Live Detection Feed", True, C_TEXT_MAIN
        )
        self._screen.blit(title_lbl, (CAM_PANEL_X + 8, 6))

        fps_lbl = self._fonts['sm'].render(
            f"{self._display_fps:.0f} FPS", True, C_TEXT_DIM
        )
        self._screen.blit(fps_lbl,
                          (CAM_PANEL_X + CAM_PANEL_W - fps_lbl.get_width() - 8, 8))

    def _draw_pause_overlay(self) -> None:
        """Semi-transparent PAUSED overlay."""
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (0, 0))
        msg = self._fonts['xl'].render("PAUSED — SPACE to resume", True, C_TEXT_YELLOW)
        self._screen.blit(msg, (
            WIN_W // 2 - msg.get_width() // 2,
            WIN_H // 2 - msg.get_height() // 2,
        ))

    # ── Drawing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _draw_dashed_line(
        surf: pygame.Surface,
        color: tuple,
        start: tuple[int, int],
        end: tuple[int, int],
        dash: int = 10,
        gap: int  = 6,
        width: int = 1,
    ) -> None:
        """Draw a dashed line between two points."""
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = max(1, int(math.hypot(dx, dy)))
        step   = dash + gap
        nx, ny = dx / length, dy / length

        pos = 0
        while pos < length:
            seg_end = min(pos + dash, length)
            sx = int(x0 + nx * pos)
            sy = int(y0 + ny * pos)
            ex = int(x0 + nx * seg_end)
            ey = int(y0 + ny * seg_end)
            pygame.draw.line(surf, color, (sx, sy), (ex, ey), width)
            pos += step

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def _save_screenshot(self) -> None:
        self._screenshot_n += 1
        fname = f"screenshot_{self._screenshot_n:03d}.png"
        pygame.image.save(self._screen, fname)
        logger.info("Screenshot saved: %s", fname)


# ---------------------------------------------------------------------------
# Pure colour utility
# ---------------------------------------------------------------------------

import math

def _density_color(ratio: float) -> tuple[int, int, int]:
    """Green → yellow → red gradient for density bars (RGB)."""
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.5:
        t = ratio * 2.0
        return (int(t * 220), 220, int((1 - t) * 60))
    else:
        t = (ratio - 0.5) * 2.0
        return (220, int((1 - t) * 220), 0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_simulation(
    state: IntersectionState,
    emergency_detector=None,
) -> PygameSimulation:
    """
    Create a PygameSimulation ready to run.

    Args:
        state:              Shared IntersectionState.
        emergency_detector: EmergencyDetector instance for keyboard shortcuts.

    Returns:
        PygameSimulation instance. Call .run() from the main thread.
    """
    return PygameSimulation(state=state, emergency_detector=emergency_detector)


# ---------------------------------------------------------------------------
# Standalone test  (python -m simulation.pygame_sim)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import threading
    from controller.state import create_state
    from detection.emergency import EmergencyDetector

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    state    = create_state()
    emrg_det = EmergencyDetector()

    # Inject mock data and cycle green arm in a background thread
    def mock_cycle():
        arms  = ARM_NAMES
        idx   = 0
        while True:
            arm = arms[idx % len(arms)]
            with state.lock:
                state.set_signal(None, 'RED')
                state.set_signal(arm, 'GREEN')
                state.current_green = arm
                state.phase         = 'normal'
                # Vary density
                for i, a in enumerate(arms):
                    state.arms[a].density   = (10 + i * 7 + (idx % 5) * 3) % 40
                    state.arms[a].flow_rate = 2.0 if a == arm else 0.4
                    state.arms[a].wait_time = 0.0 if a == arm else state.arms[a].wait_time + 8
                state.total_cycles += 1
            time.sleep(8)
            with state.lock:
                state.set_signal(arm, 'YELLOW')
            time.sleep(3)
            idx += 1

    t = threading.Thread(target=mock_cycle, daemon=True)
    t.start()

    sim = create_simulation(state, emergency_detector=emrg_det)
    sim.run()