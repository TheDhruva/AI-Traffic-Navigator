# hardware/arduino.py — PySerial Interface for Arduino Signal Controller
# Sends signal commands to Arduino Uno over USB serial.
# Gracefully degrades to simulation mode if Arduino is not connected.
#
# Command format (matches traffic_signals.ino):
#   "{ARM_INITIAL}:{PHASE}:{DURATION}\n"
#   e.g. "N:GREEN:30\n", "A:RED:0\n", "P:WALK:15\n"
#
# Arduino responds (optional):
#   "ACK:{ARM}:{PHASE}\n"

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "pyserial not installed — running in simulation mode. "
        "Install with: pip install pyserial"
    )

from config import (
    SERIAL_PORT,
    SERIAL_BAUD,
    SERIAL_TIMEOUT,
    ARDUINO_BOOT_DELAY,
    ARM_NAMES,
)

logger = logging.getLogger(__name__)

# Arm name → single-character initial used in serial protocol
ARM_INITIALS: dict[str, str] = {
    'North': 'N',
    'South': 'S',
    'East':  'E',
    'West':  'W',
    'ALL':   'A',
    'PED':   'P',
}

# Valid phase strings accepted by the Arduino sketch
VALID_PHASES = {'GREEN', 'YELLOW', 'RED', 'WALK'}


# ---------------------------------------------------------------------------
# Command dataclass
# ---------------------------------------------------------------------------

class SignalCommand:
    """Represents one serial command to send to the Arduino."""

    __slots__ = ('arm', 'phase', 'duration', 'timestamp')

    def __init__(self, arm: str, phase: str, duration: int = 0) -> None:
        """
        Args:
            arm:      Arm initial 'N'|'S'|'E'|'W'|'A'|'P'
                      OR full name 'North'|'South'|'East'|'West'|'ALL'|'PED'.
            phase:    'GREEN'|'YELLOW'|'RED'|'WALK'
            duration: Seconds (informational — Arduino doesn't enforce timing).
        """
        # Normalise arm to single initial
        if len(arm) > 1:
            self.arm = ARM_INITIALS.get(arm, arm[0].upper())
        else:
            self.arm = arm.upper()

        self.phase    = phase.upper()
        self.duration = int(duration)
        self.timestamp = time.time()

    @property
    def serial_string(self) -> str:
        """Return the wire-format command string including newline."""
        return f"{self.arm}:{self.phase}:{self.duration}\n"

    def __repr__(self) -> str:
        return f"SignalCommand({self.serial_string.strip()!r})"


# ---------------------------------------------------------------------------
# Arduino interface
# ---------------------------------------------------------------------------

class ArduinoController:
    """
    Manages the serial connection to the Arduino Uno.

    Runs an internal writer thread that drains a command queue,
    ensuring the controller thread never blocks waiting for serial I/O.

    Usage:
        arduino = ArduinoController()
        arduino.connect()                         # safe no-op if not found
        arduino.send('N', 'GREEN', 30)
        arduino.send_all_red()
        arduino.disconnect()

    Simulation mode (no Arduino):
        All send calls are no-ops that log at DEBUG level.
        The controller thread works identically — no code changes needed.
    """

    # How long to wait for an ACK from Arduino before giving up
    ACK_TIMEOUT = 2.0

    # Maximum commands to buffer before dropping oldest (prevents memory growth)
    QUEUE_MAX = 32

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = SERIAL_BAUD,
        timeout: float = SERIAL_TIMEOUT,
        boot_delay: float = ARDUINO_BOOT_DELAY,
        auto_connect: bool = True,
    ) -> None:
        """
        Args:
            port:         Serial port string (e.g. 'COM3', '/dev/ttyACM0').
            baud:         Baud rate — must match Arduino sketch (9600).
            timeout:      Serial read timeout in seconds.
            boot_delay:   Seconds to wait after opening port for Arduino reset.
            auto_connect: If True, attempt connection in __init__.
        """
        self._port        = port
        self._baud        = baud
        self._timeout     = timeout
        self._boot_delay  = boot_delay

        self._serial: Optional[serial.Serial] = None  # type: ignore[name-defined]
        self._connected   = False
        self._simulation  = not SERIAL_AVAILABLE

        # Thread-safe command queue
        self._cmd_queue: queue.Queue[Optional[SignalCommand]] = queue.Queue(
            maxsize=self.QUEUE_MAX
        )

        # Writer thread — drains _cmd_queue and writes to serial
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False

        # Statistics
        self._sent_count  = 0
        self._error_count = 0
        self._last_ack    = ""
        self._last_sent   = ""

        # Lock for stats reads (not for I/O — writer thread owns the port)
        self._stats_lock = threading.Lock()

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Attempt to open the serial port and start the writer thread.

        Returns:
            True if connected successfully, False if in simulation mode.
        """
        if self._simulation:
            logger.info("Arduino: simulation mode (pyserial not available)")
            self._start_writer()
            return False

        if self._connected:
            logger.warning("Arduino: already connected on %s", self._port)
            return True

        # Auto-detect port if configured port not found
        port = self._resolve_port(self._port)
        if port is None:
            logger.warning(
                "Arduino not found on %s — trying auto-detect...", self._port
            )
            port = self._auto_detect_port()

        if port is None:
            logger.warning(
                "Arduino not detected — falling back to simulation mode. "
                "Pygame simulation will still work correctly."
            )
            self._simulation = True
            self._start_writer()
            return False

        try:
            self._serial = serial.Serial(  # type: ignore[name-defined]
                port=port,
                baudrate=self._baud,
                timeout=self._timeout,
            )
            time.sleep(self._boot_delay)   # wait for Arduino reset on connect

            # Read READY message from Arduino startup
            ready_line = self._serial.readline().decode('utf-8', errors='ignore').strip()
            if ready_line == 'READY':
                logger.info(
                    "Arduino connected on %s @ %d baud — got READY", port, self._baud
                )
            else:
                logger.warning(
                    "Arduino on %s — expected READY, got %r", port, ready_line
                )

            self._connected = True
            self._port = port
            self._start_writer()
            return True

        except Exception as exc:  # serial.SerialException + OSError
            logger.warning(
                "Cannot open %s (%s) — simulation mode active", port, exc
            )
            self._simulation = True
            self._start_writer()
            return False

    def disconnect(self) -> None:
        """Stop the writer thread and close the serial port cleanly."""
        self._running = False
        # Poison pill — unblocks writer thread if it's waiting on queue.get()
        self._cmd_queue.put(None)

        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)

        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass

        self._connected = False
        logger.info("Arduino disconnected")

    def reconnect(self) -> bool:
        """Disconnect and reconnect — useful after USB unplug/replug."""
        self.disconnect()
        time.sleep(1.0)
        self._simulation  = not SERIAL_AVAILABLE
        self._connected   = False
        return self.connect()

    # ------------------------------------------------------------------
    # Public send API
    # ------------------------------------------------------------------

    def send(self, arm: str, phase: str, duration: int = 0) -> None:
        """
        Queue a signal command for the Arduino.

        Thread-safe. Returns immediately — I/O happens in the writer thread.

        Args:
            arm:      'N'|'S'|'E'|'W'|'A'|'P' or full arm name.
            phase:    'GREEN'|'YELLOW'|'RED'|'WALK'
            duration: Seconds (stored for logging; Arduino doesn't enforce it).
        """
        cmd = SignalCommand(arm, phase, duration)

        if cmd.phase not in VALID_PHASES:
            logger.error("Invalid phase %r — command dropped", phase)
            return

        try:
            self._cmd_queue.put_nowait(cmd)
        except queue.Full:
            # Drop oldest command to make room — a stale RED is safer than
            # a blocked queue that holds up the controller thread
            try:
                self._cmd_queue.get_nowait()
                self._cmd_queue.put_nowait(cmd)
                logger.warning("Command queue full — dropped oldest command")
            except queue.Empty:
                pass

    def send_all_red(self) -> None:
        """Convenience: set all arms RED immediately."""
        self.send('A', 'RED', 0)

    def send_arm(self, arm: str, phase: str, duration: int = 0) -> None:
        """Convenience: send command for a single named arm."""
        self.send(arm, phase, duration)

    def send_pedestrian_walk(self) -> None:
        """Convenience: activate pedestrian WALK signal."""
        self.send('P', 'WALK', 0)

    def send_pedestrian_off(self) -> None:
        """Convenience: deactivate pedestrian WALK signal."""
        self.send('P', 'RED', 0)

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if physically connected to Arduino (not simulation mode)."""
        return self._connected and not self._simulation

    @property
    def is_simulation(self) -> bool:
        """True if running without hardware."""
        return self._simulation

    def stats(self) -> dict:
        """Return a snapshot of send statistics."""
        with self._stats_lock:
            return {
                'connected':    self._connected,
                'simulation':   self._simulation,
                'port':         self._port,
                'sent_count':   self._sent_count,
                'error_count':  self._error_count,
                'queue_size':   self._cmd_queue.qsize(),
                'last_sent':    self._last_sent,
                'last_ack':     self._last_ack,
            }

    def get_send_callback(self):
        """
        Return a (arm, phase, duration) → None callback suitable for
        passing directly to SignalController.__init__().

        Usage:
            arduino = ArduinoController()
            controller = SignalController(state, send_command=arduino.get_send_callback())
        """
        return lambda arm, phase, duration: self.send(arm, phase, duration)

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _start_writer(self) -> None:
        """Start the background serial writer thread."""
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="ArduinoWriter",
            daemon=True,
        )
        self._writer_thread.start()
        logger.debug("Arduino writer thread started")

    def _writer_loop(self) -> None:
        """
        Drains the command queue and writes commands to serial.
        Runs until self._running is False and a None sentinel is received.
        """
        while self._running:
            try:
                cmd = self._cmd_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if cmd is None:
                # Poison pill — exit
                break

            self._write_command(cmd)

        logger.debug("Arduino writer thread exiting")

    def _write_command(self, cmd: SignalCommand) -> None:
        """
        Write a single command to the serial port and optionally read ACK.
        Simulation mode just logs at DEBUG level.
        """
        wire = cmd.serial_string

        if self._simulation:
            logger.debug("SIM → %s", wire.strip())
            with self._stats_lock:
                self._sent_count += 1
                self._last_sent   = wire.strip()
            return

        try:
            self._serial.write(wire.encode('utf-8'))
            self._serial.flush()

            with self._stats_lock:
                self._sent_count += 1
                self._last_sent   = wire.strip()

            logger.debug("→ Arduino: %s", wire.strip())

            # Try to read ACK (non-blocking within timeout)
            ack = self._read_ack()
            if ack:
                with self._stats_lock:
                    self._last_ack = ack
                logger.debug("← Arduino: %s", ack)
            else:
                logger.debug("No ACK for %s (may be normal)", wire.strip())

        except Exception as exc:
            with self._stats_lock:
                self._error_count += 1
            logger.error("Serial write error (%s) — cmd: %s", exc, wire.strip())

            # Attempt reconnect on persistent errors
            if self._error_count % 5 == 0:
                logger.warning("5 consecutive errors — attempting reconnect")
                self._try_reconnect()

    def _read_ack(self) -> str:
        """
        Read one line from Arduino (the ACK response).
        Returns empty string if nothing arrives within ACK_TIMEOUT.
        """
        if not self._serial or not self._serial.is_open:
            return ""
        try:
            self._serial.timeout = self.ACK_TIMEOUT
            line = self._serial.readline()
            return line.decode('utf-8', errors='ignore').strip()
        except Exception:
            return ""

    def _try_reconnect(self) -> None:
        """Non-blocking reconnect attempt from within the writer thread."""
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
            time.sleep(1.0)
            self._serial = serial.Serial(  # type: ignore[name-defined]
                port=self._port,
                baudrate=self._baud,
                timeout=self._timeout,
            )
            time.sleep(self._boot_delay)
            logger.info("Arduino reconnected on %s", self._port)
            with self._stats_lock:
                self._error_count = 0
        except Exception as exc:
            logger.error("Reconnect failed: %s", exc)

    # ------------------------------------------------------------------
    # Port detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_port(port: str) -> Optional[str]:
        """
        Check if the given port exists in the system's port list.
        Returns the port string if found, None otherwise.
        """
        if not SERIAL_AVAILABLE:
            return None
        available = [p.device for p in serial.tools.list_ports.comports()]  # type: ignore
        return port if port in available else None

    @staticmethod
    def _auto_detect_port() -> Optional[str]:
        """
        Scan all serial ports for an Arduino-like device.
        Identifies by USB vendor description containing 'Arduino' or 'CH340'.

        Returns:
            Port string if found, None otherwise.
        """
        if not SERIAL_AVAILABLE:
            return None
        for port_info in serial.tools.list_ports.comports():  # type: ignore
            desc = (port_info.description or "").lower()
            mfr  = (port_info.manufacturer or "").lower()
            if any(k in desc or k in mfr for k in ('arduino', 'ch340', 'ftdi', 'acm')):
                logger.info(
                    "Auto-detected Arduino on %s (%s)",
                    port_info.device, port_info.description,
                )
                return port_info.device
        return None

    @staticmethod
    def list_ports() -> list[dict]:
        """
        Return a list of all available serial ports with descriptions.
        Useful for debugging 'which port is my Arduino on?'
        """
        if not SERIAL_AVAILABLE:
            return []
        return [
            {
                'device':      p.device,
                'description': p.description,
                'manufacturer': p.manufacturer,
                'hwid':        p.hwid,
            }
            for p in serial.tools.list_ports.comports()
        ]

    def __repr__(self) -> str:
        mode = "hardware" if self.is_connected else "simulation"
        return f"ArduinoController({mode} port={self._port!r} sent={self._sent_count})"

    # Context manager support
    def __enter__(self) -> "ArduinoController":
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Module-level factory used by main.py
# ---------------------------------------------------------------------------

def create_arduino(
    port: str = SERIAL_PORT,
    auto_connect: bool = True,
) -> ArduinoController:
    """
    Create and return an ArduinoController.
    Safe to call regardless of whether Arduino hardware is present.

    Args:
        port:         Serial port override. Defaults to config.SERIAL_PORT.
        auto_connect: Attempt connection immediately.

    Returns:
        ArduinoController instance (hardware or simulation mode).
    """
    return ArduinoController(port=port, auto_connect=auto_connect)


# ---------------------------------------------------------------------------
# Standalone test  (python -m hardware.arduino)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    print("=== Arduino Controller Test ===\n")

    # List available ports
    ports = ArduinoController.list_ports()
    if ports:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p['device']:<16} {p['description']}")
    else:
        print("No serial ports found — will run in simulation mode\n")

    # Connect (auto-falls back to sim if no Arduino)
    arduino = create_arduino()
    print(f"\n{arduino}\n")

    # Run test sequence
    test_commands = [
        ('A', 'RED',    0,  "All red"),
        ('N', 'GREEN',  10, "North green"),
        ('N', 'YELLOW', 3,  "North yellow"),
        ('N', 'RED',    0,  "North red"),
        ('S', 'GREEN',  10, "South green"),
        ('S', 'YELLOW', 3,  "South yellow"),
        ('S', 'RED',    0,  "South red"),
        ('E', 'GREEN',  10, "East green"),
        ('E', 'RED',    0,  "East red"),
        ('W', 'GREEN',  10, "West green"),
        ('W', 'RED',    0,  "West red"),
        ('P', 'WALK',   15, "Pedestrian WALK"),
        ('A', 'RED',    0,  "All red (final)"),
    ]

    print("Sending test sequence (0.5s between commands):\n")
    for arm, phase, dur, label in test_commands:
        print(f"  {label:<22} → {arm}:{phase}:{dur}")
        arduino.send(arm, phase, dur)
        time.sleep(0.5)

    # Wait for queue to drain
    time.sleep(1.0)

    print("\nStats:")
    for k, v in arduino.stats().items():
        print(f"  {k:<16} {v}")

    arduino.disconnect()
    print("\nTest complete.")