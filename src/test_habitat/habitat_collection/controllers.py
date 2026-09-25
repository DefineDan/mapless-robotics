from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class ControlCommand:
    """A single high-level control command consumed by the collection env."""

    action: Optional[str]
    label: str
    should_quit: bool = False
    should_reset: bool = False


class Controller(Protocol):
    """Controller interface for keyboard, policy, or scripted control."""

    def next_command(self) -> ControlCommand:
        """Return the next command to apply."""

    def send_result(self, record: Dict[str, Any]) -> None:
        """Optionally return per-step state to the controller frontend."""


class KeyboardController:
    """Blocking terminal keyboard controller."""

    KEYMAP = {
        "w": ControlCommand("move_forward", "forward"),
        "a": ControlCommand("turn_left", "turn_left"),
        "d": ControlCommand("turn_right", "turn_right"),
        "s": ControlCommand(None, "record"),
        " ": ControlCommand(None, "record"),
        "r": ControlCommand(None, "reset", should_reset=True),
        "q": ControlCommand(None, "quit", should_quit=True),
        "\x1b": ControlCommand(None, "quit", should_quit=True),
    }

    def __init__(self) -> None:
        import termios
        import tty

        self._termios = termios
        self._tty = tty
        self._fd = None
        self._old_settings = None

    def __enter__(self) -> "KeyboardController":
        import sys

        self._fd = sys.stdin.fileno()
        self._old_settings = self._termios.tcgetattr(self._fd)
        self._tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            self._termios.tcsetattr(
                self._fd, self._termios.TCSADRAIN, self._old_settings
            )

    def next_command(self) -> ControlCommand:
        import sys

        while True:
            key = sys.stdin.read(1).lower()
            command = self.KEYMAP.get(key)
            if command is not None:
                return command

    def send_result(self, record: Dict[str, Any]) -> None:
        return None


class TcpKeyboardController:
    """Single-client TCP keyboard controller for use over Tailscale or SSH."""

    KEYMAP = KeyboardController.KEYMAP

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    def __enter__(self) -> "TcpKeyboardController":
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        print(f"Waiting for remote keyboard on {self.host}:{self.port}...")
        self._conn, addr = self._server.accept()
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Remote keyboard connected from {addr[0]}:{addr[1]}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()
        if self._server is not None:
            self._server.close()

    def next_command(self) -> ControlCommand:
        if self._conn is None:
            raise RuntimeError("TCP controller is not connected.")

        while True:
            data = self._conn.recv(1)
            if not data:
                return ControlCommand(None, "quit", should_quit=True)
            command = self.KEYMAP.get(data.decode("utf-8", errors="ignore").lower())
            if command is not None:
                return command

    def send_result(self, record: Dict[str, Any]) -> None:
        if self._conn is None:
            return
        message = json.dumps(record) + "\n"
        self._conn.sendall(message.encode("utf-8"))
