#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import socket
import threading
import termios
import tty
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ControlCommand:
    label: str
    key: str
    should_quit: bool = False


class KeyboardController:
    KEYMAP = {
        "w": ControlCommand("forward", "w"),
        "a": ControlCommand("turn_left", "a"),
        "d": ControlCommand("turn_right", "d"),
        "s": ControlCommand("record", "s"),
        " ": ControlCommand("record", "s"),
        "r": ControlCommand("reset", "r"),
        "q": ControlCommand("quit", "q", should_quit=True),
        "\x1b": ControlCommand("quit", "q", should_quit=True),
    }

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old_settings = None

    def __enter__(self) -> "KeyboardController":
        import sys

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def next_command(self) -> ControlCommand:
        import sys

        while True:
            command = self.KEYMAP.get(sys.stdin.read(1).lower())
            if command is not None:
                return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote keyboard client for Habitat.")
    parser.add_argument("host", help="Tailscale IP/name or SSH-forwarded localhost.")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def print_responses(sock: socket.socket) -> None:
    stream = sock.makefile("r", encoding="utf-8")
    for line in stream:
        try:
            record = json.loads(line)
            pos = record["pose"]["position"]
            print(
                f"\r{record['frame_index']:06d} {record['command']:>10} "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n",
                end="",
                flush=True,
            )
        except Exception:
            print(f"\r{line.strip()}\n", end="", flush=True)


def main() -> None:
    args = parse_args()
    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    receiver = threading.Thread(target=print_responses, args=(sock,), daemon=True)
    receiver.start()

    print("Connected. Controls: w forward, a/d turn, s or space record, r reset, q quit")
    try:
        with KeyboardController() as keyboard:
            while True:
                command = keyboard.next_command()
                sock.sendall(command.key.encode("utf-8"))
                if command.should_quit:
                    break
    finally:
        sock.close()


if __name__ == "__main__":
    main()
