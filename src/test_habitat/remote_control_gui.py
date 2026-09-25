#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import tkinter as tk
from tkinter import ttk


KEYS = {
    "w": ("forward", "w"),
    "a": ("turn_left", "a"),
    "d": ("turn_right", "d"),
    "s": ("record", "s"),
    "space": ("record", "s"),
    "r": ("reset", "r"),
    "q": ("quit", "q"),
    "escape": ("quit", "q"),
}


class RemoteControlGui:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.messages: queue.Queue[dict | str] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Habitat Remote Control")
        self.root.geometry("460x320")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value=f"Connecting to {host}:{port}...")
        self.pose_var = tk.StringVar(value="pos=(--, --, --)")
        self.frame_var = tk.StringVar(value="frame=--")
        self.command_var = tk.StringVar(value="command=--")

        self._build_ui()
        self._bind_keys()
        self._connect()
        self.root.after(50, self._poll_messages)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(outer, textvariable=self.frame_var).pack(anchor="w", pady=(12, 0))
        ttk.Label(outer, textvariable=self.command_var).pack(anchor="w")
        ttk.Label(outer, textvariable=self.pose_var).pack(anchor="w")

        controls = ttk.Frame(outer)
        controls.pack(pady=18)

        ttk.Button(controls, text="W\nForward", command=lambda: self.send("w")).grid(
            row=0, column=1, padx=6, pady=6, ipadx=14, ipady=10
        )
        ttk.Button(controls, text="A\nLeft", command=lambda: self.send("a")).grid(
            row=1, column=0, padx=6, pady=6, ipadx=14, ipady=10
        )
        ttk.Button(controls, text="S / Space\nRecord", command=lambda: self.send("s")).grid(
            row=1, column=1, padx=6, pady=6, ipadx=14, ipady=10
        )
        ttk.Button(controls, text="D\nRight", command=lambda: self.send("d")).grid(
            row=1, column=2, padx=6, pady=6, ipadx=14, ipady=10
        )
        ttk.Button(controls, text="R\nReset", command=lambda: self.send("r")).grid(
            row=2, column=0, padx=6, pady=6, ipadx=14, ipady=10
        )
        ttk.Button(controls, text="Q / Esc\nQuit", command=lambda: self.send("q")).grid(
            row=2, column=2, padx=6, pady=6, ipadx=14, ipady=10
        )

        ttk.Label(
            outer,
            text="Click this window once, then use W/A/D/S/R/Q. Rerun can stay open in another window.",
        ).pack(anchor="w", pady=(8, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._on_key)

    def _connect(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=10.0)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as exc:
            self.status_var.set(f"Connection failed: {exc}")
            return

        self.status_var.set(f"Connected to {self.host}:{self.port}")
        thread = threading.Thread(target=self._receive_loop, daemon=True)
        thread.start()

    def _receive_loop(self) -> None:
        assert self.sock is not None
        stream = self.sock.makefile("r", encoding="utf-8")
        try:
            for line in stream:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self.messages.put(line.strip())
        except OSError as exc:
            self.messages.put(f"Disconnected: {exc}")

    def _poll_messages(self) -> None:
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, dict):
                self._show_record(message)
            else:
                self.status_var.set(message)
        self.root.after(50, self._poll_messages)

    def _show_record(self, record: dict) -> None:
        pos = record["pose"]["position"]
        self.frame_var.set(f"frame={record['frame_index']:06d}")
        self.command_var.set(f"command={record['command']}")
        self.pose_var.set(f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in KEYS:
            self.send(KEYS[key][1])

    def send(self, key: str) -> None:
        if self.sock is None:
            return
        try:
            self.sock.sendall(key.encode("utf-8"))
        except OSError as exc:
            self.status_var.set(f"Send failed: {exc}")
            return
        if key == "q":
            self.close()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI remote control for Habitat.")
    parser.add_argument("host", help="Tailscale IP/name or SSH-forwarded localhost.")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RemoteControlGui(args.host, args.port).run()


if __name__ == "__main__":
    main()
