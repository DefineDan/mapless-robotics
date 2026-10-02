"""Nonblocking terminal keys for the Rerun frontier viewer."""

import os
import select
import sys
import termios
import tty

# Match the existing cv2.waitKeyEx values used by walk_frontiers.py.
ARROW_CODES = {
    b"\x1b[A": 65362,
    b"\x1b[B": 65364,
    b"\x1b[D": 65361,
    b"\x1b[C": 65363,
}


class TerminalKeyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Rerun keyboard control requires an interactive terminal"
            )
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self, timeout=0.033):
        if not select.select([self.fd], [], [], timeout)[0]:
            return -1
        key = os.read(self.fd, 1)
        if key == b"\x1b":
            # Read the rest of an arrow sequence; bare Escape still quits.
            for _ in range(2):
                if not select.select([self.fd], [], [], 0.01)[0]:
                    break
                key += os.read(self.fd, 1)
            return ARROW_CODES.get(key, 27)
        return key[0]
