"""Shared gain colors for the OpenCV and Rerun graph views."""

import numpy as np


# A perceptually ordered purple-to-yellow scale, stored as RGB.
STOPS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)
UNKNOWN_COLOR = (150, 150, 150)


def latest_gain(node):
    """Use the newest finite measurement, retaining earlier ones if later gain is absent."""
    for observation in reversed(node.observations):
        value = observation.get("gain")
        if value is None or not np.isscalar(value):
            continue
        try:
            gain = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(gain):
            return gain
    return None


def gain_colors(nodes):
    """Return RGB colors and the current finite gain range for frontier nodes."""
    gains = [latest_gain(node) for node in nodes]
    finite = [gain for gain in gains if gain is not None]
    if not finite:
        return [UNKNOWN_COLOR] * len(gains), None
    low, high = min(finite), max(finite)
    colors = []
    stops = np.asarray(STOPS, dtype=float)
    for gain in gains:
        if gain is None:
            colors.append(UNKNOWN_COLOR)
            continue
        fraction = (gain - low) / (high - low) if high > low else 0.5
        location = fraction * (len(stops) - 1)
        left = min(int(location), len(stops) - 2)
        mix = location - left
        color = np.rint(stops[left] * (1 - mix) + stops[left + 1] * mix)
        colors.append(tuple(int(channel) for channel in color))
    return colors, (low, high)
