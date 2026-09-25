from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .env import SceneSpec, StepResult


def unique_run_dir(path: Path) -> Path:
    """Return path, or path_001/path_002/... if it already exists."""
    if not path.exists():
        return path

    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused run directory for {path}")


class TrajectoryLogger:
    """JSONL trajectory logger with optional frame export."""

    def __init__(self, output_dir: Path, scene: SceneSpec, save_frames: bool) -> None:
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.rgb_dir = output_dir / "rgb"
        self.depth_dir = output_dir / "depth"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_frames:
            self.rgb_dir.mkdir(exist_ok=True)
            self.depth_dir.mkdir(exist_ok=True)

        self._jsonl = (output_dir / "trajectory.jsonl").open("w", encoding="utf-8")
        metadata = {
            "scene_id": scene.scene_id,
            "scene_path": scene.scene_path,
            "scene_dataset_config": scene.scene_dataset_config,
            "dataset_name": scene.dataset_name,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        self._jsonl.close()

    def log(self, result: StepResult, command_label: str) -> Dict[str, Any]:
        rgb_path = None
        depth_path = None

        if self.save_frames:
            color = result.observations.get("color_sensor")
            if color is not None:
                rgb_path = self.rgb_dir / f"{result.frame_index:06d}.png"
                save_rgb(rgb_path, color)

            depth = result.observations.get("depth_sensor")
            if depth is not None:
                depth_path = self.depth_dir / f"{result.frame_index:06d}.npy"
                np.save(depth_path, depth.astype(np.float32))

        record = {
            "frame_index": result.frame_index,
            "timestamp": result.timestamp,
            "command": command_label,
            "action": result.action,
            "pose": result.pose,
            "sensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in result.observations.items()
                if isinstance(value, np.ndarray)
            },
            "rgb_path": str(rgb_path) if rgb_path else None,
            "depth_path": str(depth_path) if depth_path else None,
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        return record


class RerunLogger:
    """Live Rerun visualization for collected Habitat steps."""

    def __init__(self, addr: str, spawn: bool = False) -> None:
        self.enabled = False
        self._positions = []
        self._entity_prefix = "world/trajectory"
        try:
            import rerun as rr
        except ImportError:
            print("Rerun SDK is not installed; continuing with file logging only.")
            return

        self.rr = rr
        try:
            rr.init("Habitat Collection", spawn=spawn)
            if not spawn:
                rr.connect_grpc(
                    f"rerun+http://{addr}/proxy",
                )
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            self.enabled = True
            print(f"Rerun logging enabled at {addr}")
        except Exception as exc:
            print(f"Could not connect to Rerun at {addr}: {exc}")
            print("Continuing with file logging only.")

    def new_trajectory(self, trajectory_id: Any) -> None:
        """Start a fresh trajectory line/point cloud under its own entity path.

        Without this, consecutive automatically-generated trajectories (which
        teleport the agent between episodes) would otherwise draw one
        continuous line jumping across the scene.
        """
        self._positions = []
        self._entity_prefix = f"world/trajectory_{trajectory_id}"

    def log(self, result: StepResult, command_label: str) -> None:
        if not self.enabled:
            return

        rr = self.rr
        rr.set_time("frame", sequence=result.frame_index)
        rr.log("command", rr.TextLog(command_label))

        color = result.observations.get("color_sensor")
        if color is not None:
            image = color[:, :, :3] if color.ndim == 3 and color.shape[2] == 4 else color
            rr.log("camera/image", rr.Image(image).compress(90))

        depth = result.observations.get("depth_sensor")
        if depth is not None:
            rr.log("camera/depth", rr.DepthImage(depth, meter=1.0))

        position = result.pose["position"]
        rotation = result.pose["rotation_xyzw"]
        self._positions.append(position)
        rr.log(
            f"{self._entity_prefix}/current_pose",
            rr.Transform3D(translation=position, quaternion=rotation),
        )
        rr.log(
            f"{self._entity_prefix}/points",
            rr.Points3D(self._positions, radii=0.04, colors=[255, 180, 0]),
        )
        if len(self._positions) >= 2:
            rr.log(
                f"{self._entity_prefix}/path",
                rr.LineStrips3D([self._positions], radii=0.015, colors=[255, 180, 0]),
            )


def save_rgb(path: Path, color: np.ndarray) -> None:
    import cv2

    image = color
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = np.asarray(image, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
