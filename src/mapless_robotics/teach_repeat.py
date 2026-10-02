"""Teach and repeat with LoTIS.

1. Teach: walk through the scene. Frames are recorded as you move.
2. Press Enter to stop recording. The frames are saved to data/runs/ and encoded with LoTIS.
3. Repeat: the agent is put back at the start and LoTIS drives it to where you stopped.

    python -m mapless_robotics.teach_repeat --index 25

Everything is shown in Rerun. Keys are read from the terminal, so keep it focused:
arrows = move (left/right strafe), a/d = turn, w/s = look up/down, Enter = stop recording,
q = quit (Ctrl+C during navigation).
"""

from __future__ import annotations

import argparse
import os
import sys
import termios
import tty
from datetime import UTC, datetime
from pathlib import Path

import habitat_sim
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from habitat_sim.utils.common import (
    angle_between_quats,
    quat_from_magnum,
    quat_to_magnum,
)
from PIL import Image

from mapless_robotics.walk import SCENES_DIR, make_sim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOTIS_ROOT = PROJECT_ROOT / "external/lotis"
LOTIS_CONFIG = LOTIS_ROOT / "nav_eval/configs/lotis_method.yaml"
DATA_DIR = Path("data/runs")

# Terminal key -> Habitat action (arrow keys arrive as escape sequences)
KEY_TO_ACTION = {
    "\x1b[A": "move_forward",
    "\x1b[B": "move_backward",
    "\x1b[D": "move_left",
    "\x1b[C": "move_right",
    "a": "turn_left",
    "d": "turn_right",
    "w": "look_up",
    "s": "look_down",
}
ENTER = {"\n", "\r"}
QUIT = {"q"}

# Teaching: save a frame every time we moved or turned this much since the last one
RECORD_EVERY_M = 0.25
RECORD_EVERY_DEG = 15.0

# Repeating: same settings as the LoTIS eval
DT = 0.1  # seconds per control step (10 Hz)
MAX_STEPS = 1000
GOAL_RADIUS = 0.5  # meters
HFOV = 90.0  # Habitat's default camera FOV


def turned(a: habitat_sim.AgentState, b: habitat_sim.AgentState) -> float:
    """Angle in radians between two agent orientations."""
    return float(
        angle_between_quats(quat_to_magnum(a.rotation), quat_to_magnum(b.rotation))
    )


def read_key() -> str:
    """Wait for a single key press in the terminal (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        key = sys.stdin.read(1)
        if key == "\x1b":  # arrow keys: ESC [ A/B/C/D
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return key


def teach(
    sim: habitat_sim.Simulator,
) -> tuple[list[np.ndarray], list[habitat_sim.AgentState]]:
    """Let the user walk around, recording frames and agent states. Enter stops."""
    agent = sim.get_agent(0)
    obs = sim.get_sensor_observations()
    frames, states = [], []
    step = 0
    while True:
        state = agent.get_state()
        if not states or (
            np.linalg.norm(state.position - states[-1].position) >= RECORD_EVERY_M
            or np.rad2deg(turned(state, states[-1])) >= RECORD_EVERY_DEG
        ):
            frames.append(obs["rgb"][:, :, :3].copy())
            states.append(state)

        rr.set_time("step", sequence=step)
        rr.log("camera", rr.Image(obs["rgb"][:, :, :3]))
        rr.log("last_recorded_frame", rr.Image(frames[-1]))
        print(f"\rRECORDING  {len(frames)} frames  (Enter to stop)", end="", flush=True)

        key = read_key()
        step += 1
        if key in QUIT:
            raise SystemExit
        if key in ENTER:
            if (
                states[-1] is not state
            ):  # also keep the spot where we stopped: that's the goal
                frames.append(obs["rgb"][:, :, :3].copy())
                states.append(state)
            print()
            return frames, states
        if key in KEY_TO_ACTION:
            obs = sim.step(KEY_TO_ACTION[key])


def step_velocity(sim: habitat_sim.Simulator, linear: float, angular: float) -> None:
    """Move the agent with a velocity command for DT seconds, sliding along the navmesh.

    Same as step_velocity in lotis/nav_eval/habitat/runtime.py.
    """
    agent = sim.get_agent(0)
    state = agent.get_state()
    vel = habitat_sim.physics.VelocityControl()
    vel.controlling_lin_vel = vel.controlling_ang_vel = True
    vel.lin_vel_is_local = vel.ang_vel_is_local = True
    vel.linear_velocity = np.array([0.0, 0.0, -linear])  # -z is forward in Habitat
    vel.angular_velocity = np.array([0.0, angular, 0.0])  # rotate around the up axis

    previous = habitat_sim.RigidState(quat_to_magnum(state.rotation), state.position)
    target = vel.integrate_transform(DT, previous)
    state.position = sim.step_filter(previous.translation, target.translation)
    state.rotation = quat_from_magnum(target.rotation)
    agent.set_state(state, reset_sensors=False)


def repeat(
    sim,
    method: LoTISNavigationMethod,
    start: habitat_sim.AgentState,
    goal_state: habitat_sim.AgentState,
    goal_frame,
) -> None:
    """Put the agent at the start and let LoTIS drive until it reaches the goal."""
    agent = sim.get_agent(0)
    agent.set_state(start)

    h, w = goal_frame.shape[:2]
    fx = (w / 2) / np.tan(np.deg2rad(HFOV) / 2)
    intrinsics = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)

    rr.reset_time()
    rr.log("goal", rr.Image(goal_frame), static=True)
    for step in range(MAX_STEPS):
        rgb = sim.get_sensor_observations()["rgb"][:, :, :3]
        obs = {"rgb": rgb, "rgb_pil": Image.fromarray(rgb), "intrinsics": intrinsics}
        action = method.act(obs)

        # Camera + LoTIS predictions on top (blue = start, red = end of the trajectory).
        # method.visualize is the eval's own Rerun overlay.
        rr.set_time("navigation_step", sequence=step)
        rr.log("camera", rr.Image(rgb))
        method.visualize(obs)

        step_velocity(sim, action.linear_velocity, action.angular_velocity)

        distance = np.linalg.norm(agent.get_state().position - goal_state.position)
        if distance < GOAL_RADIUS:
            print(f"Reached the goal in {step + 1} steps")
            return
    print(f"Gave up after {MAX_STEPS} steps, {distance:.2f} m from the goal")


def load_lotis_config(
    config_path=LOTIS_CONFIG, *, checkpoint=None, model_config=None, dinov3_weights=None
):
    """Resolve the bundled checkout layout or paths relative to a custom YAML.

    CLI paths and environment variables are relative to the working directory.
    The bundled method YAML uses paths relative to the LoTIS checkout, except
    its DINOv3 repo, which lives alongside LoTIS under external/.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LoTIS method config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("LoTIS method config must contain a YAML mapping")
    bundled = config_path == LOTIS_CONFIG.resolve()
    base = LOTIS_ROOT if bundled else config_path.parent

    def resolved(value, root):
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"Expected a model path, got {value!r}")
        path = Path(value).expanduser()
        return str((root / path).resolve())

    for key, override, default in (
        ("checkpoint_path", checkpoint, "final_model.pth"),
        ("config_path", model_config, "final_config.yaml"),
    ):
        config[key] = resolved(
            override if override is not None else config.get(key, default),
            Path.cwd() if override is not None else base,
        )

    repo_env = os.environ.get("DINOV3_REPO")
    repo_base = PROJECT_ROOT / "external" if bundled else base
    config["dinov3_repo"] = resolved(
        repo_env or config.get("dinov3_repo", "./dinov3"),
        Path.cwd() if repo_env else repo_base,
    )
    weights = (
        dinov3_weights
        or config.get("dinov3_weights")
        or os.environ.get("DINOV3_WEIGHTS")
    )
    weights_base = (
        Path.cwd()
        if dinov3_weights is not None or not config.get("dinov3_weights")
        else base
    )
    config["dinov3_weights"] = resolved(weights, weights_base) if weights else None

    missing = []
    for key, label, option in (
        ("checkpoint_path", "LoTIS checkpoint", "--checkpoint"),
        ("config_path", "LoTIS model config", "--model-config"),
        (
            "dinov3_weights",
            "DINOv3 ViT-B/16 weights",
            "--dinov3-weights or DINOV3_WEIGHTS",
        ),
    ):
        if config[key] is None or not Path(config[key]).is_file():
            missing.append(f"  {label}: {config[key] or 'not configured'} ({option})")
    if not (Path(config["dinov3_repo"]) / "hubconf.py").is_file():
        missing.append(f"  DINOv3 repository: {config['dinov3_repo']} (DINOV3_REPO)")
    if missing:
        raise FileNotFoundError(
            "Missing teach-and-repeat model files:\n" + "\n".join(missing)
        )
    config["use_geodesic_follower"] = False
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--index", type=int, default=25, help="HM3D scene index, e.g. 1 -> 00001-*"
    )
    parser.add_argument(
        "--scene", default=None, help="Path to a .glb scene (overrides --index)"
    )
    parser.add_argument(
        "--lotis-config",
        type=Path,
        default=LOTIS_CONFIG,
        help="LoTIS navigation-method YAML",
    )
    parser.add_argument("--checkpoint", type=Path, help="LoTIS model checkpoint (.pth)")
    parser.add_argument(
        "--model-config", type=Path, help="LoTIS model architecture YAML"
    )
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        help="DINOv3 ViT-B/16 checkpoint (or set DINOV3_WEIGHTS)",
    )
    args = parser.parse_args()

    scene = (
        Path(args.scene).expanduser()
        if args.scene
        else next(
            (PROJECT_ROOT / SCENES_DIR).glob(f"{args.index:05d}-*/*.basis.glb"), None
        )
    )
    if scene is None or not scene.is_file():
        parser.error("Scene not found. Supply --scene /path/to/scene.glb.")
    try:
        config = load_lotis_config(
            args.lotis_config,
            checkpoint=args.checkpoint,
            model_config=args.model_config,
            dinov3_weights=args.dinov3_weights,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    # Habitat's EGL/Mesa LLVM must initialize before LoTIS imports Triton's LLVM.
    print(f"Loading {scene}")
    sim = make_sim(str(scene))
    agent = sim.initialize_agent(0)
    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)

    from nav_eval.methods.lotis import LoTISNavigationMethod

    # Steer from the image only; the eval's default geodesic follower uses a map.
    print("Loading LoTIS...")
    method = LoTISNavigationMethod(config)
    method.setup()

    rr.init("teach_repeat", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="camera"),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Last recorded frame", origin="last_recorded_frame"
                    ),
                    rrb.Spatial2DView(name="Goal", origin="goal"),
                    rrb.Spatial2DView(
                        name="Closest trajectory frame",
                        origin="closest_trajectory_frame",
                    ),
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    )

    # 1. Teach
    frames, states = teach(sim)
    if len(frames) < 2:
        raise SystemExit("Recorded fewer than 2 frames, walk a bit further next time")

    out_dir = DATA_DIR / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"{i:03d}.png")
    print(f"Saved {len(frames)} frames to {out_dir}")

    # 2. Encode the trajectory (LoTIS subsamples to at most 40 frames)
    print("Encoding trajectory...")
    method.reset_episode(
        {
            "rgb_images": [Image.fromarray(f) for f in frames],
            "tracking_mode": "forward",  # goal = last frame
            "goal_point_id": None,
        }
    )

    # 3. Repeat
    print("Navigating...")
    repeat(sim, method, states[0], states[-1], frames[-1])

    sim.close()


if __name__ == "__main__":
    main()
