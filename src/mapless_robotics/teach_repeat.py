"""Teach and repeat with LoTIS.

1. Teach: walk through the scene (same keys as walk.py). Frames are recorded as you move.
2. Press Enter to stop recording. The frames are saved to runs/ and encoded with LoTIS.
3. Repeat: the agent is put back at the start and LoTIS drives it to where you stopped.

    python -m mapless_robotics.teach_repeat --index 25

Press q/esc at any time to quit.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import yaml
from habitat_sim.utils.common import (
    angle_between_quats,
    quat_from_magnum,
    quat_to_magnum,
)
from nav_eval.methods.lotis import LoTISNavigationMethod
from PIL import Image

from mapless_robotics.walk import KEY_TO_ACTION, QUIT, SCENES_DIR, make_sim

LOTIS_CONFIG = Path("lotis/nav_eval/configs/lotis_method.yaml")
DATA_DIR = Path("data/runs")
WINDOW = "teach & repeat"
ENTER = {13, 10}

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


def teach(
    sim: habitat_sim.Simulator,
) -> tuple[list[np.ndarray], list[habitat_sim.AgentState]]:
    """Let the user walk around, recording frames and agent states. Enter stops."""
    agent = sim.get_agent(0)
    obs = sim.get_sensor_observations()
    frames, states = [], []
    while True:
        state = agent.get_state()
        if not states or (
            np.linalg.norm(state.position - states[-1].position) >= RECORD_EVERY_M
            or np.rad2deg(turned(state, states[-1])) >= RECORD_EVERY_DEG
        ):
            frames.append(obs["rgb"][:, :, :3].copy())
            states.append(state)

        frame = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGBA2BGR)
        text = f"RECORDING  {len(frames)} frames  (Enter to stop)"
        cv2.putText(
            frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKeyEx(0)
        if key in QUIT:
            raise SystemExit
        if key in ENTER:
            if (
                states[-1] is not state
            ):  # also keep the spot where we stopped: that's the goal
                frames.append(obs["rgb"][:, :, :3].copy())
                states.append(state)
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


def draw(rgb: np.ndarray, result, goal: np.ndarray) -> np.ndarray:
    """Camera view with the LoTIS predictions (blue = start, red = end), goal frame next to it."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    n = len(result.visibility)
    for i, ((row, col), vis) in enumerate(zip(result.coords, result.visibility)):
        if vis > 0.5:
            color = cv2.applyColorMap(
                np.uint8([[255 * i // max(1, n - 1)]]), cv2.COLORMAP_JET
            )[0, 0]
            cv2.circle(
                frame,
                (int((col + 1) / 2 * w), int((row + 1) / 2 * h)),
                6,
                color.tolist(),
                -1,
            )
    return np.hstack([frame, cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)])


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

    for step in range(MAX_STEPS):
        rgb = sim.get_sensor_observations()["rgb"][:, :, :3]
        action = method.act({"rgb_pil": Image.fromarray(rgb), "intrinsics": intrinsics})

        cv2.imshow(WINDOW, draw(rgb, method._last_result, goal_frame))
        if cv2.waitKeyEx(1) in QUIT:
            return

        step_velocity(sim, action.linear_velocity, action.angular_velocity)

        distance = np.linalg.norm(agent.get_state().position - goal_state.position)
        if distance < GOAL_RADIUS:
            print(f"Reached the goal in {step + 1} steps")
            return
    print(f"Gave up after {MAX_STEPS} steps, {distance:.2f} m from the goal")


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
    args = parser.parse_args()

    # Load LoTIS first, so there's no wait after recording. Steer from the image only:
    # the eval's default geodesic follower would use the simulator's navmesh (a map).
    print("Loading LoTIS...")
    config = yaml.safe_load(LOTIS_CONFIG.read_text())
    config["use_geodesic_follower"] = False
    method = LoTISNavigationMethod(config)
    method.setup()

    scene = args.scene or next(SCENES_DIR.glob(f"{args.index:05d}-*/*.basis.glb"))
    print(f"Loading {scene}")
    sim = make_sim(str(scene))
    agent = sim.initialize_agent(0)
    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)

    # 1. Teach
    frames, states = teach(sim)
    if len(frames) < 2:
        raise SystemExit("Recorded fewer than 2 frames, walk a bit further next time")

    out_dir = DATA_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
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

    print("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
