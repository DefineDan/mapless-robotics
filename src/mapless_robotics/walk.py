"""Walk through a Habitat scene with the arrow keys.

    python -m mapless_robotics.walk --index 25
    python -m mapless_robotics.walk --scene path/to/scene.basis.glb

Keys: arrows = move (left/right strafe), a/d = turn, w/s = look up/down, q/esc = quit.
"""

import argparse
from pathlib import Path

import cv2
import habitat_sim

SCENES_DIR = Path("data/versioned_data/hm3d-0.2/hm3d/train")

# cv2.waitKeyEx codes for the arrow keys (macOS first, then Linux / Windows)
UP = {63232, 65362, 2490368}
DOWN = {63233, 65364, 2621440}
LEFT = {63234, 65361, 2424832}
RIGHT = {63235, 65363, 2555904}
QUIT = {ord("q"), 27}

KEY_TO_ACTION = {
    **dict.fromkeys(UP, "move_forward"),
    **dict.fromkeys(DOWN, "move_backward"),
    **dict.fromkeys(LEFT, "move_left"),
    **dict.fromkeys(RIGHT, "move_right"),
    ord("a"): "turn_left",
    ord("d"): "turn_right",
    ord("w"): "look_up",
    ord("s"): "look_down",
}


def make_sim(scene_path: str) -> habitat_sim.Simulator:
    # 1. Simulator: which scene to load
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    # 2. Sensor: an RGB camera at eye height, attached to the agent
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]  # height, width
    rgb.position = [0.0, 1.5, 0.0]

    # 3. Agent: the body carrying the sensor, plus the discrete actions it can take
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.2)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        # look_up/look_down tilt only the camera, not the agent body
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=80.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10.0, constraint=80.0)
        ),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


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

    scene = args.scene or next(SCENES_DIR.glob(f"{args.index:05d}-*/*.basis.glb"))
    print(f"Loading {scene}")
    sim = make_sim(str(scene))

    # Start at a random walkable spot. The navmesh (.navmesh next to the .glb) is loaded
    # automatically and also keeps the agent from walking through walls.
    agent = sim.initialize_agent(0)
    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    while True:
        frame = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGBA2BGR)
        cv2.imshow("habitat (arrows move, a/d turn, w/s look, q quit)", frame)

        key = cv2.waitKeyEx(0)
        if key in QUIT:
            break
        action = KEY_TO_ACTION.get(key)
        if action is not None:
            obs = sim.step(action)

    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
