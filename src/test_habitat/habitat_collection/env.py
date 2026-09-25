from __future__ import annotations

import glob
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from collections import Counter


class AttrDict(dict):
    """Small dict with attribute access for YAML configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def clone(self) -> "AttrDict":
        return to_attrdict(deepcopy(self))


def to_attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: to_attrdict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_attrdict(item) for item in value]
    return value


@dataclass(frozen=True)
class SceneSpec:
    dataset_name: str
    scene_path: str
    scene_dataset_config: str
    scene_id: str
    needs_lighting: bool = False


@dataclass(frozen=True)
class StepResult:
    observations: Dict[str, np.ndarray]
    pose: Dict[str, Any]
    action: Optional[str]
    frame_index: int
    timestamp: float


def load_config(config_path: Path) -> AttrDict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _with_collection_defaults(config)
    # Let a student point at their own scene data without editing the YAML.
    data_path_override = os.environ.get("HABITAT_DATA_PATH")
    if data_path_override:
        config["DATASET"]["DATA_PATH"] = data_path_override
    return to_attrdict(config)


def _with_collection_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(config)
    config.setdefault("GENERAL", {})
    config["GENERAL"].setdefault("RANDOM_SEED", 1)

    sim = config.setdefault("SIM_SETTINGS", {})
    sim.setdefault("WIDTH", 512)
    sim.setdefault("HEIGHT", 512)
    sim.setdefault("DEFAULT_AGENT", 0)
    sim.setdefault("SENSOR_HEIGHT", 1.5)

    sim.setdefault("AGENT_HEIGHT", 1.5)
    sim.setdefault("AGENT_RADIUS", 0.05)

    sim.setdefault("COLOR_SENSOR", True)
    sim.setdefault("DEPTH_SENSOR", True)
    sim.setdefault("SEMANTIC_SENSOR", True)
    sim.setdefault("ENABLE_PHYSICS", False)
    sim.setdefault("CAMERA_FOV", 120.0)

    actions = config.setdefault("ACTIONS", {})
    actions.setdefault("MOVE_FORWARD", 0.25)
    actions.setdefault("TURN_ANGLE", 30.0)

    trajectory = config.setdefault("TRAJECTORY_SETTINGS", {})
    trajectory.setdefault("TRAJECTORIES_PER_SCENE", 10)
    trajectory.setdefault("MIN_PATH_DISTANCE", 3.0)
    trajectory.setdefault("MAX_TRAJECTORY_LENGTH", 30.0)
    trajectory.setdefault("CONTROL_FREQUENCY", 10.0)
    trajectory.setdefault("FRAME_SKIP", 1)
    trajectory.setdefault("LOOK_AHEAD_DISTANCE", 0.4)
    trajectory.setdefault("MAX_FORWARD_SPEED", 0.7)
    trajectory.setdefault("MAX_TURN_SPEED", 0.8)
    trajectory.setdefault("ARRIVAL_DISTANCE", 0.3)
    trajectory.setdefault("SAVE_EVERY_N_FRAMES", [2, 8])

    dataset = config.setdefault("DATASET", {})
    dataset.setdefault("SCENE_DATASETS", [])
    selection = dataset.setdefault("SCENE_SELECTION", {})
    selection.setdefault("RANDOM", True)
    selection.setdefault("MAX_SCENES", 5)
    selection.setdefault("SCENE_LIST", [])
    return config


def discover_scenes(config) -> List[SceneSpec]:
    scenes: List[SceneSpec] = []
    data_path = Path(config.DATASET.DATA_PATH)

    for dataset in config.DATASET.SCENE_DATASETS:
        scenes_directory = data_path / dataset["scenes_directory"]
        scene_dataset_config = data_path / dataset["scene_dataset_config"]
        pattern = str(scenes_directory / dataset["scene_pattern"])

        for scene_file in glob.glob(pattern, recursive=True):
            scene_path = Path(scene_file)
            scene_id = str(scene_path.relative_to(scenes_directory).with_suffix(""))
            scenes.append(
                SceneSpec(
                    dataset_name=dataset["name"],
                    scene_path=str(scene_path),
                    scene_dataset_config=str(scene_dataset_config),
                    scene_id=scene_id,
                    needs_lighting=dataset.get("needs_lighting", False),
                )
            )

    selected_ids = set(config.DATASET.SCENE_SELECTION.SCENE_LIST)
    if selected_ids:
        scenes = [scene for scene in scenes if scene.scene_id in selected_ids]

    if config.DATASET.SCENE_SELECTION.RANDOM:
        random.shuffle(scenes)

    max_scenes = int(config.DATASET.SCENE_SELECTION.MAX_SCENES)
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    return scenes


class HabitatCollectionEnv:
    """Small step-based Habitat interface for interactive trajectory recording."""

    def __init__(self, config, scene: SceneSpec) -> None:
        import habitat_sim

        self.habitat_sim = habitat_sim
        self.config = config.clone()
        self.config.SIM_SETTINGS.COLOR_SENSOR = True
        self.config.SIM_SETTINGS.SEMANTIC_SENSOR = True
        self.scene = scene
        self.frame_index = 0
        self._vel_control = None
        self.sim = habitat_sim.Simulator(self._make_cfg())
        self._recompute_navmesh()
        semantic_scene = self.sim.semantic_scene

        print("Semantic objects:", len(semantic_scene.objects))
        print("Semantic regions:", len(semantic_scene.regions))

        self._semantic_id_to_region = {}

        for obj in semantic_scene.objects:
            if obj is None:
                continue

            if obj.region is None:
                continue

            self._semantic_id_to_region[int(obj.semantic_id)] = str(
                obj.region.id
            )

        print(
            "Semantic ID -> region mappings:",
            len(self._semantic_id_to_region),
        )

        for i, region in enumerate(semantic_scene.regions[:10]):
            print(
                "REGION",
                i,
                "id=",
                region.id,
            )

        seed = int(self.config.GENERAL.RANDOM_SEED)
        random.seed(seed)
        self.sim.seed(seed)
        self.reset_random()

    def close(self) -> None:
        self.sim.close()

    def reset_random(self) -> StepResult:
        # IMPORTANT: How to change the agent's position
        from habitat_sim.utils import common as quat_utils

        point = self.sample_navigable_point()
        yaw = random.uniform(0.0, 2.0 * np.pi)
        rotation = quat_utils.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        self.set_pose(point, rotation)
        return self.observe(action=None)

    def reset_frame_counter(self) -> None:
        """Restart frame numbering, e.g. at the start of a new trajectory."""
        self.frame_index = 0

    def step(self, action: Optional[str]) -> StepResult:
        if action is not None:

            if action == "move_forward":
                from habitat_sim.utils.common import quat_rotate_vector

                state = self.sim.get_agent(0).get_state()

                current = np.asarray(
                    state.position,
                    dtype=np.float32,
                )

                forward = quat_rotate_vector(
                    state.rotation,
                    np.array(
                        [0.0, 0.0, -1.0],
                        dtype=np.float32,
                    ),
                )

                target = current + (
                    forward
                    * float(self.config.ACTIONS.MOVE_FORWARD)
                )

                filtered = self.sim.pathfinder.try_step(
                    current,
                    target.astype(np.float32),
                )

                print(
                    "NAV DEBUG:",
                    "current=", current,
                    "target=", target,
                    "filtered=", filtered,
                    "target_navigable=",
                    self.sim.pathfinder.is_navigable(
                        target.astype(np.float32)
                    ),
                )

            self.sim.step(action)

        return self.observe(action=action)

    def set_pose(self, position: np.ndarray, rotation) -> None:
        """Teleport the agent (and its sensors) to an exact position/rotation."""
        state = self.sim.get_agent(0).get_state()
        state.position = np.asarray(position, dtype=np.float32)
        
        ## Habitat requires an actual unit quaternion.
        rotation = rotation.normalized()

        # Extra sanity check.
        if abs(rotation.norm() - 1.0) > 1e-10:
            raise ValueError(
                f"Quaternion normalization failed: {rotation}, norm²={rotation.norm()}"
            )
            
        state.rotation = rotation
        self.sim.get_agent(0).set_state(state, infer_sensor_states=True)

    def sample_navigable_point(self, island_index: int = -1) -> np.ndarray:
        self._ensure_navmesh_loaded()
        for _ in range(200):
            point = self.sim.pathfinder.get_random_navigable_point(island_index=island_index)
            if point is not None:
                return np.asarray(point, dtype=np.float32)
        raise RuntimeError("Could not sample a navigable point in the scene.")

    def island_of(self, point: np.ndarray) -> int:
        self._ensure_navmesh_loaded()
        return self.sim.pathfinder.get_island(np.asarray(point, dtype=np.float32))

    def compute_path(self, start: np.ndarray, end: np.ndarray):
        """Shortest navmesh path between two points.

        Returns (points, geodesic_distance), or (None, 0.0) if unreachable.
        """
        path = self.habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if not self.sim.pathfinder.find_path(path):
            return None, 0.0
        return np.array(path.points, dtype=np.float32), float(path.geodesic_distance)

    def step_velocity(self, linear_velocity: float, angular_velocity: float, dt: float) -> None:
        """Integrate one physics substep under body-frame velocities, clamped to the navmesh."""
        from habitat_sim.utils import common as quat_utils

        if self._vel_control is None:
            self._vel_control = self.habitat_sim.physics.VelocityControl()
            self._vel_control.controlling_lin_vel = True
            self._vel_control.lin_vel_is_local = True
            self._vel_control.controlling_ang_vel = True
            self._vel_control.ang_vel_is_local = True

        self._vel_control.linear_velocity = np.array([0.0, 0.0, -linear_velocity])
        self._vel_control.angular_velocity = np.array([0.0, angular_velocity, 0.0])

        agent = self.sim.get_agent(0)

        agent_state = agent.state
        previous_rigid_state = self.habitat_sim.RigidState(
            quat_utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )
        target_rigid_state = self._vel_control.integrate_transform(dt, previous_rigid_state)
        end_position = self.sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )

        agent_state.position = end_position
        agent_state.rotation = quat_utils.quat_from_magnum(target_rigid_state.rotation)
        agent.set_state(agent_state)

        if self.config.SIM_SETTINGS.ENABLE_PHYSICS:
            self.sim.step_physics(dt)

    def estimate_max_path_distance(
        self,
        samples: int = 2000,
    ) -> float:
        """Estimate the largest reachable geodesic distance in the scene."""

        best_distance = 0.0
        best_start = None
        best_end = None

        for _ in range(samples):
            start = self.sample_navigable_point()

            island = self.island_of(start)

            end = self.sample_navigable_point(
                island_index=island
            )

            path_points, distance = self.compute_path(
                start,
                end,
            )

            if (
                path_points is not None
                and distance > best_distance
            ):
                best_distance = distance
                best_start = start.copy()
                best_end = end.copy()

        print(
            f"Estimated maximum geodesic distance: "
            f"{best_distance:.2f} m"
        )

        if best_start is not None:
            print("best start:", best_start)
            print("best end:", best_end)

        return best_distance

    def _recompute_navmesh(self) -> None:
        settings = self.habitat_sim.NavMeshSettings()
        settings.set_defaults()

        settings.agent_height = float(
            self.config.SIM_SETTINGS.AGENT_HEIGHT
        )

        settings.agent_radius = float(
            self.config.SIM_SETTINGS.AGENT_RADIUS
        )

        # Finer voxelization helps with narrow openings.
        settings.cell_size = 0.025
        settings.cell_height = 0.10

        settings.include_static_objects = True

        success = self.sim.recompute_navmesh(
            self.sim.pathfinder,
            settings,
        )

        if not success:
            raise RuntimeError("Failed to recompute Habitat navmesh")

        print(
            "Recomputed navmesh:",
            f"radius={settings.agent_radius:.3f}m",
            f"height={settings.agent_height:.3f}m",
            f"cell_size={settings.cell_size:.3f}m",
            f"islands={self.sim.pathfinder.num_islands}",
            f"area={self.sim.pathfinder.navigable_area:.2f}m²",
        )

    def _ensure_navmesh_loaded(self) -> None:
        if self.sim.pathfinder.is_loaded:
            return
        navmesh_settings = self.habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)

    def observe(self, action: Optional[str]) -> StepResult:
        observations = self._sensor_observations()
        pose = self._pose_dict()
        region_id = self._region_from_semantic_observation(
            observations
        )

        pose["region_id"] = region_id
        result = StepResult(
            observations=observations,
            pose=pose,
            action=action,
            frame_index=self.frame_index,
            timestamp=time.time(),
        )
        self.frame_index += 1
        return result

    def _make_cfg(self):
        habitat_sim = self.habitat_sim
        settings = self.config.SIM_SETTINGS

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene.scene_path
        sim_cfg.scene_dataset_config_file = self.scene.scene_dataset_config
        sim_cfg.enable_physics = bool(settings.ENABLE_PHYSICS)
        sim_cfg.allow_sliding = True
        sim_cfg.load_semantic_mesh = bool(settings.SEMANTIC_SENSOR)
        sim_cfg.use_semantic_textures = False
        if self.scene.needs_lighting:
            sim_cfg.override_scene_light_defaults = True
            sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        sensor_specs = []
        if settings.COLOR_SENSOR:
            sensor_specs.append(
                self._camera_spec(
                    "color_sensor",
                    habitat_sim.SensorType.COLOR,
                    settings.SENSOR_HEIGHT,
                    settings.CAMERA_FOV,
                )
            )
        if settings.DEPTH_SENSOR:
            sensor_specs.append(
                self._camera_spec(
                    "depth_sensor",
                    habitat_sim.SensorType.DEPTH,
                    settings.SENSOR_HEIGHT,
                    settings.CAMERA_FOV,
                )
            )

        if settings.SEMANTIC_SENSOR:
            sensor_specs.append(
                self._semantic_floor_spec(
                    settings.SENSOR_HEIGHT
                )
            )

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        agent_cfg.height = float(settings.AGENT_HEIGHT)
        agent_cfg.radius = float(settings.AGENT_RADIUS)

        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(
                    amount=float(self.config.ACTIONS.MOVE_FORWARD)
                ),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(
                    amount=float(self.config.ACTIONS.TURN_ANGLE)
                ),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(
                    amount=float(self.config.ACTIONS.TURN_ANGLE)
                ),
            ),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _camera_spec(self, uuid: str, sensor_type, height: float, hfov: float):
        # IMPORTANT: Camera Setup
        habitat_sim = self.habitat_sim
        settings = self.config.SIM_SETTINGS

        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [int(settings.HEIGHT), int(settings.WIDTH)]
        spec.position = [0.0, float(height), 0.0]
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.hfov = float(hfov)
        return spec

    def _semantic_floor_spec(self, height: float):
        """Downward semantic camera used only for GT room/region labeling."""
        habitat_sim = self.habitat_sim

        spec = habitat_sim.CameraSensorSpec()

        spec.uuid = "room_semantic_sensor"
        spec.sensor_type = habitat_sim.SensorType.SEMANTIC

        # We only need the semantic ID underneath the robot.
        # No reason to render a 512x512 GT image.
        spec.resolution = [32, 32]

        spec.position = [
            0.0,
            float(height),
            0.0,
        ]

        # Habitat sensor orientation uses radians.
        # -pi/2 pitch = look downward.
        spec.orientation = [
            -np.pi / 2.0,
            0.0,
            0.0,
        ]

        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.hfov = 30.0

        return spec

    def _region_from_semantic_observation(
        self,
        observations: Dict[str, np.ndarray],
    ) -> Optional[str]:
        """
        Estimate the HM3D room/region underneath the agent.

        Uses a small central patch of the downward-looking semantic
        sensor and majority-votes the corresponding region IDs.
        """

        semantic = observations.get(
            "room_semantic_sensor"
        )

        if semantic is None:
            return None

        semantic = np.asarray(semantic)

        h, w = semantic.shape[:2]

        # 5x5 patch around the image center.
        radius = 2

        y0 = max(0, h // 2 - radius)
        y1 = min(h, h // 2 + radius + 1)

        x0 = max(0, w // 2 - radius)
        x1 = min(w, w // 2 + radius + 1)

        patch = semantic[y0:y1, x0:x1]

        region_votes = []

        for semantic_id in patch.reshape(-1):
            region_id = self._semantic_id_to_region.get(
                int(semantic_id)
            )

            if region_id is not None:
                region_votes.append(region_id)

        if not region_votes:
            return None

        counts = Counter(region_votes)

        region_id, _ = counts.most_common(1)[0]

        return region_id
        
    def _sensor_observations(self) -> Dict[str, np.ndarray]:
        return self.sim.get_sensor_observations()

    def _pose_dict(self) -> Dict[str, Any]:
        state = self.sim.get_agent(0).get_state()
        rotation = state.rotation
        return {
            "position": np.asarray(state.position, dtype=float).tolist(),
            "rotation_xyzw": [
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ],
        }
