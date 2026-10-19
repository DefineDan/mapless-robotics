# Noisy odometry

By default `auto_explore_navigate` reads the robot pose from the simulator. Pass
`--odometry-noise` to run on drifting odometry instead:

```bash
python -m mapless_robotics.auto_explore_navigate --index 10 --seed 1 --odometry-noise 0.01
```

## Model
`navigate/odometry.py` wraps the Habitat agent in an `OdometryAgent`. Motion still goes
to the real agent (`sim.step`, `step_velocity`, `get_sim_action`). Every pose *read*
(`agent.get_state()`, and therefore `capture()`) accumulates the true motion since the
last read, adds Gaussian noise, and returns body and sensor poses in the odometry frame.

For a motion of planar distance `d` (m) and turn `dθ` (rad):

- translation std `noise·sqrt(d)` on both ground-plane axes,
- yaw std `noise·sqrt(|dθ| + d)`.

Variance grows linearly with motion, so drift does not depend on step size or read
frequency. `0.01` gives about 0.4 m error after 20 m of straight driving (about 2%), and `0.05`
gives about 2 m. `0` (default) is an exact passthrough. `--odometry-seed` fixes the noise
(default: `--seed`).

Odometry starts at the ground-truth start pose, and every downstream component only sees
odometry. When the robot is "put" at a pose (e.g. `restore_view`), `OdometryAgent.set_state`
treats it as a relative motion from the *believed* pose, with noise. Before retrieval,
`teleport` places the follower at the true start and resets its odometry.

**Limitation:** the model measures the motion actually made after Habitat's `step_filter`
slides along walls. It does not model wheel slip.

## Ground truth for evaluation
- Rerun: a grey `world/robot_ground_truth` arrow next to the green odometry arrow, a
  `world/trajectory/ground_truth` trail, and `odometry/position_error_m`.
- `route.json`: `odometry_positions` and `ground_truth_positions` per archived frame,
  noise settings under `initialization`.
- `Observation.gt_robot_pose` / `gt_camera_pose` next to `robot_pose` / `camera_pose`.

Simulator stepping, collisions, and plotting stay on ground truth.

## Where the pipeline depends on pose

| Where | Pose used | Expected failure under drift |
|---|---|---|
| Frontier back-projection (`ExplorationPipeline.process`, `opencv_extrinsic(camera_pose)`) | camera pose | The same physical frontier seen at different times lands at different world points. |
| Frontier fusion (`ExplorationGraph.add_frontier`, `merge_distance` 0.5 m, `fusion_angle_degrees` 30°) | frontier positions | Once drift exceeds about 0.5 m or 30°, re-observations stop fusing (duplicate frontiers), or unrelated ones merge. |
| Visited marking / `ExplorationMap` (`add_pose`, `test_visit`) | pose nodes | Explored areas are not marked visited, so the robot re-explores them; unexplored areas can be wrongly invalidated. |
| Objective arrival (`eval_objective`, `revalidate_objective`, `visited_distance` 0.5 m) | robot and camera pose | A frontier is never "reached", or is retired before the robot gets there. |
| Frontier selection (`select_saved`, `best_saved`, `score`) | camera pose | Distance and heading terms are biased. This is a soft degradation. |
| Recovery decision (`graph.observer`, `should_recover`) | current position vs pose nodes | The nearest observing viewpoint is wrong, causing unnecessary or missed LoTIS returns. |
| Local depth planner (`DepthNavigation.set_data` / `get_command`) | camera pose | Consistent over short horizons, but accumulated depth frames smear, which can create phantom obstacles and stuck goals. |
| Stuck detection (`RecoveryBehaviour.is_stuck`) | robot pose history | Still works here because the model measures real motion. With real wheel slip, a stuck robot would look like it was moving. |
| Route recording spacing (`changed_enough`, 0.2 m / 15°) | recorded states | Frame spacing becomes slightly irregular. Harmless. |
| LoTIS route geometry (`compact_route_ids`, `route_positions`, initial yaw) | recorded odometry states | Drift is locally consistent, so this is mostly robust. Long legs get distorted lookahead lengths. |
| LoTIS endpoint slowdown and `endpoint_stalled` (`LoTISRoutePolicy`) | recorded route positions from LoTIS's nearest frame | Not affected by live drift. Only the recorded leg lengths are distorted, as above. |
| `drive` tail no-progress accept (`in_tail`, 0.1 m over 40 steps) | live relative motion | Still works here because the model measures real motion. With real wheel slip, a stuck robot would look like it was moving. |
| `ARRIVAL_BACKSTOP` (1 m) and loop-closure frame | recorded poses of LoTIS's last localized frame and the leg's end frame | Not affected by live drift. Depends on LoTIS localizing correctly. |
| `SKIP_RETURN_RADIUS` (0.2 m) | current position vs pose node | A return is skipped although the robot is not at the viewpoint, or driven although it is. |
| `restore_view` | recorded heading | The robot turns to the *believed* heading, so the true view is off by the yaw drift. LoTIS and the fresh depth start misaligned. |
| `ViewScan` headings | yaw | Scan directions are unevenly spaced. Minor. |
| Object search (`detection_position`, 0.75 m consistency radius, `approach_position`, approach controller) | camera and robot pose | Confirmations taken far apart in time stop agreeing, and the approach point is offset by the drift. |
| Retrieval (`navigate`) | none for arrival (`goal_radius=None`) | Ends on LoTIS arrival. `goal_distance_m` (explorer vs follower odometry) is only logged. |
| Start pose (`choose_start`, navmesh) | ground truth by design | Not affected; this is simulator setup. |
