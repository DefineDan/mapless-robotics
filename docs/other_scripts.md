# Other scripts

These are the building blocks of [`auto_explore_navigate`](../README.md#running), and
they are mostly useful for debugging one part on its own. They share the scene options
(`--index`, `--scenes-dir`, `--scene`), `--depth-source`, `--camera-pitch`, and the LoTIS
path options.

## Scene discovery
`--index 25` looks for a `00025-*` folder in `--scenes-dir`. Without `--scenes-dir`, it
checks `data/versioned_data/hm3d-0.2/hm3d/train` (working directory and project root),
`scene_datasets/hm3d/train`, and the versioned data path under `external/habitat-sim`.
Missing indices report the searched paths. A navmesh is built in memory if a scene has none.

## `walk`: keyboard control
Checks that Habitat runs and that you can drive with the keyboard.

```bash
python -m mapless_robotics.walk --index 10
```

## `walk_frontiers`: manual frontier exploration
Drive by hand while OpenFrontier builds the frontier graph.

```bash
python -m mapless_robotics.walk_frontiers --index 25 --depth-source habitat --viewer rerun
```

- `--exploration`: `manual` (default, press `f`), `periodic --interval 1`,
  `movement --movement-distance 0.5 --movement-angle 15`, or `off` (keyboard viewer only).
- `--viewer`: `opencv` (default, three windows) or `rerun`. `--show-navigation-points` adds
  the depth-navigation point cloud in Rerun.
- `--frontier-weights` / `--frontier-config` override the OpenFrontier checkpoint
  (`external/OpenFrontier/model_weights/rgbd_11cls.pth`) and `config/navigation.yaml`.
  Only the detector settings in that YAML are used.
- `--depth-model` selects the UniK3D model for `--depth-source estimated`.
- Controls: arrows move, `a`/`d` turn, `w`/`s` tilt, `f` analyze, `g` save, `q`/`Esc` quit.

### Frontier graph
- All coordinates are in Habitat's world frame, in metres, with Y up. Camera poses are
  converted to OpenFrontier's OpenCV world-to-camera convention.
- Each processed capture adds a pose node (body and camera transforms, timestamp, parent).
- A new frontier merges into the nearest existing one within `--merge-distance` (0.5 m).
  Every measurement keeps its own position, gain, timestamp, and observing pose.
- Frontiers are colored by their latest gain, from purple (low) to yellow (high). Blue is
  the pose path, grey lines link observations, and the green arrow is the live pose.
- Saved after every analysis to `runs/exploration/<timestamp>/graph.json` (`--graph-output`).
  Load it with `ExplorationGraph.load(path)`.

Code layout: `exploration_graph.py` (history and persistence), `exploration.py` (capture,
pipeline, triggers, worker), `graph_view.py` (visualization), `walk_frontiers.py`
(Habitat and keyboard).

## `teach_repeat`: LoTIS only
Drive a route by hand. On `Enter` the robot resets to the start and LoTIS follows the
route to the end.

```bash
python -m mapless_robotics.teach_repeat --index 10 --dinov3-weights /path/to/dinov3_vitb16_pretrain.pth
```

It defaults to `external/lotis/nav_eval/configs/lotis_method.yaml`. `--lotis-config`,
`--checkpoint`, `--model-config`, `--dinov3-weights` (or `DINOV3_WEIGHTS` / `DINOV3_REPO`)
override it. Relative paths in a custom YAML resolve against its directory. Frames are
saved under `data/runs/`. Controls: arrows, `a`/`d`, `w`/`s`, `Enter` to start navigating,
`q` to quit, `Ctrl+C` to interrupt.

To test LoTIS alone on pre-recorded videos:

```bash
cd external/lotis
python infer.py --trajectory /path/to/trajectory.mp4 --query /path/to/query.mp4
```

## `explore_navigate`: manual exploration with LoTIS
Combines the two above. Drive while OpenFrontier builds the graph, then press `Enter` and
LoTIS navigates from the start to where exploration ended. While exploring, `n` plus a
frontier ID (`F<id>` in the 3D graph) lets LoTIS drive back to the pose that saw it.

```bash
python -m mapless_robotics.explore_navigate --index 10 --depth-source habitat \
  --dinov3-weights /path/to/dinov3_vitb16_pretrain.pth
```

Outputs go to `runs/explore_navigate/<timestamp>/` (`--run-dir`, `--graph-output`).
