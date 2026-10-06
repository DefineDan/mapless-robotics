# Mapless Robotics
Visual exploration and navigation without an explicit map
representation for collaborative object search and retrieval.

## Idea
An application could be search and rescue. We want to quickly search a place for an object using one or multiple lightweight drones. Once we have found the object, we use the RGB trajectory to navigate the robot that is capable of manipulation, or we could visually guide a human, e.g. fire fighter with a smartphone camera or wearing smart glasses, to the object.

## Approach
1. **Explore:** A lightweight robot (e.g. a drone) searches the environment based on RGB only using [OpenFrontier](https://arxiv.org/abs/2603.05377). Given a natural-language query (e.g. "fire extinguisher"), it finds visual frontiers directly in the RGB images and uses a vision-language model to score how promising each one is. It then flies toward the best one. There is no dense 3D map, only a lightweight, sparse frontier memory.
2. **Detect:** A VLM checks the drone's camera stream and decides when the queried object has been found.
3. **Extract reference:** The drone's RGB video, from the start position to the object, is saved as the reference trajectory. Optimizations might be applied to reduce the path length and remove loops.
4. **Navigate (Spot / human):** [LoTIS](https://arxiv.org/abs/2602.18803) localizes this reference trajectory in image space from the follower's live camera view and guides it along the path. The follower can be Spot or a person with a phone or smart glasses. This is *cross-embodiment* navigation: the reference is recorded by a drone and followed by a different agent, so camera height and viewpoint differ.

## Setup
1. Get the submodules:
    ```bash
    git submodule update --init --recursive
    ```
2. Install dependencies and create a venv using [uv](https://docs.astral.sh/uv/):
    ```bash
    uv sync
    ```
   This also installs OpenFrontier's dependencies (including UniK3D and pywavemap,
   which are built from source). It needs a recent uv: 0.7.x can't parse the
   `config-settings-package` section in `pyproject.toml`, so run `brew upgrade uv` or
   `uv self update` first if needed.

### Data Setup
1. Download the HM3D dataset. Refer to the [LoTIS README](external/lotis/README.md).

### LoTIS Setup
Refer to the [LoTIS README](external/lotis/README.md) for detailed setup instructions.
1. Get model weights for both LoTIS and DinoV3
2. Set the DINOv3 paths in your shell, or pass `--dinov3-weights` to the scripts:
    ```bash
    export DINOV3_WEIGHTS=/path/to/dinov3_vitb16_pretrain.pth
    export DINOV3_REPO=/path/to/external/dinov3
    ```
   The default LoTIS checkpoint and architecture config are in `external/lotis/`.
   Use `--checkpoint`, `--model-config`, or `--lotis-config` to override them.
3. Check the LoTIS navigation demo using a pre-recorded trajectory:
    ```bash
    cd external/lotis
    python infer.py \
        --trajectory /path/to/trajectory.mp4 \
        --query /path/to/query.mp4
    ```
## Running

1. Check if you can run the Habitat simulation and control it with the keyboard:
```bash
python -m mapless_robotics.walk --index 10                                               
```
2. The scripts load LoTIS settings from
   [`external/lotis/nav_eval/configs/lotis_method.yaml`](external/lotis/nav_eval/configs/lotis_method.yaml)
   by default. Pass `--lotis-config` to use another method configuration.

3. Demonstrate a trajectory and follow it with LoTIS in the simulator:
```bash
python -m mapless_robotics.teach_repeat --index 10
```


## Openfrontier

### Setup
The Python dependencies come with `uv sync` (see [Setup](#setup)). There's no
separate environment and no `pip install -r requirements.txt`. You only need the
model weights:

```bash
cd external/OpenFrontier
bash scripts/download_weights.sh
cd ../..
```

The UniK3D depth model downloads from Hugging Face on first use.

## Frontier exploration and graph generation

Run from `mapless-robotics/` with the project environment activated
(`source .venv/bin/activate`, or prefix the commands with `uv run`).
`--depth-source habitat` uses simulator depth; `--depth-source estimated` uses
UniK3D. `uv sync` installs the project itself, so `PYTHONPATH=src` is optional.
The default checkpoint is `external/OpenFrontier/model_weights/rgbd_11cls.pth`;
use `--frontier-weights /path/to/rgbd_11cls.pth` for a checkpoint elsewhere.
The default detector settings come from OpenFrontier's `config/navigation.yaml`;
`--frontier-config` can select another YAML file. Map/planning/VLM settings in
that file are not used by this pipeline.

For this workspace, a scene and checkpoint are already available in the sibling
`OpenFrontier` checkout. From `mapless-robotics/`, run:

```bash
PYTHONPATH=src python -m mapless_robotics.walk_frontiers \
  --scene ../OpenFrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../OpenFrontier/model_weights/rgbd_11cls.pth \
  --exploration manual --depth-source habitat
```

The `--index 25` examples below require an installed HM3D dataset; they do not
download scenes.

```bash
# Manual analysis, with simulator depth to first verify the graph geometry.
PYTHONPATH=src python -m mapless_robotics.walk_frontiers --index 25 \
  --exploration manual --depth-source habitat

# Document pipeline: RGB -> UniK3D depth -> OpenFrontier -> graph, every second.
PYTHONPATH=src python -m mapless_robotics.walk_frontiers --index 25 \
  --exploration periodic --interval 1 --depth-source estimated

# Analyze after translation OR camera rotation exceeds a threshold.
PYTHONPATH=src python -m mapless_robotics.walk_frontiers --index 25 \
  --exploration movement --movement-distance 0.5 --movement-angle 15 \
  --interval 1 --depth-source habitat

# Original keyboard-only viewer (no perception models required).
PYTHONPATH=src python -m mapless_robotics.walk_frontiers --index 25 --exploration off
```

Use `--scene /path/to/scene.glb` instead of `--index` for another scene. For an
HM3D dataset elsewhere, pass `--scenes-dir /path/to/hm3d/train --index 25` (the
split directory must contain `00025-*` folders). Without `--scenes-dir`, discovery
checks `data/versioned_data/hm3d-0.2/hm3d/train` under the working directory and
project root, then `scene_datasets/hm3d/train` under the project root and the
versioned data path under `external/habitat-sim`. Missing indices report the
searched paths; they never silently select a different scene. If a scene has no
navmesh, one is built in memory.
Estimated depth uses UniK3D (installed by `uv sync`); the first use downloads
the pretrained model.
`--depth-model` selects the UniK3D model. Habitat depth bypasses UniK3D while
still running the real OpenFrontier detector. Manual analysis is the default;
estimated depth is the default depth source.

### Rerun viewer

Add `--viewer rerun` to stream the live RGB camera, OpenFrontier's marked
capture, and a native 3D pose/frontier graph to Rerun:

```bash
PYTHONPATH=src python -m mapless_robotics.walk_frontiers \
  --scene ../OpenFrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../OpenFrontier/model_weights/rgbd_11cls.pth \
  --exploration manual --depth-source habitat --viewer rerun
```

Add `--show-navigation-points` to show the recent 2D depth-navigation cloud
as 3D points at the latest camera's height, transformed into Habitat world
coordinates (Y is up by default).

Rerun opens in a separate window. **Keep the terminal focused for the keys**;
press `f` to analyze, arrows to move, A/D to turn, W/S to tilt, `g` to save,
and Q/Escape to quit. The blue points and lines are processed robot poses,
frontier colors run from purple (lower gain) to yellow (higher gain), gray
lines are observation associations, and the green arrow is the live robot
pose/heading. In Rerun's entity tree,
`world/observations` can be hidden to reduce clutter. The detected image
remains on the last completed capture until another analysis finishes. The
`--viewer opencv` default retains the existing three OpenCV windows and their
keyboard controls.

Controls:

- Arrow keys: forward/backward and left/right strafe.
- `a` / `d`: turn; `w` / `s`: look up/down.
- `f`: analyze the current capture, including in automatic modes.
- `g`: save the currently completed graph.
- `q` / Escape: finish any in-flight analysis, save, and quit.

RGB, optional depth, body pose, sensor pose, and wall-clock timestamp are copied
together at capture. The viewer refreshes at roughly 30 Hz, subject to rendering
cost. A single background worker processes captures without blocking movement;
pressing `f` while busy does not queue an old image. Automatic processing uses
the latest capture when the worker is free. `--interval` is the minimum time
between automatic submissions; inference may take longer. Movement mode compares
translation and full rotation of the camera with the last submitted capture,
so looking up/down also counts. An inference error is printed and switches
analysis to manual mode for retry with `f`.

After an analysis completes, a third window, **Detected frontiers**, shows the
captured image with numbered frontier markers drawn by OpenFrontier's
`FrontierDetector.get_SoM_img()`. This is the detector's resized, center-cropped
view (normally 320×320), so marks align with the pixels used for inference.
The caption identifies the processed pose and capture time. Labels in this window
are local detection numbers, independent of the graph's persistent `F` IDs. The
window keeps the last completed capture while you move or inference runs; press
`f` again or use an automatic mode to update it. A capture with zero detections
replaces the previous image with an unmarked crop and a zero count.

The second window shows the graph projected onto Habitat's X/Z ground plane:
blue poses/path edges, gain-colored frontiers, gray observation associations,
and the green current robot position/heading. All graph coordinates are in Habitat's
world frame, in metres, with Y up. The actual sensor transform (including camera
height and pitch) is converted from Habitat's camera-to-world convention to
OpenFrontier's OpenCV world-to-camera convention.

Every successfully processed capture adds a pose, even if no frontiers are found.
Pose nodes contain a full body transform, camera transform, capture timestamp,
parent ID, and Euclidean distance to the parent. These edges connect processed
poses; their distances are straight-line distances, not total keyboard travel
between captures. A new frontier merges into the nearest existing frontier within
`--merge-distance` (default 0.5 metres, 3D distance). Its original representative
position stays fixed, and every measurement retains its own position, gain,
timestamp, and observing pose ID. Invalid/nonfinite frontier positions are skipped.
No history is discarded. Deduplication is separate from detection-quality filtering.
The graph views color each frontier using its latest available gain measurement,
scaled across the currently visible gain range. A missing gain is gray; node
labels show the numeric gain.

The entire graph is saved after every completed analysis and on exit, by default
to `runs/exploration/<timestamp>/graph.json`. `--graph-output` overrides this path
(and replaces that file). Each viewer run starts a new graph. Graph snapshots can
be loaded programmatically with `ExplorationGraph.load(path)` for inspection or
continued accumulation in the same coordinate frame.

The implementation is split into `exploration_graph.py` (history and persistence),
`exploration.py` (capture, pipeline, triggers, worker), `graph_view.py` (visualization),
and `walk_frontiers.py` (Habitat/keyboard integration). `ExplorationPipeline.process(observation)`
and `ExplorationGraph.add_frontiers_graph(...)` can also be called from another
component. Navigation/goal selection and object detection are left for the next stage.

Run the focused tests without downloading models or starting Habitat:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Teach and repeat with LoTIS and Rerun

`teach_repeat.py` records a route while you drive, then returns to the start and
uses LoTIS to navigate toward the recorded endpoint. It loads its method settings
from `external/lotis/nav_eval/configs/lotis_method.yaml` independently of the
working directory. The default LoTIS checkpoint and architecture YAML live in
`external/lotis/`; the DINOv3 repository is `external/dinov3/`.

Two trained models are required: the LoTIS route-localization checkpoint and
DINOv3 ViT-B/16 image-feature weights. Download the public LoTIS checkpoint with:

```bash
.venv/bin/hf download fnnBsch/lotis final_model.pth --local-dir external/lotis
```

For DINOv3, request access through [Meta's official downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
and download the ViT-B/16 **PyTorch `.pth` checkpoint**. This loader uses Meta's
original repository format. Point the script to the downloaded file:

```bash
PYTHONPATH=src python -m mapless_robotics.teach_repeat \
  --scene ../OpenFrontier/examples/mv2HUxq3B53.glb \
  --dinov3-weights /path/to/dinov3_vitb16_pretrain.pth
```

`DINOV3_WEIGHTS` is an alternative to `--dinov3-weights`. Use `--checkpoint` and
`--model-config` for LoTIS files stored elsewhere. `--lotis-config` selects a custom
method YAML; relative paths inside a custom YAML resolve against its directory.
CLI paths resolve against the working directory. `DINOV3_REPO` can override the
backbone repository. Startup checks all required files before loading models.

Keep the **terminal focused** for controls: arrows move, A/D turn, W/S tilt the
camera, Enter finishes recording and begins automatic navigation, Q quits during
recording, and Ctrl+C interrupts navigation. Rerun shows the images and predictions;
recorded frames are saved under `data/runs/`.

## Explore and navigate in Rerun

`explore_navigate.py` combines the two workflows. Drive a route while
OpenFrontier builds the graph, then press Enter to reset the robot to the start
and let LoTIS navigate to where exploration ended. While exploring, press `n`
and type a frontier id (the `F<id>` labels in the 3D graph) to let LoTIS drive
back to the pose that saw that frontier, along the frames recorded since then.
Frontier analysis continues while LoTIS drives.
Use the terminal for controls; Rerun displays the live camera with LoTIS's
trajectory points, the marked frontier image, and the 3D graph.

From `mapless-robotics/` with the project environment activated:

```bash
PYTHONPATH=src python -m mapless_robotics.explore_navigate \
  --scene ../OpenFrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../OpenFrontier/model_weights/rgbd_11cls.pth \
  --dinov3-weights models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --exploration manual --depth-source habitat
```

Press `f` to detect frontiers, arrows to move, A/D to turn, W/S to tilt the
camera, `g` to save, `n` to navigate to a frontier (Q/Escape stops that drive
and returns to manual control), Enter to start navigating to the end, and
Q/Escape to quit. Choose
`--exploration periodic --interval 1` or `--exploration movement` for automatic
frontier captures. The route frames and graph are saved together under
`runs/explore_navigate/<timestamp>/`; `--run-dir` and `--graph-output`
override those paths. Habitat is initialized before LoTIS to avoid the LLVM

## Automatic frontier exploration with LoTIS recovery

The new launcher uses the existing exploration, depth-navigation and Rerun
components, with LoTIS returning to saved viewpoints during exploration:

```bash
python -m mapless_robotics.auto_explore_navigate \
  --index 10 --depth-source habitat \
  --dinov3-weights /path/to/dinov3_vitb16_pretrain.pth
```

Use the same `--scene`, `--scenes-dir`, `--frontier-weights`, `--lotis-config`,
`--checkpoint` and `--model-config` overrides as `explore_navigate`. The default
is `--exploration periodic --interval 0.5`; full depth/frontier checks run in the
existing background worker while navigating. `--interval` adjusts the minimum
submission interval; actual checks also depend on inference time. `f` can request
an immediate check. Depth navigation applies commands continuously between completed
analyses (default 10 Hz), so repeated `f` presses are unnecessary. Simulator
body/sensor poses and depth are used with `--depth-source habitat`; estimated
depth continues to use UniK3D and is refreshed when an analysis finishes.

The start position and heading are now randomized on each launch. Startup prints
`Start XYZ=..., yaw=..., seed=...`; that information is also saved in
`route.json`. Add `--seed 42` to reproduce a start, or try another seed to start
elsewhere. To choose a specific location, pass `--start-position X Y Z` and
optionally `--start-yaw 90` (degrees: 0 faces world -Z, 90 faces world -X). The
position must be on the scene's navmesh; invalid positions report an error.

The automatic launcher starts with the camera tilted **10 degrees downward**.
Set `--camera-pitch 15` for 15 degrees down, `--camera-pitch 0` for a level view,
or a negative value to look up (allowed range: -80 to 80 degrees). RGB and Habitat
depth use the same sensor pitch, which remains configured when the agent's
position or heading is initialized or reset. The initial pitch is saved in
`route.json`. `walk_frontiers` and `explore_navigate` accept the same option and
default to a level camera. Their camera configuration is in
`walk_frontiers.make_sim`; `walk.py` defines a separate simulator for its own
launcher. A `look_down` action before `agent.set_state(state)` is reset to the
configured pitch, so use this option to set the initial view. W/S still adjust
the camera during exploration.

The agent first detects frontiers in its current view and commits to one goal.
It refreshes frontiers regularly while driving and retains newly observed
alternatives. A higher-gain alternative does not interrupt the active goal.
After arrival, rejection with `u`, or repeated low-quality target detections,
it first analyzes the current view again. Candidates seen in that fresh capture
are preferred, ranked by their current gain; older candidates are a fallback.
When no candidates remain, If that
view is empty, it scans up to six directions around the full circle, stopping as
soon as a direction yields candidates. It waits for each stationary capture to
finish before rotating again. `--scan-views 4` changes the direction count;
`--scan-views 0` disables sweeping. If all directions are empty, the agent pauses.
Move to a new viewpoint with the keyboard, then Space to resume discovery.

Keep the terminal focused:

- `u`: mark the active frontier unreachable, check the current view, then select
  a local alternative before falling back to saved candidates from older views.
- Space: pause/resume automatic navigation. Arrow/A/D/W/S movement pauses it too.
- `f`: force a capture; `g`: save frames, route metadata and the graph.
- `r`: retry a cancelled or timed-out LoTIS return.
- Enter: declare the object found, save the retained route, reset to the start
  and follow it with LoTIS. Object detection remains an operator decision.
- Q/Escape: quit exploration/retrieval; during a LoTIS return, stop and pause.

The reduced OpenFrontier integration exposes gain. Fresh local candidates are
ranked by gain in the current capture; historical fallback candidates use **peak
observed gain**, without the ordinary graph's distance penalty or visit decay.
Historical alternatives stay in memory, including frontiers seen on detours
removed from the retrieval route. The current goal stays fixed until reached or
marked unreachable or invalidated by repeated low-quality matches; higher-gain
alternatives wait for the next selection. Each full check records target status
in `pipeline_memory.objective_check` and Rerun's pipeline status. Three consecutive
matching detections with gain below the graph's minimum (default 0.5), nonfinite
or missing gain, or an explicit invalid flag retire the goal as `invalidated`.
A missing detection marks it `not_observed` and resets this streak: absence alone
does not prove a target unreachable. A good detection resets the streak too.
Reached, rejected and invalidated frontiers are retained as history,
but are not selected again.

Before returning anywhere, selection waits for a fresh completed capture at the
current position. For a reobserved frontier, the nearest recorded observing pose
to the current robot position is used, with newer poses winning ties.
For a remote alternative, LoTIS follows the recorded route backward to its
observing body pose. If that viewpoint lies on an archived branch, it first
returns to the shared junction and then follows the archived branch forward.
It aligns with the saved heading, restores the saved camera pitch, and hands
control back to depth navigation toward that frontier. An alternative at the
current viewpoint can be selected directly. Selecting a saved goal restores its
observing heading and camera pitch, clears the old local occupancy, and uses fresh
depth from that view. Every successful return forces a full frontier/depth check, including
with `--exploration manual`; Habitat depth initializes local navigation immediately
at the returned position. Arrival currently
uses simulator ground-plane distance: 0.3 m for a return viewpoint and 0.5 m for
a frontier. A missing local path stops motion; use `u` to reject the target.

During a LoTIS return, Rerun marks the selected frontier in **yellow**, labelled
`RETURN FRONTIER F...`, and its saved observing viewpoint in **cyan**, labelled
with the pose and recorded frame IDs. The cyan arrow shows the saved heading;
the connecting line identifies the observation, rather than a planned path.
The **Return viewpoint** panel shows the saved RGB image, and **Return target**
shows the IDs and return status. These markers stay visible if the return pauses
and are cleared after a successful return. The ordinary red objective marker
updates as soon as a frontier is selected. Manual returns with `n` in
`explore_navigate` use the same markers.

The automatic return condition compares the robot with the selected observing
viewpoint: more than **0.5 m** away triggers LoTIS. A pending recovery after `u`
also enters the return flow; a viewpoint already within **0.3 m** skips the
corresponding travel leg. A higher-gain candidate alone does not interrupt an
active goal, and a missing local path alone does not automatically reject it.

After a **successful** return, the route suffix beyond that viewpoint is excluded
from retrieval, including the outbound detour and its redundant return. Archived
frame IDs remain stable, so saved frontier observations still reference the
correct image after multiple returns. The graph retains excluded pose IDs and
retired frontiers in `pipeline_memory`; the viewer shows excluded branches in
red and the retained graph path in blue. Historical frontier markers remain
visible: reached frontiers are gray; unreachable and invalidated frontiers are muted red.
Retired frontier positions
are suppressed within the larger of the merge and visit radii to prevent repeated
detections from reviving the same target. This marks an abandoned branch unusable
for retrieval; it does not label its physically traversable corridor as an
obstacle, since LoTIS needs that corridor to return.

Outputs live in `runs/auto_explore_navigate/<timestamp>/` (or `--run-dir`):

- `archive/`: recorded exploration images, including abandoned detours.
- `frames/`: the ordered, loop-pruned RGB route for retrieval.
- `route.json`: active archive frame IDs, archive parent IDs for revisiting older
  branches, excluded branches and reasons, and `retrieval_ready` (false while a return is unresolved).
- `graph.json`: observations, active/retired frontier memory and excluded poses
  (`--graph-output` can place the graph elsewhere).

The automatic launcher wraps LoTIS with a guarded visual policy for return and
retrieval. Return references omit intermediate stationary/near-duplicate poses
(0.15 m spacing, retaining both endpoints). The initial backward alignment follows
the recorded travel direction rather than blindly turning 180 degrees from the
current heading. Steering requires finite image-space predictions inside the
camera view with visibility above the configured threshold. It skips projected
self poses and uses a visible point ahead in the requested route direction.

Travel speed follows the LoTIS method's `forward_velocity` (0.4 m/s in the bundled
config), with `--lotis-speed` providing an optional lower cap. Large heading errors
enter a turn-in-place mode at 35 degrees; forward motion resumes below 20 degrees,
so noisy predictions around one threshold do not alternate forward/stop commands.
Forward speed tapers continuously during aligned steering corrections.

Return and retrieval encodings include the recorded body positions, subsampled
with exactly the same frame indices. Lookahead uses 0.35 m of recorded path spacing.
LoTIS's distance head outputs normalized values in [0, 1]; these are used to rank
closest frames, not interpreted as metres for this lookahead. If reference
positions are absent, the wrapper retains its relative-distance fallback.

Near the endpoint, slowdown uses measured simulator distance when available:
it tapers speed inside 0.75 m instead of creeping at 0.05 m/s because of a small
normalized prediction while physically farther away. The older 0.05 m/s fallback
is retained for callers that do not provide a measured distance. Arrival still
uses measured simulator distance. An endpoint approach that fails to reduce the
measured goal distance by 0.03 m for 60 steps stops for retry.

LoTIS runs synchronously: each prediction advances the simulator by 0.1 simulated
seconds. Slow inference therefore produces slower playback than the configured
m/s value. CPU execution defaults to 8 PyTorch threads (`--lotis-cpu-threads` can
override this); excessive CPU threads can make localization substantially slower.
Startup prints device, CPU thread count and speed limit. Return diagnostics
include `inference_ms`, `cycle_ms`, `control_dt_s`, `linear_mps`, `angular_rps`,
`goal_distance_m` and `goal_radius_m`, alongside localization predictions. The
reported cycle time includes capture and visualization before motion; it is
not a claim that the simulator runs in real time.

When no usable route point is visible, the agent searches in place and stops after
60 unsuccessful prediction steps. It also stops if it advances less than 0.1 m
for 80 driving steps. A stopped return pauses for `r` and keeps the archived route.
These are visual-policy safeguards, not a simulator shortest-path fallback, and
they do not guarantee LoTIS can localize every backward view.

The terminal prints return-reference frame counts and LoTIS status periodically.
Rerun's `pipeline/return_status` includes `tracking`, `searching`,
`no_visible_lookahead`, `endpoint_approach` or `lost_stop`, plus visible-point counts and the encoded
nearest/target frame indices when available. Those indices refer to the compacted,
possibly subsampled reference used by LoTIS, not archive frame IDs.

A stopped or timed-out return preserves the route for retry and prevents Enter
from starting retrieval. After `u`, the agent checks the current view first and uses historical alternatives
if that view yields none. It sweeps other directions if no candidates remain.
If discovery is exhausted, turn/move to another viewpoint and Space to resume.
This produces
a route with abandoned detours removed, not a guaranteed globally shortest path.
The launcher starts a fresh session; these outputs do not implement resumable
LoTIS/controller state.

Verify the route-memory, worker integration and navigation completion behavior
without loading models or starting Habitat:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
