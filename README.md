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
  and follow it with LoTIS. Without `--object-query`, object detection remains
  an operator decision.
- Q/Escape: quit exploration/retrieval; during a LoTIS return, stop and pause.

### Object search with Qwen and OWLv2

Add a request to the automatic launcher:

```bash
python -m mapless_robotics.auto_explore_navigate \
  --index 10 --depth-source habitat \
  --dinov3-weights /path/to/dinov3_vitb16_pretrain.pth \
  --object-query "a place to sit"
```

[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) translates the request
once into short detector labels, for example `chair`, `sofa`, `bench`, and
`stool`. Translation uses CPU by default, with thinking disabled, and releases
the model before loading the exploration models. `--query-device cuda` runs
this startup step on the GPU instead. The labels are printed before exploration
and shown in Rerun; inspect them because a language model can misinterpret a
request. This step proposes categories; it does not visually assess whether a
particular object meets the requested function.

Add `--no-qwen` to skip translation and pass the request directly to OWLv2,
for example `--object-query "chair" --no-qwen`. Qwen is not loaded in this mode.

[OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2) detects those
labels on the original RGB captures in a separate object worker. Frontier results
update the graph and local navigation as soon as frontier analysis finishes;
they do not wait for object inference. OWLv2 keeps one capture in flight and
skips submissions while busy, retaining that capture's RGB, depth, pose and
archived frame index. Results from before recovery or manual movement are discarded.
The default checkpoint is `google/owlv2-base-patch16-ensemble`; `--object-model`
accepts another OWLv2 model ID or local model directory. The first run downloads
both models and their tokenizers/processors to `models/huggingface/` (internet
required once). LoTIS continues using the configured DINOv3 checkpoint.

`--object-device auto` selects CUDA when available, otherwise CPU. CUDA object
inference uses half precision on its own stream to reduce delays to frontier
updates; CPU inference uses full precision. For faster
CPU inference, try `--object-image-size 512`; a smaller input can miss small
objects. Without the override, OWLv2 uses its native resolution. Qwen has no
callbacks or inference during navigation. The frontier trigger, recovery rules
and depth mask keep their existing settings; frontier analysis frequency depends
on depth/frontier inference time and `--interval`. Both workers still share CPU
or GPU resources, so hardware contention can affect their speed.

Rerun's **Object detections** window shows labeled boxes on the exact analyzed
capture. The graph shows depth-projected detections and a larger yellow confirmed
object marker, with a green approach point. **Object search** shows the original
request, translated labels, and confirmation/arrival status.

By default, three distinct consecutive captures must locate a candidate within
0.75 m, using median valid depth in the center of the box. Configure this with
`--object-confirmations` and `--object-consistency-radius`; `--object-threshold`
defaults to 0.2 (an uncalibrated detection score). Missing or invalid depth and
objects beyond 5 m cannot confirm a target. Estimated depth affects the position
accuracy. A confirmed object takes priority over frontier driving; the object
navigator proposes a ground-plane point `--object-stand-off 1.0` m from its
observed surface and requires a local path ending at that point. At arrival it
faces the object, requests a fresh matching detection, and pauses. Enter starts
the existing LoTIS retrieval. `o` dismisses the object location and resumes
frontier exploration; a blocked approach stops movement until a usable path
appears or the object is dismissed. `f` still requests a fresh analysis.

The request, labels, target position/score, observing archive frame, approach
point, rejected locations and arrival status are saved in `route.json` under
`object_search`. Running without `--object-query` uses the original exploration
flow. Tests run without downloading weights:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

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
retrieval. Return and retrieval references omit intermediate stationary/near-duplicate
poses (0.15 m spacing, retaining both endpoints), so in-place scans do not use up
the encoder's frame budget and leave large gaps elsewhere on the route. The initial backward alignment follows
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
Final retrieval restores the saved starting pose, then turns in place toward the
first recorded position at least 0.15 m away, skipping stationary startup scans.
LoTIS takes over once the heading is within the existing 20-degree alignment tolerance.
LoTIS's distance head outputs normalized values in [0, 1]; these are used to rank
closest frames, not interpreted as metres for this lookahead. If reference
positions are absent, the wrapper retains its relative-distance fallback.

The endpoint logic does not use the live pose. Arrival means the endpoint is
LoTIS's nearest visible route frame for 3 consecutive steps. Near the endpoint,
slowdown uses the recorded path length remaining from the frame LoTIS localizes
the robot at (`remaining_m`): it tapers speed inside the last 0.5 m instead of
creeping at 0.05 m/s because of a small normalized prediction. The 0.05 m/s
fallback is retained for references without positions. If an endpoint approach
does not localize the robot 0.03 m further along the route for 60 forward-driving
steps (turning in place does not count), the arrival is accepted (`endpoint_stalled`). Final retrieval also ends on this
LoTIS arrival, not on a radius around the recorded end position.

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

Before pausing, a leg stopped mid-route (lost or stalled, outside the last metre)
gets up to 2 automatic retries (`plan_retry`), each without re-encoding the leg:

- **U-turn:** LoTIS's nearest frame repeatedly jumps between two frames at least
  1.5 m apart along the recorded route but within 1.5 m in space, i.e. both legs of
  a U are in view. The first retry follows the U: matching is restricted to a window
  from 0.5 m behind the earlier frame to half the fold's path length ahead, so the
  other leg cannot be matched. The second retry cuts across, with the window
  centred on the frame on the other leg.
- **Lost / corner:** a window around the last matched frame (0.5 m back, 2 m ahead;
  1 m / 4 m on the second retry).
- Each retry first turns a full circle in 10-degree steps, scoring the view with
  LoTIS every 30 degrees (visibility of in-window frames ahead), and faces the best
  view. The window then slides with the robot until the leg ends. While it is on,
  a turn in place keeps its direction until a target is within 20 degrees, so a
  target switching sides between frames cannot make the robot turn back and forth.
  Normal tracking (no window) is unchanged.
- A leg stalled with the route still in view (likely an obstacle), a cancelled leg
  (`q`, also during the sweep), and the step limit are left to the operator. `r`
  starts a fresh leg without a window and with new retries.
- Final retrieval uses the same automatic retries. If it still stops, it pauses
  like a stopped return: arrows move, a/d turn, b turns around, `r` re-encodes the
  route and retries LoTIS from the current pose, `q` quits.

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

### Noisy odometry instead of ground-truth pose

By default the launcher reads the robot pose from the simulator. To make it run
on drifting odometry instead, pass `--odometry-noise`:

```bash
python -m mapless_robotics.auto_explore_navigate --index 10 --seed 1 \
  --odometry-noise 0.01
```

`navigate/odometry.py` wraps the Habitat agent in an `OdometryAgent`. Motion
still goes to the real agent (`sim.step`, `step_velocity`, `get_sim_action`).
Every pose *read* (`agent.get_state()`, and therefore `capture()`) adds up the
true motion since the last read, adds Gaussian noise, and returns the body and
sensor poses in the odometry frame.

The noise for each motion of planar distance `d` (m) and turn `dθ` (rad) is:

- translation std `noise·sqrt(d)`, applied to both ground-plane axes;
- yaw std `noise·sqrt(|dθ| + d)`.

The variance grows linearly with motion, so drift does not depend on step size
or on how often the pose is read. As a guide, `0.01` gives about 0.4 m of error
after 20 m of straight driving (about 2%), and `0.05` gives about 2 m.

`--odometry-noise 0` (the default) is an exact passthrough to ground truth.
`--odometry-seed` fixes the noise; it defaults to the start `--seed`.

Odometry starts equal to the ground-truth start pose. Every downstream component
only ever sees odometry. Ground truth is kept for evaluation only:

- In Rerun, once the two poses diverge, a grey `world/robot_ground_truth` arrow
  appears next to the green odometry arrow.
- Rerun also draws a thin grey `world/trajectory/ground_truth` trail (the graph
  already shows the odometry poses) and plots the planar error as
  `odometry/position_error_m`.
- `route.json` stores `odometry_positions` and `ground_truth_positions` for every
  archived frame, and records the noise settings under `initialization`.
- `Observation` has `gt_robot_pose` and `gt_camera_pose` next to the odometry
  `robot_pose` and `camera_pose`.

The robot sometimes has to be "put" at a pose, for example when `restore_view`
turns back to a recorded heading. `OdometryAgent.set_state` treats this as the
relative motion the robot would execute from its *believed* pose, and that
motion is itself integrated with noise. Before retrieval, `teleport` places the
follower at the true start and resets its odometry, which is where the
explorer's odometry frame began.

Limitation: the model measures the motion the robot actually made, after
Habitat's `step_filter` slides it along walls. It does not model wheel slip,
where wheels turn without the robot moving.

#### Where the pipeline depends on pose, and how drift should break it

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

Simulator stepping (`sim.step`, `step_filter` collisions) and plotting
intentionally stay on ground truth.

## Handheld RealSense + OpenVINS

`--mode explore` runs **the same `auto_explore_navigate.explore` state machine**
with live inputs and human walking instructions. It includes the initial camera
scan, frontier selection/revalidation, depth navigation, LoTIS recovery to saved
viewpoints, route pruning, optional object search/confirmation, and retrieval.
The simulator entry point remains available. Real-camera poses are read-only:
there is no simulated integration, teleport, or robot velocity publisher.

Start these in separate terminals from `mapless-robotics/`:

```bash
./scripts/start_handheld_camera.sh
./scripts/start_handheld_openvins.sh
./scripts/start_handheld_bridge.sh
```

Use the RGB-D camera script: the earlier stereo-only script disables RGB/depth.
All three scripts default to ROS domain 1. `OPENVINS_WORKSPACE` overrides
`~/openvins_ws`; the camera uses the previously patched RealSense ROS driver.
Firmware is unchanged. Keep still initially, then move gently to initialize VIO.

Start the full guided workflow, with Rerun:

```bash
./scripts/run_handheld.sh --mode explore \
  --record-dir runs/handheld/office-full-01

# Full object-search workflow, using a concrete OWLv2 label without Qwen translation:
./scripts/run_handheld.sh --mode explore \
  --record-dir runs/handheld/chair-search-01 --object-query "chair" --no-qwen
```

Omit `--no-qwen` to use the same one-time goal translation as the automatic
launcher. Other object/recovery/scan settings match `auto_explore_navigate`.
LoTIS/DINOv3 load for recovery and retrieval even before the first return is needed.
The launcher automatically uses the existing local ViT-B checkpoint when present;
`DINOV3_WEIGHTS` or `--dinov3-weights` can override it. `--lotis-device cpu` selects CPU.
Use a fresh output directory each time; old runs are retained.

Rerun shows a camera-centred top-down local map, velocity suggestions, input health,
pipeline state, the 3D pose/frontier graph, depth/obstacle data, the analyzed frontier
crop, LoTIS projections/reference views, and optional object detections. The right
column keeps the local map, a large status/target/distance message, and pipeline
state visible throughout mapping. The map follows measured position while keeping
world -Z up and +X right. Blue marks your position/heading, gold the goal, cyan the
actual local planner path, red obstacles, light cells observed free space, and grey
unknown space. The blue circle shows the configured horizontal arrival radius.
Off-screen goals are marked at the map edge with distance. `--map-span 6` controls
the map's width/height in metres. A changed goal suppresses an old goal's path;
no straight-line substitute is drawn when no local plan is available. LoTIS recovery
shows its recorded route in orange, separately labelled from a local collision
plan. The background is the raw rolling occupancy grid, before planner inflation
and endpoint clearing; grey/unknown cells can be traversed at an additional cost.
The map/status heartbeat continues during model loading/inference. Frontier, LoTIS, recovery and object
panes are tabs beneath the map. The viewer follows the shared `live_time` timeline;
if you pause playback to inspect history, return to Follow for current guidance.
Keep the **terminal focused** for keys, as with the simulator's Rerun launcher:

- Space: pause/resume; `f`: analyze; `g`: save.
- `u`: reject the selected frontier; `o`: dismiss the detected object.
- `r`: retry a stopped LoTIS recovery; `b`: physically turn around with guidance.
- Enter: confirm found/end exploration and begin the retrieval workflow.
- `q`/Escape: stop. Ctrl+C exits and saves.

Handheld frontier arrival is position-only: entering the configured
`ExplorationGraph.visited_distance` horizontal arrival radius retires that frontier
without matching its heading.
The local navigator stops at that radius without a final heading turn. Steering
toward the frontier, frontier-direction fusion, saved-view alignment and LoTIS
recovery remain active. The simulator entry point keeps its prior arrival policy.

Add `--skip-view-alignment` to disable the "Align with the recorded view" yaw/tilt
wait when selecting a saved viewpoint, finishing LoTIS recovery, or starting
retrieval. It keeps your measured orientation and continues with the current
camera view. Default behavior still requires alignment. This option does not
skip intentional scans, explicit turn-around commands, or steering along a path.
LoTIS may localize less reliably when the current view differs from the recording.
The setting is saved as `align_recorded_views` in route initialization metadata.

During scans, rotate until the measured camera heading reaches each requested
view. During recovery, turn/tilt to match the saved viewpoint; the program waits
for your actual pose rather than overwriting it. For retrieval, physically return
to the recorded start, keep the estimator running, and press Enter when close
(within 0.5 m by default). LoTIS then guides you from that start to the found object.
A new odometry session requires a new run; independent later route following can
use the archived visual route without sharing its original world origin.

Choose the depth source explicitly for frontier detection, local obstacles and
object localization:

```bash
# Aligned metric depth from the D455 (default):
./scripts/run_handheld.sh --mode explore --scan-views 0 \
  --depth-source realsense --record-dir runs/handheld/depth-camera-01

# Predicted depth from UniK3D using RGB and the calibrated intrinsics:
./scripts/run_handheld.sh --mode explore --scan-views 0 \
  --depth-source estimated --depth-model lpiccinelli/unik3d-vitl \
  --record-dir runs/handheld/depth-model-01
```

Keep your floor-height, object-query and other options on either command. The
source/model is shown in Rerun and stored in `route.json` initialization metadata.
Estimated mode passes no captured depth to analysis and does not mix in camera
depth for local-map refreshes. OpenVINS continues using stereo IR + IMU in both
modes. The ROS bridge still synchronizes RGB and aligned camera depth; this switch
selects the downstream geometry source, not a different camera launch profile.
Model weights must be available locally or downloadable. In a CPU check with two
threads, the cached ViT-L took about 12.5 s to load and 19.4 s for one 640x480 frame;
live latency depends on hardware and the selected model. Predicted metric depth
accuracy also affects floor filtering and obstacle distances.

Current frontier generation and selection:

1. OpenFrontier consumes RGB plus the selected depth. Its network predicts a
   distance field and information-gain classes on a 320x320 crop. The current
   config uses distance-field normalizer 10 and threshold 0.3.
2. Depth gradients supply image-space directions; HDBSCAN clusters frontier
   pixels using position, direction and depth. Each cluster is backprojected into
   the measured world frame, producing a 3D anchor, viewing direction and gain.
3. Captured-depth anchors must lie on valid depth pixels. The graph filters gain
   below 0.5, distance above 7 m from the observing pose, steep vertical viewing
   directions (`abs(vd_y) > 0.65`), and any configured floor-relative height bounds.
   Compatible detections merge within 0.5 m and 30 degrees by default. Previously
   viewed areas reduce gain, and reached/rejected frontiers are excluded.
4. Among candidates linked to archived RGB observations, selection maximizes
   `effective_gain / (1 + horizontal_distance)` multiplied by a forward-facing
   reward `(1 + cos(bearing_error))/2` and the configured global directional
   multiplier. Thus a nearby, useful frontier ahead is preferred. The active goal
   is kept stable; reobservations do not move it.
5. Depth navigation plans to that goal on its rolling obstacle map. Repeated
   explicit low-quality matches can invalidate the target; missing detections
   alone do not. A failed/stuck route can reject the target and trigger LoTIS
   recovery. Reaching uses the current horizontal radius without heading matching
   in handheld mode.

The graph's explicit depth-occlusion rejection is currently disabled. Local A*
planning handles obstacles but inflates them by 0.15 m and clears a 0.30 m region
around its endpoints; frontier selection alone does not certify a free goal.
The plotted frontier markers are learned 3D anchors, not occupancy-grid boundary
extraction, and may still need rejection or a different viewpoint.

For a measured floor reference, add `--camera-height` to the handheld command:

```bash
./scripts/run_handheld.sh --mode explore --scan-views 0 \
  --camera-height 1.50 --record-dir runs/handheld/floor-test-01
```

Replace `1.50` with the measured vertical height in metres from the level floor
below the camera to the RGB optical centre. Hold the camera at that height while
the first fresh bridge frame arrives. The program fixes
`floor_y = initial_camera_y - camera_height` in the current Y-up odometry frame.
It does not shift, rescale or otherwise modify camera poses. Rerun draws the floor
reference, puts the planar map/path overlays at that floor, and reports current
camera height. The reference stays fixed if you raise or lower the camera; it is
saved in the route initialization and graph settings. Restarting OpenVINS/bridge
requires a new run and a newly established reference.

With `--camera-height`, local navigation rejects ground returns within
`--floor-clearance` of the fixed floor (default `0.05` m). Ground samples contribute
free-space rays and free endpoints rather than occupied cells. Returns below that
floor band are ignored; points above the band remain obstacle candidates. Objects
at or below the clearance cannot be distinguished from ground by this height
filter. The camera-relative upper cutoff is retained unless an explicit obstacle
height band is supplied. Rerun reports ground/obstacle counts and the active band.
A bad floor reference or drifting pose still needs correction; this filter does
not estimate a floor plane or adjust the supplied height.

Additional height filtering is optional. With `--camera-height`, you can set:

- `--frontier-min-height` / `--frontier-max-height`: reject frontier anchors below
  or above those heights, in metres above the floor.
- `--obstacle-min-height` / `--obstacle-max-height`: select depth points for the
  local obstacle map using that floor-relative band. The ground-clearance band
  always applies; the effective lower limit is the larger of clearance and your
  minimum. With an explicit minimum but no maximum, the upper side is unbounded.

No frontier height limits are added merely by supplying camera height. Without a
floor reference, the existing camera-relative obstacle filter is retained.
These flags do not change arrival thresholds, collision validation,
IMU calibration or odometry. This is a manually supplied horizontal floor, not
floor-plane detection or tracking for stairs, slopes or multiple floor levels.

The bridge uses system ROS Python, synchronizes RGB/aligned metric depth with
bracketed IMU odometry, obtains IMU-to-RGB optical extrinsics from static TF,
rectifies both images consistently, and converts world Z-up/optical axes into
pipeline Y-up/-Z-forward axes. It uses reliable image QoS by default. Missing data
never receives an invented identity pose. RGB-to-IMU time offset defaults to zero;
`--camera-time-offset` on the bridge accepts a calibrated value (`t_imu=t_rgb+offset`).
The previous IR offset is not silently substituted for RGB calibration.

### Odometry failure and the large allocation error

The failed office runs contain OpenVINS positions hundreds of kilometres from the
origin. The old unbounded exploration grid tried to cover those positions,
producing the reported 4.41 TiB allocation. This is invalid estimator input, not a
reason to increase RAM or suppress the error. New checks stop before recording or
mapping such input, and both exploration/local grids have explicit allocation
budgets. No poses are clipped, recentered, or replaced with commanded motion.

If an odometry fault appears, stop guidance and restart **OpenVINS and the bridge**,
then use a new run directory. Confirm that estimated positions remain reasonable
before walking. Restarting does not prove the underlying VIO drift is resolved;
if it recurs, capture another sensor bag to diagnose it. The handheld defaults
reject radius >100 m or speed >3 m/s; these are operating-envelope checks exposed
as `--max-odometry-distance` and `--max-odometry-speed`, not estimator tuning.
The bridge has matching checks and latches a fault until restarted. Stale input
or expired commands show STOP. An inference exception prints its traceback.

Each run saves `session.rrd`, `guidance.jsonl`, `graph.json`, and the shared
pipeline's immutable `archive/` plus loop-pruned retrieval `frames/` and `route.json`.
Use `--no-spawn` or `--display terminal` to record Rerun data without opening its
viewer; `--rrd /path/session.rrd` chooses another recording path.

The lighter independent modes remain available:

```bash
./scripts/run_handheld.sh --mode preview
./scripts/run_handheld.sh --mode teach --record-dir runs/handheld/office-route
./scripts/run_handheld.sh --mode follow --route runs/handheld/office-route
# A shared-pipeline run can also be followed later using its active archived route:
./scripts/run_handheld.sh --mode follow --route runs/handheld/office-full-01
```

Preview/teach/follow retain terminal/OpenCV displays (`--display terminal` for
text only); their arrow window runs in system Python because the ML environment's
OpenCV is headless. Full exploration uses Rerun. Run one handheld consumer at a
time. ROS and ML exchange a single atomic latest snapshot under `/tmp/mapless-live`.

Focused checks, without a camera or downloads:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_handheld*.py' -v
```

## Scenes to test:

- data/hm3d-train-habitat-v0.2/00352-zmZvNTCxMZE/zmZvNTCxMZE.basis.glb
- data/hm3d-train-habitat-v0.2/00005-yPKGKBCyYx8/yPKGKBCyYx8.basis.glb
- data/hm3d-train-habitat-v0.2/00231-4h4JxvG3cip/4h4JxvG3cip.basis.glb
- data/hm3d-train-habitat-v0.2/00177-VSxVP19Cdyw/VSxVP19Cdyw.basis.glb
- data/hm3d-train-habitat-v0.2/00548-otTm4oTrHvc/otTm4oTrHvc.basis.glb
- data/hm3d-train-habitat-v0.2/00715-kjUg7BaQF1C/kjUg7BaQF1C.basis.glb
- data/hm3d-train-habitat-v0.2/00675-c6TFyURFrL4/c6TFyURFrL4.basis.glb
- data/hm3d-train-habitat-v0.2/00659-ZVPMj4YoZtK/ZVPMj4YoZtK.basis.glb
- data/hm3d-train-habitat-v0.2/00080-gPf8XzHACGn/gPf8XzHACGn.basis.glb
- data/hm3d-train-habitat-v0.2/00559-53jtKd53a1X/53jtKd53a1X.basis.glb
- data/hm3d-train-habitat-v0.2/00777-N17ddiDvJr9/N17ddiDvJr9.basis.glb
- data/hm3d-train-habitat-v0.2/00353-8qbZhbTc1wX/8qbZhbTc1wX.basis.glb
- data/hm3d-train-habitat-v0.2/00609-x1pTUWx9DPr/x1pTUWx9DPr.basis.glb
- data/hm3d-train-habitat-v0.2/00301-JiHGQpwKUvd/JiHGQpwKUvd.basis.glb
- data/hm3d-train-habitat-v0.2/00067-osQy15y8EVT/osQy15y8EVT.basis.glb


- data/hm3d-train-habitat-v0.2/00700-aosjAwX5Lnq/aosjAwX5Lnq.basis.glb
- data/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb
- data/hm3d-train-habitat-v0.2/00733-GtM3JtRvvvR/GtM3JtRvvvR.basis.glb


- data/hm3d-train-habitat-v0.2/00248-PM558qFsyi8/PM558qFsyi8.basis.glb
- data/hm3d-train-habitat-v0.2/00297-jGdNyKqGZJw/jGdNyKqGZJw.basis.glb
- data/hm3d-train-habitat-v0.2/00780-3iZkJUc7KhX/3iZkJUc7KhX.basis.glb
- 00291-Umx6CdjZfvy/Umx6CdjZfvy.basis.glb
