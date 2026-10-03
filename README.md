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
collision seen when both are loaded in the opposite order.




python -m mapless_robotics.explore_navigate \
  --index 10 \
  --scenes-dir /Users/dan/Code/kth/mapless-robotics/data/versioned_data/hm3d-0.2/hm3d/train \
  --exploration manual --depth-source habitat \
  --lotis-config /Users/dan/Code/kth/kth-mapless-robotics/src/mapless_robotics/configs/lotis_config.yaml \
  --checkpoint /Users/dan/Code/kth/mapless-robotics/models/lotis/final_model.pth \
  --model-config /Users/dan/Code/kth/mapless-robotics/models/lotis/final_config.yaml


python -m mapless_robotics.teach_repeat --index 10 --checkpoint /Users/dan/Code/kth/mapless-robotics/models/lotis/final_model.pth --lotis-config /Users/dan/Code/kth/kth-mapless-robotics/src/mapless_robotics/configs/lotis_config.yaml


python -m mapless_robotics.walk_frontiers --index 10 --scenes-dir /Users/dan/Code/kth/mapless-robotics/data/versioned_data/hm3d-0.2/hm3d/train --viewer rerun


Alright got it. Now let's advance. The idea is the following. 

We have our exploration graph and we want to use that to navigate during the exploration phase (called teach phase right now). Maybe lets rename it. We have an expolartion phase and then a navigation / retrieval phase (where we just go to the goal called repeat right now). Well the main idea is that I want to use lotis to navigate during expolration as well. Here is how it should work:

Based on the exploration graph we have for each frontier node Pose nodes that are connected to it. 