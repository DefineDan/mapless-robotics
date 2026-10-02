


## Openfrontier

### Instalation
```bash
uv venv --python 3.12 .venv_open_fron
source .venv_open_fron/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cd external/OpenFrontier/
uv pip install -r requirements.txt
bash scripts/download_weights.sh

git submodule update --init third_party/UniK3D
uv pip install -e third_party/UniK3D
uv pip install "setuptools<81"
```
## Frontier exploration and graph generation

Run from `mapless-robotics/` with an environment containing Habitat-Sim,
OpenCV with GUI support, NumPy, PyYAML, PyTorch, and OpenFrontier's inference
dependencies in **the same environment**. The project now declares OpenFrontier's
runtime dependencies in `pyproject.toml` and `uv.lock`. For an existing `.venv`
with working Habitat and PyTorch, install the missing inference packages without
rebuilding Habitat:

```bash
uv pip install --python .venv/bin/python 'hdbscan>=0.8.44' \
  'segmentation-models-pytorch==0.3.3' 'setuptools<81'
source .venv/bin/activate
```

This is sufficient for `--depth-source habitat`; UniK3D is only required for
`--depth-source estimated`. Use `PYTHONPATH=src` if this project is not installed.
The default checkpoint is `external/OpenFrontier/model_weights/rgbd_11cls.pth`;
use `--frontier-weights /path/to/rgbd_11cls.pth` for a checkpoint elsewhere.
The default detector settings come from OpenFrontier's `config/navigation.yaml`;
`--frontier-config` can select another YAML file. Map/planning/VLM settings in
that file are not used by this pipeline.

For this workspace, a scene and checkpoint are already available in the sibling
`openfrontier` checkout. From `mapless-robotics/`, run:

```bash
PYTHONPATH=src python -m mapless_robotics.walk_frontiers \
  --scene ../openfrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../openfrontier/model_weights/rgbd_11cls.pth \
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
Estimated depth requires UniK3D (`uv pip install -e external/OpenFrontier/third_party/UniK3D`)
and its model weights; the first use may download the pretrained model.
`--depth-model` selects the UniK3D model. Habitat depth bypasses UniK3D while
still running the real OpenFrontier detector. Manual analysis is the default;
estimated depth is the default depth source.

### Rerun viewer

Add `--viewer rerun` to stream the live RGB camera, OpenFrontier's marked
capture, and a native 3D pose/frontier graph to Rerun:

```bash
PYTHONPATH=src python -m mapless_robotics.walk_frontiers \
  --scene ../openfrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../openfrontier/model_weights/rgbd_11cls.pth \
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
  --scene ../openfrontier/examples/mv2HUxq3B53.glb \
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

## Teach, explore, and repeat in Rerun

`teach_repeat_frontiers.py` combines the two workflows. Drive a route while
OpenFrontier builds the graph, then press Enter to reset the robot to the start
and let LoTIS repeat the route. Frontier analysis continues during the repeat.
Use the terminal for controls; Rerun displays the live camera with LoTIS's
trajectory points, the marked frontier image, and the 3D graph.

From `mapless-robotics/` with the project environment activated:

```bash
PYTHONPATH=src python -m mapless_robotics.teach_repeat_frontiers \
  --scene ../openfrontier/examples/mv2HUxq3B53.glb \
  --frontier-weights ../openfrontier/model_weights/rgbd_11cls.pth \
  --dinov3-weights models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --exploration manual --depth-source habitat
```

Press `f` to detect frontiers, arrows to move, A/D to turn, W/S to tilt the
camera, `g` to save, Enter to start repeating, and Q/Escape to quit. Choose
`--exploration periodic --interval 1` or `--exploration movement` for automatic
frontier captures. The route frames and graph are saved together under
`runs/teach_repeat_frontiers/<timestamp>/`; `--run-dir` and `--graph-output`
override those paths. Habitat is initialized before LoTIS to avoid the LLVM
collision seen when both are loaded in the opposite order.
