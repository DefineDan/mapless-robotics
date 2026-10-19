# Mapless Robotics
Visual exploration and navigation without a dense map
representation for collaborative object search and retrieval.

https://github.com/user-attachments/assets/f7808790-3a7f-4a6f-a624-df35bfd8f7fa

## Idea
Visual navigation is already mapless: a follower, robot or human, only needs a reference trajectory of RGB images. Exploration, on the other hand, usually builds a map with SLAM. In search and rescue, for example, the area is explored and mapped first, and then the map is thrown away because the follower only uses the RGB route.

Can we explore without a map too, and build the whole system on RGB only? We explore with frontiers predicted from RGB images and navigate using a visual memory, with no dense map at any point. The only exception is a small local obstacle map that the depth planner uses for short-range driving.

## Application
Multi-robot search and retrieval or rescue. A swarm of lightweight drones explores quickly. Once one finds the target, its visual trajectory guides another agent to it: a heavier robot like a quadruped or humanoid, or a person with a smartphone camera or smart glasses.

## Approach
Everything currently runs in the Habitat simulator on HM3D scenes.

1. **Explore:** The explorer looks for frontiers (boundaries to unexplored space) directly in its camera images, using a [FrontierNet](https://arxiv.org/abs/2501.04597)-style detector (we use the implementation from [OpenFrontier](https://arxiv.org/abs/2603.05377)). Each frontier gets a predicted information gain, and the explorer picks one, drives toward it with a local depth planner, and repeats. Depth comes from the simulator or from RGB using UniK3D. There is no dense 3D map, only a sparse graph of poses and frontiers.
2. **Return to earlier viewpoints:** When the next frontier was seen from somewhere else, the explorer uses [LoTIS](https://arxiv.org/abs/2602.18803) to drive back to that viewpoint along its own recorded images, not along a map.
3. **Detect:** An open-vocabulary detector (OWLv2) looks for the queried object in the camera stream. A small language model (Qwen) can first turn a request like "a place to sit" into object labels. Once several captures agree on the object's position, the explorer drives up to it.
4. **Extract reference:** The RGB frames from the start to the object are saved as the reference route. Detours that were abandoned during exploration are removed.
5. **Navigate (follower):** LoTIS localizes the follower's live camera view within this reference route and guides it to the object. In the target application the follower is a different agent, such as Spot or a person with a phone or smart glasses, so camera height and viewpoint differ from the explorer's (*cross-embodiment* navigation).

## Setup
1. Get the submodules:
    ```bash
    git submodule update --init --recursive
    ```
2. Install dependencies and create a venv using [uv](https://docs.astral.sh/uv/):
    ```bash
    uv sync
    ```
   This also installs OpenFrontier's dependencies (UniK3D and pywavemap are built
   from source). uv 0.7.x can't parse `pyproject.toml`; run `uv self update` first if needed.
3. Download the HM3D scenes (see the [LoTIS README](external/lotis/README.md)).
4. Download the model weights:
   - **OpenFrontier:** `cd external/OpenFrontier && bash scripts/download_weights.sh`
   - **LoTIS:** `.venv/bin/hf download fnnBsch/lotis final_model.pth --local-dir external/lotis`
   - **DINOv3 ViT-B/16:** request access on [Meta's download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and download the PyTorch `.pth` checkpoint.

   UniK3D, OWLv2 and Qwen download from Hugging Face on first use.
5. Point a LoTIS config at your files. Copy
   [`src/mapless_robotics/configs/lotis_config.yaml`](src/mapless_robotics/configs/lotis_config.yaml)
   and set `checkpoint_path`, `config_path`, `dinov3_weights`, `dinov3_repo` and
   `device` (`cuda`, `mps` or `cpu`).

## Running
The main entry point is `auto_explore_navigate`. It explores a Habitat scene on its own,
searches for an object, and then uses LoTIS to guide a follower from the start back
to the object along the recorded RGB route.

```bash
python -m mapless_robotics.auto_explore_navigate \
  --scenes-dir /path/to/hm3d/train \
  --index 10 \
  --depth-source habitat \
  --lotis-config /path/to/lotis_config.yaml \
  --model-config /path/to/lotis/final_config.yaml \
  --checkpoint /path/to/lotis/final_model.pth \
  --object-query "bed" \
  --no-qwen \
  --camera-pitch 30 \
  --odometry-noise 0.01
```

Rerun opens in a separate window. **Keep the terminal focused**, because it reads the keys.

### Core parameters
| Parameter | Meaning |
|---|---|
| `--scenes-dir`, `--index` | HM3D split directory and scene number (`--index 10` loads `00010-*`). Use `--scene path.glb` for a single scene instead. |
| `--depth-source` | `habitat` uses the simulator's ground-truth depth. `estimated` (default) predicts depth from RGB with UniK3D, which is the RGB-only setting we are aiming for, but slower and noisier. |
| `--lotis-config`, `--model-config`, `--checkpoint` | LoTIS method settings, model architecture, and weights. `--dinov3-weights` overrides the DINOv3 path. |
| `--object-query` | What to search for. By default Qwen first turns a request like "a place to sit" into detector labels. `--no-qwen` passes the text straight to OWLv2, so use a plain label like "bed". Without a query you decide when the object is found (Enter). |
| `--camera-pitch` | Initial downward camera tilt in degrees (default 10). |
| `--odometry-noise` | Run on drifting odometry instead of ground-truth pose. `0.01` gives about 2% drift. Default `0`. |
| `--seed` | Reproduce a random start pose. `--start-position X Y Z` and `--start-yaw` set one explicitly. |
| `--exploration`, `--interval` | When frontier analysis runs. Default: `periodic` every 0.5 s. |

Run `python -m mapless_robotics.auto_explore_navigate --help` for the full list.

### Controls
- `Space`: pause or resume. Moving by hand (arrows, `a`/`d`, `w`/`s`) also pauses.
- `u`: mark the current frontier unreachable. `o`: dismiss a found object.
- `f`: force a frontier analysis. `g`: save the run. `r`: retry a stopped LoTIS return.
- `Enter`: declare the object found and start LoTIS retrieval from the start.
- `q` / `Esc`: quit, or stop an active LoTIS return.

Outputs are written to `runs/auto_explore_navigate/<timestamp>/`.

## Further docs
- [Automatic exploration and LoTIS returns](docs/auto_explore_navigate.md): how goals are picked, returns and retries, outputs
- [Object search](docs/object_search.md): Qwen and OWLv2 detection and confirmation
- [Noisy odometry](docs/odometry.md): noise model and where drift affects the pipeline
- [Other scripts](docs/other_scripts.md): `walk`, `walk_frontiers`, `teach_repeat`, `explore_navigate`
