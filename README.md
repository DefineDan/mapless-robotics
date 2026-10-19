# Mapless Robotics — RealSense handheld setup

Use a RealSense D455 and OpenVINS to explore an environment, record an RGB route,
and follow that route with LoTIS. ChairBot carries the camera and follows the
instructions in Rerun. This guide covers **Ubuntu 24.04 and ROS 2 Jazzy**.

For the project overview, algorithm details, and simulator usage, see the
[main branch README](https://github.com/DefineDan/mapless-robotics/blob/main/README.md).

## 1. Install the dependencies

### ROS and camera driver

Install [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
and connect the D455 over USB 3. With the ROS package repository configured:

```bash
sudo apt update
sudo apt install ros-jazzy-realsense2-camera \
  ros-jazzy-cv-bridge ros-jazzy-message-filters ros-jazzy-tf2-ros \
  python3-numpy python3-opencv
```

For a source installation or a different SDK version, follow the
[RealSense ROS installation guide](https://github.com/realsenseai/realsense-ros#installation-on-ubuntu).
Source your driver workspace after the ROS environment when using a source build.

### Project and Python environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the branch into a directory of your choice:

```bash
git clone --branch realsense_test https://github.com/DefineDan/mapless-robotics.git
cd mapless-robotics
git submodule update --init --recursive -- external/lotis external/dinov3 external/OpenFrontier
uv sync
```

Run the remaining project commands from this checkout's root. The current Python dependency set includes simulator libraries, but handheld operation does not require downloading HM3D scenes.

### Model files

Follow the main repositry guidelines.

### OpenVINS and sensor calibration

Build a ROS 2 OpenVINS workspace using the
[OpenVINS installation guide](https://docs.openvins.com/gs-installing.html).
Before continuing, its `install/setup.bash` must make the `ov_msckf` package and
`subscribe.launch.py` available in your Jazzy environment.

Prepare a **stereo infrared + IMU configuration for your camera and image
profile**. Follow the [sensor calibration guide](https://docs.openvins.com/gs-calibration.html)
for camera intrinsics/distortion, camera–IMU extrinsics, IMU parameters, and timing.
Keep the estimator YAML and its referenced calibration YAML files together.
Device-specific calibration files, driver patches, and a built OpenVINS workspace
are external prerequisites; this repository does not distribute them.

For the camera launch below, configure OpenVINS for two stereo cameras and these
input topics:

| Input | Topic |
|---|---|
| Left infrared image | `/camera/camera/infra1/image_rect_raw` |
| Right infrared image | `/camera/camera/infra2/image_rect_raw` |
| Combined IMU | `/camera/camera/imu` |

Match the calibration to **640×480 infrared images**. Validate stable VIO before
using its poses for mapping or walking guidance.

## 2. Start the sensor pipeline

Open three terminals. In **each ROS terminal**, source Jazzy and use the same
domain ID:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=1
```

### Terminal 1: RealSense

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera camera_name:=camera \
  enable_infra1:=true enable_infra2:=true \
  enable_color:=true enable_depth:=true \
  enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  gyro_fps:=200 accel_fps:=63 \
  enable_sync:=true align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  depth_module.infra_profile:=640x480x30
```

This enables RGB, stereo infrared, aligned depth, and IMU. If you change the
infrared resolution or calibration, update the OpenVINS configuration accordingly.

### Terminal 2: OpenVINS

Replace both paths with your own built workspace and calibrated estimator file:

```bash
export OPENVINS_WORKSPACE="/absolute/path/to/openvins_ws"
export OPENVINS_CONFIG="/absolute/path/to/d455_stereo/estimator_config.yaml"
source "$OPENVINS_WORKSPACE/install/setup.bash"
ros2 launch ov_msckf subscribe.launch.py config_path:="$OPENVINS_CONFIG"
```

OpenVINS uses **stereo infrared images + IMU**. Allow initialization to complete,
then confirm that `/ov_msckf/odomimu` publishes plausible motion and remains
stable when the camera is stationary. The bridge expects this odometry to
describe the IMU pose, not an already transformed RGB camera pose.

### Terminal 3: Bridge

From the project root:

```bash
./scripts/start_handheld_bridge.sh
```

Wait for `Streaming ... RGB-D with interpolated OpenVINS pose`. The bridge
synchronizes RGB and aligned depth, interpolates odometry to the RGB timestamp,
uses TF for the IMU-to-RGB transform, and exports snapshots to `/tmp/mapless-live`.
It converts ROS coordinates into the pipeline's Y-up, camera −Z-forward convention.

The default bridge inputs are:

| Data | Topic / frame |
|---|---|
| RGB | `/camera/camera/color/image_raw` |
| Aligned depth | `/camera/camera/aligned_depth_to_color/image_raw` |
| RGB intrinsics | `/camera/camera/color/camera_info` |
| VIO odometry | `/ov_msckf/odomimu` |
| IMU frame for TF lookup | `camera_imu_optical_frame` |

For different names, pass `--rgb-topic`, `--depth-topic`, `--info-topic`,
`--odom-topic`, or `--imu-frame` to the bridge. Ensure TF describes the same IMU
axes used by the estimator. If changing `--stream-dir`, supply the same directory
to the handheld process.

The launch commands above work without the development machine's helper paths.
`scripts/start_handheld_camera.sh` currently sources a local patched-driver
overlay; use the explicit camera launch above unless you adapt that path.
`scripts/start_handheld_openvins.sh` assumes a configuration under its OpenVINS
workspace; the explicit launch above accepts any calibrated configuration path.

## 3. Explore and record

Keep all three sensor processes running. In a fourth terminal, enter the project
root and set `DINOV3_WEIGHTS` as described above. Start with a new output directory:

```bash
./scripts/run_handheld.sh --mode explore \
  --depth-source estimated \
  --record-dir runs/handheld/my-route
```

Rerun opens automatically. Follow the requested scans and walking instructions.
The workflow selects frontiers, plans local motion, records RGB references, and
uses LoTIS for recovery to saved viewpoints when needed. Press Enter to confirm
the object was found and begin retrieval, or quit to save the exploration run.

To search automatically for a concrete object label:

```bash
./scripts/run_handheld.sh --mode explore \
  --depth-source estimated \
  --object-query "chair" --no-qwen \
  --record-dir runs/handheld/object-search
```

Omit `--no-qwen` to translate a natural-language request into object labels with
Qwen first.

### Depth source and optional settings

`--depth-source estimated` uses **UniK3D** for frontier geometry, obstacle mapping,
planning, and object localization. `--depth-source realsense` uses captured depth
instead; it is the current handheld default, so select `estimated` explicitly
when that is the intended configuration. OpenVINS uses stereo infrared and IMU
in either case.

**Leave camera depth enabled even in estimated mode:** the current bridge still
requires aligned RealSense depth to construct synchronized snapshots. Its depth
measurements are excluded from downstream geometry when using estimated mode.

| Option | Use |
|---|---|
| `--depth-model MODEL_ID` | UniK3D model; default `lpiccinelli/unik3d-vitl`. |
| `--camera-height METRES` | Set a fixed floor reference from the measured startup RGB camera height. |
| `--floor-clearance METRES` | Ground rejection band around that floor; default 0.05 m. |
| `--frontier-min-height` / `--frontier-max-height` | Optional frontier limits above the floor. |
| `--obstacle-min-height` / `--obstacle-max-height` | Optional obstacle-height limits above the floor. |
| `--scan-views 0` | Disable automatic multiview scans and begin with the current view. |
| `--skip-view-alignment` | Skip recorded-view yaw/tilt matching; steering and explicit turns remain active. |
| `--speed METRES_PER_SECOND` | Suggested walking speed; default 0.2 m/s. |
| `--map-span METRES` | Width/height of the local map; default 6 m. |
| `--lotis-device cpu` | Run LoTIS on CPU instead of automatic device selection. |

For a floor reference, measure the vertical distance from the level floor to the
RGB optical centre and pass it with `--camera-height`. Hold that height when the
first frame is captured. Floor-relative height limits require this reference.

### Controls

Keep the **launch terminal focused**, rather than the Rerun window.

| Key | Exploration action |
|---|---|
| Space | Pause/resume. |
| `f` / `g` | Force analysis / save. |
| `u` / `o` | Reject the frontier / dismiss the detected object. |
| `r` | Retry a stopped LoTIS recovery. |
| `b` | Turn around with guidance while a return is stopped. |
| Enter | Confirm found and begin retrieval. |
| `q` / Escape | Stop; Ctrl+C exits and saves. |

For retrieval in the same session, physically return to the recorded start and
press Enter when prompted. Keep OpenVINS and the bridge running so that workflow
retains its odometry frame.

## 4. Follow a saved route

Stop exploration before starting standalone following. Keep the sensor pipeline
running, stand at the recorded start, and face the original direction:

```bash
./scripts/run_handheld.sh --mode follow \
  --route runs/handheld/my-route \
  --direction forward
```

Replace `my-route` with a directory you recorded. Use `--direction backward` to
attempt following from the recorded end toward the start, facing the return path.
Standalone LoTIS following uses live RGB and saved reference images; it does not
run UniK3D. Only one handheld guidance process can use a stream directory at a time.

Rerun shows the reference route, goal, matched/lookahead waypoints, live RGB,
walking status, and LoTIS projections. The blue reference marker is a **matched
recorded waypoint**, not a measured current position. A new OpenVINS session has
an independent origin, so its live trail appears in a separate tab. Standalone
following does not require the original odometry origin.

Keep the terminal focused for **Space**, **`f`**, and **`q`**. `--display both`
selects the terminal/OpenCV display; `--display terminal` gives text only.

## 5. Save and view recordings

Exploration saves `route.json`, `graph.json`, `archive/`, `frames/`,
`guidance.jsonl`, and `session.rrd` in the chosen directory. `archive/` retains all
saved RGB frames; `frames/` exports the loop-pruned route.

Following saves a new timestamped session under `runs/handheld/`, leaving the
reference recording unchanged. Use `--record-dir` for a chosen output directory,
`--rrd` for a chosen recording file, and `--no-spawn` to record without opening
Rerun. Use fresh output paths for each session.

Open a saved exploration without starting the camera:

```bash
.venv/bin/rerun runs/handheld/my-route/session.rrd
```

Press Play for recorded data. For live guidance, keep the `live_time` timeline in
Follow mode.

## Troubleshooting and help

- **Bridge waits for data:** check RGB, aligned depth, camera intrinsics, TF, and
  initialized odometry. Topic/frame names must match your configuration. For
  best-effort image publishers, use `--image-qos best_effort` on the bridge.
- **USB or frame timeouts:** check the USB 3 connection and that only one process
  opens the camera.
- **Odometry fault or restarted estimator/bridge:** restart the affected sensor
  processes and begin a new exploration directory and floor reference. Mapping
  cannot continue across an odometry-session change.
- **STOP / guidance expired:** stop walking and check input freshness and model
  status in Rerun. Initial model loading and inference may take time.

Quit guidance before shutting down the sensor terminals with Ctrl+C. After code
changes, restart guidance; a recovery retry does not reload Python modules.

```bash
./scripts/run_handheld.sh --mode explore --help
./scripts/run_handheld.sh --mode follow --help
./scripts/start_handheld_bridge.sh --help
```
