# Automatic exploration and LoTIS returns

`auto_explore_navigate` explores on its own with OpenFrontier and a local depth
planner, uses LoTIS to return to saved viewpoints when it changes goal, and uses
LoTIS again for the final retrieval. See the [README](../README.md#running) for
the command and core parameters.

## Start pose and camera
- The start position and heading are random on each launch. Startup prints
  `Start XYZ=..., yaw=..., seed=...`, and `route.json` stores it. `--seed` reproduces a start.
- `--start-position X Y Z` must be on the navmesh. With `--start-yaw`, 0 faces world -Z and 90 faces world -X.
- `--camera-pitch` (-80 to 80, positive is down) applies to both RGB and Habitat depth
  and stays set across resets. `w`/`s` still tilt the camera at runtime.

## Exploration loop
- Frontier and depth analysis runs in a background worker (`--exploration periodic --interval 0.5`
  by default). The depth planner applies commands at about 10 Hz between analyses.
- The agent commits to one frontier goal. A higher-gain alternative found while driving
  does not interrupt it. The goal ends when:
  - it is reached (0.5 m ground-plane distance),
  - it is rejected with `u`,
  - or it is invalidated after 3 consecutive low-quality matches (gain below 0.5,
    missing or non-finite gain). If a frontier is not observed, the streak resets.
    Absence alone does not prove a target unreachable.
- To pick the next goal, it first re-analyzes the current view and prefers frontiers seen
  there, ranked by current gain. Older frontiers are a fallback, ranked by peak observed gain.
- If nothing is left, it scans up to `--scan-views` (default 6) directions in place.
  If all are empty it pauses. Move to a new spot by hand and press Space.
- Retired frontiers (reached, rejected, invalidated) stay in the graph as history and
  are suppressed nearby, so repeated detections don't bring them back.

## LoTIS returns
When the chosen frontier was seen from a viewpoint more than 0.5 m away, LoTIS drives
back to that viewpoint along the recorded route (backward, then forward along an archived
branch if needed). It then restores the saved heading and camera pitch, clears local
occupancy, and hands back to the depth planner. A viewpoint within 0.3 m skips the return.

After a successful return, the detour beyond that viewpoint is dropped from the retrieval
route. The graph keeps the excluded poses, and Rerun shows them in red.

In Rerun, the return frontier is yellow (`RETURN FRONTIER F...`) and its observing viewpoint
is cyan. The **Return viewpoint** and **Return target** panels show the saved image and status.

### LoTIS policy details
- References skip near-duplicate poses (0.15 m spacing), so in-place scans don't use up
  the encoder's frame budget.
- Speed follows `forward_velocity` from the LoTIS config, capped by `--lotis-speed`.
  Above 35° heading error it turns in place, and it moves forward again below 20°.
- Lookahead uses 0.35 m of recorded path. Arrival means the endpoint is LoTIS's nearest
  visible frame for 3 steps. Near the end it slows down over the last 0.5 m of recorded path.
- LoTIS runs synchronously, and each prediction steps the simulator by 0.1 s, so slow
  inference means slow playback. On CPU it uses `--lotis-cpu-threads` (default 8).

### Failure handling and retries
- **Lost:** no usable route point visible for 60 steps. **Stalled:** less than 0.1 m of progress in 80 steps.
- A leg that stops mid-route gets up to 2 automatic retries. Each first does a 360° sweep
  and faces the best view, then restricts LoTIS matching to a window:
  - **U-turn** (nearest frame jumps between two legs of a U): the first retry follows the U,
    the second cuts across.
  - **Lost / corner:** a window around the last matched frame (0.5 m back and 2 m ahead,
    then 1 m back and 4 m ahead).
- Stalls with the route still in view (likely an obstacle), cancels, and step limits are
  left to you: press `r` to retry. A stopped return blocks Enter until it is resolved.
- If final retrieval stops: arrows move, `a`/`d` turn, `b` turns around, `r` re-encodes
  and retries from the current pose.

Rerun's `pipeline/return_status` shows the state (`tracking`, `searching`, `endpoint_approach`,
`lost_stop`, ...) and frame indices into the compacted LoTIS reference (not archive IDs).

## Outputs
`runs/auto_explore_navigate/<timestamp>/` (or `--run-dir`):

- `archive/`: every recorded exploration image, including abandoned detours.
- `frames/`: the ordered, loop-pruned RGB route used for retrieval.
- `route.json`: active frame IDs, branch parents, excluded branches, start pose,
  odometry and ground-truth positions, object search results, and `retrieval_ready`.
- `graph.json`: poses, frontiers (active and retired), and excluded poses (`--graph-output` to move it).

The pruned route has abandoned detours removed. It is not guaranteed to be the
globally shortest path.
