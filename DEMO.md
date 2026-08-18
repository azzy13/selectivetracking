# GroundingDINO -> Trinity Perception

The MOT pipeline (GroundingDINO detection + ByteTrack + scene graph) publishes
its tracks as `trinity_msgs/PerceptionArray`, on the topic the rest of the ANSR
stack subscribes to.

---

## Build

```bash
cd /isis/home/hasana3/vlmtest/GroundingDINO
./docker/build_ros2.sh
```

This stages `trinity_msgs` into the build context and builds
`groundingdino_ros:latest`.

### Message version skew — read this before changing the build

The `trinity_msgs` repo is at **0.58** (`a13ec0f`). The running
`architecture_demo` stack pins its `trinity_msgs` submodule at **0.22**
(`31287a1`, "message definitions for perception 0.22"). The two revisions
define `Detection` and `Perception` differently:

| | 0.22 (what the stack runs) | 0.58 (repo HEAD) |
|---|---|---|
| `Perception` | 8 fields | adds `frame_number`, `occlusion`, `pose` |
| `Detection` | has `yaw`, `location`, `spawned_prob` | drops those; adds `tracking_id`, `occlusion_level`, `occlusion_label`, `pose_idx`, `pose_label` |
| `TrackPose` / `TrackPoseArray` | absent | present |

ROS 2 will not connect a publisher and a subscriber whose type definitions
differ, **and it fails silently** — no error, no warning, no data. So the build
pins 0.22 to match the stack. `build_ros2.sh` stages it with `git archive`:

```bash
./docker/build_ros2.sh                          # 0.22, matches the stack
TRINITY_MSGS_REF=master ./docker/build_ros2.sh  # 0.58, for when the stack moves
TRINITY_MSGS_REF=WORKTREE ./docker/build_ros2.sh
```

The node writes version-dependent fields through a `hasattr` guard, so one
source works against either revision. Under 0.22, `Perception.frame_number`
does not exist and is skipped; `PerceptionArray.frame_num` carries the frame
count instead.

---

## Run

```bash
docker run --rm --gpus all --ipc=host --network host \
  -e ROS_DOMAIN_ID=0 \
  -v ${PWD}/weights:/weights:ro \
  -v /path/to/mission_briefing:/mission_briefing:ro \
  groundingdino_ros:latest
```

### Check the output without the simulator

```bash
./docker/check_perception_output.sh              # depth projection
./docker/check_perception_output.sh --no-depth   # ground-plane fallback
./docker/check_perception_output.sh --color white
```

Runs the pipeline on `videos/color_car.mp4` against a stub AirSim publisher
(`ros2_package/sim_stub_publisher.py`) and echoes one message. The stub stamps
frames from a synthetic clock starting at t=1000s, so a sim-clock stamp is
visibly distinguishable from a wall-clock one.

### Save the results to disk

Start the container once, then run the demo script inside it:

```bash
mkdir -p outputs/demo /tmp/briefing
cat > /tmp/briefing/config.json <<'JSON'
{"entities_of_interest": [
  {"entity_id": "Car495", "entity_type": "Car",
   "attributes": {"color": "black", "class": "SEDAN.1"}}]}
JSON

docker run -d --name gd_demo --gpus all --ipc=host \
  -e ROS_DOMAIN_ID=0 -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -v "$PWD/weights:/weights:ro" \
  -v "$PWD/videos:/app/GroundingDINO/videos:ro" \
  -v "$PWD/ros2_package:/app/GroundingDINO/ros2_package:ro" \
  -v "$PWD/docker:/app/GroundingDINO/docker:ro" \
  -v /tmp/briefing:/mission_briefing:ro \
  -v "$PWD/outputs/demo:/output:rw" \
  --entrypoint sleep groundingdino_ros:latest infinity

docker exec gd_demo /app/GroundingDINO/docker/run_demo.sh --seconds 30
docker rm -f gd_demo
```

Everything lands in `outputs/demo/` on the host:

| File | Written by | Contents |
|---|---|---|
| `tracked.avi` | `video_saver.py` | annotated video, boxes + track ids |
| `perceptions.jsonl` | `perception_recorder.py` | one JSON object per `PerceptionArray` |
| `perceptions.csv` | `perception_recorder.py` | one row per perception |
| `node.log` | the node | projection path, prompt, entities |
| `stub.log`, `saver.log`, `recorder.log` | helpers | |

`HOST_UID`/`HOST_GID` hand the files back to you; the container runs as root,
so without them everything in `/output` is root-owned.

The node itself has **no disk-output flag** — it only publishes ROS topics.
`video_saver.py` and `perception_recorder.py` are separate subscriber nodes,
which is why the demo script starts four processes rather than one.

---

## The topic

`/vanderbilt/fake_perception/data` — `trinity_msgs/PerceptionArray`,
QoS `KEEP_LAST` depth 1, `RELIABLE`, `VOLATILE`.

That is what `prediction_node` subscribes to and what
`world_model/fake_perception_node.py` publishes; this pipeline is a drop-in for
the latter. The topic is the `trinity_perception_topic` parameter, and the whole
output is behind `publish_trinity_perception` (default `true`). The pre-existing
`/perception/detections` and `/perception/perceptions` publishers are unchanged
and still publish alongside.

### Example message

From `./docker/check_perception_output.sh` — two cars tracked, the black one
matched to the entity of interest:

```yaml
stamp:
  sec: 1020            # sim clock (stub started at t=1000), not wall clock
  nanosec: 200000000
frame_num: 202
perceptions:
- tracking_id: 1
  target_entity_id: ''         # no colour evidence -> no match
  detection_prob: 0.8569320440292358
  location:
  - -228.61904907226562        # metres, AirSim NED
  - 13.187518119812012
  - -0.20000000298023224
  yaw: 0.0
  entity_class: ''
  entity_color: ''
  match_prob: 0.0
- tracking_id: 2
  target_entity_id: Car495     # matched the black entity of interest
  detection_prob: 0.8610715270042419
  location:
  - -229.09007263183594
  - -57.086063385009766
  - -0.20000000298023224
  yaw: 0.0
  entity_class: ''
  entity_color: black
  match_prob: 1.0
```

Two things worth noticing. The stamp is `1020`, not a wall-clock epoch — the
node copied the image's stamp. And `location` is hundreds of metres, in the
frame the drone pose is expressed in, not a normalized `0..1` image coordinate.

Both projection paths were exercised and agree to within 0.2 m on the same
scene — the depth path returns z = -0.2 (the camera's mounting offset), the
ground-plane fallback returns z = 0.0 by construction:

```
[INFO] [groundingdino_node]: Depth stream active (1366x768, 32FC1): projecting with depth
[INFO] [groundingdino_node]: Projecting via depth (depth=2)

[INFO] [groundingdino_node]: Projecting via ground_plane (ground_plane=2)   # --no-depth
```


### Field mapping

| Field | Source | Notes |
|---|---|---|
| `stamp` | image `header.stamp` | sim clock, not wall clock |
| `frame_num` | frame counter | uint16, wraps at 65536 |
| `tracking_id` | ByteTrack `track_id` | uint16 |
| `detection_prob` | track score | |
| `location` | `geometry.GroundProjector` | **metres, AirSim NED**, from the bbox bottom-centre |
| `entity_color` | scene graph LAB classifier | remapped to the episodes' vocabulary |
| `target_entity_id` | mission briefing | set only on a colour match |
| `match_prob` | scene-graph mission score | for the matched entity |
| `yaw` | — | always 0.0, not estimated |
| `entity_class` | — | always "", see below |

---

## Talking points

- **Pipeline stages.** A frame goes GroundingDINO (open-vocabulary detection
  from a text prompt built out of the mission briefing) -> scale-aware
  confidence filter -> ByteTrack (Kalman + IoU association, giving stable track
  ids) -> scene graph (per-track LAB colour voting, region, motion/heading) ->
  per-entity mission filter, which scores each track's accumulated colour
  evidence against each entity of interest. The last two stages existed for the
  offline eval path but the ROS node never instantiated them; it does now, and
  that is what makes `entity_color`, `target_entity_id` and `match_prob`
  possible.

- **The ROS interface.** In: scene camera, depth camera + `CameraInfo`, drone
  pose, and `/mission_briefing/config.json`. Out: `PerceptionArray` on
  `/vanderbilt/fake_perception/data`, plus the pre-existing detection,
  visualization and debug topics. Two failure modes are handled by degrading
  rather than stopping: no `CameraInfo` falls back to FOV-derived intrinsics,
  and no depth falls back to intersecting the pixel ray with the ground plane.
  The node logs which projection path it is using.

- **`location` is the field that took real work.** Everything else is a
  rename; this one needed the camera geometry. The bbox bottom-centre (where
  the vehicle meets the ground) is unprojected through the depth sample, then
  through `C · Dᵀ · S · Rᵀ · T` — the same world->camera chain
  `point_cloud_node3.get_extrinsic_from_pose()` uses — to get metres NED.
  Verified against hand-computed synthetic poses in `tests/test_geometry.py`.

- **What is missing, deliberately.**
  - `entity_class` is left empty. The episodes name classes as
    `SEDAN.POLICE`, `SUV.1`, `TRUCK.PICKUP`, `MINIVAN.LARGE` — 11 tokens that
    an open-vocabulary detector cannot distinguish. It can tell you "car".
    Emitting a guessed token would be worse than emitting nothing, because
    consumers string-match it. For the same reason association matches on
    colour only.
  - `yaw` is 0.0. The scene graph tracks a heading, but in *image* space; the
    conversion to world yaw is not implemented.
  - `occlusion` / `pose` (0.58) and `occlusion_level` / `occlusion_label` /
    `pose_idx` / `pose_label` on `Detection` have no definition anywhere —
    no producer, no consumer, no documentation in the trinity repo or the
    stack. They need an enum from SRI before they can be filled meaningfully.
  - `TrackPose` / `TrackPoseArray` are defined in trinity_msgs 0.58 and
    published by nothing and subscribed to by nothing anywhere in this tree.
    Filling one needs a 6x6 pose covariance; ByteTrack's Kalman filter carries
    an 8x8 covariance in image `xyah` space, so it would have to be propagated
    through the unprojection Jacobian. Not built.
  - The colour classifier's hue bins come from measured sRGB swatches and
    have not been validated against real episode footage. Its *achromatic*
    thresholds are inherited and strict: a patch is only "white" above
    L > 200, so the white car in the sample video votes 10/16 "gray" and
    comes out as `entity_color: ''` rather than `white`. That is the
    intended failure mode — an empty colour is ignored downstream, a wrong
    one gets matched against — but the L thresholds are the first thing to
    tune against real footage. They were left alone here because the
    offline eval path shares the classifier.

- **AirSim wiring — what exists (inventoried, not built).**
  - Consumed by this node: `.../Drone1/sensors/front_center1/scene_camera/image`,
    `.../front_center1/depth_planar_camera/{image,camera_info}`,
    `.../Drone1/actual_pose`.
  - Available and unused: `.../depth_planar_camera/{point_cloud,pose}`,
    `.../front_center1/segmentation_camera/image`, `.../Drone1/odom_local_ned`,
    `.../Drone1/actual_pose_10hz`, and per-entity ground truth on
    `/viaduct/Sim/SceneDroneSensors/env_actors/<entity_id>/actual_kinematics`
    — that last one is how `fake_perception_node` cheats, and is the obvious
    way to score this pipeline's `location` against truth.
  - **Topic prefix varies.** Some nodes use `/Sim/SceneDroneSensors/...` and
    others `/viaduct/Sim/SceneDroneSensors/...`. All four input topics are
    parameters; check which prefix the deployment uses before wiring up.
  - ADK boundary: `/adk_node/output/{scenario_starting,scenario_ending}`
    (this node ignores both — it does not shut down on scenario end the way
    the world_model nodes do), and `/adk_node/input/perception`, which takes
    `adk_node/TargetPerception`, a different type reached by going through
    the world model.
  - `docker/docker-compose.groundingdino.yml` exists and expects an external
    `ros_network_$USER` and a running `adk` service. The SRI perception
    container it would displace is commented out in
    `airsim/release_installer/docker/docker-compose.yml`
    (`sri-vanderbilt/perception:latest`, `DETECTION_MODEL=yolo_v8`).
