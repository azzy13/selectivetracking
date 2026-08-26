# GroundingDINO ROS2
## Prerequisites Check

```bash

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Download Model Weights

```bash
cd /isis/home/hasana3/vlmtest/GroundingDINO
mkdir -p weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```

### 2. Build Docker Image

```bash
./docker/build_ros2.sh
```

Self-contained — the `trinity_msgs` message definitions the node publishes are
vendored at `ros2_package/trinity_msgs`, so no `trinity_msgs` checkout is
needed. See "Message version skew" in `DEMO.md` to build against a different
revision.

### 3. Run GroundingDINO Node

```bash
docker run \
  --name groundingdino_node \
  --rm \
  --gpus all \
  --network host \
  --ipc=host \
  -v ${PWD}/weights:/weights:ro \
  -v ${PWD}/outputs:/outputs:rw \
  -e ROS_DOMAIN_ID=0 \
  groundingdino_ros:latest
```

### 4. Run Inference on a Video

Basic inference (detection + tracking):
```bash
docker run --rm --gpus all \
  -v ${PWD}/weights:/app/GroundingDINO/weights:ro \
  -v ${PWD}/videos:/app/GroundingDINO/videos:ro \
  -v ${PWD}/outputs:/app/GroundingDINO/outputs:rw \
  groundingdino_ros:latest \
  python3 demo/inference_w_worker.py \
    --video videos/carla1.mp4 \
    --output outputs/carla1_tracked.mp4 \
    --fp16 \
    --text-prompt "car. "
```

With MoGe-2 depth estimation (adds distance overlay on each bbox):
```bash
docker run --rm --gpus all \
  -v ${PWD}/weights:/app/GroundingDINO/weights:ro \
  -v ${PWD}/videos:/app/GroundingDINO/videos:ro \
  -v ${PWD}/outputs:/app/GroundingDINO/outputs:rw \
  groundingdino_ros:latest \
  python3 demo/inference_w_worker.py \
    --video videos/carla1.mp4 \
    --output outputs/carla1_depth.mp4 \
    --fp16 \
    --text-prompt "car." \
    --depth
```

Output video lands in `outputs/` on your host. Change `--text-prompt` (default: `"red car."`) to detect different objects.

### 6. Test with Sample Video (ROS2 — In Another Terminal)

```bash
docker run -d \
  --name test_publisher \
  --network host \
  --ipc=host \
  -e ROS_DOMAIN_ID=0 \
  -v /isis/home/hasana3/vlmtest/GroundingDINO:/app/groundingdino:ro \
  groundingdino_ros:latest \
  bash -c "cd /app/groundingdino/ros2_package && \
           python3 test_publisher.py --video /app/groundingdino/videos/carla1.mp4 --fps 30"
```

### 7. Verify It's Working

```bash
# Check topics are publishing
ros2 topic list | grep groundingdino

# See detections
ros2 topic echo /groundingdino/tracks --once

# Check FPS
ros2 topic hz /groundingdino/visualization
```

## Common Issues
### "_C is not defined"
→ Rebuild image: `./docker/build_ros2.sh`

## Clean Up

When you're done testing:

```bash
# Stop containers
docker stop groundingdino_node test_publisher
```

## Directory Structure After Setup

```
GroundingDINO/
├── docker/
│   ├── Dockerfile.ros2
│   ├── build_ros2.sh
│   ├── ros2_entrypoint.sh
│   └── README.md
├── weights/
│   └── groundingdino_swint_ogc.pth  ← Downloaded
├── outputs/                          ← Created automatically
│   ├── frames/                       ← Saved visualizations
│   └── tracking.mp4                  ← Video output
└── ros2_package/
    ├── groundingdino_ros/
    └── trinity_msgs/                 ← Vendored msg definitions (0.22)
```



Terminal 1 — start the container once:

cd /isis/home/hasana3/vlmtest/GroundingDINO
mkdir -p outputs/demo /tmp/briefing

cat > /tmp/briefing/config.json <<'JSON'
{"entities_of_interest": [
  {"entity_id": "Car495", "entity_type": "Car",
   "attributes": {"color": "black", "class": "SEDAN.1"}}]}
JSON

docker run -d --name gd_demo --gpus all --ipc=host \
  --user $(id -u):$(id -g) \
  -e ROS_DOMAIN_ID=0 \
  -v "$PWD/weights:/weights:ro" \
  -v "$PWD/videos:/app/GroundingDINO/videos:ro" \
  -v "$PWD/ros2_package:/app/GroundingDINO/ros2_package:ro" \
  -v "$PWD/docker:/app/GroundingDINO/docker:ro" \
  -v /tmp/briefing:/mission_briefing:ro \
  -v "$PWD/outputs/demo:/output:rw" \
  --entrypoint sleep groundingdino_ros:latest infinity

Run inference:

docker exec gd_demo /app/GroundingDINO/docker/run_demo.sh --seconds 30

Options: --video videos/carla1.mp4, --seconds 60, --no-depth, --level,
--save_video (off by default; without it no tracked.avi is written).

Look at the results (on the host, no sudo needed):

ls -la outputs/demo/
vlc outputs/demo/tracked.avi          # needs --save_video; or scp it back
column -s, -t outputs/demo/perceptions.csv | head

Cleanup: docker rm -f gd_demo


docker exec gd_run /app/GroundingDINO/docker/run_demo.sh \
  --full --lockstep --video videos/carla1.mp4 --fps 60 --track-buffer 90
docker rm -f gd_run