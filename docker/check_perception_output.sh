#!/bin/bash
#
# Runs the pipeline on a short sample video and echoes one PerceptionArray
# from the perception output topic, so the field mapping can be eyeballed.
#
# Everything runs inside one container against a stub AirSim publisher --
# no simulator required. Usage:
#
#   ./docker/check_perception_output.sh                    # depth path
#   ./docker/check_perception_output.sh --no-depth         # ground-plane path
#   ./docker/check_perception_output.sh --color white
#   ./docker/check_perception_output.sh --video videos/carla1.mp4
#
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE="${IMAGE:-groundingdino_ros:latest}"
VIDEO="videos/color_car.mp4"
DEPTH="--depth"
CAMERA_RPY="[0.0,-90.0,0.0]"   # nadir: the ground plane is in view
TIMEOUT="${TIMEOUT:-180}"
ENTITY_COLOR="${ENTITY_COLOR:-black}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)    VIDEO="$2"; shift 2 ;;
        --no-depth) DEPTH=""; shift ;;
        --color)    ENTITY_COLOR="$2"; shift 2 ;;
        --level)    CAMERA_RPY="[0.0,0.0,0.0]"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# A mission briefing naming one entity of interest, so target_entity_id has
# something to match against. Colour is the knob that decides whether a
# track matches; see DEMO.md.
BRIEFING="$(mktemp -d)"
trap 'rm -rf "$BRIEFING"' EXIT
cat > "$BRIEFING/config.json" <<JSON
{
  "entities_of_interest": [
    {
      "entity_id": "Car495",
      "entity_type": "Car",
      "attributes": {"color": "${ENTITY_COLOR}", "class": "SEDAN.1"}
    }
  ],
  "scenario_id": "perception-output-check"
}
JSON

echo "========================================"
echo "Perception output check"
echo "========================================"
echo "image:    $IMAGE"
echo "video:    $VIDEO"
echo "depth:    ${DEPTH:-off (ground-plane fallback)}"
echo "entity:   Car495, colour ${ENTITY_COLOR}"
echo ""

docker run --rm --gpus all --ipc=host \
    -e ROS_DOMAIN_ID=0 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -v "$PROJECT_ROOT/weights:/weights:ro" \
    -v "$PROJECT_ROOT/videos:/app/GroundingDINO/videos:ro" \
    -v "$PROJECT_ROOT/ros2_package:/app/GroundingDINO/ros2_package:ro" \
    -v "$BRIEFING:/mission_briefing:ro" \
    "$IMAGE" \
    bash -c "
set -e
source /opt/ros/humble/setup.bash
source /app/ros2_ws/install/setup.bash
export PYTHONPATH=/app/GroundingDINO:/app/GroundingDINO/eval:\$PYTHONPATH
cd /app/GroundingDINO

echo '--- msgs definitions built into this image ---'
ros2 interface show msgs/msg/Perception
echo ''

echo '--- starting groundingdino node ---'
ros2 run groundingdino_ros groundingdino_node --ros-args \
    -p model_weights:=/weights/groundingdino_swinb_cogcoor.pth \
    -p camera_rpy_deg:='${CAMERA_RPY}' \
    -p text_threshold:=0.25 \
    -p box_threshold:=0.30 \
    > /tmp/node.log 2>&1 &
NODE_PID=\$!

# Wait for the model to load and the node to advertise the topic.
for i in \$(seq 1 ${TIMEOUT}); do
    if ros2 topic list 2>/dev/null | grep -q '/vanderbilt/fake_perception/data'; then
        echo \"node ready after \${i}s\"
        break
    fi
    if ! kill -0 \$NODE_PID 2>/dev/null; then
        echo 'NODE DIED:'; tail -40 /tmp/node.log; exit 1
    fi
    sleep 1
done

echo '--- starting sim stub publisher ---'
python3 ros2_package/sim_stub_publisher.py \
    --video '${VIDEO}' --fps 10 --loop ${DEPTH} \
    > /tmp/stub.log 2>&1 &
STUB_PID=\$!
sleep 20
if ! kill -0 \$STUB_PID 2>/dev/null; then
    echo 'STUB DIED:'; tail -30 /tmp/stub.log; exit 1
fi

echo ''
echo '=============================================='
echo 'PerceptionArray on /vanderbilt/fake_perception/data'
echo '=============================================='
timeout 60 ros2 topic echo --once --full-length \
    /vanderbilt/fake_perception/data || {
        echo 'NO MESSAGE RECEIVED'
        echo '--- node log ---';  tail -40 /tmp/node.log
        echo '--- stub log ---';  tail -20 /tmp/stub.log
        echo '--- topics ---';    ros2 topic list
        exit 1; }

echo ''
echo '--- node log (projection path, entities) ---'
grep -E 'Perception|Projecting|Depth stream|entities|Entities|CameraInfo|prompt' \
    /tmp/node.log | tail -20 || true

kill \$STUB_PID \$NODE_PID 2>/dev/null || true
wait 2>/dev/null || true
"
