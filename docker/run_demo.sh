#!/bin/bash
#
# Runs the full pipeline INSIDE the container and saves everything to /output.
#
# Intended use: start the container in one terminal, then run this script in
# it. See DEMO.md. Produces, in /output:
#
#   tracked.avi        annotated video (boxes + track ids)
#   perceptions.jsonl  one JSON object per PerceptionArray
#   perceptions.csv    one row per perception
#   node.log, stub.log
#
# Usage (inside the container):
#   /app/GroundingDINO/docker/run_demo.sh
#   /app/GroundingDINO/docker/run_demo.sh --seconds 60 --video videos/carla1.mp4
#
set -euo pipefail

VIDEO="videos/color_car.mp4"
SECONDS_TO_RUN=30
OUT_DIR="/output"
FPS=10
DEPTH="--depth"
CAMERA_RPY="[0.0,-90.0,0.0]"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)    VIDEO="$2"; shift 2 ;;
        --seconds)  SECONDS_TO_RUN="$2"; shift 2 ;;
        --out)      OUT_DIR="$2"; shift 2 ;;
        --fps)      FPS="$2"; shift 2 ;;
        --no-depth) DEPTH=""; shift ;;
        --level)    CAMERA_RPY="[0.0,0.0,0.0]"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ROS setup scripts reference unbound variables; -u must be off while they run.
set +u
source /opt/ros/humble/setup.bash
source /app/ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/app/GroundingDINO:/app/GroundingDINO/eval:${PYTHONPATH:-}
cd /app/GroundingDINO

mkdir -p "$OUT_DIR"

echo "========================================"
echo "GroundingDINO -> trinity Perception"
echo "========================================"
echo "video:    $VIDEO"
echo "duration: ${SECONDS_TO_RUN}s at ${FPS} fps"
echo "output:   $OUT_DIR"
echo "depth:    ${DEPTH:-off (ground-plane fallback)}"
echo ""

PIDS=()
cleanup() {
    echo ""
    echo "--- stopping ---"
    # SIGINT so the writers flush and close their files. The detection node
    # takes a few seconds to tear down its CUDA context; without the grace
    # period it gets SIGKILLed and bash reports a "Killed" job.
    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 15); do
        still_running=0
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && still_running=1
        done
        [ "$still_running" -eq 0 ] && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/4] starting detection + tracking node (loads the model, ~10s)"
ros2 run groundingdino_ros groundingdino_node --ros-args \
    -p model_weights:=/weights/groundingdino_swinb_cogcoor.pth \
    -p camera_rpy_deg:="${CAMERA_RPY}" \
    -p box_threshold:=0.30 \
    -p text_threshold:=0.25 \
    > "$OUT_DIR/node.log" 2>&1 &
PIDS+=($!)
NODE_PID=$!

for i in $(seq 1 180); do
    if ros2 topic list 2>/dev/null | grep -q '/vanderbilt/fake_perception/data'; then
        echo "      node ready after ${i}s"
        break
    fi
    if ! kill -0 $NODE_PID 2>/dev/null; then
        echo "      NODE DIED:"; tail -40 "$OUT_DIR/node.log"; exit 1
    fi
    sleep 1
done

echo "[2/4] starting perception recorder -> $OUT_DIR/perceptions.{jsonl,csv}"
python3 ros2_package/perception_recorder.py --output "$OUT_DIR/perceptions" \
    > "$OUT_DIR/recorder.log" 2>&1 &
PIDS+=($!)

echo "[3/4] starting video saver -> $OUT_DIR/tracked.avi"
python3 ros2_package/video_saver.py --output "$OUT_DIR/tracked.avi" --fps "$FPS" \
    > "$OUT_DIR/saver.log" 2>&1 &
PIDS+=($!)

echo "[4/4] starting AirSim stub publisher ($VIDEO)"
python3 ros2_package/sim_stub_publisher.py \
    --video "$VIDEO" --fps "$FPS" --loop ${DEPTH} \
    > "$OUT_DIR/stub.log" 2>&1 &
PIDS+=($!)

echo ""
echo "running for ${SECONDS_TO_RUN}s ..."
sleep "$SECONDS_TO_RUN"

cleanup
trap - EXIT

# Container runs as root, so everything in the mounted /output lands
# root-owned. Hand it back if the caller passed their ids.
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" "$OUT_DIR" 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "Output in $OUT_DIR"
echo "========================================"
ls -la "$OUT_DIR"
echo ""
echo "--- projection path used ---"
grep -E 'Projecting via|Depth stream|entities of interest' "$OUT_DIR/node.log" | tail -5 || true
echo ""
echo "--- first perceptions (csv) ---"
head -6 "$OUT_DIR/perceptions.csv" || true
echo ""
echo "--- tracks matched to an entity of interest ---"
awk -F, 'NR>1 && $4 != "" {print $4}' "$OUT_DIR/perceptions.csv" 2>/dev/null \
    | sort | uniq -c || true
