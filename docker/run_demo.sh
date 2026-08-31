#!/bin/bash
#
# Runs the full pipeline INSIDE the container and saves everything to /output.
#
# Intended use: start the container in one terminal, then run this script in
# it. See DEMO.md. Produces, in /output:
#
#   perceptions.jsonl  one JSON object per PerceptionArray
#   perceptions.csv    one row per perception
#   node.log, stub.log
#   tracked.avi        annotated video -- only with --save_video
#
# Video is off by default. Writing it costs an annotated-frame render and an
# image publish per frame, which is the most expensive thing in the pipeline
# after inference, and the detection output does not depend on it.
#
# Usage (inside the container):
#   /app/GroundingDINO/docker/run_demo.sh
#   /app/GroundingDINO/docker/run_demo.sh --seconds 60 --video videos/carla1.mp4
#   /app/GroundingDINO/docker/run_demo.sh --save_video
#
set -euo pipefail

VIDEO="videos/color_car.mp4"
SECONDS_TO_RUN=30
PLAY_ONCE=0
OUT_DIR="/output"
# Publish rate. Also handed to ByteTrack as frame_rate, and to the video
# writer. "auto" reads the rate out of the video file itself.
FPS=30
TRACK_BUFFER=""
LOCKSTEP=""
DEPTH="--depth"
CAMERA_RPY="[0.0,-90.0,0.0]"
SAVE_VIDEO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)    VIDEO="$2"; shift 2 ;;
        --seconds)  SECONDS_TO_RUN="$2"; shift 2 ;;
        --out)      OUT_DIR="$2"; shift 2 ;;
        --fps)      FPS="$2"; shift 2 ;;
        --track-buffer) TRACK_BUFFER="$2"; shift 2 ;;
        --full)     PLAY_ONCE=1; shift ;;
        --lockstep) LOCKSTEP="--lockstep"; shift ;;
        --no-depth) DEPTH=""; shift ;;
        --level)    CAMERA_RPY="[0.0,0.0,0.0]"; shift ;;
        --save_video) SAVE_VIDEO=1; shift ;;
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

# Read the video's own frame rate so we can resolve --fps auto and warn when
# the publish rate does not match what the file was recorded at.
read -r SRC_FRAMES SRC_FPS <<<"$(python3 -c "
import cv2
c = cv2.VideoCapture('$VIDEO')
n, f = int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS)
c.release()
print(n, round(f, 3) if f and f > 0 else 0)" 2>/dev/null || echo "0 0")"

if [ "$FPS" = "auto" ]; then
    if [ "${SRC_FPS%%.*}" -gt 0 ] 2>/dev/null; then
        FPS="${SRC_FPS%%.*}"
        echo "--fps auto: using the file's own rate, ${FPS} fps"
    else
        FPS=30
        echo "--fps auto: could not read a rate from $VIDEO, falling back to 30"
    fi
fi

# Another node on the same ROS_DOMAIN_ID would publish to the same topic and
# the recorder would mix them together. Containers sharing --ipc=host find
# each other over shared memory even without a shared network.
OTHER=$(ros2 node list 2>/dev/null | grep -c groundingdino_node || true)
if [ "${OTHER:-0}" -gt 0 ]; then
    echo ""
    echo "ERROR: ${OTHER} groundingdino_node(s) already running on"
    echo "       ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}. Their output would be"
    echo "       recorded alongside this run and the results would be wrong."
    echo "       Stop the other containers (docker ps / docker rm -f), or"
    echo "       give this one a different ROS_DOMAIN_ID."
    exit 1
fi

echo "========================================"
echo "GroundingDINO -> Perception"
echo "========================================"
echo "video:    $VIDEO"
echo "source:   ${SRC_FRAMES} frames, recorded at ${SRC_FPS} fps"
if [ -n "$LOCKSTEP" ]; then
    echo "publish:  lockstep, timestamps ${FPS} fps apart (no frames dropped)"
else
    echo "publish:  ${FPS} fps wall clock (frames the node misses are dropped)"
fi
if [ "$PLAY_ONCE" -eq 1 ]; then
    echo "duration: whole video once"
else
    echo "duration: ${SECONDS_TO_RUN}s (looping)"
fi
echo "output:   $OUT_DIR"
echo "depth:    ${DEPTH:-off (ground-plane fallback)}"
if [ "$SAVE_VIDEO" -eq 1 ]; then
    echo "annotated: $OUT_DIR/tracked.avi"
else
    echo "annotated: off (pass --save_video to write $OUT_DIR/tracked.avi)"
fi
echo ""

PIDS=()

# `ros2 run` is a wrapper: it forks the node and waits.  $! is the wrapper, so
# signalling only the recorded pid leaves the node itself alive -- reparented to
# pid 1, still on the topic, and the next run then aborts on the cross-talk
# guard.  Signal the whole descendant tree instead, deepest first.
kill_tree() {
    local pid=$1 sig=$2 child
    for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
    kill "-$sig" "$pid" 2>/dev/null || true
}
tree_alive() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && return 0
        pgrep -P "$pid" >/dev/null 2>&1 && return 0
    done
    return 1
}

cleanup() {
    echo ""
    echo "--- stopping ---"
    # SIGINT so the writers flush and close their files. The detection node
    # takes a few seconds to tear down its CUDA context; without the grace
    # period it gets SIGKILLed and bash reports a "Killed" job.
    for pid in "${PIDS[@]}"; do kill_tree "$pid" INT; done
    for _ in $(seq 1 15); do
        tree_alive || break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do kill_tree "$pid" KILL; done
    wait 2>/dev/null || true

    # Belt and braces: anything of ours that got reparented away from this
    # shell, so `pgrep -P` can no longer see it from the recorded pids.
    pkill -9 -u "$(id -u)" -f 'groundingdino_ros/groundingdino_node' 2>/dev/null || true
    pkill -9 -u "$(id -u)" -f 'ros2_package/(video_saver|perception_recorder|sim_stub_publisher)\.py' 2>/dev/null || true
}
trap cleanup EXIT

# ByteTrack drops a track after int(frame_rate/30 * track_buffer) missed
# frames, so the tracker has to be told the real frame rate or tracks expire
# in a fraction of a second at high fps and ids churn.
TRACK_ARGS=(-p "frame_rate:=${FPS}")
if [ -n "$TRACK_BUFFER" ]; then
    TRACK_ARGS+=(-p "track_buffer:=${TRACK_BUFFER}")
fi

# Without --save_video nothing subscribes to the annotated image topic, so stop
# the node drawing and publishing frames for no reader.  Detection, tracking and
# the PerceptionArray output are unaffected -- this gates rendering only.
if [ "$SAVE_VIDEO" -eq 1 ]; then
    TRACK_ARGS+=(-p "output_visualization:=true")
else
    TRACK_ARGS+=(-p "output_visualization:=false")
fi

# Steps are numbered for the operator; the video saver is one of them only when
# it actually runs.
N_STEPS=$((3 + SAVE_VIDEO))
STEP=0
step() { STEP=$((STEP + 1)); echo "[${STEP}/${N_STEPS}] $*"; }
LOST_FRAMES=$(python3 -c "
buf = ${TRACK_BUFFER:-30}
print(int(${FPS} / 30.0 * buf))" 2>/dev/null || echo "?")
echo "      tracker: frame_rate=${FPS}, track_buffer=${TRACK_BUFFER:-30}"\
     "-> a track survives ~${LOST_FRAMES} missed frames"

step "starting detection + tracking node (loads the model, ~10s)"
ros2 run groundingdino_ros groundingdino_node --ros-args \
    -p model_weights:=/weights/groundingdino_swinb_cogcoor.pth \
    -p camera_rpy_deg:="${CAMERA_RPY}" \
    -p box_threshold:=0.30 \
    -p text_threshold:=0.25 \
    "${TRACK_ARGS[@]}" \
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

step "starting perception recorder -> $OUT_DIR/perceptions.{jsonl,csv}"
python3 ros2_package/perception_recorder.py --output "$OUT_DIR/perceptions" \
    > "$OUT_DIR/recorder.log" 2>&1 &
PIDS+=($!)

if [ "$SAVE_VIDEO" -eq 1 ]; then
    step "starting video saver -> $OUT_DIR/tracked.avi"
    python3 ros2_package/video_saver.py --output "$OUT_DIR/tracked.avi" --fps "$FPS" \
        > "$OUT_DIR/saver.log" 2>&1 &
    PIDS+=($!)
fi

LOOP="--loop"
[ "$PLAY_ONCE" -eq 1 ] && LOOP=""

step "starting AirSim stub publisher ($VIDEO)"
python3 ros2_package/sim_stub_publisher.py \
    --video "$VIDEO" --fps "$FPS" ${LOOP} ${DEPTH} ${LOCKSTEP} \
    > "$OUT_DIR/stub.log" 2>&1 &
STUB_PID=$!
PIDS+=($STUB_PID)

echo ""
if [ "$PLAY_ONCE" -eq 1 ]; then
    EST=$(python3 -c "print(f'{$SRC_FRAMES/$FPS:.0f}')" 2>/dev/null || echo "?")
    echo "playing all ${SRC_FRAMES} frames at ${FPS} fps -> ~${EST}s if the"
    echo "pipeline keeps up; longer if it cannot, which is harmless because"
    echo "the stub stamps frames from a synthetic clock, not wall clock."
    # The stub exits on its own once the video runs out.
    wait "$STUB_PID" || true
    echo "stub finished"
else
    echo "running for ${SECONDS_TO_RUN}s ..."
    sleep "$SECONDS_TO_RUN"
fi

cleanup
trap - EXIT

# The container now runs as a non-root user whose uid matches the host's, so
# output already lands owned by the caller and this is a no-op.  Kept for images
# built before that change, and for `docker exec -u root`, where it still does
# the work.  Skipped when we are not root -- chown would only fail.
if [ "$(id -u)" = "0" ] && [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
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
echo "--- frames: published vs processed ---"
PUBLISHED=$(grep -oP 'published \K[0-9]+' "$OUT_DIR/stub.log" | tail -1 || echo "?")
PROCESSED=$(wc -l < "$OUT_DIR/perceptions.jsonl" 2>/dev/null || echo 0)
echo "published=${PUBLISHED} processed=${PROCESSED}"
if [ "$PUBLISHED" != "?" ] && [ "${PROCESSED:-0}" -gt 0 ]; then
    python3 -c "
pub, proc = $PUBLISHED, $PROCESSED
if proc < pub * 0.95:
    print(f'  {100*(pub-proc)/pub:.0f}% of frames were dropped: the node could'
          f' not keep up at this publish rate.')
    print('  Re-run with --lockstep to process every frame.')
else:
    print('  every frame processed')" 2>/dev/null || true
fi

echo ""
echo "--- distinct track ids ---"
awk -F, 'NR>1{print $3}' "$OUT_DIR/perceptions.csv" 2>/dev/null | sort -un | wc -l

echo ""
echo "--- tracks matched to an entity of interest ---"
awk -F, 'NR>1 && $4 != "" {print $4}' "$OUT_DIR/perceptions.csv" 2>/dev/null \
    | sort | uniq -c || true
