#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRINITY_MSGS_SRC="$(dirname "$PROJECT_ROOT")/trinity_msgs"
TRINITY_MSGS_STAGE="$PROJECT_ROOT/_trinity_msgs_build"

echo "========================================"
echo "Building GroundingDINO ROS2 Docker Image"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Docker context: $PROJECT_ROOT"
echo ""

# Which trinity_msgs revision to build against.
#
# The running architecture_demo stack pins its trinity_msgs submodule at
# 31287a1 ("message definitions for perception 0.22").  ROS 2 refuses to connect
# a publisher and subscriber whose type definitions differ, and it does so
# SILENTLY -- no error, no data.  So we build the same revision the stack runs,
# not the repo's HEAD (0.58), which has extra fields and would not connect.
#
# Override to build a different revision, e.g. to test against 0.58 head:
#   TRINITY_MSGS_REF=master ./docker/build_ros2.sh
# Use TRINITY_MSGS_REF=WORKTREE to copy the working tree as-is instead.
TRINITY_MSGS_REF="${TRINITY_MSGS_REF:-31287a1}"

# Stage trinity_msgs into the build context (it lives outside GroundingDINO/)
if [ ! -d "$TRINITY_MSGS_SRC" ]; then
    echo "ERROR: trinity_msgs not found at $TRINITY_MSGS_SRC"
    exit 1
fi

mkdir -p "$TRINITY_MSGS_STAGE"
if [ "$TRINITY_MSGS_REF" = "WORKTREE" ]; then
    echo "Staging trinity_msgs working tree from $TRINITY_MSGS_SRC ..."
    cp -r "$TRINITY_MSGS_SRC/." "$TRINITY_MSGS_STAGE/"
else
    echo "Staging trinity_msgs @ $TRINITY_MSGS_REF from $TRINITY_MSGS_SRC ..."
    git -C "$TRINITY_MSGS_SRC" archive "$TRINITY_MSGS_REF" \
        | tar -x -C "$TRINITY_MSGS_STAGE"
fi
echo "  message files staged:"
ls "$TRINITY_MSGS_STAGE/msg" | sed 's/^/    /'

cleanup() {
    rm -rf "$TRINITY_MSGS_STAGE"
}
trap cleanup EXIT

# Build the Docker image
docker build \
    -f "$SCRIPT_DIR/Dockerfile.ros2" \
    -t groundingdino_ros:latest \
    "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo "Image: groundingdino_ros:latest"
echo ""
echo "Quick test (no GPU):"
echo "  docker run --rm groundingdino_ros:latest python3 -c \"from trinity_msgs.msg import DetectionArray, PerceptionArray; print('OK')\""
echo ""
echo "Full smoke test (needs GPU + weights volume):"
echo "  docker run --rm --gpus all -v /path/to/weights:/weights:ro groundingdino_ros:latest ros2 pkg list | grep -E 'groundingdino|trinity'"
echo ""
echo "To run with docker-compose:"
echo "  cd ../airsim/release_installer"
echo "  docker-compose -f docker/docker-compose.yml up groundingdino"
echo ""
