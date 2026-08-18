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

# Stage trinity_msgs into the build context (it lives outside GroundingDINO/)
if [ -d "$TRINITY_MSGS_SRC" ]; then
    echo "Staging trinity_msgs from $TRINITY_MSGS_SRC ..."
    cp -r "$TRINITY_MSGS_SRC" "$TRINITY_MSGS_STAGE"
else
    echo "ERROR: trinity_msgs not found at $TRINITY_MSGS_SRC"
    exit 1
fi

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
