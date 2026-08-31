#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MSGS_SRC="$PROJECT_ROOT/ros2_package/msgs"
MSGS_STAGE="$PROJECT_ROOT/_msgs_build"

echo "========================================"
echo "Building GroundingDINO ROS2 Docker Image"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Docker context: $PROJECT_ROOT"
echo ""

# groundingdino_ros publishes msgs/msg/PerceptionArray.  ROS 2 matches a
# publisher to a subscriber by the fully-qualified type name and the message
# definition, so the image has to contain a package named msgs with these
# fields -- and if a subscriber was built against different definitions, the
# two do not connect, SILENTLY: no error, no warning, no data.
#
# The definitions live at ros2_package/msgs.  They are staged into the Docker
# context here because the build needs them inside it.  See that README.

rm -rf "$MSGS_STAGE"
mkdir -p "$MSGS_STAGE"

if [ ! -d "$MSGS_SRC/msg" ]; then
    echo "ERROR: msgs package missing at $MSGS_SRC"
    echo "       It is checked into this repo; restore it with:"
    echo "         git checkout -- ros2_package/msgs"
    exit 1
fi

echo "Staging msgs from ros2_package/msgs ..."
cp -r "$MSGS_SRC/." "$MSGS_STAGE/"
rm -f "$MSGS_STAGE/README.md"

echo "  message files staged:"
ls "$MSGS_STAGE/msg" | sed 's/^/    /'

cleanup() {
    rm -rf "$MSGS_STAGE"
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
echo "  docker run --rm groundingdino_ros:latest python3 -c \"from msgs.msg import DetectionArray, PerceptionArray; print('OK')\""
echo ""
echo "Full smoke test (needs GPU + weights volume):"
echo "  docker run --rm --gpus all -v /path/to/weights:/weights:ro groundingdino_ros:latest ros2 pkg list | grep -E 'groundingdino|msgs'"
echo ""
echo "To run with docker-compose:"
echo "  cd ../airsim/release_installer"
echo "  docker-compose -f docker/docker-compose.yml up groundingdino"
echo ""
