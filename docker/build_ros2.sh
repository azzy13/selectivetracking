#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRINITY_MSGS_VENDORED="$PROJECT_ROOT/ros2_package/trinity_msgs"
TRINITY_MSGS_STAGE="$PROJECT_ROOT/_trinity_msgs_build"

echo "========================================"
echo "Building GroundingDINO ROS2 Docker Image"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Docker context: $PROJECT_ROOT"
echo ""

# groundingdino_ros must use the same trinity_msgs definitions as the rest of
# the ROS stack. Use the vendored definitions by default; set TRINITY_MSGS_REF
# and TRINITY_MSGS_SRC to build against another checkout.
#
# Examples:
#   TRINITY_MSGS_REF=master TRINITY_MSGS_SRC=/path/to/trinity_msgs ./docker/build_ros2.sh
#   TRINITY_MSGS_REF=WORKTREE TRINITY_MSGS_SRC=/path/to/trinity_msgs ./docker/build_ros2.sh
TRINITY_MSGS_REF="${TRINITY_MSGS_REF:-VENDORED}"

rm -rf "$TRINITY_MSGS_STAGE"
mkdir -p "$TRINITY_MSGS_STAGE"

if [ "$TRINITY_MSGS_REF" = "VENDORED" ]; then
    if [ ! -d "$TRINITY_MSGS_VENDORED/msg" ]; then
        echo "ERROR: vendored trinity_msgs missing at $TRINITY_MSGS_VENDORED"
        echo "       It is checked into this repo; restore it with:"
        echo "         git checkout -- ros2_package/trinity_msgs"
        exit 1
    fi
    echo "Staging vendored trinity_msgs (perception 0.22) from ros2_package/trinity_msgs ..."
    cp -r "$TRINITY_MSGS_VENDORED/." "$TRINITY_MSGS_STAGE/"
    rm -f "$TRINITY_MSGS_STAGE/README.md"
else

    # Non-vendored revisions require a local trinity_msgs checkout.
    TRINITY_MSGS_SRC="${TRINITY_MSGS_SRC:-$(dirname "$PROJECT_ROOT")/trinity_msgs}"
    if [ ! -d "$TRINITY_MSGS_SRC" ]; then
        echo "ERROR: TRINITY_MSGS_REF=$TRINITY_MSGS_REF needs a trinity_msgs checkout,"
        echo "       and none is at $TRINITY_MSGS_SRC"
        echo "       Set TRINITY_MSGS_SRC=/path/to/trinity_msgs, or unset"
        echo "       TRINITY_MSGS_REF to build the vendored 0.22 definitions."
        exit 1
    fi
    if [ "$TRINITY_MSGS_REF" = "WORKTREE" ]; then
        echo "Staging trinity_msgs working tree from $TRINITY_MSGS_SRC ..."
        cp -r "$TRINITY_MSGS_SRC/." "$TRINITY_MSGS_STAGE/"
        rm -rf "$TRINITY_MSGS_STAGE/.git"
    else
        echo "Staging trinity_msgs @ $TRINITY_MSGS_REF from $TRINITY_MSGS_SRC ..."
        git -C "$TRINITY_MSGS_SRC" archive "$TRINITY_MSGS_REF" \
            | tar -x -C "$TRINITY_MSGS_STAGE"
    fi
fi
echo "  message files staged:"
ls "$TRINITY_MSGS_STAGE/msg" | sed 's/^/    /'

cleanup() {
    rm -rf "$TRINITY_MSGS_STAGE"
}
trap cleanup EXIT

# Build the image from the project root so the staged messages are in context.
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
