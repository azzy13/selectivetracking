#!/bin/bash
set -e

echo "========================================"
echo "GroundingDINO ROS2 Container Starting"
echo "========================================"

# Source ROS2 Humble
echo "Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f "$ROS2_WS/install/setup.bash" ]; then
    echo "Sourcing ROS2 workspace..."
    source $ROS2_WS/install/setup.bash
else
    echo "Warning: ROS2 workspace not built yet"
fi

# Add GroundingDINO to Python path
export PYTHONPATH=$GROUNDINGDINO_PATH:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Print environment info
echo "----------------------------------------"
echo "Environment Information:"
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "GROUNDINGDINO_PATH: $GROUNDINGDINO_PATH"
echo "ROS2_WS: $ROS2_WS"
echo "Running as: $(id -un 2>/dev/null || echo uid=$(id -u)) (uid=$(id -u) gid=$(id -g))"
echo "----------------------------------------"

# The container is non-root so that results land on the host owned by you and
# not by root.  The flip side: if /output is bind-mounted from a host directory
# owned by someone else, we cannot write to it.  Say so now, with the fix,
# rather than letting the node die on its first write half a minute in.
for d in /output "$HOME"; do
    [ -d "$d" ] || continue
    if [ ! -w "$d" ]; then
        owner=$(stat -c '%u:%g' "$d" 2>/dev/null || echo "?")
        echo "ERROR: $d is not writable by uid=$(id -u) gid=$(id -g) (it is owned by $owner)"
        echo ""
        echo "  The host directory mounted there belongs to a different user."
        echo "  Either chown it on the host:"
        echo "      sudo chown -R \$(id -u):\$(id -g) /path/to/that/dir"
        echo "  or run the container as its owner:"
        echo "      docker run --user ${owner%%:*}:${owner##*:} ..."
        exit 1
    fi
done
echo "----------------------------------------"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Devices:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "----------------------------------------"
else
    echo "Warning: CUDA not available"
    echo "----------------------------------------"
fi

echo "Starting GroundingDINO node..."
echo "========================================"
echo ""

# Execute command
exec "$@"
