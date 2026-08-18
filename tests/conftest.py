"""Put eval/ and the ROS2 package on sys.path so tests import their modules
the same way the eval scripts and the ROS2 node do
(``from query_parser import parse``, ``from geometry import ...``)."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EVAL_DIR = os.path.join(_ROOT, "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# groundingdino_ros modules are importable bare, as they are when colcon
# installs them flat into lib/groundingdino_ros/.
_ROS_PKG_DIR = os.path.join(
    _ROOT, "ros2_package", "groundingdino_ros", "groundingdino_ros")
if _ROS_PKG_DIR not in sys.path:
    sys.path.insert(0, _ROS_PKG_DIR)
