#!/usr/bin/env python3
"""
Minimal AirSim stand-in for checking the Perception output.

Publishes everything groundingdino_node needs to place a track in the
world, so the output can be inspected without the simulator running:

    scene camera image   from a video file
    depth image          synthetic constant-depth plane (optional)
    camera_info          intrinsics from a horizontal FOV
    drone pose           a fixed NED pose, nadir-looking by default

Image stamps come from a synthetic clock that starts at --start-time and
advances one frame period per frame, standing in for the sim clock. That
is the value the node copies into PerceptionArray.stamp, so a wall-clock
stamp is visibly distinguishable from a sim-clock one in the output.

Usage:
    python3 sim_stub_publisher.py --video videos/carla1.mp4 --fps 10
"""

import argparse
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from msgs.msg import PerceptionArray

PREFIX = "/viaduct/Sim/SceneDroneSensors/robots/Drone1"
SCENE_TOPIC = f"{PREFIX}/sensors/front_center1/scene_camera/image"
DEPTH_TOPIC = f"{PREFIX}/sensors/front_center1/depth_planar_camera/image"
INFO_TOPIC = f"{PREFIX}/sensors/front_center1/depth_planar_camera/camera_info"
POSE_TOPIC = f"{PREFIX}/actual_pose"
PERCEPTION_TOPIC = "/vanderbilt/fake_perception/data"

QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class SimStubPublisher(Node):

    def __init__(self, args):
        super().__init__("sim_stub_publisher")

        self.bridge = CvBridge()
        self.args = args

        self.cap = cv2.VideoCapture(args.video)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fx = (self.width / 2.0) / math.tan(math.radians(args.fov) / 2.0)

        self.get_logger().info(
            f"Video {self.width}x{self.height}, fov={args.fov} deg -> fx={self.fx:.1f}")
        self.get_logger().info(
            f"Drone at NED ({args.north}, {args.east}, {args.down}), "
            f"depth {'on' if args.depth else 'off'}")

        self.image_pub = self.create_publisher(Image, SCENE_TOPIC, QOS)
        self.info_pub = self.create_publisher(CameraInfo, INFO_TOPIC, QOS)
        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, QOS)
        self.depth_pub = self.create_publisher(Image, DEPTH_TOPIC, QOS) \
            if args.depth else None

        # Spacing between frame timestamps. In lockstep this is only the
        # sim-clock step -- wall-clock pacing comes from the node instead.
        self.sim_step = 1.0 / args.fps
        self.sim_time = float(args.start_time)
        self.frame_count = 0
        self.done = False

        # Lockstep: publish the next frame only once the node has answered
        # for the previous one. Without it the node's depth-1 image queue
        # drops every frame it cannot keep up with -- on a 2560x1400 video
        # that is most of them, and tracks break across the gaps.
        self.lockstep = args.lockstep
        self.waiting_since = None
        if self.lockstep:
            self.create_subscription(
                PerceptionArray, PERCEPTION_TOPIC, self.on_perception, QOS)
            self.get_logger().info(
                "lockstep: pacing off the node, every frame gets processed")
            self.timer = self.create_timer(0.005, self.tick)
        else:
            self.timer = self.create_timer(self.sim_step, self.tick)

    def stamp(self) -> Time:
        """Synthetic sim-clock stamp for the current frame."""
        msg = Time()
        msg.sec = int(math.floor(self.sim_time))
        msg.nanosec = int(round((self.sim_time - msg.sec) * 1e9))
        return msg

    def on_perception(self, msg):
        """The node answered for the frame in flight; release the next."""
        self.waiting_since = None

    def tick(self):
        if self.lockstep and self.waiting_since is not None:
            # Safety valve: never deadlock if a frame produces no output.
            if (self.get_clock().now() - self.waiting_since).nanoseconds < 3e9:
                return
            self.get_logger().warn("no response for a frame, moving on",
                                   throttle_duration_sec=10.0)
            self.waiting_since = None

        ok, frame = self.cap.read()
        if not ok:
            if not self.args.loop:
                self.timer.cancel()
                self.get_logger().info(
                    f"Video finished: published {self.frame_count} frames, "
                    f"sim t={self.sim_time:.1f}s. Shutting down.")
                # Let the consumers drain the last frames before we go.
                self.create_timer(self.args.drain, self._finish)
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return

        stamp = self.stamp()

        # Pose first: the node needs it before the frame it projects.
        self.pose_pub.publish(self.make_pose(stamp))
        self.info_pub.publish(self.make_info(stamp))
        if self.depth_pub is not None:
            self.depth_pub.publish(self.make_depth(stamp))

        image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image.header.stamp = stamp
        image.header.frame_id = "front_center1"
        self.image_pub.publish(image)

        if self.lockstep:
            self.waiting_since = self.get_clock().now()

        self.frame_count += 1
        self.sim_time += self.sim_step
        if self.frame_count % 20 == 0:
            self.get_logger().info(
                f"frame {self.frame_count}, sim t={self.sim_time:.1f}s")

    def _finish(self):
        """Signal main()'s spin loop to stop.

        Calling rclpy.try_shutdown() from inside a callback does not
        reliably break out of rclpy.spin(), so main() drives spin_once and
        watches this flag instead.
        """
        self.done = True

    def make_pose(self, stamp) -> PoseStamped:
        """Fixed NED pose. Yaw only -- camera pitch is a node parameter."""

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world_ned"
        msg.pose.position.x = float(self.args.north)
        msg.pose.position.y = float(self.args.east)
        msg.pose.position.z = float(self.args.down)

        half = math.radians(self.args.yaw_deg) / 2.0
        msg.pose.orientation.w = math.cos(half)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(half)
        return msg

    def make_info(self, stamp) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "front_center1"
        msg.width = self.width
        msg.height = self.height
        msg.k = [self.fx, 0.0, self.width / 2.0,
                 0.0, self.fx, self.height / 2.0,
                 0.0, 0.0, 1.0]
        return msg

    def make_depth(self, stamp) -> Image:
        """Constant planar depth: a flat surface at --depth-metres."""

        depth = np.full((self.height, self.width),
                        float(self.args.depth_metres), dtype=np.float32)
        msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = "front_center1"
        return msg

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--fov", type=float, default=120.0,
                        help="horizontal FOV in degrees")
    parser.add_argument("--loop", action="store_true",
                        help="restart at the end instead of exiting")
    parser.add_argument("--lockstep", action="store_true",
                        help="wait for the node before sending each frame, so "
                             "no frame is dropped (slower than real time)")
    parser.add_argument("--drain", type=float, default=5.0,
                        help="seconds to keep spinning after the last frame, "
                             "so consumers finish writing")
    parser.add_argument("--start-time", type=float, default=1000.0,
                        help="synthetic sim clock start, seconds")
    parser.add_argument("--north", type=float, default=-213.0)
    parser.add_argument("--east", type=float, default=-25.0)
    parser.add_argument("--down", type=float, default=-40.0,
                        help="NED down: -40 means 40 m altitude")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--depth", action="store_true",
                        help="publish a synthetic depth plane")
    parser.add_argument("--depth-metres", type=float, default=40.0)
    args = parser.parse_args()

    rclpy.init()
    node = SimStubPublisher(args)
    try:
        # spin_once rather than spin, so the end-of-video flag can stop us
        # and the process exits for a caller waiting on it.
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
