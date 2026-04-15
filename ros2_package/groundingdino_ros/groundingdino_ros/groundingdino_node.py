#!/usr/bin/env python3
"""
GroundingDINO ROS2 Node - Wraps Worker class for detection and tracking.

This node subscribes to camera images, runs GroundingDINO detection with ByteTrack/CLIP tracking,
and publishes tracking results as custom ROS2 messages.
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# Add GroundingDINO root to path
GROUNDINGDINO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(GROUNDINGDINO_ROOT))

from worker_simple import Worker
from groundingdino_ros.msg import GroundingDINOTrack, GroundingDINOTrackArray

# Import mission parser (same directory when installed)
try:
    from groundingdino_ros.mission_parser import get_text_prompt_from_mission
except ModuleNotFoundError:
    from mission_parser import get_text_prompt_from_mission


class GroundingDINONode(Node):
    """ROS2 node for GroundingDINO detection and tracking."""

    def __init__(self):
        super().__init__('groundingdino_node')

        # Declare parameters
        self._declare_parameters()

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load text prompt from mission config
        self.text_prompt = self._load_text_prompt()
        self.get_logger().info(f"Text prompt: '{self.text_prompt}'")

        # Initialize Worker
        self.worker = self._init_worker()
        self.get_logger().info(
            f"Worker initialized with tracker: {self.worker.tracker_type}"
        )

        # Subscribe to camera images
        camera_topic = self.get_parameter('camera_topic').value
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to: {camera_topic}")

        # Publishers
        self.tracks_pub = self.create_publisher(
            GroundingDINOTrackArray,
            '/groundingdino/tracks',
            10
        )

        self.viz_pub = self.create_publisher(
            Image,
            '/groundingdino/visualization',
            10
        )

        self.debug_pub = self.create_publisher(
            String,
            '/groundingdino/debug',
            10
        )

        # State
        self.frame_id = 0
        self.last_prompt_reload = self.get_clock().now()

        # Mission config reload timer (check every 5 seconds)
        if self.get_parameter('use_mission_classes').value:
            self.config_timer = self.create_timer(
                5.0,
                self.reload_mission_config_callback
            )

        self.get_logger().info("GroundingDINO node initialized and ready")

    def _declare_parameters(self):
        """Declare all ROS2 parameters."""

        # Model configuration
        self.declare_parameter('model_config', '/app/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py')
        self.declare_parameter('model_weights', '/weights/groundingdino_swinb_cogcoor.pth')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('use_fp16', True)

        # Detection thresholds
        self.declare_parameter('box_threshold', 0.42)
        self.declare_parameter('text_threshold', 0.50)

        # Tracker configuration (ByteTrack only)
        self.declare_parameter('tracker_type', 'bytetrack')
        self.declare_parameter('track_thresh', 0.5)
        self.declare_parameter('track_buffer', 30)
        self.declare_parameter('match_thresh', 0.8)

        # Input/output
        self.declare_parameter('camera_topic', '/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image')
        self.declare_parameter('mission_config_path', '/mission_briefing/config.json')
        self.declare_parameter('output_visualization', True)

        # Text prompts
        self.declare_parameter('default_classes', ['car', 'pedestrian'])
        self.declare_parameter('use_mission_classes', True)

        # Performance
        self.declare_parameter('frame_rate', 10)

    def _load_text_prompt(self) -> str:
        """Load text prompt from mission config or use defaults."""

        if self.get_parameter('use_mission_classes').value:
            config_path = self.get_parameter('mission_config_path').value
            default_classes = self.get_parameter('default_classes').value

            prompt = get_text_prompt_from_mission(config_path, default_classes)
            self.get_logger().info(f"Loaded text prompt from mission config")
        else:
            default_classes = self.get_parameter('default_classes').value
            prompt = ". ".join(default_classes) + "."
            self.get_logger().info(f"Using default text prompt")

        return prompt

    def _init_worker(self) -> Worker:
        """Initialize Worker instance with ROS2 parameters."""

        tracker_kwargs = {
            'track_thresh': self.get_parameter('track_thresh').value,
            'track_buffer': self.get_parameter('track_buffer').value,
            'match_thresh': self.get_parameter('match_thresh').value,
        }

        worker = Worker(
            config_path=self.get_parameter('model_config').value,
            weights_path=self.get_parameter('model_weights').value,
            text_prompt=self.text_prompt,
            box_thresh=self.get_parameter('box_threshold').value,
            text_thresh=self.get_parameter('text_threshold').value,
            use_fp16=self.get_parameter('use_fp16').value,
            device=self.get_parameter('device').value,
            tracker_kwargs=tracker_kwargs,
            frame_rate=self.get_parameter('frame_rate').value,
        )

        return worker

    def reload_mission_config_callback(self):
        """Periodically reload mission config to pick up changes."""

        # Only reload every 5 seconds minimum
        now = self.get_clock().now()
        if (now - self.last_prompt_reload).nanoseconds < 5e9:
            return

        new_prompt = self._load_text_prompt()

        if new_prompt != self.text_prompt:
            self.get_logger().info(
                f"Text prompt changed: '{self.text_prompt}' -> '{new_prompt}'"
            )
            self.text_prompt = new_prompt
            self.worker.text_prompt = new_prompt

        self.last_prompt_reload = now

    def image_callback(self, msg: Image):
        """Process incoming camera images."""

        try:
            # Convert ROS image to OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            orig_h, orig_w = frame.shape[:2]

            # Preprocess frame (Worker handles resizing to 800px)
            tensor = self.worker.preprocess_frame(frame)

            # Run detection
            dets_xyxy = self.worker.predict_detections(
                frame_bgr=frame,
                tensor_image=tensor,
                orig_h=orig_h,
                orig_w=orig_w
            )

            # Run tracking (ByteTrack via worker_simple)
            tracks = self.worker.update_tracker(dets_xyxy, orig_h, orig_w)

            # Publish tracking results
            self._publish_tracks(tracks, msg.header, orig_h, orig_w)

            # Optionally publish visualization
            if self.get_parameter('output_visualization').value:
                self._publish_visualization(frame, tracks, msg.header)

            # Publish debug info
            debug_msg = String()
            debug_msg.data = f"Frame {self.frame_id}: {len(dets_xyxy)} detections, {len(tracks)} tracks"
            self.debug_pub.publish(debug_msg)

            self.frame_id += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}", throttle_duration_sec=1.0)

    def _publish_tracks(self, tracks, header, img_h: int, img_w: int):
        """Publish tracking results as ROS2 message."""

        track_array_msg = GroundingDINOTrackArray()
        track_array_msg.header = header
        track_array_msg.frame_id = self.frame_id
        track_array_msg.text_prompt = self.text_prompt
        track_array_msg.tracker_type = self.worker.tracker_type

        for track in tracks:
            if not track.is_activated:
                continue

            track_msg = GroundingDINOTrack()
            track_msg.header = header

            # Track info
            track_msg.track_id = int(track.track_id)
            track_msg.frame_id = self.frame_id
            track_msg.class_name = "object"  # TODO: extract from phrases if available
            track_msg.confidence = float(track.score)

            # Bounding box in pixels (tlwh format)
            tlwh = track.tlwh
            track_msg.x = float(tlwh[0])
            track_msg.y = float(tlwh[1])
            track_msg.width = float(tlwh[2])
            track_msg.height = float(tlwh[3])

            # Bounding box in normalized coordinates
            track_msg.cx_norm = float((tlwh[0] + tlwh[2] / 2) / img_w)
            track_msg.cy_norm = float((tlwh[1] + tlwh[3] / 2) / img_h)
            track_msg.width_norm = float(tlwh[2] / img_w)
            track_msg.height_norm = float(tlwh[3] / img_h)

            # Track status
            track_msg.is_activated = bool(track.is_activated)
            track_msg.tracklet_len = int(track.tracklet_len)

            track_array_msg.tracks.append(track_msg)

        track_array_msg.total_tracks = len(track_array_msg.tracks)
        self.tracks_pub.publish(track_array_msg)

    def _publish_visualization(self, frame, tracks, header):
        """Publish annotated image with bounding boxes."""

        vis_frame = frame.copy()

        # Color palette for different track IDs
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

        for track in tracks:
            if not track.is_activated:
                continue

            # Get bounding box
            tlwh = track.tlwh
            x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
            x2, y2 = x1 + w, y1 + h

            # Get color for this track ID
            track_id = track.track_id
            color = tuple(map(int, colors[track_id % len(colors)]))

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{track_id} ({track.score:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Label background
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Add frame info
        info_text = f"Frame: {self.frame_id} | Tracks: {len(tracks)} | Tracker: {self.worker.tracker_type}"
        cv2.putText(
            vis_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Convert back to ROS message and publish
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(vis_frame, encoding='bgr8')
            viz_msg.header = header
            self.viz_pub.publish(viz_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing visualization: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = GroundingDINONode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
