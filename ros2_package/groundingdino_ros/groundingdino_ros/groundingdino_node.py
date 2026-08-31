#!/usr/bin/env python3
"""
GroundingDINO ROS2 Node - Wraps Worker class for detection and tracking.

This node subscribes to camera images, runs GroundingDINO detection with ByteTrack/CLIP tracking,
and publishes tracking results.
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import numpy as np

# Add GroundingDINO root to path
GROUNDINGDINO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(GROUNDINGDINO_ROOT))

from worker_simple import Worker
from scene_graph import SceneGraphBuilder, SceneGraphMissionFilter
from msgs.msg import Detection, DetectionArray, Perception, PerceptionArray

# Import mission parser (same directory when installed)
try:
    from groundingdino_ros.mission_parser import (
        get_text_prompt_from_mission, parse_entities_of_interest)
    from groundingdino_ros.geometry import GroundProjector
    from groundingdino_ros import perception_publisher as pp
except ModuleNotFoundError:
    from mission_parser import (
        get_text_prompt_from_mission, parse_entities_of_interest)
    from geometry import GroundProjector
    import perception_publisher as pp


# QoS the rest of the stack uses for perception topics: fake_perception_node
# publishes with it and prediction_node subscribes with it.
PERCEPTION_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def _set_if_present(msg, field: str, value) -> bool:
    """Assign msg.field only if this msgs build actually defines it.

    Some builds of msgs carry optional extra fields (frame_number,
    occlusion, pose) that this repo's definitions do not.  rosidl message
    classes use __slots__, so assigning a field the built package does not
    define raises AttributeError and kills the frame.  This keeps one node
    source working against either field set.
    """
    if hasattr(msg, field):
        setattr(msg, field, value)
        return True
    return False


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

        # Publishers (shared message types for plug-and-play compatibility)
        self.detections_pub = self.create_publisher(
            DetectionArray,
            self.get_parameter('detections_topic').value,
            10
        )

        self.perceptions_pub = self.create_publisher(
            PerceptionArray,
            self.get_parameter('perceptions_topic').value,
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

        # Corrected Perception output, alongside the legacy topics
        self.perception_output_enabled = bool(
            self.get_parameter('publish_perception_output').value)
        self.perception_output_pub = None
        self.projector = None
        self.sg_builder = None
        self.entities = []
        self.mission_filters = {}
        self.depth_available = False
        self._last_projection_method = None
        if self.perception_output_enabled:
            self._init_perception_output()

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

        # Output topic names
        self.declare_parameter('detections_topic', '/perception/detections')
        self.declare_parameter('perceptions_topic', '/perception/perceptions')

        # Corrected Perception output. Published alongside the
        # legacy topics above, which are left exactly as they were.
        self.declare_parameter('publish_perception_output', True)
        self.declare_parameter('perception_output_topic',
                               '/vanderbilt/fake_perception/data')

        # Inputs needed to place a track in the world
        self.declare_parameter(
            'pose_topic',
            '/viaduct/Sim/SceneDroneSensors/robots/Drone1/actual_pose')
        self.declare_parameter(
            'depth_image_topic',
            '/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/'
            'front_center1/depth_planar_camera/image')
        self.declare_parameter(
            'depth_info_topic',
            '/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/'
            'front_center1/depth_planar_camera/camera_info')
        self.declare_parameter('use_depth', True)

        # Camera mounting on the drone body, from the mission config's
        # controllable_vehicles[].sensors[] entry.
        self.declare_parameter('camera_offset_xyz', [0.3, 0.0, -0.2])
        self.declare_parameter('camera_rpy_deg', [0.0, 0.0, 0.0])
        self.declare_parameter('camera_fov_degrees', 120.0)

        # Ground-plane fallback and sanity bound
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('max_projection_range', 200.0)

        # Minimum scene-graph mission score to call a track a match
        self.declare_parameter('mission_score_thresh', 0.10)

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
            if self.perception_output_enabled:
                self._reload_entities()

        self.last_prompt_reload = now

    def _reload_entities(self):
        """Re-read entities of interest after a mission config change."""

        config_path = self.get_parameter('mission_config_path').value
        entities = parse_entities_of_interest(config_path)
        if not entities:
            return

        known = {e.entity_id for e in self.entities}
        self.entities = entities
        thresh = float(self.get_parameter('mission_score_thresh').value)
        for entity in entities:
            if entity.entity_id in known:
                continue  # keep its accumulated colour evidence
            prompt = f"{entity.color} {entity.prompt_noun}".strip()
            self.mission_filters[entity.entity_id] = SceneGraphMissionFilter(
                text_prompt=prompt, hard_mode=False, score_thresh=thresh)
        self.get_logger().info(
            f"Entities of interest now: {[e.entity_id for e in entities]}")

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

            # Publish tracking results (legacy topics, unchanged)
            self._publish_tracks(tracks, msg.header, orig_h, orig_w)

            # Corrected Perception output
            if self.perception_output_enabled:
                self._publish_perception_output(
                    tracks, msg, orig_h, orig_w, frame)

            # Optionally publish visualization
            if self.get_parameter('output_visualization').value:
                self._publish_visualization(frame, tracks, msg.header)

            # Publish debug info
            active_tracks = sum(1 for t in tracks if t.is_activated)
            debug_msg = String()
            debug_msg.data = f"Frame {self.frame_id}: {len(dets_xyxy)} detections, {active_tracks} active tracks"
            self.debug_pub.publish(debug_msg)

            self.frame_id += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}", throttle_duration_sec=1.0)

    def _publish_tracks(self, tracks, header, img_h: int, img_w: int):
        """Publish tracking results as ROS2 messages."""

        frame_num = self.frame_id % 65536  # uint16 wrap

        det_array_msg = DetectionArray()
        det_array_msg.image_header = header
        det_array_msg.pointcloud_header = Header()

        perc_array_msg = PerceptionArray()
        perc_array_msg.stamp = self.get_clock().now().to_msg()
        perc_array_msg.frame_num = frame_num

        for track in tracks:
            if not track.is_activated:
                continue

            tlwh = track.tlwh
            x1 = int(max(0, tlwh[0]))
            y1 = int(max(0, tlwh[1]))
            x2 = int(min(img_w, tlwh[0] + tlwh[2]))
            y2 = int(min(img_h, tlwh[1] + tlwh[3]))
            cx_norm = float((tlwh[0] + tlwh[2] / 2) / img_w)
            cy_norm = float((tlwh[1] + tlwh[3] / 2) / img_h)
            score = float(track.score)
            track_id = int(track.track_id)

            det_msg = Detection()
            det_msg.frame_number = frame_num
            det_msg.detection_model = "groundingdino"
            det_msg.detection_prob = score
            det_msg.bounding_box = [x1, y1, x2, y2]
            det_msg.center_world_coords = [cx_norm, cy_norm, 0.0]
            det_msg.car_type_names = ["object"]
            det_msg.car_type_probs = [score]
            det_msg.color_names = []
            det_msg.color_probs = []
            # Optional fields; absent from this repo's definitions
            _set_if_present(det_msg, "occlusion_level", 0)
            _set_if_present(det_msg, "occlusion_label", "")
            _set_if_present(det_msg, "pose_idx", 0)
            _set_if_present(det_msg, "pose_label", "")
            det_array_msg.detections.append(det_msg)

            perc_msg = Perception()
            perc_msg.tracking_id = track_id % 65536
            perc_msg.target_entity_id = str(track_id)
            perc_msg.detection_prob = score
            perc_msg.location = [cx_norm, cy_norm, 0.0]
            perc_msg.yaw = 0.0
            perc_msg.entity_class = "object"
            perc_msg.entity_color = ""
            # Optional fields; absent from this repo's definitions
            _set_if_present(perc_msg, "occlusion", "")
            _set_if_present(perc_msg, "pose", "")
            perc_msg.match_prob = score
            perc_array_msg.perceptions.append(perc_msg)

        self.detections_pub.publish(det_array_msg)
        self.perceptions_pub.publish(perc_array_msg)

    def _init_perception_output(self):
        """Set up the corrected Perception output.

        Publishes PerceptionArray on the topic prediction_node actually
        subscribes to, with locations in metres NED and target_entity_id
        filled from the mission briefing.
        """

        topic = self.get_parameter('perception_output_topic').value
        self.perception_output_pub = self.create_publisher(
            PerceptionArray, topic, PERCEPTION_QOS)
        self.get_logger().info(f"Perception output on: {topic}")

        # --- world projection -----------------------------------------
        self.projector = GroundProjector(
            camera_xyz=self.get_parameter('camera_offset_xyz').value,
            camera_rpy_deg=self.get_parameter('camera_rpy_deg').value,
            ground_z=float(self.get_parameter('ground_z').value),
            max_range=float(self.get_parameter('max_projection_range').value),
        )

        self.create_subscription(
            PoseStamped, self.get_parameter('pose_topic').value,
            self.pose_callback, PERCEPTION_QOS)
        self.create_subscription(
            CameraInfo, self.get_parameter('depth_info_topic').value,
            self.camera_info_callback, PERCEPTION_QOS)

        if self.get_parameter('use_depth').value:
            self.create_subscription(
                Image, self.get_parameter('depth_image_topic').value,
                self.depth_callback, PERCEPTION_QOS)
        else:
            self.get_logger().info(
                "use_depth is false: ground-plane projection only")

        # Intrinsics fall back to the configured FOV until CameraInfo shows
        # up, so a missing camera_info topic degrades instead of failing.
        # Applied on the first frame, when the real image size is known.
        self.camera_fov_degrees = float(
            self.get_parameter('camera_fov_degrees').value)

        # --- mission entities and scene graph -------------------------
        config_path = self.get_parameter('mission_config_path').value
        self.entities = parse_entities_of_interest(config_path)
        if not self.entities:
            self.get_logger().warn(
                f"No entities of interest in {config_path}: "
                "target_entity_id will stay empty and prediction_node will "
                "ignore these messages")

        # gt_vocabulary: name colours the way the episodes do.
        # max_frames: this node runs for a whole mission, so the graph
        # history is capped rather than retained for a final save_jsonl().
        self.sg_builder = SceneGraphBuilder(
            text_prompt=self.text_prompt, gt_vocabulary=True, max_frames=60)

        # One mission filter per entity, prompted with that entity's colour,
        # so its score is evidence for *that* entity rather than the prompt.
        thresh = float(self.get_parameter('mission_score_thresh').value)
        self.mission_filters = {}
        for entity in self.entities:
            prompt = f"{entity.color} {entity.prompt_noun}".strip()
            self.mission_filters[entity.entity_id] = SceneGraphMissionFilter(
                text_prompt=prompt, hard_mode=False, score_thresh=thresh)

    # ------------------------------------------------------------------
    # Inputs for world projection
    # ------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        """Drone body pose in NED."""
        self.projector.set_pose(msg.pose)

    def camera_info_callback(self, msg: CameraInfo):
        """Depth camera intrinsics; overrides the FOV-derived fallback."""
        self.projector.set_camera_info(msg)

    def depth_callback(self, msg: Image):
        """Planar depth image, converted to metres."""

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}",
                                    throttle_duration_sec=5.0)
            return

        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        # 16UC1 depth is millimetres; 32FC1 is already metres.
        if np.issubdtype(depth.dtype, np.integer):
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        self.projector.set_depth(depth)
        if not self.depth_available:
            self.depth_available = True
            self.get_logger().info(
                f"Depth stream active ({msg.width}x{msg.height}, "
                f"{msg.encoding}): projecting with depth")

    def _publish_perception_output(self, tracks, image_msg, img_h, img_w, frame_bgr):
        """Publish the corrected PerceptionArray.

        Differs from _publish_tracks in every field that matters:
          - stamp is the image's sim-clock stamp, not wall clock
          - location is metres NED, not normalized image coords
          - entity_color uses the episodes' colour vocabulary
          - target_entity_id / match_prob come from the scene-graph match
        """

        activated = [t for t in tracks if t.is_activated]

        if not self.projector.has_intrinsics and self.camera_fov_degrees > 0.0:
            self.projector.set_intrinsics_from_fov(
                img_w, img_h, self.camera_fov_degrees)
            self.get_logger().warn(
                f"No CameraInfo yet: using {self.camera_fov_degrees} deg FOV "
                f"intrinsics for {img_w}x{img_h}", once=True)

        msg = PerceptionArray()
        # Sim clock: prediction_node derives simulation_time from this.
        msg.stamp = image_msg.header.stamp
        msg.frame_num = self.frame_id % 65536

        # Scene graph over all activated tracks: colour and mission scores.
        frame_graph = self.sg_builder.update(
            self.frame_id, activated, img_h, img_w, frame_bgr=frame_bgr)
        nodes = pp.nodes_by_track_id(frame_graph)

        # {entity_id: {track_id: score}} -- one filter per entity
        scores = {entity_id: mission_filter.score_tracks(frame_graph)
                  for entity_id, mission_filter in self.mission_filters.items()}

        methods = {}
        for track in activated:
            node = nodes.get(int(track.track_id), {})
            entity_color = pp.to_gt_color(node.get("color"))

            per_entity = {entity_id: track_scores.get(int(track.track_id), 0.0)
                          for entity_id, track_scores in scores.items()}
            target_entity_id, match_prob = pp.match_entity(
                entity_color, self.entities, per_entity)

            location, method = self._project_track(track, img_h, img_w)
            methods[method] = methods.get(method, 0) + 1

            msg.perceptions.append(pp.build_perception(
                Perception, track,
                frame_number=self.frame_id,
                location=location,
                entity_color=entity_color,
                target_entity_id=target_entity_id,
                match_prob=match_prob,
                set_field=_set_if_present,
            ))

        self.perception_output_pub.publish(msg)
        self._log_projection_methods(methods)

    def _project_track(self, track, img_h, img_w):
        """Project a track's bbox bottom-centre to an NED world point.

        Bottom-centre rather than the box centre because that is where the
        vehicle meets the ground, which is what both projection paths assume.
        """

        x, y, w, h = track.tlwh
        u = float(x + w / 2.0)
        v = float(y + h)

        # Clamp to the image: a box can extend past the frame edge.
        u = min(max(u, 0.0), img_w - 1.0)
        v = min(max(v, 0.0), img_h - 1.0)

        return self.projector.project(u, v, img_w, img_h)

    def _log_projection_methods(self, methods):
        """Report which projection path is in use, when it changes."""

        if not methods:
            return

        dominant = max(methods, key=methods.get)
        if dominant == self._last_projection_method:
            return
        self._last_projection_method = dominant

        summary = ", ".join(f"{name}={count}"
                            for name, count in sorted(methods.items()))
        if dominant in ("depth", "ground_plane"):
            self.get_logger().info(f"Projecting via {dominant} ({summary})")
        else:
            self.get_logger().warn(
                f"Cannot project tracks ({summary}): "
                "location will be zeros. Check pose/camera_info topics.")

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
