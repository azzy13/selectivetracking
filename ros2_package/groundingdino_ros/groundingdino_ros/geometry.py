"""
Image-plane -> AirSim NED world projection for GroundingDINO tracks.

msgs Perception.location is float32[3] in metres, AirSim NED
(x=North, y=East, z=Down, ground at z=0). Consumers treat it as such:
prediction_node feeds location[0:2] straight into its map.

Two projection paths, in order of preference:

  "depth"        sample the depth camera at the pixel and unproject.
  "ground_plane" intersect the pixel ray with the z=0 plane. Used when no
                 depth image is available, or depth is invalid at that
                 pixel. Assumes the target sits on flat ground, which the
                 benchmark episodes do (all entities have z=0.0).

The world->camera transform mirrors point_cloud_node3.get_extrinsic_from_pose():

    E = C . D^T . S . R^T . T

  T  translate by -drone_position          (world axes, origin at drone)
  R^T rotate world axes into body axes     (R = body->world from the pose quaternion)
  S  translate by -camera mounting offset  (body axes)
  D^T rotate body axes into mount axes     (D = Rz(yaw).Ry(pitch).Rx(roll))
  C  permute mount (x fwd, y right, z down) -> optical (x right, y down, z fwd)

so X_cam = E . X_world, and we unproject with inv(E). The rotation helpers
are reimplemented here rather than pulled from open3d, which is not
installed in the groundingdino_ros image.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# mount (x forward, y right, z down) -> optical (x right, y down, z forward)
_C = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def rotation_from_quaternion(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit-quaternion -> 3x3 rotation matrix (open3d's (w,x,y,z) order)."""

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def rotation_from_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rz(yaw) . Ry(pitch) . Rx(roll), radians -- open3d's zyx convention."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    return rz @ ry @ rx


class GroundProjector:
    """Turns image pixels into NED world points, given camera state.

    Feed it camera_info, poses and (optionally) depth images as they
    arrive; call project() per track. Everything is optional until
    project() is called, which reports why it could not produce a point.
    """

    def __init__(
        self,
        camera_xyz=(0.3, 0.0, -0.2),
        camera_rpy_deg=(0.0, 0.0, 0.0),
        ground_z: float = 0.0,
        max_range: float = 200.0,
    ):
        # Camera mounting offset on the drone body, from the mission config
        # (controllable_vehicles[].sensors[].xyz / rpy-deg).
        self.camera_xyz = tuple(float(v) for v in camera_xyz)
        self.camera_rpy_deg = tuple(float(v) for v in camera_rpy_deg)
        self.ground_z = float(ground_z)
        self.max_range = float(max_range)

        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.info_size: Optional[Tuple[int, int]] = None
        self.pose = None
        self.depth: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State intake
    # ------------------------------------------------------------------

    def set_camera_info(self, msg) -> None:
        """Store intrinsics from a sensor_msgs/CameraInfo."""

        k = msg.k if hasattr(msg, "k") else msg.K
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0.0 or fy <= 0.0:
            return
        self.intrinsics = (fx, fy, cx, cy)
        self.info_size = (int(msg.width), int(msg.height))

    def set_intrinsics_from_fov(self, width: int, height: int,
                                fov_degrees: float) -> None:
        """Derive intrinsics from a horizontal FOV, as AirSim specifies them.

        Fallback for when no CameraInfo topic is published. AirSim uses a
        square pixel and a centred principal point, so fy == fx.
        """

        if width <= 0 or height <= 0 or not (0.0 < fov_degrees < 180.0):
            return
        fx = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
        self.intrinsics = (fx, fx, width / 2.0, height / 2.0)
        self.info_size = (int(width), int(height))

    def set_pose(self, pose) -> None:
        """Store a geometry_msgs/Pose: the drone body pose in NED."""

        self.pose = pose

    def set_depth(self, depth: Optional[np.ndarray]) -> None:
        """Store a depth image in metres (float), or None to clear it."""

        self.depth = depth

    @property
    def has_pose(self) -> bool:
        return self.pose is not None

    @property
    def has_intrinsics(self) -> bool:
        return self.intrinsics is not None

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _world_to_camera(self) -> Optional[np.ndarray]:
        """Build E = C . D^T . S . R^T . T from the current pose."""

        if self.pose is None:
            return None

        position = self.pose.position
        orientation = self.pose.orientation

        t = np.eye(4)
        t[:3, 3] = (-position.x, -position.y, -position.z)

        r = np.eye(4)
        r[:3, :3] = rotation_from_quaternion(
            orientation.w, orientation.x, orientation.y, orientation.z)

        s = np.eye(4)
        s[:3, 3] = (-self.camera_xyz[0], -self.camera_xyz[1], -self.camera_xyz[2])

        d = np.eye(4)
        roll, pitch, yaw = (math.radians(v) for v in self.camera_rpy_deg)
        d[:3, :3] = rotation_from_zyx(yaw, pitch, roll)

        return _C @ d.T @ s @ r.T @ t

    def _scaled_intrinsics(self, image_w: int, image_h: int):
        """Rescale intrinsics if the detection image differs from CameraInfo.

        The scene and depth cameras are the same AirSim sensor, so this is
        normally a no-op, but the node must not silently mis-project if the
        two streams are configured at different resolutions.
        """

        fx, fy, cx, cy = self.intrinsics
        if not self.info_size:
            return fx, fy, cx, cy

        info_w, info_h = self.info_size
        if info_w == image_w and info_h == image_h:
            return fx, fy, cx, cy

        sx, sy = image_w / float(info_w), image_h / float(info_h)
        return fx * sx, fy * sy, cx * sx, cy * sy

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, u: float, v: float, image_w: int, image_h: int
                ) -> Tuple[Optional[np.ndarray], str]:
        """Project pixel (u, v) to an NED world point.

        Returns (point, method). method is "depth" or "ground_plane" on
        success, or a short reason string with point None on failure.
        """

        if self.intrinsics is None:
            return None, "no_intrinsics"

        extrinsic = self._world_to_camera()
        if extrinsic is None:
            return None, "no_pose"

        cam_to_world = np.linalg.inv(extrinsic)
        fx, fy, cx, cy = self._scaled_intrinsics(image_w, image_h)

        # Ray through the pixel, in the optical frame (z forward).
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

        depth = self._sample_depth(u, v, image_w, image_h)
        if depth is not None:
            point_cam = np.append(ray_cam * depth, 1.0)
            point = (cam_to_world @ point_cam)[:3]
            if self._plausible(point):
                return point, "depth"

        point = self._intersect_ground(ray_cam, cam_to_world)
        if point is not None and self._plausible(point):
            return point, "ground_plane"

        return None, "no_intersection"

    def _sample_depth(self, u: float, v: float,
                      image_w: int, image_h: int) -> Optional[float]:
        """Median planar depth in metres over a small patch around (u, v)."""

        if self.depth is None:
            return None

        dh, dw = self.depth.shape[:2]
        # Detection pixels are in the scene image; depth may be a different size.
        du = int(round(u * dw / float(image_w)))
        dv = int(round(v * dh / float(image_h)))
        if not (0 <= du < dw and 0 <= dv < dh):
            return None

        half = 2
        patch = self.depth[max(0, dv - half): min(dh, dv + half + 1),
                           max(0, du - half): min(dw, du + half + 1)]
        patch = patch[np.isfinite(patch)]
        patch = patch[(patch > 0.0) & (patch < self.max_range)]
        if patch.size == 0:
            return None

        return float(np.median(patch))

    def _intersect_ground(self, ray_cam: np.ndarray,
                          cam_to_world: np.ndarray) -> Optional[np.ndarray]:
        """Intersect the camera ray with the z = ground_z plane."""

        origin = cam_to_world[:3, 3]
        direction = cam_to_world[:3, :3] @ ray_cam

        # NED: z is down, so the ray must be pointing downward and the
        # camera must be above the ground plane.
        if direction[2] <= 1e-6:
            return None

        distance = (self.ground_z - origin[2]) / direction[2]
        if distance <= 0.0:
            return None

        return origin + distance * direction

    def _plausible(self, point: np.ndarray) -> bool:
        """Reject non-finite or absurdly distant points."""

        if not np.all(np.isfinite(point)):
            return False
        if self.pose is None:
            return True

        origin = np.array([self.pose.position.x,
                           self.pose.position.y,
                           self.pose.position.z])
        return bool(np.linalg.norm(point - origin) <= self.max_range)
