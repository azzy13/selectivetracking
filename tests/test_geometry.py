"""Projection tests for groundingdino_ros.geometry.

Synthetic camera poses with hand-checkable answers -- no sim required.
Frame is AirSim NED: x=North, y=East, z=Down, ground at z=0.
"""

import math
import types

import numpy as np
import pytest

from geometry import GroundProjector, rotation_from_quaternion, rotation_from_zyx


# 640x360 @ 120 deg horizontal FOV, matching mission_briefing/config.json
WIDTH, HEIGHT, FOV = 640, 360, 120.0
FX = (WIDTH / 2.0) / math.tan(math.radians(FOV) / 2.0)


def make_pose(x, y, z, yaw_rad=0.0):
    """geometry_msgs/Pose stand-in: NED position + yaw about the down axis."""
    return types.SimpleNamespace(
        position=types.SimpleNamespace(x=x, y=y, z=z),
        orientation=types.SimpleNamespace(
            w=math.cos(yaw_rad / 2.0), x=0.0, y=0.0, z=math.sin(yaw_rad / 2.0)),
    )


def down_looking(**kwargs):
    """Projector for a camera pitched 90 deg down, no mounting offset."""
    projector = GroundProjector(
        camera_xyz=(0.0, 0.0, 0.0), camera_rpy_deg=(0.0, -90.0, 0.0), **kwargs)
    projector.set_intrinsics_from_fov(WIDTH, HEIGHT, FOV)
    return projector


# ---------------------------------------------------------------- helpers

def test_quaternion_identity_and_yaw():
    assert np.allclose(rotation_from_quaternion(1, 0, 0, 0), np.eye(3))

    # 90 deg about the NED down axis: North -> East
    rot = rotation_from_quaternion(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_zyx_pitch_down_maps_forward_to_down():
    # -90 deg pitch takes the mount's forward axis to the body's down axis
    rot = rotation_from_zyx(0.0, math.radians(-90.0), 0.0)
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-9)


# ------------------------------------------------------- ground-plane path

def test_principal_ray_lands_directly_below_drone():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "ground_plane"
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-6)


def test_offset_pixel_lands_east_when_facing_north():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))

    # one focal length right of centre == 45 deg off-axis == 10 m at 10 m up
    point, method = projector.project(WIDTH / 2 + FX, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "ground_plane"
    assert np.allclose(point, [0.0, 10.0, 0.0], atol=1e-6)


def test_drone_yaw_rotates_the_projection():
    projector = down_looking()
    # yawed 90 deg (facing East): image-right now points South
    projector.set_pose(make_pose(0.0, 0.0, -10.0, yaw_rad=math.pi / 2))

    point, _ = projector.project(WIDTH / 2 + FX, HEIGHT / 2, WIDTH, HEIGHT)

    assert np.allclose(point, [-10.0, 0.0, 0.0], atol=1e-6)


def test_drone_position_offsets_the_projection():
    projector = down_looking()
    projector.set_pose(make_pose(100.0, -50.0, -20.0))

    point, _ = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert np.allclose(point, [100.0, -50.0, 0.0], atol=1e-6)


def test_horizon_ray_has_no_ground_intersection():
    # level camera: the principal ray is horizontal and never reaches z=0
    projector = GroundProjector(camera_xyz=(0.0, 0.0, 0.0), camera_rpy_deg=(0.0, 0.0, 0.0))
    projector.set_intrinsics_from_fov(WIDTH, HEIGHT, FOV)
    projector.set_pose(make_pose(0.0, 0.0, -10.0))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert point is None
    assert method == "no_intersection"


# -------------------------------------------------------------- depth path

def test_depth_path_agrees_with_ground_plane():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    # planar depth == altitude for a camera looking straight down
    projector.set_depth(np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "depth"
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-6)


def test_depth_path_reports_object_above_ground():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    # a rooftop 4 m up reads 6 m of planar depth
    projector.set_depth(np.full((HEIGHT, WIDTH), 6.0, dtype=np.float32))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "depth"
    assert np.allclose(point, [0.0, 0.0, -4.0], atol=1e-6)


def test_invalid_depth_falls_back_to_ground_plane():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    projector.set_depth(np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "ground_plane"
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-6)


def test_depth_image_resolution_is_rescaled():
    projector = down_looking()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    # half-resolution depth stream; the sampler must scale the pixel
    depth = np.full((HEIGHT // 2, WIDTH // 2), 1000.0, dtype=np.float32)
    depth[HEIGHT // 4, WIDTH // 4] = 10.0
    projector.set_depth(depth)

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "depth"
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-6)


def test_out_of_range_depth_is_rejected():
    projector = down_looking(max_range=50.0)
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    projector.set_depth(np.full((HEIGHT, WIDTH), 5000.0, dtype=np.float32))

    point, method = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert method == "ground_plane"


# ------------------------------------------------------------- degradation

def test_missing_pose_is_reported():
    projector = down_looking()
    assert projector.project(0, 0, WIDTH, HEIGHT) == (None, "no_pose")


def test_missing_intrinsics_is_reported():
    projector = GroundProjector()
    projector.set_pose(make_pose(0.0, 0.0, -10.0))
    assert projector.project(0, 0, WIDTH, HEIGHT) == (None, "no_intrinsics")


def test_camera_info_intrinsics_match_fov_derivation():
    info = types.SimpleNamespace(
        k=[FX, 0.0, WIDTH / 2, 0.0, FX, HEIGHT / 2, 0.0, 0.0, 1.0],
        width=WIDTH, height=HEIGHT)

    from_info = GroundProjector()
    from_info.set_camera_info(info)
    from_fov = GroundProjector()
    from_fov.set_intrinsics_from_fov(WIDTH, HEIGHT, FOV)

    assert np.allclose(from_info.intrinsics, from_fov.intrinsics)


def test_mounting_offset_shifts_the_camera():
    # camera 2 m forward of the body origin, drone facing North
    projector = GroundProjector(
        camera_xyz=(2.0, 0.0, 0.0), camera_rpy_deg=(0.0, -90.0, 0.0))
    projector.set_intrinsics_from_fov(WIDTH, HEIGHT, FOV)
    projector.set_pose(make_pose(0.0, 0.0, -10.0))

    point, _ = projector.project(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT)

    assert np.allclose(point, [2.0, 0.0, 0.0], atol=1e-6)
