#!/usr/bin/env python3
"""
robot_control.py — Agibot G02 Desktop Object Grasping: robot/camera/perception/motion.

No Flask dependency. Everything the web GUI needs to drive (dry_run, grasp_loop,
plan_grasp, gripper/home helpers, shared constants) lives here; web_server.py
imports from this module and never touches GDK/YOLO/motion code directly.

Vision:      YOLO (ultralytics)
Trajectory:  GDK end_effector_pose_control() with 50 Hz smoothstep interpolation
SDK:         agibot_gdk v3.38
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import agibot_gdk as gdk
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# YOLO fuse compatibility patch
# ══════════════════════════════════════════════════════════════════════════════

def _patch_yolo_fuse(model) -> None:
    """
    Patch the underlying model's fuse() to tolerate missing 'bn' attributes.

    Root cause: some .pt files saved with older ultralytics store BatchNorm as
    a plain Python attribute, not a registered nn.Module submodule.  Newer
    ultralytics fuse() calls delattr() through torch.nn.Module.__delattr__,
    which looks only in _parameters/_buffers/_modules and raises AttributeError
    when the attribute was set via object.__setattr__ instead.

    This patch replaces fuse() on the model instance with a version that catches
    the AttributeError and nulls the attribute so forward_fuse can still be used.
    """
    import types as _types
    import torch.nn as _nn

    def _safe_fuse(self, verbose=True):
        for m in self.modules():
            if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "forward_fuse"):
                try:
                    if isinstance(m.bn, _nn.BatchNorm2d):
                        from ultralytics.utils.torch_utils import fuse_conv_and_bn
                        m.conv = fuse_conv_and_bn(m.conv, m.bn)
                        try:
                            delattr(m, "bn")
                        except AttributeError:
                            object.__setattr__(m, "bn", None)
                        m.forward = m.forward_fuse
                except Exception:
                    pass
        return self

    model.model.fuse = _types.MethodType(_safe_fuse, model.model)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENSOR_DIR = Path(__file__).parent / "sensor"
DEFAULT_MODEL_PATH = str(Path(__file__).parent / "yolov8n.pt")

# Detection
CONFIDENCE_THRESHOLD = 0.50
DETECTION_SAMPLES    = 5        # frames to median-blend for stable detection
DEPTH_MIN_MM         = 80
DEPTH_MAX_MM         = 3500
DEPTH_PATCH_RADIUS   = 8        # px half-width for center-patch fallback

# Grasp geometry (base_link, metres)
APPROACH_HEIGHT    = 0.20   # clearance above grasp_z when positioning at standoff
APPROACH_STANDOFF  = 0.18   # X distance in front of bottle for lateral approach
GRASP_Z_OFFSET     = 0.02   # final height above object surface
TABLE_CLEARANCE    = 0.08   # minimum z above estimated table surface
LIFT_AFTER_GRASP   = 0.20   # lift height after closing gripper

# Obstacle avoidance (lightweight, single-shot — plan_grasp() only, never per-tick)
OBSTACLE_IOU_MATCH_THRESH = 0.30   # box IoU with `best` above this = same object, not an obstacle
OBSTACLE_RADIUS_MIN       = 0.02   # m — clamp floor to reject depth/bbox noise
OBSTACLE_RADIUS_MAX       = 0.25   # m — clamp ceiling to reject depth/bbox noise
OBSTACLE_CLEARANCE_MARGIN = 0.05   # m — required standoff between EE path and obstacle surface
OBSTACLE_HEIGHT_BUMP      = 0.10   # m — single clear_z retry bump to fly over a blocked transit

# Motion
TRAJ_HZ     = 50    # interpolation Hz for end-effector commands
LIFE_TIME   = 0.18  # GDK EndEffectorPose life_time (s)

# Arm selection is by the sign of y, but y near the centreline is only accurate to
# a few cm, so the sign flips on noise and the arm alternates between plans. 5 of
# the 6 logged grasp failures had |y| ≤ 0.11 m. Within this band the previously
# chosen arm is kept unless the object is clearly on the other side.
ARM_SELECT_HYSTERESIS = 0.06   # m

# Workspace safety limits (base_link, metres)
#
# X upper bound widened 0.85 → 0.95: 27 of 49 logged rejections were targets at
# x = 0.85–0.93, i.e. genuinely reachable but clipped by the old hard limit.
# The arm still reaches them — the bound is a sanity guard against bogus depth,
# not a reach model, so it is set just outside the useful envelope.
#
# Y bounds widened past the centreline by ARM_SELECT_HYSTERESIS: _select_arm()
# deliberately KEEPS the previous arm for any |y| <= ARM_SELECT_HYSTERESIS (the
# sign is noise there, not signal), so a target that hysteresis pins to "right"
# can still legitimately read y up to +ARM_SELECT_HYSTERESIS. Without this the
# two policies fought each other: hysteresis kept the arm locked while the
# workspace check rejected every plan for exactly the y range hysteresis was
# defending, so a target sitting a few cm on the "wrong" side of the centreline
# — same side every sample, never far enough to trip the hysteresis switch —
# looped forever (repeated "outside {arm} workspace" + waist-scan retries,
# 2026-09-11 field log). The two must agree on where the boundary actually is.
WORKSPACE = {
    "left":  {"x": (0.15, 0.95), "y": (-ARM_SELECT_HYSTERESIS,  0.55), "z": (0.25, 1.30)},
    "right": {"x": (0.15, 0.95), "y": (-0.55,  ARM_SELECT_HYSTERESIS), "z": (0.25, 1.30)},
}

# Gripper (position ranges per readme_v3.38 move_ee_pos() spec)
#   omnipicker : [-0.785, 0]   -0.785=open   0=close
#   dahuan     : [0, 0.025]     0=open       0.025=close
#   ctek90d    : [-0.91, 0]    -0.91=open    0=close
GRIPPER_OPEN_POS  = {"omnipicker": [-0.785], "dahuan": [0.0],   "ctek90d": [-0.91]}
GRIPPER_CLOSE_POS = {"omnipicker": [0.0],    "dahuan": [0.025], "ctek90d": [0.0]}

# Valid CLI/API choices — shared between argparse (grasp_pipeline.py) and the
# web GUI's /api/config (web_server.py) so the two can never drift apart.
_GRIPPER_CHOICES = ["omnipicker", "dahuan", "ctek90d"]
_ARM_CHOICES     = ["auto", "right", "left"]

# Whole-body joint names as they appear in get_joint_states() / JointControlReq.
_HOME_JOINT_NAMES = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7",
]
_ARM_L_JOINT_NAMES = _HOME_JOINT_NAMES[8:15]    # idx21..idx27_arm_l_joint1..7
_ARM_R_JOINT_NAMES = _HOME_JOINT_NAMES[15:22]   # idx61..idx67_arm_r_joint1..7

# Grasp verification thresholds. Gripper thresholds are fractions of each
# gripper's *own* travel, not absolute radians: the three supported grippers span
# 0.785 / 0.025 / 0.91 rad, so any absolute margin that suits the omnipicker is
# wider than the dahuan's entire range (it would report HOLDING unconditionally).
# The fractions below reproduce the previous absolute omnipicker values exactly
# (0.05/0.785 and 0.02/0.785).
# Gripper: finger stopped >this fraction of travel short of close → object blocked it
GRIPPER_HOLD_POS_FRAC = 0.064
# Vision: target must stay within this pixel radius to be "still on table"
GRASP_CHECK_PIXEL_RADIUS = 70    # px

# The gripper cannot use move_ee_pos(): readme_v3.38 states it "不可与伺服接口同时
# 使用，若有与伺服接口同时使用的需求请通过伺服接口控制末端执行器", and this arm runs
# under end_effector_pose_control() — a servo interface. EndEffectorPose carries no
# gripper field (readme_v3.38:577 — only group / left+right pose / life_time), so
# the cartesian channel cannot carry the finger either. That leaves
# joint_servo_control(), the one servo interface whose joint_names may include
# end-effector joints (readme_v3.38:3589, and the arm+EE example at :3800).
#
# So the finger is streamed at 100 Hz alongside the arm's own joints, held at their
# measured positions so the arm does not drift while the finger closes.
_SERVO_HZ           = 100.0  # joint_servo_control's required rate
_SERVO_PERIOD_S     = 0.01   # req.control_period, matching the readme example
_GRIPPER_QUIESCE_S  = 0.30   # > LIFE_TIME — let the cartesian channel lapse first
# The readme example sweeps the omnipicker's full -0.785→0 range over 3.0 s. That
# is demo pacing rather than a documented limit, but 0.785 rad in a fraction of a
# second would outrun the finger and make the interpolation meaningless, so keep the
# ramp in the same order of magnitude. Raise this if the finger lags behind.
_GRIPPER_TRAVEL_S   = 1.00   # ramp the finger from current position to target
_GRIPPER_SETTLE_S   = 0.30   # keep streaming the target after arriving
_GRIPPER_MOVED_FRAC = 0.025  # of travel — moved less than this ⇒ never left open

# Head camera — always used for detection
_CAM_COLOR       = gdk.CameraType.kHeadColor
_CAM_DEPTH       = gdk.CameraType.kHeadDepth
_TF_PARENT       = "head_link3"
_EXTR_FILE       = "extrinsic_end_T_head_front_rgbd.json"
_INTR_DEPTH_FILE = "intrinsic_head_front_depth.json"

# Home joint position (matches bottle_workflow_live.py)

_HOME_JOINT_POS = [
    # body 1-5
    -1.0793,  1.9992, -0.8723, 0.0, 0.0,
    # head 1-3
     0.0,     0.0,    0.3,
    # left arm 1-7
     1.5708, -1.5708, -1.5708, -1.5708,  1.5000, 0.0, 0.0,
    # right arm 1-7
    -1.5708, -1.5708,  1.5708, -1.5708, -1.5000, 0.0, 0.0,
]
_HOME_JOINT_VEL = 0.3   # rad/s
_HOME_WAIT_S    = 6.0   # seconds to wait for joints to settle

# Waist scan — rotate to find the target when detection fails
# body_joint4 (index 3 in the 5-joint body list) is the likely yaw/rotation joint.
# Tune _SCAN_YAW_IDX to 4 (body_joint5) if joint 4 does not rotate left/right.
_SCAN_YAW_IDX    = 4          # index into the 5 body-joint positions
_SCAN_STEPS      = [0.0, 0.2, 0.35, -0.2, -0.35]   # rad offsets (~0°, ±14°, ±26°)
_SCAN_VEL        = 0.25       # rad/s — slow to avoid disturbing arms
_SCAN_WAIT_S     = 4        # seconds to wait after each rotation for camera to stabilise
_SCAN_FAIL_LIMIT = 3          # consecutive plan_grasp() failures before starting a scan

# ══════════════════════════════════════════════════════════════════════════════
# Torso height (crouch fraction c: 1.0 = lowest home pose, 0.0 = fully
# upright/vertical). Hardware-tested body joint limits (NOT the URDF, which
# disagrees with reality):
#   body_joint1 [-1.08, 0.0]   body_joint2 [0, 2.62]   body_joint3 [-1.85, 0.95]
#   body_joint4 [-0.4, 0.4]    body_joint5 [-2.95, 2.95] (yaw)
# All-zero == fully upright. joint_i(c) = c * home_value linearly interpolates
# between the two already-verified-safe endpoints.
#
# joint1's lower bound is -1.08, not the -1.07 originally written here: the
# hardware-tested home value is -1.0793, so a -1.07 limit clamped c=1.0 by
# 0.0093 rad and _torso_body_positions(1.0) was NOT the home pose. Every bound
# must admit its own endpoint or the "c=1.0 is a no-op" invariant is false.
# ══════════════════════════════════════════════════════════════════════════════
_TORSO_LIMITS = [(-1.08, 0.0), (0.0, 2.62), (-1.85, 0.95), (-0.4, 0.4), (-2.95, 2.95)]
# c is a crouch fraction on the line from fully-upright (0.0) to the lowest
# verified home pose (1.0). 0.8 is the *default* ceiling — comfortable working
# range, what the auto mapper is tuned against. 1.0 is the hard ceiling: it is
# the home pose itself, on hardware, so it is known-safe. Nothing above 1.0 is
# valid and the GUI's adjustable range may not be raised past _TORSO_C_HARD_MAX.
_TORSO_C_HARD_MAX = 1.0        # absolute ceiling — never exceeded, not adjustable
_TORSO_C_MIN, _TORSO_C_MAX = 0.0, 0.8
_TORSO_HYSTERESIS = 0.24        # min |Δc| to trigger a re-adjustment in auto mode
_TORSO_VEL         = 0.35      # rad/s — same slow default as waist-scan
_TORSO_WAIT_S      = 1.5       # seconds to wait for torso joints to settle

# Waist-settle feedback (_wait_for_yaw). move_waist_joint() returns when the
# command is *accepted*, not when the joints have arrived, so the scan polls the
# measured yaw instead of assuming a fixed delay. 0.02 rad ≈ 7 mm at 0.35 m.
YAW_SETTLE_TOL_RAD   = 0.02
YAW_SETTLE_TIMEOUT_S = 8.0
YAW_SETTLE_POLL_S    = 0.1

# Auto height mapping: target height above table (m) → crouch fraction c.
# Short objects keep today's behavior (c=1.0); tall objects raise the torso.
_AUTO_TORSO_HEIGHT_LOW  = 0.4   # m — at/below this, c = 1.0 (lowest, unchanged)
_AUTO_TORSO_HEIGHT_HIGH = 1.70   # m — at/above this, c = 0.0 (fully upright)

_torso_state = {"mode": "auto", "c": 1.0, "yaw": 0.0}

# Bumped every time the torso physically moves (set_torso_height / scan_for_target).
# The grasp loop compares it against the value its current plan was built with to
# notice that the camera pose changed under it — in manual mode there is no
# auto-adjust to trigger a re-plan, but the operator can still move the torso
# from the GUI mid-run.
_torso_epoch = {"n": 0}

# Last arm auto-selected by _select_arm(), so a repeated plan of the same object
# near the centreline doesn't flip arms between attempts.
_arm_state = {"auto": None}

# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def _load_intrinsic(path: Path) -> dict:
    with path.open() as f:
        d = json.load(f)
    return {"fx": float(d["Fx"]), "fy": float(d["Fy"]),
            "cx": float(d["Cx"]), "cy": float(d["Cy"])}


def _load_extrinsic(path: Path) -> np.ndarray:
    """Load a rotation+translation extrinsic JSON → 4×4 homogeneous matrix."""
    with path.open() as f:
        d = json.load(f)
    r = d["rotation"]
    qw, qx, qy, qz = float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"])
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)
    t = d["translation"]
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = [float(t["x"]), float(t["y"]), float(t["z"])]
    return M

# ══════════════════════════════════════════════════════════════════════════════
# Coordinate geometry
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _tf_to_matrix(tf_pose) -> np.ndarray:
    R = _quat_to_rot(tf_pose.rotation.x, tf_pose.rotation.y,
                     tf_pose.rotation.z, tf_pose.rotation.w)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = [tf_pose.translation.x, tf_pose.translation.y, tf_pose.translation.z]
    return M


def _pixel_to_base(px: int, py: int, depth_m: float,
                   intr: dict, cam_to_base: np.ndarray) -> np.ndarray:
    x = (px - intr["cx"]) * depth_m / intr["fx"]
    y = (py - intr["cy"]) * depth_m / intr["fy"]
    return (cam_to_base @ np.array([x, y, depth_m, 1.0]))[:3]


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _slerp(q0: list, q1: list, t: float) -> list:
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0:
        q1 = [-v for v in q1]; dot = -dot
    if dot > 0.9995:
        r = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
        n = math.sqrt(sum(v * v for v in r))
        return [v / n for v in r]
    th0 = math.acos(dot)
    th  = th0 * t
    s0  = math.cos(th) - dot * math.sin(th) / math.sin(th0)
    s1  = math.sin(th) / math.sin(th0)
    return [s0 * q0[i] + s1 * q1[i] for i in range(4)]


# ══════════════════════════════════════════════════════════════════════════════
# GDK helpers
# ══════════════════════════════════════════════════════════════════════════════

def _color_frame(camera, cam_type) -> np.ndarray | None:
    """Fetch the latest color frame as a BGR uint8 numpy array."""
    try:
        obj = camera.get_latest_image(cam_type, 500.0)
    except RuntimeError:
        return None
    # GDK returns JPEG-compressed bytes — must decode, not reshape
    return cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)


def _depth_frame(camera, cam_type) -> np.ndarray | None:
    """Fetch the latest depth frame as uint16 millimetres."""
    try:
        obj = camera.get_latest_image(cam_type, 500.0)
    except RuntimeError:
        return None
    return np.frombuffer(obj.data, np.uint16).reshape(obj.height, obj.width)


def _cam_to_base(tf_module, parent_frame: str, cam_to_parent: np.ndarray) -> np.ndarray | None:
    """Compute 4×4 transform: camera frame → base_link."""
    try:
        pose = tf_module.get_tf_from_base_link(parent_frame)
    except RuntimeError:
        return None
    return _tf_to_matrix(pose) @ cam_to_parent


def _ee_pose(tf_module, arm: str) -> tuple[list, list] | None:
    """Return ([x,y,z], [qx,qy,qz,qw]) of the named arm EE, or None."""
    frame = "arm_l_end_link" if arm == "left" else "arm_r_end_link"
    try:
        pose = tf_module.get_tf_from_base_link(frame)
    except RuntimeError:
        return None
    return (
        [pose.translation.x, pose.translation.y, pose.translation.z],
        [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Depth sampling
# ══════════════════════════════════════════════════════════════════════════════

def _depth_patch_median(depth: np.ndarray, px: int, py: int,
                        radius: int = DEPTH_PATCH_RADIUS) -> float | None:
    h, w = depth.shape
    x1, x2 = max(0, px - radius), min(w, px + radius + 1)
    y1, y2 = max(0, py - radius), min(h, py + radius + 1)
    valid = depth[y1:y2, x1:x2]
    valid = valid[(valid >= DEPTH_MIN_MM) & (valid <= DEPTH_MAX_MM)]
    return float(np.median(valid)) / 1000.0 if valid.size > 0 else None


def _depth_body_patch(depth: np.ndarray, det: dict,
                      frame_shape: tuple) -> tuple[float | None, int, int]:
    """
    Sample from the object body (middle 44%×54% of bbox) to avoid cap/highlight noise.
    Falls back to centre-patch median if too few valid pixels.
    Returns (depth_m, sample_px, sample_py).
    """
    fh, fw = frame_shape[:2]
    dh, dw = depth.shape[:2]
    sx, sy = dw / float(fw), dh / float(fh)

    x1 = int(round(det["x1"] * sx));  x2 = int(round(det["x2"] * sx))
    y1 = int(round(det["y1"] * sy));  y2 = int(round(det["y2"] * sy))

    bx1 = max(0, int(x1 + 0.28 * (x2 - x1)))
    bx2 = min(dw, int(x1 + 0.72 * (x2 - x1)))
    by1 = max(0, int(y1 + 0.28 * (y2 - y1)))
    by2 = min(dh, int(y1 + 0.82 * (y2 - y1)))

    body  = depth[by1:by2, bx1:bx2]
    valid = body[(body >= DEPTH_MIN_MM) & (body <= DEPTH_MAX_MM)]
    if valid.size >= 12:
        return float(np.median(valid)) / 1000.0, (bx1 + bx2) // 2, (by1 + by2) // 2

    cx = int(round(det["cx"] * sx))
    cy = int(round(det["cy"] * sy))
    return _depth_patch_median(depth, cx, cy), cx, cy


def _estimate_table_z(depth: np.ndarray, det: dict,
                      frame_shape: tuple, intr: dict,
                      cam_to_base: np.ndarray) -> float | None:
    """Estimate table surface Z in base_link from the strip just below the bbox."""
    fh, fw = frame_shape[:2]
    dh, dw = depth.shape[:2]
    sx, sy = dw / float(fw), dh / float(fh)

    cx  = int(round(det["cx"] * sx))
    y2d = int(round(det["y2"] * sy))
    width = max(8, int(round((det["x2"] - det["x1"]) * sx)))

    x1 = max(0, cx - width // 2);     x2 = min(dw, cx + width // 2)
    y1 = max(0, min(dh - 1, y2d + 6)); y2 = min(dh, y1 + max(12, width // 3))
    if x2 <= x1 or y2 <= y1:
        return None

    strip = depth[y1:y2, x1:x2]
    valid = strip[(strip >= DEPTH_MIN_MM) & (strip <= DEPTH_MAX_MM)]
    if valid.size < 12:
        return None

    d_m   = float(np.median(valid)) / 1000.0
    point = _pixel_to_base((x1 + x2) // 2, (y1 + y2) // 2, d_m, intr, cam_to_base)
    return float(point[2])


def _estimate_obstacles(depth: np.ndarray, obstacle_dets: list[dict],
                        frame_shape: tuple, intr: dict,
                        cam_to_base: np.ndarray) -> list[dict]:
    """
    Turn non-target detection boxes into approximate 3D spheres in base_link.
    Reuses _depth_body_patch + _pixel_to_base — the same localisation path used
    for the grasp target itself. Called once per plan_grasp(), not per tick.
    """
    fh, fw = frame_shape[:2]
    dh, dw = depth.shape[:2]
    sx = dw / float(fw)

    obstacles = []
    for det in obstacle_dets:
        depth_m, dpx, dpy = _depth_body_patch(depth, det, frame_shape)
        if depth_m is None:
            continue
        point = _pixel_to_base(dpx, dpy, depth_m, intr, cam_to_base)
        width_px_depth = (det["x2"] - det["x1"]) * sx
        radius = 0.5 * width_px_depth * depth_m / intr["fx"]
        radius = max(OBSTACLE_RADIUS_MIN, min(OBSTACLE_RADIUS_MAX, radius))
        obstacles.append({"pos": point, "radius": radius,
                          "name": det["name"], "conf": det["conf"]})
    return obstacles


# ══════════════════════════════════════════════════════════════════════════════
# Safety
# ══════════════════════════════════════════════════════════════════════════════

def _check_workspace(pos, arm: str = "right", label: str = "") -> bool:
    ws = WORKSPACE[arm]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if (ws["x"][0] <= x <= ws["x"][1] and
            ws["y"][0] <= y <= ws["y"][1] and
            ws["z"][0] <= z <= ws["z"][1]):
        return True
    print(f"[safety] {label} ({x:.3f},{y:.3f},{z:.3f}) outside {arm} workspace")
    return False


def _select_arm(oy: float, previous: str | None = None) -> str:
    """
    Choose arm based on object Y position in base_link frame.
    Left arm workspace covers y > 0; right arm covers y ≤ 0.

    Inside the ±ARM_SELECT_HYSTERESIS band around the centreline the sign of y is
    not meaningful (y itself carries a few cm of error), so keep `previous` if
    there was one. That stops the arm from alternating between successive plans
    of the same object — and a cross-body reach is exactly where grasps failed.
    """
    if previous is not None and abs(oy) <= ARM_SELECT_HYSTERESIS:
        return previous
    return "left" if oy > 0.0 else "right"


def _has_collision(robot) -> bool:
    try:
        status = robot.get_motion_control_status()
        if status.collision_pairs_1:
            pairs = list(zip(status.collision_pairs_1, status.collision_pairs_2))
            print(f"[collision] pairs={pairs}")
            return True
    except (RuntimeError, AttributeError):
        pass
    return False


def _segment_clear(p0, p1, obstacles: list[dict], margin: float, label: str = "") -> bool:
    """
    Check the straight-line segment p0→p1 keeps `margin` clearance from every
    obstacle sphere's surface. Mirrors _check_workspace()'s print+bool style.
    """
    a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    for obs in obstacles:
        c = obs["pos"]
        t = 0.0 if ab_len_sq < 1e-9 else max(0.0, min(1.0, float(np.dot(c - a, ab) / ab_len_sq)))
        closest = a + t * ab
        clearance = float(np.linalg.norm(c - closest)) - obs["radius"]
        if clearance < margin:
            print(f"[obstacle] {label} segment too close to '{obs['name']}' "
                  f"(clearance={clearance:.3f}m < margin={margin:.3f}m)")
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Perception
# ══════════════════════════════════════════════════════════════════════════════

def _detect(model, frame: np.ndarray,
            target_class: str, conf_thresh: float,
            include_all: bool = False) -> list[dict]:
    dets = []
    for r in model(frame, verbose=False):
        for box in r.boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            is_target = (name == target_class)
            if conf < conf_thresh or (not include_all and not is_target):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append({
                "name": name, "conf": round(conf, 3),
                "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "is_target": is_target,
            })
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def _bbox_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / float(area_a + area_b - inter)


def _detect_stable(camera, cam_type_color, model,
                   target_class: str, conf_thresh: float,
                   samples: int) -> tuple[np.ndarray, list[dict], dict] | None:
    """
    Collect `samples` frames, pick the highest-confidence anchor, median-blend
    spatially-close candidates to suppress jitter.
    """
    candidates: list[tuple] = []
    for _ in range(max(1, samples)):
        frame = _color_frame(camera, cam_type_color)
        if frame is None:
            time.sleep(0.05)
            continue
        all_dets = _detect(model, frame, target_class, conf_thresh, include_all=True)
        target_dets = [d for d in all_dets if d["is_target"]]
        if target_dets:
            candidates.append((frame, all_dets, target_dets[0]))
        time.sleep(0.06)

    if not candidates:
        return None

    anchor = max(candidates, key=lambda c: c[2]["conf"])[2]
    close  = [c for c in candidates
              if abs(c[2]["cx"] - anchor["cx"]) < 80
              and abs(c[2]["cy"] - anchor["cy"]) < 80] or candidates

    frame, dets, _ = close[-1]   # dets = ALL boxes (target + obstacles) for that frame
    merged = dict(anchor)
    for k in ("x1", "y1", "x2", "y2", "cx", "cy"):
        merged[k] = int(round(float(np.median([c[2][k] for c in close]))))
    merged["conf"] = round(float(np.median([c[2]["conf"] for c in close])), 3)
    return frame, dets, merged


# ══════════════════════════════════════════════════════════════════════════════
# Gripper
# ══════════════════════════════════════════════════════════════════════════════

def _gripper_travel(gripper_type: str) -> float:
    """Full open→close span of a gripper, in rad. Used to scale all thresholds."""
    op = GRIPPER_OPEN_POS.get(gripper_type,  [-0.785])[0]
    cl = GRIPPER_CLOSE_POS.get(gripper_type, [0.0])[0]
    return max(abs(cl - op), 1e-6)


def _gripper(robot, positions: list, gripper_type: str, arm: str = "right"):
    """
    Drive the gripper to `positions[0]` by streaming it through joint_servo_control.

    move_ee_pos() is unusable here — see the _SERVO_HZ comment block above. The
    documented alternative (readme_v3.38:3800) sends one joint_servo_control request
    per tick whose joint_names are the arm's 7 joints followed by the end-effector
    joints. The arm values are read once from get_joint_states() and held constant,
    so it stays put while the finger ramps to target.

    Only one approximate close/open value is needed (GRIPPER_CLOSE_POS /
    GRIPPER_OPEN_POS): the omnipicker is compliant, so commanding the full close
    position grips whatever is in the way and stops there. Every end-effector joint
    on this side is driven to that same value, as in the readme example.
    """
    goal      = float(positions[0])
    arm_names = _ARM_L_JOINT_NAMES if arm == "left" else _ARM_R_JOINT_NAMES

    # The cartesian channel must lapse before another servo interface takes over.
    time.sleep(_GRIPPER_QUIESCE_S)

    try:
        name_to_pos = {s["name"]: s["motor_position"]
                       for s in robot.get_joint_states()["states"]}
        arm_hold = [float(name_to_pos[n]) for n in arm_names]

        side     = robot.get_end_state()["left_end_state" if arm == "left"
                                         else "right_end_state"]
        ee_names = list(side.get("names", []))
        ee_start = [float(s["position"]) for s in side.get("end_states", [])]
    except Exception as e:
        print(f"[gripper] state read failed ({e}) — gripper not commanded")
        return

    if not ee_names or len(ee_names) != len(ee_start):
        print(f"[gripper] no usable end-effector joints for arm={arm} "
              f"(names={ee_names}, positions={len(ee_start)})")
        return

    all_names = arm_names + ee_names
    dt        = 1.0 / _SERVO_HZ
    n_ramp    = max(1, int(_GRIPPER_TRAVEL_S * _SERVO_HZ))
    n_settle  = max(1, int(_GRIPPER_SETTLE_S * _SERVO_HZ))

    def _send(ee_values) -> bool:
        req = gdk.JointServoControlReq()
        req.control_period  = _SERVO_PERIOD_S
        req.joint_names     = all_names
        req.joint_positions = arm_hold + [float(v) for v in ee_values]
        try:
            res = robot.joint_servo_control(req)
        except Exception as e:
            print(f"[gripper] joint_servo_control raised: {e}")
            return False
        if res != 0:
            print(f"[gripper] joint_servo_control returned {res} — stream aborted")
            return False
        return True

    deadline = time.time()
    for i in range(n_ramp):
        t = float(i) / (n_ramp - 1) if n_ramp > 1 else 1.0
        if not _send([p + t * (goal - p) for p in ee_start]):
            return
        deadline = _sleep_until(deadline + dt)

    for _ in range(n_settle):
        if not _send([goal] * len(ee_names)):
            return
        deadline = _sleep_until(deadline + dt)

    print(f"[gripper] {gripper_type} {arm}: {ee_start[0]:.3f} → {goal:.3f} "
          f"({len(all_names)} joints @ {_SERVO_HZ:.0f} Hz)")


def _sleep_until(deadline: float) -> float:
    """Sleep until the absolute time `deadline`, then return it."""
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return deadline


def open_gripper(robot, gripper_type: str, arm: str = "right"):
    _gripper(robot, GRIPPER_OPEN_POS.get(gripper_type, [-0.785]), gripper_type, arm)


def close_gripper(robot, gripper_type: str, arm: str = "right"):
    _gripper(robot, GRIPPER_CLOSE_POS.get(gripper_type, [0.0]), gripper_type, arm)


def check_gripper_holding(robot, gripper_type: str, arm: str) -> bool | None:
    """
    Read end-effector state after close and return whether an object is held.

    Tool/gripper joints live in get_end_state()'s left_end_state/right_end_state
    (not get_joint_states(), which only covers whole-body/arm joints).

    Position is the only criterion: the finger stopped more than
    GRIPPER_HOLD_POS_FRAC of its travel short of the commanded close position, i.e.
    an object blocked it. (The end_states "current" field reads 0.000 A on this
    hardware, so motor load cannot be used.)

    A finger still sitting at the fully-open position is reported as *not* holding —
    that means the close command never reached it, not that something is gripped.

    Returns True (holding), False (empty), or None (joints not found / read error).
    """
    try:
        end_state = robot.get_end_state()
        side = end_state["left_end_state" if arm == "left" else "right_end_state"]
        names       = side.get("names", [])
        end_states  = side.get("end_states", [])
        if not end_states:
            print(f"[grip check] no end-effector joints found for arm={arm}")
            return None

        close_targets = GRIPPER_CLOSE_POS.get(gripper_type, [0.0])
        open_pos      = GRIPPER_OPEN_POS.get(gripper_type, [-0.785])
        travel        = _gripper_travel(gripper_type)
        hold_margin   = GRIPPER_HOLD_POS_FRAC * travel
        moved_eps     = _GRIPPER_MOVED_FRAC   * travel
        for i, (joint, target) in enumerate(zip(end_states, close_targets)):
            name    = names[i] if i < len(names) else f"joint{i}"
            pos_err = abs(joint["position"] - target)
            # Finger never left the open position → the command did not reach it.
            # This is NOT "blocked by an object"; report no-hold so the pipeline
            # does not falsely claim a successful grasp.
            if abs(joint["position"] - open_pos[0]) < moved_eps:
                print(f"[grip check] {name}  pos={joint['position']:.3f} — finger DID NOT MOVE "
                      f"(still at open). Control failure, not a block.")
                return False
            print(f"[grip check] {name}  pos={joint['position']:.3f}  "
                  f"target={target:.3f}  err={pos_err:.3f}")
            if pos_err > hold_margin:
                print("[grip check] HOLDING — finger stopped short of close target")
                return True

        print("[grip check] EMPTY — gripper reached close target")
        return False
    except Exception as e:
        print(f"[grip check] read error: {e}")
        return None


def check_target_lifted(camera, model, plan: dict,
                        conf_thresh: float) -> bool | None:
    """
    After the lift waypoint, verify the target is no longer at its original
    table position by running YOLO on the head camera.

    Returns True  (object not on table → successfully lifted),
            False (object still detected at original location → grasp failed),
            None  (camera/detection error → assume success to avoid false resets).
    """
    best = plan.get("best")
    if best is None:
        return None
    orig_cx, orig_cy = best["cx"], best["cy"]
    target_class     = best["name"]

    try:
        obj   = camera.get_latest_image(gdk.CameraType.kHeadColor, 500.0)
        frame = cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)
        dets  = _detect(model, frame, target_class, conf_thresh)
        for d in dets:
            if (abs(d["cx"] - orig_cx) < GRASP_CHECK_PIXEL_RADIUS and
                    abs(d["cy"] - orig_cy) < GRASP_CHECK_PIXEL_RADIUS):
                print(f"[lift check] target still at ({d['cx']},{d['cy']}) "
                      f"— orig ({orig_cx},{orig_cy}) — GRASP FAILED")
                return False
        print(f"[lift check] target not at original location ({orig_cx},{orig_cy}) "
              f"— LIFTED OK")
        return True
    except Exception as e:
        print(f"[lift check] camera error: {e}")
        return None


def move_to_home(robot, keep_torso: bool = False) -> None:
    """Send all joints to home position and wait for them to settle.

    keep_torso=True holds body joints 1-5 at the torso's current
    auto/manual height (_torso_state) instead of snapping them back to the
    lowest home pose — use this for every reset *during* a grasp run so an
    auto/manual height adjustment survives a failed grasp, a successful
    placement, or an abort. Leave it False only for a full end-of-run/
    shutdown reset to the one posture verified safe on hardware."""
    positions = list(_HOME_JOINT_POS)
    if keep_torso:
        positions[:5] = _torso_body_positions(_torso_state["c"], _torso_state["yaw"])
    req = gdk.JointControlReq()
    req.life_time        = _HOME_WAIT_S
    req.joint_names      = _HOME_JOINT_NAMES
    req.joint_positions  = positions
    req.joint_velocities = [_HOME_JOINT_VEL] * len(_HOME_JOINT_NAMES)
    try:
        robot.joint_control_request(req)
    except RuntimeError as e:
        print(f"[home] joint_control_request failed: {e}"); return
    time.sleep(_HOME_WAIT_S)
    print("[home] reached" +
          (f" (torso kept at c={_torso_state['c']:.2f})" if keep_torso else ""))


# ══════════════════════════════════════════════════════════════════════════════
# Torso height
# ══════════════════════════════════════════════════════════════════════════════

def _torso_body_positions(c: float, yaw: float = 0.0) -> list[float]:
    """5 body joint positions for crouch fraction c (1.0=lowest home, 0.0=upright).
    joint5 (yaw) is passed through unchanged/overridden via `yaw`.

    Clamps against the HARD ceiling, not the adjustable one: the GUI is allowed
    to *tighten* c_max, but a caller that already holds a higher c (e.g. the
    auto mapper before the operator moved the slider) must not have its pose
    silently altered mid-run."""
    c = max(_TORSO_C_MIN, min(_TORSO_C_HARD_MAX, c))
    positions = [c * _HOME_JOINT_POS[i] for i in range(4)] + [yaw]
    return [max(lo, min(hi, p)) for p, (lo, hi) in zip(positions, _TORSO_LIMITS)]


def retract_arms(robot) -> None:
    """Fold both arms to their home posture without touching torso/head joints.

    Raises RuntimeError if the command is rejected — a torso move must not
    proceed with the arms still extended, and a silent return here was
    indistinguishable from a successful retract in the GUI."""
    names = _HOME_JOINT_NAMES[8:]      # arm_l (7) + arm_r (7)
    pos   = _HOME_JOINT_POS[8:]
    req = gdk.JointControlReq()
    req.life_time        = _HOME_WAIT_S
    req.joint_names      = names
    req.joint_positions  = pos
    req.joint_velocities = [_HOME_JOINT_VEL] * len(names)
    try:
        robot.joint_control_request(req)
    except RuntimeError as e:
        raise RuntimeError(f"retract_arms failed: {e}") from e
    time.sleep(_HOME_WAIT_S)
    print("[torso] arms retracted")


def set_torso_height(robot, c: float, yaw: float | None = None) -> tuple[float, bool]:
    """Retract both arms, then move the torso to crouch fraction c (0=upright,
    1=today's lowest home pose), keeping the torso perpendicular to ground.

    `c` is clamped to [_TORSO_C_MIN, _TORSO_C_HARD_MAX] and the clamped value is
    returned as `applied_c` so the GUI can show what was really commanded rather
    than echoing back what was typed.

    `yaw` defaults to the torso's *current* yaw (`_torso_state["yaw"]`), not 0.0 —
    a height change must not silently undo a waist rotation left over from a
    successful scan_for_target().

    Returns (applied_c, converged). Raises RuntimeError if the motion could not
    be commanded at all — the caller surfaces that to the browser instead of
    leaving a silent no-op."""
    if yaw is None:
        yaw = _torso_state["yaw"]
    c = max(_TORSO_C_MIN, min(_TORSO_C_HARD_MAX, c))

    retract_arms(robot)          # raises on failure — do not move the torso with arms out
    positions = _torso_body_positions(c, yaw)
    try:
        robot.move_waist_joint(positions, [_TORSO_VEL] * 5)
    except RuntimeError as e:
        raise RuntimeError(f"move_waist_joint failed: {e}") from e

    measured, ok = _wait_for_yaw(robot, positions[_SCAN_YAW_IDX])
    if not ok:
        print("[torso] waist yaw did not settle — torso height may be inaccurate")
    time.sleep(_TORSO_WAIT_S)

    _torso_state["c"]   = c
    # Store the measured yaw: the next plan's coordinates are only valid for the
    # orientation the camera actually ended up at.
    _torso_state["yaw"] = measured if ok else positions[_SCAN_YAW_IDX]
    print(f"[torso] height set to c={_torso_state['c']:.3f} yaw={_torso_state['yaw']:.3f}")
    _torso_epoch["n"] += 1
    return _torso_state["c"], ok


def _auto_torso_c(target_height_m: float) -> float:
    """Map target height above table (m) to crouch fraction c. Short objects
    (<= _AUTO_TORSO_HEIGHT_LOW) keep today's lowest pose (c=1.0); tall objects
    (>= _AUTO_TORSO_HEIGHT_HIGH) raise the torso toward fully upright (c=0.0)."""
    lo, hi = _AUTO_TORSO_HEIGHT_LOW, _AUTO_TORSO_HEIGHT_HIGH
    if target_height_m <= lo:
        return 1.0
    if target_height_m >= hi:
        return 0.0
    frac = (target_height_m - lo) / (hi - lo)
    return 1.0 - frac


# ══════════════════════════════════════════════════════════════════════════════
# Motion: 50 Hz end-effector interpolation, kBothArms
# ══════════════════════════════════════════════════════════════════════════════

def _move_ee(robot, tf_module, arm: str,
             goal_pos: list, goal_ori: list,
             duration: float, smooth: bool = True) -> bool:
    """
    Smoothstep (or linear) trajectory to goal_pos/ori at 50 Hz.
    The idle arm is read from TF and held at its current pose.
    """
    left  = _ee_pose(tf_module, "left")
    right = _ee_pose(tf_module, "right")
    if left  is None: left  = ([0.30,  0.20, 0.90], [0.0, 0.0, 0.0, 1.0])
    if right is None: right = ([0.30, -0.20, 0.90], [0.0, 0.0, 0.0, 1.0])

    start_pos, start_ori = (left if arm == "left" else right)
    hold_pos,  hold_ori  = (right if arm == "left" else left)

    n      = max(int(duration * TRAJ_HZ), 2)
    t_step = duration / n

    for i in range(n):
        if _has_collision(robot):
            print("[motion] collision — stopping")
            return False

        t = float(i) / (n - 1) if n > 1 else 1.0
        if smooth:
            t = _smoothstep(t)

        pos  = [start_pos[j] + t * (goal_pos[j]  - start_pos[j])  for j in range(3)]
        ori  = _slerp(list(start_ori), list(goal_ori), t)

        lp, lq = (pos, ori)  if arm == "left"  else (hold_pos, hold_ori)
        rp, rq = (pos, ori)  if arm == "right" else (hold_pos, hold_ori)

        req = gdk.EndEffectorPose()
        req.group     = int(gdk.EndEffectorControlGroup.kBothArms)
        req.life_time = LIFE_TIME

        req.left_end_effector_pose.position.x     = lp[0]
        req.left_end_effector_pose.position.y     = lp[1]
        req.left_end_effector_pose.position.z     = lp[2]
        req.left_end_effector_pose.orientation.x  = lq[0]
        req.left_end_effector_pose.orientation.y  = lq[1]
        req.left_end_effector_pose.orientation.z  = lq[2]
        req.left_end_effector_pose.orientation.w  = lq[3]
        req.right_end_effector_pose.position.x    = rp[0]
        req.right_end_effector_pose.position.y    = rp[1]
        req.right_end_effector_pose.position.z    = rp[2]
        req.right_end_effector_pose.orientation.x = rq[0]
        req.right_end_effector_pose.orientation.y = rq[1]
        req.right_end_effector_pose.orientation.z = rq[2]
        req.right_end_effector_pose.orientation.w = rq[3]

        robot.end_effector_pose_control(req)
        time.sleep(t_step)

    return True


def _read_body_yaw(robot) -> float | None:
    """Actual position of the waist-yaw joint, or None if it can't be read."""
    name = _HOME_JOINT_NAMES[_SCAN_YAW_IDX]
    try:
        states = robot.get_joint_states()["states"]
    except Exception:
        return None
    for s in states:
        if s["name"] == name:
            return float(s["motor_position"])
    return None


def _wait_for_yaw(robot, target_yaw: float,
                  tol: float = YAW_SETTLE_TOL_RAD,
                  timeout: float = YAW_SETTLE_TIMEOUT_S) -> tuple[float, bool]:
    """
    Block until the waist yaw joint has actually reached `target_yaw`.

    move_waist_joint() returns as soon as the command is accepted, so a fixed
    sleep is a guess: come up short and every downstream coordinate was computed
    against a camera pose the robot had not finished reaching yet — the plan is
    then correct for the *old* orientation and the gripper lands beside the
    object. Polls the measured joint position instead.

    Returns (measured_yaw, converged).
    """
    deadline = time.time() + timeout
    last     = float("nan")
    while time.time() < deadline:
        last = _read_body_yaw(robot)
        if last is not None and abs(last - target_yaw) <= tol:
            return last, True
        time.sleep(YAW_SETTLE_POLL_S)
    return last, False


# ══════════════════════════════════════════════════════════════════════════════
# Perception + Planning
# ══════════════════════════════════════════════════════════════════════════════

def scan_for_target(robot, camera, tf_module, model,
                    cam_to_parent, intr_depth,
                    target_class, conf_thresh, samples, arm) -> "dict | None":
    """
    Rotate the waist incrementally to search for the target when straight-ahead
    detection has failed repeatedly.  Tries each angle in _SCAN_STEPS (offsets
    from home for body joint _SCAN_YAW_IDX), attempts plan_grasp at each stop,
    and returns the first successful plan.  Restores the body yaw to 0.0
    before returning None if nothing is found.

    Joints 1-4 are held at the torso's *current* crouch fraction (not forced
    back to the lowest home pose) so scanning never undoes an auto/manual
    height adjustment already in effect.

    At each stop the waist is waited on until it has physically arrived before
    the camera is read — see _wait_for_yaw() for why a fixed sleep is not enough.
    """
    base_body = _torso_body_positions(_torso_state["c"], _torso_state["yaw"])

    for i, offset in enumerate(_SCAN_STEPS):
        positions = list(base_body)
        positions[_SCAN_YAW_IDX] += offset

        print(f"[scan] step {i+1}/{len(_SCAN_STEPS)} — "
              f"body_joint{_SCAN_YAW_IDX+1} offset {offset:+.2f} rad")
        try:
            robot.move_waist_joint(positions, [_SCAN_VEL] * 5)
        except Exception as exc:
            print(f"[scan] waist move error: {exc}")
            continue

        measured, ok = _wait_for_yaw(robot, positions[_SCAN_YAW_IDX])
        if not ok:
            print(f"[scan] waist did not settle: commanded "
                  f"{positions[_SCAN_YAW_IDX]:+.3f} rad, measured "
                  f"{measured if measured is not None else float('nan'):+.3f} "
                  f"— skipping this step (coordinates would be wrong)")
            continue
        # Camera pose lags the joint; let one more frame land at the final pose.
        time.sleep(_SCAN_WAIT_S)

        # Coordinates are only valid for the pose the camera was actually at, so
        # record the measured yaw — not the commanded one. The waist has moved,
        # so any plan made earlier is stale: bump the epoch.
        _torso_state["yaw"] = measured
        _torso_epoch["n"] += 1

        plan = plan_grasp(camera, tf_module, model,
                          _CAM_COLOR, _CAM_DEPTH,
                          _TF_PARENT, cam_to_parent, intr_depth,
                          target_class, conf_thresh, samples, arm)
        if plan is not None:
            print(f"[scan] target found at body yaw offset {offset:+.2f} rad "
                  f"(measured yaw {measured:+.3f})")
            return plan

    # Nothing found — restore home yaw
    print("[scan] target not found after full sweep — restoring home yaw")
    try:
        robot.move_waist_joint(base_body, [_SCAN_VEL] * 5)
        measured, ok = _wait_for_yaw(robot, base_body[_SCAN_YAW_IDX])
        if ok:
            _torso_state["yaw"] = measured
            _torso_epoch["n"] += 1      # waist moved back — anything planned is stale
        else:
            print("[scan] home yaw restore did not settle")
    except Exception as exc:
        print(f"[scan] restore waist error: {exc}")

    return None


def plan_grasp(camera, tf_module, model,
               cam_type_color, cam_type_depth,
               tf_parent_frame: str,
               cam_to_parent: np.ndarray,
               intr_depth: dict,
               target_class: str, conf_thresh: float, samples: int,
               arm: str = "auto") -> dict | None:
    """
    Detect target → 3D localise → plan Cartesian waypoints.
    arm="auto" selects left/right based on object Y position.
    Returns a plan dict (including "arm" key) or None on failure.
    """
    c2b = _cam_to_base(tf_module, tf_parent_frame, cam_to_parent)
    if c2b is None:
        print("[plan] TF lookup failed"); return None

    result = _detect_stable(camera, cam_type_color, model,
                            target_class, conf_thresh, samples)
    if result is None:
        print("[plan] no target detected"); return None
    frame, dets, best = result
    print(f"[plan] detected '{best['name']}' conf={best['conf']:.2f}")

    depth = _depth_frame(camera, cam_type_depth)
    if depth is None:
        print("[plan] depth frame unavailable"); return None

    depth_m, dpx, dpy = _depth_body_patch(depth, best, frame.shape)
    if depth_m is None:
        print("[plan] invalid depth at detection"); return None

    obj = _pixel_to_base(dpx, dpy, depth_m, intr_depth, c2b)
    ox, oy, oz = float(obj[0]), float(obj[1]), float(obj[2])

    table_z = _estimate_table_z(depth, best, frame.shape, intr_depth, c2b)
    if table_z is not None:
        oz = max(oz, table_z + TABLE_CLEARANCE)
    # None (not 0.0) when the table surface can't be estimated — grasp_loop()
    # must then leave the torso at its current height rather than assume "short".
    target_height_m = (oz - table_z) if table_z is not None else None

    # Auto arm selection: left arm covers y > 0, right arm covers y ≤ 0
    if arm == "auto":
        selected_arm = _select_arm(oy, _arm_state["auto"])
        _arm_state["auto"] = selected_arm
    else:
        selected_arm = arm
    print(f"[plan] base_link target: x={ox:.3f} y={oy:.3f} z={oz:.3f}  "
          f"depth={depth_m:.3f} m  arm={selected_arm}")

    grasp_z  = oz + GRASP_Z_OFFSET
    clear_z  = grasp_z + APPROACH_HEIGHT   # height that clears the bottle top

    # Obstacles: any detected box that isn't the grasp target (matched by IoU,
    # not class name, so a second same-class object still counts as an obstacle).
    obstacle_dets = [d for d in dets if _bbox_iou(d, best) < OBSTACLE_IOU_MATCH_THRESH]
    obstacles = (_estimate_obstacles(depth, obstacle_dets, frame.shape, intr_depth, c2b)
                 if obstacle_dets else [])
    if obstacles:
        print(f"[plan] {len(obstacles)} obstacle(s): " +
              ", ".join(f"{o['name']}@r={o['radius']:.2f}m" for o in obstacles))

    # Lateral (front-to-back) approach — never descends onto the bottle cap:
    #   1. Align in front of bottle at clearance height (EE midpoint alignment)
    #   2. Lower to barrel height while still at standoff
    #   3. Slide horizontally into grasp position  → close gripper
    #   4. Lift straight up
    def _build_waypoints(cz: float) -> list[tuple[str, list, float, bool]]:
        return [
            # (label, [x, y, z], duration_s, smooth)
            ("align",     [ox - APPROACH_STANDOFF, oy, grasp_z],      1.5, True),
            #("lower",     [ox - APPROACH_STANDOFF, oy, grasp_z], 1.2, True),
            ("slide in",  [ox-0.02,                     oy, grasp_z], 1.5, False),
            # gripper closes after "slide in"
            ("lift",      [ox-0.02,                     oy, cz],      2, True),
        ]

    waypoints = _build_waypoints(clear_z)

    for label, pos, _, _ in waypoints:
        if not _check_workspace(pos, arm=selected_arm, label=label):
            return None

    if obstacles:
        ee_now = _ee_pose(tf_module, selected_arm)
        start_pos = ee_now[0] if ee_now else waypoints[0][1]   # fail-soft on TF miss
        seg_labels = [wp[0] for wp in waypoints]
        NEAR_OBJECT = {"lower", "slide in"}
        TRANSIT     = {"align", "lift"}

        def _segments(wps):
            points = [start_pos] + [wp[1] for wp in wps]
            return list(zip(seg_labels, points[:-1], points[1:]))

        # Hard-fail first: obstacles right next to the object can't be fixed by a height bump
        for label, p0, p1 in _segments(waypoints):
            if label in NEAR_OBJECT and not _segment_clear(
                    p0, p1, obstacles, OBSTACLE_CLEARANCE_MARGIN, label):
                print(f"[plan] near-object segment '{label}' blocked by obstacle — aborting plan")
                return None

        transit_blocked = any(
            label in TRANSIT and not _segment_clear(p0, p1, obstacles, OBSTACLE_CLEARANCE_MARGIN, label)
            for label, p0, p1 in _segments(waypoints)
        )
        if transit_blocked:
            bumped = min(clear_z + OBSTACLE_HEIGHT_BUMP, WORKSPACE[selected_arm]["z"][1])
            if bumped <= clear_z:
                print("[plan] transit segment blocked, no headroom to bump clear_z — aborting plan")
                return None
            print(f"[plan] transit segment blocked — retrying with clear_z bumped to {bumped:.3f}")
            clear_z = bumped
            waypoints = _build_waypoints(clear_z)
            for label, pos, _, _ in waypoints:
                if not _check_workspace(pos, arm=selected_arm, label=label):
                    return None
            still_blocked = any(
                label in TRANSIT and not _segment_clear(p0, p1, obstacles, OBSTACLE_CLEARANCE_MARGIN, label)
                for label, p0, p1 in _segments(waypoints)
            )
            if still_blocked:
                print("[plan] obstacle clearance still violated after height bump — aborting plan")
                return None

    return {
        "obj":             (ox, oy, oz),
        "arm":             selected_arm,
        "frame":           frame,
        "dets":            dets,
        "best":            best,
        "depth_m":         depth_m,
        "waypoints":       waypoints,
        "target_height_m": target_height_m,
        # Torso epoch this plan's coordinates are valid for. Every coordinate
        # above was derived from the camera pose at this epoch; if the torso
        # moves afterwards the whole plan is stale and must be rebuilt.
        "torso_epoch":     _torso_epoch["n"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════════

def execute_grasp(robot, tf_module, gripper_type: str,
                  plan: dict, arm: str = "right",
                  stop_event: threading.Event | None = None) -> bool:
    """
    Move through waypoints, close gripper after 'lower to grasp', verify hold.

    Returns False if a waypoint motion aborts (collision / TF loss / stop_event).
    Returns True once all waypoints complete — caller should then run
    check_target_lifted() for visual confirmation.
    """
    ee = _ee_pose(tf_module, arm)
    ori = ee[1] if ee else [0.0, 0.707, 0.0, 0.707]

    for label, pos, dur, smooth in plan["waypoints"]:
        if stop_event is not None and stop_event.is_set():
            print(f"[grasp] abort requested before '{label}'")
            return False

        print(f"  → {label}  {[round(v,3) for v in pos]}  {dur:.1f}s")
        ok = _move_ee(robot, tf_module, arm, pos, ori, dur, smooth=smooth)
        if not ok:
            print(f"[grasp] aborted at '{label}'")
            return False

        if label == "slide in":
            print("  closing gripper...")
            close_gripper(robot, gripper_type, arm)
            time.sleep(0.5)   # let the fingers travel before reading them back

            # Hold check: how far short of the close target the finger stopped
            holding = check_gripper_holding(robot, gripper_type, arm)
            if holding is False:
                print("  [grasp] gripper reports empty — continuing to lift for visual check")
            elif holding is True:
                print("  [grasp] gripper confirms hold — proceeding to lift")

    print("[grasp] motion complete")
    return True


def reset_arm(robot, tf_module, gripper_type: str,
              plan: dict, arm: str = "right"):
    """Reverse back through waypoints then open gripper (used on grasp failure)."""
    ee  = _ee_pose(tf_module, arm)
    ori = ee[1] if ee else [0.0, 0.707, 0.0, 0.707]

    for label, pos, dur, smooth in reversed(plan["waypoints"]):
        print(f"  [reset] → {label}")
        _move_ee(robot, tf_module, arm, pos, ori, dur, smooth=smooth)

    open_gripper(robot, gripper_type, arm)
    time.sleep(0.5)
    print("[reset] done")


def place_object(robot, tf_module, gripper_type: str,
                 plan: dict, arm: str = "right") -> None:
    """
    Lower to release height, open gripper, slide back, retract.
    Mirrors the lateral approach in reverse so the gripper never sweeps over the bottle.
    """
    ox, oy, oz = plan["obj"]
    grasp_z = oz + GRASP_Z_OFFSET
    clear_z = grasp_z + APPROACH_HEIGHT

    ee  = _ee_pose(tf_module, arm)
    ori = ee[1] if ee else [0.0, 0.707, 0.0, 0.707]

    for label, pos, dur in [
        ("lower to release", [ox,                     oy, clear_z], 1.2),
        ("place",            [ox,                     oy, grasp_z], 1.0),
    ]:
        print(f"  → {label}  {[round(v,3) for v in pos]}")
        _move_ee(robot, tf_module, arm, pos, ori, dur, smooth=True)

    print("  opening gripper — releasing object")
    open_gripper(robot, gripper_type, arm)
    
    time.sleep(0.4)

    for label, pos, dur in [
        ("slide back", [ox - APPROACH_STANDOFF, oy, grasp_z], 1.0),
        #("retract",    [ox - APPROACH_STANDOFF, oy, clear_z], 1.5),
    ]:
        print(f"  → {label}  {[round(v,3) for v in pos]}")
        _move_ee(robot, tf_module, arm, pos, ori, dur, smooth=True)

    print("[place] object released")


# ══════════════════════════════════════════════════════════════════════════════
# Dry-run (perception + plan only, no robot motion)
# ══════════════════════════════════════════════════════════════════════════════

def dry_run(camera, tf_module, model,
            cam_type_color, cam_type_depth,
            tf_parent_frame, cam_to_parent, intr_depth,
            target_class, conf_thresh, samples, arm, attempts=5,
            stop_event: threading.Event | None = None) -> bool:
    print("=" * 60)
    print("DRY-RUN: perception + planning only — robot will not move")
    print("=" * 60)
    for i in range(1, attempts + 1):
        if stop_event is not None and stop_event.is_set():
            print("[dry-run] abort requested"); return False

        print(f"\n[dry-run {i}/{attempts}]")
        plan = plan_grasp(camera, tf_module, model,
                          cam_type_color, cam_type_depth,
                          tf_parent_frame, cam_to_parent, intr_depth,
                          target_class, conf_thresh, samples, arm)
        if plan is None:
            print("  → failed, retrying..."); time.sleep(0.5); continue

        ox, oy, oz = plan["obj"]
        print(f"  target     : x={ox:.3f} y={oy:.3f} z={oz:.3f}")
        print(f"  depth_m    : {plan['depth_m']:.3f}")
        print("  waypoints  :")
        for label, pos, dur, smooth in plan["waypoints"]:
            print(f"    {label:24s}  {[round(v,3) for v in pos]}  {dur:.1f}s  smooth={smooth}")
        print("\nDRY-RUN complete — zero motion commands issued")
        return True

    print("DRY-RUN: all attempts failed")
    return False

# ══════════════════════════════════════════════════════════════════════════════
# Main grasp loop
# ══════════════════════════════════════════════════════════════════════════════

def grasp_loop(camera, robot, tf_module, model,
               cam_to_parent, intr_depth,
               gripper_type, target_class, conf_thresh, samples,
               arm: str = "auto",
               stop_event: threading.Event | None = None, on_loop=None):
    print(f"Grasp loop  target={target_class}  gripper={gripper_type}  arm={arm}")

    print("Initialising: moving to home position...")
    move_to_home(robot, keep_torso=True)
    # Open gripper(s) at start — if arm is "auto" open both since we don't yet know
    # which arm will be used for the first grasp.
    if arm == "auto":
        open_gripper(robot, gripper_type, "right")
        open_gripper(robot, gripper_type, "left")
    else:
        open_gripper(robot, gripper_type, arm)

    loop = 0
    detect_fail_count = 0   # consecutive detection-failure counter

    while stop_event is None or not stop_event.is_set():
        loop += 1
        print(f"\n──── Loop {loop} ────")
        if on_loop is not None:
            on_loop(loop)

        plan = plan_grasp(camera, tf_module, model,
                          _CAM_COLOR, _CAM_DEPTH,
                          _TF_PARENT, cam_to_parent, intr_depth,
                          target_class, conf_thresh, samples, arm)

        if plan is None:
            detect_fail_count += 1
            if detect_fail_count >= _SCAN_FAIL_LIMIT:
                print(f"[{detect_fail_count} consecutive detection failures] "
                      f"starting waist scan...")
                plan = scan_for_target(robot, camera, tf_module, model,
                                       cam_to_parent, intr_depth,
                                       target_class, conf_thresh, samples, arm)
                if plan is None:
                    print("scan complete — target not found; retrying straight-ahead")
                    detect_fail_count = 0
                    time.sleep(0.5)
                    continue
                # plan found during scan — reset counter and proceed
                detect_fail_count = 0
            else:
                print(f"perception/planning failed ({detect_fail_count}) — retrying")
                time.sleep(0.5)
                continue
        else:
            detect_fail_count = 0

        # Auto torso height: re-plan against the fresh camera pose if the
        # torso moved, since plan_grasp()'s TF lookup is stale after a move.
        # target_height_m is None when the table surface couldn't be estimated —
        # in that case leave the torso at whatever height it already has (the
        # pose from before this grasp attempt), don't assume "short" and drop
        # to the lowest pose.
        #
        # In *manual* mode the torso is not moved here, but the operator may have
        # moved it from the GUI between loops. plan_grasp()'s TF lookup is stale
        # for that move too, so re-plan once whenever the torso pose changed
        # since this plan was computed.
        target_height_m = plan.get("target_height_m")
        if _torso_state["mode"] == "auto" and target_height_m is not None:
            desired_c = _auto_torso_c(target_height_m)
            if abs(desired_c - _torso_state["c"]) > _TORSO_HYSTERESIS:
                print(f"[torso] auto height adjust: c {_torso_state['c']:.2f} -> {desired_c:.2f}")
                set_torso_height(robot, desired_c)
                plan = plan_grasp(camera, tf_module, model,
                                  _CAM_COLOR, _CAM_DEPTH,
                                  _TF_PARENT, cam_to_parent, intr_depth,
                                  target_class, conf_thresh, samples, arm)
                if plan is None:
                    print("[torso] re-plan after height change failed — retrying next loop")
                    time.sleep(0.5)
                    continue

        # Manual mode never moves the torso here, but the operator can move it
        # from the GUI while the loop is running — and a plan's coordinates are
        # only valid for the camera pose it was built against. Rebuild rather
        # than grasp at a target the camera is no longer looking at.
        if plan.get("torso_epoch") != _torso_epoch["n"]:
            print(f"[torso] torso moved since this plan was built "
                  f"(epoch {plan.get('torso_epoch')} -> {_torso_epoch['n']}) — re-planning")
            replanned = plan_grasp(camera, tf_module, model,
                                   _CAM_COLOR, _CAM_DEPTH,
                                   _TF_PARENT, cam_to_parent, intr_depth,
                                   target_class, conf_thresh, samples, arm)
            if replanned is None:
                print("[torso] re-plan after manual torso move failed — retrying next loop")
                time.sleep(0.5)
                continue
            plan = replanned

        # Arm is resolved per plan (auto-selected or manually locked)
        plan_arm = plan["arm"]

        ok = execute_grasp(robot, tf_module, gripper_type, plan, plan_arm,
                          stop_event=stop_event)
        if not ok:
            print("grasp motion aborted — resetting")
            reset_arm(robot, tf_module, gripper_type, plan, plan_arm)
            move_to_home(robot, keep_torso=True)
            open_gripper(robot, gripper_type, plan_arm)
            continue

        # Secondary hold check: head camera delta (object no longer on table?)
        lifted = check_target_lifted(camera, model, plan, conf_thresh)
        if lifted is False:
            print("grasp failed (object still on table) — resetting")
            reset_arm(robot, tf_module, gripper_type, plan, plan_arm)
            move_to_home(robot, keep_torso=True)
            open_gripper(robot, gripper_type, plan_arm)
            continue

        print(f"grasp SUCCESS ({plan_arm} arm) — placing object")
        place_object(robot, tf_module, gripper_type, plan, plan_arm)
        move_to_home(robot, keep_torso=True)
        open_gripper(robot, gripper_type, plan_arm)

    # Loop exited via stop_event — leave the arm in a safe, known state.
    if stop_event is not None and stop_event.is_set():
        print("grasp loop stopped — returning to home")
        move_to_home(robot, keep_torso=True)
        if arm == "auto":
            open_gripper(robot, gripper_type, "right")
            open_gripper(robot, gripper_type, "left")
        else:
            open_gripper(robot, gripper_type, arm)
