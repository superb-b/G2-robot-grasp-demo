# G2_grape_235_v1 — G02 Desktop Object Grasping

Agibot G02 humanoid robot — a desktop grasping pipeline that uses the head camera + YOLO detection + end-effector pose control.

[Download Link, Wheels are ready, easy to deploy](https://drive.google.com/file/d/1r5a-BBBn4JGMq6piQwH8MH14ah3oe2t9/view?usp=sharing)

## Prerequisites

This demo is programmed under the system version 235. The following must be configured correctly, otherwise the gripper will not work properly. Open the tablet, go to **Settings** in the lower-right corner of the tablet, and tap **System Update**. If `235` and `thor` are shown, the system can run the scripts below normally. If not, contact support promptly.
If `thor` is shown, you need to connect the first Ethernet port on the back of the robot to the host. Configure the host Ethernet port to a static IP `10.42.1.102` with subnet `255.255.255.0`.

1. Open a terminal and enter `ssh agi@10.42.1.101` to log into the robot. The password please ask the after sales team.
2. Before use, first `vim /home/agi/app/config/arbitrator_config_base.json`. Press `i` once to enter insert mode, use the arrow keys to move the cursor to the last line `end_effector_pose_control`, and change `priority` to `40`. Then press `ESC` once, enter `:wq!` to save.
```json
{"source_topic": "/gdk/end_effector_pose_control","target_topic": "/wbc/end_effector_pose_control",  "msg_type": "genie_msgs.msg.pb.EndEffectorPoseControl",   "priority": 40,  "lock_decay_timeout_ms": 5000, "group": "control"}
```

3. Then enter:

```bash
source ~/app/env.sh
cd app/bin/
mode_switch --mode base
```

The terminal will display `switch to base success`.

4. Restart the robot.


## Deploying on a New Machine

After the robot restarts, first use the tablet to connect it to a network — internet access is needed to download a small tool.

```bash

# 1. Recreate the virtual environment on the robot
cd ~/G2_grape_235_v1
sudo apt install python3-venv
# Press Y

# Create a virtual environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Each time you open a terminal, run the commands below first to load the environment variables
cd ~/G2_grape_235_v1
source .venv/bin/activate
source ~/app/env.sh
source ~/app/gdk/scripts/env.sh
```

Then run the bundled script to verify. Once it passes, the run command will be displayed:

```bash
./install.sh            # Run on the target machine
```
Enter the command below to run:
```bash
python3 grasp_pipeline.py --stream true
# Press CTRL and C to exit the program
```

After a successful run, the terminal will display two IP addresses. Open either `http://10.41.1.101:5000` or `http://<robot-ip>:5000` in a browser. The robot IP is the second one printed in the terminal.
Open `http://<robot-ip>:5000` in a browser.

## Runtime Parameters

The constants at the top of `robot_control.py` control the main behavior. Please read the comments before changing them — most values were calibrated from real-robot logs:

| Constant | Current value | Description |
|---|---|---|
| `WORKSPACE` x upper limit | 0.95 | Relaxed from 0.85 on 2026-09-10; the old value would reject a large number of actually reachable targets |
| `ARM_SELECT_HYSTERESIS` | 0.06 | Within ±6 cm of the midline, keep the previously selected arm to avoid back-and-forth flipping; the y boundary of `WORKSPACE` **leaves this 0.06 m of extra margin inward toward the midline on each side** (2026-09-11). Otherwise, a target that persistently sits a few centimeters on one side of the midline but never crosses the 0.06 m threshold will be locked by hysteresis to one arm — and that arm's workspace happens to reject the target in those few centimeters. The two policies clash, showing up as repeated "outside workspace" + waist-scan failures with no grasp |
| `YAW_SETTLE_TOL_RAD` | 0.02 | Waist settling criterion (≈7 mm @ 0.35 m) |
| `_SCAN_STEPS` | `[0, ±0.2, ±0.35]` | Scan angles; the duplicate 0 step has been removed |
| `APPROACH_STANDOFF` | 0.18 | X distance from the target at which the gripper is held during side approach |
| `_TEST_GRIPPER_EE_POS` | `False` | When `True`, open/close additionally goes through the `move_ee_pos` channel |
| `_TORSO_C_HARD_MAX` | 1.0 | **Absolute** upper limit of `c`; not editable from the GUI either. 1.0 = the home pose verified on the real robot |
| `_TORSO_C_MIN` / `_TORSO_C_MAX` | 0.0 / 0.8 | **Default** values for the GUI's adjustable range; editable from the web page, but cannot exceed `_TORSO_C_HARD_MAX` |
| `_TORSO_LIMITS[0]` | `(-1.08, 0.0)` | Relaxed from -1.07 on 2026-09-11: the home value is -1.0793, and the old limit would silently clip c=1.0 by 0.0093 rad |

## Torso Height (crouch fraction `c`)

`c` is the "crouch fraction": `joint_i(c) = c * home_i` (body joints 1–4); joint 5 (waist yaw) is controlled separately.

- `c = 1.0` — today's home pose itself, verified on the real robot; this is the **lowest** pose.
- `c = 0.0` — all zeros, fully upright.
- Both ends are verified safe; the in-between range is linear interpolation in joint space.

**The upper and lower limits can be adjusted on the web page.** In the sidebar under "Torso Height":

- `AUTO` / `MANUAL` toggle. AUTO is driven by `_auto_torso_c()`, which selects `c` based on the target height; MANUAL is filled in by the operator.
- `c` input box — editable only in MANUAL mode.
- The two input boxes below are the **adjustable range** (`c_min` / `c_max`). Click `SET HEIGHT` to submit them together.
  The server validates: both values must fall within `[0, _TORSO_C_HARD_MAX]`, and `c_min < c_max`.
  If either condition fails, the entire request is rejected (400), and the server-side range **stays unchanged** — it cannot be polluted by a half-bad request.
- The value actually applied is fed back into the input box as `applied_c`: if you fill in an out-of-range value, it gets clamped to `c_max`, which is visible on the page.

**Behavior details:**

- Before each height change, `retract_arms()` is called first to retract the arms; if retraction fails it **raises an exception** (HTTP 500), and the torso will not move.
  Previously this returned silently, which was indistinguishable from "set successfully" on the web page.
- The `yaw` parameter of `set_torso_height()` defaults to `_torso_state["yaw"]`, not 0.0 ——
  changing the height will not silently rotate the waist back from the pose reached by scanning.
- Each time the torso moves, `_torso_epoch` is incremented by 1. The dictionary returned by `plan_grasp()` carries the epoch at which it was built;
  when `grasp_loop()` finds a mismatch it **replans** (in MANUAL mode the operator can change the height at runtime; the camera pose changes, and old coordinates become invalid).
- The web page **does not poll** `/api/torso`. Previously, polling once per second would flip MANUAL back to AUTO,
  appearing as "can't toggle manual mode".

**Known coordinate-sync gap:** the color frame, depth frame, and TF lookup are sampled at three different moments in time — no timestamp synchronization is performed.
This is currently the largest remaining error source, larger than `YAW_SETTLE_TOL_RAD` (0.02 rad ≈ 7 mm @ 0.35 m).

## Directory Contents

```
grasp_pipeline.py    Entry point: parses args, initializes GDK/YOLO/calibration, starts
robot_control.py     Robot/vision/planning/motion logic (main body)
web_server.py        Flask control panel (MJPEG stream + start/stop)
web/index.html       Control panel frontend
test_grippers.py     Dual-arm gripper self-check (run on the robot)
sensor/              Camera calibration JSON
yolov8n.pt           Default YOLO model
wheelhouse/          Offline wheels (aarch64 + any)
requirements.txt     Pinned-version dependencies (65 items)
test_log             Full log of one run
```

## Dependency Prerequisites

- **`agibot_gdk` (SDK v3.38) is not in this directory and cannot be installed via pip.** It is shipped with the robot's system image and only available on the robot body itself. Neither the `.venv` nor `wheelhouse/` in this directory contains it — this is intentional, not a missing item.
- The `.venv` in this directory is **aarch64** and **non-portable** (the shebang and `pyvenv.cfg` contain absolute paths to `/usr/bin`). **Do not copy `.venv` to a new machine**; rebuild it on the new machine.
- `wheelhouse/` — wheels.

## Known Issues

- The frame path in `web_server.py` (`_producer`) and the grasp detection in `robot_control.py` (`_detect_stable`) are **two independent detection paths** with different parameter sources (the former uses CLI defaults, the latter uses parameters POSTed by the browser); the two can be inconsistent.
- `check_target_lifted()` uses "is there still a same-class box within 70 px of the original pixel" to decide success, and can produce false positives: the log has shown cases where the gripper closed to `-0.000` (empty grasp) yet was reported as `LIFTED OK`.

## Ctrl+C / Interruption Behavior

Since 2026-09-11, when `grasp_pipeline.py` encounters Ctrl+C (or any escaped exception), it **no longer auto-returns to home**; it only prints a prompt and holds the current pose in place. Reasons:

- **Non-`--stream` direct-run mode**: `grasp_loop()` runs in the main thread, and Ctrl+C may land inside `_move_ee()` — the gripper may be in any pose at that moment, e.g., just doing a "lower"/"slide in" near the table.
  The old code unconditionally called `move_to_home(keep_torso=False)` in `finally` — this is a direct joint-space interpolation with no collision checking, throwing the arm from a pose close to the table straight back to home (while also straightening the waist). The path is not guaranteed to avoid the table.
- **`--stream` (web page) mode**: grasping runs in a background thread (`_run_worker`), and Ctrl+C only interrupts the main thread's own `while True: time.sleep(0.5)` idle loop — **it cannot stop the background thread**. The old code set `_RUN_STOP` and then waited at most 2 seconds; if the wait timed out, it still called `move_to_home()` from the main thread — if the background thread hadn't exited yet (its own cleanup was actually doing a safer `move_to_home(keep_torso=True)`), two threads would be issuing motion commands to the same robot at the same time. The commands would clash, showing up as "probing down hits the table".

Current behavior: in `finally`, `move_to_home()` is called only when the run finishes **normally, without exception or interruption**; on interruption, or when the background thread fails to confirm stop within the wait window, the robot simply holds its current pose and prints a prompt so a human can check and handle it manually.