#!/usr/bin/env python3
"""
grasp_pipeline.py — G02 Desktop Object Grasping Pipeline: entry point.

Robot/camera/perception/motion logic lives in robot_control.py.
The Flask control panel (camera stream + start/abort GUI) lives in web_server.py.
This file only parses CLI args, initialises GDK/YOLO/calibration, and wires the
two together — mirroring the original single-file script's behavior exactly.
"""
import argparse
import time
from pathlib import Path

import agibot_gdk as gdk

import robot_control as rc

try:
    import web_server as ws
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False


def main():
    parser = argparse.ArgumentParser(description="G02 Desktop Object Grasping Pipeline")
    parser.add_argument("--arm",        choices=rc._ARM_CHOICES, default="auto",
                        help="arm to use: auto (default, head-cam decides), right, or left")
    parser.add_argument("--gripper",    choices=rc._GRIPPER_CHOICES,
                        default="omnipicker")
    parser.add_argument("--model",      default=rc.DEFAULT_MODEL_PATH)
    parser.add_argument("--target",     default="bottle",
                        help="YOLO class name to grasp (e.g. bottle, cup)")
    parser.add_argument("--confidence", type=float, default=rc.CONFIDENCE_THRESHOLD)
    parser.add_argument("--samples",    type=int,   default=rc.DETECTION_SAMPLES)
    parser.add_argument("--sensor-dir", type=Path,  default=rc.DEFAULT_SENSOR_DIR)
    parser.add_argument("--dry-run",    action="store_true",
                        help="perception + planning only, no motion "
                             "(ignored together with --stream — use the browser's "
                             "mode toggle instead)")
    parser.add_argument("--stream",     default=True,
                        help="start Flask control panel (camera + start/abort GUI) on --stream-port")
    parser.add_argument("--stream-port", type=int, default=5000)
    args = parser.parse_args()

    if not rc._YOLO_OK:
        raise ImportError("ultralytics not installed — pip install ultralytics")

    # ── Load calibration (head camera always) ────────────────────────────────
    sensor_dir    = args.sensor_dir
    intr_depth    = rc._load_intrinsic(sensor_dir / rc._INTR_DEPTH_FILE)
    cam_to_parent = rc._load_extrinsic(sensor_dir / rc._EXTR_FILE)
    print(f"Calibration loaded  camera=head  tf_parent={rc._TF_PARENT}")

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print(f"Loading YOLO model: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)
    rc._patch_yolo_fuse(model)
    valid_names = set(model.names.values())
    if args.target not in valid_names:
        print(f"[warning] '{args.target}' not in model classes; valid: "
              f"{sorted(valid_names)[:20]} ...")

    # ── GDK init ──────────────────────────────────────────────────────────────
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init() failed")
    time.sleep(1)

    robot       = None   # hoisted so finally can reach it for home reset
    camera      = None
    stream_stop = None   # stop event for the MJPEG producer thread
    interrupted = False  # True on Ctrl+C or any escaping exception — gates the
                         # automatic home reset in finally (see the note there)
    try:
        robot  = gdk.Robot()
        camera = gdk.Camera()
        tf     = gdk.TF()
        time.sleep(2)
        print("GDK ready")

        if args.stream:
            if not _FLASK_OK:
                print("[stream] Flask/web_server not available — pip install flask")
            else:
                stream_stop = ws.start_stream(camera, model, robot, tf,
                                              cam_to_parent, intr_depth,
                                              args.target, args.gripper, args.confidence,
                                              args.samples, args.arm, args.stream_port)
                print(f"Control panel ready → http://<robot-ip>:{args.stream_port}  "
                      f"(waiting for Start in the browser; Ctrl+C to quit)")
                while True:
                    time.sleep(0.5)
        elif args.dry_run:
            rc.dry_run(camera, tf, model,
                      rc._CAM_COLOR, rc._CAM_DEPTH,
                      rc._TF_PARENT, cam_to_parent, intr_depth,
                      args.target, args.confidence, args.samples, args.arm)
        else:
            rc.grasp_loop(camera, robot, tf, model,
                         cam_to_parent, intr_depth,
                         args.gripper, args.target, args.confidence, args.samples,
                         args.arm)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — holding current pose (see [finally] note below).")
    except Exception as e:
        interrupted = True
        print(f"[error] {e}"); raise
    finally:
        # Whether the background grasp worker (--stream mode) is still moving the
        # robot when we get here. Give it a window to notice _RUN_STOP and unwind
        # through its own shutdown path (grasp_loop's stop_event tail already does
        # move_to_home(keep_torso=True) — the safe version, called only between
        # waypoints, never mid-motion).
        worker_still_running = False
        if _FLASK_OK and ws._RUN_STOP is not None:
            ws._RUN_STOP.set()
            for _ in range(40):        # wait up to ~4s for the worker to unwind
                worker_still_running = ws._run_state_snapshot()["running"]
                if not worker_still_running:
                    break
                time.sleep(0.1)
        if stream_stop is not None:
            stream_stop.set()        # tell producer thread to exit
            time.sleep(0.25)         # let it finish its current frame fetch
        if robot is not None:
            if interrupted or worker_still_running:
                # Do NOT call move_to_home() here. Two distinct hazards, both of
                # which showed up as "the arm hits the table on the way down":
                #  1. Interrupted mid-motion (Ctrl+C in non-stream/CLI mode, where
                #     grasp_loop runs on this same thread): the arm can be at any
                #     arbitrary pose — mid-lower, mid-slide-in, right at the table.
                #     move_to_home() is a single blind joint-space interpolation
                #     straight to home with no collision check; from a low, near-
                #     table pose that path is not guaranteed to clear the table,
                #     especially combined with keep_torso=False snapping the torso
                #     upright at the same time.
                #  2. --stream mode: the grasp worker runs on its own thread, so
                #     Ctrl+C here only interrupts this thread's idle wait loop, not
                #     the worker. If the worker hasn't confirmed stopping within the
                #     wait above, it may *still* be mid-shutdown (or mid-grasp) on
                #     the robot right now — calling move_to_home() from this thread
                #     races a second motion command against whatever it's doing.
                # Either way, holding the last commanded pose and leaving cleanup to
                # a human (or to the worker's own already-safe shutdown) is the only
                # move here that's guaranteed not to make things worse.
                print("[finally] holding current pose — NOT auto-returning home "
                      "(interrupted mid-operation or the grasp worker hasn't "
                      "confirmed it stopped). Check the robot/table clearance "
                      "manually before restarting.")
            else:
                try:
                    rc.move_to_home(robot)
                    rc.open_gripper(robot, args.gripper, args.arm)
                except Exception as e:
                    print(f"[finally] home reset failed: {e}")
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        try:
            gdk.gdk_release()
            print("GDK released")
        except Exception:
            pass


if __name__ == "__main__":
    main()
