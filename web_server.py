#!/usr/bin/env python3
"""
web_server.py — Flask control panel for the G02 grasp pipeline.

Owns the Flask app, MJPEG stream producer, run-state tracking, and the
background worker thread that drives dry_run()/grasp_loop() on behalf of the
browser GUI (web/index.html). All robot/camera/perception/motion logic is
imported from robot_control.py — this file never touches GDK/YOLO directly
beyond reading camera frames for the stream mosaic.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import agibot_gdk as gdk

from flask import Flask, Response, send_from_directory, jsonify, request

import robot_control as rc

_YOLO_OK = rc._YOLO_OK
_DEPTH_NEAR_MM = rc.DEPTH_MIN_MM
_DEPTH_FAR_MM  = 2000


# ══════════════════════════════════════════════════════════════════════════════
# Run-state (background worker driven by the browser's Start/Abort buttons)
# ══════════════════════════════════════════════════════════════════════════════

_STREAM_JPEG: bytes = b""
_STREAM_LOCK = threading.Lock()

_RUN_LOCK  = threading.Lock()
_RUN_STATE = {
    "running": False, "mode": None, "phase": "idle", "message": "",
    "loop": 0, "arm": None, "target": None, "gripper": None,
    "started_at": None, "ended_at": None,
}
_RUN_STOP: threading.Event | None = None


def _run_state_update(**kwargs) -> None:
    with _RUN_LOCK:
        _RUN_STATE.update(kwargs)


def _run_state_snapshot() -> dict:
    with _RUN_LOCK:
        return dict(_RUN_STATE)


def _run_worker(mode: str, params: dict,
                camera, robot, tf_module, model,
                cam_to_parent, intr_depth,
                stop_event: threading.Event) -> None:
    _run_state_update(
        running=True, mode=mode, phase="starting", message="",
        loop=0, arm=params["arm"], target=params["target"],
        gripper=params["gripper"], started_at=time.time(), ended_at=None,
    )
    try:
        _run_state_update(phase="running", message="operation in progress")

        if mode == "dry_run":
            ok = rc.dry_run(
                camera, tf_module, model,
                rc._CAM_COLOR, rc._CAM_DEPTH,
                rc._TF_PARENT, cam_to_parent, intr_depth,
                params["target"], params["confidence"], params["samples"],
                params["arm"], attempts=5, stop_event=stop_event,
            )
            _run_state_update(
                phase=("done" if ok else "error"),
                message=("dry-run complete" if ok else "dry-run failed — see server log"),
            )
        else:
            def _on_loop(loop_idx: int) -> None:
                _run_state_update(loop=loop_idx, message=f"loop {loop_idx}")

            rc.grasp_loop(
                camera, robot, tf_module, model,
                cam_to_parent, intr_depth,
                params["gripper"], params["target"], params["confidence"],
                params["samples"], arm=params["arm"],
                stop_event=stop_event, on_loop=_on_loop,
            )
            _run_state_update(
                phase=("stopped" if stop_event.is_set() else "done"),
                message="grasp loop stopped" if stop_event.is_set() else "grasp loop finished",
            )
    except Exception as e:
        _run_state_update(phase="error", message=f"error: {e}")
    finally:
        _run_state_update(running=False, ended_at=time.time())


# ══════════════════════════════════════════════════════════════════════════════
# Visualization helpers (MJPEG mosaic)
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_W, _PANEL_H = 480, 360
_LABEL_H = 28
_DET_COLORS = {True: (56, 189, 248), False: (245, 158, 11)}   # target=cyan, obstacle=amber


def _annotate(frame: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        color = _DET_COLORS.get(d.get("is_target", False), (200, 200, 200))
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.putText(out, label, (d["x1"], max(0, d["y1"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth, _DEPTH_NEAR_MM, _DEPTH_FAR_MM).astype(np.float32)
    norm = ((clipped - _DEPTH_NEAR_MM) / max(1, (_DEPTH_FAR_MM - _DEPTH_NEAR_MM)) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)


def _make_panel(frame: np.ndarray | None, title: str,
                w: int = _PANEL_W, h: int = _PANEL_H, label_h: int = _LABEL_H) -> np.ndarray:
    panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    panel[:label_h, :, :] = (24, 27, 33)
    cv2.putText(panel, title, (10, label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 232, 235), 1, cv2.LINE_AA)
    if frame is not None:
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (nw, nh))
        y0 = label_h + (h - nh) // 2
        x0 = (w - nw) // 2
        panel[y0:y0 + nh, x0:x0 + nw] = resized
    else:
        cv2.putText(panel, "no signal", (w // 2 - 45, label_h + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 125, 135), 1, cv2.LINE_AA)
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# Flask app + MJPEG stream + control API
# ══════════════════════════════════════════════════════════════════════════════

def start_stream(camera, model, robot, tf_module, cam_to_parent, intr_depth,
                 target_class, gripper_type, conf_thresh, samples, arm,
                 port: int = 5000) -> threading.Event:
    app = Flask(__name__)
    stop = threading.Event()
    web_dir = Path(__file__).parent / "web"

    def _producer() -> None:
        global _STREAM_JPEG
        while not stop.is_set():
            try:
                try:
                    obj = camera.get_latest_image(gdk.CameraType.kHeadColor, 500.0)
                    f1  = cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)
                    f1  = _annotate(f1, rc._detect(model, f1, target_class, conf_thresh))
                except Exception:
                    f1 = None

                try:
                    obj  = camera.get_latest_image(gdk.CameraType.kHeadDepth, 500.0)
                    dep  = np.frombuffer(obj.data, np.uint16).reshape(obj.height, obj.width)
                    f2   = _colorize_depth(dep)
                    valid = dep[(dep >= _DEPTH_NEAR_MM) & (dep <= _DEPTH_FAR_MM)]
                    if valid.size > 0:
                        cv2.putText(f2, f"min {valid.min()}mm  max {valid.max()}mm",
                                    (6, f2.shape[0] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                except Exception:
                    f2 = None

                try:
                    obj = camera.get_latest_image(gdk.CameraType.kHandRightColor, 500.0)
                    f3  = cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)
                    f3  = _annotate(f3, rc._detect(model, f3, target_class, conf_thresh))
                except Exception:
                    f3 = None

                try:
                    obj = camera.get_latest_image(gdk.CameraType.kHandLeftColor, 500.0)
                    f4  = cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)
                    f4  = _annotate(f4, rc._detect(model, f4, target_class, conf_thresh))
                except Exception:
                    f4 = None

                mosaic = np.hstack([
                    _make_panel(f1, "Head Color (YOLO)"),
                    _make_panel(f2, f"Head Depth [{_DEPTH_NEAR_MM}-{_DEPTH_FAR_MM} mm]"),
                    _make_panel(f3, "Hand-R Color (YOLO)"),
                    _make_panel(f4, "Hand-L Color (YOLO)"),
                ])
                ok, buf = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with _STREAM_LOCK:
                        _STREAM_JPEG = buf.tobytes()
            except Exception as e:
                print(f"[stream producer] {e}")
            time.sleep(0.05)

    @app.route("/")
    def _index():
        return send_from_directory(str(web_dir), "index.html")

    @app.route("/stream")
    def _stream():
        def _gen():
            while True:
                with _STREAM_LOCK:
                    frame = _STREAM_JPEG
                if frame:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.033)
        return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/config")
    def _config():
        target_choices = sorted(model.names.values()) if _YOLO_OK else [target_class]
        return jsonify({
            "target_choices": target_choices,
            "gripper_choices": rc._GRIPPER_CHOICES,
            "arm_choices": rc._ARM_CHOICES,
            "defaults": {
                "target": target_class, "gripper": gripper_type, "arm": arm,
                "confidence": conf_thresh, "samples": samples,
            },
        })

    @app.route("/api/status")
    def _status():
        return jsonify(_run_state_snapshot())

    @app.route("/api/torso")
    def _torso_get():
        # c_min / c_max are the GUI's *adjustable* range (so the operator can
        # tighten it); c_hard_max is the ceiling they may not raise past.
        return jsonify(dict(rc._torso_state,
                            c_min=rc._TORSO_C_MIN,
                            c_max=rc._TORSO_C_MAX,
                            c_hard_max=rc._TORSO_C_HARD_MAX))

    @app.route("/api/torso", methods=["POST"])
    def _torso_post():
        if _run_state_snapshot()["running"]:
            return jsonify({"error": "operation already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        new_mode = body.get("mode", rc._torso_state["mode"])
        if new_mode not in ("auto", "manual"):
            return jsonify({"error": f"invalid mode '{new_mode}'"}), 400

        # Optional range override from the GUI. Validate the whole request
        # BEFORE writing any of it: a rejected request must leave the server's
        # range exactly as it found it, or one bad POST poisons every later one.
        new_min, new_max = rc._TORSO_C_MIN, rc._TORSO_C_MAX
        for key, target in (("c_min", "min"), ("c_max", "max")):
            if key not in body:
                continue
            try:
                v = float(body[key])
            except (TypeError, ValueError):
                return jsonify({"error": f"invalid '{key}'"}), 400
            if v < 0.0 or v > rc._TORSO_C_HARD_MAX:
                return jsonify({"error": f"'{key}' {v} outside [0, "
                                         f"{rc._TORSO_C_HARD_MAX}]"}), 400
            if target == "min":
                new_min = v
            else:
                new_max = v
        if new_min >= new_max:
            return jsonify({"error": f"c_min ({new_min}) must be < "
                                     f"c_max ({new_max})"}), 400
        rc._TORSO_C_MIN, rc._TORSO_C_MAX = new_min, new_max

        rc._torso_state["mode"] = new_mode

        # Only drive the torso when an actual height is supplied. A bare
        # {"mode": "manual"} just switches modes — the operator then sets c.
        applied = None
        if new_mode == "manual" and "c" in body:
            try:
                c = float(body["c"])
            except (TypeError, ValueError):
                return jsonify({"error": "invalid 'c'"}), 400
            if c > rc._TORSO_C_HARD_MAX:
                return jsonify({"error": f"c {c} exceeds hard max "
                                         f"{rc._TORSO_C_HARD_MAX}"}), 400
            try:
                applied, converged = rc.set_torso_height(robot, c)
            except RuntimeError as e:
                return jsonify({"error": str(e)}), 500
            if not converged:
                print("[torso] yaw did not converge during height set")

        return jsonify(dict(rc._torso_state,
                            c_min=rc._TORSO_C_MIN,
                            c_max=rc._TORSO_C_MAX,
                            c_hard_max=rc._TORSO_C_HARD_MAX,
                            applied_c=applied))

    @app.route("/api/start", methods=["POST"])
    def _start():
        global _RUN_STOP
        if _run_state_snapshot()["running"]:
            return jsonify({"error": "operation already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        mode = body.get("mode", "dry_run")
        if mode not in ("dry_run", "live"):
            return jsonify({"error": f"invalid mode '{mode}'"}), 400

        params = {
            "target":     body.get("target", target_class),
            "gripper":    body.get("gripper", gripper_type),
            "arm":        body.get("arm", arm),
            "confidence": float(body.get("confidence", conf_thresh)),
            "samples":    int(body.get("samples", samples)),
        }
        if params["gripper"] not in rc._GRIPPER_CHOICES:
            return jsonify({"error": f"invalid gripper '{params['gripper']}'"}), 400
        if params["arm"] not in rc._ARM_CHOICES:
            return jsonify({"error": f"invalid arm '{params['arm']}'"}), 400

        _RUN_STOP = threading.Event()
        threading.Thread(
            target=_run_worker,
            args=(mode, params, camera, robot, tf_module, model, cam_to_parent, intr_depth, _RUN_STOP),
            daemon=True,
        ).start()
        return jsonify({"ok": True})

    @app.route("/api/abort", methods=["POST"])
    def _abort():
        if not _run_state_snapshot()["running"]:
            return jsonify({"error": "no operation running"}), 409
        if _RUN_STOP is not None:
            _RUN_STOP.set()
        _run_state_update(phase="aborting", message="abort requested")
        return jsonify({"ok": True})

    threading.Thread(target=_producer, daemon=True).start()
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False),
        daemon=True,
    ).start()

    return stop
