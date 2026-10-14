# -*- coding: utf-8 -*-
"""qrnav_stream.py — QR 面板 JPEG 编码 + 位姿同步/实时估计后台循环

依赖: qrnav_config, qrnav_geometry
"""
import os
import io
import time
import tempfile
import contextlib

import numpy as np
import cv2

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_pose_lock, _qrnav_pose_box, _qrnav_pose_ver,
    _qrnav_pose_msg_box, _qrnav_img_w_box, _qrnav_img_h_box,
    _qrnav_jpg_buf,
)
from qrnav_geometry import pose_to_command, _compute_fill_ratio

def _encode_qr_panel_jpeg(frame, pose):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = _QR_NAV_CFG["STREAM_SCALE"]
    nw, nh = int(w * scale), int(h * scale)
    if nw > 0 and nh > 0 and (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)

    if pose is not None:
        if "image_points" in pose:
            pts = np.array(pose["image_points"], dtype=int).reshape(-1, 2)
            if len(pts) == 4:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        cmd = pose_to_command(pose)
        txt1 = f"d={cmd['distance']:.2f}m  lat={cmd['lateral_x']:+.2f}m"
        cv2.putText(frame, txt1, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        txt2 = f"fill={cmd['fill_ratio']*100:.1f}%  yaw={cmd['qr_yaw']:.1f}"
        cv2.putText(frame, txt2, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if cmd.get("zoom_needed"):
            cv2.putText(frame, f"ZOOM {cmd['zoom_factor']:.1f}x",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    else:
        cv2.putText(frame, "no QR", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    ok, jpg = cv2.imencode(".jpg", frame,
                           [cv2.IMWRITE_JPEG_QUALITY,
                            _QR_NAV_CFG["STREAM_QUALITY"]])
    return jpg.tobytes() if ok else None


def _qrnav_pose_sync_loop(estimator, stop_evt):
    fps = _QR_NAV_CFG["STREAM_FPS"]
    interval = 1.0 / max(1.0, fps)
    next_t = time.time()
    while not stop_evt.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
            continue
        next_t = now + interval
        pose = getattr(estimator, "latest_pose_data", None)
        frame = getattr(estimator, "current_frame", None)
        img_w = int(frame.shape[1]) if frame is not None else 0
        img_h = int(frame.shape[0]) if frame is not None else 0
        with _qrnav_pose_lock:
            _qrnav_pose_box[0] = pose
            _qrnav_img_w_box[0] = img_w
            _qrnav_img_h_box[0] = img_h
            if pose is not None:
                fill_disp = _compute_fill_ratio(pose, img_w)
                _qrnav_pose_msg_box[0] = (
                    f"距 {pose.get('distance', 0):.2f}m  "
                    f"占比 {fill_disp*100:.1f}%")
            _qrnav_pose_ver[0] += 1
        if frame is not None:
            jpg = _encode_qr_panel_jpeg(frame, pose)
            if jpg:
                _qrnav_jpg_buf.update(jpg)


def _qrnav_ascii_tmp(name="_qrnav_live_pose.jpg"):
    cands = [os.getcwd(),
             os.environ.get("TEMP", ""), os.environ.get("TMP", ""),
             tempfile.gettempdir(), "C:\\Windows\\Temp", "D:\\"]
    for d in cands:
        if not d:
            continue
        p = os.path.join(d, name)
        if any(ord(c) > 127 for c in p):
            continue
        try:
            os.makedirs(d, exist_ok=True)
            with open(p, "wb"):
                pass
            os.remove(p)
            return p
        except Exception:
            continue
    return os.path.join(tempfile.gettempdir(), name)


def _qrnav_live_pose_loop(estimator, stop_evt):
    fps = float(_QR_NAV_CFG.get("LIVE_POSE_FPS", 0.0) or 0.0)
    if fps <= 0:
        return
    interval = 1.0 / max(0.2, min(30.0, fps))
    ema_a = float(_QR_NAV_CFG.get("LIVE_POSE_EMA", 0.55))
    tmp_path = _qrnav_ascii_tmp("_qrnav_live_pose.jpg")
    ema_tvec = None
    ema_pos = None
    next_t = 0.0
    while not stop_evt.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(min(0.05, max(0.0, next_t - now)))
            continue
        next_t = now + interval
        try:
            frame = getattr(estimator, "current_frame", None)
            if frame is None:
                continue
            if getattr(estimator, "_est_busy", False):
                continue
            if not cv2.imwrite(tmp_path, np.ascontiguousarray(frame)):
                continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok, pd, _ = estimator.estimate_pose_from_image(tmp_path,
                                                               debug=False)
            if not ok or pd is None:
                continue
            if "tvec" in pd:
                tv = np.array(pd["tvec"], dtype=float).flatten()
                ema_tvec = tv if ema_tvec is None else (ema_a * tv + (1 - ema_a) * ema_tvec)
                pd["tvec"] = ema_tvec.reshape(-1, 1).tolist()
            if "camera_position" in pd:
                cp = np.array(pd["camera_position"], dtype=float).flatten()
                ema_pos = cp if ema_pos is None else (ema_a * cp + (1 - ema_a) * ema_pos)
                pd["camera_position"] = ema_pos.tolist()
                pd["distance"] = float(np.linalg.norm(ema_pos))
            pd["live"] = True
            try:
                estimator.latest_pose_data = pd
            except Exception:
                pass
            with _qrnav_pose_lock:
                _qrnav_pose_box[0] = pd
                _qrnav_pose_msg_box[0] = f"实时 距 {pd.get('distance', 0):.2f}m"
                _qrnav_pose_ver[0] += 1
        except Exception:
            pass
    try:
        os.remove(tmp_path)
    except Exception:
        pass
