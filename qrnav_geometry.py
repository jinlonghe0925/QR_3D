# -*- coding: utf-8 -*-
"""qrnav_geometry.py — 位姿/几何换算 (纯计算, 不发任何硬件指令)

职责: pose → cmd 字典 / 云台位置作为机器狗位置 / fill 占比
      / 真实 FOV 缓存 + camera_matrix 重建 / pose 人话报告
依赖: qrnav_config
"""
import math
import numpy as np
import cv2

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_estimator_holder,
    _qrnav_pose_lock, _qrnav_img_w_box,
    _qrnav_focal_cache, _qrnav_focal_cache_ttl, _qrnav_last_known_zoom,
    _qrnav_fov_sensor_w_mm,
)

# ============================================================
#  Helpers (与 V1 保持一致)
# ============================================================
def _get_gimbal_angles():
    try:
        est = _qrnav_estimator_holder[0]
        if est is None:
            return 0.0, 0.0, 0.0
        gimbal = getattr(est, "gimbal", None)
        if gimbal is None:
            return 0.0, 0.0, 0.0
        ya, yp, yr = gimbal.get_current_angles()
        return float(ya), float(yp), float(yr)
    except Exception:
        return 0.0, 0.0, 0.0


def _compute_fill_ratio(pose, img_w):
    if pose is None or img_w <= 0:
        return 0.0
    img_pts = pose.get("image_points")
    if img_pts is None or len(img_pts) < 4:
        return 0.0
    try:
        pts = np.asarray(img_pts, dtype=float).reshape(-1, 2)
        if len(pts) < 4:
            return 0.0
        s01 = float(np.linalg.norm(pts[0] - pts[1]))
        s12 = float(np.linalg.norm(pts[1] - pts[2]))
        s23 = float(np.linalg.norm(pts[2] - pts[3]))
        s30 = float(np.linalg.norm(pts[3] - pts[0]))
        return (s01 + s12 + s23 + s30) / 4.0 / float(img_w)
    except Exception:
        return 0.0


def _qr_position_in_camera(pose):
    """二维码原点在相机系的位置；PnP 满足 camera_position = -R.T @ tvec。"""
    if pose.get("tvec") is not None:
        position = np.asarray(pose["tvec"], dtype=float).reshape(3)
    else:
        camera_position = np.asarray(pose["camera_position"], dtype=float).reshape(3)
        if pose.get("rmat") is not None:
            rotation = np.asarray(pose["rmat"], dtype=float).reshape(3, 3)
        else:
            rotation, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=float).reshape(3, 1))
        position = -rotation @ camera_position
    if not np.isfinite(position).all():
        raise ValueError("二维码相机系位置不是有限数值")
    return position


def _camera_pos_to_base(cam_pos, gimbal_yaw_deg, pose=None):
    """将传入 pose 的二维码位置按 yaw 投影到基座方向，不读取 latest_pose_data。

    基座轴约定为 x 右、y 下、z 前；当前仅补偿云台 yaw。
    cam_pos 是相机在二维码系的位置，缺少旋转矩阵时不能直接取负当 tvec。
    """
    yaw_sign = _QR_NAV_CFG.get("YAW_SIGN", +1)
    yaw = np.radians(gimbal_yaw_deg) * yaw_sign
    cy, sy = np.cos(yaw), np.sin(yaw)
    try:
        tx, ty, tz = _qr_position_in_camera(pose or {"camera_position": cam_pos})
    except (KeyError, TypeError, ValueError, cv2.error):
        return None, None, None
    base_x = cy * tx + sy * tz
    base_z = -sy * tx + cy * tz
    base_y = ty
    return float(base_x), float(base_y), float(base_z)


def _qr_size_est():
    try:
        return float(getattr(_qrnav_estimator_holder[0], "qr_size", 0.0) or 0.0)
    except Exception:
        return 0.0


def _qr_size_resolved():
    return _qr_size_est() or float(_QR_NAV_CFG.get("QR_SIZE_M", 0.12))


def pose_to_command(pose):
    """直接使用 camera_position 作为狗的位置，横移量取其 X，不做基座投影。"""
    if pose is None:
        return {
            "ready": False, "distance": None, "lateral_x": None,
            "camera_y": None, "camera_z": None,
            "lateral_world": None, "forward_world": None,
            "gimbal_yaw_deg": None,
            "turn_dir": "中", "turn_sign": 0, "rotate_hint_deg": 0.0,
            "fill_ratio": None, "zoom_needed": False, "zoom_factor": 1.0,
            "angle_to_plane": None, "qr_yaw": 0.0, "qr_pitch": 0.0,
            "decoded": "", "hint": "等待 QR 检测…",
        }

    cfg = _QR_NAV_CFG
    if "camera_position" in pose:
        cam = np.array(pose["camera_position"], dtype=float).flatten()
    else:
        rvec = np.array(pose["rvec"], dtype=float).reshape(3, 1)
        tvec = np.array(pose["tvec"], dtype=float).reshape(3, 1)
        rmat, _ = cv2.Rodrigues(rvec)
        cam = (-rmat.T @ tvec).flatten()
    x, y, z = float(cam[0]), float(cam[1]), float(cam[2])
    dist = float(np.linalg.norm(cam))
    lateral_x = x

    if pose.get("gimbal_yaw_deg") is not None:
        gimbal_yaw_deg = float(pose["gimbal_yaw_deg"])
        gimbal_pitch_deg = float(pose.get("gimbal_pitch_deg", 0.0))
    else:
        gimbal_yaw_deg, gimbal_pitch_deg, _ = _get_gimbal_angles()
    # 暂时将云台位置视为狗的位置，数值与终端打印的 camera_position 完全一致。
    # 保留旧 JSON 字段名供兼容调用者使用，字段值直接对应相机位置 X/Z。
    lateral_world = x
    forward_world = z

    if lateral_world is None:
        turn_dir, turn_sign = "?", 0
    else:
        lat_eff = lateral_world * cfg.get("LATERAL_SIGN", +1)
        if   lat_eff >  0.02: turn_dir, turn_sign = "右", -1
        elif lat_eff < -0.02: turn_dir, turn_sign = "左", +1
        else:                 turn_dir, turn_sign = "中",  0

    if lateral_world is not None and abs(lateral_world) > cfg["ALIGN_LAT"] * 1.5:
        rotate_hint_deg = -lateral_world * 30.0
        rotate_hint_deg = max(-15.0, min(15.0, rotate_hint_deg))
    else:
        rotate_hint_deg = 0.0

    with _qrnav_pose_lock:
        img_w = _qrnav_img_w_box[0]
    fill = _compute_fill_ratio(pose, img_w)
    zoom_needed = fill < cfg["ZOOM_TARGET"]
    if zoom_needed and fill > 1e-4:
        zoom_factor = (cfg["ZOOM_TARGET"] / fill) ** 0.5
        zoom_factor = min(zoom_factor, 5.0)
    else:
        zoom_factor = 1.0

    if "rmat" in pose:
        rmat = np.array(pose["rmat"], dtype=float)
        Z_axis = rmat[:, 2]
        cos_a = float(np.clip(abs(float(Z_axis[2])), -1.0, 1.0))
        angle_to_plane = float(np.degrees(np.arccos(cos_a)))
    else:
        angle_to_plane = 0.0

    if lateral_world is not None and abs(lateral_world) > cfg["ALIGN_LAT"]:
        step = abs(x)
        hint = f"→ 向{turn_dir}横移目标 {step:.3f}m (相机位置 X={x:+.3f}m，作为狗的位置)"
    elif zoom_needed:
        lx_str = f"{lateral_world:+.3f}" if lateral_world is not None else "n/a"
        hint = (f"🔍 需要变倍 {zoom_factor:.2f}x (QR 占比 {fill*100:.1f}% < "
                f"{cfg['ZOOM_TARGET']*100:.0f}%, 相机位置 X={lx_str}m)")
    else:
        hint = (f"✓ 已对齐 (相机位置 |X|={abs(x):.3f}m ≤ {cfg['ALIGN_LAT']}m, "
                f"QR 占比 {fill*100:.1f}%, 云台 yaw={gimbal_yaw_deg:.1f}°)")

    return {
        "ready": True, "distance": dist, "lateral_x": lateral_x,
        "camera_y": y, "camera_z": z,
        "position_source": "camera_position",
        "lateral_world": lateral_world,
        "forward_world": forward_world,
        "gimbal_yaw_deg": gimbal_yaw_deg,
        "gimbal_pitch_deg": gimbal_pitch_deg,
        "turn_dir": turn_dir, "turn_sign": turn_sign,
        "rotate_hint_deg": rotate_hint_deg,
        "fill_ratio": fill, "zoom_needed": zoom_needed, "zoom_factor": zoom_factor,
        "angle_to_plane": angle_to_plane,
        "qr_yaw": float(pose.get("yaw", 0.0)),
        "qr_pitch": float(pose.get("pitch", 0.0)),
        "decoded": str(pose.get("decoded", "")),
        "qr_size":      _qr_size_resolved(),
        "qr_size_src":  ("estimator ✓" if _qr_size_est() else "常量0.12"),
        "qr_size_pose": float(pose.get("qr_size", 0.0) or 0.0),
        "hint": hint,
    }


def _query_real_fov_from_gimbal(timeout=1.5):
    """同步调用云台 query_focal_length() 拿真实 FOV，失败返回 None。

    外层不能另起短超时线程：底层查询自身会等待响应，外层提前返回会留下仍在
    运行的查询，与下一次 Pose 查询争用同一个 Event 和返回值。
    """
    est = _qrnav_estimator_holder[0]
    if est is None:
        return None
    gimbal = getattr(est, "gimbal", None)
    if gimbal is None or not hasattr(gimbal, "query_focal_length"):
        return None
    try:
        try:
            values = gimbal.query_focal_length(timeout=timeout)
        except TypeError:
            values = gimbal.query_focal_length()
        if not values:
            return None
        fov_x, fov_y, fx_mm, fy_mm = values
        if fov_x and fov_x > 0:
            return (float(fov_x), float(fov_y or 0),
                    float(fx_mm or 0), float(fy_mm or 0))
    except Exception:
        return None
    return None


def _rebuild_camera_matrix(img_w, img_h, fov_x_deg, fov_y_deg):
    """用真实 FOV 重建 camera_matrix (跟参照主页面 _init_camera 一致)"""
    est = _qrnav_estimator_holder[0]
    if est is None or img_w <= 0 or img_h <= 0:
        return None
    try:
        fx = img_w / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        fy = img_h / (2.0 * math.tan(math.radians(fov_y_deg) / 2.0))
        cam = np.array([
            [fx, 0.0, img_w / 2.0],
            [0.0, fy, img_h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        est.camera_matrix = cam
        return cam
    except Exception:
        return None


def _ensure_focal_for_zoom(cur_zoom, img_w, img_h, force_query=False):
    """按当前 zoom 拿到真实 (fov_x, fov_y), 同步刷新 est.camera_matrix

    优先级:
      1) cache 命中且未过期 (60s) —— 直接返
      2) pose.camera_matrix 反算 —— zoom 不变时
      3) 实时 query gimbal, 缓存并刷 est.camera_matrix
      4) 退回 FOV_X / zoom 公式 (最后兜底)
    """
    cfg = _QR_NAV_CFG
    est = _qrnav_estimator_holder[0]
    if est is None or img_w <= 0 or img_h <= 0:
        fx = float(cfg.get("FOV_X", 60.0))
        return fx, fx * (img_h / max(1, img_w))
    cur_zoom = max(1.0, round(float(cur_zoom or 1.0), 2))
    now = time.time()

    # 1) cache 命中
    cached = _qrnav_focal_cache.get(cur_zoom)
    if not force_query and cached and (now - cached[2] < _qrnav_focal_cache_ttl):
        return cached[0], cached[1]

    # 2) 已有 camera_matrix 且 zoom 没变 —— 用现成 fx 反算
    last_zoom = _qrnav_last_known_zoom[0]
    cam = getattr(est, "camera_matrix", None)
    if (not force_query and cam is not None and last_zoom == cur_zoom
            and float(cam[0, 0]) > 0):
        try:
            fx = float(cam[0, 0])
            fov_x = 2.0 * math.degrees(math.atan(
                _qrnav_fov_sensor_w_mm / (2.0 * fx
                * (_qrnav_fov_sensor_w_mm / float(img_w)))))
            # 上面是真实几何; 但 cam 的 fx 是像素单位, 转回 FOV 用 W/2 / fx
            fov_x = 2.0 * math.degrees(math.atan(float(img_w) / (2.0 * fx)))
            fy = float(cam[1, 1]) if cam.shape[0] > 1 and cam.shape[1] > 1 else fx
            fov_y = 2.0 * math.degrees(math.atan(float(img_h) / (2.0 * fy)))
            _qrnav_focal_cache[cur_zoom] = (fov_x, fov_y, now)
            return fov_x, fov_y
        except Exception:
            pass

    # 3) 实时查云台
    res = _query_real_fov_from_gimbal()
    if res is not None:
        fov_x, fov_y, fx_mm, fy_mm = res
        fov_y = 2.0 * math.degrees(math.atan(
            math.tan(math.radians(fov_x) / 2.0)
            * float(img_h) / float(img_w)))
        _qrnav_focal_cache[cur_zoom] = (fov_x, fov_y, now)
        # ★ 同步刷新 est.camera_matrix, 让后续 PnP 用对的内参
        _rebuild_camera_matrix(img_w, img_h, fov_x, fov_y)
        _qrnav_last_known_zoom[0] = cur_zoom
        return fov_x, fov_y

    # 4) 兜底: 经验公式
    fov_x_base = float(cfg.get("FOV_X", 60.0))
    fov_x = fov_x_base / cur_zoom
    fov_y = fov_x * (float(img_h) / float(img_w))
    if force_query:
        _qrnav_focal_cache[cur_zoom] = (fov_x, fov_y, now)
    return fov_x, fov_y


def _maybe_refresh_camera_matrix(img_w, img_h):
    """检测 zoom 是否变了, 变了就刷 camera_matrix (PnP 用真内参)"""
    est = _qrnav_estimator_holder[0]
    if est is None:
        return
    cur_zoom = max(1.0, round(float(getattr(est, "current_zoom_mag", 1.0) or 1.0), 2))
    last = _qrnav_last_known_zoom[0]
    if last is None or abs(cur_zoom - last) > 0.05:  # zoom 变了 (5% 容差)
        _ensure_focal_for_zoom(cur_zoom, img_w, img_h, force_query=False)


# ============================================================
#  ★ 需求: 「云台基于二维码数据的位置和方向」报告 (终端 + 页面2 显示)
# ============================================================
def describe_pose_report(pose, cmd=None):
    """报告云台位置（暂作狗的位置）、横移目标和二维码朝向。

    内容:
      1) 云台当前角度 (yaw/pitch)
      2) 相机在二维码坐标系的位置 camera_position，作为狗的位置
      3) 横移目标为 |camera_position[0]|
      4) QR 平面朝向偏角 (qr_yaw/qr_pitch, 相对相机)
      5) 方向结论: QR 在云台哪一侧多远, 建议动作
    """
    if pose is None:
        return "— 无 pose —"
    cmd = cmd or pose_to_command(pose)
    gy = cmd.get("gimbal_yaw_deg") or 0.0
    gp = cmd.get("gimbal_pitch_deg") or 0.0
    lat = cmd.get("lateral_x")
    cam_y = cmd.get("camera_y")
    cam_z = cmd.get("camera_z")
    if lat is not None and cam_y is not None and cam_z is not None:
        cam_str = f"X={lat:+.3f}m Y={cam_y:+.3f}m Z={cam_z:+.3f}m"
    else:
        cam_str = "n/a"
    lat_eff = (lat if lat is not None else 0.0) * _QR_NAV_CFG.get("LATERAL_SIGN", +1)
    align_lat = float(_QR_NAV_CFG.get("EXEC_MIN_LAT", 0.005))
    if lat is None:
        suggest = "相机位置 X 未就绪"
    elif abs(lat) <= align_lat:
        suggest = f"|X| ≤ {align_lat}m，无需横移"
    else:
        suggest = f"横移指令 {'right' if lat_eff > 0 else 'left'} (按当前左右配置)"
    dist = cmd.get("distance")
    dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
    fov_x = pose.get("fov_x")
    fov_y = pose.get("fov_y")
    fov_source = str(pose.get("fov_source", "unknown"))
    try:
        fov_str = f"H={float(fov_x):.2f}° V={float(fov_y):.2f}° ({fov_source})"
    except (TypeError, ValueError):
        fov_str = "n/a"
    return "\n".join([
        "[云台→二维码 报告]",
        f"  云台角度: yaw={gy:+.1f}°  pitch={gp:+.1f}°",
        f"  本次 Pose FOV: {fov_str}",
        f"  机器狗位置（按云台位置）: {cam_str} (二维码坐标系，距离 {dist_str})",
        (f"  执行横移目标: |相机位置 X|={abs(lat):.3f}m" if lat is not None
         else "  执行横移目标: 未就绪"),
        f"  QR 平面朝向: yaw={cmd.get('qr_yaw', 0.0):+.1f}° "
        f"pitch={cmd.get('qr_pitch', 0.0):+.1f}° (相对相机)",
        f"  方向结论: {suggest}",
    ])
