#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gimbal_qr_dog_auto_drive.py
============================
QR 位姿 → 机器狗 自动导航 (附录模式: 完整保留参照的云台控制台)

设计原则:
    ┌─────────────────────────────────────────────────────┐
    │  1. 复用参照文件 gimbal_pose_estimation_ascend.py    │
    │     - 云台控制 / RTSP / YOLO / 3D 视图 / 狗控制按钮 │
    │     - 完整保留 HTML 控制台 (端口 5003)              │
    │  2. 本文件只追加 QR 导航面板                         │
    │     - /qr_nav      面板页面 (iframe 友好)           │
    │     - /qr_nav/pose 实时位姿 JSON                    │
    │     - /qr_nav/auto 自动对齐+前进 (单步)              │
    │     - /qr_nav/align 仅横向对齐 (单步)                │
    │  3. 位姿数据直接读 estimator.latest_pose_data       │
    │  4. 狗控制直接调参照的 /dog/<action> 路由            │
    └─────────────────────────────────────────────────────┘

调用方式严格参照 D:\\hjl\\QR_ASCEND\\gimbal_pose_estimation_ascend.py:

    python3 gimbal_qr_dog_auto_drive.py --mode video --model models/qr_yolov8.om

参数完全兼容参照文件 (14 个同名参数) + 7 个本文件独有参数.
"""

import os
import sys
import io
import time
import json
import socket
import logging
import tempfile
import threading
import argparse
import contextlib

import numpy as np
import cv2


# ============================================================
#  关键: 复用参照文件 (云台 / YOLO / 视频流 / HTTP 控制台)
# ============================================================
import gimbal_pose_estimation_ascend as _gpea


# ============================================================
#  全局状态 (QR 导航面板的 pose/msg buffer)
# ============================================================

class _JpegBuffer:
    """线程安全 + Condition 零空转 JPEG 帧缓冲"""
    def __init__(self):
        self._cond = threading.Condition()
        self._jpg = b""

    def update(self, jpg: bytes):
        with self._cond:
            self._jpg = jpg
            self._cond.notify_all()

    def wait(self, timeout: float = 0.5) -> bytes:
        with self._cond:
            self._cond.wait(timeout)
            return self._jpg


# QR 导航面板的 pose 缓存 (从 estimator.latest_pose_data 周期同步)
_qrnav_pose_lock    = threading.Lock()
_qrnav_pose_box     = [None]      # 最近一次的 pose_data (dict)
_qrnav_pose_ver     = [0]         # 版本号 (前端轮询用)
_qrnav_pose_msg_box = ["等待 QR 检测…"]
_qrnav_img_w_box    = [0]         # 最近一次 pose 对应的图像宽度 (像素) — fill_ratio 算分母
_qrnav_poll_busy    = threading.Lock()

# QR 导航面板的视频流 JPEG
_qrnav_jpg_buf = _JpegBuffer()


# ── pose_data → UI 字段 (5个数字) + 狗动作建议 ──
def _get_gimbal_angles():
    """从参照 estimator 读云台当前 yaw/pitch/roll (度)

    失败时返回 (0.0, 0.0, 0.0)
    """
    try:
        est = _qrnav_estimator_holder[0]
        if est is None:
            return 0.0, 0.0, 0.0
        gimbal = getattr(est, "gimbal", None)
        if gimbal is None:
            return 0.0, 0.0, 0.0
        ya, yp, _ = gimbal.get_current_angles()
        return float(ya), float(yp), 0.0
    except Exception:
        return 0.0, 0.0, 0.0


def _compute_fill_ratio(pose, img_w):
    """QR 占图比例 = qr 像素边长 / 图像宽度 (按参照 robot_dog_bridge_win.py 推导)

    参照 D:\\hjl\\QR_3D\\robot_dog_bridge_win.py:1300-1313:
        side_01/12/23/30 = QR 4 角点相邻像素距离
        qr_px_side = mean(4 边长)
        fill_ratio = qr_px_side / img_w   # 占画面宽度的比例
    目标范围: 0.15 ~ 0.40 (15%~40%)

    Args:
        pose:   pose_data 字典 (含 image_points 4 角点)
        img_w:  图像宽度 (像素, 0 时无法算)

    Returns:
        fill_ratio ∈ [0, 1], img_w 无效或 pose 无 image_points 时返回 0
    """
    if pose is None or img_w <= 0:
        return 0.0
    img_pts = pose.get("image_points")
    if img_pts is None or len(img_pts) < 4:
        return 0.0
    try:
        pts = np.asarray(img_pts, dtype=float).reshape(-1, 2)
        if len(pts) < 4:
            return 0.0
        side_01 = float(np.linalg.norm(pts[0] - pts[1]))
        side_12 = float(np.linalg.norm(pts[1] - pts[2]))
        side_23 = float(np.linalg.norm(pts[2] - pts[3]))
        side_30 = float(np.linalg.norm(pts[3] - pts[0]))
        qr_px_side = (side_01 + side_12 + side_23 + side_30) / 4.0
        return qr_px_side / float(img_w)
    except Exception:
        return 0.0


def _camera_pos_to_base(cam_pos, gimbal_yaw_deg):
    """把 QR 在相机局部坐标系下的位置 (tvec) 投影到云台基座坐标系

    推导 (核心):
        PnP 输出的 tvec = QR 在相机局部坐标系下的位置向量 (3,)
        当云台 yaw=0 时, 相机坐标系与基座坐标系重合 (X/Z), 因此:
            QR 在基座下 = tvec
        当云台 yaw=θ 时, 相机绕基座 Y 轴旋转:
            R_gimbal(θ) = 基座→相机 = [[cosθ,0,-sinθ],[0,1,0],[sinθ,0,cosθ]]
            R_gimbal^(-1)(θ) = R.T = [[cosθ,0,sinθ],[0,1,0],[-sinθ,0,cosθ]]
        因此 QR 在基座下 (水平):
            base_x = cosθ * tvec[0] + sinθ * tvec[2]
            base_z = -sinθ * tvec[0] + cosθ * tvec[2]

        ⚠ 用 tvec (QR 在相机局部下), 不是 -rmat.T @ tvec = camera_position.
        之前用 camera_position 推导, 符号错了导致左右判断反向.

    YAW_SIGN 开关:
        +1 = 假设 yaw>0 = 相机右转 (默认)
        -1 = 假设 yaw>0 = 相机左转 (云台协议反着时改这个)

    Args:
        cam_pos: ndarray (3,)  = camera_position  (保留兼容, 实际从 pose 取 tvec)
        gimbal_yaw_deg: 云台当前偏航角 (度)

    Returns:
        (base_x, base_y, base_z): QR 在云台基座下的 3D 位置 (米)
        base_x > 0  → QR 在基座右侧
        base_x < 0  → QR 在基座左侧
        base_z > 0  → QR 在基座前方
        base_z < 0  → QR 在基座后方
    """
    yaw_sign = _QR_NAV_CFG.get("YAW_SIGN", +1)
    yaw = np.radians(gimbal_yaw_deg) * yaw_sign
    cy, sy = np.cos(yaw), np.sin(yaw)

    # 优先从 estimator.latest_pose_data 读 tvec, 与传入 cam_pos 一致
    # 上一版用 estimator 读 tvec + 传 cam_pos 双源, 容易时序错位, 改用单源: 先 tvec, 再 cam_pos 兜底
    tx = ty = tz = 0.0
    src = "none"
    try:
        est = _qrnav_estimator_holder[0]
        pose = getattr(est, "latest_pose_data", None) if est else None
        if pose is not None and "tvec" in pose:
            tvec = np.array(pose["tvec"], dtype=float).flatten()
            if len(tvec) >= 3 and not (tvec[0] == 0 and tvec[2] == 0):
                tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
                src = "tvec"
        if tx == 0 and tz == 0 and cam_pos is not None:
            # 兜底: -cam_pos (rmat≈I 时等价于 tvec, 即 base_x = -cam_x)
            tx = -float(cam_pos[0])
            ty = -float(cam_pos[1])
            tz = -float(cam_pos[2])
            src = "cam_pos_fallback"
    except Exception:
        pass

    base_x = cy * tx + sy * tz
    base_z = -sy * tx + cy * tz
    base_y = ty
    return float(base_x), float(base_y), float(base_z)


def _qr_size_est():
    """estimator 实例里真实的 qr_size (即 PnP object_points 用的值); 取不到返回 0.0"""
    try:
        return float(getattr(_qrnav_estimator_holder[0], "qr_size", 0.0) or 0.0)
    except Exception:
        return 0.0


def _qr_size_resolved():
    """最终报告给 UI 的 QR 尺寸: estimator 真值优先, 否则配置常量 (默认 0.12).

    不再退回 pose dict —— 曾出现 pose['qr_size'] 带 0.049/0.05 导致面板显示错值。
    """
    return _qr_size_est() or float(_QR_NAV_CFG.get("QR_SIZE_M", 0.12))


def pose_to_command(pose):
    """把 estimator 的 pose_data 转成 UI 字段 + 自动驾驶建议

    输入 pose 字段 (与参照文件 estimate_pose_from_image 返回一致):
        rvec / tvec / rmat
        yaw / pitch / roll         (度, QR 相对相机的姿态)
        camera_position (3,)        (米, 相机在 QR 坐标系下的位置)
        distance                    (米, 相机到 QR 中心距离)
        qr_size                     (米)
        image_points (4, 2)
        qr_area / fill_ratio
        decoded                     (QR 内容)

    关键字段:
        lateral_x        : 相机局部 X (=camera_position[0]), 反映 QR 偏离相机光轴的左右
        lateral_world    : 投影到云台基座坐标系下的左右偏移 (← 用户要的「云台估计」)
        forward_world    : 投影到云台基座坐标系下的前后距离
    """
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

    # 兼容: rvec/tvec 可能是 ndarray 或 0-d / list
    if "camera_position" in pose:
        cam = np.array(pose["camera_position"], dtype=float).flatten()
    else:
        # 自己从 rvec/tvec 算
        rvec = np.array(pose["rvec"], dtype=float).reshape(3, 1)
        tvec = np.array(pose["tvec"], dtype=float).reshape(3, 1)
        rmat, _ = cv2.Rodrigues(rvec)
        cam = (-rmat.T @ tvec).flatten()

    x, y, z = float(cam[0]), float(cam[1]), float(cam[2])
    dist = float(np.linalg.norm(cam))
    lateral_x = x

    # ── 关键: 把 camera_position 投影到云台基座水平面 ──
    # 用户诉求: 左右移动数据按云台估计的 3D 坐标
    gimbal_yaw_deg, gimbal_pitch_deg, _ = _get_gimbal_angles()
    base_x, base_y, base_z = _camera_pos_to_base(cam, gimbal_yaw_deg)
    # _camera_pos_to_base 投影失败时 (tvec + cam_pos 都不可用) 返回 (None, None, None)
    lateral_world = base_x if base_x is not None else None
    forward_world = base_z if base_z is not None else None   # 基座下 QR 距狗前后方向的距离 (负值=在狗前方)

    # 转角方向 (UI 显示 + 建议旋转) - 用基座下左右
    # LATERAL_SIGN = -1: 实测默认 (基座 x>0=右 → 提示"左", 因为现场狗需反向走)
    # LATERAL_SIGN = +1: 反之
    if lateral_world is None:
        turn_dir, turn_sign = "?", 0
    else:
        lat_eff = lateral_world * cfg.get("LATERAL_SIGN", -1)
        if   lat_eff >  0.02: turn_dir, turn_sign = "右", -1
        elif lat_eff < -0.02: turn_dir, turn_sign = "左", +1
        else:                 turn_dir, turn_sign = "中",  0

    if lateral_world is not None and abs(lateral_world) > cfg["ALIGN_LAT"] * 1.5:
        rotate_hint_deg = -lateral_world * 30.0
        rotate_hint_deg = max(-15.0, min(15.0, rotate_hint_deg))
    else:
        rotate_hint_deg = 0.0

    # 变倍提示 (QR 占比 < 阈值 → 需变倍)
    # 参照 D:\hjl\QR_3D\robot_dog_bridge_win.py:1300-1313:
    #   fill = qr 4 角点像素边长均值 / 图像宽度
    # (pose_data 里没有 fill_ratio 字段, 必须用 image_points 实时算)
    with _qrnav_pose_lock:
        img_w = _qrnav_img_w_box[0]
    fill = _compute_fill_ratio(pose, img_w)
    zoom_needed = fill < cfg["ZOOM_TARGET"]
    if zoom_needed and fill > 1e-4:
        zoom_factor = (cfg["ZOOM_TARGET"] / fill) ** 0.5
        zoom_factor = min(zoom_factor, 5.0)
    else:
        zoom_factor = 1.0

    # 与 QR 平面夹角 (相机光轴与 QR 法线的夹角)
    if "rmat" in pose:
        rmat = np.array(pose["rmat"], dtype=float)
        Z_axis = rmat[:, 2]
        cos_a = float(np.clip(abs(float(Z_axis[2])), -1.0, 1.0))
        angle_to_plane = float(np.degrees(np.arccos(cos_a)))
    else:
        angle_to_plane = 0.0

    # 单步建议 (新需求: 仅左右移动 + 变倍, 旋转/前进交给用户手动)
    if lateral_world is not None and abs(lateral_world) > cfg["ALIGN_LAT"]:
        step = min(abs(lateral_world), cfg["STEP_LAT"])
        hint = f"→ 向{turn_dir}横移 {step:.2f}m (基座横偏 {lateral_world:+.3f}m)"
    elif zoom_needed:
        lx_str = f"{lateral_world:+.3f}" if lateral_world is not None else "n/a"
        hint = (f"🔍 需要变倍 {zoom_factor:.2f}x (QR 占比 {fill*100:.1f}% < "
                f"{cfg['ZOOM_TARGET']*100:.0f}%, 横偏 {lx_str} 已对齐)")
    else:
        lx_str = f"{lateral_world:+.3f}" if lateral_world is not None else "n/a"
        hint = (f"✓ 已对齐 (横偏 {lx_str} ≤ {cfg['ALIGN_LAT']}m, "
                f"QR 占比 {fill*100:.1f}%, 云台 yaw={gimbal_yaw_deg:.1f}°)")

    return {
        "ready": True, "distance": dist, "lateral_x": lateral_x,
        "camera_y": y, "camera_z": z,
        "lateral_world": lateral_world,
        "forward_world": forward_world,
        "gimbal_yaw_deg": gimbal_yaw_deg,
        "turn_dir": turn_dir, "turn_sign": turn_sign,
        "rotate_hint_deg": rotate_hint_deg,
        "fill_ratio": fill, "zoom_needed": zoom_needed, "zoom_factor": zoom_factor,
        "angle_to_plane": angle_to_plane,
        "qr_yaw": float(pose.get("yaw", 0.0)),
        "qr_pitch": float(pose.get("pitch", 0.0)),
        "decoded": str(pose.get("decoded", "")),
        # qr_size 统一到 _QR_NAV_CFG["QR_SIZE_M"] (0.12): estimator 真值优先, 否则常量
        "qr_size":      _qr_size_resolved(),
        "qr_size_src":  ("estimator ✓" if _qr_size_est() else "常量0.12"),
        "qr_size_pose": float(pose.get("qr_size", 0.0) or 0.0),
        "hint": hint,
    }


# ============================================================
#  QR 导航面板的 pose 同步 (后台线程, 节流)
# ============================================================
_QR_NAV_CFG = {
    "ALIGN_LAT":      0.10,    # 米: 横偏 < 此值算对齐 (决策用)
    "STEP_LAT":       0.15,    # 米: 单步横移
    # ── 横移执行解锁 (绕开参照单次 0.3m 安全锁) ──
    #   参照 /dog/<action> → dog.move(max_move_m=DOG_MAX_MOVE_M=0.3),
    #   横偏 0.45m 时单次超限被整个拒绝 → 狗一动不动("被安全距离锁死")。
    #   这里按 DOG_MAX_STEP_M 拆段执行, 累计 = 计算出的完整偏移量。
    "DOG_MAX_STEP_M": 0.30,    # 米: 狗单次平移上限 (与参照 DOG_MAX_MOVE_M 对齐, 可调大)
    "EXEC_MIN_LAT":   0.005,   # 米: 执行按钮死区 (|横偏| ≤ 此值才算"无需移动", 远小于 ALIGN_LAT)
    # 方向开关 (实测反了时一键翻转, 不影响其它维度)
    "LATERAL_SIGN":   -1,      # 左右反向 (实测确定): 单独翻横移方向
                              # 实测: 基座横偏 x>0 (QR 在基座右侧) 时,
                              #       必须发 /dog/left 才能让狗向右移动把 QR 拉回中心,
                              #       即"现场狗的左右与基座 x 反号", 故默认 -1。
                              # 若另一台设备实测相反, 点前端『↔ 翻转左右方向』切回 +1
    "YAW_SIGN":       +1,      # 云台 yaw 方向: 翻基座投影 (会同时翻前后/左右, 一般用 LATERAL_SIGN)
    # 视频推流 + 变倍
    "ZOOM_TARGET":    0.15,    # QR 占比阈值 (fill_ratio 低于此值则需要变倍)
    "STREAM_SCALE":   0.50,    # JPEG 缩放
    "STREAM_QUALITY": 60,
    "STREAM_FPS":     8.0,     # 视频推流帧率 (节流)
    "POSE_POLL_HZ":   4.0,     # pose 同步频率
    # 实时位姿估计 (可选): 周期用当前帧做单帧 PnP 回写, 让面板值实时跟随。
    # 默认 0 = 关闭 = 快照模式: 面板值只在「📸 捕获一次」后更新, 之后冻结不变。
    "LIVE_POSE_FPS":  0.0,     # >0 开启实时(会持续浮动); 0 = 快照(按捕获才变)
    "LIVE_POSE_EMA":  0.55,    # 实时位姿 EMA 新值权重 (单帧 PnP 抖动大, 需平滑)
    "QR_SIZE_M":      0.12,    # QR 物理尺寸 (米): 整条链路统一 (PnP + 面板显示)
}


def _encode_qr_panel_jpeg(frame, pose):
    """把 estimator.current_frame 编码成 JPEG, 画 QR 框 + 数字 (供 /qr_nav/stream)"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = _QR_NAV_CFG["STREAM_SCALE"]
    nw, nh = int(w * scale), int(h * scale)
    if nw > 0 and nh > 0 and (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)

    if pose is not None:
        # 画 QR 框
        if "image_points" in pose:
            pts = np.array(pose["image_points"], dtype=int).reshape(-1, 2)
            if len(pts) == 4:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        # 画位姿文字
        cmd = pose_to_command(pose)
        cam = (cmd["camera_x"] if "camera_x" in cmd else cmd.get("lateral_x", 0))
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
    """后台线程: 周期从 estimator 同步 pose / frame 到 QR 导航面板"""
    fps = _QR_NAV_CFG["STREAM_FPS"]
    interval = 1.0 / max(1.0, fps)
    next_t = time.time()
    while not stop_evt.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
            continue
        next_t = now + interval

        # 1) 同步 pose_data
        pose = getattr(estimator, "latest_pose_data", None)
        # 2) 同步视频帧 (从 estimator.current_frame 拿)
        frame = getattr(estimator, "current_frame", None)
        img_w = int(frame.shape[1]) if frame is not None else 0
        with _qrnav_pose_lock:
            _qrnav_pose_box[0] = pose
            _qrnav_img_w_box[0] = img_w
            if pose is not None:
                # fill 实时算: qr 4 角点像素边长 / 图像宽度 (按 D:\hjl\QR_3D\robot_dog_bridge_win.py 推导)
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
    """挑一个纯 ASCII 的临时文件路径.

    cv2.imread / cv2.imwrite 在含中文的路径下会失败 (Windows 非 ASCII 编码),
    tempfile.gettempdir() 有时正好是 C:\\Users\\<中文用户名>\\... →
    写图失败 → 实时位姿线程空转, 面板永远不刷新。
    这里优先 CWD (通常是项目目录, ASCII), 再退到 TEMP, 最后兜底。
    """
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
    """后台线程: 周期性用『当前帧』做单帧 PnP, 实时刷新面板位姿.

    为什么需要这个线程:
        参照 gimbal_pose_estimation_ascend 的 latest_pose_data 只在按键 [c] /
        /detect 之类的『多帧采集』流程里更新一次; 视频流模式 (_video_display_loop)
        只读它来画字, 从不持续刷新 → 面板上『左右偏移量 / 前距』等会一直停在
        最后一次采集的快照上不动, 移动 QR 或狗都没有反应。
    本线程周期 (LIVE_POSE_FPS) 从 estimator.current_frame 取最新帧:
        cv2.imwrite 临时图 → estimate_pose_from_image(debug=False) → 轻量 EMA →
        回写 estimator.latest_pose_data 和 _qrnav_pose_box, 让面板实时跟随。

    静默: 用 contextlib.redirect_stdout 吞掉 estimate_pose_from_image 的大量
          print, 避免每秒刷屏 (代价: 该 ~几十毫秒内其它线程的 print 会被吞)。
    """
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
            # 有人正在做多帧采集 (按键 [c]) 时让路, 避免抢推理/覆盖平滑结果
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
            # 轻量 EMA: 单帧 PnP 抖动大, 不平滑会导致面板数字乱跳
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


# ============================================================
#  自动驾驶 (调用参照的 /dog/<action> HTTP 路由, 不直连狗)
# ============================================================
def _call_dog_http(action, value=0.0, gait=None, timeout=8.0):
    """通过 HTTP 调参照的 /dog/<action> 路由, 保持狗控制单点入口

    timeout 默认 8s: 参照 /dog/stand → dog.init() 内部阻塞约 4s,
    加网络抖动, 5s 容易误判超时; 8s 更稳。
    """
    import urllib.request
    import urllib.parse
    qs = {"v": value}
    if gait is not None:
        qs["gait"] = gait
    url = f"http://127.0.0.1:{_qrnav_dog_port_holder[0]}/dog/{action}?" + \
          urllib.parse.urlencode(qs)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"ERR 调狗失败: {e}"


_qrnav_dog_port_holder = [5003]   # 实际 HTTP 端口 (main() 注入)
_qrnav_estimator_holder = [None]  # 参照 estimator 实例 (main() 注入, 零 HTTP 读 dog)
_qrnav_last_init_ts = [0.0]       # 上次「一键初始化」时间戳 (冷却锁, 防连点)
_QR_NAV_INIT_COOLDOWN = 6.0       # 初始化冷却秒数 (init 全程约 4~5s)


def auto_step_nav():
    """根据『云台基座坐标系下的 3D 坐标』决策单步 (只横移 + 变倍)

    输入 (来自 pose_to_command):
        lat (lateral_world)  : QR 在基座下 X 偏移 (m), >0 右 / <0 左
        fwd (forward_world)  : QR 在基座下 Z 距离 (m), >0 在基座前方
        yaw (gimbal_yaw_deg) : 当前云台 yaw (度)
        fill (fill_ratio)    : QR 占比

    决策优先级 (新需求: 不再自动旋转/自动前进):
      1. 横偏 > 阈值 → 横移 (消除左右偏差)
      2. QR 占比 < ZOOM_TARGET → 变倍
      3. 都满足 → "已对齐"

    旋转和前进交由用户手动按钮 (/dog/tl /dog/tr /dog/fwd)

    输出: 文本结果 (dog_http 返回文本 + 决策说明)
    """
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "无 QR 数据, 跳过"

    cmd = pose_to_command(pose)
    if not cmd["ready"]:
        return "无 QR 数据, 跳过"
    lat  = cmd["lateral_world"]
    fwd  = cmd["forward_world"]
    yaw  = cmd["gimbal_yaw_deg"]
    fill = cmd.get("fill_ratio") or 0.0
    zoom_needed = cmd.get("zoom_needed", False)

    if lat is None or fwd is None:
        return "基座 3D 坐标未就绪 (云台角度? QR 位姿?)"

    abs_lat = abs(lat)

    # 1) 横偏 → 横移 (完整横偏分段执行, 绕开参照单次 0.3m 安全锁)
    if abs_lat > cfg["ALIGN_LAT"]:
        direction, step, n, per, ret = _move_lateral_steps(lat)
        seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
        msgs = [f"{cmd['turn_dir']}: 横移 {step:.2f}m ({seg}) "
                f"(基座横偏 {lat:+.3f}m, 前距 {fwd:.2f}m, "
                f"yaw={yaw:.1f}°)" + " | " + ret]
        # 横移后顺便看下需不需要变倍
        if zoom_needed:
            msgs.append(_auto_zoom_step(fill))
        return " | ".join(msgs)

    # 2) 已对齐 (横偏 ≤ 阈值): 判断变倍
    if zoom_needed:
        return (f"✓ 横偏已对齐 ({lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m) | "
                + _auto_zoom_step(fill) + " | (前进/旋转请手动按钮)")

    # 3) 完全对齐
    return (f"✓ 已对齐 (横偏 {lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m, "
            f"QR 占比 {fill*100:.1f}% ≥ {cfg['ZOOM_TARGET']*100:.0f}%, "
            f"前距 {fwd:.2f}m, yaw={yaw:.1f}°)")


def _auto_zoom_step(fill):
    """按当前 fill_ratio 自动变倍 (云台给的 fill → 目标倍率)"""
    cfg = _QR_NAV_CFG
    est = _qrnav_estimator_holder[0]
    if est is None or getattr(est, "gimbal", None) is None:
        return f"🔍 需变倍 (fill {fill*100:.1f}% < {cfg['ZOOM_TARGET']*100:.0f}%, 无云台句柄)"
    cur_z = getattr(est, "current_zoom_mag", 1.0)
    if fill < 1e-4:
        return "🔍 需变倍 (fill=0, 无法算倍率)"
    k = (cfg["ZOOM_TARGET"] / fill) ** 0.5
    zmin = getattr(est, "_zoom_min", 1.0)
    zmax = getattr(est, "_zoom_max", 30.0)
    target = max(zmin, min(zmax, cur_z * k))
    est.gimbal.zoom_to_magnification(target)
    est.current_zoom_mag = target
    return f"🔍 变倍→{target:.1f}x (fill {fill*100:.1f}% → {cfg['ZOOM_TARGET']*100:.0f}%)"


def align_step_nav():
    """仅横向对齐 (用 lateral_world)"""
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "无 QR 数据"
    cmd = pose_to_command(pose)
    if not cmd["ready"]:
        return "无 QR 数据"
    lat = cmd["lateral_world"]
    yaw = cmd["gimbal_yaw_deg"]
    if lat is None:
        return "基座横偏未就绪"
    dead = float(cfg.get("EXEC_MIN_LAT", 0.005))
    if abs(lat) <= dead:
        return (f"✓ 已接近 0 (基座横偏 {lat:+.3f}m ≤ {dead}m, "
                f"yaw={yaw:.1f}°)")
    direction, step, n, per, ret = _move_lateral_steps(lat)
    seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
    return (f"↔ 横移 {step:.2f}m 向{direction} ({seg}) | {ret} "
            f"(基座横偏 {lat:+.3f}m, yaw={yaw:.1f}°)")


# ============================================================
#  云台估计 → 两个独立单步 (① 左右偏移 / ② 变倍)
#  需求: 变倍 与 左右偏移 分开成两个按钮各自执行,
#        并在云台估计后『输出这两个值』(左右偏移值 / 变倍数值)
# ============================================================
def _est_values():
    """云台估计一次, 返回 (lat, fwd, yaw, fill, cur_z, lat_src) 供两个单步共用

    lat     : 左右偏移值 (米, 已统一为基座 x 符号: + 右 / - 左)
    fwd     : 前距 (米)
    yaw     : 当前云台 yaw (度, 0 表示云台未就绪)
    fill    : QR 占比 (变倍依据)
    cur_z   : 当前变倍倍数
    lat_src : "base" / "fallback" — 横偏数值的来源
              base: 来自基座投影 (lateral_world)
              fallback: 基座为 None, 用 -camera_position[0] 还原
    返回 None 表示无 pose / 仍未检出 QR
    """
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return None
    cmd = pose_to_command(pose)
    if not cmd.get("ready"):
        return None
    lat_g = cmd.get("lateral_world")
    lat_x = cmd.get("lateral_x")
    fwd   = cmd.get("forward_world")
    yaw   = cmd.get("gimbal_yaw_deg")
    fill  = cmd.get("fill_ratio") or 0.0
    est   = _qrnav_estimator_holder[0]
    cur_z = getattr(est, "current_zoom_mag", 1.0) if est else 1.0

    # 优先用基座横偏; 若为 None (云台未投影成功), 兜底用相机局部 x 取反还原
    if lat_g is not None:
        lat, lat_src = float(lat_g), "base"
    elif lat_x is not None:
        lat, lat_src = -float(lat_x), "fallback"   # camera_x 取反 = 基座 x
    else:
        return None
    if fwd is None:
        return None
    return (lat, float(fwd), float(yaw or 0.0),
            float(fill), float(cur_z), lat_src)


def _move_lateral_steps(lat, max_step=None):
    """按『完整横偏量』分段驱动狗横移, 绕开参照的单次 0.3m 安全锁.

    问题: 参照 /dog/<action> → dog.move(max_move_m=DOG_MAX_MOVE_M=0.3),
          单次超限直接拒绝(返回 False) → 横偏 0.45m 时整个动作被"安全距离"
          锁死, 狗一动不动。
    做法: 拆成 N 段, 每段 ≤ DOG_MAX_STEP_M(默认 0.30), 累计 = 计算出的完整
          偏移量, 真正做到『算出多少移多少』。

    Returns: (direction, step, n, per, ret_text)
    """
    cfg = _QR_NAV_CFG
    step = abs(float(lat))
    lat_eff = float(lat) * cfg.get("LATERAL_SIGN", -1)
    direction = "right" if lat_eff > 0 else "left"
    mx = float(max_step if max_step else
               cfg.get("DOG_MAX_STEP_M", 0.30) or 0.30)
    if mx <= 0:
        mx = step or 0.30
    n = int(step // mx)                       # 段数 (不用 math 模块)
    if n * mx < step - 1e-9:
        n += 1
    n = max(1, n)
    per = step / n
    if n == 1:
        return (direction, step, 1, per,
                f"{per:.2f}m → " + _call_dog_http(direction, value=per))
    rets = []
    for i in range(n):
        rets.append(f"[{i+1}/{n}] "
                    + _call_dog_http(direction, value=per))
    return direction, step, n, per, " | ".join(rets)


def lateral_step_only():
    """① 仅执行『左右偏移』一次 (云台估计的左右偏移值 → 狗横移)

    数值来源: 优先用『基座横偏 base_x』, 缺失时兜底用『相机局部 x 取反』
              不论来源, 显示的 x 单位都是米, 符号都是『基座坐标』 (+ 右 / - 左)

    关键诊断: 返回字符串里会带 base_x / camera_x / yaw / 前距 原始值,
              即便没动也能看清"为什么"
    """
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return ("✗ 横偏数值未就绪: 暂无 QR 帧 (摄像头/检测器还没出图, "
                "或刚启动还没收到首帧位姿)")
    cmd = pose_to_command(pose)
    lat_g  = cmd.get("lateral_world")
    lat_x  = cmd.get("lateral_x")
    fwd    = cmd.get("forward_world")
    yaw    = cmd.get("gimbal_yaw_deg")
    fill   = cmd.get("fill_ratio") or 0.0
    qr_s   = cmd.get("qr_size") or 0.0

    # 数值: 优先基座, 兜底 -camera_x
    if lat_g is not None:
        lat, src = float(lat_g), "base"
    elif lat_x is not None:
        lat, src = -float(lat_x), "fallback"
    else:
        return (f"✗ 横偏数值未就绪: base_x=None, camera_x=None "
                f"(yaw={yaw or 0:.1f}°, 前距={fwd or 0:.2f}m, "
                f"fill={fill*100:.1f}%, qr_size={qr_s:.3f}m)")

    # 前距: 基座为 None 时退到 camera_position[2] (相机局部 z 距离), 不阻塞横移
    fwd_str = f"{fwd:.2f}m" if fwd is not None else "n/a"

    dead = float(cfg.get("EXEC_MIN_LAT", 0.005))
    extra = (f"(来源={src}, 原始 base_x={lat_g}, camera_x={lat_x}, "
             f"yaw={yaw or 0:.1f}°, 前距 {fwd_str}, "
             f"fill={fill*100:.1f}%, qr_size={qr_s:.3f}m)")

    # 死区: 远小于 ALIGN_LAT, 只有真的接近 0 才不动 → 按钮"解锁"
    if abs(lat) <= dead:
        return (f"✓ 横偏 x={lat:+.3f}m ≤ {dead}m 已接近 0, 无需移动  "
                f"{extra}")

    direction, step, n, per, ret = _move_lateral_steps(lat)
    if n > 1:
        seg = (f"分{n}段×{per:.2f}m "
               f"(绕开参照单次 {cfg.get('DOG_MAX_STEP_M', 0.3)}m 安全锁)")
    else:
        seg = f"单段 {per:.2f}m"
    return (f"✓ 横移 {step:.2f}m 向{direction} | {seg} | {ret}  {extra}")


def zoom_step_only():
    """② 仅执行『变倍』一次 (云台估计的 QR 占比 → 目标倍率)

    数值来源: QR 占比 fill 由当前 pose 的 image_points 实时算出;
              缺失时 (fill=0) 返回诊断而非静默不动
    """
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "✗ 变倍数值未就绪: 暂无 QR 帧"
    cmd = pose_to_command(pose)
    fill  = cmd.get("fill_ratio") or 0.0
    qr_s  = cmd.get("qr_size") or 0.0
    yaw   = cmd.get("gimbal_yaw_deg")
    fwd   = cmd.get("forward_world")
    est   = _qrnav_estimator_holder[0]
    cur_z = getattr(est, "current_zoom_mag", 1.0) if est else 1.0

    if fill <= 1e-4:
        return (f"✗ 无法算变倍数值: QR 占比 fill=0 (未检出 QR?) "
                f"当前 {cur_z:.1f}x, qr_size={qr_s:.3f}m, "
                f"yaw={yaw or 0:.1f}°, 前距={fwd or 0:.2f}m")
    if fill >= cfg["ZOOM_TARGET"]:
        return (f"✓ 无需变倍: QR 占比 {fill*100:.1f}% ≥ "
                f"{cfg['ZOOM_TARGET']*100:.0f}% (当前 {cur_z:.1f}x, "
                f"qr_size={qr_s:.3f}m)")
    if est is None or getattr(est, "gimbal", None) is None:
        return (f"✗ 无云台句柄, 无法变倍 "
                f"(占比 {fill*100:.1f}%, 当前 {cur_z:.1f}x, qr_size={qr_s:.3f}m)")
    k = (cfg["ZOOM_TARGET"] / fill) ** 0.5
    zmin = getattr(est, "_zoom_min", 1.0)
    zmax = getattr(est, "_zoom_max", 30.0)
    target = max(zmin, min(zmax, cur_z * k))
    est.gimbal.zoom_to_magnification(target)
    est.current_zoom_mag = target
    return (f"变倍数值 {cur_z:.1f}x → {target:.1f}x "
            f"(QR 占比 {fill*100:.1f}% → 目标 {cfg['ZOOM_TARGET']*100:.0f}%, "
            f"qr_size={qr_s:.3f}m, yaw={yaw or 0:.1f}°)")


# ============================================================
#  HTML 页面 (QR 导航面板 — 作为附录路由, 兼容 iframe)
# ============================================================
HTML_QR_NAV_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>QR → 狗 自动导航</title>
<style>
  body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#1a1a1f;color:#eee;padding:6px}
  h3{margin:4px 0;color:#0cf;font-size:14px}
  .container{display:flex;flex-wrap:wrap;gap:8px}
  .panel{background:#26262e;border-radius:5px;padding:8px}
  .vid{position:relative;background:#000;border:1px solid #444}
  .vid img{display:block;width:480px;height:auto}
  .vid .overlay{position:absolute;top:4px;left:4px;background:rgba(0,0,0,0.6);
                padding:3px 6px;font:11px monospace;color:#0f0;border-radius:3px}
  .pose{font-family:Consolas,monospace;font-size:12px;line-height:1.6}
  .pose b{color:#0cf;display:inline-block;min-width:120px}
  .ok{color:#0f0}.warn{color:#f80}.err{color:#f44}
  button{padding:5px 10px;margin:2px;border:none;border-radius:3px;cursor:pointer;
         background:#0af;color:#fff;font-size:12px}
  button:hover{background:#08d}
  button:disabled{background:#555;cursor:not-allowed}
  button.danger{background:#c33}
  .row{margin:3px 0}
  .grid{display:grid;grid-template-columns:repeat(3,46px);gap:3px;margin:4px 0}
  .grid button{padding:5px 0;font-size:14px}
  .ctr{text-align:center}
  .hint{color:#bbb;font-size:11px;margin:2px 0;padding:4px;background:#333;border-radius:3px}
</style>
</head>
<body>
<h3>🐕📷 QR 位姿 → 机器狗 自动导航 (附录面板, 完整云台控制台请见 / )</h3>
<div class="container">
  <div class="panel">
    <div class="vid">
      <img id="qr_img" src="/qr_nav/stream" alt="QR">
      <div class="overlay" id="overlay">无 QR</div>
    </div>
  </div>
  <div class="panel" style="min-width:340px">
    <!-- 📊 云台估计值 (捕获快照: 按「📸 捕获一次」刷新; 不点则冻结在上次值) -->
    <div class="pose" style="background:#00263a;border:2px solid #0ff;padding:8px;margin-bottom:6px">
      <b style="color:#0ff;font-size:14px">📊 云台估计值 (捕获快照)</b>
      <span style="color:#888;font-size:10px;margin-left:6px">← 值冻结在上次捕获, 直到再次捕获</span><br>
      <table style="width:100%;font-size:13px;color:#fff;margin-top:6px">
        <tr>
          <td style="color:#0f0;width:90px">↔ 左右偏移量:</td>
          <td><span id="ge_lat" style="color:#ff0;font-size:20px;font-weight:bold">--</span> m</td>
          <td style="color:#f80;font-size:11px"><span id="ge_latdir">--</span></td>
        </tr>
        <tr>
          <td style="color:#0f0">⤵ 前距:</td>
          <td><span id="ge_fwd" style="color:#ff0;font-size:16px">--</span> m</td>
          <td></td>
        </tr>
        <tr>
          <td style="color:#0f0">🔍 变倍数值:</td>
          <td><span id="ge_zoom" style="color:#ff0;font-size:16px">--</span></td>
          <td style="color:#f80;font-size:11px"><span id="ge_zoomdir">--</span></td>
        </tr>
        <tr>
          <td style="color:#0f0">📐 QR 尺寸:</td>
          <td><span id="ge_qs" style="color:#ff0;font-size:16px">--</span> m</td>
          <td style="color:#888;font-size:10px"><span id="ge_qs_src">--</span></td>
        </tr>
      </table>
      <button id="btn_capture" style="background:#f80;color:#222;font-weight:bold;width:100%;margin-top:6px"
              onclick="captureOnce()">📸 捕获一次 (刷新上面的值)</button>
      <div id="cap_hint" style="color:#888;font-size:10px;margin-top:3px">--</div>
    </div>
    <div class="pose" id="pose">
      <b style="color:#0f0">基座前距 (m):</b> <span id="p_fwd">--</span> &nbsp;
         <b style="color:#0f0">基座横偏 (m):</b> <span id="p_latw">--</span><br>
      <b style="color:#0f0">云台 yaw:</b> <span id="p_yaw_d">--</span>°<br>
      ─────────<br>
      <b>距离 (m):</b> <span id="p_dist">--</span><br>
      <b>相机 X (m):</b> <span id="p_lat">--</span><br>
      <b>相机 Y (m):</b> <span id="p_hei">--</span><br>
      <b>相机 Z (m):</b> <span id="p_dep">--</span><br>
      <b>转角方向:</b> <span id="p_turn">--</span> &nbsp;
         <b>建议旋转:</b> <span id="p_rot">--</span>°<br>
      <b>QR 占比:</b> <span id="p_fill">--</span> &nbsp;
         <b>变倍提示:</b> <span id="p_zoom">--</span><br>
      <b>与QR平面夹角:</b> <span id="p_ang">--</span>°<br>
      <b>QR yaw / pitch:</b> <span id="p_qy">--</span> / <span id="p_pitch">--</span>°<br>
      <b>QR 内容:</b> <span id="p_dec" style="color:#ff0">--</span><br>
      <b style="color:#f80">QR 物理尺寸 qr_size:</b> <span id="p_qs" style="color:#f80">--</span> m
         <span style="color:#888;font-size:10px">(估计器硬编码: 0.12m, 与 PnP 一致才算准)</span>
    </div>
    <div class="pose" id="plan" style="background:#003;border:1px solid #0ff;padding:6px;margin-top:6px">
      <b style="color:#0ff">🎯 云台估计输出 (① 左右偏移值 / ② 变倍数值)</b><br>
      <table style="width:100%;font-size:13px;color:#fff;margin-top:4px">
        <tr>
          <td style="color:#aaa">↔ 左右移动:</td>
          <td><span id="pl_lat" style="color:#0ff;font-weight:bold">--</span> m
              <span id="pl_latdir" style="color:#f80"></span></td>
        </tr>
        <tr>
          <td style="color:#aaa">🔍 变倍倍数:</td>
          <td>当前 <span id="pl_curz" style="color:#fff">--</span>x
              → 目标 <span id="pl_zm" style="color:#0ff;font-weight:bold">--</span>x</td>
        </tr>
        <tr>
          <td style="color:#aaa;font-size:10px">(参考)</td>
          <td style="font-size:10px;color:#888">前距 <span id="pl_fwd">--</span> m,
              yaw <span id="pl_yaw_d">--</span>° (前进/旋转请手动按钮)</td>
        </tr>
      </table>
      <div id="pl_hint" style="color:#888;font-size:10px;margin-top:4px">--</div>
    </div>
    <div class="hint" id="hint">等待数据…</div>
  </div>
  <div class="panel">
    <div class="row">
      <button id="btn_init" style="background:#0c0;color:#000;font-weight:bold;width:100%"
              onclick="call('/qr_nav/init_all')">
        ① 🦿 一键初始化 (起立+力控+开始运动)
      </button>
    </div>
    <div class="row" style="font-size:11px;color:#aaa">
      ⚠ 顺序不能漏, 缺「力控」移动指令被丢, 缺「开始运动」狗腿不动<br>
      <span id="step_stand">①</span>·
      <span id="step_force">②</span>·
      <span id="step_motion">③</span> &nbsp; 状态:
      <span id="init_state" style="color:#f80">未初始化</span>
    </div>
    <div class="row">
      <button style="background:#08f;color:#fff;font-weight:bold"
              onclick="call('/qr_nav/do_lateral')">↔️ ① 执行左右偏移</button>
      <button style="background:#0a8;color:#fff;font-weight:bold"
              onclick="call('/qr_nav/do_zoom')">🔍 ② 执行变倍</button>
    </div>
    <div class="ctr" style="color:#8f8;font-size:10px">
      ✅ 横移已解锁: 按完整横偏执行 (算出多少移多少),
      单次上限 <span id="lat_max_step">0.30</span>m, 超出自动分段
    </div>
    <div class="row">
      <button style="background:#f80;color:#222" onclick="call('/qr_nav/auto')">🎯 自动一步 (横移+变倍)</button>
      <button onclick="call('/qr_nav/align')">↔️ 仅横向对齐</button>
    </div>
    <div class="row">
      <button id="btn_exec" style="background:#f0f;color:#000;font-weight:bold;width:100%"
              onclick="execPlanLoop()">🚀 一键执行 (循环横移+变倍)</button>
    </div>
    <div class="ctr" style="color:#888;font-size:10px">
      决策: 横偏&gt;阈值→横移; QR 占比&lt;15%→变倍 (旋转/前进请用下方手动按钮)
    </div>
    <div class="row" style="margin-top:6px">
      <div class="ctr" style="color:#888;font-size:10px">镜头变倍 (→ 云台 zoom_to_magnification)</div>
      <div class="grid" style="grid-template-columns:repeat(4,72px);margin:auto">
        <button onclick="call('/qr_nav/zoom/out')">🔍- 变倍-</button>
        <button style="background:#08f;color:#fff" onclick="call('/qr_nav/zoom/auto')">🎯 自动变倍</button>
        <button onclick="call('/qr_nav/zoom/in')">🔍+ 变倍+</button>
        <button onclick="call('/qr_nav/zoom/focus')">🔲 自动对焦</button>
      </div>
    </div>
    <div class="row" style="margin-top:6px">
      <div class="ctr" style="color:#888;font-size:10px">单步微调 (→ 调参照 /dog/*)</div>
      <div class="grid" style="margin:auto">
        <div></div><button onclick="call('/dog/fwd')">↑</button><div></div>
        <button onclick="call('/dog/left')">←</button>
        <button class="danger" onclick="call('/dog/down')">⏹</button>
        <button onclick="call('/dog/right')">→</button>
        <div></div><button onclick="call('/dog/back')">↓</button><div></div>
      </div>
      <div class="grid" style="grid-template-columns:repeat(2,70px);margin:auto">
        <button onclick="call('/dog/tl')">↺ 左转</button>
        <button onclick="call('/dog/tr')">↻ 右转</button>
      </div>
    </div>
    <div class="row" style="margin-top:4px">
      <button class="danger" onclick="call('/dog/damp')">💧 阻尼</button>
      <button class="danger" onclick="call('/dog/stop')">⏹ 停止并趴下</button>
    </div>
    <div class="row">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="call('/qr_nav/dog_ready')">🔍 检查狗状态</button>
    </div>
    <div class="row">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="call('/qr_nav/lateral_sign')">↔ 翻转左右方向 (左右反了点这个)</button>
    </div>
    <div class="row">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="call('/qr_nav/yaw_sign')">🔄 反转 yaw 方向 (云台左右反了用这个)</button>
    </div>
  </div>
</div>
<script>
var callBusy = false;
async function call(path){
  if (callBusy) return;          // 串行化: 上一条没回来, 忽略本次 (防连点)
  callBusy = true;
  var e = document.getElementById('hint');
  // 初始化/移动是阻塞操作 (init≈4s), 先给「执行中」反馈, 避免用户以为卡死而连点
  if (e) e.textContent = '⏳ 执行中: ' + path + ' ...';
  try {
    const r = await fetch(path);
    const t = await r.text();
    if (e) e.textContent = path + ' → ' + t;
    setTimeout(pollDogState, 200);
  } catch(err) {
    if (e) e.textContent = '✗ ' + path + ' 请求失败: ' + err;
  } finally {
    callBusy = false;
  }
}
async function pollDogState(){
  try {
    const r = await fetch('/qr_nav/dog_ready');
    const j = await r.json();
    const setLight = (id, ok) => {
      const el = document.getElementById(id);
      if (el) el.textContent = ok ? '🟢' : '⚪';
    };
    setLight('step_stand',  j.stand);
    setLight('step_force',  j.force);
    setLight('step_motion', j.motion);
    const s = document.getElementById('init_state');
    if (s) {
      if (j.motion)      { s.textContent = '✅ 已就绪, 可运动'; s.style.color = '#0f0'; }
      else if (j.force)  { s.textContent = '⚠ 已力控, 还差「开始运动」'; s.style.color = '#f80'; }
      else if (j.stand)  { s.textContent = '⚠ 已起立, 还差「力控+开始运动」'; s.style.color = '#f80'; }
      else               { s.textContent = '❌ 未初始化, 请按「① 一键初始化」'; s.style.color = '#f44'; }
    }
  } catch(e) {}
}
async function poll(){
  // 每个字段独立 try: 任一字段异常都不拖累其它 (根治"某一项卡住不动")
  const S = (id, fn) => { try { const el = document.getElementById(id);
                                 if (el) el.textContent = fn(); } catch(e){} };
  const n3 = v => (v>=0?'+':'') + Number(v).toFixed(3);
  try {
    const r = await fetch('/qr_nav/pose');
    const j = await r.json();
    S('lat_max_step', () => Number(j.lat_max_step || 0.30).toFixed(2));
    if (!j.ready){
      ['p_dist','p_lat','p_hei','p_dep','p_turn','p_rot',
       'p_fill','p_zoom','p_ang','p_qy','p_pitch','p_dec',
       'p_fwd','p_latw','p_yaw_d','p_qs',
       'ge_lat','ge_latdir','ge_fwd','ge_zoom','ge_zoomdir','ge_qs']
        .forEach(id => S(id, () => '--'));
      S('hint', () => j.hint || '等待 QR…');
      S('overlay', () => '无 QR');
    } else {
      S('p_dist', () => Number(j.distance).toFixed(3));
      S('p_lat',  () => n3(j.lateral_x));
      S('p_hei',  () => Number(j.camera_y).toFixed(3));
      S('p_dep',  () => Number(j.camera_z).toFixed(3));
      // 基座坐标 (核心, 用云台 yaw 投影)
      S('p_fwd',  () => Number(j.forward_world).toFixed(3));
      S('p_latw', () => n3(j.lateral_world));
      try {
        const fwdEl  = document.getElementById('p_fwd');
        const latwEl = document.getElementById('p_latw');
        fwdEl.style.color  = (j.forward_world < -0.9) ? '#f80' : '#0f0';
        latwEl.style.color = (Math.abs(j.lateral_world) > 0.10) ? '#f80' : '#0f0';
      } catch(e) {}
      S('p_yaw_d', () => Number(j.gimbal_yaw_deg).toFixed(1));
      try {
        const turnEl = document.getElementById('p_turn');
        turnEl.textContent = j.turn_dir;
        turnEl.className = j.turn_dir==='中' ? 'ok' : 'warn';
      } catch(e) {}
      S('p_rot', () => Number(j.rotate_hint_deg).toFixed(1));
      S('p_fill', () => Number(j.fill_ratio*100).toFixed(1) + '%');
      S('p_zoom', () => j.zoom_needed ? `需要变倍 (≈${Number(j.zoom_factor).toFixed(2)}x)` : 'OK');
      try { document.getElementById('p_zoom').className = j.zoom_needed ? 'warn' : 'ok'; } catch(e) {}
      // ===== 📊 云台估计值 高亮框 =====
      S('ge_lat', () => n3(j.lateral_world));
      try {
        const ldir = document.getElementById('ge_latdir');
        if (j.lateral_world > 0.10)       { ldir.textContent = '← 狗应左移'; ldir.style.color = '#f80'; }
        else if (j.lateral_world < -0.10) { ldir.textContent = '狗应右移 →'; ldir.style.color = '#f80'; }
        else                              { ldir.textContent = '已对齐 ✓';   ldir.style.color = '#0f0'; }
      } catch(e) {}
      S('ge_fwd', () => Number(j.forward_world).toFixed(3));
      // 注意: /qr_nav/pose 不含 zoom_mag/cur_zoom (那是 /qr_nav/plan), 这里用 zoom_factor
      S('ge_zoom', () => j.zoom_needed ? ('需 ≈' + Number(j.zoom_factor).toFixed(2) + 'x') : '无需变倍');
      try {
        const zdir = document.getElementById('ge_zoomdir');
        zdir.textContent = j.zoom_needed ? '需变倍' : 'OK';
        zdir.style.color = j.zoom_needed ? '#f80' : '#0f0';
      } catch(e) {}
      S('ge_qs', () => Number(j.qr_size).toFixed(3));
      S('ge_qs_src', () => j.qr_size_src || '');
      S('p_ang', () => Number(j.angle_to_plane).toFixed(1));
      S('p_qy', () => Number(j.qr_yaw).toFixed(1));
      S('p_pitch', () => Number(j.qr_pitch).toFixed(1));
      S('p_dec', () => j.decoded || '(未解码)');
      S('p_qs',  () => Number(j.qr_size || 0).toFixed(3));
      S('hint',  () => j.hint);
      S('overlay', () => `前 ${Number(j.forward_world).toFixed(2)}m  横 ${Number(j.lateral_world).toFixed(2)}m  yaw=${Number(j.gimbal_yaw_deg).toFixed(1)}°`);
    }
  } catch(e) {}
  setTimeout(poll, 250);
}
poll();

// ── 📸 捕获一次: 触发多帧位姿估计 (等同参照按键 [c]), 刷新上面的值 ──
//   面板网页按 [c] 不会传给参照的视频窗口, 所以给一个显式按钮/路由。
async function captureOnce(){
  const b = document.getElementById('btn_capture');
  const h = document.getElementById('cap_hint');
  if (!b || b.disabled) return;
  const old = b.innerText;
  b.disabled = true; b.innerText = '📸 捕获中...';
  h.textContent = '多帧采集 + YOLO + PnP (约 2~5 秒)，请稍候…';
  try {
    const r = await fetch('/qr_nav/capture');
    h.textContent = (await r.text()) || '✓ 完成';
  } catch(e) {
    h.textContent = '✗ 捕获失败: ' + e;
  }
  b.disabled = false; b.innerText = old;
}
setInterval(pollDogState, 3000);   // 3s: 读内存 dog.ready, 零 HTTP 双跳
pollDogState();

// ── 目标指令轮询 (云台估计, 新需求: 仅横移 + 变倍) ──
var lastPlan = null;
async function pollPlan(){
  try {
    const r = await fetch('/qr_nav/plan');
    const j = await r.json();
    lastPlan = j;
    if (!j.ready) {
      ['pl_lat','pl_zm','pl_curz','pl_fwd','pl_yaw_d'].forEach(id =>
        document.getElementById(id).textContent = '--');
      document.getElementById('pl_latdir').textContent = '';
      document.getElementById('pl_hint').textContent = j.hint || '等待 QR…';
      return;
    }
    document.getElementById('pl_lat').textContent =
      j.lateral_m > 0 ? j.lateral_m.toFixed(2) : '0.00';
    document.getElementById('pl_latdir').textContent =
      j.lateral_dir ? (j.lateral_dir==='right'?'→ 右':'← 左') : '';
    document.getElementById('pl_curz').textContent = j.cur_zoom.toFixed(1);
    document.getElementById('pl_zm').textContent = j.zoom_mag.toFixed(1);
    document.getElementById('pl_fwd').textContent = j.forward_world.toFixed(2);
    document.getElementById('pl_yaw_d').textContent = j.gimbal_yaw_deg.toFixed(1);
    document.getElementById('pl_hint').textContent = j.hint;
  } catch(e) {}
  setTimeout(pollPlan, 500);
}
pollPlan();

// ── 一键执行: 循环调 /qr_nav/exec_plan 直到横偏对齐 + 变倍 OK ──
var execBusy = false;
var execStop = false;
async function execPlanLoop(){
  if (execBusy) return;
  execBusy = true;
  execStop = false;
  const btn = document.getElementById('btn_exec');
  const oldTxt = btn.innerText;
  const hint = document.getElementById('pl_hint');
  btn.disabled = true;
  let i = 0;
  try {
    while (!execStop && lastPlan && lastPlan.ready) {
      // 防止无限循环: 50 步硬限
      if (i++ > 50) { hint.textContent = '⏹ 超过 50 步, 自动停止'; break; }
      // 检查是否已对齐: 横偏 ≤ 阈值 AND 变倍 OK
      const lat_ok = lastPlan.lateral_m < 1e-3;
      const zm_ok  = Math.abs(lastPlan.zoom_mag - lastPlan.cur_zoom) < 0.05;
      if (lat_ok && zm_ok) {
        hint.textContent = '✓ 已对齐 (横偏≤阈值 + 变倍 OK)';
        break;
      }
      btn.innerText = `🚀 执行中 ${i}…`;
      hint.textContent = `🚀 第 ${i} 步...`;
      const r = await fetch('/qr_nav/exec_plan');
      const t = await r.text();
      hint.textContent = t;
      // 重新拉 plan 看是否还有下一步
      await new Promise(r => setTimeout(r, 600));
      await pollPlanOnce();
    }
  } catch(e) {
    hint.textContent = '✗ ' + e;
  } finally {
    btn.innerText = oldTxt;
    btn.disabled = false;
    execBusy = false;
  }
}
async function pollPlanOnce(){
  try {
    const r = await fetch('/qr_nav/plan');
    lastPlan = await r.json();
  } catch(e) {}
}
</script>
</body>
</html>"""


# ============================================================
#  把 QR 导航路由挂到参照的 Flask app 上
# ============================================================


def _get_dog_obj():
    """直接拿参照 estimator 里的 dog 对象 (同进程, 零 HTTP 开销)

    参照的 GimbalPoseEstimator._get_dog() 懒加载 Dog 实例。
    这里直接读 estimator._dog, 避免 HTTP 双跳 (那会加重 Flask 阻塞)。
    """
    est = _qrnav_estimator_holder[0]
    if est is None:
        return None
    return getattr(est, "_dog", None)


def _init_dog_3steps():
    """一键初始化: 只发 ① stand (新版 init 已含 力控+开始运动), 不重复发

    为什么不再补发 force/motion?
    ─────────────────────────────────────────────────────────
    参照的 /dog/stand → robot_interface.Dog.init() 内部已是完整 5 步:
        心跳 → 起立 → 手动 → ★力控 → ★开始运动 → gait   (约 4 秒)
    若再补发 force/motion, 等于对已经使能的腿再发一遍:
        - 重复「力控」→ 腿重新进柔性模式 → 抖一下
        - 重复「开始运动」→ 重启步态循环 → 卡顿
    ← 这就是用户「重新站立偶尔卡顿」的根因。
    因此默认只发 stand; 仅当读内存发现 dog.ready 仍为 False 时才补发。
    """
    # ── 冷却锁: N 秒内只允许初始化一次 (防连点造成重复 init) ──
    now = time.time()
    elapsed = now - _qrnav_last_init_ts[0]
    if elapsed < _QR_NAV_INIT_COOLDOWN:
        return (f"⏳ 冷却中 (距上次初始化 {elapsed:.1f}s < "
                f"{_QR_NAV_INIT_COOLDOWN:.0f}s), 请稍候再按")
    _qrnav_last_init_ts[0] = now

    msgs = []
    # ① 起立 (新版 init 一次到位: 含 力控 + 开始运动)
    r = _call_dog_http("stand", value=0.0)
    msgs.append(f"① {r}")
    if "ERR" in r or "失败" in r or "拒绝" in r:
        return " | ".join(msgs)

    # ── 优先读内存判断: ready=True 说明 3 步已含, 不再补发 (防卡顿) ──
    if _is_dog_fully_ready():
        return " | ".join(msgs) + " ✓ 已就绪 (起立+力控+开始运动)"

    # ── 兜底: 某些旧版 robot_interface 的 init() 不含后两步时才补发 ──
    r = _call_dog_http("force", value=0.0)
    msgs.append(f"② {r}")
    if "ERR" in r or "失败" in r or "拒绝" in r:
        return " | ".join(msgs)
    r = _call_dog_http("motion", value=0.0, gait=0x21010301)
    msgs.append(f"③ {r}")
    if "ERR" in r or "失败" in r or "拒绝" in r:
        return " | ".join(msgs)
    return " | ".join(msgs) + " ✓ 补发完成, 狗已就绪"


def _get_dog_step_state():
    """返回 ①②③ 各步状态 (优先读内存 dog.ready, 零 HTTP 开销)

    因为 robot_interface.Dog.init() 是完整 5 步, dog.ready=True
    即代表 ①②③ 都完成 (不再区分单步), 所以内存路径下三者同亮。
    内存不可用时才退化为 HTTP 查 /dog_msg (不会频繁调用)。
    """
    state = {"stand": False, "force": False, "motion": False, "ready": False}

    # ── 主路径: 同进程读 dog.ready (最快, 无 HTTP) ──
    dog = _get_dog_obj()
    if dog is not None:
        if bool(getattr(dog, "ready", False)):
            state["stand"] = state["force"] = state["motion"] = True
            state["ready"] = True
        return state

    # ── 兜底: dog 对象还没创建, 查一次 /dog_msg (仅首次) ──
    try:
        import urllib.request
        url = f"http://127.0.0.1:{_qrnav_dog_port_holder[0]}/dog_msg"
        with urllib.request.urlopen(url, timeout=1) as r:
            msg = r.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return state
    if "未连接" in msg or msg == "":
        return state
    state["stand"] = state["force"] = state["motion"] = True
    state["ready"] = True
    return state


def _is_dog_fully_ready():
    """检查狗是否完全就绪 (①②③ 都已发)"""
    try:
        return _get_dog_step_state().get("motion", False)
    except Exception:
        return False


def _ensure_dog_ready(force_init=False):
    """确保狗已就绪; 返回 (ok: bool, msg: str)

    force_init=True : 无条件重新初始化 (「一键初始化」按钮)
    force_init=False: 已就绪则直接放行 (移动按钮, 避免重复站立造成卡顿)
    """
    if not force_init and _is_dog_fully_ready():
        return True, "狗已就绪"
    msg = _init_dog_3steps()
    if "ERR" in msg or "失败" in msg:
        return False, msg
    if "冷却" in msg:
        # 冷却中: 已就绪则放行; 未就绪则拦住 (提示等一会)
        return _is_dog_fully_ready(), msg
    return True, msg


def _register_qr_nav_routes(app):
    """把 QR 导航面板路由注册到参照的 Flask app

    参照的 _make_flask_app 已经建好 app, 我们直接挂路由
    """
    from flask import Response

    @app.route("/qr_nav/stream")
    def _qrnav_stream():
        def gen():
            while True:
                jpg = _qrnav_jpg_buf.wait(0.5)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/qr_nav/pose")
    def _qrnav_pose():
        with _qrnav_pose_lock:
            pose = _qrnav_pose_box[0]
        data = pose_to_command(pose)
        # 顺带把「横移单次上限」带给面板 (解锁提示里显示用)
        try:
            data["lat_max_step"] = float(
                _QR_NAV_CFG.get("DOG_MAX_STEP_M", 0.30))
        except Exception:
            pass
        return Response(json.dumps(data, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav/capture")
    def _qrnav_capture():
        """面板内「📸 捕获一次」: 触发一次多帧位姿估计 (等同参照按键 [c]),
        完成后把结果填进面板。面板值只在捕获后更新 → 不点则冻结。
        """
        est = _qrnav_estimator_holder[0]
        if est is None:
            return Response("✗ estimator 未就绪 (未注入?)",
                            mimetype="text/plain; charset=utf-8")
        frame = getattr(est, "current_frame", None)
        if frame is None:
            return Response("✗ 无视频帧 (视频流未启动 / 还没收到帧)",
                            mimetype="text/plain; charset=utf-8")
        if getattr(est, "_est_busy", False):
            return Response("⏳ 上一步位姿估计还没跑完, 请稍等",
                            mimetype="text/plain; charset=utf-8")
        try:
            est._start_estimate_async(frame)
        except Exception as e:
            return Response(f"✗ 触发失败: {e}",
                            mimetype="text/plain; charset=utf-8")
        # 等采集完成 (_est_busy: False → True → False), 最多 15s
        time.sleep(0.3)                      # 让 _est_busy 置位
        t0 = time.time()
        while time.time() - t0 < 15.0:
            if not getattr(est, "_est_busy", False):
                break
            time.sleep(0.1)
        # 主动同步一次, 不等后台线程
        pd = getattr(est, "latest_pose_data", None)
        if pd is None:
            return Response("✗ 估计失败 (未检测到 QR / 位姿解算失败)",
                            mimetype="text/plain; charset=utf-8")
        with _qrnav_pose_lock:
            _qrnav_pose_box[0] = pd
            _qrnav_pose_ver[0] += 1
        cmd = pose_to_command(pd)
        return Response(
            f"✓ 已捕获: 横偏 {cmd.get('lateral_world')} m, "
            f"前距 {cmd.get('forward_world')} m, "
            f"占比 {(cmd.get('fill_ratio') or 0)*100:.1f}%, "
            f"QR尺寸 {cmd.get('qr_size')} m",
            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/pose_ver")
    def _qrnav_pose_ver_route():
        # 注意: 处理函数名不能叫 _qrnav_pose_ver, 否则会遮蔽同名的全局列表
        with _qrnav_pose_lock:
            return Response(str(_qrnav_pose_ver[0]),
                            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/msg")
    def _qrnav_msg():
        with _qrnav_pose_lock:
            return Response(_qrnav_pose_msg_box[0],
                            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/auto")
    def _qrnav_auto():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = auto_step_nav()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/plan")
    def _qrnav_plan():
        """返回基于云台估计的『目标指令』 (新需求: 仅横移 + 变倍, 旋转/前进交给用户手动)
        返回 JSON: {
            "ready": bool,
            "lateral_m":  横移距离 (绝对值 m, 0 表示无需横移)
            "lateral_dir": "left"/"right"/""
            "zoom_mag":   变倍目标倍率 (1.0 表示无需变倍)
            "fill_ratio": 当前 QR 占比
            "lateral_world": 横偏原始值
            "forward_world": 前距 (仅供参考, 不再自动走)
            "gimbal_yaw_deg": 当前云台 yaw
            "hint":       简短提示
        }
        """
        with _qrnav_pose_lock:
            pose = _qrnav_pose_box[0]
        if pose is None:
            return Response('{"ready":false,"hint":"等待 QR…"}',
                            mimetype="application/json")
        try:
            cmd = pose_to_command(pose)
        except Exception as e:
            return Response(f'{{"ready":false,"hint":"ERR {e}"}}',
                            mimetype="application/json")
        if not cmd.get("ready"):
            return Response('{"ready":false,"hint":"无 QR 数据"}',
                            mimetype="application/json")
        cfg = _QR_NAV_CFG
        lat  = cmd.get("lateral_world")
        fwd  = cmd.get("forward_world")
        yaw  = cmd.get("gimbal_yaw_deg", 0.0)
        if lat is None or fwd is None:
            return Response('{"ready":false,"hint":"基座坐标未就绪"}',
                            mimetype="application/json")

        abs_lat = abs(lat)

        # 横移目标: 用云台算的完整横偏 (不截断到 STEP_LAT)
        if abs_lat > cfg["ALIGN_LAT"]:
            lat_m = abs_lat
            lat_dir = "right" if (lat * cfg.get("LATERAL_SIGN", -1)) > 0 else "left"
        else:
            lat_m, lat_dir = 0.0, ""

        # 变倍目标: fill < ZOOM_TARGET 才有
        fill = cmd.get("fill_ratio") or 0.0
        est = _qrnav_estimator_holder[0]
        cur_z = getattr(est, "current_zoom_mag", 1.0) if est else 1.0
        if fill > 1e-4 and fill < cfg["ZOOM_TARGET"] and est is not None:
            k = (cfg["ZOOM_TARGET"] / fill) ** 0.5
            zoom_mag = max(getattr(est, "_zoom_min", 1.0),
                           min(getattr(est, "_zoom_max", 30.0), cur_z * k))
        else:
            zoom_mag = cur_z   # 当前倍率, 1.0 表示无需变倍

        # 汇总提示
        parts = []
        if lat_dir:  parts.append(f"横移 {lat_m:.2f}m {lat_dir}")
        if abs(zoom_mag - cur_z) > 0.05:
            parts.append(f"变倍→{zoom_mag:.1f}x")
        hint = " / ".join(parts) if parts else "✓ 已对齐 (无需移动)"

        plan = {
            "ready": True,
            "lateral_m": round(lat_m, 3),
            "lateral_dir": lat_dir,
            "zoom_mag":   round(zoom_mag, 2),
            "cur_zoom":   round(cur_z, 2),
            "fill_ratio": round(fill, 4),
            "lateral_world": round(lat, 3),
            "forward_world": round(fwd, 3),
            "gimbal_yaw_deg": round(yaw, 1),
            "hint": hint,
        }
        return Response(json.dumps(plan, ensure_ascii=False),
                        mimetype="application/json")

    @app.route("/qr_nav/exec_plan")
    def _qrnav_exec_plan():
        """按 /qr_nav/plan 给出的当前建议, 串行执行一次 (新需求: 仅横移 + 变倍)

        前端『一键执行』循环调这个直到 plan.hint 出现『已对齐』
        旋转/前进由用户手动按钮 (/dog/tl /dog/tr /dog/fwd) 控制
        """
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")

            # 1) 复算 plan (和 /qr_nav/plan 一致, 避免另存)
            with _qrnav_pose_lock:
                pose = _qrnav_pose_box[0]
            if pose is None:
                return "无 QR 数据"
            cmd = pose_to_command(pose)
            if not cmd.get("ready"):
                return "无 QR 数据"
            cfg = _QR_NAV_CFG
            lat  = cmd.get("lateral_world")
            fwd  = cmd.get("forward_world")
            if lat is None or fwd is None:
                return "基座坐标未就绪"
            abs_lat = abs(lat)

            msgs = []
            # 1) 横移: 完整横偏分段执行 (绕开参照单次 0.3m 安全锁)
            if abs_lat > cfg["ALIGN_LAT"]:
                ld, step, n, per, ret = _move_lateral_steps(lat)
                seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
                msgs.append(f"↔ 横移 {step:.2f}m 向{ld} ({seg}): " + ret)
            else:
                msgs.append(f"✓ 横偏已对齐 ({lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m)")

            # 2) 变倍 (按需, 与横移独立)
            est = _qrnav_estimator_holder[0]
            if est is not None and getattr(est, "gimbal", None) is not None:
                fill = cmd.get("fill_ratio") or 0.0
                cur_z = getattr(est, "current_zoom_mag", 1.0)
                if fill > 1e-4 and fill < cfg["ZOOM_TARGET"]:
                    k = (cfg["ZOOM_TARGET"] / fill) ** 0.5
                    zmin = getattr(est, "_zoom_min", 1.0)
                    zmax = getattr(est, "_zoom_max", 30.0)
                    target = max(zmin, min(zmax, cur_z * k))
                    est.gimbal.zoom_to_magnification(target)
                    est.current_zoom_mag = target
                    msgs.append(f"🔍 变倍→{target:.1f}x (fill {fill*100:.1f}% → {cfg['ZOOM_TARGET']*100:.0f}%)")
                else:
                    msgs.append(f"🔍 变倍 OK (fill {fill*100:.1f}% ≥ {cfg['ZOOM_TARGET']*100:.0f}%)")

            # 3) 提示: 前进/旋转请用户手动
            msgs.append(f"前距 {fwd:.2f}m / yaw {cmd.get('gimbal_yaw_deg',0.0):.1f}° (前进/旋转请手动按钮)")

            return " | ".join(msgs)
        except Exception as e:
            return f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()

    @app.route("/qr_nav/align")
    def _qrnav_align():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = align_step_nav()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/do_lateral")
    def _qrnav_do_lateral():
        """① 仅执行『左右偏移』一次 (云台估计的左右偏移值 → 狗横移)

        与变倍完全独立: 只做横移, 不动镜头
        """
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = lateral_step_only()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/do_zoom")
    def _qrnav_do_zoom():
        """② 仅执行『变倍』一次 (云台估计的 QR 占比 → 目标倍率)

        与横移完全独立: 只变倍, 不移动狗 (仅动云台镜头, 不需狗就绪)
        """
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            msg = zoom_step_only()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/init_all")
    def _qrnav_init_all():
        """一键初始化 (只发 ① stand, 新版 init 已含力控+开始运动)
        防重复发 force/motion 造成狗腿卡顿.
        """
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, msg = _ensure_dog_ready(force_init=True)
            if not ok and "冷却" not in msg:
                msg = f"初始化失败: {msg}"
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/dog_ready")
    def _qrnav_dog_ready():
        """前端轮询: 返回 ①②③ 各步的状态 (用于 UI 指示灯)"""
        try:
            state = _get_dog_step_state()
        except Exception as e:
            return Response(
                json.dumps({"stand": False, "force": False,
                            "motion": False, "err": str(e)}),
                mimetype="application/json; charset=utf-8")
        return Response(json.dumps(state, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav/yaw_sign")
    def _qrnav_yaw_sign():
        """切换 YAW_SIGN (+1 / -1), 用户实测发现左右反时一键切换"""
        cur = _QR_NAV_CFG.get("YAW_SIGN", +1)
        new = -cur
        _QR_NAV_CFG["YAW_SIGN"] = new
        return Response(
            f"✓ YAW_SIGN 已切换: {cur:+d} → {new:+d} "
            f"({'右转正' if new>0 else '左转正'})",
            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/lateral_sign")
    def _qrnav_lateral_sign():
        """切换 LATERAL_SIGN (+1 / -1), 单独翻转横移方向 (不影响前后/旋转)

        用法: 用户实测发现『点右移按钮→狗往左走』时, 点这个按钮一键翻转
        """
        cur = _QR_NAV_CFG.get("LATERAL_SIGN", -1)
        new = -cur
        _QR_NAV_CFG["LATERAL_SIGN"] = new
        return Response(
            f"✓ LATERAL_SIGN 已切换: {cur:+d} → {new:+d} "
            f"(横移方向{'正常' if new>0 else '已翻转'})",
            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/zoom/<action>")
    def _qrnav_zoom(action):
        """变倍控制: in 步进+ / out 步进- / auto 按 QR 占比自动 / focus 对焦

        与参照 estimator 的 [A]/[D] 按键一致 (zoom_to_magnification,
        走到目标倍率自停, 不会无限持续)
        """
        est = _qrnav_estimator_holder[0]
        gimbal = getattr(est, "gimbal", None)
        if est is None or gimbal is None:
            return Response("✗ 云台未就绪 (estimator/gimbal 未注入)",
                            mimetype="text/plain; charset=utf-8")
        try:
            act = str(action).lower().strip()
            zmin = getattr(est, "_zoom_min", 1.0)
            zmax = getattr(est, "_zoom_max", 30.0)
            zstep = getattr(est, "zoom_step_val", 1.0)
            cur = getattr(est, "current_zoom_mag", 1.0)

            if act in ("in", "+"):
                target = min(zmax, cur + zstep)
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = f"✓ 变倍+ → {target:.1f}x (范围 {zmin:.0f}~{zmax:.0f}x)"
            elif act in ("out", "-"):
                target = max(zmin, cur - zstep)
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = f"✓ 变倍- → {target:.1f}x (范围 {zmin:.0f}~{zmax:.0f}x)"
            elif act == "auto":
                # 按 QR 占比算目标倍率: fill * k² = ZOOM_TARGET → k=√(T/fill)
                with _qrnav_pose_lock:
                    pose = _qrnav_pose_box[0]
                if pose is None:
                    return Response("✗ 无 QR 数据, 无法自动变倍",
                                    mimetype="text/plain; charset=utf-8")
                cmd = pose_to_command(pose)
                fill = cmd.get("fill_ratio") or 0.0
                if fill <= 1e-4:
                    return Response("✗ QR 占比无效, 无法自动变倍",
                                    mimetype="text/plain; charset=utf-8")
                k = (_QR_NAV_CFG["ZOOM_TARGET"] / fill) ** 0.5
                target = max(zmin, min(zmax, cur * k))
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = (f"✓ 自动变倍 → {target:.1f}x "
                       f"(占比 {fill*100:.1f}% → 目标 "
                       f"{_QR_NAV_CFG['ZOOM_TARGET']*100:.0f}%)")
            elif act == "focus":
                gimbal.auto_focus()
                msg = "✓ 自动对焦已触发"
            else:
                msg = f"未知变倍动作: {act} (可用: in/out/auto/focus)"
            return Response(msg, mimetype="text/plain; charset=utf-8")
        except Exception as e:
            return Response(f"✗ 变倍失败: {e}",
                            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav")
    def _qrnav_page():
        # no-store: 防止浏览器缓存旧页面 (改版后按钮/字段对不上)
        resp = Response(HTML_QR_NAV_PAGE,
                        mimetype="text/html; charset=utf-8")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/qr_nav_index")
    def _qrnav_index():
        """主页面: 完整保留参照控制台 + iframe 嵌入 QR 导航面板"""
        return Response(
            """<!DOCTYPE html><html><head><meta charset="utf-8">
            <title>QR→狗 + 云台控制台 (合并页)</title>
            <style>body{margin:0;background:#111}iframe{border:0;display:block}
            #cloud{width:100%;height:680px;border-bottom:1px solid #333}
            #qnav{width:100%;height:520px}</style></head>
            <body>
              <iframe id="cloud" src="/"></iframe>
              <iframe id="qnav" src="/qr_nav"></iframe>
            </body></html>""",
            mimetype="text/html; charset=utf-8")

    return app


# ============================================================
#  启动合并 HTTP (参照 + 附录), 端口默认 5003
# ============================================================
def start_combined_http_stream(port, owner=None):
    """启动 Flask: 参照的全部路由 (/, /raw, /detect, /3d, /dog, ...) +
    QR 导航附录 (/qr_nav, /qr_nav/pose, /qr_nav/auto, ...)"""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    # 1) 建参照的 app (含所有原版路由)
    app = _gpea._make_flask_app(owner=owner)
    # 2) 挂 QR 导航附录
    _register_qr_nav_routes(app)
    # 3) 后台启动
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               debug=False, use_reloader=False),
        daemon=True)
    t.start()
    return t


# ============================================================
#  main()
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='QR位姿 → 机器狗 自动导航 (附录模式, 完整保留参照的云台控制台)')

    # ── 与参照文件完全一致的参数 (14 个) ──
    parser.add_argument('--gimbal_ip', type=str, default='192.168.1.99',
                       help='云台 IP (默认: 192.168.1.99)')
    parser.add_argument('--gimbal_port', type=int, default=6001,
                       help='云台控制端口 (默认: 6001)')
    parser.add_argument('--qr_size', type=float, default=0.10,
                       help='二维码物理尺寸 米 (默认: 0.10)')
    parser.add_argument('--fov_x', type=float, default=60.0,
                       help='相机水平视场角 度 (默认: 60.0)')
    parser.add_argument('--model', type=str, default='./qr_yolov8.om',
                       help='OM 模型路径 (默认: ./qr_yolov8.om)')
    parser.add_argument('--device-id', type=int, default=0,
                       help='Ascend NPU 设备 ID (默认: 0)')
    parser.add_argument('--detector', type=str, default='opencv',
                       choices=['opencv', 'pyzbar'],
                       help='二维码检测器类型 (默认: opencv)')
    parser.add_argument('--square_size', type=int, default=800,
                       help='矫正后正方形图像边长 (默认: 800)')
    parser.add_argument('--conf_threshold', type=float, default=0.2,
                       help='YOLO 置信度阈值 (默认: 0.2)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='YOLO NMS 阈值 (默认: 0.5)')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['single', 'interactive', 'video'],
                       help='运行模式 (默认: video, 与参照一致)')
    parser.add_argument('--image', type=str, default=None,
                       help='直接使用指定图像 (仅 --mode single 生效)')
    parser.add_argument('--video', action='store_true',
                       help='启动时自动打开实时视频流 (默认开)')
    parser.add_argument('--display_pose', action='store_true',
                       help='在视频上显示位姿信息 (默认开)')

    # ── 本文件独有参数 (7 个) ──
    parser.add_argument('--cam', type=int, default=0,
                       help='本地相机设备号 (默认: 0, 备用)')
    parser.add_argument('--cam-w', type=int, default=1280,
                       help='本地相机宽度 (默认: 1280)')
    parser.add_argument('--cam-h', type=int, default=720,
                       help='本地相机高度 (默认: 720)')
    parser.add_argument('--port', type=int, default=5003,
                       help='Web 端口 (默认: 5003, 承载云台控制台 + QR 面板)')
    parser.add_argument('--dog-ip', type=str, default='192.168.1.103',
                       help='机器狗 IP (默认: 192.168.1.103, 传给参照)')
    parser.add_argument('--dog-port', type=int, default=43893,
                       help='机器狗端口 (默认: 43893, 传给参照)')
    parser.add_argument('--source', type=str, default='gimbal',
                       choices=['gimbal', 'cam'],
                       help='视频源: gimbal=RTSP(--gimbal_ip), cam=本地 (默认: gimbal)')

    args = parser.parse_args()

    # ── 配置回填 ──
    _QR_NAV_CFG["FOV_X"] = args.fov_x
    _qrnav_dog_port_holder[0] = args.port

    print("=" * 60)
    print("QR 位姿 → 机器狗 自动导航 (附录模式)")
    print("=" * 60)
    print(f"  参照参数 (与 gimbal_pose_estimation_ascend 同款):")
    print(f"    mode={args.mode}  model={args.model}")
    print(f"    gimbal={args.gimbal_ip}:{args.gimbal_port}")
    print(f"    qr_size={args.qr_size}  fov_x={args.fov_x}")
    print(f"    detector={args.detector}")
    print(f"  本文件参数:")
    print(f"    port={args.port}  dog={args.dog_ip}:{args.dog_port}")
    print("=" * 60)

    # ── 1) 创建参照的 estimator (与参照文件 main() 同款) ──
    try:
        estimator = _gpea.GimbalPoseEstimator(
            gimbal_ip=args.gimbal_ip,
            gimbal_port=args.gimbal_port,
            qr_size=args.qr_size,
            fov_x=args.fov_x,
            model_path=args.model,
            device_id=args.device_id,
            detector_type=args.detector,
            square_size=args.square_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
    except Exception as e:
        print(f"\n✗ 初始化 estimator 失败: {e}")
        return

    # ── 2) 启动合并 HTTP (参照全路由 + QR 导航附录) ──
    print(f"\n[HTTP] 启动合并服务器 (含云台控制台 + QR 导航附录) ...")
    start_combined_http_stream(port=args.port, owner=estimator)
    _qrnav_dog_port_holder[0] = args.port
    _qrnav_estimator_holder[0] = estimator   # 注入: 零 HTTP 读 dog.ready
    # 强制 QR 物理尺寸, 统一整条链路 (PnP object_points + 面板显示), 防止
    # config.json 里的旧值 (0.049) 覆盖成错值
    try:
        estimator.qr_size = float(_QR_NAV_CFG.get("QR_SIZE_M", 0.12))
        print(f"  QR 物理尺寸已强制统一: {estimator.qr_size:.3f} m "
              f"(覆盖 config/默认, 保证 PnP 与面板一致)")
    except Exception as _e:
        print(f"  ⚠ 强制 qr_size 失败: {_e}")
    print(f"  云台控制台: http://0.0.0.0:{args.port}/          ← 完整保留参照")
    print(f"  QR 导航    : http://0.0.0.0:{args.port}/qr_nav    ← 本文件新增")
    print(f"  合并页面   : http://0.0.0.0:{args.port}/qr_nav_index")

    # 阻止 estimator.start_video_stream() 重复启动 HTTP
    estimator._http_stream_started = True
    estimator._http_port = args.port

    # ── 3) 启动 QR 导航后台同步线程 (从 estimator 读 pose+frame) ──
    stop_evt = threading.Event()
    sync_t = threading.Thread(
        target=_qrnav_pose_sync_loop, args=(estimator, stop_evt), daemon=True)
    sync_t.start()
    print(f"  QR 导航同步线程已启动 (读取 estimator.latest_pose_data)")

    # 实时位姿线程: 周期用当前帧做单帧 PnP, 回写 latest_pose_data, 面板值实时刷新
    if float(_QR_NAV_CFG.get("LIVE_POSE_FPS", 0.0) or 0.0) > 0:
        live_t = threading.Thread(
            target=_qrnav_live_pose_loop, args=(estimator, stop_evt),
            daemon=True)
        live_t.start()
        print(f"  QR 导航实时位姿线程已启动 "
              f"({_QR_NAV_CFG['LIVE_POSE_FPS']:.1f} fps, 单帧 PnP)")

    # ── 4) 启动视频流 (参照的 start_video_stream: RTSP + YOLO + 推流 + 按键) ──
    print(f"\n[VIDEO] 启动视频流 (参照的 start_video_stream 同款) ...")
    if args.mode == 'video' or args.video or args.display_pose:
        estimator.start_video_stream(display_pose=True)
        try:
            while estimator.video_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            stop_evt.set()
            estimator.stop_video_stream()
            estimator.cleanup()
    else:
        print(f"  当前 mode={args.mode}, 视频流未启动 (与参照一致)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            stop_evt.set()
            estimator.cleanup()


if __name__ == "__main__":
    main()