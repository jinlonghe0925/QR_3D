# -*- coding: utf-8 -*-
"""qrnav_phases.py — 两段式捕获、横移、云台与变倍状态机。"""
import time
import threading
import copy
import math

import numpy as np

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_pose_lock, _qrnav_pose_box, _qrnav_pose_ver,
    _qrnav_estimator_holder,
)
from qrnav_geometry import (
    pose_to_command, _qr_size_est,
    _get_gimbal_angles, _ensure_focal_for_zoom, describe_pose_report,
)
from qrnav_dogctl import (
    _move_lateral_steps, _ensure_dog_ready, _is_dog_fully_ready,
    _dog_msg_bad,
)
# ============================================================
#  V2 新增: 两段式状态机
# ============================================================
#  状态字段:
#    phase            : "idle" | "first_captured" | "first_executed"
#                    | "dog_ready" | "second_captured" | "done"
#    first_pose       : dict | None   第一次捕获的 pose_data
#    second_pose      : dict | None   第二次捕获的 pose_data
#    first_capture_msg: str           ① 捕获结果文本
#    first_exec_msg   : str           ① 执行结果文本
#    second_capture_msg: str          ② 捕获结果文本
#    second_exec_msg  : str           ② 执行结果文本
#    first_capture_ts : float         ① 捕获时间戳
#    second_capture_ts: float         ② 捕获时间戳
_TWO_PHASE_STATE = {
    "phase":             "idle",
    "first_pose":        None,
    "second_pose":       None,
    "first_capture_msg": "",
    "first_pose_report": "",
    "first_exec_msg":    "",
    "second_capture_msg":"",
    "second_pose_report":"",
    "second_exec_msg":   "",
    "first_capture_ts":  0.0,
    "second_capture_ts": 0.0,
    "last_update_ts":    0.0,
}
_TWO_PHASE_LOCK = threading.Lock()


def _phase_get():
    """返回状态快照 (供前端轮询) — 屏蔽不可序列化的 pose 对象"""
    with _TWO_PHASE_LOCK:
        s = dict(_TWO_PHASE_STATE)
    # 锁外计算概要，避免 pose_to_command 内部读取位姿锁时扩大临界区。
    s["first_pose_summary"] = _pose_summary(s.get("first_pose"))
    s["second_pose_summary"] = _pose_summary(s.get("second_pose"))
    s.pop("first_pose", None)
    s.pop("second_pose", None)
    return s


def _pose_summary(pose):
    """把 pose_data 压缩成 5 个关键数字 + 1 个变倍建议, 供 UI 显示"""
    if pose is None:
        return None
    try:
        cmd = pose_to_command(pose)
        return {
            "lateral_x":      cmd.get("lateral_x"),
            "camera_z":       cmd.get("camera_z"),
            "position_source":cmd.get("position_source"),
            "lateral_world":  cmd.get("lateral_world"),
            "forward_world":  cmd.get("forward_world"),
            "fill_ratio":     cmd.get("fill_ratio"),
            "qr_size":        cmd.get("qr_size"),
            "gimbal_yaw_deg": cmd.get("gimbal_yaw_deg"),
            "distance":       cmd.get("distance"),
            "hint":           cmd.get("hint"),
        }
    except Exception:
        return None


def _phase_set(**kw):
    with _TWO_PHASE_LOCK:
        for k, v in kw.items():
            _TWO_PHASE_STATE[k] = v
        _TWO_PHASE_STATE["last_update_ts"] = time.time()


def _phase_reset():
    with _TWO_PHASE_LOCK:
        for k in list(_TWO_PHASE_STATE.keys()):
            if k == "phase":
                _TWO_PHASE_STATE[k] = "idle"
            elif k == "last_update_ts":
                _TWO_PHASE_STATE[k] = time.time()
            else:
                _TWO_PHASE_STATE[k] = (0.0 if k.endswith("_ts") else
                                        (None if k.endswith("_pose") else ""))


# ── 统一的『多帧采集』底层 (两次捕获都调这个) ──
def _do_capture_once(phase_name):
    """两次捕获的底层算法 (完全相同): 触发多帧采集 + 同步结果到全局 pose_box

    Returns:
        (ok, pose_data, msg, report)
    """
    est = _qrnav_estimator_holder[0]
    if est is None:
        return False, None, "✗ estimator 未注入", ""
    frame = getattr(est, "current_frame", None)
    if frame is None:
        return False, None, "✗ 无视频帧 (视频流未启动 / 还没收到帧)", ""
    if getattr(est, "_est_busy", False):
        return False, None, "⏳ 上一次采集还没跑完, 请稍等", ""
    # estimate_pose_from_image 会在每个 PnP 帧前同步读取真实 FOV。
    # 此处不再预查询，避免两个查询同时争用云台的 FOV Event/返回值。
    try:
        previous_pose = getattr(est, "latest_pose_data", None)
        capture_angles = _get_gimbal_angles()
        history = getattr(est, "_pose_history", None)
        if history is not None:
            history.clear()
        est._start_estimate_async(frame)
    except Exception as e:
        return False, None, f"✗ 触发失败: {e}", ""
    time.sleep(0.3)
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if not getattr(est, "_est_busy", False):
            break
        time.sleep(0.1)
    if getattr(est, "_est_busy", False):
        return False, None, "✗ 采集超时，本次横偏未更新", ""
    pd = getattr(est, "latest_pose_data", None)
    if pd is None or pd is previous_pose:
        return False, None, "✗ 估计失败 (未检测到 QR / 位姿解算失败)", ""
    pd = copy.deepcopy(pd)
    (pd["gimbal_yaw_deg"], pd["gimbal_pitch_deg"],
     pd["gimbal_roll_deg"]) = capture_angles
    with _qrnav_pose_lock:
        _qrnav_pose_box[0] = pd
        _qrnav_pose_ver[0] += 1
    cmd = pose_to_command(pd)
    # ★ Bug 修复: cmd 字典没有 'qr_size' 字段, 直接用 None 格式化会抛 TypeError
    #   用 (cmd.get(...) or 0) 兜底, 让 msg 字符串必能生成
    qr_size_val = _qr_size_est() or 0.0
    lat_val  = cmd.get('lateral_x')
    fwd_val  = cmd.get('camera_z')
    fill_val = cmd.get('fill_ratio') or 0.0
    yaw_val  = cmd.get('gimbal_yaw_deg')
    lat_str  = f"{lat_val:+.3f}m"  if lat_val  is not None else "--"
    fwd_str  = f"{fwd_val:.2f}m"   if fwd_val  is not None else "--"
    yaw_str  = f"{yaw_val:.1f}°"   if yaw_val  is not None else "--"
    msg = (f"✓ {phase_name} 捕获成功: 相机位置 X={lat_str} (作为狗的位置), "
           f"位置 Z={fwd_str}, "
           f"占比 {fill_val*100:.1f}%, "
           f"QR尺寸 {qr_size_val:.3f}m, "
           f"yaw={yaw_str}")
    report = describe_pose_report(pd, cmd)
    print(report)
    return True, pd, msg, report


def first_capture_once():
    """① 第一次捕获: 触发多帧采集, 保存到 first_pose

    与第二次的区别: 仅『存数据』, 不动云台, 不动狗
    """
    with _TWO_PHASE_LOCK:
        phase = _TWO_PHASE_STATE["phase"]
    # 允许从 idle / 任何非执行中阶段重抓 (用户能自由重拍)
    if phase in ("first_executed", "dog_ready", "second_captured", "done"):
        return (f"⚠ 当前阶段 {phase}, 若想重新 ① 捕获请先 ↺ 重置")
    ok, pd, msg, report = _do_capture_once("① 第一次")
    if not ok:
        return msg
    _phase_set(phase="first_captured",
               first_pose=pd,
               first_capture_msg=msg,
               first_pose_report=report,
               first_capture_ts=time.time())
    return msg


def second_capture_once():
    """② 第二次捕获: 触发多帧采集, 保存到 second_pose

    底层算法与第一次完全相同; 区别只是『保存到 second_pose + 状态推进』
    """
    with _TWO_PHASE_LOCK:
        phase = _TWO_PHASE_STATE["phase"]
    if phase not in ("dog_ready", "first_executed"):
        return (f"✗ 当前阶段 {phase}, 应先完成 ① (捕获→横移→标记狗就位)")
    ok, pd, msg, report = _do_capture_once("② 第二次")
    if not ok:
        return msg
    _phase_set(phase="second_captured",
               second_pose=pd,
               second_capture_msg=msg,
               second_pose_report=report,
               second_capture_ts=time.time())
    return msg


def mark_dog_ready():
    """✓ 标记: 狗已就位 (一般由 exec_phase1_lateral 自动推进, 这按钮是兜底)"""
    with _TWO_PHASE_LOCK:
        phase = _TWO_PHASE_STATE["phase"]
    if phase == "dog_ready":
        return "✓ 已就绪 (跳过, 无需重复标记)"
    if phase != "first_executed":
        return f"✗ 当前阶段 {phase}, 应先完成 ① 捕获 + 横移"
    _phase_set(phase="dog_ready")
    return "✓ 已标记: 狗就位 → 可按 ② 第二次捕获"


def reset_phase():
    """↺ 重置: 清空所有阶段, 回到 idle"""
    _phase_reset()
    return "✓ 已重置: 回到阶段 0 (idle)"


# ============================================================
#  一键全流程 (▶ 后台线程跑完整 6 段, 实时更新状态给前端轮询)
#  把原来 6 次手动点击 → 1 次, 前端轮询 /qr_nav2/auto_full_status
#  即可看到当前步骤 / 进度 / 状态文本
# ============================================================
_qrnav_auto_full_status = [None]   # {running, stage, msg, pct, ts}

def _set_af_status(running, stage, msg, pct):
    _qrnav_auto_full_status[0] = {
        "running": running, "stage": stage, "msg": msg, "pct": pct,
        "ts": time.time()
    }

def _is_af_running():
    s = _qrnav_auto_full_status[0]
    return s is not None and s.get("running", False)

def _is_phase_ok(msg):
    return bool(msg) and ("ERR" not in msg) and ("✗" not in msg)

def _auto_full_worker():
    """后台线程: 跑完整 6 段流程, 每步更新 _qrnav_auto_full_status"""
    try:
        # 0) 初始化狗 (如未就绪)
        if not _is_dog_fully_ready():
            _set_af_status(True, "init", "🦿 0/6 初始化狗 (起立+力控+开始运动) ...", 5)
            ok, msg = _ensure_dog_ready(force_init=True)
            if not ok:
                _set_af_status(False, "error", f"✗ 狗初始化失败: {msg}", -1)
                return
        else:
            _set_af_status(True, "init", "✓ 0/6 狗已就绪, 跳过初始化", 5)

        # 1) 第一次捕获
        _set_af_status(True, "capture1", "📸 1/6 第一次捕获 (多帧+PnP, 约 3-5 秒) ...", 20)
        msg = first_capture_once()
        if not _is_phase_ok(msg):
            _set_af_status(False, "error", f"✗ 第一次捕获失败: {msg}", -1)
            return

        # 2) 执行横移 (auto-mark dog_ready 在 exec_phase1_lateral 内部完成)
        _set_af_status(True, "lateral", "↔️ 2/6 执行横移 (狗对齐 QR 中线) ...", 40)
        msg = exec_phase1_lateral()
        if not _is_phase_ok(msg):
            _set_af_status(False, "error", f"✗ 横移失败: {msg}", -1)
            return

        # 3) 等待 2 秒让狗稳定停下 (避免第二次捕获时还在动)
        for i in range(4):
            time.sleep(0.5)
            _set_af_status(True, "settle", f"⏸ 3/6 等待狗稳定 ({i+1}/4 × 0.5s) ...", 55 + i * 2)

        # 4) 第二次捕获
        _set_af_status(True, "capture2", "📸 4/6 第二次捕获 (底层算法同①) ...", 65)
        msg = second_capture_once()
        if not _is_phase_ok(msg):
            _set_af_status(False, "error", f"✗ 第二次捕获失败: {msg}", -1)
            return

        # 5) 云台+变倍
        _set_af_status(True, "gimbal_zoom", "🔄 5/6 云台+变倍 (QR 拉回中心 + 变倍) ...", 85)
        msg = exec_phase2_gimbal_zoom()
        if not _is_phase_ok(msg):
            _set_af_status(False, "error", f"✗ 云台+变倍失败: {msg}", -1)
            return

        _set_af_status(False, "done", "✓ 6/6 全流程完成 (狗就位 + QR 居中 + 变倍)", 100)
    except Exception as e:
        _set_af_status(False, "error", f"✗ ERR {e}", -1)


def exec_phase1_lateral():
    """① 执行横移: 基于 first_pose，自动开始、横移、停止，不动云台。

    ★ 设计: 只算 1 次 pose (在「📸 第一次捕获」时), 横移直接用这个 pose
      - ① 按钮只做 capture+pose；↔️ 按钮完成运动使能、横移和零速停止
      - 不再每次横移都重新 capture, 避免延迟 / 重复计算
      - 兜底: pose 为空时让用户去按 ① 第一次捕获
    运动指令流及停止帧发送成功后推进到 dog_ready；当前接口没有实际位移反馈。
    """
    with _TWO_PHASE_LOCK:
        phase = _TWO_PHASE_STATE["phase"]
        pose = _TWO_PHASE_STATE["first_pose"]

    if phase == "idle":
        return f"✗ 当前阶段 {phase}, 应先按 📸 ① 第一次捕获 (算 pose, 约 3 秒)"
    if pose is None:
        return f"✗ first_pose 为空 (phase={phase}), 请先按 📸 ① 第一次捕获"

    # 直接使用捕获①的相机位置 X 作为狗的横移量，不做基座投影。
    with _qrnav_pose_lock:
        _qrnav_pose_box[0] = pose
    cmd = pose_to_command(pose)
    lat = cmd.get("lateral_x")
    if lat is None or not math.isfinite(float(lat)):
        return "✗ 相机位置 X 未就绪"

    dead = float(_QR_NAV_CFG.get("EXEC_MIN_LAT", 0.005))
    if abs(lat) <= dead:
        msg = (f"✓ ① 横偏 x={lat:+.3f}m ≤ {dead}m 已对齐, 无需移动  "
               f"(来源=camera_position[0], 相机位置作为狗的位置)")
    else:
        direction, step, n, per, ret = _move_lateral_steps(lat)
        if _dog_msg_bad(ret) or n <= 0:
            msg = f"✗ ① 横移失败 (相机位置 X={lat:+.3f}m) | {ret}"
            _phase_set(phase="first_captured", first_exec_msg=msg)
            return msg
        seg = (f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m")
        msg = (f"✓ ① 已自动开始并完成横移，目标 {step:.3f}m 向{direction} ({seg})，已停止 | {ret}  "
               f"(来源=camera_position[0], 相机位置 X={lat:+.3f}m)")

    # 运动使能、指令流及停止帧全部成功后推进；到位精度取决于速度标定。
    _phase_set(phase="dog_ready", first_exec_msg=msg)
    return msg


def _compute_gimbal_center_offset(pose, img_w, img_h):
    """根据 QR 在图像中的位置, 计算让 QR 居中所需的云台 yaw/pitch 增量 (度)

    推导:
      图像中心   = (img_w/2, img_h/2)
      QR 中心     = image_points 4 角均值
      像素偏移   = (qr_cx - img_w/2, qr_cy - img_h/2)
      当前 FOV   = FOV_X / zoom (横向), FOV_Y / zoom (纵向)
                   FOV_Y 按宽高比推 (FOV_X * h / w)
      角速度     = 1 像素 ≈ (FOV / 边长) 度
      yaw 增量   = +dx_px * (cur_fov_x / img_w) * sign_yaw
        QR 偏右 → dx>0 → 若 sign=+1, 云台 + yaw (相机右转, 把 QR 带回中心)
      pitch 增量 = -dy_px * (cur_fov_y / img_h) * sign_pitch
        QR 偏下 → dy>0 → 若 sign=+1, -dy → 云台 - pitch (相机上抬, 把 QR 带回中心)

    Returns:
        (yaw_offset_deg, pitch_offset_deg, cur_fov_x, cur_fov_y, qr_cx, qr_cy)
    """
    cfg = _QR_NAV_CFG
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 居中计算使用生成当前 pose 的同一份内参；旧 pose 没保存内参时才查询兜底。
    try:
        cam = np.asarray(pose.get("camera_matrix"), dtype=float).reshape(3, 3)
        fx, fy = float(cam[0, 0]), float(cam[1, 1])
        if not (math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0):
            raise ValueError("invalid focal pixels")
        cur_fov_x = 2.0 * math.degrees(math.atan(float(img_w) / (2.0 * fx)))
        cur_fov_y = 2.0 * math.degrees(math.atan(float(img_h) / (2.0 * fy)))
    except (TypeError, ValueError):
        est = _qrnav_estimator_holder[0]
        cur_zoom = float(getattr(est, "current_zoom_mag", 1.0) or 1.0)
        cur_zoom = max(1.0, cur_zoom)
        cur_fov_x, cur_fov_y = _ensure_focal_for_zoom(
            cur_zoom, int(img_w), int(img_h), force_query=True)
    img_pts = pose.get("image_points")
    if img_pts is None or len(img_pts) < 4:
        return 0.0, 0.0, cur_fov_x, cur_fov_y, img_w / 2.0, img_h / 2.0
    pts = np.asarray(img_pts, dtype=float).reshape(-1, 2)
    qr_cx, qr_cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    dx_px = qr_cx - img_w / 2.0
    dy_px = qr_cy - img_h / 2.0
    sign_y   = float(cfg.get("CENTER_YAW_SIGN",   +1))
    sign_p   = float(cfg.get("CENTER_PITCH_SIGN", +1))
    yaw_off   =  sign_y * dx_px * (cur_fov_x / float(img_w))
    pitch_off = -sign_p * dy_px * (cur_fov_y / float(img_h))
    return float(yaw_off), float(pitch_off), cur_fov_x, cur_fov_y, qr_cx, qr_cy


def exec_phase2_gimbal_zoom():
    """② 执行云台+变倍: 基于 second_pose, 旋转云台到图像中心 + 按需变倍

    这是 V2 关键变化:
      - 第一次执行只动狗, 不动云台
      - 第二次执行才让云台把 QR 转到画面中心
      - 变倍仍按 fill_ratio 走 V1 公式
    """
    with _TWO_PHASE_LOCK:
        phase = _TWO_PHASE_STATE["phase"]
        pose = _TWO_PHASE_STATE["second_pose"]
    if phase != "second_captured":
        return f"✗ 当前阶段 {phase}, 应先按 ② 第二次捕获"
    if pose is None:
        return "✗ 第二次 pose 为空"

    est = _qrnav_estimator_holder[0]
    gimbal = getattr(est, "gimbal", None) if est else None
    if gimbal is None:
        return "✗ 云台未就绪 (gimbal 句柄为空)"

    frame = getattr(est, "current_frame", None)
    if frame is None:
        return "✗ 无视频帧 (无法算图像尺寸 / 中心)"
    img_h, img_w = frame.shape[:2]

    yaw_off, pitch_off, fov_x, fov_y, qr_cx, qr_cy = \
        _compute_gimbal_center_offset(pose, img_w, img_h)

    # 死区: 像素偏移 < CENTER_DEAD_PX 就不动云台
    dead_px = float(_QR_NAV_CFG.get("CENTER_DEAD_PX", 4.0))
    dx_px = qr_cx - img_w / 2.0
    dy_px = qr_cy - img_h / 2.0

    msgs = []
    if abs(dx_px) < dead_px and abs(dy_px) < dead_px:
        msgs.append(f"✓ QR 已居中 (dx={dx_px:+.1f}px, dy={dy_px:+.1f}px, "
                    f"死区 {dead_px}px) — 云台无需旋转")
    else:
        # 限幅: 单次增量不超过当前 FOV/2
        yaw_off   = max(-fov_x / 2.0, min(fov_x / 2.0, yaw_off))
        pitch_off = max(-fov_y / 2.0, min(fov_y / 2.0, pitch_off))
        try:
            cur_yaw, cur_pitch, _ = gimbal.get_current_angles()
        except Exception:
            cur_yaw, cur_pitch = 0.0, 0.0
        target_yaw   = max(-180.0, min(180.0, cur_yaw + yaw_off))
        target_pitch = max(-90.0,  min(90.0,  cur_pitch + pitch_off))
        try:
            est._gimbal_rotate(target_yaw, target_pitch, 0.0)
        except Exception as e:
            return f"✗ 云台旋转失败: {e}"
        msgs.append(
            f"🔄 云台 yaw {cur_yaw:.1f}°→{target_yaw:.1f}° "
            f"pitch {cur_pitch:.1f}°→{target_pitch:.1f}° "
            f"(增量 {yaw_off:+.2f}°/{pitch_off:+.2f}°, FOV_x={fov_x:.1f}°, "
            f"QR 偏 {dx_px:+.1f}px/{dy_px:+.1f}px)")

    # 变倍 (按需)
    with _qrnav_pose_lock:
        _qrnav_pose_box[0] = pose
    cmd = pose_to_command(pose)
    fill = cmd.get("fill_ratio") or 0.0
    if fill > 1e-4 and fill < _QR_NAV_CFG["ZOOM_TARGET"]:
        cur_z = getattr(est, "current_zoom_mag", 1.0)
        k = (_QR_NAV_CFG["ZOOM_TARGET"] / fill) ** 0.5
        zmin = getattr(est, "_zoom_min", 1.0)
        zmax = getattr(est, "_zoom_max", 30.0)
        target_z = max(zmin, min(zmax, cur_z * k))
        try:
            gimbal.zoom_to_magnification(target_z)
            est.current_zoom_mag = target_z
            msgs.append(f"🔍 变倍 {cur_z:.1f}x→{target_z:.1f}x "
                        f"(fill {fill*100:.1f}%→"
                        f"{_QR_NAV_CFG['ZOOM_TARGET']*100:.0f}%)")
        except Exception as e:
            msgs.append(f"⚠ 变倍失败: {e}")
    else:
        msgs.append(f"🔍 变倍 OK (fill {fill*100:.1f}%)")

    msg = " | ".join(msgs)
    _phase_set(phase="done", second_exec_msg=msg)
    return msg



