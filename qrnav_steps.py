# -*- coding: utf-8 -*-
"""qrnav_steps.py — V1 兼容单步导航 (横移/变倍/auto/align/plan)

依赖: qrnav_config, qrnav_geometry, qrnav_dogctl
"""
from qrnav_config import (
    _QR_NAV_CFG, _qrnav_pose_lock, _qrnav_pose_box, _qrnav_estimator_holder,
)
from qrnav_geometry import pose_to_command
from qrnav_dogctl import _move_lateral_steps

# ============================================================
#  V1 路由函数 (保留, /qr_nav 走原版, /qr_nav2 走 V2 状态机)
# ============================================================
def lateral_step_only():
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return ("✗ 横偏数值未就绪: 暂无 QR 帧")
    cmd = pose_to_command(pose)
    lat_x  = cmd.get("lateral_x")
    fwd    = cmd.get("camera_z")
    yaw    = cmd.get("gimbal_yaw_deg")
    fill   = cmd.get("fill_ratio") or 0.0
    qr_s   = cmd.get("qr_size") or 0.0
    if lat_x is None:
        return "✗ 相机位置 X 未就绪"
    lat = float(lat_x)
    fwd_str = f"{fwd:.2f}m" if fwd is not None else "n/a"
    dead = float(cfg.get("EXEC_MIN_LAT", 0.005))
    extra = (f"(来源=camera_position[0], 相机位置 X={lat_x}, "
             f"yaw={yaw or 0:.1f}°, 位置 Z={fwd_str}, "
             f"fill={fill*100:.1f}%, qr_size={qr_s:.3f}m)")
    if abs(lat) <= dead:
        return (f"✓ 横偏 x={lat:+.3f}m ≤ {dead}m 已接近 0, 无需移动  {extra}")
    direction, step, n, per, ret = _move_lateral_steps(lat)
    if n > 1:
        seg = (f"分{n}段×{per:.2f}m "
               f"(绕开参照单次 {cfg.get('DOG_MAX_STEP_M', 0.3)}m 安全锁)")
    else:
        seg = f"单段 {per:.2f}m"
    return (f"✓ 横移 {step:.2f}m 向{direction} | {seg} | {ret}  {extra}")


def zoom_step_only():
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
        return (f"✗ 无法算变倍数值: QR 占比 fill=0")
    if fill >= cfg["ZOOM_TARGET"]:
        return (f"✓ 无需变倍: QR 占比 {fill*100:.1f}% ≥ "
                f"{cfg['ZOOM_TARGET']*100:.0f}% (当前 {cur_z:.1f}x)")
    if est is None or getattr(est, "gimbal", None) is None:
        return (f"✗ 无云台句柄, 无法变倍")
    k = (cfg["ZOOM_TARGET"] / fill) ** 0.5
    zmin = getattr(est, "_zoom_min", 1.0)
    zmax = getattr(est, "_zoom_max", 30.0)
    target = max(zmin, min(zmax, cur_z * k))
    est.gimbal.zoom_to_magnification(target)
    est.current_zoom_mag = target
    return (f"变倍数值 {cur_z:.1f}x → {target:.1f}x "
            f"(QR 占比 {fill*100:.1f}% → 目标 {cfg['ZOOM_TARGET']*100:.0f}%)")


# ============================================================
#  V1 兼容函数 (V2 页面也用: 自动一步 / 仅横向对齐 / 一键执行)
#  把 V1 的 auto/align/exec_plan 算法并到 V2, 这样 V2 页面就能独立完成
#  所有"机器人运动"操作, 不必再切回 V1 页面
# ============================================================
def _auto_zoom_step(fill):
    """V1 风格: 按当前 fill_ratio 自动变倍一次 (云台给的 fill → 目标倍率)"""
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


def auto_step_nav():
    """V1 风格: 自动一步 (横移 + 变倍 串行)

    与 V2 的 6 段式不同 — 这是一键 (用最近一次 pose, 不重新捕获)
    """
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "无 QR 数据, 跳过"
    cmd = pose_to_command(pose)
    if not cmd["ready"]:
        return "无 QR 数据, 跳过"
    lat  = cmd["lateral_x"]
    fwd  = cmd["camera_z"]
    yaw  = cmd["gimbal_yaw_deg"]
    fill = cmd.get("fill_ratio") or 0.0
    zoom_needed = cmd.get("zoom_needed", False)
    if lat is None or fwd is None:
        return "相机位置未就绪"

    abs_lat = abs(lat)
    if abs_lat > cfg["ALIGN_LAT"]:
        direction, step, n, per, ret = _move_lateral_steps(lat)
        seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
        msgs = [f"{cmd['turn_dir']}: 横移 {step:.2f}m ({seg}) "
                f"(相机位置 X={lat:+.3f}m, 位置 Z={fwd:.2f}m, "
                f"yaw={yaw:.1f}°)" + " | " + ret]
        if zoom_needed:
            msgs.append(_auto_zoom_step(fill))
        return " | ".join(msgs)
    if zoom_needed:
        return (f"✓ 横偏已对齐 ({lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m) | "
                + _auto_zoom_step(fill) + " | (前进/旋转请手动按钮)")
    return (f"✓ 已对齐 (横偏 {lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m, "
            f"QR 占比 {fill*100:.1f}% ≥ {cfg['ZOOM_TARGET']*100:.0f}%, "
            f"位置 Z={fwd:.2f}m, yaw={yaw:.1f}°)")


def align_step_nav():
    """V1 风格: 仅横向对齐 (横移, 不动变倍)"""
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "无 QR 数据"
    cmd = pose_to_command(pose)
    if not cmd["ready"]:
        return "无 QR 数据"
    lat = cmd["lateral_x"]
    yaw = cmd["gimbal_yaw_deg"]
    if lat is None:
        return "相机位置 X 未就绪"
    dead = float(cfg.get("EXEC_MIN_LAT", 0.005))
    if abs(lat) <= dead:
        return (f"✓ 已接近 0 (相机位置 X={lat:+.3f}m ≤ {dead}m, "
                f"yaw={yaw:.1f}°)")
    direction, step, n, per, ret = _move_lateral_steps(lat)
    seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
    return (f"↔ 横移 {step:.2f}m 向{direction} ({seg}) | {ret} "
            f"(相机位置 X={lat:+.3f}m, yaw={yaw:.1f}°)")


def exec_plan_step():
    """V1 风格: 一次性按 plan 执行横移 + 变倍 (用于前端循环"一键执行")

    V2 中也可用, 但要注意它不触发云台中心对齐 (那是 ② ② 的事)
    """
    cfg = _QR_NAV_CFG
    with _qrnav_pose_lock:
        pose = _qrnav_pose_box[0]
    if pose is None:
        return "无 QR 数据"
    cmd = pose_to_command(pose)
    if not cmd.get("ready"):
        return "无 QR 数据"
    lat  = cmd.get("lateral_x")
    fwd  = cmd.get("camera_z")
    if lat is None or fwd is None:
        return "相机位置未就绪"
    abs_lat = abs(lat)
    msgs = []
    if abs_lat > cfg["ALIGN_LAT"]:
        ld, step, n, per, ret = _move_lateral_steps(lat)
        seg = f"分{n}段×{per:.2f}m" if n > 1 else f"单段{per:.2f}m"
        msgs.append(f"↔ 横移 {step:.2f}m 向{ld} ({seg}): " + ret)
    else:
        msgs.append(f"✓ 横偏已对齐 ({lat:+.3f}m ≤ {cfg['ALIGN_LAT']}m)")
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
            msgs.append(f"🔍 变倍→{target:.1f}x (fill {fill*100:.1f}% → "
                        f"{cfg['ZOOM_TARGET']*100:.0f}%)")
        else:
            msgs.append(f"🔍 变倍 OK (fill {fill*100:.1f}%)")
    msgs.append(f"位置 Z={fwd:.2f}m / yaw {cmd.get('gimbal_yaw_deg',0.0):.1f}° "
                f"(前进/旋转请手动按钮)")
    return " | ".join(msgs)
