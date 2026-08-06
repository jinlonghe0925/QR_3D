#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机器狗导航模块
包含坐标系转换、移动参数计算、以及 Windows 模拟导航器。

对应 app_config.json 的 dog_navigator 节:
    ip:                机器狗 IP (仅 Linux 有效)
    port:              机器狗端口 (仅 Linux 有效)
    stop_distance_m:   安全停止距离 (米)
    sim_mode:          是否模拟模式 (Windows 强制 True)
    auto_stand:        是否自动起立
"""

import time
import math
from typing import Optional, Tuple


# ============================================================
# QR 坐标系 → 狗体相对坐标
# ============================================================
def qr_position_to_dog(camera_position) -> Tuple[float, float, float]:
    """QR Pose 中的 camera_position → 狗体相对坐标。

    坐标系定义:
        QR 坐标系 (竖直安装): X=右, Y=上, Z=向板内(相机方向)
        Dog 坐标系:          X=前, Y=右, Z=上

    转换关系:
        Dog_X =  QR_Z    (前进距离)
        Dog_Y = -QR_X    (右侧距离)
        Dog_Z = -QR_Y    (高度差, 通常忽略)

    参数:
        camera_position: [x, y, z] 相机在 QR 坐标系中的位置 (米)

    返回:
        (dog_x, dog_y, dog_z) 狗体相对坐标

    异常:
        ValueError: camera_position 格式无效
    """
    if camera_position is None or len(camera_position) != 3:
        raise ValueError("camera_position 必须为 [x, y, z]")
    x_qr, y_qr, z_qr = (float(v) for v in camera_position)
    return z_qr, -x_qr, -y_qr


def compute_dog_move(x_dog: float, y_dog: float, z_dog: float,
                     stop_distance: float = 0.8) -> Optional[Tuple[float, float, float]]:
    """将狗体相对坐标转换为 Dog.move(x, y, z) 参数。

    移动距离会减去 stop_distance，确保停在 QR 前方安全距离而非撞上去。

    参数:
        x_dog:         狗体前方距离 (米)
        y_dog:         狗体右侧距离 (米)
        z_dog:         狗体高度差 (米, 未使用)
        stop_distance: 安全停止距离 (米)

    返回:
        (dx, dy, dz_deg) 或 None (已在安全距离内)
    """
    horizontal_dist = math.hypot(x_dog, y_dog)
    if horizontal_dist <= stop_distance:
        return None

    # 按比例扣除安全距离
    ratio = (horizontal_dist - stop_distance) / horizontal_dist
    dx = x_dog * ratio
    dy = -y_dog * ratio      # 狗体 Y(右+) → SDK y(左+)，符号取反
    dz = math.degrees(math.atan2(y_dog, x_dog))
    return dx, dy, dz


# ============================================================
# 机器狗导航器 (Windows 模拟模式)
# ============================================================
class RobotDogNavigator:
    """X30 机器狗导航控制器 (Windows 模拟版本)。

    Windows 上无法连接真狗，所有运动指令打印到控制台。
    API 与 Linux robot_dog_bridge.RobotDogNavigator 完全一致。

    用法:
        nav = RobotDogNavigator(config=app_config)
        nav.connect()
        nav.go_to_from_pose(pose_data)
        nav.disconnect()
    """

    def __init__(self, ip: str = "192.168.1.103", port: int = 43893,
                 auto_stand: bool = True, stop_distance: float = 0.8,
                 config: dict = None):
        """初始化导航器。

        参数:
            ip:            机器狗 IP (仅 Linux)
            port:          机器狗端口 (仅 Linux)
            auto_stand:    是否自动起立
            stop_distance: 安全停止距离 (米)
            config:        应用配置字典 (app_config.json), 会覆盖上述参数
        """
        # ── 从配置覆盖 ──
        dog_cfg = config.get("dog_navigator", {}) if config else {}
        self.stop_distance = dog_cfg.get("stop_distance_m", stop_distance)
        self._dog = None
        self._ready = False
        self._sim = dog_cfg.get("sim_mode", True)  # Windows 始终模拟
        self._auto_stand = dog_cfg.get("auto_stand", auto_stand)

    def connect(self) -> bool:
        """起立并准备运动 (Win 跳过，直接进模拟模式)。"""
        self._ready = True
        self._sim = True
        print("[Dog] Windows 模拟模式 — 无法连接真狗，运动指令仅打印日志")
        return True

    def disconnect(self):
        """停止运动，关闭连接。"""
        self._ready = False
        print("[Dog] 已断开")

    @property
    def ready(self) -> bool:
        """导航器是否就绪。"""
        return self._ready

    def go_to(self, x: float, y: float, z: float) -> bool:
        """相对移动指令。

        参数:
            x: 前进距离 (米)
            y: 左移距离 (米)
            z: 旋转角度 (度)
        """
        if not self._ready:
            print("[Dog] 未就绪")
            return False

        print(f"[Dog] move(x={x:.3f}m, y={y:.3f}m, z={z:.2f}°)")

        if self._sim:
            return self._sim_move(x, y, z)

        return False  # Win 永远不进真狗路径

    def go_to_from_pose(self, pose: dict) -> bool:
        """由 QR Pose 生成狗体位移并导航。

        流程:
            1. camera_position → dog_relative (qr_position_to_dog)
            2. dog_relative → Dog.move(dx, dy, dz) (compute_dog_move)
            3. 发送 move 指令

        参数:
            pose: 位姿数据 (含 camera_position)

        返回:
            True=导航成功/已到位, False=失败
        """
        cam = pose.get("camera_position")
        if cam is None:
            print("[Dog] Pose 中缺少 camera_position")
            return False

        try:
            x_dog, y_dog, z_dog = qr_position_to_dog(cam)
        except (TypeError, ValueError) as e:
            print(f"[Dog] QR 坐标转换失败: {e}")
            return False

        print("[Transform] QR cam=({:.3f}, {:.3f}, {:.3f}) m".format(
            float(cam[0]), float(cam[1]), float(cam[2])))
        print("[Transform] QR->Dog=({:.3f}, {:.3f}, {:.3f}) m".format(
            x_dog, y_dog, z_dog))

        horizontal_dist = math.hypot(x_dog, y_dog)
        if horizontal_dist <= self.stop_distance:
            print("[Dog] 二维码水平距离 {:.3f}m，"
                  "已在安全距离 {:.3f}m 内".format(
                horizontal_dist, self.stop_distance))
            return True

        move_params = compute_dog_move(x_dog, y_dog, z_dog,
                                       stop_distance=self.stop_distance)
        if move_params is None:
            return True  # 已在安全距离

        dx, dy, dz = move_params
        return self.go_to(dx, dy, dz)

    def _sim_move(self, x, y, z) -> bool:
        """模拟移动：打印目标 → sleep → 返回成功。"""
        print("[Dog][SIM] ({:.2f}, {:.2f}, {:.1f}°) → 到达 ✓".format(x, y, z))
        time.sleep(0.3)
        return True
