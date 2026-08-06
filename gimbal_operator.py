#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云台控制模块
封装云台旋转、变倍、对焦等操作。

从 app_config.json 的 gimbal 节读取参数:
    rotation_step_deg:   手动微调步长 (度), 默认 5.0
    zoom_step:           变倍步长, 默认 1.0
    zoom_min / zoom_max: 变倍范围, 默认 1.0~30.0
    angle_dead_zone_deg: 角度死区 (度), 默认 0.1
    gimbal_stable_delay_sec: 旋转后稳定等待 (秒), 默认 0.5
"""

import time


class GimbalOperator:
    """云台操作器 — 封装旋转、变倍、对焦等操作。

    用法:
        op = GimbalOperator(gimbal=gimbal_controller, config=app_config)
        op.rotate_to_target()
        op.manual_adjust(dyaw=5.0, dpitch=0.0)
        op.zoom(+1.0)
        op.auto_focus()
    """

    def __init__(self, gimbal=None, config: dict = None):
        # ── 云台控制器实例 ──
        self.gimbal = gimbal

        # ── 从配置读取参数 ──
        gimbal_cfg = config.get("gimbal", {}) if config else {}

        # 旋转目标角度 (由位姿计算得出)
        self.target_yaw = 0.0
        self.target_pitch = 0.0

        # 当前云台角度 (运行时更新)
        self.current_gimbal_yaw = 0.0
        self.current_gimbal_pitch = 0.0

        # 当前变倍倍率
        self.current_zoom_mag = 1.0

        # 手动微调步长 (度)
        self.rotation_step = gimbal_cfg.get("rotation_step_deg", 5.0)
        self._rotation_step_min = gimbal_cfg.get("rotation_step_min", 1.0)
        self._rotation_step_max = gimbal_cfg.get("rotation_step_max", 45.0)

        # 变倍参数
        self.zoom_step_val = gimbal_cfg.get("zoom_step", 1.0)
        self._zoom_min = gimbal_cfg.get("zoom_min", 1.0)
        self._zoom_max = gimbal_cfg.get("zoom_max", 30.0)

        # 角度死区
        self.angle_dead_zone = gimbal_cfg.get("angle_dead_zone_deg", 0.1)

        # 旋转后稳定等待时间
        self.stable_delay = gimbal_cfg.get("gimbal_stable_delay_sec", 0.5)

    def set_gimbal(self, gimbal):
        """设置/更新云台控制器实例。"""
        self.gimbal = gimbal

    def rotate_to_target(self):
        """旋转云台到已计算的目标角度。"""
        if self.gimbal is None:
            print("[Gimbal] 云台未连接，无法旋转")
            return

        print(f"\n[Gimbal] 旋转到目标: Yaw={self.target_yaw:.2f}°, Pitch={self.target_pitch:.2f}°")
        try:
            self.gimbal.follow(self.target_yaw, self.target_pitch, 0.0)
            print(f"  ✓ 旋转完成")
        except Exception as e:
            print(f"  ✗ 旋转失败: {e}")

    def manual_adjust(self, dyaw=0.0, dpitch=0.0):
        """手动微调云台角度。

        参数:
            dyaw:   yaw 增量 (度), 正=右转
            dpitch: pitch 增量 (度), 正=上仰
        """
        if self.gimbal is None:
            print("[Gimbal] 云台未连接")
            return

        try:
            gy, gp, _ = self.gimbal.get_current_angles()
            self.current_gimbal_yaw = gy
            self.current_gimbal_pitch = gp

            ny = gy + dyaw
            np_val = gp + dpitch

            self.gimbal.follow(ny, np_val, 0.0)
            print(f"[Gimbal] 调整: Yaw={ny:.1f}° (Δ{dyaw:+.0f}), Pitch={np_val:.1f}° (Δ{dpitch:+.0f})")
        except Exception as e:
            print(f"[Gimbal] 调整失败: {e}")

    def adjust_rotation_step(self, delta: float):
        """调整手动微调步长。

        参数:
            delta: 步长增量, 正=增大, 负=减小
        """
        self.rotation_step = max(self._rotation_step_min,
                                  min(self._rotation_step_max,
                                      self.rotation_step + delta))
        print(f"[Gimbal] 步长调整为: {self.rotation_step:.0f}°")

    def zoom(self, delta: float):
        """变倍控制。

        参数:
            delta: 变倍增量 (正=放大, 负=缩小)
        """
        if self.gimbal is None:
            print("[Gimbal] 云台未连接，无法变倍")
            return

        try:
            self.current_zoom_mag += delta
            self.current_zoom_mag = max(self._zoom_min, min(self.current_zoom_mag, self._zoom_max))
            self.gimbal.zoom_to_magnification(self.current_zoom_mag)
            print(f"[Gimbal] 变倍: {self.current_zoom_mag:.1f}x (Δ{delta:+.0f})")
            time.sleep(self.stable_delay)
            self.gimbal.auto_focus()
            print(f"  ✓ 自动对焦完成")
        except Exception as e:
            print(f"[Gimbal] 变倍失败: {e}")

    def auto_focus(self):
        """触发自动对焦。"""
        if self.gimbal is None:
            print("[Gimbal] 云台未连接，无法对焦")
            return

        try:
            self.gimbal.auto_focus()
            print("[Gimbal] ✓ 自动对焦完成")
        except Exception as e:
            print(f"[Gimbal] 自动对焦失败: {e}")

    def update_current_angles(self):
        """读取并更新当前云台角度。"""
        if self.gimbal is None:
            return
        try:
            gy, gp, _ = self.gimbal.get_current_angles()
            self.current_gimbal_yaw = gy
            self.current_gimbal_pitch = gp
        except Exception:
            pass

    def query_fov(self, fallback_fov: float = 60.0) -> float:
        """从云台查询实时 FOV。

        参数:
            fallback_fov: 查询失败时的回退值 (度)

        返回:
            fov_x (度)
        """
        if self.gimbal is None:
            return fallback_fov
        try:
            fov_x, _, _, _ = self.gimbal.query_focal_length(timeout=1.0)
            if fov_x is not None:
                return fov_x
        except Exception:
            pass
        return fallback_fov
