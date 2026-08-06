#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D 可视化模块
绘制 QR 平面 + 相机视锥体 (旋转前/后对比)。

对应 app_config.json 的 visualizer_3d 节:
    fig_width, fig_height: 窗口尺寸, 默认 12x9
    viz_scale:             可视化缩放, 默认 1.0
    frustum_far_ratio:     视锥远裁剪面比例 (相对于距离), 默认 1.2
    before_color / before_lw: 旋转前视锥颜色/线宽
    after_color  / after_lw:  旋转后视锥颜色/线宽
    elev / azim:            3D 视角 (仰角/方位角)
"""

import math
import numpy as np


def draw_frustum(ax, pose, fov_x, img_w, img_h, qr_center,
                  color='r', label='Frustum', lw=2.5,
                  frustum_far_ratio: float = 1.2):
    """在给定 axes 上绘制单个视锥体 + 视线。返回 (distance, fov_y)。

    参数:
        ax:                matplotlib 3D axes
        pose:              位姿数据 dict，需含 rmat, camera_position
        fov_x:             水平 FOV (度)
        img_w:             图像宽度 (像素)
        img_h:             图像高度 (像素)
        qr_center:         QR 中心的世界坐标 (3,)
        color:             视锥颜色
        label:             图例标签
        lw:                线宽
        frustum_far_ratio: 远裁剪面距离 / 相机到QR距离, 默认 1.2
    """
    rmat = pose.get("rmat")
    cam_pos = pose.get("camera_position", np.zeros(3)).flatten()

    dist = np.linalg.norm(cam_pos)
    fov_y = fov_x * img_h / img_w
    far = dist * frustum_far_ratio

    # 相机坐标系各轴在世界坐标系中的方向 (rmat.T = R_c2w)
    cam_dir = rmat.T[:, 2]    # 光轴 (Z)
    cam_up = rmat.T[:, 1]     # 上方向 (Y)
    cam_right = rmat.T[:, 0]  # 右方向 (X)

    fc = cam_pos + cam_dir * far
    hh = far * math.tan(math.radians(fov_y) / 2)
    hw = far * math.tan(math.radians(fov_x) / 2)

    # 远裁剪面四个角点
    far_rect = np.array([
        fc + cam_right * (-hw) + cam_up * (-hh),
        fc + cam_right * (hw) + cam_up * (-hh),
        fc + cam_right * (hw) + cam_up * (hh),
        fc + cam_right * (-hw) + cam_up * (hh),
    ])

    # 视锥棱线 (相机 → 四个角点)
    for j in range(4):
        ax.plot([cam_pos[0], far_rect[j, 0]], [cam_pos[1], far_rect[j, 1]],
                [cam_pos[2], far_rect[j, 2]], color, linewidth=lw, alpha=0.8)

    # 远裁剪面边框
    far_loop = np.vstack([far_rect, far_rect[0]])
    ax.plot(far_loop[:, 0], far_loop[:, 1], far_loop[:, 2],
            color, linewidth=lw + 0.5, alpha=0.9, label=label)

    # 相机→QR 视线 (虚线)
    ax.plot([cam_pos[0], qr_center[0]], [cam_pos[1], qr_center[1]],
            [cam_pos[2], qr_center[2]], color, linestyle='--', linewidth=1.0, alpha=0.5)

    return dist, fov_y


class FrustumVisualizer:
    """3D 视锥可视化器。

    用法:
        viz = FrustumVisualizer(config=app_config)
        viz.show(pose_before, pose_after, fov_x, img_w, img_h)
    """

    def __init__(self, config: dict = None):
        """初始化可视化器。

        参数:
            config: 应用配置字典 (app_config.json 内容)
        """
        viz_cfg = config.get("visualizer_3d", {}) if config else {}

        self._fig3d = None
        self._ax3d = None

        # ── 可视化参数 ──
        self.fig_width = viz_cfg.get("fig_width", 12)
        self.fig_height = viz_cfg.get("fig_height", 9)
        self.viz_scale = viz_cfg.get("viz_scale", 1.0)
        self.frustum_far_ratio = viz_cfg.get("frustum_far_ratio", 1.2)

        # 旋转前视锥样式
        self.before_color = viz_cfg.get("before_color", "r")
        self.before_lw = viz_cfg.get("before_lw", 1.5)

        # 旋转后视锥样式
        self.after_color = viz_cfg.get("after_color", "b")
        self.after_lw = viz_cfg.get("after_lw", 3.0)

        # 3D 视角
        self.elev = viz_cfg.get("elev", 25)
        self.azim = viz_cfg.get("azim", -60)

    def show(self, pose_before, pose_after, fov_x, img_w, img_h):
        """3D 可视化 (完整版, 含 pause 循环, 会阻塞直到窗口关闭)。

        参数:
            pose_before: 旋转前的位姿数据 (原始 PnP 结果)
            pose_after:  旋转后的位姿数据 (含修正后的 rmat, 中心对齐后)
            fov_x:       水平 FOV (度)
            img_w:       图像宽度 (像素)
            img_h:       图像高度 (像素)
        """
        import matplotlib.pyplot as plt
        self._render(pose_before, pose_after, fov_x, img_w, img_h)
        # 阻塞等待用户关闭窗口
        while self._fig3d is not None and plt.fignum_exists(self._fig3d.number):
            plt.pause(0.1)
        # 注意: 调用者需在主循环中自行调用 plt.pause() 保持窗口响应

    def show_nonblocking(self, pose_before, pose_after, fov_x, img_w, img_h):
        """非阻塞版 show: 只创建窗口并触发首次渲染，不进入 pause 循环。

        调用者需要在主循环中自行调用 plt.pause() 来保持窗口响应。
        此方法仅做渲染，不阻塞。
        """
        self._render(pose_before, pose_after, fov_x, img_w, img_h)

    def _render(self, pose_before, pose_after, fov_x, img_w, img_h):
        """内部渲染逻辑 (不包含 plt.show 后的 pause 循环)。"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("  ✗ matplotlib 不可用，跳过 3D 可视化")
            return

        rmat = pose_before.get("rmat")
        qr_size = pose_before.get("qr_size", 0.1)
        object_points = pose_before.get("object_points")
        roll = pose_before.get("roll", 0.0)
        pitch = pose_before.get("pitch", 0.0)

        if rmat is None:
            print("  ✗ 无旋转矩阵，跳过 3D 可视化")
            return

        # ── 创建/重置图形 ──
        if self._fig3d is not None and plt.fignum_exists(self._fig3d.number):
            plt.close(self._fig3d)
        self._fig3d = plt.figure("QR-Camera 3D Frustum", figsize=(self.fig_width, self.fig_height))
        ax = self._fig3d.add_subplot(111, projection='3d')
        self._ax3d = ax

        # ── QR 平面 ──
        if object_points is not None:
            qr_obj = np.array(object_points) * self.viz_scale
        else:
            half = qr_size * self.viz_scale / 2.0
            qr_obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        qr_center = np.mean(qr_obj, axis=0)

        qr_loop = np.vstack([qr_obj, qr_obj[0]])
        ax.plot(qr_loop[:, 0], qr_loop[:, 1], qr_loop[:, 2], 'k-', linewidth=2.5, label='QR Code')
        ax.add_collection3d(Poly3DCollection([qr_obj], alpha=0.15, facecolor='gray', edgecolor='none'))
        ax.text(qr_center[0], qr_center[1], qr_center[2], 'QR Code', fontsize=10, ha='center', color='black', weight='bold')

        # ── 世界坐标轴 ──
        world_len = qr_size * self.viz_scale * 1.5
        ax.quiver(0, 0, 0, world_len, 0, 0, color='r', arrow_length_ratio=0.15, linewidth=1.5, label='World X (→)')
        ax.quiver(0, 0, 0, 0, world_len, 0, color='g', arrow_length_ratio=0.15, linewidth=1.5, label='World Y (↑)')
        ax.quiver(0, 0, 0, 0, 0, world_len, color='b', arrow_length_ratio=0.15, linewidth=1.5, label='World Z (⊙)')

        # ── 相机位置 ──
        cam_pos = pose_before.get("camera_position", np.zeros(3)).flatten()
        ax.scatter(*cam_pos, c='red', s=120, marker='^', edgecolors='black', linewidth=1, label='Camera', zorder=10)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.02, 'Camera', fontsize=9, color='red', ha='center')

        # ── 红色视锥: 旋转前 ──
        dist1, fov_y = draw_frustum(ax, pose_before, fov_x, img_w, img_h, qr_center,
                                     color=self.before_color, label='Before (red)',
                                     lw=self.before_lw, frustum_far_ratio=self.frustum_far_ratio)
        # ── 蓝色视锥: 旋转后 ──
        draw_frustum(ax, pose_after, fov_x, img_w, img_h, qr_center,
                     color=self.after_color, label='After (blue)',
                     lw=self.after_lw, frustum_far_ratio=self.frustum_far_ratio)

        # ── 视角信息 ──
        tvec = pose_before.get("tvec", np.zeros(3))
        tvn = np.linalg.norm(tvec)
        view_angle = math.degrees(math.acos(min(abs(tvec[2]) / tvn, 1.0))) if tvn > 1e-6 else 0.0

        max_range = max(abs(cam_pos).max(), qr_size) * 1.5
        ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range]); ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        yt = pitch; pt = roll - 180.0
        if abs(pt) >= 180: pt = (360.0 + pt) % 360.0

        ax.set_title(f'Red=Before Blue=After | FOV={fov_x:.1f}° | ViewAngle={view_angle:.1f}° | Dist={dist1:.2f}m\n'
                     f'Algorithm→Gimbal: Yaw={yt:.1f}° Pitch={pt:.1f}°')
        ax.legend(loc='upper left', fontsize=7)

        info_text = (f"Camera: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m\n"
                     f"Distance: {dist1:.3f} m\nFOV: {fov_x:.1f}° x {fov_y:.1f}°\nView Angle: {view_angle:.1f}°\n"
                     f"Roll: {roll:.1f}°  Pitch: {pitch:.1f}°\nImage: {img_w}x{img_h}")
        self._fig3d.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.view_init(elev=self.elev, azim=self.azim)
        self._fig3d.tight_layout()
        print("  ✓ 3D 视锥窗口已打开 (红色=旋转前, 蓝色加粗=旋转后) — 关闭窗口继续")
        plt.show(block=False)
        plt.pause(0.1)  # 触发一次渲染

    @property
    def fig(self):
        """返回当前 matplotlib figure 对象。"""
        return self._fig3d
