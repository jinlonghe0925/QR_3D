#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D 可视化模块
绘制 QR 平面 + 相机视锥体 (旋转前/后对比)。

PC 端 (有GUI): 使用 matplotlib 弹出窗口
板端 (无GUI): 使用 Plotly 生成 HTML，通过 HTTP 提供
"""

import math
import numpy as np


# ═══════════════════════════════════════════════════════════
# 纯数据计算 (共享)
# ═══════════════════════════════════════════════════════════
def _build_frustum_lines(pose, fov_x, img_w, img_h, qr_center, frustum_far_ratio=1.2):
    """计算视锥体的线段数据。"""
    rmat = pose.get("rmat")
    cam_pos = pose.get("camera_position", np.zeros(3)).flatten()

    dist = np.linalg.norm(cam_pos)
    fov_y = fov_x * img_h / img_w
    far = dist * frustum_far_ratio

    cam_dir = rmat.T[:, 2]
    cam_up = rmat.T[:, 1]
    cam_right = rmat.T[:, 0]
    fc = cam_pos + cam_dir * far
    hh = far * math.tan(math.radians(fov_y) / 2)
    hw = far * math.tan(math.radians(fov_x) / 2)

    far_rect = np.array([
        fc + cam_right * (-hw) + cam_up * (-hh),
        fc + cam_right * (hw) + cam_up * (-hh),
        fc + cam_right * (hw) + cam_up * (hh),
        fc + cam_right * (-hw) + cam_up * (hh),
    ])

    # 视锥棱线: [(cam_pos, corner), ...]
    lines = []
    for j in range(4):
        lines.append((cam_pos.copy(), far_rect[j].copy()))

    # 远裁剪面边框
    for j in range(4):
        lines.append((far_rect[j].copy(), far_rect[(j + 1) % 4].copy()))

    # 视线
    lines.append((cam_pos.copy(), qr_center.copy()))

    return lines, dist, fov_y


def _build_3d_data(pose_before, pose_after, fov_x, img_w, img_h, config=None):
    """构建 3D 可视化所需的所有数据。"""
    viz_cfg = config.get("visualizer_3d", {}) if config else {}
    viz_scale = viz_cfg.get("viz_scale", 1.0)
    frustum_far_ratio = viz_cfg.get("frustum_far_ratio", 1.2)

    rmat = pose_before.get("rmat")
    qr_size = pose_before.get("qr_size", 0.1)
    object_points = pose_before.get("object_points")
    roll = pose_before.get("roll", 0.0)
    pitch = pose_before.get("pitch", 0.0)

    if rmat is None:
        return None

    # QR 平面角点
    if object_points is not None:
        qr_obj = np.array(object_points) * viz_scale
    else:
        half = qr_size * viz_scale / 2.0
        qr_obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    qr_center = np.mean(qr_obj, axis=0)

    cam_pos = pose_before.get("camera_position", np.zeros(3)).flatten()
    tvec = pose_before.get("tvec", np.zeros(3))
    tvn = np.linalg.norm(tvec)
    view_angle = math.degrees(math.acos(min(abs(tvec[2]) / tvn, 1.0))) if tvn > 1e-6 else 0.0

    # 红色视锥 (旋转前)
    red_lines, dist1, fov_y = _build_frustum_lines(
        pose_before, fov_x, img_w, img_h, qr_center, frustum_far_ratio)
    # 蓝色视锥 (旋转后)
    blue_lines, _, _ = _build_frustum_lines(
        pose_after, fov_x, img_w, img_h, qr_center, frustum_far_ratio)

    return {
        "qr_obj": qr_obj, "qr_center": qr_center, "qr_size": qr_size,
        "cam_pos": cam_pos, "roll": roll, "pitch": pitch,
        "red_lines": red_lines, "blue_lines": blue_lines,
        "dist1": dist1, "fov_y": fov_y, "fov_x": fov_x,
        "view_angle": view_angle, "img_w": img_w, "img_h": img_h,
    }


# ═══════════════════════════════════════════════════════════
# Plotly HTML 渲染 (板端无 GUI)
# ═══════════════════════════════════════════════════════════
# matplotlib 简写 → Plotly 标准颜色名
_COLOR_MAP = {'r': 'red', 'g': 'green', 'b': 'blue', 'k': 'black',
              'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'w': 'white'}


def _add_lines_to_plotly(fig, lines, color, width, name, dash=None):
    """将线段添加到 Plotly figure。"""
    import plotly.graph_objects as go
    color = _COLOR_MAP.get(color, color)  # 转换简写颜色
    for (p1, p2) in lines:
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=width, dash=dash),
            name=name, showlegend=False,
        ))
    # 只显示一次图例
    if lines:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color, width=width),
            name=name,
        ))


def render_plotly_html(data: dict, config: dict = None) -> str:
    """生成 Plotly 3D HTML 字符串。

    参数:
        data:   _build_3d_data 返回的字典
        config: 应用配置字典

    返回:
        完整的 HTML 字符串
    """
    import plotly.graph_objects as go

    if data is None:
        return "<html><body><h3>3D 可视化数据为空</h3></body></html>"

    viz_cfg = config.get("visualizer_3d", {}) if config else {}

    fig = go.Figure()

    # ── QR 平面 ──
    qr = data["qr_obj"]
    qr_loop = np.vstack([qr, qr[0]])
    fig.add_trace(go.Scatter3d(
        x=qr_loop[:, 0], y=qr_loop[:, 1], z=qr_loop[:, 2],
        mode='lines', line=dict(color='black', width=3),
        name='QR Code',
    ))
    # QR 半透明面
    fig.add_trace(go.Mesh3d(
        x=qr[:, 0], y=qr[:, 1], z=qr[:, 2],
        color='gray', opacity=0.2, name='QR Plane', showlegend=False,
    ))

    # ── 世界坐标轴 ──
    world_len = data["qr_size"] * 1.5
    for axis, color, name in [([world_len, 0, 0], 'red', 'X'), 
                                ([0, world_len, 0], 'green', 'Y'),
                                ([0, 0, world_len], 'blue', 'Z')]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]],
            mode='lines', line=dict(color=color, width=2),
            name=f'World {name}',
        ))

    # ── 相机位置 ──
    cp = data["cam_pos"]
    fig.add_trace(go.Scatter3d(
        x=[cp[0]], y=[cp[1]], z=[cp[2]],
        mode='markers+text', marker=dict(color='red', size=8, symbol='diamond'),
        text=['Camera'], textposition='top center',
        name='Camera',
    ))

    # ── 红色视锥 (旋转前) ──
    before_color = viz_cfg.get("before_color", "red")
    before_lw = viz_cfg.get("before_lw", 2)
    _add_lines_to_plotly(fig, data["red_lines"], before_color, before_lw, "Before (red)")

    # ── 蓝色视锥 (旋转后) ──
    after_color = viz_cfg.get("after_color", "blue")
    after_lw = viz_cfg.get("after_lw", 4)
    _add_lines_to_plotly(fig, data["blue_lines"], after_color, after_lw, "After (blue)")

    # ── 布局 ──
    max_range = max(abs(cp).max(), data["qr_size"]) * 1.5
    yt = data["pitch"]
    pt = data["roll"] - 180.0
    if abs(pt) >= 180:
        pt = (360.0 + pt) % 360.0

    fig.update_layout(
        title=f'Red=Before Blue=After | FOV={data["fov_x"]:.1f}° | Dist={data["dist1"]:.2f}m<br>'
              f'Yaw={yt:.1f}° Pitch={pt:.1f}° | ViewAngle={data["view_angle"]:.1f}°',
        scene=dict(
            xaxis=dict(title='X (m)', range=[-max_range, max_range]),
            yaxis=dict(title='Y (m)', range=[-max_range, max_range]),
            zaxis=dict(title='Z (m)', range=[-max_range, max_range]),
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.01, y=0.99),
        template='plotly_dark',
    )

    # 'inline' = plotly.js 内嵌进 HTML (~4MB), 完全离线可用 (板子无 CDN 网络)
    return fig.to_html(include_plotlyjs='inline', full_html=True)


# ═══════════════════════════════════════════════════════════
# FrustumVisualizer
# ═══════════════════════════════════════════════════════════
class FrustumVisualizer:
    """3D 视锥可视化器。

    PC 端: show() / show_nonblocking() 使用 matplotlib
    板端:  render_html() 生成 Plotly HTML
    """

    def __init__(self, config: dict = None):
        viz_cfg = config.get("visualizer_3d", {}) if config else {}
        self._fig3d = None
        self._ax3d = None
        self._html = ""  # Plotly HTML 缓存

        self.fig_width = viz_cfg.get("fig_width", 12)
        self.fig_height = viz_cfg.get("fig_height", 9)
        self.viz_scale = viz_cfg.get("viz_scale", 1.0)
        self.frustum_far_ratio = viz_cfg.get("frustum_far_ratio", 1.2)
        self.before_color = viz_cfg.get("before_color", "r")
        self.before_lw = viz_cfg.get("before_lw", 1.5)
        self.after_color = viz_cfg.get("after_color", "b")
        self.after_lw = viz_cfg.get("after_lw", 3.0)
        self.elev = viz_cfg.get("elev", 25)
        self.azim = viz_cfg.get("azim", -60)
        self._config = config

    # ── Plotly 渲染 (板端) ──
    def render_html(self, pose_before, pose_after, fov_x, img_w, img_h) -> str:
        """生成 Plotly 3D HTML (板端使用)。

        返回: 完整的 HTML 字符串，可通过 HTTP 提供。
        """
        data = _build_3d_data(pose_before, pose_after, fov_x, img_w, img_h, self._config)
        if data is None:
            self._html = "<html><body><h3>无旋转矩阵</h3></body></html>"
        else:
            self._html = render_plotly_html(data, self._config)
        print("  ✓ 3D 可视化 HTML 已生成")
        return self._html

    @property
    def html(self) -> str:
        """最近一次生成的 Plotly HTML。"""
        return self._html

    # ── Matplotlib 渲染 (PC 端) ──
    def show(self, pose_before, pose_after, fov_x, img_w, img_h):
        """Matplotlib 3D 窗口 (PC 端, 阻塞)。"""
        import matplotlib.pyplot as plt
        self._render_mpl(pose_before, pose_after, fov_x, img_w, img_h)
        while self._fig3d is not None and plt.fignum_exists(self._fig3d.number):
            plt.pause(0.1)

    def show_nonblocking(self, pose_before, pose_after, fov_x, img_w, img_h):
        """Matplotlib 非阻塞版。"""
        self._render_mpl(pose_before, pose_after, fov_x, img_w, img_h)

    def _render_mpl(self, pose_before, pose_after, fov_x, img_w, img_h):
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("  ✗ matplotlib 不可用")
            return

        rmat = pose_before.get("rmat")
        qr_size = pose_before.get("qr_size", 0.1)
        object_points = pose_before.get("object_points")
        roll = pose_before.get("roll", 0.0)
        pitch = pose_before.get("pitch", 0.0)

        if rmat is None:
            return

        if self._fig3d is not None and plt.fignum_exists(self._fig3d.number):
            plt.close(self._fig3d)
        self._fig3d = plt.figure("QR-Camera 3D", figsize=(self.fig_width, self.fig_height))
        ax = self._fig3d.add_subplot(111, projection='3d')

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

        world_len = qr_size * self.viz_scale * 1.5
        ax.quiver(0, 0, 0, world_len, 0, 0, color='r', arrow_length_ratio=0.15, linewidth=1.5, label='World X')
        ax.quiver(0, 0, 0, 0, world_len, 0, color='g', arrow_length_ratio=0.15, linewidth=1.5, label='World Y')
        ax.quiver(0, 0, 0, 0, 0, world_len, color='b', arrow_length_ratio=0.15, linewidth=1.5, label='World Z')

        cam_pos = pose_before.get("camera_position", np.zeros(3)).flatten()
        ax.scatter(*cam_pos, c='red', s=120, marker='^', edgecolors='black', linewidth=1, label='Camera', zorder=10)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.02, 'Camera', fontsize=9, color='red', ha='center')

        # 复用的底层函数
        from visualizer_3d import draw_frustum as _df
        dist1, fov_y = _df(ax, pose_before, fov_x, img_w, img_h, qr_center,
                            color=self.before_color, label='Before (red)', lw=self.before_lw,
                            frustum_far_ratio=self.frustum_far_ratio)
        _df(ax, pose_after, fov_x, img_w, img_h, qr_center,
            color=self.after_color, label='After (blue)', lw=self.after_lw,
            frustum_far_ratio=self.frustum_far_ratio)

        tvec = pose_before.get("tvec", np.zeros(3))
        tvn = np.linalg.norm(tvec)
        view_angle = math.degrees(math.acos(min(abs(tvec[2]) / tvn, 1.0))) if tvn > 1e-6 else 0.0

        max_range = max(abs(cam_pos).max(), qr_size) * 1.5
        ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range]); ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        yt = pitch; pt = roll - 180.0
        if abs(pt) >= 180: pt = (360.0 + pt) % 360.0
        ax.set_title(f'Red=Before Blue=After | FOV={fov_x:.1f}° | ViewAngle={view_angle:.1f}° | Dist={dist1:.2f}m\n'
                     f'Yaw={yt:.1f}° Pitch={pt:.1f}°')
        ax.legend(loc='upper left', fontsize=7)

        info_text = (f"Camera: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m\n"
                     f"Distance: {dist1:.3f} m\nFOV: {fov_x:.1f}° x {fov_y:.1f}°\n"
                     f"View Angle: {view_angle:.1f}°\nRoll: {roll:.1f}°  Pitch: {pitch:.1f}°\nImage: {img_w}x{img_h}")
        self._fig3d.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.view_init(elev=self.elev, azim=self.azim)
        self._fig3d.tight_layout()
        print("  ✓ 3D 视锥窗口已打开 (红色=旋转前, 蓝色加粗=旋转后)")
        plt.show(block=False)
        plt.pause(0.1)

    @property
    def fig(self):
        return self._fig3d
