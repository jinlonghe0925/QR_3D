# -*- coding: utf-8 -*-
"""qrnav_config.py — V2 QR 导航: 全局配置 + 共享状态 (零业务逻辑)

被 geometry / stream / dogctl / phases / steps / web / 主入口 共同引用。
依赖方向单向: 本模块不 import 任何兄弟模块, 不存在循环依赖。
"""
import threading

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


_qrnav_pose_lock    = threading.Lock()
_qrnav_pose_box     = [None]
_qrnav_pose_ver     = [0]
_qrnav_pose_msg_box = ["等待 QR 检测…"]
_qrnav_img_w_box    = [0]
_qrnav_img_h_box    = [0]   # V2 新增: 给中心换算用
_qrnav_poll_busy    = threading.Lock()

_qrnav_jpg_buf = _JpegBuffer()

# ============================================================
#  配置 (V1 全部 + V2 新增 FOV / 中心方向)
# ============================================================
_QR_NAV_CFG = {
    "ALIGN_LAT":      0.10,
    "STEP_LAT":       0.15,
    "DOG_MAX_STEP_M": 0.30,
    "DOG_LATERAL_SPEED_MPS": 0.30,  # 行走 gait 满摇杆实测侧移速度；用于距离→时长
    "DOG_MOTION_SETTLE_S": 1.0,     # 开始运动+切 gait 后等待运控稳定，再发送横移轴指令
    "EXEC_MIN_LAT":   0.005,
    # 现场实测方向：相机位置 X>0 → 右移，X<0 → 左移。
    "LATERAL_SIGN":   +1,
    "YAW_SIGN":       +1,
    "ZOOM_TARGET":    0.15,
    "STREAM_SCALE":   0.50,
    "STREAM_QUALITY": 60,
    "STREAM_FPS":     8.0,
    "POSE_POLL_HZ":   4.0,
    "LIVE_POSE_FPS":  0.0,
    "LIVE_POSE_EMA":  0.55,
    "QR_SIZE_M":      0.12,
    # ── V2 新增: 图像中心换算 ──
    "FOV_X":          60.0,    # 相机水平 FOV (度, 1x 时), 走 main() args.fov_x 注入
    # 中心换算符号 (实测反了时一键翻):
    #   dx>0 = QR 偏右; 想让 QR 回到中心, 云台应该让图像往左移 (target 减小 dx 对应角度)
    #   默认: yaw 增量 = +dx_px * (fov_x / img_w)   ← "右移相机" 让 QR 居中
    #         pitch 增量 = -dy_px * (fov_y / img_h)  ← "上抬相机" 让 QR 居中
    # 若云台协议左右/上下是反的, 把下面两个 -1 翻成 +1
    "CENTER_YAW_SIGN":   +1,
    "CENTER_PITCH_SIGN": +1,
    # 中心换算阈值: 像素偏移小于此值就视为已居中, 不发云台命令
    "CENTER_DEAD_PX":    4.0,
}
_qrnav_dog_port_holder = [5003]
_qrnav_estimator_holder = [None]
_qrnav_last_init_ts = [0.0]
_QR_NAV_INIT_COOLDOWN = 2.0        # 6s 太长 → 「再按一次」被拒, 没法停止
# ============================================================
#  Bug 2 修复: 真实 FOV 缓存 (按 zoom 维度)
#  旧逻辑: FOV_X/zoom —— 这是经验公式, 实际不同镜头在不同变倍下
#         焦距-变倍不是严格反比 (尤其带畸变/电动变焦镜头)
#  新逻辑: 调云台 query_focal_length() 取真实 FOV, 按 zoom 缓存 60s
#         同步刷新 est.camera_matrix 让后续 PnP 用对的内参
# ============================================================
_qrnav_focal_cache       = {}      # zoom(float) -> (fov_x, fov_y, ts)
_qrnav_focal_cache_ttl   = 60.0    # 秒
_qrnav_last_known_zoom   = [None]  # 跟踪 zoom 变化, 变了就刷新 camera_matrix
_qrnav_fov_sensor_w_mm   = 6.17    # IMX415 1/2.8" 感光元件宽, 仅作兜底
_qrnav_fov_sensor_h_mm   = 4.55    # 感光元件高
