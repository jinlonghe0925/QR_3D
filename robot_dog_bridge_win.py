#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR 位姿 → 机器狗导航 (Windows 版本)
============================================
严格遵循 gimbal_pose_estimation_ascend.py 的位姿估计管线，
仅替换无法在 Windows 上运行的组件：
- YOLO QR 检测: ONNX Runtime (CPU) ← 替代 Ascend OM
- 位姿估计: numpy + cv2.solvePnP (CPU) ← 完全一致
- 狗运动: 始终模拟模式 ← 替代真狗 UDP
- 相机内参: camera_intrinsics.txt / fov_x ← 替代 gimbal.query_focal_length()

云台: Windows 通过 UDP 控制云台 (socket + cv2 RTSP)，与 Linux 完全兼容。
无需: Ascend NPU / 真狗 / TDRobot / ROS2 / Linux

坐标系转换 (二维码竖直安装):
  QR 坐标系: X=右, Y=上, Z=向板内(相机方向)
  Dog 坐标系: X=前, Y=右, Z=上
  转换:  Dog_X =  QR_Z   (前进距离)
         Dog_Y = -QR_X   (右侧距离)
         Dog_Z = -QR_Y   (高度，忽略)

环境:
    pip install opencv-python numpy onnxruntime

用法:
    # === 本地图片模式 (无云台) ===
    # 单张图片检测
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx
    python robot_dog_bridge_win.py --mode single --model ../qrdet-s.onnx --image ./test.jpg

    # 循环模式
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --mode loop
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --mode loop --interval 1.0

    # 使用 camera_intrinsics.txt
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --camera-intrinsics ../camera_intrinsics.txt

    # 自定义参数
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --qr-size 0.15 --fov-x 70.0
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --conf 0.3 --nms 0.6
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --square-size 600 --stop-distance 0.5
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --detector pyzbar

    # 调试模式 + 弹窗
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --debug --show

    # 完整组合
    python robot_dog_bridge_win.py --image ./test.jpg --model ../qrdet-s.onnx --camera-intrinsics ../camera_intrinsics.txt --debug --mode loop

    # === 云台模式 (Windows 通过 UDP 连接云台) ===
    # 云台 IP: 192.168.1.99, UDP 控制端口: 6001, RTSP: 554
    # 可见光: rtsp://192.168.1.99:554/ch1/stream0
    # 红外:   rtsp://192.168.1.99:554/ch2/stream0
    # 主视频: rtsp://192.168.1.99:554/ch0/stream0

    # 云台抓图 + 位姿估计 (无本地图片，从 RTSP 拉图)
    python robot_dog_bridge_win.py --model ../qrdet-s.onnx --gimbal-ip 192.168.1.99 --gimbal-port 6001
    python robot_dog_bridge_win.py --model ../qrdet-s.onnx --gimbal-ip 192.168.1.99 --fov-x 70.0

    # 云台循环模式 (每轮从 RTSP 抓图)
    python robot_dog_bridge_win.py --model ../qrdet-s.onnx --gimbal-ip 192.168.1.99 --mode loop --interval 2.0

    # 云台 + 调试 (输出所有中间图片)
    python robot_dog_bridge_win.py --model ../qrdet-s.onnx --gimbal-ip 192.168.1.99 --debug
"""

import sys
import time
import math
import os
import re
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

# ── 把当前目录加入 path，方便 import 同级模块 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qr_detection
import pose_estimation
import yolo_detection_ascend as yolo_detection

# ── 云台控制 (Windows / Linux 均可通过 UDP 控制) ──
try:
    from gimbal_demo import GimbalController, capture_from_rtsp
    _GIMBAL_AVAILABLE = True
except ImportError:
    _GIMBAL_AVAILABLE = False


# ============================================================
# 相机内参加载
# ============================================================
def load_camera_intrinsics(txt_path: str) -> dict:
    """从 camera_intrinsics.txt 读取相机内参和 QR 参数。"""
    info = {}
    matrix_lines = []
    in_matrix = False
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line and "[" not in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.split("#")[0].strip()
                try:
                    info[key] = float(val)
                except ValueError:
                    info[key] = val
            if line.startswith("camera_matrix"):
                in_matrix = True
                continue
            if line.startswith("]"):
                in_matrix = False
                continue
            if in_matrix:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) == 3:
                    matrix_lines.append([float(v) for v in nums])

    if len(matrix_lines) == 3:
        info["camera_matrix"] = np.array(matrix_lines, dtype=np.float64)
    info["dist_coeffs"] = np.zeros((5, 1), dtype=np.float32)

    if "fov_x" not in info and "fov_x_deg" in info:
        info["fov_x"] = info["fov_x_deg"]
    if "fov_y" not in info and "fov_y_deg" in info:
        info["fov_y"] = info["fov_y_deg"]

    return info


# ============================================================
# QR 坐标系 → 狗体相对坐标
# ============================================================
def qr_position_to_dog(camera_position) -> Tuple[float, float, float]:
    """QR Pose 中的 camera_position → 狗体相对坐标。

    QR 坐标系 (竖直安装): X=右, Y=上, Z=向板内(相机)
    Dog 坐标系:          X=前, Y=右, Z=上

    转换:
        Dog_X =  QR_Z    (前进距离)
        Dog_Y = -QR_X    (右侧距离)
        Dog_Z = -QR_Y    (高度差)
    """
    if camera_position is None or len(camera_position) != 3:
        raise ValueError("camera_position 必须为 [x, y, z]")
    x_qr, y_qr, z_qr = (float(v) for v in camera_position)
    return z_qr, -x_qr, -y_qr


def compute_dog_move(x_dog: float, y_dog: float, z_dog: float,
                     stop_distance: float = 0.8) -> Optional[Tuple[float, float, float]]:
    """将狗体相对坐标转换为 Dog.move(x, y, z) 参数。

    注意: 移动距离会减去 stop_distance，确保停在 QR 前方而非撞上去。
    """
    horizontal_dist = math.hypot(x_dog, y_dog)
    if horizontal_dist <= stop_distance:
        return None

    ratio = (horizontal_dist - stop_distance) / horizontal_dist
    dx = x_dog * ratio
    dy = -y_dog * ratio      # 狗体 Y(右+) → SDK y(左+)，符号取反
    dz = math.degrees(math.atan2(y_dog, x_dog))
    return dx, dy, dz


# ============================================================
# Windows 位姿估计器 (无云台，纯 CPU)
# ============================================================
class WinPoseEstimator:
    """Windows 位姿估计器 — 严格遵循 GimbalPoseEstimator 的管线。

    与 gimbal_pose_estimation_ascend.GimbalPoseEstimator 的区别:
    - YOLO 后端: ONNX Runtime 替代 Ascend OM
    - 图像源: 本地文件 / 云台 RTSP 抓图 (通过 UDP 控制)
    - 内参: camera_intrinsics.txt / fov_x / gimbal.query_focal_length()
    """

    def __init__(self, model_path: str = "../qrdet-s.onnx",
                 qr_size: float = 0.10,
                 config_file: str = None,
                 fov_x: float = 60.0,
                 conf_threshold: float = 0.2,
                 nms_threshold: float = 0.5,
                 square_size: int = 800,
                 detector_type: str = "opencv",
                 gimbal_ip: Optional[str] = None,
                 gimbal_port: int = 6001):
        # ── 从 config.json 读取参数 (与 GimbalPoseEstimator 一致) ──
        self.config_file = config_file
        if config_file and Path(config_file).exists():
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if qr_size is None:
                qr_size = config_data.get("estimate", {}).get("qr_size", 0.1)

        self.qr_size = qr_size
        self.fov_x = fov_x
        self.square_size = square_size
        self.detector_type = detector_type
        self._override_camera_matrix = None
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        # 云台相关输出目录 (与 Linux 版保持一致)
        self.output_dir = "gimbal_pose_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # ── 视频流相关属性 ──
        self.video_capture = None
        self.video_thread = None
        self.video_running = False
        self.video_paused = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.latest_result_image = None
        self.result_image_lock = threading.Lock()

        # ── 旋转角度相关属性 ──
        self.target_yaw = 0.0            # 旋转目标 yaw (由位姿计算得出)
        self.target_pitch = 0.0          # 旋转目标 pitch (由位姿计算得出)
        self.rotation_step = 5.0         # 手动微调步长 (度)
        self.current_gimbal_yaw = 0.0    # 当前云台 yaw
        self.current_gimbal_pitch = 0.0  # 当前云台 pitch
        self.auto_rotate = True          # 识别后是否自动旋转云台

        # ── 变倍 (Zoom) 相关属性 ──
        self.current_zoom_mag = 1.0      # 当前变倍倍率
        self.zoom_step = 1.0             # 变倍步长

        # ── 视频叠加标注 ──
        self.qr_center = None            # 二维码中心像素坐标 (cx, cy), 用于视频窗口叠加

        # ── 3D 可视化窗口 ──
        self._fig3d = None
        self._ax3d = None
        self._pending_3d_data = None  # 待主线程显示的 3D 数据

        # ── 云台控制器 (Windows 通过 UDP 连接，与 Linux 完全兼容) ──
        self.gimbal = None
        self.rtsp_tv = None
        self.gimbal_ip = gimbal_ip
        self.gimbal_port = gimbal_port
        if gimbal_ip is not None and _GIMBAL_AVAILABLE:
            self._connect_gimbal(gimbal_ip, gimbal_port)

        print("[Detector] 加载 ONNX 模型: {}".format(model_path))
        self.detector = yolo_detection.load_yolov8_detector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        if self.detector is None:
            raise RuntimeError("ONNX 检测器加载失败，请确认已安装 onnxruntime 且模型路径正确")

    def _connect_gimbal(self, gimbal_ip: str, gimbal_port: int):
        """连接云台控制器 (与 GimbalPoseEstimator 一致)。"""
        print(f"\n[Gimbal] 连接云台: {gimbal_ip}:{gimbal_port}")
        try:
            self.gimbal = GimbalController(ip=gimbal_ip, port=gimbal_port)
            self.rtsp_tv = f"rtsp://{gimbal_ip}:554/ch1/stream0"
            print(f"  ✓ 云台连接成功")
            print(f"  可见光 RTSP: {self.rtsp_tv}")
        except Exception as e:
            print(f"  ✗ 云台连接失败: {e}")
            print(f"  将回退到本地图片模式")
            self.gimbal = None
            self.rtsp_tv = None

    def cleanup(self):
        """清理资源 (与 GimbalPoseEstimator 保持一致)。"""
        self.stop_video_stream()
        if self.gimbal is not None:
            try:
                self.gimbal.stop()
                print("[Gimbal] 已停止")
            except Exception:
                pass
            self.gimbal = None
        print("[Estimator] 已清理")

    def capture_visible_image(self, output_path=None, timeout=5):
        """从云台 RTSP 流抓取可见光图像 (与 GimbalPoseEstimator 一致)。

        返回: (success, image_path)
        """
        if self.gimbal is None or self.rtsp_tv is None:
            print("[Capture] 云台未连接，无法抓图")
            return False, None

        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"capture_tv_{timestamp}.jpg")

        print(f"\n[Capture] RTSP 抓图: {self.rtsp_tv}")
        success = capture_from_rtsp(self.rtsp_tv, output_path, timeout=timeout)

        if success:
            img = cv2.imread(output_path)
            if img is not None:
                print(f"  ✓ 图像已保存: {output_path} ({img.shape[1]}x{img.shape[0]})")
                return True, output_path
            else:
                print(f"  ✗ 图像保存成功但无法读取")
                return False, None
        else:
            print(f"  ✗ RTSP 抓图失败")
            return False, None

    def estimate_pose_from_gimbal(self, debug=False, debug_dir=None):
        """从云台抓图 → 位姿估计 (与 GimbalPoseEstimator.run_single_estimation 一致)。

        返回: (success, pose_data, result_image)
        """
        success, image_path = self.capture_visible_image()
        if not success:
            return False, None, None

        return self.estimate_pose_from_image(image_path, debug=debug, debug_dir=debug_dir)

    def _fast_detect_points(self, image: np.ndarray, H_mat: np.ndarray,
                             warped_size: tuple, alpha_mask_big: np.ndarray,
                             quad_points: np.ndarray) -> Optional[np.ndarray]:
        """轻量角点检测：跳过 YOLO，复用第一帧的 H_mat 做透视变换 + 角点检测。

        返回: points_corrected (4x2) 或 None
        """
        # 透视变换 + Alpha 合成
        warped = qr_detection.warp_perspective_with_homography(image, H_mat, warped_size)
        warped_alpha = qr_detection.warp_perspective_with_homography(alpha_mask_big, H_mat, warped_size)
        warped_alpha_3ch = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR)
        warped_alpha_f = warped_alpha_3ch / 255.0
        warped_comp = (warped_alpha_f * warped / 255.0 * 255.0).astype(np.uint8)
        border = ((1.0 - warped_alpha_f) * 255.0).astype(np.uint8)
        warped_comp = cv2.add(warped_comp, border)

        # 角点检测
        success, points_warped, _ = qr_detection.detect_qr_in_warped_image(
            warped_comp, self.detector_type
        )
        if success:
            H_inv = np.linalg.inv(H_mat)
            pts = qr_detection.transform_points_back(points_warped, H_inv)
            pts = pts[:, :2]
            pts = qr_detection.order_points_clockwise2(pts)
            return pts
        else:
            # fallback: 用 YOLO 的 quad_points
            return quad_points.copy()

    # ── 完全复制自 GimbalPoseEstimator.estimate_pose_from_image ──
    def estimate_pose_from_image(self, image_path: str, debug: bool = False,
                                 debug_dir: Optional[str] = None) -> Tuple[bool, Optional[dict], Optional[np.ndarray]]:
        """从图片做完整的 QR 检测 + 位姿估计。

        严格遵循 gimbal_pose_estimation_ascend.GimbalPoseEstimator.estimate_pose_from_image
        的每一步。与 Linux 版的区别仅在于相机内参来源 (fov_x / camera_intrinsics.txt vs gimbal查询)。
        """
        import os
        import time as _time

        print("\n" + "-" * 60)
        print("步骤2: 位姿估计（实际相机）")
        print("-" * 60)

        # 初始化调试目录
        if debug:
            if debug_dir is None:
                timestamp = int(_time.time())
                debug_dir = f"debug_output_{timestamp}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"\n  调试模式已启用，中间图片将保存到: {debug_dir}")

        # ================================================================
        # 2.0 读取图像
        # ================================================================
        print(f"\n  读取图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ 无法读取图像")
            return False, None, None

        H, W = image.shape[:2]
        print(f"  图像尺寸: {W}x{H}")

        # 保存原始图像 (01_original_image)
        if debug:
            filepath = os.path.join(debug_dir, "01_original_image.jpg")
            cv2.imwrite(filepath, image)
            print(f"  调试: 图像已保存到 {filepath}")

        # ================================================================
        # 2.1 YOLOv8 检测二维码
        # ================================================================
        print(f"\n  2.1 YOLOv8检测二维码...")
        detections = yolo_detection.detect_qr_yolov8(image, self.detector)
        print(f"  检测到 {len(detections)} 个二维码")

        if len(detections) == 0:
            print(f"  ✗ 未检测到二维码")
            return False, None, None

        detection = detections[0]
        polygon_xy = detection["polygon_xy"].astype(np.float32)
        confidence = detection["confidence"]
        print(f"  使用检测结果 1 (置信度: {confidence:.2f})")

        # 保存YOLO初始检测结果图 (02_yolo_initial_detect)
        image_init_detect = qr_detection.draw_qr_boundary(image.copy(),
                                                           qr_detection.order_points_clockwise2(
                                                               qr_detection.approximate_quadrilateral(polygon_xy)
                                                           ))
        if debug:
            filepath = os.path.join(debug_dir, "02_yolo_initial_detect.jpg")
            cv2.imwrite(filepath, image_init_detect)
            print(f"  调试: 图像已保存到 {filepath}")

        # ================================================================
        # 2.2 近似四边形
        # ================================================================
        print(f"\n  2.2 近似四边形...")
        try:
            quad_points = qr_detection.approximate_quadrilateral(polygon_xy)
            quad_points = qr_detection.order_points_clockwise2(quad_points)
            print(f"  ✓ 四边形角点: {quad_points.shape}")
        except ValueError as e:
            print(f"  ✗ 四边形近似失败: {e}")
            return False, None, None

        # ================================================================
        # 2.2b Alpha 掩码生成 (YOLO框 → 二值 → 膨胀)
        # ================================================================
        # 计算YOLO框的边界
        x_coords = quad_points[:, 0]
        y_coords = quad_points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        max_x_diff = x_max - x_min
        max_y_diff = y_max - y_min
        max_side = max(max_x_diff, max_y_diff)

        print(f"  YOLO框中心: ({center_x}, {center_y})")
        print(f"  YOLO框最大边长: {max_side} pixels")

        # 创建二值图像（与 image 同分辨率）
        alpha_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = quad_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(alpha_mask, [pts], 255)

        # 保存二值图像 (02_yolo_initial_detect_alpha)
        if debug:
            alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_alpha.jpg")
        else:
            alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_alpha.jpg")
        cv2.imwrite(alpha_filepath, alpha_mask)
        print(f"  调试: 二值图像已保存到 {alpha_filepath}")

        # 对二值图像进行膨胀操作 (55x55 核)
        kernel = np.ones((55, 55), np.uint8)
        alpha_mask_big = cv2.dilate(alpha_mask, kernel, iterations=1)

        # 保存膨胀后的二值图像
        if debug:
            big_alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_big_alpha.jpg")
        else:
            big_alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_big_alpha.jpg")
        cv2.imwrite(big_alpha_filepath, alpha_mask_big)
        print(f"  调试: 膨胀后二值图像已保存到 {big_alpha_filepath}")

        # ================================================================
        # 2.3 透视变换矫正 + Alpha 合成
        # ================================================================
        print(f"\n  2.3 透视变换矫正...")

        margin_qr = 1500
        shift_qr = margin_qr / 2.0

        dst_square = np.array([
            [shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, self.square_size - 1 + shift_qr],
            [shift_qr, self.square_size - 1 + shift_qr]
        ], dtype=np.float32)

        H_mat = qr_detection.compute_homography(quad_points, dst_square)
        if H_mat is None:
            print(f"  ✗ 单应矩阵计算失败")
            return False, None, None

        warped_size = (self.square_size + margin_qr, self.square_size + margin_qr)

        # 原图 warped
        warped = qr_detection.warp_perspective_with_homography(image, H_mat, warped_size)

        # Alpha 掩码 warped
        warped_alpha = qr_detection.warp_perspective_with_homography(alpha_mask_big, H_mat, warped_size)

        # 保存 warped_alpha (03_warped_alpha_image)
        if debug:
            filepath = os.path.join(debug_dir, "03_warped_alpha_image.jpg")
            cv2.imwrite(filepath, warped_alpha)
            print(f"  调试: 图像已保存到 {filepath}")

        print(f"  ✓ 透视变换完成")
        print(f"  透视变换尺寸: {warped_size[0]}x{warped_size[1]}")

        # 保存 warped (03_warped_image)
        if debug:
            filepath = os.path.join(debug_dir, "03_warped_image.jpg")
            cv2.imwrite(filepath, warped)
            print(f"  调试: 图像已保存到 {filepath}")

        # ── Alpha 合成 ──
        # warped_alpha: (H, W) → (H, W, 3), 归一化到 [0, 1]
        warped_alpha_3ch = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR)
        warped_alpha_f = warped_alpha_3ch / 255.0
        warped_f = warped / 255.0

        # 合成: alpha 区域保留原图，其余为黑色
        warped_comp = warped_alpha_f * warped_f
        warped_comp = warped_comp * 255.0
        warped_comp = warped_comp.astype(np.uint8)

        # 补白: 非 alpha 区域填白色
        border_image = (1.0 - warped_alpha_f) * 255.0
        border_image = border_image.astype(np.uint8)
        warped_comp = cv2.add(warped_comp, border_image)

        # 保存合成图 (03_warped_image_comp)
        if debug:
            filepath = os.path.join(debug_dir, "03_warped_image_comp.jpg")
            cv2.imwrite(filepath, warped_comp)
            print(f"  调试: 图像已保存到 {filepath}")

        # ================================================================
        # 2.4 在矫正图像上检测二维码角点 (多次检测 + 中值滤波抗噪)
        # ================================================================
        N_DETECT = 5
        print(f"\n  2.4 检测二维码角点 (重复{N_DETECT}次, 中值滤波)...")
        all_warped = []  # 收集多次检测的 points_warped (4x2)
        decoded_info = ""
        for _k in range(N_DETECT):
            ok, pts_w, info = qr_detection.detect_qr_in_warped_image(
                warped_comp, self.detector_type
            )
            if ok and pts_w is not None:
                all_warped.append(pts_w)
                if not decoded_info and info:
                    decoded_info = info
        success = len(all_warped) >= 2

        if success:
            # 对每个角点分别取中值
            stacked_w = np.array(all_warped)  # (M, 4, 2)
            points_warped = np.median(stacked_w, axis=0).astype(np.float32)
            mad_w = np.median(np.abs(stacked_w - points_warped), axis=0) if len(all_warped) > 1 else np.zeros_like(points_warped)
            print(f"  ✓ 检测成功 ({len(all_warped)}/{N_DETECT}次, 中值滤波)")
            print(f"    解码: {decoded_info}")
            for j in range(4):
                print(f"    角点{j+1}: ({points_warped[j,0]:.1f}, {points_warped[j,1]:.1f})  MAD=({mad_w[j,0]:.1f}, {mad_w[j,1]:.1f})px")

            H_inv = np.linalg.inv(H_mat)
            points_corrected = qr_detection.transform_points_back(points_warped, H_inv)
            points_corrected = points_corrected[:, :2]
            # 去掉 Z 轴旋转的自由度 (与 gimbal 版一致)
            points_corrected = qr_detection.order_points_clockwise2(points_corrected)
            print(f"  ✓ 角点已映射回原始图像")

            # 保存矫正图检测 (04_warped_detect)
            if debug:
                image_warped_detect = qr_detection.draw_qr_boundary(warped_comp.copy(), points_warped)
                filepath = os.path.join(debug_dir, "04_warped_detect.jpg")
                cv2.imwrite(filepath, image_warped_detect)
                print(f"  调试: 图像已保存到 {filepath}")
        else:
            print(f"  ⚠ 矫正图像未检测到二维码，使用YOLO检测的角点")
            points_corrected = quad_points

        # ================================================================
        # 2.5 位姿估计
        # ================================================================
        print(f"\n  2.5 位姿估计...")
        print(f"  二维码实际尺寸: {self.qr_size} m")

        object_points = pose_estimation.define_qr_object_points(qr_size=self.qr_size)

        # 相机内参: 优先使用 camera_intrinsics.txt 中的矩阵，否则用 fov_x 构造
        if self._override_camera_matrix is not None:
            camera_matrix = self._override_camera_matrix
            print(f"  使用 camera_intrinsics.txt 中的内参矩阵")
        else:
            camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=self.fov_x)

        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        print(f"  相机内参矩阵:\n{camera_matrix}")

        # 保存相机内参到 txt (方便离线测试)
        if debug and debug_dir is not None:
            cam_txt_path = os.path.join(debug_dir, "camera_intrinsics.txt")
            with open(cam_txt_path, "w") as f:
                f.write("# 相机内参矩阵 camera_matrix (3x3)\n")
                f.write("# 格式: fx,  0, cx\n")
                f.write("#        0, fy, cy\n")
                f.write("#        0,  0,  1\n\n")
                f.write(f"image_width = {W}\n")
                f.write(f"image_height = {H}\n")
                f.write(f"fov_x = {self.fov_x}\n")
                f.write(f"qr_size = {self.qr_size:.4f}\n\n")
                f.write("camera_matrix = [\n")
                for row in camera_matrix:
                    f.write(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],\n")
                f.write("]\n\n")
                f.write("dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]\n")
            print(f"  ✓ 相机内参已保存到: {cam_txt_path}")

        # PnP 估计
        success_pose, rvec, tvec, rmat = pose_estimation.estimate_pose(
            object_points, points_corrected, camera_matrix, dist_coeffs
        )

        if not success_pose:
            print(f"  ✗ 位姿估计失败")
            return False, None, None

        print(f"  ✓ 位姿估计成功")

        # 计算欧拉角
        yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(rmat)

        # 计算相机位置（世界坐标系）
        camera_position = (-rmat.T @ tvec).flatten()

        # 整理位姿数据
        pose_data = {
            "rvec": rvec.flatten(),
            "tvec": tvec.flatten(),
            "rmat": rmat,
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
            "camera_position": camera_position,
            "distance": float(np.linalg.norm(camera_position)),
            "qr_size": self.qr_size,
            "image_points": points_corrected,
            "object_points": object_points,
            "camera_matrix": camera_matrix,
        }

        # ================================================================
        # 2.6 绘制结果
        # ================================================================
        print(f"\n  2.6 绘制结果...")
        result_image = image.copy()
        result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
        result_image = pose_estimation.draw_axes(
            result_image, camera_matrix, dist_coeffs, rvec, tvec
        )
        print(f"  ✓ 结果图像已绘制")

        # 保存最终结果图 (05_result_image)
        if debug and points_corrected is not None:
            filepath = os.path.join(debug_dir, "05_result_image.jpg")
            cv2.imwrite(filepath, result_image)
            print(f"  调试: 图像已保存到 {filepath}")
            print(f"  调试: 中间图片已保存到 {debug_dir}")

        return True, pose_data, result_image

    # ── 完全复制自 GimbalPoseEstimator.print_pose_info ──
    def print_pose_info(self, pose_data: dict):
        """打印位姿信息 (与 GimbalPoseEstimator 一致)。"""
        print("\n" + "=" * 60)
        print("位姿估计结果")
        print("=" * 60)

        print(f"\n【旋转向量】 (rvec)")
        print(f"  [{pose_data['rvec'][0]:.6f}, {pose_data['rvec'][1]:.6f}, {pose_data['rvec'][2]:.6f}]")

        print(f"\n【平移向量】 (tvec) - 相机在二维码坐标系中的位置")
        print(f"  X: {pose_data['tvec'][0]:.6f} 米")
        print(f"  Y: {pose_data['tvec'][1]:.6f} 米")
        print(f"  Z: {pose_data['tvec'][2]:.6f} 米")

        print(f"\n【欧拉角】 (ZYX顺序, 度)")
        print(f"  Yaw   (绕Z轴, 偏航):   {pose_data['yaw']:.2f}°")
        print(f"  Pitch (绕Y轴, 俯仰):   {pose_data['pitch']:.2f}°")
        print(f"  Roll  (绕X轴, 横滚):   {pose_data['roll']:.2f}°")

        print(f"\n【相机位置】 (世界坐标系)")
        print(f"  X: {pose_data['camera_position'][0]:.6f} 米")
        print(f"  Y: {pose_data['camera_position'][1]:.6f} 米")
        print(f"  Z: {pose_data['camera_position'][2]:.6f} 米")
        print(f"  距离: {pose_data['distance']:.6f} 米")

        print(f"\n【其他信息】")
        print(f"  二维码尺寸: {pose_data['qr_size']} 米")
        print(f"  图像角点坐标:")
        for i, pt in enumerate(pose_data["image_points"]):
            print(f"    角点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

        # ── 云台坐标转换 (与 GimbalPoseEstimator 一致) ──
        yam_camera_input = pose_data["pitch"]
        pitch_camera_input = pose_data["roll"] - 180.0

        if abs(pitch_camera_input) >= 180:
            pitch_camera_input = (360.0 + pitch_camera_input) % 360.0

        print(f"\n【变量角度】")
        print(f"  add  Yaw:   {yam_camera_input:.2f}°")
        print(f"  add Pitch: {pitch_camera_input:.2f}°")

        # 获取当前云台角度并叠加
        if self.gimbal is not None:
            try:
                gimbal_yaw, gimbal_pitch, gimbal_roll = self.gimbal.get_current_angles()
                print(f"\n【云台当前角度】")
                print(f"  Gimbal Yaw:   {gimbal_yaw:.2f}°")
                print(f"  Gimbal Pitch: {gimbal_pitch:.2f}°")
                print(f"  Gimbal Roll:  {gimbal_roll:.2f}°")

                yam_camera_input += gimbal_yaw
                pitch_camera_input += gimbal_pitch

                print(f"\n【叠加后角度】 (位姿角度 + 云台当前角度)")
            except Exception as e:
                print(f"\n[Gimbal] 无法获取云台当前角度: {e}")
        else:
            print(f"\n[Gimbal] 云台未连接，仅输出相机相对角度")

        print("\n" + "=" * 60)
        print("手动调整云台朝向 - 参考值")
        print("=" * 60)
        print(f"\n根据位姿估计结果，建议的云台角度（仅供参考）:")
        print(f"  Yaw   (水平旋转): {yam_camera_input:.2f}°")
        print(f"  Pitch (上下俯仰): {pitch_camera_input:.2f}°")
        print(f"\n注意: 实际调整时需要考虑坐标系定义和安装方式")
        print("=" * 60)


    # ── 视频流方法 (参照 GimbalPoseEstimator) ──
    def start_video_stream(self):
        """启动实时视频流显示，按 'c' 触发二维码识别。"""
        if self.video_running:
            print("⚠ 视频流已经在运行")
            return

        if self.gimbal is None or self.rtsp_tv is None:
            print("✗ 云台未连接，无法启动视频流")
            return

        print("\n" + "-" * 60)
        print("启动实时视频流")
        print("-" * 60)

        rtsp_url = self.rtsp_tv
        print(f"  连接RTSP流: {rtsp_url}")

        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size=1024|probesize=32|analyzeduration=0'
        self.video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.video_capture.isOpened():
            print(f"  ✗ 无法打开RTSP流")
            self.video_capture = None
            return

        w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"  ✓ 视频流已打开 — {w}x{h}, {fps:.1f} FPS")

        self.video_running = True
        self.video_paused = False

        self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
        self.video_thread.start()

        print(f"\n  视频窗口已打开")
        print(f"  按 'c' - 捕获当前帧并估计位姿 (刷新ROI, 自动旋转)")
        print(f"  按 'r' - 旋转云台到上次识别目标角度")
        print(f"  按 '<- ->' - Yaw 微调 (±{self.rotation_step:.0f}°)")
        print(f"  按 '^ v'  - Pitch 微调 (±{self.rotation_step:.0f}°)")
        print(f"  按 '+/-' - 调整微调步长 (当前: {self.rotation_step:.0f}°)")
        print(f"  按 'A/D' - 变倍 放大/缩小 (自动对焦, 步长: {self.zoom_step:.0f}x)")
        print(f"  按 'f' - 手动触发自动对焦")
        print(f"  按 't' - 切换自动旋转 ({'ON' if self.auto_rotate else 'OFF'})")
        print(f"  按 'q' - 退出视频")
        print("-" * 60)

    def _video_display_loop(self):
        """视频显示循环（在线程中运行），参照 GimbalPoseEstimator._video_display_loop。"""
        window_name = "Gimbal Video Stream (Win)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        result_window_name = "Detection & ROI Result"
        cv2.namedWindow(result_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(result_window_name, 1280, 720)

        frame_count = 0

        try:
            while self.video_running:
                if self.video_paused:
                    time.sleep(0.05)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.video_paused = not self.video_paused
                    continue

                ret, frame = self.video_capture.read()

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                display_frame = frame.copy()

                with self.frame_lock:
                    self.current_frame = frame.copy()

                # ── 刷新当前云台角度 ──
                if self.gimbal is not None and frame_count % 10 == 0:
                    try:
                        gy, gp, _ = self.gimbal.get_current_angles()
                        self.current_gimbal_yaw = gy
                        self.current_gimbal_pitch = gp
                    except Exception:
                        pass

                # ── 视频叠加: 角度信息 ──
                h, w = display_frame.shape[:2]
                overlay_y = 30
                cv2.putText(display_frame, f"Gimbal Yaw: {self.current_gimbal_yaw:.1f}  Pitch: {self.current_gimbal_pitch:.1f}  Zoom: {self.current_zoom_mag:.1f}x",
                            (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if self.target_yaw != 0.0 or self.target_pitch != 0.0:
                    cv2.putText(display_frame, f"Target  Yaw: {self.target_yaw:.1f}  Pitch: {self.target_pitch:.1f}  [r] Rotate",
                                (10, overlay_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # ── 视频叠加: 操作提示 ──
                hint_lines = [
                    "[<- ->] Yaw +/-  [^ v] Pitch +/-",
                    f"[+/-] Step={self.rotation_step:.0f}deg  [t] AutoRotate={'ON' if self.auto_rotate else 'OFF'}",
                    f"[A/D] Zoom +/-{self.zoom_step:.0f}x  [f] Focus  [c] Detect QR  [q] Quit",
                ]
                for i, line in enumerate(hint_lines):
                    cv2.putText(display_frame, line,
                                (10, h - 30 - (len(hint_lines) - 1 - i) * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # ── 视频叠加: 图像中心(叉) 和 二维码中心(圆圈) ──
                img_cx, img_cy = w // 2, h // 2
                # 图像中心 — 叉号 (黄色)
                cross_len = 20
                cv2.line(display_frame, (img_cx - cross_len, img_cy - cross_len),
                         (img_cx + cross_len, img_cy + cross_len), (0, 255, 255), 2)
                cv2.line(display_frame, (img_cx - cross_len, img_cy + cross_len),
                         (img_cx + cross_len, img_cy - cross_len), (0, 255, 255), 2)

                # 二维码中心 — 圆圈+十字 (绿色)
                if self.qr_center is not None:
                    cv2.circle(display_frame, self.qr_center, 18, (0, 255, 0), 2)
                    cv2.line(display_frame,
                             (self.qr_center[0], self.qr_center[1] - 8),
                             (self.qr_center[0], self.qr_center[1] + 8),
                             (0, 255, 0), 1)
                    cv2.line(display_frame,
                             (self.qr_center[0] - 8, self.qr_center[1]),
                             (self.qr_center[0] + 8, self.qr_center[1]),
                             (0, 255, 0), 1)

                cv2.imshow(window_name, display_frame)

                # ── 显示待处理的 3D 可视化 (子线程，不阻塞主线程视频刷新) ──
                if self._pending_3d_data is not None:
                    pdata_before, pdata_after, pfov, pw, ph = self._pending_3d_data
                    self._pending_3d_data = None
                    threading.Thread(
                        target=self._visualize_3d_frustum,
                        args=(pdata_before, pdata_after, pfov, pw, ph),
                        daemon=True
                    ).start()

                # 在 ROI 窗口中保持上次识别结果
                with self.result_image_lock:
                    if self.latest_result_image is not None:
                        cv2.imshow(result_window_name, self.latest_result_image)

                key_raw = cv2.waitKeyEx(1)          # 支持方向键等扩展键
                key = key_raw & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # 主线程只截第1帧，子线程自行等后续帧 (避免阻塞主循环)
                    print(f"\n[Video] 截取5帧进行位姿估计...")
                    with self.frame_lock:
                        first_frame = self.current_frame.copy() if self.current_frame is not None else None
                    if first_frame is not None:
                        threading.Thread(
                            target=self._capture_and_estimate_multi,
                            args=(first_frame,),
                            daemon=True
                        ).start()
                    else:
                        print(f"  ✗ 当前帧为空，取消")

                # ── 旋转控制按键 ──
                elif key == ord('r'):
                    # 旋转到目标角度
                    if self.gimbal is not None:
                        threading.Thread(
                            target=self._rotate_gimbal_to_target, daemon=True
                        ).start()
                    else:
                        print("[Gimbal] 云台未连接，无法旋转")

                elif key == ord('t'):
                    # 切换自动旋转
                    self.auto_rotate = not self.auto_rotate
                    print(f"[Gimbal] 自动旋转: {'开启' if self.auto_rotate else '关闭'}")

                # ── 变倍控制 ──
                elif key == ord('a'):
                    self._zoom_step(+self.zoom_step)
                elif key == ord('d'):
                    self._zoom_step(-self.zoom_step)
                elif key == ord('f'):
                    self._auto_focus()

                elif key == ord('+') or key == ord('='):
                    self.rotation_step = min(45.0, self.rotation_step + 5.0)
                    print(f"[Gimbal] 步长调整为: {self.rotation_step:.0f}°")

                elif key == ord('-') or key == ord('_'):
                    self.rotation_step = max(1.0, self.rotation_step - 5.0)
                    print(f"[Gimbal] 步长调整为: {self.rotation_step:.0f}°")

                # ── 方向键手动旋转 ──
                elif key_raw in (0x250000, 2424832, 65361, 1113937):   # Left → Yaw-
                    self._manual_rotate(dyaw=-self.rotation_step, dpitch=0)
                elif key_raw in (0x270000, 2555904, 65363, 1113939):   # Right → Yaw+
                    self._manual_rotate(dyaw=self.rotation_step, dpitch=0)
                elif key_raw in (0x260000, 2490368, 65362, 1113938):   # Up → Pitch+
                    self._manual_rotate(dyaw=0, dpitch=self.rotation_step)
                elif key_raw in (0x280000, 2621440, 65364, 1113940):   # Down → Pitch-
                    self._manual_rotate(dyaw=0, dpitch=-self.rotation_step)

                # 调试: 打印未匹配的非零按键码 (方便排查)
                elif key_raw != -1 and key_raw != 0:
                    print(f"[DEBUG] Unhandled key_raw={key_raw}, key={key} (char={chr(key) if 32 <= key < 127 else '?'})")

        except Exception as e:
            print(f"✗ 视频显示错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(result_window_name)
            print("\n视频流已停止")

    def _capture_and_estimate_multi(self, first_frame):
        """多帧 points_corrected 平均 → 位姿估计 → 计算云台目标角度。

        第1帧完整 YOLO + 透视矫正 + 角点检测，后续帧复用 H_mat 做快速检测。
        子线程自行通过 frame_lock 从 self.current_frame 取后续帧。
        """
        N_FRAMES = 5

        print(f"\n{'='*50}")
        print(f"多帧角点估计 (共{N_FRAMES}帧, 第1帧完整YOLO, 后续帧快速)")
        print(f"{'='*50}")

        all_points = []       # 收集每帧的 points_corrected (4x2)
        first_pose_data = None
        H_mat = None
        warped_size = None
        alpha_mask_big = None
        quad_points = None
        margin_qr = 1500

        for i in range(N_FRAMES):
            # 取帧: 第1帧用传入的, 后续帧从 self.current_frame 取
            if i == 0:
                frame_i = first_frame
            else:
                time.sleep(0.06)  # 子线程等 60ms 让新帧进来
                with self.frame_lock:
                    frame_i = self.current_frame.copy() if self.current_frame is not None else None
                if frame_i is None:
                    print(f"    ✗ 帧 {i+1} 获取失败，跳过")
                    continue
            temp_path = os.path.join(self.output_dir, f"video_capture_{i}.jpg")
            cv2.imwrite(temp_path, frame_i)

            if i == 0:
                # ── 第1帧: 完整流程 ──
                success, pose_data, _ = self.estimate_pose_from_image(
                    temp_path, debug=False
                )
                if success and pose_data is not None:
                    # 缓存 H_mat 等参数给后续帧用
                    image = cv2.imread(temp_path)
                    detections = yolo_detection.detect_qr_yolov8(image, self.detector)
                    if len(detections) > 0:
                        polygon_xy = detections[0]["polygon_xy"].astype(np.float32)
                        try:
                            quad_points = qr_detection.approximate_quadrilateral(polygon_xy)
                            quad_points = qr_detection.order_points_clockwise2(quad_points)
                        except ValueError:
                            quad_points = None
                        if quad_points is not None:
                            shift_qr = margin_qr / 2.0
                            dst_square = np.array([
                                [shift_qr, shift_qr],
                                [self.square_size - 1 + shift_qr, shift_qr],
                                [self.square_size - 1 + shift_qr, self.square_size - 1 + shift_qr],
                                [shift_qr, self.square_size - 1 + shift_qr]
                            ], dtype=np.float32)
                            H_mat = qr_detection.compute_homography(quad_points, dst_square)
                            warped_size = (self.square_size + margin_qr, self.square_size + margin_qr)
                            alpha_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(alpha_mask, [quad_points.reshape((-1,1,2)).astype(np.int32)], 255)
                            alpha_mask_big = cv2.dilate(alpha_mask, np.ones((55,55), np.uint8), iterations=1)

                    img_pts = pose_data.get("image_points", None)
                    if img_pts is not None and len(img_pts) >= 4:
                        all_points.append(img_pts.copy())
                        first_pose_data = pose_data
                        print(f"    帧 1: ✓ (完整YOLO)")
                    else:
                        print(f"    帧 1: ✓ 位姿OK，但无角点")
                else:
                    print(f"    帧 1: ✗ 检测失败")
            else:
                # ── 后续帧: 快速路径 ──
                if H_mat is not None and alpha_mask_big is not None and quad_points is not None:
                    image_i = cv2.imread(temp_path)
                    if image_i is not None:
                        pts = self._fast_detect_points(
                            image_i, H_mat, warped_size, alpha_mask_big, quad_points
                        )
                        if pts is not None:
                            all_points.append(pts)
                            print(f"    帧 {i+1}: ✓ (快速)")
                        else:
                            print(f"    帧 {i+1}: ✗ 快速检测失败")
                    else:
                        print(f"    帧 {i+1}: ✗ 读取失败")
                else:
                    print(f"    帧 {i+1}: ✗ 缺少H_mat参数")

        if len(all_points) < 2:
            print(f"\n✗ 有效帧不足 ({len(all_points)}/{N_FRAMES})，位姿估计失败")
            fail_frame = first_frame.copy()
            cv2.putText(fail_frame, "QR NOT DETECTED",
                        (fail_frame.shape[1] // 2 - 120, fail_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            with self.result_image_lock:
                self.latest_result_image = fail_frame
            return

        # ── 对 points_corrected 做平均 ──
        stacked = np.array(all_points)  # (N, 4, 2)
        points_corrected = np.mean(stacked, axis=0).astype(np.float32)
        print(f"\n  角点平均 ({len(all_points)}/{N_FRAMES} 帧有效):")
        for j in range(4):
            std = np.std(stacked[:, j, :], axis=0)
            print(f"    角点{j+1}: ({points_corrected[j,0]:.1f}, {points_corrected[j,1]:.1f})  STD=({std[0]:.1f}, {std[1]:.1f})px")

        # ── 用平均角点重新做位姿估计 ──
        base_pose = first_pose_data
        camera_matrix = base_pose.get("camera_matrix")
        object_points = base_pose.get("object_points")
        if camera_matrix is None:
            h_img, w_img = first_frame.shape[:2]
            camera_matrix = pose_estimation.create_camera_matrix((w_img, h_img), fov_x=self.fov_x)
        if object_points is None:
            object_points = pose_estimation.define_qr_object_points(qr_size=self.qr_size)

        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        success_pose, rvec, tvec, rmat = pose_estimation.estimate_pose(
            object_points, points_corrected, camera_matrix, dist_coeffs
        )

        if not success_pose:
            print(f"  ✗ 平均角点位姿估计失败")
            fail_frame = first_frame.copy()
            cv2.putText(fail_frame, "POSE FAILED",
                        (fail_frame.shape[1] // 2 - 120, fail_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            with self.result_image_lock:
                self.latest_result_image = fail_frame
            return

        yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(rmat)
        camera_position = (-rmat.T @ tvec).flatten()

        pose_data = {
            "rvec": rvec.flatten(),
            "tvec": tvec.flatten(),
            "rmat": rmat,
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
            "camera_position": camera_position,
            "distance": float(np.linalg.norm(camera_position)),
            "qr_size": self.qr_size,
            "image_points": points_corrected,
            "object_points": object_points,
            "camera_matrix": camera_matrix,
        }

        self.print_pose_info(pose_data)

        # 用第一帧画结果图
        result_image = cv2.imread(os.path.join(self.output_dir, "video_capture_0.jpg"))
        if result_image is not None:
            result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
            result_image = pose_estimation.draw_axes(result_image, camera_matrix, dist_coeffs, rvec, tvec)

        with self.result_image_lock:
            self.latest_result_image = result_image.copy() if result_image is not None else None

        # ── 保存二维码中心像素坐标 ──
        center_pt = np.mean(points_corrected, axis=0)
        self.qr_center = (int(center_pt[0]), int(center_pt[1]))

        # ===== 步骤3: 计算云台目标角度 =====
        h, w = first_frame.shape[:2]
        qr_cx, qr_cy = self.qr_center
        dx = qr_cx - w / 2.0
        dy = qr_cy - h / 2.0

        # 从云台读取实时 FOV
        fov_x_actual = self.fov_x  # fallback
        if self.gimbal is not None:
            try:
                fov_x_actual, _, _, _ = self.gimbal.query_focal_length(timeout=1.0)
                if fov_x_actual is None:
                    fov_x_actual = self.fov_x
            except Exception:
                pass

        # 焦距 (像素): W / (2 * tan(FOV_x / 2))
        f_eff = w / (2.0 * math.tan(math.radians(fov_x_actual) / 2.0))

        dyaw   = math.degrees(math.atan(dx / f_eff))
        dpitch = -math.degrees(math.atan(dy / f_eff))

        # 死区: 偏差 < 0.1° 不调整
        ANGLE_DEAD = 0.1
        if abs(dyaw) < ANGLE_DEAD:
            dyaw = 0.0
        if abs(dpitch) < ANGLE_DEAD:
            dpitch = 0.0

        print(f"\n  QR像素偏移: ΔX={int(dx):+d}px ΔY={int(dy):+d}px | FOV={fov_x_actual:.1f}°(云台实时) → ΔYaw={dyaw:+.2f}° ΔPitch={dpitch:+.2f}°")

        if self.gimbal is not None:
            try:
                gim_yaw, gim_pitch, gim_roll = self.gimbal.get_current_angles()
                self.current_gimbal_yaw = gim_yaw
                self.current_gimbal_pitch = gim_pitch
                self.target_yaw = gim_yaw + dyaw
                self.target_pitch = gim_pitch + dpitch
            except Exception:
                self.target_yaw = dyaw
                self.target_pitch = dpitch
        else:
            self.target_yaw = dyaw
            self.target_pitch = dpitch

        print(f"  目标角度: Yaw={self.target_yaw:.2f}°, Pitch={self.target_pitch:.2f}°")

        # ── 水平偏移判断 (基于 camera_position 的 X 分量) ──
        cam_pos = pose_data.get("camera_position", np.zeros(3))
        cam_x = cam_pos[0]

        LAT_DEAD = 0.02
        if cam_x > -LAT_DEAD:
            print(f"  ▶ 相机在QR【右侧】(X={cam_x:+.3f}m)，请向【左→】移动机器人")
        elif cam_x < LAT_DEAD:
            print(f"  ▶ 相机在QR【左侧】(X={cam_x:+.3f}m)，请向【右←】移动机器人")
        else:
            print(f"  ✓ 水平方向【已对齐】(X={cam_x:+.3f}m)")

        # ===== 修正 pose_data: 叠加中心对齐角度到欧拉角 =====
        # 算法角度 → 云台角度: yaw←pitch, pitch←roll-180
        print(f"\n{'─'*40}")
        print(f"【姿态修正前】 (算法原始欧拉角)")
        print(f"  Yaw={pose_data['yaw']:.2f}°  Pitch={pose_data['pitch']:.2f}°  Roll={pose_data['roll']:.2f}°")
        print(f"  → 云台映射: Yaw_gim={pose_data['pitch']:.2f}°  Pitch_gim={pose_data['roll']-180:.2f}°")

        aligned_pose = pose_data.copy()
        aligned_yaw   = pose_data['pitch'] - dyaw          # 算法 pitch - 中心补偿 → 云台 yaw (取反使蓝色视锥朝QR中心转)
        aligned_pitch = pose_data['roll'] - 180.0 - dpitch  # 算法 roll-180 - 中心补偿 → 云台 pitch (取反使蓝色视锥朝QR中心转)
        if abs(aligned_pitch) >= 180:
            aligned_pitch = (360.0 + aligned_pitch) % 360.0

        # 根据修正后的云台角度重建旋转矩阵 (云台 yaw/pitch → 算法欧拉角 → rmat)
        # 逆向映射: 云台 yaw → 算法 pitch, 云台 pitch → 算法 roll
        aligned_algorithm_pitch = aligned_yaw               # 云台 yaw → 算法 pitch
        aligned_algorithm_roll  = aligned_pitch + 180.0     # 云台 pitch → 算法 roll
        aligned_algorithm_yaw   = pose_data['yaw']          # 算法 yaw 不变

        # 用欧拉角构造旋转矩阵
        yaw_rad   = math.radians(aligned_algorithm_yaw)
        pitch_rad = math.radians(aligned_algorithm_pitch)
        roll_rad  = math.radians(aligned_algorithm_roll)
        Rz = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                        [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                        [0, 0, 1]])
        Ry = np.array([[math.cos(pitch_rad), 0, math.sin(pitch_rad)],
                        [0, 1, 0],
                        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])
        Rx = np.array([[1, 0, 0],
                        [0, math.cos(roll_rad), -math.sin(roll_rad)],
                        [0, math.sin(roll_rad),  math.cos(roll_rad)]])
        aligned_rmat = Rz @ Ry @ Rx

        aligned_pose['yaw']   = aligned_yaw
        aligned_pose['pitch'] = aligned_pitch
        aligned_pose['roll']  = aligned_algorithm_roll % 360.0  # 更新 roll 为对齐后的值
        aligned_pose['rmat']  = aligned_rmat
        # 用新 rmat 重建 tvec (相机在QR坐标系中的位置)
        aligned_pose['tvec']  = (-aligned_rmat @ aligned_pose['camera_position']).flatten()

        print(f"【姿态修正后】 (中心对齐 + 云台映射)")
        print(f"  Yaw={aligned_yaw:.2f}°  Pitch={aligned_pitch:.2f}°  Roll={aligned_pose['roll']:.1f}°")
        print(f"  中心补偿: ΔYaw={dyaw:+.2f}°  ΔPitch={dpitch:+.2f}°")
        print(f"{'─'*40}")

        # ── 先旋转云台到中心对齐位置 ──
        if self.auto_rotate and self.gimbal is not None:
            try:
                self.gimbal.follow(self.target_yaw, self.target_pitch, 0.0)
                print(f"\n  ✓ 云台旋转到中心对齐: Yaw={self.target_yaw:.2f}°, Pitch={self.target_pitch:.2f}°")
                time.sleep(0.5)  # 等云台稳定
            except Exception as e:
                print(f"\n  ✗ 云台旋转失败: {e}")

        # ===== 终止条件判断：修正前 (红色原始姿态) =====
        print(f"\n【最佳成像评估】(基于修正前姿态 — 红色视锥)")
        self._evaluate_imaging_quality(pose_data, fov_x_actual, w, h)

        # ===== 终止条件判断：修正后 (蓝色对齐姿态) =====
        print(f"\n【最佳成像评估】(基于修正后姿态 — 蓝色视锥)")
        self._evaluate_imaging_quality(aligned_pose, fov_x_actual, w, h)

        # ===== 3D 可视化：推迟到主线程显示 (同时显示旋转前后视锥) =====
        print(f"\n【3D可视化】已准备数据 (红色=旋转前, 蓝色=旋转后)")
        self._pending_3d_data = (pose_data, aligned_pose, fov_x_actual, w, h)

        print(f"\n✓ 位姿估计完成 ({len(all_points)}/{N_FRAMES} 帧有效)，ROI 已刷新")

    def _evaluate_imaging_quality(self, pose_data, fov_x, img_w, _img_h):
        """评估当前成像是否达到最佳状态，给出终止建议。

        判断条件:
          1. 正对程度: |roll - 180°| < 阈值 (越正对越好)
          2. QR 占图比例: QR 在画面中的像素占比是否在目标范围
          3. 视锥-平面夹角: 相机光轴与 QR 法线的夹角

        参数:
          pose_data: 位姿数据
          fov_x:     当前水平 FOV (度)
          img_w:     图像宽度
        """
        roll = pose_data.get("roll", 0.0)
        tvec = pose_data.get("tvec", np.zeros(3))
        img_pts = pose_data.get("image_points", None)

        # ── 条件 1: 正对程度 ──
        pitch_dev = abs(roll - 180.0)
        if pitch_dev > 180:
            pitch_dev = abs(360.0 - pitch_dev)% 360.0

        ALIGN_THRESH = 15.0  # 正对阈值 (度)
        is_aligned = pitch_dev < ALIGN_THRESH

        # ── 条件 2: QR 占图比例 ──
        if img_pts is not None and len(img_pts) >= 4:
            # QR 在画面中的像素边长
            side_01 = np.linalg.norm(img_pts[0] - img_pts[1])
            side_12 = np.linalg.norm(img_pts[1] - img_pts[2])
            side_23 = np.linalg.norm(img_pts[2] - img_pts[3])
            side_30 = np.linalg.norm(img_pts[3] - img_pts[0])
            qr_px_side = np.mean([side_01, side_12, side_23, side_30])
            qr_fill_ratio = qr_px_side / img_w  # QR 占画面宽度的比例

            TARGET_RATIO_MIN = 0.15   # 目标占比下限 (15%)
            TARGET_RATIO_MAX = 0.40   # 目标占比上限 (40%)
            is_good_size = TARGET_RATIO_MIN <= qr_fill_ratio <= TARGET_RATIO_MAX
        else:
            qr_fill_ratio = 0
            is_good_size = False

        # ── 条件 3: 视线-平面夹角 (相机→QR位置向量 vs QR 法线) ──
        # tvec = [X, Y, Z] 是相机在 QR 坐标系中的位置
        # QR 法线方向 = [0, 0, 1] (板面向外)
        # 视线方向 (相机 → QR 原点) = -tvec / |tvec|
        tvec_norm = np.linalg.norm(tvec)
        if tvec_norm > 1e-6:
            cos_angle = abs(tvec[2]) / tvec_norm
            view_angle = math.degrees(math.acos(min(cos_angle, 1.0)))
        else:
            view_angle = 0.0
        ANGLE_THRESH = 15.0
        is_good_angle = view_angle < ANGLE_THRESH

        # ── 条件 4: 成像平面-二维码平面夹角 (相机光轴 vs QR 法线) ──
        # 相机光轴方向 = rmat.T[:, 2] (相机 Z 轴在世界坐标系中的方向)
        # QR 法线方向 = [0, 0, 1]
        # 两平面夹角 = 两法线夹角 = arccos(|光轴·QR法线|)
        rmat = pose_data.get("rmat")
        if rmat is not None:
            cam_z_axis = rmat.T[:, 2]          # 相机光轴在世界坐标系中的方向
            qr_normal = np.array([0.0, 0.0, 1.0])  # QR 法线 (Z轴正方向)
            cos_plane = abs(np.dot(cam_z_axis, qr_normal))
            plane_angle = math.degrees(math.acos(min(cos_plane, 1.0)))
        else:
            plane_angle = 0.0
        PLANE_ANGLE_THRESH = 15.0
        is_good_plane = plane_angle < PLANE_ANGLE_THRESH

        # ── 综合评估 ──
        conditions_met = sum([is_aligned, is_good_size, is_good_angle, is_good_plane])
        all_good = conditions_met == 4

        print(f"  ① 正对程度: Pitch偏差={pitch_dev:.1f}° (阈值<{ALIGN_THRESH}°) → {'✓ 已正对' if is_aligned else '✗ 需调整'}")
        print(f"  ② QR占图比: {qr_fill_ratio:.1%} (目标{TARGET_RATIO_MIN:.0%}~{TARGET_RATIO_MAX:.0%}) → {'✓ 合适' if is_good_size else '✗ 需变倍'}")
        print(f"  ③ 视线-平面夹角: {view_angle:.1f}° (阈值<{ANGLE_THRESH}°) → {'✓ 理想' if is_good_angle else '✗ 偏斜'}")
        print(f"  ④ 成像平面-二维码平面夹角: {plane_angle:.1f}° (阈值<{PLANE_ANGLE_THRESH}°) → {'✓ 正对' if is_good_plane else '✗ 偏斜'}")

        if all_good:
            print(f"\n  {'='*40}")
            print(f"  ✓✓✓ 已达到最佳成像状态！({conditions_met}/4 条件满足)")
            print(f"  {'='*40}")
            if not is_good_size and qr_fill_ratio < TARGET_RATIO_MIN:
                print(f"  建议: QR 太小，增大变倍 (当前{self.current_zoom_mag:.0f}x → 目标约{self.current_zoom_mag * TARGET_RATIO_MIN / max(qr_fill_ratio, 0.01):.0f}x)")
            elif not is_good_size and qr_fill_ratio > TARGET_RATIO_MAX:
                print(f"  建议: QR 太大，减小变倍")
            if not is_good_angle:
                print(f"  建议: 移动机器人使正对 QR")
        else:
            print(f"  状态: {conditions_met}/4 条件满足，继续调整")

    def _draw_frustum(self, ax, pose, fov_x, img_w, img_h, qr_center,
                       color='r', label='Frustum', lw=2.5):
        """在给定 axes 上绘制单个视锥体 + 视线。返回 (distance, fov_y)。"""
        rmat = pose.get("rmat")
        cam_pos = pose.get("camera_position", np.zeros(3)).flatten()

        dist = np.linalg.norm(cam_pos)
        fov_y = fov_x * img_h / img_w
        far = dist * 1.2

        cam_dir = rmat.T[:, 2]
        cam_up = rmat.T[:, 1]
        cam_right = rmat.T[:, 0]
        fc = cam_pos + cam_dir * far
        hh = far * math.tan(math.radians(fov_y) / 2)
        hw = far * math.tan(math.radians(fov_x) / 2)

        far_rect = np.array([
            fc + cam_right * (-hw) + cam_up * (-hh),
            fc + cam_right * ( hw) + cam_up * (-hh),
            fc + cam_right * ( hw) + cam_up * ( hh),
            fc + cam_right * (-hw) + cam_up * ( hh),
        ])
        for j in range(4):
            ax.plot([cam_pos[0], far_rect[j, 0]], [cam_pos[1], far_rect[j, 1]],
                    [cam_pos[2], far_rect[j, 2]], color, linewidth=lw, alpha=0.8)
        far_loop = np.vstack([far_rect, far_rect[0]])
        ax.plot(far_loop[:, 0], far_loop[:, 1], far_loop[:, 2],
                color, linewidth=lw + 0.5, alpha=0.9, label=label)
        # 视线
        ax.plot([cam_pos[0], qr_center[0]], [cam_pos[1], qr_center[1]],
                [cam_pos[2], qr_center[2]], color, linestyle='--', linewidth=1.0, alpha=0.5)
        return dist, fov_y

    def _visualize_3d_frustum(self, pose_before, pose_after, fov_x, img_w, img_h):
        """3D 可视化：红色视锥=旋转前，蓝色加粗视锥=旋转后。"""
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

        if self._fig3d is not None and plt.fignum_exists(self._fig3d.number):
            plt.close(self._fig3d)
        self._fig3d = plt.figure("QR-Camera 3D Frustum", figsize=(12, 9))
        ax = self._fig3d.add_subplot(111, projection='3d')
        self._ax3d = ax

        # QR 平面 (OpenCV: X=右, Y=上, Z=板外)
        viz_scale = 1.0
        if object_points is not None:
            qr_obj = np.array(object_points) * viz_scale
        else:
            half = qr_size * viz_scale / 2.0
            qr_obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        qr_center = np.mean(qr_obj, axis=0)

        qr_loop = np.vstack([qr_obj, qr_obj[0]])
        ax.plot(qr_loop[:, 0], qr_loop[:, 1], qr_loop[:, 2], 'k-', linewidth=2.5, label='QR Code')
        ax.add_collection3d(Poly3DCollection([qr_obj], alpha=0.15, facecolor='gray', edgecolor='none'))
        ax.text(qr_center[0], qr_center[1], qr_center[2], 'QR Code', fontsize=10, ha='center', color='black', weight='bold')

        # 世界坐标轴 (X=右红, Y=上绿, Z=板外蓝)
        world_len = qr_size * viz_scale * 1.5
        ax.quiver(0, 0, 0, world_len, 0, 0, color='r', arrow_length_ratio=0.15, linewidth=1.5, label='World X (→)')
        ax.quiver(0, 0, 0, 0, world_len, 0, color='g', arrow_length_ratio=0.15, linewidth=1.5, label='World Y (↑)')
        ax.quiver(0, 0, 0, 0, 0, world_len, color='b', arrow_length_ratio=0.15, linewidth=1.5, label='World Z (⊙)')

        # 相机位置 (OpenCV 坐标系)
        cam_pos = pose_before.get("camera_position", np.zeros(3)).flatten()
        ax.scatter(*cam_pos, c='red', s=120, marker='^', edgecolors='black', linewidth=1, label='Camera', zorder=10)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.02, 'Camera', fontsize=9, color='red', ha='center')

        # 红色视锥: 旋转前
        dist1, fov_y = self._draw_frustum(ax, pose_before, fov_x, img_w, img_h, qr_center,
                                            color='r', label='Before (red)', lw=1.5)
        # 蓝色加粗视锥: 旋转后
        _dist2, _ = self._draw_frustum(ax, pose_after, fov_x, img_w, img_h, qr_center,
                                        color='b', label='After (blue)', lw=3.0)

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

        ax.view_init(elev=25, azim=-60)
        self._fig3d.tight_layout()
        print("  ✓ 3D 视锥窗口已打开 (红色=旋转前, 蓝色加粗=旋转后) — 关闭窗口继续")
        plt.show(block=False)
        # 非阻塞等待，保持窗口响应同时让视频继续
        while plt.fignum_exists(self._fig3d.number):
            plt.pause(0.1)

    def _rotate_gimbal_to_target(self):
        """旋转云台到已计算的目标角度。"""
        if self.gimbal is None:
            print("[Gimbal] 云台未连接")
            return

        print(f"\n[Gimbal] 旋转到目标: Yaw={self.target_yaw:.2f}°, Pitch={self.target_pitch:.2f}°")
        try:
            self.gimbal.follow(self.target_yaw, self.target_pitch, 0.0)
            print(f"  ✓ 旋转完成")
        except Exception as e:
            print(f"  ✗ 旋转失败: {e}")

    def _manual_rotate(self, dyaw=0.0, dpitch=0.0):
        """手动微调云台角度 (增量)。

        参数:
            dyaw:   yaw 增量 (度)
            dpitch: pitch 增量 (度)
        """
        if self.gimbal is None:
            print("[Gimbal] 云台未连接")
            return

        try:
            gy, gp, _ = self.gimbal.get_current_angles()
            ny = gy + dyaw
            np_val = gp + dpitch
            self.gimbal.follow(ny, np_val, 0.0)
            self.current_gimbal_yaw = ny
            self.current_gimbal_pitch = np_val
            print(f"[Gimbal] 调整: Yaw={ny:.1f}° (Δ{dyaw:+.0f}), Pitch={np_val:.1f}° (Δ{dpitch:+.0f})")
        except Exception as e:
            print(f"[Gimbal] 手动旋转失败: {e}")

    def _zoom_step(self, delta: float):
        """变倍控制 (增量)，变倍后自动触发对焦。

        参数:
            delta: 变倍增量，正数为放大，负数为缩小
        """
        if self.gimbal is None:
            print("[Gimbal] 云台未连接，无法变倍")
            return

        try:
            self.current_zoom_mag += delta
            self.current_zoom_mag = max(1.0, min(self.current_zoom_mag, 30.0))
            self.gimbal.zoom_to_magnification(self.current_zoom_mag)
            print(f"[Gimbal] 变倍: {self.current_zoom_mag:.1f}x (Δ{delta:+.0f})")
            # 变倍后自动对焦
            time.sleep(0.5)
            self.gimbal.auto_focus()
            print(f"  ✓ 自动对焦完成")
        except Exception as e:
            print(f"[Gimbal] 变倍失败: {e}")

    def _auto_focus(self):
        """触发自动对焦。"""
        if self.gimbal is None:
            print("[Gimbal] 云台未连接，无法对焦")
            return

        try:
            self.gimbal.auto_focus()
            print("[Gimbal] ✓ 自动对焦完成")
        except Exception as e:
            print(f"[Gimbal] 自动对焦失败: {e}")

    def stop_video_stream(self):
        """停止视频流，与 GimbalPoseEstimator.stop_video_stream 类似。"""
        if not self.video_running:
            return

        print("\n停止视频流...")
        self.video_running = False

        if self.video_thread is not None:
            self.video_thread.join(timeout=2.0)
            self.video_thread = None

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        cv2.destroyAllWindows()
        print("  ✓ 视频流已停止")


# ============================================================
# 机器狗导航器 (始终模拟模式)
# ============================================================
class RobotDogNavigator:
    """X30 机器狗导航控制器 (Windows 模拟版本)。

    Windows 上无法连接真狗，所有运动指令打印到控制台。
    API 与 Linux robot_dog_bridge.RobotDogNavigator 完全一致。
    """

    def __init__(self, ip: str = "192.168.1.103", port: int = 43893,
                 auto_stand: bool = True, stop_distance: float = 0.8):
        self.stop_distance = stop_distance
        self._dog = None
        self._ready = False
        self._sim = True
        self._auto_stand = auto_stand

    def connect(self) -> bool:
        """起立并准备运动 (Win 跳过，直接进模拟)"""
        self._ready = True
        self._sim = True
        print("[Dog] Windows 模拟模式 — 无法连接真狗，运动指令仅打印日志")
        return True

    def disconnect(self):
        """停止运动，关闭连接"""
        self._ready = False
        print("[Dog] 已断开")

    @property
    def ready(self) -> bool:
        return self._ready

    def go_to(self, x: float, y: float, z: float) -> bool:
        """相对移动"""
        if not self._ready:
            print("[Dog] 未就绪")
            return False

        print(f"[Dog] move(x={x:.3f}m, y={y:.3f}m, z={z:.2f}°)")

        if self._sim:
            return self._sim_move(x, y, z)

        return False  # Win 永远不进这里

    def go_to_from_pose(self, pose: dict) -> bool:
        """由 QR Pose 生成狗体位移并导航。

        流程:
            1. camera_position → dog_relative (qr_position_to_dog)
            2. dog_relative → Dog.move(dx, dy, dz) (compute_dog_move)
            3. 发送 move 指令
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
        print("[Dog][SIM] ({:.2f}, {:.2f}, {:.1f}°) → 到达 ✓".format(x, y, z))
        time.sleep(0.3)
        return True


# ============================================================
# 主管线
# ============================================================
def pipeline(
    estimator: "WinPoseEstimator",
    navigator: RobotDogNavigator,
    image_path: Optional[str] = None,
    debug: bool = False,
    show: bool = False,
    camera_matrix: Optional[np.ndarray] = None,
) -> bool:
    """单次: 位姿估计 → 机器狗导航"""
    if image_path:
        print(f"\n[Pipeline] 图片模式: {image_path}")
        if camera_matrix is not None:
            estimator._override_camera_matrix = camera_matrix
        success, pose, result_img = estimator.estimate_pose_from_image(
            image_path, debug=debug)
    elif estimator.gimbal is not None:
        # ── 云台模式: 从 RTSP 抓图 ──
        print("\n[Pipeline] 云台抓图模式 ...")
        if camera_matrix is not None:
            estimator._override_camera_matrix = camera_matrix
        success, pose, result_img = estimator.estimate_pose_from_gimbal(debug=debug)
    else:
        print("\n[Pipeline] 无输入图片且云台未连接，退出")
        return False

    if not success or pose is None:
        print("[Pipeline] 位姿估计失败，跳过导航")
        return False

    estimator.print_pose_info(pose)

    if show and result_img is not None:
        cv2.imshow("QR Pose Result (Windows)", result_img)
        print("[Pipeline] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return navigator.go_to_from_pose(pose)


# ============================================================
def main():
    import argparse

    p = argparse.ArgumentParser(description="AR位姿 → X30机器狗导航 (Windows 版本)")
    # ── 云台 / 模型参数 ──
    p.add_argument("--gimbal-ip", default=None,
                   help="云台 IP 地址 (如 192.168.1.99)，若指定则连接云台")
    p.add_argument("--gimbal-port", type=int, default=6001, help="云台 UDP 控制端口")
    p.add_argument("--qr-size", type=float, default=0.10, help="QR物理尺寸(m)")
    p.add_argument("--fov-x", type=float, default=60.0, help="相机水平视场角(°)")
    p.add_argument("--model", default="../qrdet-s.onnx", help="ONNX 模型路径")
    p.add_argument("--device-id", type=int, default=0, help="(仅 Linux NPU 有效)")
    p.add_argument("--detector", default="opencv", choices=["opencv", "pyzbar"])
    p.add_argument("--square-size", type=int, default=800, help="矫正图边长")
    p.add_argument("--conf", type=float, default=0.2, help="YOLO置信度")
    p.add_argument("--nms", type=float, default=0.5, help="NMS阈值")
    # ── 输入源 (--image 和 --gimbal-ip 至少指定一个) ──
    p.add_argument("--image", default=None, help="输入图片路径 (不指定则需连接云台)")
    p.add_argument("--mode", default="single", choices=["single", "loop", "video"],
                   help="single=单次, loop=循环, video=实时视频流(按c识别)")
    # ── 机器狗 (Windows 始终模拟) ──
    p.add_argument("--dog-ip", default="192.168.1.103", help="(仅 Linux 有效)")
    p.add_argument("--dog-port", type=int, default=43893, help="(仅 Linux 有效)")
    p.add_argument("--sim", action="store_true", help="模拟模式 (Windows 强制开启)")
    p.add_argument("--stop-distance", type=float, default=0.8,
                   help="机器狗与二维码的水平安全距离(m)，默认0.8")
    p.add_argument("--no-stand", action="store_true", help="不自动起立 (Win 无影响)")
    # ── 其他 ──
    p.add_argument("--interval", type=float, default=2.0, help="循环间隔(秒)")
    p.add_argument("--debug", action="store_true", help="保存调试图像")
    p.add_argument("--show", action="store_true", help="显示结果窗口")
    p.add_argument("--camera-intrinsics", default=None,
                   help="camera_intrinsics.txt 路径（读取内参和QR参数）")
    args = p.parse_args()

    # 检查输入源: --image 和 --gimbal-ip 至少指定一个
    if args.image is None and args.gimbal_ip is None:
        p.error("必须指定 --image 或 --gimbal-ip 至少一个")

    # 如果指定了 camera_intrinsics.txt，读取并覆盖参数
    if args.camera_intrinsics:
        intrinsics = load_camera_intrinsics(args.camera_intrinsics)
        print(f"[Intrinsics] 从 {args.camera_intrinsics} 加载:")
        for k, v in intrinsics.items():
            if k != "camera_matrix":
                print(f"  {k} = {v}")
        if "fov_x" in intrinsics:
            args.fov_x = float(intrinsics["fov_x"])
        if "qr_size" in intrinsics:
            args.qr_size = float(intrinsics["qr_size"])
        args._camera_matrix = intrinsics.get("camera_matrix", None)

    # 1) 初始化位姿估计器
    print("=" * 55)
    print("AR Pose → X30 Dog Pipeline  [Windows/CPU]")
    print("=" * 55)
    estimator = WinPoseEstimator(
        model_path=args.model,
        qr_size=args.qr_size,
        fov_x=args.fov_x,
        square_size=args.square_size,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        detector_type=args.detector,
        gimbal_ip=args.gimbal_ip,
        gimbal_port=args.gimbal_port,
    )

    # 2) 初始化机器狗导航 (Windows 始终模拟)
    navigator = RobotDogNavigator(
        ip=args.dog_ip, port=args.dog_port,
        auto_stand=not args.no_stand,
        stop_distance=args.stop_distance)
    print("[Dog] 模拟模式 (Windows 不连真狗)")
    navigator._ready = True
    navigator._sim = True

    # 提取 camera_matrix（如果有）
    cam_mat = getattr(args, '_camera_matrix', None)

    try:
        if args.mode == "video":
            # ── 实时视频流模式: 按 'c' 触发识别 ──
            print("\n" + "=" * 55)
            print("实时视频流模式 — 按 'c' 捕获并识别二维码")
            print("按 'q' 退出")
            print("=" * 55)
            estimator.start_video_stream()

            try:
                while estimator.video_running:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\n用户中断")
        elif args.mode == "loop":
            if args.image:
                print(f"\n[Loop] 图片固定: {args.image}，每 {args.interval}s 一轮，按 Ctrl+C 停止\n")
            else:
                print(f"\n[Loop] 云台 RTSP 抓图，每 {args.interval}s 一轮，按 Ctrl+C 停止\n")
            count = 0
            while True:
                count += 1
                print(f"\n{'─' * 40}\n 第 {count} 轮\n{'─' * 40}")
                pipeline(estimator, navigator, args.image,
                         debug=args.debug, show=args.show,
                         camera_matrix=cam_mat)
                time.sleep(args.interval)
        else:
            pipeline(estimator, navigator, args.image,
                     debug=args.debug, show=args.show,
                     camera_matrix=cam_mat)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        navigator.disconnect()
        estimator.cleanup()


if __name__ == "__main__":
    main()
