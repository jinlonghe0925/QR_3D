#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR 位姿 → 机器狗导航 (Windows 版本 v2 — 模块化 + 统一配置)
================================================================
基于 robot_dog_bridge_win.py 的模块化重构版本，所有参数统一由 app_config.json 管理。

模块结构:
    config_utils.py       — 配置加载 (app_config.json + camera_intrinsics.txt)
    dog_navigator.py      — 坐标系转换 + 机器狗导航器
    pose_pipeline.py      — 位姿估计核心 (QR检测→PnP→姿态修正→成像评估)
    gimbal_operator.py    — 云台控制 (旋转、变倍、对焦)
    visualizer_3d.py      — 3D 视锥可视化

配置文件:
    app_config.json       — 所有运行参数 (模型、相机、云台、成像评估、狗导航等)
    camera_intrinsics.txt — 相机内参矩阵 (可选, 通过 --camera-intrinsics 指定)

原文件 robot_dog_bridge_win.py 保留不变作为测试参考。

用法:
    # 云台实时视频模式 (使用默认 app_config.json)
    python robot_dog_bridge_win_v2.py --gimbal-ip 192.168.1.99 --mode video

    # 自定义配置文件
    python robot_dog_bridge_win_v2.py --config my_config.json --mode video

    # 单张图片模式
    python robot_dog_bridge_win_v2.py --image ./test.jpg --mode single

    # 循环模式
    python robot_dog_bridge_win_v2.py --image ./test.jpg --mode loop --interval 1.0

    # 命令行覆盖配置 (--model, --qr-size, --fov-x 等会覆盖 app_config.json)
    python robot_dog_bridge_win_v2.py --model ../qrdet-s.onnx --qr-size 0.15 --fov-x 70.0 --mode video
"""

import sys
import time
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# ── 路径注入 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── 基础模块 ──
import qr_detection
import pose_estimation
import yolo_detection_ascend as yolo_detection

# ── 配置模块 ──
from config_utils import load_camera_intrinsics, load_app_config, get_config_value

# ── 功能模块 ──
from dog_navigator import RobotDogNavigator
from pose_pipeline import (
    estimate_pose_from_image,
    fast_detect_points,
    print_pose_info,
    compute_gimbal_target,
    judge_lateral_position,
    build_aligned_pose,
    evaluate_imaging_quality,
)
from gimbal_operator import GimbalOperator
from visualizer_3d import FrustumVisualizer

# ── 云台控制 (条件导入) ──
try:
    from gimbal_demo import GimbalController, capture_from_rtsp
    _GIMBAL_AVAILABLE = True
except ImportError:
    _GIMBAL_AVAILABLE = False


# ============================================================
# WinPoseEstimatorV2 — 组合式位姿估计器
# ============================================================
class WinPoseEstimatorV2:
    """模块化版本的 Windows 位姿估计器。

    将位姿估计、云台控制、3D 可视化分别委托给独立模块。
    所有参数从 app_config.json 加载，命令行参数可覆盖。
    """

    def __init__(self, config: dict,
                 gimbal_ip: Optional[str] = None,
                 gimbal_port: Optional[int] = None):
        """初始化估计器。

        参数:
            config:      应用配置字典 (load_app_config 返回值)
            gimbal_ip:   云台 IP (命令行覆盖)
            gimbal_port: 云台端口 (命令行覆盖)
        """
        self._cfg = config  # 保存完整配置引用

        # ── 模型参数 ──
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("model_path", "../qrdet-s.onnx")
        self.conf_threshold = model_cfg.get("conf_threshold", 0.2)
        self.nms_threshold = model_cfg.get("nms_threshold", 0.5)

        # ── 相机参数 ──
        camera_cfg = config.get("camera", {})
        self.qr_size = camera_cfg.get("qr_size", 0.10)
        self.fov_x = camera_cfg.get("fov_x", 60.0)
        self._override_camera_matrix = None

        # ── 位姿估计参数 ──
        pe_cfg = config.get("pose_estimation", {})
        self.square_size = pe_cfg.get("square_size", 800)
        self.margin_qr = pe_cfg.get("margin_qr", 1500)
        self.detector_type = pe_cfg.get("detector_type", "opencv")
        self.warped_detect_count = pe_cfg.get("warped_detect_count", 5)
        self.warped_detect_min_success = pe_cfg.get("warped_detect_min_success", 2)
        self.dilate_kernel_size = pe_cfg.get("dilate_kernel_size", 55)
        self.output_dir = pe_cfg.get("output_dir", "gimbal_pose_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # ── 多帧参数 ──
        mf_cfg = config.get("multi_frame", {})
        self.n_frames = mf_cfg.get("n_frames", 5)
        self.frame_interval = mf_cfg.get("frame_interval_sec", 0.06)
        self.min_valid_frames = mf_cfg.get("min_valid_frames", 2)

        # ── 视频流 ──
        self.video_capture = None
        self.video_thread = None
        self.video_running = False
        self.video_paused = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.latest_result_image = None
        self.result_image_lock = threading.Lock()

        video_cfg = config.get("video", {})
        self._win_w = video_cfg.get("window_width", 1280)
        self._win_h = video_cfg.get("window_height", 720)
        self._angle_refresh = video_cfg.get("angle_refresh_interval", 10)
        self._cross_len = video_cfg.get("cross_len", 20)
        self._qr_circle_r = video_cfg.get("qr_circle_radius", 18)

        # ── 云台操作器 (从配置初始化) ──
        self.gimbal_op = GimbalOperator(config=config)
        self.auto_rotate = config.get("gimbal", {}).get("auto_rotate", True)

        # ── 视频叠加 ──
        self.qr_center = None

        # ── 3D 可视化 (从配置初始化) ──
        self._viz3d = FrustumVisualizer(config=config)
        self._pending_3d_data = None

        # ── 云台控制器 ──
        gimbal_cfg = config.get("gimbal", {})
        self.gimbal = None
        self.rtsp_tv = None
        # 命令行参数优先
        _gimbal_ip = gimbal_ip if gimbal_ip else gimbal_cfg.get("ip")
        _gimbal_port = gimbal_port if gimbal_port else gimbal_cfg.get("port", 6001)
        self.gimbal_ip = _gimbal_ip
        self.gimbal_port = _gimbal_port

        if _gimbal_ip is not None and _GIMBAL_AVAILABLE:
            self._connect_gimbal(_gimbal_ip, _gimbal_port)
        self.gimbal_op.set_gimbal(self.gimbal)

        # ── YOLO 检测器 ──
        print("[Detector] 加载 ONNX 模型: {}".format(self.model_path))
        self.detector = yolo_detection.load_yolov8_detector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        if self.detector is None:
            raise RuntimeError("ONNX 检测器加载失败，请确认已安装 onnxruntime 且模型路径正确")

    def update_params_from_args(self, model_path=None, qr_size=None, fov_x=None,
                                 square_size=None, conf=None, nms=None, detector=None,
                                 camera_matrix=None):
        """用命令行参数覆盖配置值。"""
        if model_path is not None:
            self.model_path = model_path
        if qr_size is not None:
            self.qr_size = qr_size
        if fov_x is not None:
            self.fov_x = fov_x
        if square_size is not None:
            self.square_size = square_size
        if conf is not None:
            self.conf_threshold = conf
        if nms is not None:
            self.nms_threshold = nms
        if detector is not None:
            self.detector_type = detector
        if camera_matrix is not None:
            self._override_camera_matrix = camera_matrix

    def _connect_gimbal(self, gimbal_ip: str, gimbal_port: int):
        """连接云台控制器。"""
        print(f"\n[Gimbal] 连接云台: {gimbal_ip}:{gimbal_port}")
        try:
            self.gimbal = GimbalController(ip=gimbal_ip, port=gimbal_port)
            # RTSP 路径: 从配置读取
            rtsp_path = self._cfg.get("gimbal", {}).get("rtsp_path", "/ch1/stream0")
            self.rtsp_tv = f"rtsp://{gimbal_ip}:554{rtsp_path}"
            print(f"  ✓ 云台连接成功")
            print(f"  可见光 RTSP: {self.rtsp_tv}")
        except Exception as e:
            print(f"  ✗ 云台连接失败: {e}")
            self.gimbal = None
            self.rtsp_tv = None

    def cleanup(self):
        """清理资源。"""
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
        """从云台 RTSP 流抓取可见光图像。"""
        if self.gimbal is None or self.rtsp_tv is None:
            print("[Capture] 云台未连接")
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
                print(f"  ✗ 图像读取失败")
                return False, None
        else:
            print(f"  ✗ RTSP 抓图失败")
            return False, None

    def estimate_pose_from_gimbal(self, debug=False, debug_dir=None):
        """从云台抓图 → 位姿估计。"""
        success, image_path = self.capture_visible_image()
        if not success:
            return False, None, None

        image = cv2.imread(image_path)
        if image is None:
            return False, None, None

        return self._run_pose_estimation(image, debug=debug, debug_dir=debug_dir)

    def _run_pose_estimation(self, image: np.ndarray, debug=False, debug_dir=None):
        """执行完整位姿估计管线 (委托给 pose_pipeline 模块)。"""
        success, pose_data, result_image, _extra = estimate_pose_from_image(
            image=image,
            detector=self.detector,
            qr_size=self.qr_size,
            fov_x=self.fov_x,
            square_size=self.square_size,
            detector_type=self.detector_type,
            output_dir=self.output_dir,
            override_camera_matrix=self._override_camera_matrix,
            debug=debug,
            debug_dir=debug_dir,
            yolo_detection_module=yolo_detection,
            margin_qr=self.margin_qr,
            dilate_kernel_size=self.dilate_kernel_size,
            warped_detect_count=self.warped_detect_count,
            warped_detect_min_success=self.warped_detect_min_success,
        )
        return success, pose_data, result_image

    # ================================================================
    # 视频流
    # ================================================================
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

        step = self.gimbal_op.rotation_step
        zoom_step = self.gimbal_op.zoom_step_val
        print(f"\n  视频窗口已打开")
        print(f"  按 'c' - 捕获当前帧并估计位姿 (刷新ROI, 自动旋转)")
        print(f"  按 'r' - 旋转云台到上次识别目标角度")
        print(f"  按 '<- ->' - Yaw 微调 (±{step:.0f}°)")
        print(f"  按 '^ v'  - Pitch 微调 (±{step:.0f}°)")
        print(f"  按 '+/-' - 调整微调步长 (当前: {step:.0f}°)")
        print(f"  按 'A/D' - 变倍 放大/缩小 (步长: {zoom_step:.0f}x)")
        print(f"  按 'f' - 手动触发自动对焦")
        print(f"  按 't' - 切换自动旋转 ({'ON' if self.auto_rotate else 'OFF'})")
        print(f"  按 'q' - 退出视频")
        print("-" * 60)

    def _video_display_loop(self):
        """视频显示主循环 (在线程中运行)。"""
        window_name = "Gimbal Video Stream (Win v2)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self._win_w, self._win_h)

        result_window_name = "Detection & ROI Result"
        cv2.namedWindow(result_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(result_window_name, self._win_w, self._win_h)

        frame_count = 0
        op = self.gimbal_op

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

                # 刷新云台角度
                if self.gimbal is not None and frame_count % self._angle_refresh == 0:
                    op.update_current_angles()

                # ── 视频叠加 ──
                h, w = display_frame.shape[:2]
                overlay_y = 30
                cv2.putText(display_frame,
                            f"Gimbal Yaw: {op.current_gimbal_yaw:.1f}  Pitch: {op.current_gimbal_pitch:.1f}  Zoom: {op.current_zoom_mag:.1f}x",
                            (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if op.target_yaw != 0.0 or op.target_pitch != 0.0:
                    cv2.putText(display_frame,
                                f"Target  Yaw: {op.target_yaw:.1f}  Pitch: {op.target_pitch:.1f}  [r] Rotate",
                                (10, overlay_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                hint_lines = [
                    "[<- ->] Yaw +/-  [^ v] Pitch +/-",
                    f"[+/-] Step={op.rotation_step:.0f}deg  [t] AutoRotate={'ON' if self.auto_rotate else 'OFF'}",
                    f"[A/D] Zoom +/-{op.zoom_step_val:.0f}x  [f] Focus  [c] Detect QR  [q] Quit",
                ]
                for i, line in enumerate(hint_lines):
                    cv2.putText(display_frame, line,
                                (10, h - 30 - (len(hint_lines) - 1 - i) * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # 图像中心 (黄色叉号) + QR 中心 (绿色圆圈)
                img_cx, img_cy = w // 2, h // 2
                cl = self._cross_len
                cv2.line(display_frame, (img_cx - cl, img_cy - cl),
                         (img_cx + cl, img_cy + cl), (0, 255, 255), 2)
                cv2.line(display_frame, (img_cx - cl, img_cy + cl),
                         (img_cx + cl, img_cy - cl), (0, 255, 255), 2)

                if self.qr_center is not None:
                    r = self._qr_circle_r
                    cv2.circle(display_frame, self.qr_center, r, (0, 255, 0), 2)
                    cv2.line(display_frame,
                             (self.qr_center[0], self.qr_center[1] - 8),
                             (self.qr_center[0], self.qr_center[1] + 8), (0, 255, 0), 1)
                    cv2.line(display_frame,
                             (self.qr_center[0] - 8, self.qr_center[1]),
                             (self.qr_center[0] + 8, self.qr_center[1]), (0, 255, 0), 1)

                cv2.imshow(window_name, display_frame)

                # 3D 可视化 (主线程非阻塞: 只在无3D窗口时处理按键)
                if self._pending_3d_data is not None:
                    pdata_before, pdata_after, pfov, pw, ph = self._pending_3d_data
                    self._pending_3d_data = None
                    self._viz3d.show_nonblocking(pdata_before, pdata_after, pfov, pw, ph)

                # ROI 窗口
                with self.result_image_lock:
                    if self.latest_result_image is not None:
                        cv2.imshow(result_window_name, self.latest_result_image)

                # ── 键盘处理 (仅在没有3D窗口时响应, 避免matplotlib拦截按键) ──
                try:
                    import matplotlib.pyplot as _plt
                    _has_3d = self._viz3d.fig is not None and _plt.fignum_exists(self._viz3d.fig.number)
                except Exception:
                    _has_3d = False

                if _has_3d:
                    # 3D窗口存在: 仅响应 'q' 退出和关闭窗口
                    key_raw = cv2.waitKeyEx(1)
                    key = key_raw & 0xFF
                    if key == ord('q'):
                        break
                    # 用 plt.pause 保持3D窗口响应
                    _plt.pause(0.05)
                else:
                    key_raw = cv2.waitKeyEx(1)
                    key = key_raw & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        print(f"\n[Video] 截取{self.n_frames}帧进行位姿估计...")
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
                    elif key == ord('r'):
                        if self.gimbal is not None:
                            threading.Thread(target=op.rotate_to_target, daemon=True).start()
                        else:
                            print("[Gimbal] 云台未连接，无法旋转")
                    elif key == ord('t'):
                        self.auto_rotate = not self.auto_rotate
                        print(f"[Gimbal] 自动旋转: {'开启' if self.auto_rotate else '关闭'}")
                    elif key == ord('a'):
                        op.zoom(+op.zoom_step_val)
                    elif key == ord('d'):
                        op.zoom(-op.zoom_step_val)
                    elif key == ord('f'):
                        op.auto_focus()
                    elif key == ord('+') or key == ord('='):
                        op.adjust_rotation_step(+5.0)
                    elif key == ord('-') or key == ord('_'):
                        op.adjust_rotation_step(-5.0)
                    elif key_raw == 2424832:  # 左箭头
                        op.manual_adjust(dyaw=-op.rotation_step, dpitch=0)
                    elif key_raw == 2555904:  # 右箭头
                        op.manual_adjust(dyaw=op.rotation_step, dpitch=0)
                    elif key_raw == 2490368:  # 上箭头
                        op.manual_adjust(dyaw=0, dpitch=op.rotation_step)
                    elif key_raw == 2621440:  # 下箭头
                        op.manual_adjust(dyaw=0, dpitch=-op.rotation_step)

        except Exception as e:
            print(f"\n[Video] 循环异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.video_running = False
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(result_window_name)
            print("\n视频流已停止")

    def _capture_and_estimate_multi(self, first_frame):
        """多帧角点平均 → 位姿估计 → 计算云台目标角度。

        第1帧完整 YOLO+透视矫正，后续帧复用 H_mat 做快速检测。
        """
        all_points = []
        first_pose_data = None
        H_mat = None
        warped_size = None
        alpha_mask_big = None
        quad_points = None

        print(f"\n{'='*50}")
        print(f"多帧角点估计 (共{self.n_frames}帧, 第1帧完整YOLO, 后续帧快速)")
        print(f"{'='*50}")

        for i in range(self.n_frames):
            if i == 0:
                frame_i = first_frame
            else:
                time.sleep(self.frame_interval)
                with self.frame_lock:
                    frame_i = self.current_frame.copy() if self.current_frame is not None else None
                if frame_i is None:
                    print(f"    ✗ 帧 {i+1} 获取失败，跳过")
                    continue
            temp_path = os.path.join(self.output_dir, f"video_capture_{i}.jpg")
            cv2.imwrite(temp_path, frame_i)

            if i == 0:
                success, pose_data, result_img, extra = estimate_pose_from_image(
                    image=frame_i,
                    detector=self.detector,
                    qr_size=self.qr_size,
                    fov_x=self.fov_x,
                    square_size=self.square_size,
                    detector_type=self.detector_type,
                    output_dir=self.output_dir,
                    override_camera_matrix=self._override_camera_matrix,
                    debug=False,
                    yolo_detection_module=yolo_detection,
                    margin_qr=self.margin_qr,
                    dilate_kernel_size=self.dilate_kernel_size,
                    warped_detect_count=self.warped_detect_count,
                    warped_detect_min_success=self.warped_detect_min_success,
                )
                if success and pose_data is not None:
                    H_mat = extra.get("H_mat")
                    warped_size = extra.get("warped_size")
                    alpha_mask_big = extra.get("alpha_mask_big")
                    quad_points = extra.get("quad_points")

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
                if H_mat is not None and alpha_mask_big is not None and quad_points is not None:
                    pts = fast_detect_points(
                        frame_i, H_mat, warped_size, alpha_mask_big, quad_points,
                        detector_type=self.detector_type,
                    )
                    if pts is not None:
                        all_points.append(pts)
                        print(f"    帧 {i+1}: ✓ (快速)")
                    else:
                        print(f"    帧 {i+1}: ✗ 快速检测失败")
                else:
                    print(f"    帧 {i+1}: ✗ 缺少H_mat参数")

        if len(all_points) < self.min_valid_frames:
            print(f"\n✗ 有效帧不足 ({len(all_points)}/{self.n_frames})")
            fail_frame = first_frame.copy()
            cv2.putText(fail_frame, "QR NOT DETECTED",
                        (fail_frame.shape[1] // 2 - 120, fail_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            with self.result_image_lock:
                self.latest_result_image = fail_frame
            return

        # 角点平均
        stacked = np.array(all_points)
        points_corrected = np.mean(stacked, axis=0).astype(np.float32)
        print(f"\n  角点平均 ({len(all_points)}/{self.n_frames} 帧有效):")
        for j in range(4):
            std = np.std(stacked[:, j, :], axis=0)
            print(f"    角点{j+1}: ({points_corrected[j,0]:.1f}, {points_corrected[j,1]:.1f})  STD=({std[0]:.1f}, {std[1]:.1f})px")

        # 用平均角点重新 PnP
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

        print_pose_info(pose_data, gimbal=self.gimbal)

        # 画结果图
        result_image = cv2.imread(os.path.join(self.output_dir, "video_capture_0.jpg"))
        if result_image is not None:
            result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
            result_image = pose_estimation.draw_axes(result_image, camera_matrix, dist_coeffs, rvec, tvec)
        with self.result_image_lock:
            self.latest_result_image = result_image.copy() if result_image is not None else None

        # QR 中心像素
        center_pt = np.mean(points_corrected, axis=0)
        self.qr_center = (int(center_pt[0]), int(center_pt[1]))

        # ── 计算云台目标角度 ──
        h, w = first_frame.shape[:2]
        qr_cx, qr_cy = self.qr_center
        dx = qr_cx - w / 2.0
        dy = qr_cy - h / 2.0

        fov_x_actual = self.gimbal_op.query_fov(fallback_fov=self.fov_x)
        angle_dead = self.gimbal_op.angle_dead_zone
        dyaw, dpitch, target_yaw, target_pitch = compute_gimbal_target(
            dx, dy, fov_x_actual, w, h, gimbal=self.gimbal,
            angle_dead_zone_deg=angle_dead,
        )
        self.gimbal_op.target_yaw = target_yaw
        self.gimbal_op.target_pitch = target_pitch

        # ── 水平偏移判断 → 机器人移动建议 ──
        lat_dead = get_config_value(self._cfg, "lateral_judge", "dead_zone_m", default=0.02)
        judge_lateral_position(pose_data, lat_dead=lat_dead)

        # ── 修正位姿数据 ──
        print(f"\n{'─'*40}")
        print(f"【姿态修正前】 (算法原始欧拉角)")
        print(f"  Yaw={pose_data['yaw']:.2f}°  Pitch={pose_data['pitch']:.2f}°  Roll={pose_data['roll']:.2f}°")
        print(f"  → 云台映射: Yaw_gim={pose_data['pitch']:.2f}°  Pitch_gim={pose_data['roll']-180:.2f}°")

        aligned_pose = build_aligned_pose(pose_data, dyaw, dpitch)
        print(f"【姿态修正后】 (中心对齐 + 云台映射)")
        print(f"  Yaw={aligned_pose['yaw']:.2f}°  Pitch={aligned_pose['pitch']:.2f}°  Roll={aligned_pose['roll']:.1f}°")
        print(f"  中心补偿: ΔYaw={dyaw:+.2f}°  ΔPitch={dpitch:+.2f}°")
        print(f"{'─'*40}")

        # ── 旋转云台到中心对齐位置 ──
        if self.auto_rotate and self.gimbal is not None:
            try:
                self.gimbal.follow(target_yaw, target_pitch, 0.0)
                delay = self.gimbal_op.stable_delay
                print(f"\n  ✓ 云台旋转到中心对齐: Yaw={target_yaw:.2f}°, Pitch={target_pitch:.2f}°")
                time.sleep(delay)
            except Exception as e:
                print(f"\n  ✗ 云台旋转失败: {e}")

        # ── 成像质量评估：修正前 (红色原始姿态) ──
        iq_cfg = self._cfg.get("imaging_quality", {})
        print(f"\n【最佳成像评估】(基于修正前姿态 — 红色视锥)")
        evaluate_imaging_quality(
            pose_data, fov_x_actual, w, h,
            align_thresh_deg=iq_cfg.get("align_thresh_deg", 15.0),
            qr_fill_ratio_min=iq_cfg.get("qr_fill_ratio_min", 0.15),
            qr_fill_ratio_max=iq_cfg.get("qr_fill_ratio_max", 0.40),
            view_angle_thresh_deg=iq_cfg.get("view_angle_thresh_deg", 15.0),
            plane_angle_thresh_deg=iq_cfg.get("plane_angle_thresh_deg", 15.0),
        )

        # ── 成像质量评估：修正后 (蓝色对齐姿态) ──
        print(f"\n【最佳成像评估】(基于修正后姿态 — 蓝色视锥)")
        evaluate_imaging_quality(
            aligned_pose, fov_x_actual, w, h,
            align_thresh_deg=iq_cfg.get("align_thresh_deg", 15.0),
            qr_fill_ratio_min=iq_cfg.get("qr_fill_ratio_min", 0.15),
            qr_fill_ratio_max=iq_cfg.get("qr_fill_ratio_max", 0.40),
            view_angle_thresh_deg=iq_cfg.get("view_angle_thresh_deg", 15.0),
            plane_angle_thresh_deg=iq_cfg.get("plane_angle_thresh_deg", 15.0),
        )

        # ── 3D 可视化 (红色=旋转前, 蓝色=旋转后) ──
        print(f"\n【3D可视化】已准备数据 (红色=旋转前, 蓝色=旋转后)")
        self._pending_3d_data = (pose_data, aligned_pose, fov_x_actual, w, h)

        print(f"\n✓ 位姿估计完成 ({len(all_points)}/{self.n_frames} 帧有效)，ROI 已刷新")

    def stop_video_stream(self):
        """停止视频流。"""
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
# 主管线
# ============================================================
def pipeline(estimator: WinPoseEstimatorV2,
             navigator: RobotDogNavigator,
             image_path: Optional[str] = None,
             debug: bool = False,
             show: bool = False,
             camera_matrix: Optional[np.ndarray] = None) -> bool:
    """单次: 位姿估计 → 机器狗导航。

    参数:
        estimator:      位姿估计器
        navigator:      机器狗导航器
        image_path:     输入图片路径 (None=从云台RTSP抓图)
        debug:          是否输出调试图片
        show:           是否弹窗显示结果
        camera_matrix:  相机内参矩阵 (覆盖配置文件)

    返回:
        True=导航成功, False=失败
    """
    if image_path:
        print(f"\n[Pipeline] 图片模式: {image_path}")
        if camera_matrix is not None:
            estimator._override_camera_matrix = camera_matrix
        image = cv2.imread(image_path)
        if image is None:
            print("  ✗ 无法读取图片")
            return False
        success, pose, result_img = estimator._run_pose_estimation(image, debug=debug)
    elif estimator.gimbal is not None:
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

    print_pose_info(pose, gimbal=estimator.gimbal)

    if show and result_img is not None:
        cv2.imshow("QR Pose Result (Windows)", result_img)
        print("[Pipeline] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return navigator.go_to_from_pose(pose)


# ============================================================
# 入口
# ============================================================
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="AR位姿 → X30机器狗导航 (Windows 模块化版本 v2)",
        epilog="所有参数也可通过 app_config.json 配置, 命令行参数优先级更高。"
    )
    # ── 配置文件 ──
    p.add_argument("--config", default=None, help="应用配置文件路径 (默认: app_config.json)")
    # ── 云台 / 模型参数 (命令行覆盖配置文件) ──
    p.add_argument("--gimbal-ip", default=None, help="云台 IP 地址 (覆盖配置文件)")
    p.add_argument("--gimbal-port", type=int, default=None, help="云台 UDP 控制端口")
    p.add_argument("--qr-size", type=float, default=None, help="QR物理尺寸(m)")
    p.add_argument("--fov-x", type=float, default=None, help="相机水平视场角(°)")
    p.add_argument("--model", default=None, help="ONNX 模型路径")
    p.add_argument("--device-id", type=int, default=0, help="(仅 Linux NPU 有效)")
    p.add_argument("--detector", default=None, choices=["opencv", "pyzbar"])
    p.add_argument("--square-size", type=int, default=None, help="矫正图边长")
    p.add_argument("--conf", type=float, default=None, help="YOLO置信度")
    p.add_argument("--nms", type=float, default=None, help="NMS阈值")
    # ── 输入源 ──
    p.add_argument("--image", default=None, help="输入图片路径")
    p.add_argument("--mode", default="single", choices=["single", "loop", "video"],
                   help="single=单次, loop=循环, video=实时视频流(按c识别)")
    # ── 机器狗 (Windows 始终模拟) ──
    p.add_argument("--dog-ip", default=None, help="(仅 Linux 有效)")
    p.add_argument("--dog-port", type=int, default=None, help="(仅 Linux 有效)")
    p.add_argument("--sim", action="store_true", help="模拟模式 (Windows 强制开启)")
    p.add_argument("--stop-distance", type=float, default=None, help="安全距离(m)")
    p.add_argument("--no-stand", action="store_true", help="不自动起立 (Win 无影响)")
    # ── 其他 ──
    p.add_argument("--interval", type=float, default=None, help="循环间隔(秒)")
    p.add_argument("--debug", action="store_true", help="保存调试图像")
    p.add_argument("--show", action="store_true", help="显示结果窗口")
    p.add_argument("--camera-intrinsics", default=None,
                   help="camera_intrinsics.txt 路径（读取内参和QR参数）")
    args = p.parse_args()

    # ── 1) 加载配置文件 ──
    config = load_app_config(args.config)
    print(f"[Config] 配置文件已加载")

    # ── 2) 加载 camera_intrinsics.txt (可选) ──
    cam_mat = None
    if args.camera_intrinsics:
        intrinsics = load_camera_intrinsics(args.camera_intrinsics)
        print(f"[Intrinsics] 从 {args.camera_intrinsics} 加载:")
        for k, v in intrinsics.items():
            if k != "camera_matrix":
                print(f"  {k} = {v}")
        # 用 intrinsics 覆盖 config
        if "fov_x" in intrinsics:
            config.setdefault("camera", {})["fov_x"] = float(intrinsics["fov_x"])
        if "qr_size" in intrinsics:
            config.setdefault("camera", {})["qr_size"] = float(intrinsics["qr_size"])
        cam_mat = intrinsics.get("camera_matrix", None)

    # ── 3) 命令行参数覆盖配置文件 ──
    if args.fov_x is not None:
        config.setdefault("camera", {})["fov_x"] = args.fov_x
    if args.qr_size is not None:
        config.setdefault("camera", {})["qr_size"] = args.qr_size
    if args.model is not None:
        config.setdefault("model", {})["model_path"] = args.model
    if args.conf is not None:
        config.setdefault("model", {})["conf_threshold"] = args.conf
    if args.nms is not None:
        config.setdefault("model", {})["nms_threshold"] = args.nms
    if args.square_size is not None:
        config.setdefault("pose_estimation", {})["square_size"] = args.square_size
    if args.detector is not None:
        config.setdefault("pose_estimation", {})["detector_type"] = args.detector
    if args.stop_distance is not None:
        config.setdefault("dog_navigator", {})["stop_distance_m"] = args.stop_distance
    if args.interval is not None:
        config.setdefault("loop", {})["interval_sec"] = args.interval

    # 检查输入源
    if args.image is None and args.gimbal_ip is None and config.get("gimbal", {}).get("ip") is None:
        p.error("必须指定 --image 或 --gimbal-ip 至少一个 (或在 app_config.json 中配置 gimbal.ip)")

    print("=" * 55)
    print("AR Pose → X30 Dog Pipeline  [Windows/CPU v2]")
    print("=" * 55)

    # ── 4) 初始化估计器 ──
    estimator = WinPoseEstimatorV2(
        config=config,
        gimbal_ip=args.gimbal_ip,
        gimbal_port=args.gimbal_port,
    )

    # ── 5) 初始化机器狗导航 ──
    navigator = RobotDogNavigator(config=config)
    if args.dog_ip is not None:
        navigator._dog_ip = args.dog_ip
    print("[Dog] 模拟模式 (Windows 不连真狗)")
    navigator._ready = True
    navigator._sim = True

    # ── 6) 运行 ──
    loop_interval = config.get("loop", {}).get("interval_sec", 2.0)

    try:
        if args.mode == "video":
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
                print(f"\n[Loop] 图片固定: {args.image}，每 {loop_interval}s 一轮\n")
            else:
                print(f"\n[Loop] 云台 RTSP 抓图，每 {loop_interval}s 一轮\n")
            count = 0
            while True:
                count += 1
                print(f"\n{'─' * 40}\n 第 {count} 轮\n{'─' * 40}")
                pipeline(estimator, navigator, args.image,
                         debug=args.debug, show=args.show,
                         camera_matrix=cam_mat)
                time.sleep(loop_interval)
        else:
            pipeline(estimator, navigator, args.image,
                     debug=args.debug, show=args.show,
                     camera_matrix=cam_mat)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        estimator.cleanup()
        print("\nDone.")


if __name__ == "__main__":
    main()
