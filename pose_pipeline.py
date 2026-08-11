#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""位姿估计核心模块
纯位姿估计逻辑，不依赖云台、视频流。

对应 app_config.json 配置节:
    pose_estimation:    square_size, margin_qr, detector_type,
                        warped_detect_count, dilate_kernel_size, output_dir
    multi_frame:        n_frames, frame_interval_sec, min_valid_frames
    imaging_quality:    align_thresh_deg, qr_fill_ratio_min/max, view_angle_thresh_deg
    lateral_judge:      dead_zone_m
    gimbal:             angle_dead_zone_deg
"""

import os
import time as _time
import math
import numpy as np
import cv2
from typing import Tuple, Optional

# ── 依赖同目录模块 ──
import qr_detection
import pose_estimation


def estimate_pose_from_image(image: np.ndarray, detector,
                              qr_size: float, fov_x: float,
                              square_size: int, detector_type: str,
                              output_dir: str = "gimbal_pose_output",
                              override_camera_matrix: Optional[np.ndarray] = None,
                              debug: bool = False,
                              debug_dir: Optional[str] = None,
                              yolo_detection_module=None,
                              # ── 可调参数 (来自 app_config.json) ──
                              margin_qr: int = 1500,
                              dilate_kernel_size: int = 55,
                              warped_detect_count: int = 5,
                              warped_detect_min_success: int = 2,
                              ) -> Tuple[bool, Optional[dict], Optional[np.ndarray], Optional[dict]]:
    """从图片做完整的 QR 检测 + 位姿估计。

    严格遵循 gimbal_pose_estimation_ascend.GimbalPoseEstimator.estimate_pose_from_image 的每一步。

    参数:
        image:                    BGR 图像 (numpy array)
        detector:                 YOLO 检测器实例
        qr_size:                  QR 码物理尺寸 (米)
        fov_x:                    水平 FOV (度)
        square_size:              矫正图边长 (像素)
        detector_type:            角点检测器类型 ("opencv" / "pyzbar")
        output_dir:               输出目录
        override_camera_matrix:   外部指定的相机内参矩阵 (3x3)
        debug:                    是否输出调试图片
        debug_dir:                调试输出目录
        yolo_detection_module:    yolo_detection 模块引用 (避免循环导入)
        margin_qr:                透视变换边距 (像素), 默认 1500
        dilate_kernel_size:       Alpha掩码膨胀核大小, 默认 55
        warped_detect_count:      矫正图角点检测重复次数, 默认 5
        warped_detect_min_success: 最少成功次数, 默认 2

    返回:
        (success, pose_data, result_image, extra)
        extra 包含: H_mat, warped_size, alpha_mask_big, quad_points, margin_qr
    """
    if yolo_detection_module is None:
        import yolo_detection_ascend as yolo_detection_module

    print("\n" + "-" * 60)
    print("步骤2: 位姿估计（实际相机）")
    print("-" * 60)

    # ── 调试目录初始化 ──
    if debug:
        if debug_dir is None:
            timestamp = int(_time.time())
            debug_dir = f"debug_output_{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"\n  调试模式已启用，中间图片将保存到: {debug_dir}")

    # ================================================================
    # 2.0 读取图像
    # ================================================================
    H, W = image.shape[:2]
    print(f"  图像尺寸: {W}x{H}")

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "01_original_image.jpg"), image)
        print(f"  调试: 01_original_image.jpg 已保存")

    # ================================================================
    # 2.1 YOLOv8 检测二维码
    # ================================================================
    print(f"\n  2.1 YOLOv8检测二维码... (detector={type(detector).__name__}, size={image.shape})")
    # 确保 contiguous (RTSP 帧可能 stride 不对齐)
    image_contig = np.ascontiguousarray(image)
    detections = yolo_detection_module.detect_qr_yolov8(image_contig, detector)
    print(f"  检测到 {len(detections)} 个二维码")

    if len(detections) == 0:
        print(f"  ✗ 未检测到二维码")
        return False, None, None, None

    detection = detections[0]
    polygon_xy = detection["polygon_xy"].astype(np.float32)
    confidence = detection["confidence"]
    print(f"  使用检测结果 1 (置信度: {confidence:.2f})")

    if debug:
        image_init_detect = qr_detection.draw_qr_boundary(
            image.copy(),
            qr_detection.order_points_clockwise2(
                qr_detection.approximate_quadrilateral(polygon_xy)))
        cv2.imwrite(os.path.join(debug_dir, "02_yolo_initial_detect.jpg"), image_init_detect)
        print(f"  调试: 02_yolo_initial_detect.jpg 已保存")

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
        return False, None, None, None

    # ================================================================
    # 2.2b Alpha 掩码生成 (YOLO框 → 二值 → 膨胀)
    # ================================================================
    x_coords = quad_points[:, 0]
    y_coords = quad_points[:, 1]
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    max_side = max(x_max - x_min, y_max - y_min)

    print(f"  YOLO框中心: ({center_x}, {center_y})")
    print(f"  YOLO框最大边长: {max_side} pixels")

    # 创建二值图像
    alpha_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = quad_points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(alpha_mask, [pts], 255)

    if debug:
        alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_alpha.jpg")
    else:
        alpha_filepath = os.path.join(output_dir, "02_yolo_initial_detect_alpha.jpg")
    cv2.imwrite(alpha_filepath, alpha_mask)
    print(f"  调试: 二值图像已保存到 {alpha_filepath}")

    # 膨胀操作 (扩大掩码区域)
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    alpha_mask_big = cv2.dilate(alpha_mask, kernel, iterations=1)

    if debug:
        big_alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_big_alpha.jpg")
    else:
        big_alpha_filepath = os.path.join(output_dir, "02_yolo_initial_detect_big_alpha.jpg")
    cv2.imwrite(big_alpha_filepath, alpha_mask_big)
    print(f"  调试: 膨胀后二值图像已保存到 {big_alpha_filepath}")

    # ================================================================
    # 2.3 透视变换矫正 + Alpha 合成
    # ================================================================
    print(f"\n  2.3 透视变换矫正...")

    shift_qr = margin_qr / 2.0
    dst_square = np.array([
        [shift_qr, shift_qr],
        [square_size - 1 + shift_qr, shift_qr],
        [square_size - 1 + shift_qr, square_size - 1 + shift_qr],
        [shift_qr, square_size - 1 + shift_qr]
    ], dtype=np.float32)

    H_mat = qr_detection.compute_homography(quad_points, dst_square)
    if H_mat is None:
        print(f"  ✗ 单应矩阵计算失败")
        return False, None, None, None

    warped_size = (square_size + margin_qr, square_size + margin_qr)

    # 原图 warped
    warped = qr_detection.warp_perspective_with_homography(image, H_mat, warped_size)
    # Alpha 掩码 warped
    warped_alpha = qr_detection.warp_perspective_with_homography(alpha_mask_big, H_mat, warped_size)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "03_warped_alpha_image.jpg"), warped_alpha)
        cv2.imwrite(os.path.join(debug_dir, "03_warped_image.jpg"), warped)

    print(f"  ✓ 透视变换完成")
    print(f"  透视变换尺寸: {warped_size[0]}x{warped_size[1]}")

    # Alpha 合成: alpha区域保留原图，其余填白
    warped_alpha_3ch = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR)
    warped_alpha_f = warped_alpha_3ch / 255.0
    warped_f = warped / 255.0
    warped_comp = (warped_alpha_f * warped_f * 255.0).astype(np.uint8)
    border_image = ((1.0 - warped_alpha_f) * 255.0).astype(np.uint8)
    warped_comp = cv2.add(warped_comp, border_image)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "03_warped_image_comp.jpg"), warped_comp)

    # ================================================================
    # 2.4 在矫正图像上检测二维码角点 (多次检测 + 中值滤波抗噪)
    # ================================================================
    print(f"\n  2.4 检测二维码角点 (重复{warped_detect_count}次, 中值滤波)...")
    all_warped = []
    decoded_info = ""
    for _k in range(warped_detect_count):
        ok, pts_w, info = qr_detection.detect_qr_in_warped_image(
            warped_comp, detector_type
        )
        if ok and pts_w is not None:
            all_warped.append(pts_w)
            if not decoded_info and info:
                decoded_info = info
    success_detect = len(all_warped) >= warped_detect_min_success

    if success_detect:
        stacked_w = np.array(all_warped)  # (M, 4, 2)
        points_warped = np.median(stacked_w, axis=0).astype(np.float32)
        mad_w = np.median(np.abs(stacked_w - points_warped), axis=0) if len(all_warped) > 1 else np.zeros_like(points_warped)
        print(f"  ✓ 检测成功 ({len(all_warped)}/{warped_detect_count}次, 中值滤波)")
        print(f"    解码: {decoded_info}")
        for j in range(4):
            print(f"    角点{j+1}: ({points_warped[j,0]:.1f}, {points_warped[j,1]:.1f})  MAD=({mad_w[j,0]:.1f}, {mad_w[j,1]:.1f})px")

        H_inv = np.linalg.inv(H_mat)
        points_corrected = qr_detection.transform_points_back(points_warped, H_inv)
        points_corrected = points_corrected[:, :2]
        points_corrected = qr_detection.order_points_clockwise2(points_corrected)
        print(f"  ✓ 角点已映射回原始图像")

        if debug:
            image_warped_detect = qr_detection.draw_qr_boundary(warped_comp.copy(), points_warped)
            cv2.imwrite(os.path.join(debug_dir, "04_warped_detect.jpg"), image_warped_detect)
    else:
        print(f"  ⚠ 矫正图像未检测到二维码，使用YOLO检测的角点")
        points_corrected = quad_points

    # ================================================================
    # 2.5 位姿估计 (PnP)
    # ================================================================
    print(f"\n  2.5 位姿估计...")
    print(f"  二维码实际尺寸: {qr_size} m")

    object_points = pose_estimation.define_qr_object_points(qr_size=qr_size)

    # 相机内参: 优先外部指定 → fov_x 构造
    if override_camera_matrix is not None:
        camera_matrix = override_camera_matrix
        print(f"  使用外部指定的内参矩阵")
    else:
        camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=fov_x)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    print(f"  相机内参矩阵:\n{camera_matrix}")

    # 调试时保存内参
    if debug and debug_dir is not None:
        cam_txt_path = os.path.join(debug_dir, "camera_intrinsics.txt")
        with open(cam_txt_path, "w") as f:
            f.write("# 相机内参矩阵 camera_matrix (3x3)\n")
            f.write("# 格式: fx,  0, cx\n#        0, fy, cy\n#        0,  0,  1\n\n")
            f.write(f"image_width = {W}\nimage_height = {H}\n")
            f.write(f"fov_x = {fov_x}\nqr_size = {qr_size:.4f}\n\n")
            f.write("camera_matrix = [\n")
            for row in camera_matrix:
                f.write(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],\n")
            f.write("]\n\ndist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]\n")
        print(f"  ✓ 相机内参已保存到: {cam_txt_path}")

    success_pose, rvec, tvec, rmat = pose_estimation.estimate_pose(
        object_points, points_corrected, camera_matrix, dist_coeffs
    )

    if not success_pose:
        print(f"  ✗ 位姿估计失败")
        return False, None, None, None

    print(f"  ✓ 位姿估计成功")

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
        "qr_size": qr_size,
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

    if debug and points_corrected is not None:
        cv2.imwrite(os.path.join(debug_dir, "05_result_image.jpg"), result_image)
        print(f"  调试: 中间图片已保存到 {debug_dir}")

    # 额外返回中间参数给多帧复用
    extra = {
        "H_mat": H_mat,
        "warped_size": warped_size,
        "alpha_mask_big": alpha_mask_big,
        "quad_points": quad_points,
        "margin_qr": margin_qr,
    }

    return True, pose_data, result_image, extra


def fast_detect_points(image: np.ndarray, H_mat: np.ndarray,
                        warped_size: tuple, alpha_mask_big: np.ndarray,
                        quad_points: np.ndarray,
                        detector_type: str = "opencv") -> Optional[np.ndarray]:
    """轻量角点检测：跳过 YOLO，复用第一帧的 H_mat 做透视变换 + 角点检测。

    用于多帧检测中第2~N帧的快速路径。

    参数:
        image:            原始图像
        H_mat:            第1帧计算的单应矩阵
        warped_size:      透视变换后的尺寸 (W, H)
        alpha_mask_big:   膨胀后的Alpha掩码
        quad_points:      回退用的四边形角点 (YOLO检测结果)
        detector_type:    角点检测器类型

    返回:
        points_corrected (4x2) 或 None (失败时返回 quad_points.copy())
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
        warped_comp, detector_type
    )
    if success:
        H_inv = np.linalg.inv(H_mat)
        pts = qr_detection.transform_points_back(points_warped, H_inv)
        pts = pts[:, :2]
        pts = qr_detection.order_points_clockwise2(pts)
        return pts
    else:
        # 回退: 使用 YOLO 的 quad_points
        return quad_points.copy()


def print_pose_info(pose_data: dict, gimbal=None):
    """打印位姿信息 (与 GimbalPoseEstimator 一致)。

    参数:
        pose_data: 位姿数据字典 (来自 estimate_pose_from_image)
        gimbal:    云台控制器 (可选), 用于读取当前角度
    """
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

    # ── 云台坐标转换 (算法角度 → 云台角度: yaw←pitch, pitch←roll-180) ──
    yam_camera_input = pose_data["pitch"]
    pitch_camera_input = pose_data["roll"] - 180.0
    if abs(pitch_camera_input) >= 180:
        pitch_camera_input = (360.0 + pitch_camera_input) % 360.0

    print(f"\n【变量角度】")
    print(f"  add  Yaw:   {yam_camera_input:.2f}°")
    print(f"  add Pitch: {pitch_camera_input:.2f}°")

    if gimbal is not None:
        try:
            gimbal_yaw, gimbal_pitch, gimbal_roll = gimbal.get_current_angles()
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


def compute_gimbal_target(dx: float, dy: float, fov_x_actual: float,
                           img_w: float, img_h: float,
                           gimbal=None,
                           angle_dead_zone_deg: float = 0.1) -> Tuple[float, float, float, float]:
    """根据 QR 像素偏移计算云台目标角度。

    参数:
        dx, dy:              QR 中心相对于图像中心的像素偏移
        fov_x_actual:        实际 FOV (度)
        img_w, img_h:        图像尺寸 (像素)
        gimbal:              云台控制器 (可选), 用于读取当前角度
        angle_dead_zone_deg: 角度死区 (度), 小于此值不调整

    返回:
        (dyaw, dpitch, target_yaw, target_pitch) 单位: 度
    """
    # 焦距 (像素): W / (2 * tan(FOV_x / 2))
    f_eff = img_w / (2.0 * math.tan(math.radians(fov_x_actual) / 2.0))

    # 角度偏移: QR偏右 → Yaw+, QR偏上 → Pitch-
    dyaw = math.degrees(math.atan(dx / f_eff))
    dpitch = -math.degrees(math.atan(dy / f_eff))

    # 死区
    if abs(dyaw) < angle_dead_zone_deg:
        dyaw = 0.0
    if abs(dpitch) < angle_dead_zone_deg:
        dpitch = 0.0

    print(f"\n  QR像素偏移: ΔX={int(dx):+d}px ΔY={int(dy):+d}px | FOV={fov_x_actual:.1f}° → ΔYaw={dyaw:+.2f}° ΔPitch={dpitch:+.2f}°")

    # 目标角度 = 当前角度 + 偏移
    if gimbal is not None:
        try:
            gim_yaw, gim_pitch, _ = gimbal.get_current_angles()
            target_yaw = gim_yaw + dyaw
            target_pitch = gim_pitch + dpitch
        except Exception:
            target_yaw = dyaw
            target_pitch = dpitch
    else:
        target_yaw = dyaw
        target_pitch = dpitch

    print(f"  目标角度: Yaw={target_yaw:.2f}°, Pitch={target_pitch:.2f}°")
    return dyaw, dpitch, target_yaw, target_pitch


def judge_lateral_position(pose_data: dict, lat_safe: float = 0.2, lat_dead: float = 0.02) -> dict:
    """判断相机在 QR 的左右/前后位置，给出机器人移动建议。

    参数:
        pose_data: 位姿数据 (含 camera_position)
        lat_safe:  水平安全区 (米), 默认 0.2, 超出才移动
        lat_dead:  水平死区 (米), 默认 0.02

    返回:
        {"dog_move": {"x": 0, "y": 0, "need_move": False}, "info": "..."}
        dog_move.x: 前进距离 (正=前)
        dog_move.y: 侧移距离 (正=左, SDK约定)
    """
    cam_pos = pose_data.get("camera_position", np.zeros(3))
    cam_x = cam_pos[0]   # QR 坐标系 X: 右+
    cam_z = cam_pos[2]   # QR 坐标系 Z: 板外(相机方向)
    distance = np.linalg.norm(cam_pos)

    print(f"\n  {'─'*40}")
    print(f"  【机器狗导航建议】")
    print(f"  {'─'*40}")
    print(f"  相机位置 (QR坐标系): X={cam_x:+.3f}m  Y={cam_pos[1]:+.3f}m  Z={cam_z:+.3f}m")
    print(f"  相机到QR距离: {distance:.3f}m")

    # 左右偏移 (lat_safe 以内视为安全, 不需要移动)
    lat_safe = max(lat_dead, lat_safe)
    dog_y = 0.0  # 侧移: 正=左移(SDK)
    if cam_x > lat_safe:
        dog_y = -(cam_x - lat_safe)  # QR右+ → 狗左移(SDK y+)
        print(f"  ▶ 水平偏移: 偏右 {cam_x:+.3f}m → 机器狗左移 {abs(dog_y):.3f}m")
    elif cam_x < -lat_safe:
        dog_y = (-cam_x - lat_safe)  # QR左+ → 狗右移(SDK y-)
        print(f"  ▶ 水平偏移: 偏左 {cam_x:+.3f}m → 机器狗右移 {abs(dog_y):.3f}m")
    else:
        print(f"  ✓ 水平方向已对齐 (偏差 {cam_x:+.3f}m, 安全区 ±{lat_safe}m)")

    # 前后距离
    STOP_DIST = 0.8
    dog_x = 0.0  # 前进: 正=前(SDK)
    if cam_z > STOP_DIST:
        dog_x = cam_z - STOP_DIST
        print(f"  ▶ 前后距离: {cam_z:.3f}m > {STOP_DIST}m → 机器狗前进 {dog_x:.3f}m")
    elif cam_z < -STOP_DIST:
        dog_x = cam_z + STOP_DIST  # 负值=后退
        print(f"  ▶ 前后距离: {cam_z:.3f}m < 0 → 机器狗后退 {-dog_x:.3f}m")
    else:
        print(f"  ✓ 前后距离: {cam_z:.3f}m (安全范围)")

    # 综合指令
    cmd_parts = []
    if abs(dog_y) > 0.01:
        cmd_parts.append(f"y={dog_y:+.3f}m")
    if abs(dog_x) > 0.01:
        cmd_parts.append(f"x={dog_x:+.3f}m")

    if cmd_parts:
        print(f"  → [Dog.move] " + ", ".join(cmd_parts))
    else:
        print(f"  → [Dog.move] 无需移动，已在目标位置")
    print(f"  {'─'*40}")

    return {
        "need_move": abs(dog_x) > 0.01 or abs(dog_y) > 0.01,
        "x": dog_x,
        "y": dog_y,
        "distance": distance,
        "cam_z": cam_z,
    }


def build_aligned_pose(pose_data: dict, dyaw: float, dpitch: float) -> dict:
    """在原始 pose_data 上叠加中心对齐角度，返回修正后的位姿数据。

    算法角度 → 云台角度: yaw←pitch, pitch←roll-180

    参数:
        pose_data: 原始 PnP 位姿数据
        dyaw:      中心对齐 yaw 补偿 (度)
        dpitch:    中心对齐 pitch 补偿 (度)

    返回:
        aligned_pose: 修正后的位姿数据 (rmat 已重建)
    """
    aligned_pose = pose_data.copy()
    aligned_yaw = pose_data['pitch'] - dyaw           # 算法 pitch - 中心补偿
    aligned_pitch = pose_data['roll'] - 180.0 - dpitch  # 算法 roll-180 - 中心补偿
    if abs(aligned_pitch) >= 180:
        aligned_pitch = (360.0 + aligned_pitch) % 360.0

    # 逆向映射重建旋转矩阵: 云台 yaw→算法 pitch, 云台 pitch→算法 roll
    aligned_algorithm_pitch = aligned_yaw
    aligned_algorithm_roll = aligned_pitch + 180.0
    aligned_algorithm_yaw = pose_data['yaw']

    yaw_rad = math.radians(aligned_algorithm_yaw)
    pitch_rad = math.radians(aligned_algorithm_pitch)
    roll_rad = math.radians(aligned_algorithm_roll)
    Rz = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                   [math.sin(yaw_rad), math.cos(yaw_rad), 0],
                   [0, 0, 1]])
    Ry = np.array([[math.cos(pitch_rad), 0, math.sin(pitch_rad)],
                   [0, 1, 0],
                   [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll_rad), -math.sin(roll_rad)],
                   [0, math.sin(roll_rad), math.cos(roll_rad)]])
    aligned_rmat = Rz @ Ry @ Rx

    aligned_pose['yaw'] = aligned_yaw
    aligned_pose['pitch'] = aligned_pitch
    aligned_pose['roll'] = aligned_algorithm_roll % 360.0  # 更新 roll 为对齐后的值
    aligned_pose['rmat'] = aligned_rmat
    # 用新 rmat 重建 tvec (相机在QR坐标系中的位置)
    aligned_pose['tvec'] = (-aligned_rmat @ aligned_pose['camera_position']).flatten()

    return aligned_pose


def evaluate_imaging_quality(pose_data: dict, fov_x: float, img_w: int, _img_h: int,
                              align_thresh_deg: float = 15.0,
                              qr_fill_ratio_min: float = 0.15,
                              qr_fill_ratio_max: float = 0.40,
                              view_angle_thresh_deg: float = 15.0,
                              plane_angle_thresh_deg: float = 15.0) -> bool:
    """评估当前成像是否达到最佳状态，给出终止建议。

    四个判断条件:
      ① 正对程度: |roll - 180°| < align_thresh_deg
      ② QR 占图比例: qr_fill_ratio_min ~ qr_fill_ratio_max
      ③ 视线-平面夹角: 相机→QR位置向量 vs QR法线, < view_angle_thresh_deg
      ④ 成像平面-二维码平面夹角: 相机光轴 vs QR法线, < plane_angle_thresh_deg

    参数:
        pose_data:              位姿数据
        fov_x:                  水平 FOV (度)
        img_w:                  图像宽度 (像素)
        _img_h:                 图像高度 (像素, 未使用)
        align_thresh_deg:       正对阈值 (度), 默认 15
        qr_fill_ratio_min:      QR 最小占比, 默认 0.15
        qr_fill_ratio_max:      QR 最大占比, 默认 0.40
        view_angle_thresh_deg:  视线夹角阈值 (度), 默认 15
        plane_angle_thresh_deg: 成像平面夹角阈值 (度), 默认 15

    返回:
        True=已达到最佳成像状态, False=需继续调整
    """
    roll = pose_data.get("roll", 0.0)
    tvec = pose_data.get("tvec", np.zeros(3))
    img_pts = pose_data.get("image_points", None)

    # ── 条件 ①: 正对程度 ──
    pitch_dev = abs(roll - 180.0)
    if pitch_dev > 180:
        pitch_dev = abs(360.0 - pitch_dev) % 360.0
    is_aligned = pitch_dev < align_thresh_deg

    # ── 条件 ②: QR 占图比例 ──
    if img_pts is not None and len(img_pts) >= 4:
        side_01 = np.linalg.norm(img_pts[0] - img_pts[1])
        side_12 = np.linalg.norm(img_pts[1] - img_pts[2])
        side_23 = np.linalg.norm(img_pts[2] - img_pts[3])
        side_30 = np.linalg.norm(img_pts[3] - img_pts[0])
        qr_px_side = np.mean([side_01, side_12, side_23, side_30])
        qr_fill_ratio = qr_px_side / img_w
        is_good_size = qr_fill_ratio_min <= qr_fill_ratio <= qr_fill_ratio_max
    else:
        qr_fill_ratio = 0
        is_good_size = False

    # ── 条件 ③: 视线-平面夹角 (相机→QR位置向量 vs QR 法线) ──
    tvec_norm = np.linalg.norm(tvec)
    if tvec_norm > 1e-6:
        cos_angle = abs(tvec[2]) / tvec_norm
        view_angle = math.degrees(math.acos(min(cos_angle, 1.0)))
    else:
        view_angle = 0.0
    is_good_angle = view_angle < view_angle_thresh_deg

    # ── 条件 ④: 成像平面-二维码平面夹角 (相机光轴 vs QR 法线) ──
    rmat = pose_data.get("rmat")
    if rmat is not None:
        cam_z_axis = rmat.T[:, 2]                    # 相机光轴在世界坐标系中的方向
        qr_normal = np.array([0.0, 0.0, 1.0])       # QR 法线 (Z轴正方向)
        cos_plane = abs(np.dot(cam_z_axis, qr_normal))
        plane_angle = math.degrees(math.acos(min(cos_plane, 1.0)))
    else:
        plane_angle = 0.0
    is_good_plane = plane_angle < plane_angle_thresh_deg

    # ── 综合评估 ──
    conditions_met = sum([is_aligned, is_good_size, is_good_angle, is_good_plane])
    all_good = conditions_met == 4

    print(f"  ① 正对程度: Pitch偏差={pitch_dev:.1f}° (阈值<{align_thresh_deg}°) → {'✓ 已正对' if is_aligned else '✗ 需调整'}")
    print(f"  ② QR占图比: {qr_fill_ratio:.1%} (目标{qr_fill_ratio_min:.0%}~{qr_fill_ratio_max:.0%}) → {'✓ 合适' if is_good_size else '✗ 需变倍'}")
    print(f"  ③ 视线-平面夹角: {view_angle:.1f}° (阈值<{view_angle_thresh_deg}°) → {'✓ 理想' if is_good_angle else '✗ 偏斜'}")
    print(f"  ④ 成像平面-二维码平面夹角: {plane_angle:.1f}° (阈值<{plane_angle_thresh_deg}°) → {'✓ 正对' if is_good_plane else '✗ 偏斜'}")

    if all_good:
        print(f"\n  {'='*40}")
        print(f"  ✓✓✓ 已达到最佳成像状态！({conditions_met}/4 条件满足)")
        print(f"  {'='*40}")
    else:
        print(f"  状态: {conditions_met}/4 条件满足，继续调整")
        if not is_good_size and qr_fill_ratio < qr_fill_ratio_min:
            print(f"  建议: QR 太小，增大变倍")
        elif not is_good_size and qr_fill_ratio > qr_fill_ratio_max:
            print(f"  建议: QR 太大，减小变倍")
        if not is_good_angle:
            print(f"  建议: 移动机器人使正对 QR")

    return all_good
