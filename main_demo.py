#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维码检测与位姿估计主程序 - 修复版
只保存中间检测图片（JPG格式）
"""
import os
import sys
import argparse
import numpy as np
import cv2
from typing import Tuple, List, Optional

# 导入三个功能模块
import yolo_detection
import qr_detection
import pose_estimation

# 调试支持
import json
import time
from datetime import datetime

def save_debug_image(image, filename: str, debug_dir: str = "debug_output"):
    """保存调试图像为JPG格式
    
    参数:
        image: 要保存的图像（numpy数组）
        filename: 文件名（不含路径和扩展名）
        debug_dir: 调试输出目录
    """
    os.makedirs(debug_dir, exist_ok=True)
    # 确保文件扩展名为.jpg
    if not filename.endswith('.jpg'):
        filename = filename.rsplit('.', 1)[0] + '.jpg'
    filepath = os.path.join(debug_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"  调试: 图像已保存到 {filepath}")

def setup_debug_dir(base_dir: str = "debug_output") -> str:
    """设置调试输出目录，返回完整路径
    
    参数:
        base_dir: 基础目录名
        
    返回:
        debug_dir: 完整的调试目录路径（包含时间戳）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"{base_dir}_{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"调试模式已启用，中间图片将保存到: {debug_dir}")
    return debug_dir

def generate_default_qr_template(size: int = 512) -> np.ndarray:
    """生成默认的二维码模板图像（黑白棋盘图案）"""
    # 创建512x512的白色图像
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # 添加黑色方块模拟二维码图案
    block_size = size // 16  # 16x16网格
    for i in range(16):
        for j in range(16):
            # 在特定位置添加黑色块，模拟二维码的定位图案和编码区域
            if (i < 3 and j < 3) or (i < 3 and j >= 13) or (i >= 13 and j < 3):
                # 三个定位图案区域（左上、右上、左下）
                img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0
            elif (i + j) % 4 == 0:
                # 随机黑白图案
                img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0
    
    # 添加白色边框
    border = size // 32
    img[:border, :] = 255
    img[-border:, :] = 255
    img[:, :border] = 255
    img[:, -border:] = 255
    
    return img

def load_qr_template(image_path: str) -> np.ndarray:
    """加载二维码模板图像，确保是正方形"""
    if not os.path.exists(image_path):
        print(f"警告: 二维码图像不存在: {image_path}，生成默认模板...")
        img = generate_default_qr_template()
        cv2.imwrite(image_path, img)
        print(f"  已生成默认模板: {image_path} (512x512)")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 确保是正方形
    h, w = img.shape[:2]
    if h != w:
        size = min(h, w)
        img = img[:size, :size]
        print(f"警告: 图像非正方形，已裁剪为 {size}x{size}")
    
    return img

def upsample_qr_image(image: np.ndarray, target_size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """上采样二维码图像到目标尺寸"""
    if len(target_size) == 2:
        h, w = target_size
    else:
        h = w = target_size[0]
    
    # 使用INTER_CUBIC插值进行上采样
    upsampled = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return upsampled

def project_qr_image(qr_image: np.ndarray, camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                     output_size: Tuple[int, int], qr_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将二维码图像投影到相机成像平面
    
    注意：空白区域（背景）设为白色(255)，保持二维码本身的黑白像素不变
    """
    # 定义二维码的3D角点（世界坐标系，单位：米）
    half = qr_size / 2.0
    object_points = np.array([
        [-half, half, 0.0],   # 左上
        [half, half, 0.0],    # 右上
        [half, -half, 0.0],   # 右下
        [-half, -half, 0.0]   # 左下
    ], dtype=np.float32)
    
    # 投影3D点到2D图像平面
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2)
    
    # 创建输出图像 - 初始化为白色背景
    output = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255
    
    # 定义源图像（二维码）的四个角点
    src_pts = np.array([
        [0, 0],
        [qr_image.shape[1] - 1, 0],
        [qr_image.shape[1] - 1, qr_image.shape[0] - 1],
        [0, qr_image.shape[0] - 1]
    ], dtype=np.float32)
    
    # 计算从二维码图像到投影图像的单应矩阵
    H, _ = cv2.findHomography(src_pts, image_points, cv2.RANSAC, 5.0)
    
    # 进行透视变换 - 设置白色背景
    warped = cv2.warpPerspective(qr_image, H, (output_size[0], output_size[1]), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255, 255, 255))
    
    # 将投影区域复制到输出图像（保留二维码的原始像素值）
    # 使用非白色像素作为mask，避免复制白色背景
    mask = np.any(warped < 255, axis=2)
    output[mask] = warped[mask]
    
    return output, image_points, object_points

def process_image_for_pose(image_path: str, qr_size: float = 0.1, model_size: str = 's',
                          detector_type: str = 'opencv', square_size: int = 800,
                          conf_threshold: float = 0.2, nms_threshold: float = 0.5,
                          debug: bool = False, debug_dir: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """处理单张图像，检测二维码并返回角点坐标
    
    这是对qr_homography_correction.py中process_image函数的简化版本
    
    参数:
        image_path: 图像路径
        qr_size: 二维码物理尺寸（米）
        model_size: YOLOv8模型大小
        detector_type: 检测器类型
        square_size: 矫正后正方形图像的边长
        conf_threshold: YOLOv8置信度阈值
        nms_threshold: YOLOv8非极大值抑制阈值
        debug: 是否启用调试模式（保存中间图片）
        debug_dir: 调试数据保存目录（默认：debug_output_时间戳）
        
    返回:
        success: 成功标志
        points: 角点坐标 (4,2)
        image: 原始图像
    """
    # 初始化调试目录
    if debug:
        if debug_dir is None:
            debug_dir = setup_debug_dir()
        else:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"调试模式已启用，中间图片将保存到: {debug_dir}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        return False, None, None
    
    print(f"处理图像：{image_path}")
    print(f"图像尺寸：{image.shape[1]}x{image.shape[0]}")
    
    # 1. 保存原始图像
    if debug:
        save_debug_image(image, "01_original_image", debug_dir)
    
    # 加载YOLOv8检测器
    detector = yolo_detection.load_yolov8_detector(model_size=model_size, 
                                   conf_threshold=conf_threshold, 
                                   nms_threshold=nms_threshold)
    if detector is None:
        return False, None, None
    
    # 检测二维码
    detections = yolo_detection.detect_qr_yolov8(image, detector)
    print(f"YOLOv8检测到 {len(detections)} 个二维码")
    
    if len(detections) == 0:
        print("未检测到二维码")
        return False, None, None
    
    # 只处理第一个检测到的二维码
    detection = detections[0]
    polygon_xy = detection['polygon_xy'].astype(np.float32)
    confidence = detection['confidence']
    print(f"置信度：{confidence:.2f}")
    
    # 近似为四边形
    try:
        quad_points_ = qr_detection.approximate_quadrilateral(polygon_xy)
    except ValueError as e:
        print(f"无法近似四边形：{e}")
        return False, None, None
    
    # 对YOLOv8检测的点进行排序
    quad_points = qr_detection.order_points_clockwise2(quad_points_)
    
    # 2. 保存YOLO初始检测结果图（带边界框）
    if debug:
        image_init_detect = qr_detection.draw_qr_boundary(image.copy(), quad_points)
        save_debug_image(image_init_detect, "02_yolo_initial_detect", debug_dir)
    
    # 定义目标正方形（用于透视变换）
    margin_qr = 1500
    shift_qr = margin_qr/ 2.0

    dst_square = np.array([
        [shift_qr, shift_qr],
        [square_size - 1 + shift_qr, shift_qr],
        [square_size - 1 + shift_qr, square_size - 1 + shift_qr],
        [shift_qr, square_size - 1 + shift_qr]
    ], dtype=np.float32)
    
    # 计算单应矩阵（从原始四边形到正方形）
    H = qr_detection.compute_homography(quad_points, dst_square)
    if H is None:
        print("无法计算单应矩阵")
        return False, None, None
    
    # 进行透视变换
    warped = qr_detection.warp_perspective_with_homography(image, H, (square_size + margin_qr, square_size + margin_qr))
    
    # 3. 保存透视变换后的矫正图像
    if debug:
        save_debug_image(warped, "03_warped_image", debug_dir)
    
    # 在矫正后的图像上检测二维码
    success, points_warped, decoded_info = qr_detection.detect_qr_in_warped_image(warped, detector_type)
    
    if not success:
        print("在矫正后的图像上未能检测到二维码")
        points_corrected = quad_points
    else:
        print(f"在矫正图像上检测到二维码：{decoded_info}")
        # 计算单应矩阵的逆
        H_inv = np.linalg.inv(H)
        # 将点映射回原始图像
        points_corrected = qr_detection.transform_points_back(points_warped, H_inv)
        print(f"映射回原始图像的角点：{points_corrected}")
        
        # 4. 保存矫正图像上的检测结果（带边界框）
        if debug:
            image_warped_detect = qr_detection.draw_qr_boundary(warped.copy(), points_warped)
            save_debug_image(image_warped_detect, "04_warped_detect", debug_dir)
    
    # 5. 保存最终结果图（带正确角点顺序和坐标系）
    if debug and points_corrected is not None:
        result_image = image.copy()
        
        # 绘制二维码边界
        result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
        
        # 估计位姿以绘制坐标系
        if points_corrected is not None and len(points_corrected) == 4:
            # 定义二维码的3D角点（世界坐标系）
            object_points = pose_estimation.define_qr_object_points(qr_size=qr_size)
            
            # 创建相机内参（使用图像尺寸和默认fov）
            H, W = image.shape[:2]
            camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=60.0)
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            
            # 估计位姿
            success_pose, rvec, tvec, rmat = pose_estimation.estimate_pose(
                object_points, points_corrected, camera_matrix, dist_coeffs
            )
            
            if success_pose:
                # 1. 绘制二维码坐标系（X-红，Y-绿，Z-蓝）
                axis_length = qr_size * 0.5
                result_image = pose_estimation.draw_axes(result_image, camera_matrix, dist_coeffs, rvec, tvec, length=axis_length)
                print(f"  已绘制二维码坐标系")
                
                # 2. 在图像角落绘制相机坐标系示意图
                # 创建一个小的坐标系指示器（在图像左上角）
                indicator_size = 100
                indicator_origin = (50, 50)
                
                # 相机坐标系的轴在相机坐标系中的方向
                # 注意：这里的rvec和tvec描述的是世界坐标系到相机坐标系的变换
                # 所以我们需要显示的是：如果我们在相机位置看世界，世界坐标系的轴是什么方向
                
                # 更简单的方法：直接绘制相机坐标系（从图像中心出发，表示相机的朝向）
                # 计算相机在世界坐标系中的位置和朝向
                camera_position = -rmat.T @ tvec
                
                # 在图像上标记相机投影位置（这实际上是相机光心的投影，通常在图像中心附近）
                # 但更有用的是：绘制从相机到二维码的视线
                
                # 绘制相机坐标系示意图（在图像右上角）
                indicator_x, indicator_y = W - 150, 50
                
                # 相机坐标系：X-右(红), Y-下(绿), Z-前(蓝)
                # 在示意图中，我们绘制一个固定的坐标系
                axis_len = 40
                # X轴 - 红色
                cv2.arrowedLine(result_image, 
                               (indicator_x, indicator_y), 
                               (indicator_x + axis_len, indicator_y),
                               (0, 0, 255), 2, tipLength=0.3)
                cv2.putText(result_image, 'X', (indicator_x + axis_len + 5, indicator_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Y轴 - 绿色
                cv2.arrowedLine(result_image, 
                               (indicator_x, indicator_y), 
                               (indicator_x, indicator_y + axis_len),
                               (0, 255, 0), 2, tipLength=0.3)
                cv2.putText(result_image, 'Y', (indicator_x - 10, indicator_y + axis_len + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Z轴 - 蓝色（指向图片外，用圆圈表示）
                cv2.circle(result_image, (indicator_x, indicator_y), 5, (255, 0, 0), -1)
                cv2.putText(result_image, 'Z', (indicator_x - 15, indicator_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 添加标签
                cv2.putText(result_image, 'Camera Coords', (indicator_x - 50, indicator_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 3. 绘制从相机到二维码中心的连线（表示视线）
                # 二维码中心在图像上的位置
                qr_center_2d = np.mean(points_corrected, axis=0).astype(int)
                
                # 绘制视线（黄色虚线）
                # 从图像边缘（表示相机位置）到二维码中心
                # 相机通常在图像中心附近"看"向二维码
                image_center = (W // 2, H // 2)
                cv2.line(result_image, tuple(qr_center_2d), tuple(image_center), 
                        (0, 255, 255), 1)
                cv2.putText(result_image, 'View Direction', 
                           ((qr_center_2d[0] + image_center[0]) // 2, 
                            (qr_center_2d[1] + image_center[1]) // 2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                print(f"  已绘制相机坐标系指示器")
                print(f"  相机位置（世界坐标系）: [{camera_position[0,0]:.3f}, {camera_position[1,0]:.3f}, {camera_position[2,0]:.3f}] 米")
                print(f"  相机距离二维码中心: {np.linalg.norm(camera_position):.3f} 米")
        
        save_debug_image(result_image, "05_result_image", debug_dir)
        print(f"调试: 中间图片已保存到 {debug_dir}")
    
    return True, points_corrected, image

def run_camera_simulation(qr_image_path: str = 'qr_template.jpg', output_path: str = 'simulated_camera_output.png',
                         output_size: Tuple[int, int] = (640, 480), qr_size: float = 0.1,
                         tx: float = 0.0, ty: float = 0.0, tz: float = 0.5,
                         angle_x: float = 0.0, angle_y: float = 0.0, angle_z: float = 0.0,
                         fov_x: float = 60.0, draw_axes: bool = False, 
                         visualize_3d: bool = False, save_points: Optional[str] = None,
                         debug: bool = False, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行相机模拟流程，生成投影图像
    
    这是对qr_camera_simulation.py中main函数的简化版本
    
    参数:
        qr_image_path: 二维码模板图像路径
        output_path: 输出图像路径
        output_size: 输出图像尺寸
        qr_size: 二维码物理尺寸（米）
        tx, ty, tz: 相机在二维码坐标系中的位置
        angle_x, angle_y, angle_z: 绕X、Y、Z轴的旋转角度（度）
        fov_x: 水平视场角（度）
        draw_axes: 是否绘制坐标轴
        visualize_3d: 是否显示3D场景
        save_points: 保存投影角点坐标的文件路径
        debug: 是否启用调试模式（保存中间图片）
        debug_dir: 调试数据保存目录（默认：debug_output_时间戳）
        
    返回:
        projected_image: 投影后的图像
        image_points: 投影角点坐标
        object_points: 3D角点坐标
    """
    # 初始化调试目录
    if debug:
        if debug_dir is None:
            debug_dir = setup_debug_dir()
        else:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"调试模式已启用，中间图片将保存到: {debug_dir}")
    
    # 1. 加载二维码模板图像
    print("步骤1: 加载二维码模板图像...")
    qr_image = load_qr_template(qr_image_path)
    print(f"  已加载图像: {qr_image_path} ({qr_image.shape[1]}x{qr_image.shape[0]})")
    if debug:
        save_debug_image(qr_image, "01_original_qr_image", debug_dir)
    
    # 超分辨率处理：将二维码图像上采样到2048x2048
    print("  进行超分辨率处理（上采样到2048x2048）...")
    target_size = (2048, 2048)
    if qr_image.shape[0] != target_size[0] or qr_image.shape[1] != target_size[1]:
        qr_image = upsample_qr_image(qr_image, target_size)
        print(f"  已上采样: {qr_image.shape[1]}x{qr_image.shape[0]}")
        if debug:
            save_debug_image(qr_image, "02_upsampled_qr_image", debug_dir)
    else:
        print("  图像已经是2048x2048，无需处理")
    
    # 2. 设置相机内参
    print("步骤2: 设置相机内参...")
    camera_matrix = pose_estimation.create_camera_matrix(output_size, fov_x)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    print(f"  相机矩阵:\n{camera_matrix}")
    print(f"  畸变系数: {dist_coeffs.flatten()}")
    
    # 3. 设置相机外参
    print("步骤3: 设置相机外参...")
    rvec, tvec, R = pose_estimation.create_default_extrinsics(tx=tx, ty=ty, tz=tz, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
    print(f"  相机位置: X={tx:.3f}米, Y={ty:.3f}米, Z={tz:.3f}米")
    print(f"  输入旋转角度: X={angle_x}°, Y={angle_y}°, Z={angle_z}°")
    
    # 计算欧拉角（ZYX顺序）
    yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(R)
    print(f"  计算欧拉角（ZYX顺序）:")
    print(f"    yaw (绕Z轴): {np.degrees(yaw):.2f}°")
    print(f"    pitch (绕Y轴): {np.degrees(pitch):.2f}°")
    print(f"    roll (绕X轴): {np.degrees(roll):.2f}°")
    print(f"  平移向量: {tvec.flatten()} 米")
    
    # 4. 相机模拟输出图像
    print("步骤4: 生成模拟相机输出图像...")
    projected_image, image_points, object_points = project_qr_image(
        qr_image, camera_matrix, dist_coeffs, rvec, tvec, output_size, qr_size=qr_size
    )
    print(f"  投影图像尺寸: {output_size[0]}x{output_size[1]}")
    if debug:
        save_debug_image(projected_image, "03_projected_image", debug_dir)
    
    # 5. 可选：绘制3D坐标系
    if draw_axes:
        print("步骤5: 绘制3D坐标系...")
        projected_image = pose_estimation.draw_axes(projected_image, camera_matrix, dist_coeffs, rvec, tvec)
        if debug:
            save_debug_image(projected_image, "04_image_with_axes", debug_dir)
    
    # 6. 保存输出图像
    cv2.imwrite(output_path, projected_image)
    print(f"  输出图像已保存: {output_path}")
    
    # 7. 可选：显示3D场景
    if visualize_3d:
        print("步骤6.5: 显示3D场景...")
        pose_estimation.visualize_3d_scene(camera_matrix, dist_coeffs, rvec, tvec, object_points, qr_size)
    
    # 8. 可选：保存投影角点坐标
    if save_points:
        np.save(save_points, image_points)
        print(f"  投影角点坐标已保存: {save_points}")
    
    # 9. 调试：保存参数汇总（只打印，不保存文件）
    if debug:
        print(f"调试图片已保存到目录: {debug_dir}")
    
    return projected_image, image_points, object_points

def main_DEMO():
    """主函数：解析命令行参数并运行演示"""
    parser = argparse.ArgumentParser(description='二维码检测与相机模拟演示程序')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='选择运行模式')
    
    # 相机模拟模式
    sim_parser = subparsers.add_parser('simulate', help='运行相机模拟（生成投影图像）')
    
    # 配置文件（必须放在最前面，以便后续参数覆盖）
    sim_parser.add_argument('--config', type=str, default=None,
                           help='JSON配置文件路径（参数将从此文件加载，命令行参数优先级更高）')
    
    # 输入参数
    sim_parser.add_argument('--qr_image', type=str, default='qr_template.jpg',
                           help='二维码模板图像路径（默认: qr_template.jpg）')
    sim_parser.add_argument('--output', type=str, default='simulated_camera_output.png',
                           help='输出图像路径（默认: simulated_camera_output.png）')
    sim_parser.add_argument('--output_size', type=int, nargs=2, default=[640, 480],
                           help='输出图像尺寸 [宽度 高度]（默认: 640 480）')
    sim_parser.add_argument('--qr_size', type=float, default=0.1,
                           help='二维码物理尺寸（米）（默认: 0.1）')
    
    # 位姿参数
    sim_parser.add_argument('--tx', type=float, default=0.0,
                           help='相机X坐标（米）（默认: 0.0）')
    sim_parser.add_argument('--ty', type=float, default=0.0,
                           help='相机Y坐标（米）（默认: 0.0）')
    sim_parser.add_argument('--tz', type=float, default=0.5,
                           help='相机Z坐标（米）（默认: 0.5）')
    sim_parser.add_argument('--angle_x', type=float, default=0.0,
                           help='绕X轴旋转角度（度）（默认: 0.0）')
    sim_parser.add_argument('--angle_y', type=float, default=0.0,
                           help='绕Y轴旋转角度（度）（默认: 0.0）')
    sim_parser.add_argument('--angle_z', type=float, default=0.0,
                           help='绕Z轴旋转角度（度）（默认: 0.0）')
    sim_parser.add_argument('--fov_x', type=float, default=60.0,
                           help='水平视场角（度）（默认: 60.0）')
    
    # 功能选项
    sim_parser.add_argument('--draw_axes', action='store_true',
                           help='在输出图像上绘制3D坐标系')
    sim_parser.add_argument('--show', action='store_true',
                           help='显示输出图像')
    sim_parser.add_argument('--visualize_3d', action='store_true',
                           help='显示相机和二维码的3D场景（需要matplotlib库）')
    sim_parser.add_argument('--save_points', type=str,
                           help='保存投影角点坐标到文件（.npy格式）')
    sim_parser.add_argument('--estimate_pose', action='store_true',
                           help='估计位姿并输出旋转向量、平移向量、旋转角度')
    
    # 调试选项
    sim_parser.add_argument('--debug', action='store_true',
                           help='启用调试模式，保存中间过程图片')
    sim_parser.add_argument('--debug_dir', type=str, default=None,
                           help='调试图片保存目录（默认：debug_output_时间戳）')
    
    # 位姿估计参数
    sim_parser.add_argument('--model_size', type=str, default='s',
                           choices=['n', 's', 'm', 'l', 'x'],
                           help='YOLOv8模型大小（默认s）')
    sim_parser.add_argument('--detector', type=str, default='opencv',
                           choices=['opencv', 'pyzbar'],
                           help='矫正后图像使用的检测器（默认opencv）')
    sim_parser.add_argument('--square_size', type=int, default=800,
                           help='矫正后正方形图像的边长（像素，默认800）')
    sim_parser.add_argument('--conf_threshold', type=float, default=0.2,
                           help='YOLOv8置信度阈值（默认0.2）')
    sim_parser.add_argument('--nms_threshold', type=float, default=0.5,
                           help='YOLOv8非极大值抑制阈值（默认0.5）')
    
    # 位姿估计模式
    pose_parser = subparsers.add_parser('estimate', help='从图像估计二维码位姿')
    pose_parser.add_argument('--config', type=str, default=None,
                            help='JSON配置文件路径（参数将从此文件加载，命令行参数优先级更高）')
    pose_parser.add_argument('--image', type=str, required=True,
                            help='输入图像路径')
    pose_parser.add_argument('--qr_size', type=float, default=0.1,
                            help='二维码物理尺寸（米）（默认: 0.1）')
    pose_parser.add_argument('--fov_x', type=float, default=60.0,
                            help='水平视场角（度）（默认: 60.0）')
    pose_parser.add_argument('--model_size', type=str, default='s',
                            choices=['n', 's', 'm', 'l', 'x'],
                            help='YOLOv8模型大小（默认s）')
    pose_parser.add_argument('--detector', type=str, default='opencv',
                            choices=['opencv', 'pyzbar'],
                            help='矫正后图像使用的检测器（默认opencv）')
    pose_parser.add_argument('--square_size', type=int, default=800,
                            help='矫正后正方形图像的边长（像素，默认800）')
    pose_parser.add_argument('--conf_threshold', type=float, default=0.2,
                            help='YOLOv8置信度阈值（默认0.2）')
    pose_parser.add_argument('--nms_threshold', type=float, default=0.5,
                            help='YOLOv8非极大值抑制阈值（默认0.5）')
    pose_parser.add_argument('--output', type=str, default=None,
                            help='保存结果图像路径')
    pose_parser.add_argument('--show', action='store_true',
                            help='显示结果图像')
    pose_parser.add_argument('--debug', action='store_true',
                            help='启用调试模式，保存中间过程图片')
    pose_parser.add_argument('--debug_dir', type=str, default=None,
                            help='调试图片保存目录（默认：debug_output_时间戳）')
    
    # 解析参数
    # 先获取默认值以便后续比较
    defaults = {
        'simulate': {
            'qr_image': 'qr_template.jpg',
            'output': 'simulated_camera_output.png',
            'output_size': [640, 480],
            'qr_size': 0.1,
            'tx': 0.0,
            'ty': 0.0,
            'tz': 0.5,
            'angle_x': 0.0,
            'angle_y': 0.0,
            'angle_z': 0.0,
            'fov_x': 60.0,
            'draw_axes': False,
            'show': False,
            'visualize_3d': False,
            'save_points': None,
            'estimate_pose': False,
            'debug': False,
            'debug_dir': None,
            'model_size': 's',
            'detector': 'opencv',
            'square_size': 800,
            'conf_threshold': 0.2,
            'nms_threshold': 0.5,
            'config': None
        },
        'estimate': {
            'image': None,
            'qr_size': 0.1,
            'fov_x': 60.0,
            'model_size': 's',
            'detector': 'opencv',
            'square_size': 800,
            'conf_threshold': 0.2,
            'nms_threshold': 0.5,
            'output': None,
            'show': False,
            'debug': False,
            'debug_dir': None,
            'config': None
        }
    }
    
    args = parser.parse_args()
    
    # 默认自动加载 config.json（如果存在）
    # 优先级：命令行参数 > 指定的配置文件 > 默认 config.json > 程序默认值
    config_paths = []
    
    # 1. 首先尝试加载默认的 config.json
    default_config = 'config.json'
    if os.path.exists(default_config):
        config_paths.append(default_config)
        print(f"✓ 检测到默认配置文件: {default_config}")
    
    # 2. 如果用户通过 --config 指定了其他配置文件，也加载它（优先级更高）
    if args.config is not None:
        if os.path.exists(args.config):
            config_paths.append(args.config)
            print(f"✓ 加载指定的配置文件: {args.config}")
        else:
            print(f"警告: 指定的配置文件不存在: {args.config}")
    
    # 按顺序加载配置文件（后面的覆盖前面的）
    for config_path in config_paths:
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 根据命令类型获取对应的配置
        if args.command in config:
            cmd_config = config[args.command]
            cmd_defaults = defaults.get(args.command, {})
            
            # 将配置文件中的参数应用到 args（仅当命令行使用默认值时）
            for key, value in cmd_config.items():
                if hasattr(args, key) and key in cmd_defaults:
                    current_value = getattr(args, key)
                    default_value = cmd_defaults[key]
                    # 如果当前值等于默认值，说明用户未通过命令行指定，使用配置文件的值
                    if current_value == default_value:
                        setattr(args, key, value)
                        print(f"  ✓ {key} = {value} (从配置文件)")
                    else:
                        print(f"  - {key} = {current_value} (保留命令行参数)")
                else:
                    if key != 'config':  # 忽略 config 本身
                        print(f"  警告: 配置文件中的参数 '{key}' 未被识别，已忽略")
        else:
            print(f"  警告: 配置文件中未找到命令 '{args.command}' 的配置")
    
    if not config_paths:
        print("提示: 未找到配置文件，使用程序默认值")
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'simulate':
        # 运行相机模拟流程
        output_size = tuple(args.output_size)
        projected_image, image_points, object_points = run_camera_simulation(
            qr_image_path=args.qr_image,
            output_path=args.output,
            output_size=output_size,
            qr_size=args.qr_size,
            tx=args.tx,
            ty=args.ty,
            tz=args.tz,
            angle_x=args.angle_x,
            angle_y=args.angle_y,
            angle_z=args.angle_z,
            fov_x=args.fov_x,
            draw_axes=args.draw_axes,
            visualize_3d=args.visualize_3d,
            save_points=args.save_points,
            debug=args.debug,
            debug_dir=args.debug_dir
        )
        
        # 估计位姿
        if args.estimate_pose:
            print("\n" + "="*60)
            print("估计二维码位姿")
            print("="*60 + "\n")
            
            # 处理投影图像以估计位姿
            success, detected_points, _ = process_image_for_pose(
                image_path=args.output,
                qr_size=args.qr_size,
                model_size=args.model_size,
                detector_type=args.detector,
                square_size=args.square_size,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                debug=args.debug,
                debug_dir=args.debug_dir
            )
            
            if success and detected_points is not None:
                # 定义二维码的3D角点（世界坐标系）
                object_points = pose_estimation.define_qr_object_points(qr_size=args.qr_size)
                
                # 估计位姿
                success_pose, rvec, tvec, _ = pose_estimation.estimate_pose(
                    object_points=object_points,
                    image_points=detected_points,
                    camera_matrix=pose_estimation.create_camera_matrix(output_size, args.fov_x),
                    dist_coeffs=np.zeros((5, 1), dtype=np.float32)
                )
                
                if success_pose:
                    # 输出位姿信息
                    print("\n估计的位姿：")
                    print(f"  旋转向量 (rvec): {rvec.flatten()}")
                    print(f"  平移向量 (tvec): {tvec.flatten()}")
                    
                    # 计算欧拉角
                    R = cv2.Rodrigues(rvec)[0]
                    yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(R)
                    print(f"  欧拉角（ZYX顺序）:")
                    print(f"    yaw (绕Z轴): {np.degrees(yaw):.2f}°")
                    print(f"    pitch (绕Y轴): {np.degrees(pitch):.2f}°")
                    print(f"    roll (绕X轴): {np.degrees(roll):.2f}°")
                    
                    # 在图像上绘制坐标轴
                    if args.draw_axes:
                        result_image = projected_image.copy()
                        result_image = pose_estimation.draw_axes(result_image, 
                                                              pose_estimation.create_camera_matrix(output_size, args.fov_x),
                                                              np.zeros((5, 1), dtype=np.float32),
                                                              rvec, tvec)
                        cv2.imwrite(args.output.replace('.png', '_with_axes.png'), result_image)
                        print(f"  带坐标轴的结果已保存: {args.output.replace('.png', '_with_axes.png')}")
                else:
                    print("位姿估计失败")
            else:
                print("未检测到二维码角点或角点坐标无效")
        
        # 显示结果
        if args.show:
            print("显示结果图像...")
            cv2.imshow('Projected Image', projected_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.command == 'estimate':
        # 估计位姿模式
        if not os.path.exists(args.image):
            print(f"错误: 图像文件不存在: {args.image}")
            return
        
        # 处理图像以估计位姿
        success, detected_points, image = process_image_for_pose(
            image_path=args.image,
            qr_size=args.qr_size,
            model_size=args.model_size,
            detector_type=args.detector,
            square_size=args.square_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            debug=args.debug,
            debug_dir=args.debug_dir
        )
        
        if success and detected_points is not None:
            # 估计位姿
            output_size = (image.shape[1], image.shape[0])
            
            # 定义二维码的3D角点（世界坐标系）
            object_points = pose_estimation.define_qr_object_points(qr_size=args.qr_size)
            
            success_pose, rvec, tvec, _ = pose_estimation.estimate_pose(
                object_points=object_points,
                image_points=detected_points,
                camera_matrix=pose_estimation.create_camera_matrix(output_size, args.fov_x),
                dist_coeffs=np.zeros((5, 1), dtype=np.float32)
            )
            
            if success_pose:
                # 输出位姿信息
                print("\n估计的位姿：")
                print(f"  旋转向量 (rvec): {rvec.flatten()}")
                print(f"  平移向量 (tvec): {tvec.flatten()}")
                
                # 计算欧拉角
                R = cv2.Rodrigues(rvec)[0]
                yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(R)
                print(f"  欧拉角（ZYX顺序）:")
                print(f"    yaw (绕Z轴): {np.degrees(yaw):.2f}°")
                print(f"    pitch (绕Y轴): {np.degrees(pitch):.2f}°")
                print(f"    roll (绕X轴): {np.degrees(roll):.2f}°")
                
                # 在图像上绘制坐标轴
                result_image = image.copy()
                result_image = pose_estimation.draw_axes(result_image, 
                                                      pose_estimation.create_camera_matrix(output_size, args.fov_x),
                                                      np.zeros((5, 1), dtype=np.float32),
                                                      rvec, tvec)
                
                # 保存或显示结果
                if args.output:
                    cv2.imwrite(args.output, result_image)
                    print(f"结果已保存: {args.output}")
                
                if args.show:
                    print("显示结果图像...")
                    cv2.imshow('Pose Estimation Result', result_image)
                    print("按任意键关闭窗口...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("位姿估计失败")
        else:
            print("未检测到二维码角点或角点坐标无效")

if __name__ == "__main__":
    main_DEMO()
