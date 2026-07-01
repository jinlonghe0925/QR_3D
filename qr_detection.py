#!/usr/bin/env python3
"""
二维码检测模块
包含OpenCV二维码检测、角点处理、单应性计算等功能
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List

# 检查pyzbar是否可用
try:
    import pyzbar.pyzbar as pyzbar_lib
    HAS_PYZBAR = True
except (ImportError, OSError) as e:
    HAS_PYZBAR = False
    import warnings
    warnings.warn(f"pyzbar不可用: {e}. 将使用OpenCV检测器。")

def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """将四个点按顺时针顺序排序（左上、右上、右下、左下）
    
    输入：形状为(4,2)的numpy数组
    返回：排序后的点
    """
    # 计算中心点
    center = np.mean(points, axis=0)
    
    # 计算每个点相对于中心的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    # 按角度排序（从左上开始，顺时针）
    sorted_indices = np.argsort(angles)
    # 调整顺序，使左上角第一个（角度在-135度到-45度之间？）
    # 简单处理：直接按角度排序后，可能需要重新排列
    sorted_points = points[sorted_indices]
    
    # 另一种方法：按y坐标排序，然后按x坐标排序
    # 但这里采用更稳健的方法：找到x+y最小的点作为左上角
    # 计算每个点的x+y和x-y
    sum_xy = points[:, 0] + points[:, 1]
    diff_xy = points[:, 0] - points[:, 1]
    
    # 左上角：x+y最小
    top_left_idx = np.argmin(sum_xy)
    
    # 右下角：x+y最大
    bottom_right_idx = np.argmax(sum_xy)
    # 右上角：x-y最大
    top_right_idx = np.argmax(diff_xy)
    # 左下角：x-y最小
    bottom_left_idx = np.argmin(diff_xy)
    
    # 构建顺序：左上、右上、右下、左下
    ordered = np.array([
        points[top_left_idx],
        points[top_right_idx],
        points[bottom_right_idx],
        points[bottom_left_idx]
    ])
    
    return ordered

def order_points_clockwise2(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """使用象限符号法将四个点按顺时针顺序排序（左上、右上、右下、左下）
    
    算法：计算中心点，根据每个点相对于中心点的坐标符号确定象限：
      (-,-) -> 左上 (0)
      (+,-) -> 右上 (1)
      (+,+) -> 右下 (2)
      (-,+) -> 左下 (3)
      
    输入：形状为(4,2)的numpy数组
    返回：排序后的点
    """
    if len(points) != 4:
        raise ValueError("输入必须是4个点")
    
    # 计算中心点
    center = np.mean(points, axis=0)
    
    # 计算每个点相对于中心点的偏移
    offsets = points - center
    
    # 为每个点分配象限索引
    quadrant_indices = []
    for i, (dx, dy) in enumerate(offsets):
        if dx < -eps and dy < -eps:
            quadrant = 0  # 左上
        elif dx >= eps and dy < -eps:
            quadrant = 1  # 右上
        elif dx >= eps and dy >= eps:
            quadrant = 2  # 右下
        elif dx < -eps and dy >= eps:
            quadrant = 3  # 左下
        else:
            # 如果点在轴上（接近0），根据角度决定
            angle = np.arctan2(dy, dx)
            # 将角度映射到象限
            if angle >= -np.pi and angle < -np.pi/2:
                quadrant = 0
            elif angle >= -np.pi/2 and angle < 0:
                quadrant = 1
            elif angle >= 0 and angle < np.pi/2:
                quadrant = 2
            else:
                quadrant = 3
    
        quadrant_indices.append((quadrant, i))
    
    # 按象限排序（0,1,2,3）
    quadrant_indices.sort(key=lambda x: x[0])
    
    # 提取排序后的点
    ordered = np.array([points[idx] for _, idx in quadrant_indices])
    
    return ordered

def approximate_quadrilateral(points: np.ndarray) -> np.ndarray:
    """将点集近似为四边形（4个点）
    
    输入：形状为(n,2)的numpy数组，n>=4
    返回：形状为(4,2)的四边形角点
    """
    if len(points) == 4:
        return points
    elif len(points) > 4:
        # 计算凸包
        hull = cv2.convexHull(points)
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2)
        else:
            # 如果还不是4个点，取凸包的四个极值点
            # 按x+y排序取四个点
            hull_points = hull.reshape(-1, 2)
            if len(hull_points) >= 4:
                # 取x+y最小、x-y最大、x+y最大、x-y最小的四个点
                sum_xy = hull_points[:, 0] + hull_points[:, 1]
                diff_xy = hull_points[:, 0] - hull_points[:, 1]
                
                indices = [
                    np.argmin(sum_xy),  # 左上
                    np.argmax(diff_xy),  # 右上
                    np.argmax(sum_xy),  # 右下
                    np.argmin(diff_xy)   # 左下
                ]
                return hull_points[indices]
            else:
                # 点太少，直接取前四个点
                return hull_points[:4]
    else:
        # 点不足4个，无法构成四边形
        raise ValueError(f"点数不足4个：{len(points)}")

def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """计算从src_points到dst_points的单应矩阵
    
    输入：形状为(4,2)的源点和目标点
    返回：3x3单应矩阵
    """
    # 确保点是浮点型
    src_points = src_points.astype(np.float32)
    dst_points = dst_points.astype(np.float32)
    
    # 计算单应矩阵
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

def warp_perspective_with_homography(image: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """使用单应矩阵进行透视变换
    
    参数:
        image: 输入图像
        H: 单应矩阵 (3x3)
        output_size: 输出图像尺寸 (宽, 高)
        
    返回:
        warped: 透视变换后的图像
    """
    warped = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped

def detect_qr_in_warped_image(warped_image: np.ndarray, detector_type: str = 'opencv') -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """在矫正后的图像上检测二维码角点
    
    参数:
        warped_image: 矫正后的图像
        detector_type: 'opencv' 或 'pyzbar'
        
    返回:
        success: 成功标志
        points: 角点坐标（形状(4,2)）
        decoded_info: 解码信息
    """
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    # 直接二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if detector_type == 'opencv':
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(binary)
        
        if retval and points is not None:
            # 只取第一个检测到的二维码
            points = points[0].astype(np.float32)
            return True, points, decoded_info[0] if decoded_info else ""
        else:
            return False, None, None
    
    elif detector_type == 'pyzbar':
        if not HAS_PYZBAR:
            print("pyzbar不可用，自动回退到OpenCV检测器")
            # 回退到opencv检测器
            detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(binary)
            
            if retval and points is not None:
                # 只取第一个检测到的二维码
                points = points[0].astype(np.float32)
                return True, points, decoded_info[0] if decoded_info else ""
            else:
                return False, None, None
        
        try:
            decoded_objects = pyzbar_lib.decode(binary)
            if decoded_objects:
                obj = decoded_objects[0]
                points = np.array(obj.polygon, dtype=np.float32)
                
                # 确保是4个点
                if len(points) != 4:
                    # 近似为四边形
                    hull = cv2.convexHull(points)
                    epsilon = 0.02 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    
                    if len(approx) == 4:
                        points = approx.reshape(4, 2).astype(np.float32)
                    else:
                        # 取凸包的四个极值点
                        hull_points = hull.reshape(-1, 2)
                        if len(hull_points) >= 4:
                            sum_xy = hull_points[:, 0] + hull_points[:, 1]
                            diff_xy = hull_points[:, 0] - hull_points[:, 1]
                            indices = [
                                np.argmin(sum_xy),
                                np.argmax(diff_xy),
                                np.argmax(sum_xy),
                                np.argmin(diff_xy)
                            ]
                            points = hull_points[indices]
                        else:
                            return False, None, None
                
                # 排序点：左上、右上、右下、左下
                #points = order_points_clockwise(points)
                return True, points, obj.data.decode('utf-8')
            else:
                return False, None, None
        except Exception as e:
            print(f"警告: pyzbar解码失败: {e}")
            print("将自动回退到opencv检测器")
            # 回退到opencv检测器
            detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(binary)
            
            if retval and points is not None:
                # 只取第一个检测到的二维码
                points = points[0].astype(np.float32)
                return True, points, decoded_info[0] if decoded_info else ""
            else:
                return False, None, None
    
    else:
        raise ValueError(f"不支持的检测器类型：{detector_type}")

def transform_points_back(points_warped: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """使用单应矩阵的逆变换将点映射回原始图像空间
    
    输入：形状为(n,2)的点（矫正后图像中的坐标）
    返回：形状为(n,2)的点（原始图像中的坐标）
    """
    # 转换为齐次坐标
    points_homogeneous = np.hstack([points_warped, np.ones((len(points_warped), 1))])
    
    # 应用逆变换
    points_original_homogeneous = points_homogeneous @ H_inv.T
    
    # 转换回笛卡尔坐标
    points_original = points_original_homogeneous[:, :2] / points_original_homogeneous[:, 2:]
    
    return points_original

def draw_qr_boundary(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """在图像上绘制二维码边界，并在四个顶点标上顺序1、2、3、4
    
    参数:
        image: 输入图像
        points: 角点坐标 (4,2)
        
    返回:
        image_with_boundary: 绘制了边界和角点编号的图像
    """
    if points is None:
        return image
    
    points = points.astype(int)
    # 绘制四边形边界
    cv2.polylines(image, [points], True, (0, 255, 0), 2)
    # 标记角点，顺序为1、2、3、4
    for i, pt in enumerate(points):
        cv2.circle(image, tuple(pt), 6, (0, 0, 255), -1)  # 红色圆点
        # 显示顶点编号（1-4），位置稍向下偏移
        text_pos = (pt[0] - 10, pt[1] + 20)  # 调整文字位置
        cv2.putText(image, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    5.8, (255, 200, 0), 12)  # 青色文字，更大更粗
    return image

def draw_results(image: np.ndarray, points_original: np.ndarray, points_yolo: Optional[np.ndarray] = None, 
                 warped_image: Optional[np.ndarray] = None) -> np.ndarray:
    """在原始图像上绘制结果
    
    参数:
        image: 原始图像
        points_original: 矫正后的角点坐标
        points_yolo: YOLOv8检测的角点坐标（可选）
        warped_image: 矫正后的图像（可选）
        
    返回:
        result_image: 绘制了结果的图像
    """
    result = image.copy()
    
    # 绘制矫正后的角点（顺序正确）
    if points_original is not None:
        points_original_int = points_original.astype(np.int32)
        cv2.polylines(result, [points_original_int], True, (0, 255, 0), 2)  # 绿色：矫正后
        # 标记矫正后角点（顺序1-4）
        for i, pt in enumerate(points_original_int):
            cv2.circle(result, tuple(pt), 6, (0, 255, 0), -1)
            text_pos = (pt[0] - 10, pt[1] + 20)
            cv2.putText(result, str(i+1), text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    
    # 如果提供了矫正后的图像，在旁边显示
    if warped_image is not None:
        # 调整大小以适应显示
        h, w = result.shape[:2]
        warped_resized = cv2.resize(warped_image, (w // 4, h // 4))
        # 放置在右上角
        result[10:10 + warped_resized.shape[0], w - warped_resized.shape[1] - 10:w - 10] = warped_resized
        # 绘制边框
        cv2.rectangle(result, 
                     (w - warped_resized.shape[1] - 12, 8),
                     (w - 8, 8 + warped_resized.shape[0] + 2),
                     (255, 255, 255), 1)
    
    return result