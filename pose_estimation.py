#!/usr/bin/env python3
"""
三维计算模块
包含相机参数、位姿估计、3D可视化等功能
"""

import numpy as np
import cv2
from typing import Tuple, Optional

def define_qr_object_points(qr_size: float = 0.1) -> np.ndarray:
    """定义二维码的3D角点（世界坐标系，单位：米）
    
    假设二维码在XY平面上，Z=0，中心在原点
    角点顺序：左上、右上、右下、左下（顺时针）
    
    参数:
        qr_size: 二维码物理尺寸（米）
        
    返回:
        object_points: (4, 3) 的numpy数组
    """
    half = qr_size / 2.0
    object_points = np.array([
        [-half, half, 0.0],   # 左上
        [half, half, 0.0],    # 右上
        [half, -half, 0.0],   # 右下
        [-half, -half, 0.0]   # 左下
    ], dtype=np.float32)
    return object_points

def create_camera_matrix(image_size: Tuple[int, int] = (640, 480), fov_x: float = 60.0, fov_y: Optional[float] = None) -> np.ndarray:
    """创建默认相机内参矩阵
    
    参数:
        image_size: 图像尺寸 (宽, 高)
        fov_x: 水平视场角（度）
        fov_y: 垂直视场角（度），如果为None则使用fov_x计算（假设正方形像素）
        
    返回:
        camera_matrix: 3x3相机内参矩阵
    """
    W, H = image_size
    # 根据水平视场角计算水平焦距（像素）
    f_x = W / (2 * np.tan(np.radians(fov_x) / 2))
    
    # 如果提供了垂直视场角，则使用它计算垂直焦距
    if fov_y is not None:
        f_y = H / (2 * np.tan(np.radians(fov_y) / 2))
        print("fov_y is set")
    else:
        f_y = f_x  # 假设正方形像素
    
    camera_matrix = np.array([
        [f_x, 0.0, W/2],
        [0.0, f_y, H/2],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return camera_matrix

def create_default_extrinsics(tx: float = 0.0, ty: float = 0.0, tz: float = 0.5,
                             angle_x: float = 0.0, angle_y: float = 0.0, angle_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建默认外参（相机位姿）
    
    参数:
        tx, ty, tz: 相机在二维码坐标系中的位置（米）
        angle_x, angle_y, angle_z: 绕X、Y、Z轴的旋转角度（度）
        
    返回:
        rvec: 旋转向量（3x1）
        tvec: 平移向量（3x1）
        R: 旋转矩阵（3x3）
    """
    # 旋转矩阵（欧拉角，ZYX顺序）
    angles_rad = np.radians([angle_x, angle_y, angle_z])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                   [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])
    
    Ry = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                   [0, 1, 0],
                   [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])
    
    Rz = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                   [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # 平移向量：相机在二维码坐标系中的位置
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float32)
    
    # 将旋转矩阵转换为旋转向量
    rvec, _ = cv2.Rodrigues(R)
    
    return rvec, tvec, R

def rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    """将旋转矩阵转换为欧拉角（ZYX顺序，即yaw, pitch, roll）
    
    参数:
        rmat: 3x3旋转矩阵
        
    返回:
        (yaw, pitch, roll) 单位为弧度
        
    异常:
        ValueError: 旋转矩阵无效
    """
    # 确保旋转矩阵是有效的
    if rmat.shape != (3, 3):
        raise ValueError("旋转矩阵必须是3x3")
    
    # 提取旋转矩阵元素
    r00, r01, r02 = rmat[0, 0], rmat[0, 1], rmat[0, 2]
    r10, r11, r12 = rmat[1, 0], rmat[1, 1], rmat[1, 2]
    r20, r21, r22 = rmat[2, 0], rmat[2, 1], rmat[2, 2]
    
    # 检查万向节锁情况
    if np.abs(r20) < 1e-6:
        r20 = 0.0
    
    if np.abs(r20) < 1 - 1e-6:
        # 正常情况：无万向节锁
        pitch = np.arcsin(-r20)
        yaw = np.arctan2(r10, r00)
        roll = np.arctan2(r21, r22)
    else:
        # 万向节锁情况
        if r20 > 0:  # pitch = -90度
            pitch = -np.pi/2
            yaw = -np.arctan2(-r12, r11)
            roll = 0.0
        else:  # pitch = 90度
            pitch = np.pi/2
            yaw = np.arctan2(-r12, r11)
            roll = 0.0
    
    return yaw, pitch, roll

def load_camera_parameters(W: int, H: int, 
                          camera_matrix_path: Optional[str] = None, 
                          dist_coeffs_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参和畸变系数
    
    如果未提供，返回默认值（仅用于演示，实际使用时需要标定）
    
    参数:
        W: 图像宽度
        H: 图像高度
        camera_matrix_path: 相机内参矩阵文件路径（.npy格式）
        dist_coeffs_path: 畸变系数文件路径（.npy格式）
        
    返回:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    if camera_matrix_path and dist_coeffs_path:
        camera_matrix = np.load(camera_matrix_path)
        dist_coeffs = np.load(dist_coeffs_path)
        return camera_matrix, dist_coeffs
    else:
        # 默认相机矩阵
        camera_matrix = np.array([[W, 0., W/2],
                                  [0., H, H/2],
                                  [0., 0., 1.]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 无畸变
        return camera_matrix, dist_coeffs

def estimate_pose(object_points: np.ndarray, image_points: np.ndarray, 
                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """使用solvePnP估计位姿
    
    返回旋转向量、平移向量、旋转矩阵
    
    参数:
        object_points: 3D角点坐标 (N,3)
        image_points: 2D图像角点坐标 (N,2)
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        
    返回:
        success: 成功标志
        rvec: 旋转向量 (3,1)
        tvec: 平移向量 (3,1)
        rmat: 旋转矩阵 (3,3)
    """
    # 使用迭代法（SOLVEPNP_ITERATIVE）求解PnP问题
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return False, None, None, None
    
    # 将旋转向量转换为旋转矩阵
    rmat, _ = cv2.Rodrigues(rvec)
    return True, rvec, tvec, rmat

def draw_axes(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
              rvec: np.ndarray, tvec: np.ndarray, length: float = 0.05) -> np.ndarray:
    """在图像上绘制3D坐标系（X红，Y绿，Z蓝）
    
    参数:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        rvec: 旋转向量
        tvec: 平移向量
        length: 轴长度（米）
        
    返回:
        image_with_axes: 绘制了坐标轴的图像
    """
    # 定义坐标系轴点（世界坐标）
    axis_points = np.float32([
        [0, 0, 0],           # 原点
        [length, 0, 0],      # X轴
        [0, length, 0],      # Y轴
        [0, 0, -length]      # Z轴（OpenCV中Z轴指向相机）
    ]).reshape(-1, 3)
    
    # 投影到图像平面
    axis_proj, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    axis_proj = axis_proj.reshape(-1, 2).astype(int)
    
    # 绘制轴
    origin = tuple(axis_proj[0])
    cv2.line(image, origin, tuple(axis_proj[1]), (0, 0, 255), 2)  # X轴：红色
    cv2.line(image, origin, tuple(axis_proj[2]), (0, 255, 0), 2)  # Y轴：绿色
    cv2.line(image, origin, tuple(axis_proj[3]), (255, 0, 0), 2)  # Z轴：蓝色
    
    # 添加标签
    cv2.putText(image, 'X', tuple(axis_proj[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(image, 'Y', tuple(axis_proj[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(image, 'Z', tuple(axis_proj[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    return image

def visualize_3d_scene(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                       rvec: np.ndarray, tvec: np.ndarray, object_points: np.ndarray, 
                       qr_size: float = 0.1):
    """使用matplotlib显示相机和二维码的3D场景
    
    参数:
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 畸变系数 (5x1)
        rvec: 旋转向量 (3x1) - 从世界坐标系到相机坐标系
        tvec: 平移向量 (3x1) - 相机原点在世界坐标系中的位置
        object_points: 二维码的3D角点坐标 (Nx3)
        qr_size: 二维码物理尺寸（米）
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.patches as mpatches
    except ImportError:
        print("错误: 需要matplotlib库进行3D可视化")
        print("请安装: pip install matplotlib")
        return
    
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 计算相机在世界坐标系中的位置 (相机原点)
    # tvec = -R * camera_position，所以 camera_position = -R^T * tvec
    camera_position = -R.T @ tvec  # (3,1)
    camera_position = camera_position.flatten()  # (3,)
    
    # 计算相机坐标系轴的方向（在世界坐标系中表示）
    # 相机坐标系的X轴（右）、Y轴（下）、Z轴（前）在世界坐标系中的方向
    cam_x_axis = R.T @ np.array([[1], [0], [0]])  # 相机X轴在世界坐标系中
    cam_y_axis = R.T @ np.array([[0], [1], [0]])  # 相机Y轴在世界坐标系中
    cam_z_axis = R.T @ np.array([[0], [0], [1]])  # 相机Z轴在世界坐标系中
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 1. 绘制二维码平面
    qr_points = object_points
    # 定义二维码平面的四个顶点（顺序：左上、右上、右下、左下）
    vertices = qr_points[[0, 1, 2, 3, 0]]  # 闭合多边形
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            'k-', linewidth=2, label='QR Code Border')
    
    # 填充二维码平面（半透明灰色）
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([qr_points], alpha=0.2, color='gray')
    ax.add_collection3d(poly)
    
    # 在二维码中心添加文本标签
    qr_center = np.mean(qr_points, axis=0)
    ax.text(qr_center[0], qr_center[1], qr_center[2], 
            'QR Code', fontsize=10, ha='center')
    
    # 2. 绘制相机位置和坐标系
    # 相机位置点
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
               c='red', s=100, marker='^', label='Camera')
    
    # 相机坐标系轴（长度为0.1米）
    axis_length = 0.05
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_x_axis[0], cam_x_axis[1], cam_x_axis[2],
              length=axis_length, color='r', arrow_length_ratio=0.1, label='Camera X')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_y_axis[0], cam_y_axis[1], cam_y_axis[2],
              length=axis_length, color='g', arrow_length_ratio=0.1, label='Camera Y')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_z_axis[0], cam_z_axis[1], cam_z_axis[2],
              length=axis_length, color='b', arrow_length_ratio=0.1, label='Camera Z')
    
    # 3. 绘制世界坐标系轴（在二维码中心）
    world_axis_length = qr_size * 0.8
    ax.quiver(0, 0, 0, world_axis_length, 0, 0, color='r', 
              arrow_length_ratio=0.1, linestyle='--', label='World X')
    ax.quiver(0, 0, 0, 0, world_axis_length, 0, color='g', 
              arrow_length_ratio=0.1, linestyle='--', label='World Y')
    ax.quiver(0, 0, 0, 0, 0, world_axis_length, color='b', 
              arrow_length_ratio=0.1, linestyle='--', label='World Z')
    
    # 4. 绘制从相机到二维码中心的视线
    ax.plot([camera_position[0], qr_center[0]],
            [camera_position[1], qr_center[1]],
            [camera_position[2], qr_center[2]],
            'y--', alpha=0.5, label='Line of Sight')
    
    # 5. 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('QR Code and Camera 3D Scene')
    
    # 设置坐标轴比例相等
    max_range = max(qr_size, np.linalg.norm(camera_position)) * 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # 添加信息文本
    info_text = f"Camera Position: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}] m\n"
    info_text += f"Camera Distance: {np.linalg.norm(camera_position):.2f} m\n"
    info_text += f"QR Code Size: {qr_size} m"
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    print("3D场景可视化已显示，关闭窗口后程序将继续运行...")