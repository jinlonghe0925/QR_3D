#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云台位姿估计接口
功能：
1. 从云台抓取可见光图片
2. 使用实际相机估计接口进行位姿估计（非仿真）
3. 打印估计的位姿信息，用于手动调整云台朝向
4. 实时显示视频流

使用说明：
- 确保云台已连接并上电
- 确保二维码在可见光相机视野内
- 运行后，程序会自动抓取图像并估计位姿
- 根据输出的位姿信息，手动调整云台角度
- 按 'v' 键可以打开/关闭实时视频流显示
"""

import sys
import time
import os
import threading
import numpy as np
import cv2

# 导入云台控制模块
try:
    from gimbal_demo import GimbalController, capture_from_rtsp
except ImportError:
    print("错误: 无法导入 gimbal_demo 模块")
    print("请确保 gimbal_demo.py 在同一目录下")
    sys.exit(1)

# 导入 main_demo 模块以使用其 save_debug_image 函数
try:
    import main_demo
except ImportError:
    print("警告: 无法导入 main_demo 模块，将使用内置的保存函数")
    main_demo = None

# 导入二维码检测和位姿估计模块
import yolo_detection
import qr_detection
import pose_estimation


class GimbalPoseEstimator:
    """云台位姿估计器"""
    
    def __init__(self, 
                 gimbal_ip="192.168.1.99", 
                 gimbal_port=6001,
                 config_file="config.json",
                 qr_size=None,
                 fov_x=60.0,
                 model_size='s',
                 detector_type='opencv',
                 square_size=800,
                 conf_threshold=0.2,
                 nms_threshold=0.5):
        """
        初始化云台位姿估计器
        
        参数:
            gimbal_ip: 云台IP地址
            gimbal_port: 云台控制端口
            config_file: 配置文件路径（用于读取qr_size等参数）
            qr_size: 二维码物理尺寸（米），如果为None则从config.json读取
            fov_x: 相机水平视场角（度）
            model_size: YOLOv8模型大小
            detector_type: 二维码检测器类型
            square_size: 矫正后正方形图像边长
            conf_threshold: YOLOv8置信度阈值
            nms_threshold: YOLOv8非极大值抑制阈值
        """
        print("=" * 60)
        print("初始化云台位姿估计器")
        print("=" * 60)
        
        # 从配置文件读取参数
        self.config_file = config_file
        config_data = self._load_config(config_file)
        
        # 如果未提供qr_size，从配置文件读取
        if qr_size is None:
            qr_size = config_data.get("estimate", {}).get("qr_size", 0.1)
            print(f"  从配置文件读取 qr_size -------------------------: {qr_size} 米")
        qr_size = 0.05
        #qr_size = 0.12

        print(f" qr_size -------------------------: {qr_size} 米")

        # 保存配置参数
        self.gimbal_ip = gimbal_ip
        self.qr_size = qr_size
        self.fov_x = fov_x
        self.model_size = model_size
        self.detector_type = detector_type
        self.square_size = square_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 初始化云台控制器
        print(f"\n1. 连接云台: {gimbal_ip}:{gimbal_port}")
        try:
            self.gimbal = GimbalController(ip=gimbal_ip, port=gimbal_port)
            print("   ✓ 云台连接成功")
        except Exception as e:
            print(f"   ✗ 云台连接失败: {e}")
            raise
        
        # 构建RTSP地址
        self.rtsp_tv = f"rtsp://{gimbal_ip}:554/ch1/stream0"  # 可见光
        self.rtsp_ir = f"rtsp://{gimbal_ip}:554/ch2/stream0"  # 红外
        print(f"   可见光地址: {self.rtsp_tv}")
        print(f"   红外地址:   {self.rtsp_ir}")
        
        # 加载YOLOv8检测器
        print(f"\n2. 加载YOLOv8检测器 (模型: {model_size})")
        self.detector = yolo_detection.load_yolov8_detector(
            model_size=model_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
        if self.detector is None:
            raise RuntimeError("YOLOv8检测器加载失败")
        print("   ✓ 检测器加载成功")
        
        # 创建输出目录
        self.output_dir = "gimbal_pose_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n3. 输出目录: {self.output_dir}")
        
        # 视频流相关属性
        self.video_capture = None           # OpenCV VideoCapture对象
        self.video_thread = None            # 视频显示线程
        self.video_running = False          # 视频流运行标志
        self.video_paused = False           # 视频暂停标志
        self.current_frame = None           # 当前帧
        self.frame_lock = threading.Lock()  # 帧锁
        self.latest_pose_data = None       # 最新的位姿数据
        self.show_pose_on_video = False    # 是否在视频上显示位姿信息
        
        # Review 模式相关属性
        self.review_mode = True           # 是否处于 review 模式
        self.review_detections = []        # 保存YOLO检测结果供循环绘制
        
        print("\n" + "=" * 60)
        print("初始化完成")
        print("=" * 60)

    def _load_config(self, config_file):
        """
        从JSON配置文件读取参数

        参数:
            config_file: 配置文件路径

        返回:
            配置字典，如果文件不存在则返回空字典
        """
        import json
        config_data = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                print(f"  已加载配置文件: {config_file}")
            except Exception as e:
                print(f"  警告: 无法读取配置文件 {config_file}: {e}")
        else:
            print(f"  警告: 配置文件不存在: {config_file}，使用默认参数")

        return config_data

    def capture_visible_image(self, output_path=None, use_rtsp=True):
        """
        步骤1: 从云台抓取可见光图片
        
        参数:
            output_path: 输出图像路径（如果为None，则自动生成）
            use_rtsp: 是否使用RTSP流（True）或直接抓取（False）
        
        返回:
            success: 是否成功
            image_path: 保存的图像路径
        """
        print("\n" + "-" * 60)
        print("步骤1: 抓取可见光图像")
        print("-" * 60)
        
        # 生成输出文件路径
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"capture_tv_{timestamp}.jpg")
        
        print(f"  输出路径: {output_path}")
        
        # 方法1: 使用RTSP流抓取（推荐）
        if use_rtsp:
            print(f"  方法: RTSP流 ({self.rtsp_tv})")
            success = capture_from_rtsp(self.rtsp_tv, output_path, timeout=5)
        
        # 方法2: 使用云台命令直接抓取
        else:
            print(f"  方法: 云台命令 (通道0=可见光)")
            success = self.gimbal.capture_image(
                channel=0,  # 0=可见光
                output_path=output_path,
                timeout=10.0
            )
        
        if success:
            print(f"  ✓ 图像抓取成功: {output_path}")
            # 验证图像可读性
            img = cv2.imread(output_path)
            if img is not None:
                print(f"  图像尺寸: {img.shape[1]}x{img.shape[0]}")
                return True, output_path
            else:
                print(f"  ✗ 图像保存成功但无法读取")
                return False, None
        else:
            print(f"  ✗ 图像抓取失败")
            return False, None
    
    def estimate_pose_from_image(self, image_path, debug=True, debug_dir=None):
        """
        步骤2: 使用实际相机估计接口进行位姿估计
        
        注意：这不是仿真，而是实际相机的位姿估计
        严格按照main_demo.py的process_image_for_pose函数调用方式
        
        参数:
            image_path: 输入图像路径
            debug: 是否保存调试图像
            debug_dir: 调试图像保存目录
        
        返回:
            success: 是否成功
            pose_data: 位姿数据字典
            result_image: 结果图像（带标注）
        """
        print("\n" + "-" * 60)
        print("步骤2: 位姿估计（实际相机）")
        print("-" * 60)
        
        # 初始化调试目录（按照main_demo.py的方式）
        if debug:
            if debug_dir is None:
                timestamp = int(time.time())
                debug_dir = f"debug_output_{timestamp}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"\n  调试模式已启用，中间图片将保存到: {debug_dir}")
        
        # 读取图像
        print(f"\n  读取图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ 无法读取图像")
            return False, None, None
        
        print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 1. 保存原始图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(image, "01_original_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "01_original_image.jpg")
                cv2.imwrite(filepath, image)
                print(f"  调试: 图像已保存到 {filepath}")
        
        # 使用YOLOv8检测二维码
        print(f"\n  2.1 YOLOv8检测二维码...")
        detections = yolo_detection.detect_qr_yolov8(image, self.detector)
        print(f"  检测到 {len(detections)} 个二维码")
        
        if len(detections) == 0:
            print(f"  ✗ 未检测到二维码")
            return False, None, None
        
        # 只处理第一个检测到的二维码
        detection = detections[0]
        polygon_xy = detection['polygon_xy'].astype(np.float32)
        confidence = detection['confidence']
        print(f"  使用检测结果 1 (置信度: {confidence:.2f})")
        
        # 2. 保存YOLO初始检测结果图（带边界框）（按照main_demo.py）
        # 绘制YOLO框到图像上
        image_init_detect = qr_detection.draw_qr_boundary(image.copy(), 
                                                        qr_detection.order_points_clockwise2(
                                                            qr_detection.approximate_quadrilateral(polygon_xy)
                                                        ))
        
        # 保存带YOLO框的图像，用于detection窗口显示
        self.review_frame_with_yolo = image_init_detect.copy()
        
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(image_init_detect, "02_yolo_initial_detect", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "02_yolo_initial_detect.jpg")
                cv2.imwrite(filepath, image_init_detect)
                print(f"  调试: 图像已保存到 {filepath}")
        

        
        # 近似为四边形
        print(f"\n  2.2 近似四边形...")
        try:
            quad_points = qr_detection.approximate_quadrilateral(polygon_xy)
            quad_points = qr_detection.order_points_clockwise2(quad_points)
            print(f"  ✓ 四边形角点: {quad_points.shape}")
        except ValueError as e:
            print(f"  ✗ 四边形近似失败: {e}")
            return False, None, None



        #--根据yolo增加遮罩，方便后续alpha合成 -------------------------------------------

        # 生成二值图像（YOLO框内为255，框外为0）
        # 计算YOLO框的边界
        x_coords = quad_points[:, 0]
        y_coords = quad_points[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # 计算中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 计算最大边长（最大y差和最大x差的较大值）
        max_x_diff = x_max - x_min
        max_y_diff = y_max - y_min
        max_side = max(max_x_diff, max_y_diff)
        
        print(f"  YOLO框中心: ({center_x}, {center_y})")
        print(f"  YOLO框最大边长: {max_side} pixels")
        
        # 创建二值图像（与image_init_detect同分辨率）
        alpha_mask = np.zeros(image_init_detect.shape[:2], dtype=np.uint8)
        
        # 在YOLO框内填充255（使用多边形填充）
        pts = quad_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(alpha_mask, [pts], 255)
        
        # 保存二值图像
        if debug:
            alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_alpha.jpg")
        else:
            alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_alpha.jpg")
        cv2.imwrite(alpha_filepath, alpha_mask)
        print(f"  调试: 二值图像已保存到 {alpha_filepath}")
        
        # 对二值图像进行膨胀操作，让白色部分变大
        # 定义膨胀核（5x5的矩形核）
        kernel = np.ones((55, 55), np.uint8)
        alpha_mask_big = cv2.dilate(alpha_mask, kernel, iterations=1)
        
        # 保存膨胀后的二值图像
        if debug:
            big_alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_big_alpha.jpg")
        else:
            big_alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_big_alpha.jpg")
        cv2.imwrite(big_alpha_filepath, alpha_mask_big)
        print(f"  调试: 膨胀后二值图像已保存到 {big_alpha_filepath}")
        
        #--------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------

        # 透视变换矫正 - 按照main_demo.py的方式
        print(f"\n  2.3 透视变换矫正...")
        
        # 按照main_demo.py的方式计算dst_square
        margin_qr =1500  # 边距
        shift_qr = margin_qr / 2.0  # 偏移量
        
        dst_square = np.array([
            [shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, self.square_size - 1 + shift_qr],
            [shift_qr, self.square_size - 1 + shift_qr]
        ], dtype=np.float32)
        
        H = qr_detection.compute_homography(quad_points, dst_square)
        if H is None:
            print(f"  ✗ 单应矩阵计算失败")
            return False, None, None
        
        # 透视变换目标尺寸按照main_demo.py的方式
        warped = qr_detection.warp_perspective_with_homography(
            image, H, (self.square_size + margin_qr, self.square_size + margin_qr)
        )
        #--------------------------------------------------------------------------------------------

        warped_alpha = qr_detection.warp_perspective_with_homography(
            alpha_mask_big, H, (self.square_size + margin_qr, self.square_size + margin_qr)
        )
        #3. 保存透视变换后的alpha图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped_alpha, "03_warped_alpha_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_alpha_image.jpg")
                cv2.imwrite(filepath, warped_alpha)
                print(f"  调试: 图像已保存到 {filepath}")


        #----------------------------------------------------------------------------------------------


        print(f"  ✓ 透视变换完成")
        print(f"  透视变换尺寸: {self.square_size + margin_qr}x{self.square_size + margin_qr}")
        
        # 3. 保存透视变换后的矫正图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped, "03_warped_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_image.jpg")
                cv2.imwrite(filepath, warped)
                print(f"  调试: 图像已保存到 {filepath}")
        


        # alpha合成
        warped_alpha = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR)
        warped_alpha = warped_alpha / 255.0
        warped_01 = warped / 255.0
        warped_comp = warped_alpha * warped_01 
        warped_comp = warped_comp * 255.0
        warped_comp = warped_comp.astype(np.uint8)

        #step2 : 补白---------------------------------------------------------------------
        border_image =(1-warped_alpha)* 255.0
        border_image = border_image.astype(np.uint8)
        warped_comp = cv2.add(warped_comp, border_image)


        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped_comp, "03_warped_image_comp", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_image_comp.jpg")
                cv2.imwrite(filepath, warped_comp)
                print(f"  调试: 图像已保存到 {filepath}")
        
        # 在矫正图像上检测二维码角点
        print(f"\n  2.4 检测二维码角点...")
        # success, points_warped, decoded_info = qr_detection.detect_qr_in_warped_image(
        #     warped, self.detector_type
        # )

        success, points_warped, decoded_info = qr_detection.detect_qr_in_warped_image(
            warped, self.detector_type
        )
        
        if success:
            print(f"  ✓ 检测到二维码: {decoded_info}")
            # 映射回原始图像
            H_inv = np.linalg.inv(H)
            points_corrected = qr_detection.transform_points_back(points_warped, H_inv)


            points_corrected = points_corrected[:, :2]
            #---------------由于相机缺失一个自由度，不能绕Z轴旋转，所以需要将Z轴旋转的角度去掉-----------------
            points_corrected = qr_detection.order_points_clockwise2(points_corrected)
            #---------------
            print(f"  ✓ 角点已映射回原始图像")
            
            # 4. 保存矫正图像上的检测结果（带边界框）（按照main_demo.py）
            if debug:
                image_warped_detect = qr_detection.draw_qr_boundary(warped.copy(), points_warped)
                if main_demo is not None:
                    main_demo.save_debug_image(image_warped_detect, "04_warped_detect", debug_dir)
                else:
                    os.makedirs(debug_dir, exist_ok=True)
                    filepath = os.path.join(debug_dir, "04_warped_detect.jpg")
                    cv2.imwrite(filepath, image_warped_detect)
                    print(f"  调试: 图像已保存到 {filepath}")
        else:
            print(f"  ⚠ 矫正图像未检测到二维码，使用YOLO检测的角点")
            points_corrected = quad_points
        
        
        # 定义二维码的3D角点（世界坐标系）
        print(f"\n  2.5 位姿估计...")
        print(f"二维码实际尺寸...", self.qr_size)
        object_points = pose_estimation.define_qr_object_points(qr_size=self.qr_size)
        
        # 查询当前焦距和视场角，用于计算相机内参矩阵
        print(f"\n  2.5.1 查询当前焦距和视场角...")
        fov_x, fov_y, focal_x, focal_y = self.gimbal.query_focal_length()
        if fov_x is not None and fov_y is not None:
            print(f"  ✓ 查询成功: fov_x={fov_x:.1f}°, fov_y={fov_y:.1f}°, focal_x={focal_x:.1f}mm, focal_y={focal_y:.1f}mm")
            # 使用查询到的视场角创建相机内参矩阵
            H, W = image.shape[:2]
            camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=fov_x, fov_y=fov_y)
        else:
            print(f"  ⚠ 查询失败，使用默认fov_x={self.fov_x}°")
            H, W = image.shape[:2]
            camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=self.fov_x)
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        print(f"  相机内参矩阵:\n{camera_matrix}")
        
        # 估计位姿
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
        camera_position = -rmat.T @ tvec
        
        # 整理位姿数据
        pose_data = {
            'rvec': rvec.flatten(),                    # 旋转向量
            'tvec': tvec.flatten(),                    # 平移向量
            'rmat': rmat,                             # 旋转矩阵
            'yaw': np.degrees(yaw),                   # 偏航角（度）
            'pitch': np.degrees(pitch),                # 俯仰角（度）
            'roll': np.degrees(roll),                  # 横滚角（度）
            'camera_position': camera_position.flatten(),  # 相机位置（米）
            'distance': np.linalg.norm(camera_position),   # 距离（米）
            'qr_size': self.qr_size,                  # 二维码尺寸（米）
            'image_points': points_corrected,          # 图像角点
            'object_points': object_points,            # 二维码3D角点（世界坐标系）
            'camera_matrix': camera_matrix             # 相机内参矩阵
        }
        
        # 绘制结果图像
        print(f"\n  2.6 绘制结果...")
        result_image = image.copy()
        
        # 绘制二维码边界
        result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
        
        # 绘制坐标轴
        result_image = pose_estimation.draw_axes(
            result_image, camera_matrix, dist_coeffs, rvec, tvec
        )
        
        print(f"  ✓ 结果图像已绘制")
        
        # 5. 保存最终结果图（带正确角点顺序和坐标系）（按照main_demo.py）
        if debug and points_corrected is not None:
            if main_demo is not None:
                main_demo.save_debug_image(result_image, "05_result_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "05_result_image.jpg")
                cv2.imwrite(filepath, result_image)
                print(f"  调试: 图像已保存到 {filepath}")
            print(f"  调试: 中间图片已保存到 {debug_dir}")
        
        return True, pose_data, result_image
    
    def print_pose_info(self, pose_data):
        """
        步骤3: 打印位姿信息
        
        这些信息可以手动输入到云台控制系统中
        
        参数:
            pose_data: 位姿数据字典
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
        for i, pt in enumerate(pose_data['image_points']):
            print(f"    角点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")


        #------云台坐标转换------------------------------------------------

        yam_camera_input = pose_data['pitch']
        pitch_camera_input = pose_data['roll'] - 180.0

        if(abs(pitch_camera_input)>=180):
            pitch_camera_input_temp = (360.0 + pitch_camera_input)% 360.0
            pitch_camera_input = pitch_camera_input_temp



        #-----获知当前云台朝向，并计算云台需要调整的角度-------------------

        # 获取当前云台角度并加上相机角度
        try:
            #gimbal_yaw, gimbal_pitch = self.gimbal.get_current_angles()
            gimbal_yaw, gimbal_pitch, gimbal_roll = self.gimbal.get_current_angles()

            print(f"\n【变量角度】")
            print(f"  add  Yaw:   {yam_camera_input:.2f}°")
            print(f"  add Pitch: {pitch_camera_input:.2f}°")


            print(f"\n【云台当前角度】")
            print(f"  Gimbal Yaw:   {gimbal_yaw:.2f}°")
            print(f"  Gimbal Pitch: {gimbal_pitch:.2f}°")
            print(f"  Gimbal Roll: {gimbal_roll:.2f}°")

            # 加上云台当前角度
            yam_camera_input += gimbal_yaw
            pitch_camera_input += gimbal_pitch

            print(f"\n【叠加后角度】 (位姿角度 + 云台当前角度)")
        except Exception as e:
            print(f"\n警告: 无法获取云台当前角度: {e}")
            print(f"\n【位姿角度】 (未叠加云台角度)")

        #-----------------------------------------------------------------
        
        print("\n" + "=" * 60)
        print("手动调整云台朝向 - 参考值")
        print("=" * 60)
        print(f"\n根据位姿估计结果，建议的云台角度（仅供参考）:")
        print(f"  Yaw   (水平旋转): {yam_camera_input:.2f}°")
        print(f"  Pitch (上下俯仰): {pitch_camera_input:.2f}°")
        print(f"\n注意: 实际调整时需要考虑坐标系定义和安装方式")
        print("=" * 60)
    
    def save_result(self, result_image, pose_data, output_prefix=None):
        """
        保存结果图像和位姿数据
        
        参数:
            result_image: 结果图像
            pose_data: 位姿数据
            output_prefix: 输出文件前缀
        """
        if output_prefix is None:
            output_prefix = os.path.join(self.output_dir, f"pose_result_{int(time.time())}")
        
        # 保存结果图像
        result_image_path = f"{output_prefix}.jpg"
        cv2.imwrite(result_image_path, result_image)
        print(f"\n结果图像已保存: {result_image_path}")
        
        # 保存位姿数据（NPY格式）
        pose_data_path = f"{output_prefix}_pose.npy"
        np.save(pose_data_path, pose_data)
        print(f"位姿数据已保存: {pose_data_path}")
        
        # 保存位姿数据（文本格式，方便查看）
        pose_text_path = f"{output_prefix}_pose.txt"
        with open(pose_text_path, 'w', encoding='utf-8') as f:
            f.write("位姿估计结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"旋转向量 (rvec):\n")
            f.write(f"  {pose_data['rvec']}\n\n")
            f.write(f"平移向量 (tvec):\n")
            f.write(f"  {pose_data['tvec']}\n\n")
            f.write(f"欧拉角 (度):\n")
            f.write(f"  Yaw:   {pose_data['yaw']:.2f}°\n")
            f.write(f"  Pitch: {pose_data['pitch']:.2f}°\n")
            f.write(f"  Roll:  {pose_data['roll']:.2f}°\n\n")
            f.write(f"相机位置 (米):\n")
            f.write(f"  X: {pose_data['camera_position'][0]:.6f}\n")
            f.write(f"  Y: {pose_data['camera_position'][1]:.6f}\n")
            f.write(f"  Z: {pose_data['camera_position'][2]:.6f}\n")
            f.write(f"  距离: {pose_data['distance']:.6f}\n")
        print(f"位姿数据(文本)已保存: {pose_text_path}")
    
    def run_single_estimation(self, save_result=True):
        """
        运行单次位姿估计
        
        流程:
        1. 抓取可见光图像
        2. 位姿估计
        3. 打印结果
        4. 可选：保存结果
        
        返回:
            success: 是否成功
            pose_data: 位姿数据
        """
        print("\n" + "=" * 60)
        print("开始位姿估计流程")
        print("=" * 60)
        
        # 步骤1: 抓取图像
        success, image_path = self.capture_visible_image(use_rtsp=True)
        if not success:
            print("\n✗ 流程终止: 图像抓取失败")
            return False, None
        
        # 步骤2: 位姿估计
        success, pose_data, result_image = self.estimate_pose_from_image(image_path)
        if not success:
            print("\n✗ 流程终止: 位姿估计失败")
            return False, None
        
        # 步骤3: 打印位姿信息
        self.print_pose_info(pose_data)
        
        # 步骤4: 保存结果（可选）
        if save_result and result_image is not None:
            self.save_result(result_image, pose_data)
        
        print("\n" + "=" * 60)
        print("位姿估计流程完成")
        print("=" * 60)
        
        return True, pose_data
    
    def run_interactive_mode(self):
        """
        运行交互模式
        
        用户可以多次抓取图像并估计位姿，
        根据结果手动调整云台角度
        可以打开实时视频流查看画面
        """
        print("\n" + "=" * 60)
        print("交互模式")
        print("=" * 60)
        print("\n命令说明:")
        print("  c - 抓取图像并估计位姿")
        print("  v - 打开/关闭实时视频流")
        print("  s - 调整云台角度")
        print("  z - 变倍/变焦/调焦控制")
        print("  a - 自动瞄准模式（等间隔识别+自动旋转）")
        print("  q - 退出")
        print("=" * 60)
        print("\n视频流窗口快捷键:")
        print("  q - 关闭视频")
        print("  p - 暂停/继续")
        print("  c - 捕获当前帧并估计位姿")
        print("  s - 显示/隐藏位姿信息")
        print("=" * 60)
        
        while True:
            cmd = input("\n请选择命令 (c/v/s/q): ").strip().lower()
            
            if cmd == 'q':
                print("\n退出程序")
                # 停止视频流
                if self.video_running:
                    self.stop_video_stream()
                break
            
            elif cmd == 'c':
                # 抓取并估计
                success, pose_data = self.run_single_estimation(save_result=True)
                
                if success:
                    print("\n✓ 位姿估计完成，请根据上面的信息调整云台")
            
            elif cmd == 'v':
                # 打开/关闭视频流
                if self.video_running:
                    print("\n关闭视频流...")
                    self.stop_video_stream()
                else:
                    print("\n打开视频流...")
                    self.start_video_stream(display_pose=True)
            
            elif cmd == 's':
                # 调整云台角度
                print("\n调整云台角度:")
                print("(输入角度，单位: 度)")
                try:
                    yaw = float(input("  Yaw (偏航角): "))
                    pitch = float(input("  Pitch (俯仰角): "))
                    roll = float(input("  roll (横滚角): "))

                    print(f"\n发送云台控制命令...")
                    self.gimbal.follow(yaw, pitch, roll)
                    print(f"✓ 云台已调整到: Yaw={yaw}°, Pitch={pitch}°, Roll={roll}°")
                
                except ValueError:
                    print("✗ 输入无效，请输入数字")
                except Exception as e:
                    print(f"✗ 云台控制失败: {e}")
            
            elif cmd == 'z':
                # 变倍/变焦/调焦控制
                print("\n=== 变倍/变焦/调焦控制 ===")
                print("  1. 变倍+ (Zoom In)")
                print("  2. 变倍- (Zoom Out)")
                print("  3. 变倍停 (Zoom Stop)")
                print("  4. 广视场/短焦 (Wide)")
                print("  5. 窄视场/长焦 (Narrow)")
                print("  6. 变焦到指定焦距 (Zoom to Focal Length)")
                print("  7. 调焦+ (Focus Near)")
                print("  8. 调焦- (Focus Far)")
                print("  9. 调焦停 (Focus Stop)")
                print(" 10. 变焦到指定视场角 (Zoom to FOV)")
                print(" 11. 变倍到指定倍率 (Zoom to Magnification)")
                print(" 12. 变倍级别控制 (1-50级)")
                print(" 13. 自动对焦 (Auto Focus)")
                print(" 14. 查询当前焦距 (Query Focal Length)")
                print("  r. 读取焦距、变倍、调焦信息")
                print("  0. 返回")
                print("=" * 30)
                
                sub_cmd = input("\n请选择 (1-11/0/r): ").strip().lower()
                
                try:
                    if sub_cmd == '1':
                        self.gimbal.zoom_in()
                        print("✓ 变倍+")
                    elif sub_cmd == '2':
                        self.gimbal.zoom_out()
                        print("✓ 变倍-")
                    elif sub_cmd == '3':
                        self.gimbal.zoom_stop()
                        print("✓ 变倍停")
                    elif sub_cmd == '4':
                        self.gimbal.focus_wide()
                        print("✓ 广视场(短焦)")
                    elif sub_cmd == '5':
                        self.gimbal.focus_narrow()
                        print("✓ 窄视场(长焦)")
                    elif sub_cmd == '6':
                        focal_length = float(input("  输入焦距 (mm): "))
                        self.gimbal.zoom_to_focal_length(focal_length)
                        print(f"✓ 变焦到指定焦距: {focal_length}mm")
                    elif sub_cmd == '7':
                        self.gimbal.focus_near()
                        print("✓ 调焦+")
                    elif sub_cmd == '8':
                        self.gimbal.focus_far()
                        print("✓ 调焦-")
                    elif sub_cmd == '9':
                        self.gimbal.focus_stop()
                        print("✓ 调焦停")
                    elif sub_cmd == '10':
                        fov = float(input("  输入视场角 (度): "))
                        self.gimbal.zoom_to_fov(fov)
                        print(f"✓ 变焦到指定视场角: {fov}°")
                    elif sub_cmd == '11':
                        mag = float(input("  输入倍率: "))
                        self.gimbal.zoom_to_magnification(mag)
                        print(f"✓ 变倍到指定倍率: {mag}x")
                    elif sub_cmd == '12':
                        # 变倍级别控制 (1-50级，1=最小倍率，50=最大倍率)
                        print("\n  变倍级别控制 (1-50级)")
                        print("  1 = 最小倍率, 50 = 最大倍率")
                        try:
                            level = int(input("  输入变倍级别 (1-50): "))
                            if 1 <= level <= 50:
                                # 假设变倍范围是1x到50x，将级别转换为倍率
                                mag = float(level)  # 1级=1x, 50级=50x
                                self.gimbal.zoom_to_magnification(mag)
                                print(f"✓ 变倍到级别 {level} (倍率: {mag}x)")
                            else:
                                print("✗ 级别必须在1-50之间")
                        except ValueError:
                            print("✗ 输入无效，请输入1-50的整数")
                    elif sub_cmd == '13':
                        # 自动对焦
                        self.gimbal.auto_focus()
                        print("✓ 自动对焦已触发")
                    elif sub_cmd == '14':
                        # 查询当前焦距和视场角
                        print("\n查询当前焦距和视场角...")
                        fov_x, fov_y, focal_x, focal_y = self.gimbal.query_focal_length()
                        if fov_x is not None and fov_y is not None:
                            print(f"✓ fov_x: {fov_x:.1f}°, fov_y: {fov_y:.1f}°")
                            print(f"✓ focal_length_x: {focal_x:.1f}mm, focal_length_y: {focal_y:.1f}mm")
                        else:
                            print("✗ 查询失败或超时")
                    elif sub_cmd == 'r':
                        self.gimbal.get_zoom_info()
                    elif sub_cmd == '0':
                        print("返回主菜单")
                    else:
                        print("✗ 无效选择")
                except ValueError:
                    print("✗ 输入无效，请输入数字")
                except Exception as e:
                    print(f"✗ 控制失败: {e}")
            
            elif cmd == 'a':
                # 自动瞄准模式
                print("\n" + "=" * 60)
                print("自动瞄准模式")
                print("=" * 60)
                try:
                    interval_str = input("  识别间隔（秒，回车=1.0）: ").strip()
                    interval = float(interval_str) if interval_str else 1.0

                    max_retries_str = input("  最大连续失败次数（回车=3）: ").strip()
                    max_retries = int(max_retries_str) if max_retries_str else 3
                except ValueError:
                    print("✗ 输入无效，使用默认值")
                    interval = 1.0
                    max_retries = 3

                print(f"\n启动自动瞄准: 间隔={interval}秒, 最大失败={max_retries}次")
                print("  按 Ctrl+C 停止自动瞄准，返回主菜单")
                print("=" * 60)

                # 调用自动瞄准（内部捕获 KeyboardInterrupt，返回交互模式）
                self.auto_aim(interval=interval, max_retries=max_retries)

                print("\n已退出自动瞄准模式，返回主菜单")
                print("=" * 60)

            else:
                print("✗ 无效命令")
    
    def start_video_stream(self, display_pose=False):
        """
        启动实时视频流显示
        
        参数:
            display_pose: 是否在视频上显示位姿信息
        """
        if self.video_running:
            print("⚠ 视频流已经在运行")
            return
        
        print("\n" + "-" * 60)
        print("启动实时视频流")
        print("-" * 60)
        
        # 打开RTSP流
        rtsp_url = self.rtsp_tv
        print(f"  连接RTSP流: {rtsp_url}")
        
        # 使用TCP协议以获得更好的稳定性
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        self.video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # 设置缓冲区大小（减少延迟）
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.video_capture.isOpened():
            print(f"  ✗ 无法打开RTSP流")
            self.video_capture = None
            return
        
        print(f"  ✓ 视频流已打开")
        print(f"  分辨率: {self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  帧率: {self.video_capture.get(cv2.CAP_PROP_FPS)} FPS")
        
        # 设置标志
        self.video_running = True
        self.video_paused = False
        self.show_pose_on_video = display_pose
        
        # 启动视频显示线程
        self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
        self.video_thread.start()
        
        print(f"\n  视频窗口已打开")
        print(f"  按 'q' - 退出视频")
        print(f"  按 'p' - 暂停/继续")
        print(f"  按 'c' - 捕获当前帧并估计位姿")
        print(f"  按 's' - 显示/隐藏位姿信息")
        print("-" * 60)
    
    def _video_display_loop(self):
        """视频显示循环（在线程中运行）"""
        window_name = "Gimbal Video Stream - for Viewing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Detection 窗口 - 按"v"启动视频流时就创建，复制推流数据
        detection_window_name = "detection Video"
        cv2.namedWindow(detection_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(detection_window_name, 1280, 720)
        
        frame_count = 0
        
        try:
            while self.video_running:
                if self.video_paused:
                    time.sleep(0.1)
                    continue
                
                # 读取帧
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    print("⚠ 视频帧读取失败，重试中...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                
                # 在视频上显示位姿信息（仅在原窗口显示文本）
                if self.show_pose_on_video and self.latest_pose_data is not None:
                    pose_data = self.latest_pose_data
                    
                    # 显示位姿文本信息
                    info_y = 30
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display_frame, f"Yaw: {pose_data.get('yaw', 0):.1f}°", 
                               (10, info_y), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Pitch: {pose_data.get('pitch', 0):.1f}°", 
                               (10, info_y + 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Dist: {pose_data.get('distance', 0):.2f}m", 
                               (10, info_y + 60), font, 0.7, (0, 255, 0), 2)
                
                # 显示帧计数和状态
                cv2.putText(display_frame, f"Frame: {frame_count}", 
                           (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.video_paused:
                    cv2.putText(display_frame, "PAUSED", 
                               (display_frame.shape[1] // 2 - 50, display_frame.shape[0] // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # 显示原视频帧（推流窗口）
                cv2.imshow(window_name, display_frame)
                
                # 如果处于 review 模式，显示带YOLO框的图像
                if self.review_mode and hasattr(self, 'review_frame_with_yolo'):
                    # 使用带YOLO框的图像（从image_init_detect保存）
                    detection_frame = self.review_frame_with_yolo.copy()
                    
                    # 在 detection 窗口上绘制坐标轴
                    if self.show_pose_on_video and self.latest_pose_data is not None:
                        pose_data = self.latest_pose_data
                        try:
                            H, W = detection_frame.shape[:2]
                            camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=self.fov_x)
                            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                            
                            # 使用最新的位姿数据绘制坐标轴
                            if 'rvec' in pose_data and 'tvec' in pose_data:
                                rvec = pose_data['rvec'].reshape(3, 1)
                                tvec = pose_data['tvec'].reshape(3, 1)
                                detection_frame = pose_estimation.draw_axes(
                                    detection_frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05
                                )
                        except Exception as e:
                            pass  # 忽略绘制错误
                    
                    # 显示提示信息
                    cv2.putText(detection_frame, "Press 'y' to reshoot, 'n' to continue", 
                               (10, detection_frame.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # 正常模式：复制推流数据到 detection Video 窗口
                    detection_frame = display_frame.copy()
                
                # 显示 detection 窗口（始终显示，复制推流数据）
                cv2.imshow(detection_window_name, detection_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    # 退出视频
                    break
                elif key == ord('p') or key == ord('P'):
                    # 暂停/继续
                    self.video_paused = not self.video_paused
                    print(f"\n{'暂停' if self.video_paused else '继续'}视频")
                elif key == ord('s') or key == ord('S'):
                    # 显示/隐藏位姿信息
                    self.show_pose_on_video = not self.show_pose_on_video
                    print(f"\n位姿信息显示: {'开启' if self.show_pose_on_video else '关闭'}")
                elif self.review_mode:
                    # 在 review 模式下，检查用户按键
                    if key == ord('y') or key == ord('Y'):
                        # 重新截图
                        print("\n  用户选择重新截图...")
                        self.review_mode = False
                        if hasattr(self, 'review_detections'):
                            delattr(self, 'review_detections')
                        # 设置标志，在下一轮循环中重新捕获
                        self.need_recapture = True
                    elif key == ord('n') or key == ord('N'):
                        # 继续
                        print("\n  用户选择继续...")
                        self.review_mode = False
                        if hasattr(self, 'review_detections'):
                            delattr(self, 'review_detections')
                
                # 检查是否需要重新捕获
                if hasattr(self, 'need_recapture') and self.need_recapture:
                    self.need_recapture = False
                    print(f"\n[视频] 重新捕获当前帧进行位姿估计...")
                    self._capture_and_estimate(frame)
                elif not self.review_mode and (key == ord('c') or key == ord('C')):
                    # 捕获当前帧并估计位姿
                    print(f"\n[视频] 捕获当前帧进行位姿估计...")
                    self._capture_and_estimate(frame)
        
        except Exception as e:
            print(f"✗ 视频显示错误: {e}")
        
        finally:
            # 关闭窗口
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(detection_window_name)
            print("\n视频流已停止")
    
    def _capture_and_estimate(self, frame):
        """
        捕获当前帧并进行位姿估计
        
        参数:
            frame: 当前视频帧
        """
        try:
            # 保存当前帧到临时文件
            temp_path = os.path.join(self.output_dir, "temp_capture.jpg")
            cv2.imwrite(temp_path, frame)
            
            # 进行位姿估计
            success, pose_data, result_image = self.estimate_pose_from_image(temp_path)
            
            if success:
                # 保存最新的位姿数据
                self.latest_pose_data = pose_data
                
                # 打印位姿信息
                self.print_pose_info(pose_data)
                
                # 保存结果
                timestamp = int(time.time())
                output_prefix = os.path.join(self.output_dir, f"video_pose_{timestamp}")
                self.save_result(result_image, pose_data, output_prefix)
                
                print(f"\n✓ 位姿估计完成（来自视频流）")
                
                # 读取刚保存的帧（确保使用正确的图像更新YOLO框）
                saved_frame = cv2.imread(temp_path)
                if saved_frame is not None:
                    # 启动 review 模式，在视频流上显示 YOLO 框
                    self._start_review_mode(saved_frame)
                else:
                    print(f"\n✗ 无法读取保存的帧: {temp_path}")
            else:
                print(f"\n✗ 位姿估计失败（来自视频流）")
        
        except Exception as e:
            print(f"✗ 捕获和估计失败: {e}")
    
    def _start_review_mode(self, frame):
        """
        启动 review 模式，在视频流上显示 YOLO 检测框
        
        参数:
            frame: 捕获的帧
        """
        print("\n" + "=" * 60)
        print("启动 Review 模式 - 请在视频窗口查看 YOLO 检测框")
        print("=" * 60)
        
        # 使用 YOLOv8 检测二维码
        try:
            detections = yolo_detection.detect_qr_yolov8(frame.copy(), self.detector)
            # 保存检测结果，供视频循环绘制使用
            self.review_detections = detections
            print(f"  检测到 {len(detections)} 个二维码")
            
            # 保存带YOLO框的图像，用于detection窗口显示
            if len(detections) > 0:
                # 在帧上绘制YOLO框
                frame_with_yolo = frame.copy()
                for det in detections:
                    polygon_xy = det['polygon_xy'].astype(np.float32)
                    conf = det['confidence']
                    
                    # 绘制多边形边界（红色）
                    pts = polygon_xy.reshape((-1, 1, 2))
                    cv2.polylines(frame_with_yolo, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                    
                    # 绘制角点（红色）
                    for pt in polygon_xy:
                        cv2.circle(frame_with_yolo, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                    
                    # 显示置信度（红色）
                    cv2.putText(frame_with_yolo, f"{conf:.2f}", 
                               (int(polygon_xy[0][0]), int(polygon_xy[0][1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 保存带YOLO框的图像
                self.review_frame_with_yolo = frame_with_yolo
                print(f"  ✓ 已保存带YOLO框的图像")
        except Exception as e:
            print(f"  ⚠ YOLO 检测失败: {e}")
            self.review_detections = []
        
        # 只有当检测到YOLO框时，才进入review模式
        if hasattr(self, 'review_detections') and len(self.review_detections) > 0:
            self.review_mode = True
            print("\n  提示: 在视频窗口按 'y' 重新截图，按 'n' 继续")
        else:
            # 没有检测到YOLO框，不进入review模式
            self.review_mode = False
            print("\n  ⚠ 未检测到YOLO框，不进入review模式")
            if hasattr(self, 'review_detections'):
                delattr(self, 'review_detections')
        
        print("=" * 60)
    
    def stop_video_stream(self):
        """停止视频流"""
        if not self.video_running:
            return
        
        print("\n停止视频流...")
        self.video_running = False
        
        # 等待线程结束
        if self.video_thread is not None:
            self.video_thread.join(timeout=2.0)
            self.video_thread = None
        
        # 释放VideoCapture
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("  ✓ 视频流已停止")
    
    def visualize_3d_scene(self, pose_records):
        """
        将等间隔的位姿数据、二维码及世界坐标系画在三维场景中
        
        参数:
            pose_records: list of pose_data dicts，每条记录包含:
                - object_points: 二维码3D角点
                - qr_size: 二维码物理尺寸（米）
                - camera_position: 相机在世界坐标系中的位置 [x, y, z]
                - rmat: 旋转矩阵 (3x3)
                - distance: 相机到原点距离（米）
                - yaw, pitch, roll: 欧拉角（度）
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("错误: 需要matplotlib库进行3D可视化")
            print("请安装: pip install matplotlib")
            return
        
        if not pose_records:
            print("没有位姿数据可供可视化")
            return
        
        print(f"\n正在生成3D场景可视化（共 {len(pose_records)} 个位姿）...")
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=45)
        
        # 从第一条记录获取二维码信息（所有记录应该一致）
        first_pose = pose_records[0]
        object_points_raw = np.array(first_pose['object_points'])
        qr_size = first_pose['qr_size']
        
        # 放大二维码平面使其在3D场景中更明显（保持世界坐标系比例不变）
        viz_scale = 5.0
        object_points = object_points_raw * viz_scale
        qr_center = np.mean(object_points, axis=0)
        
        # ======= 1. 绘制二维码平面（放大版） =======
        # 二维码四边
        vertices = object_points[[0, 1, 2, 3, 0]]
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                'k-', linewidth=2.5, label='QR Code Border')
        
        # 半透明填充二维码平面
        poly = Poly3DCollection([object_points], alpha=0.12, color='gray')
        ax.add_collection3d(poly)
        
        # 二维码中心标签
        ax.text(qr_center[0], qr_center[1], qr_center[2],
                'QR Code', fontsize=12, ha='center', color='black', weight='bold')
        
        # 标记四个角点（1=左上, 2=右上, 3=右下, 4=左下）
        corner_labels = ['1 (TL)', '2 (TR)', '3 (BR)', '4 (BL)']
        for i, pt in enumerate(object_points):
            ax.scatter(pt[0], pt[1], pt[2], c='black', s=50, marker='s')
            ax.text(pt[0], pt[1], pt[2] + object_points_raw.max() * 0.3, corner_labels[i],
                    fontsize=9, ha='center', color='black')
        
        # ======= 2. 绘制世界坐标系轴（二维码中心） =======
        world_axis_length = qr_size * viz_scale * 1.5
        ax.quiver(0, 0, 0, world_axis_length, 0, 0,
                  color='r', arrow_length_ratio=0.15, linewidth=1.5, label='World X')
        ax.quiver(0, 0, 0, 0, world_axis_length, 0,
                  color='g', arrow_length_ratio=0.15, linewidth=1.5, label='World Y')
        ax.quiver(0, 0, 0, 0, 0, -world_axis_length,
                  color='b', arrow_length_ratio=0.15, linewidth=1.5, label='World Z')
        
        # ======= 3. 绘制所有相机位置（渐变色轨迹） =======
        colors = plt.cm.jet(np.linspace(0.1, 0.9, len(pose_records)))
        all_positions = []
        
        for i, pose in enumerate(pose_records):
            cam_pos = np.array(pose['camera_position'])
            all_positions.append(cam_pos)
            rmat = np.array(pose['rmat'])
            color = colors[i]
            
            # 相机位置点（渐变颜色）
            ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                       c=[color], s=40, marker='o', alpha=0.8, zorder=5)
            
            # 标注位姿顺序编号（1, 2, 3...）
            avg_cam_pos = np.mean(np.abs(cam_pos))
            label_offset = max(object_points_raw.max(), avg_cam_pos) * 0.04
            ax.text(cam_pos[0] + label_offset, cam_pos[1] + label_offset, 
                    cam_pos[2] + label_offset,
                    str(i + 1), fontsize=7, ha='left', va='bottom',
                    color='black', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # 首尾两个位置画相机坐标系轴
            if i == 0 or i == len(pose_records) - 1:
                axis_len = qr_size * 0.3
                cam_x = rmat.T @ np.array([[1], [0], [0]])
                cam_y = rmat.T @ np.array([[0], [1], [0]])
                cam_z = rmat.T @ np.array([[0], [0], [1]])
                
                # 相机坐标系（X红=右, Y绿=下, Z蓝=前）
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_x[0, 0], cam_x[1, 0], cam_x[2, 0],
                          length=axis_len, color='r', arrow_length_ratio=0.25, alpha=0.6)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_y[0, 0], cam_y[1, 0], cam_y[2, 0],
                          length=axis_len, color='g', arrow_length_ratio=0.25, alpha=0.6)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_z[0, 0], cam_z[1, 0], cam_z[2, 0],
                          length=axis_len, color='b', arrow_length_ratio=0.25, alpha=0.6)
                
                label_text = 'Start' if i == 0 else 'End'
                marker = '^' if i == 0 else 's'
                ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                           c=[color], s=100, marker=marker, edgecolors='black', linewidth=1,
                           label=f'Camera ({label_text})', zorder=10)
                ax.text(cam_pos[0] + 0.01, cam_pos[1] + 0.01, cam_pos[2] + 0.01,
                        label_text, fontsize=9, ha='left', color=color)
        
        all_positions = np.array(all_positions)
        
        # ======= 4. 绘制相机移动轨迹连线 =======
        if len(all_positions) > 1:
            ax.plot(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                    'y--', linewidth=1.5, alpha=0.7, label='Camera Path')
        
        # ======= 5. 绘制视线（首帧和末帧到二维码中心的连线） =======
        ax.plot([all_positions[0, 0], qr_center[0]],
                [all_positions[0, 1], qr_center[1]],
                [all_positions[0, 2], qr_center[2]],
                'c:', alpha=0.4, linewidth=1, label='Start LOS')
        ax.plot([all_positions[-1, 0], qr_center[0]],
                [all_positions[-1, 1], qr_center[1]],
                [all_positions[-1, 2], qr_center[2]],
                'm:', alpha=0.4, linewidth=1, label='End LOS')
        
        # ======= 6. 设置图形属性 =======
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'QR Code & Camera 3D Scene ({len(pose_records)} poses)')
        
        # 设置坐标轴范围
        max_range = max(qr_size * 2, np.max(np.abs(all_positions))) * 1.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        
        # 信息文本
        first_dist = pose_records[0]['distance']
        last_dist = pose_records[-1]['distance']
        avg_dist = np.mean([p['distance'] for p in pose_records])
        
        info_text = (
            f"Total Poses: {len(pose_records)}\n"
            f"First Distance: {first_dist:.3f} m\n"
            f"Last Distance:  {last_dist:.3f} m\n"
            f"Avg Distance:   {avg_dist:.3f} m\n"
            f"QR Code Size:   {qr_size} m\n"
            f"Interval:       auto_aim sampling"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))
        
        plt.tight_layout()
        print("3D场景已生成，关闭窗口后程序继续...")
        plt.show()
    
    def auto_aim(self, interval=1.0, max_retries=3):
        """
        自动瞄准循环：等时间间隔识别二维码，估计位姿后自动旋转云台到对应位置
        
        参数:
            interval: 识别间隔时间（秒），默认 1.0 秒
            max_retries: 连续失败最大重试次数，超过则自动退出
        
        控制逻辑:
            1. 抓取可见光图像
            2. 二维码检测 + 位姿估计
            3. 计算云台目标角度:
               yam_camera_input  = pose_data['pitch']        (Yaw 目标)
               pitch_camera_input = pose_data['roll'] - 180.0  (Pitch 目标)
            4. 将目标角度直接发送给云台（不叠加当前角度）
            5. 等待 interval 秒后重复
        
        退出方式:
            按 Ctrl+C 停止循环
        """
        print("\n" + "=" * 60)
        print("自动瞄准模式")
        print("=" * 60)
        print(f"  识别间隔: {interval} 秒")
        print(f"  最大连续失败次数: {max_retries}")
        print(f"  提示: 按 Ctrl+C 停止")
        print("=" * 60)
        
        retry_count = 0
        pose_records = []  # 收集每次成功的位姿数据，用于退出时3D可视化
        
        try:
            while True:
                loop_start = time.time()
                
                # ===== 步骤1: 抓取图像 =====
                print(f"\n[{time.strftime('%H:%M:%S')}] 抓取图像...")
                success, image_path = self.capture_visible_image(use_rtsp=True)
                
                if not success:
                    retry_count += 1
                    print(f"  ✗ 图像抓取失败 ({retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        print(f"  ✗ 连续失败 {max_retries} 次，自动退出")
                        break
                    time.sleep(interval)
                    continue
                
                # ===== 步骤2: 位姿估计 =====
                print(f"  估计位姿...")
                success, pose_data, result_image = self.estimate_pose_from_image(
                    image_path, debug=False, debug_dir=None
                )
                
                if not success or pose_data is None:
                    retry_count += 1
                    print(f"  ✗ 位姿估计失败 ({retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        print(f"  ✗ 连续失败 {max_retries} 次，自动退出")
                        break
                    time.sleep(interval)
                    continue
                
                # 成功，重置失败计数，并记录位姿数据
                retry_count = 0
                pose_records.append(pose_data)  # 收集等间隔位姿数据，用于3D可视化

                #----------------------------------------

                cam_pos = pose_data['camera_position']
                print("\n")
                print("╔" + "═" * 56 + "╗")
                print("║" + " " * 16 + "📷  相 机 坐 标" + " " * 25 + "║")
                print("╠" + "═" * 56 + "╣")
                print(f"║  X (水平方向):  {cam_pos[0]:>10.4f}  m" + " " * 24 + "║")
                print(f"║  Y (垂直方向):  {cam_pos[1]:>10.4f}  m" + " " * 24 + "║")
                print(f"║  Z (深度方向):  {cam_pos[2]:>10.4f}  m" + " " * 24 + "║")
                print("╚" + "═" * 56 + "╝")
                print("")
                
                # ===== 步骤3: 计算云台目标角度 =====
                # 直接从位姿结果计算目标角度（不叠加云台当前角度）
                yaw_target   = pose_data['pitch']           # 相机 pitch → 云台 yaw
                pitch_target = pose_data['roll'] - 180.0    # 相机 roll  → 云台 pitch
                


                if(abs(pitch_target)>=180):
                    pitch_target = (360.0 + pitch_target)% 360.0

                # 修正角度
                gimbal_yaw, gimbal_pitch, _ = self.gimbal.get_current_angles()

                yaw_target += gimbal_yaw
                pitch_target += gimbal_pitch

                # ===== 步骤4: 发送云台控制命令 =====
                print(f"  目标角度: Yaw={yaw_target:.2f}°, Pitch={pitch_target:.2f}°")
                try:
                    self.gimbal.follow(yaw_target, pitch_target, 0.0)
                    print(f"  ✓ 云台控制命令已发送")
                except Exception as e:
                    print(f"  ✗ 云台控制失败: {e}")
                
                # ===== 步骤5: 等待到下一个周期 =====
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n自动瞄准已停止（用户中断）")
        
        finally:
            print("=" * 60)
            
            # ===== 退出时绘制3D场景可视化 =====
            if pose_records:
                print(f"\n共收集到 {len(pose_records)} 个等间隔位姿数据，正在绘制3D场景...")
                self.visualize_3d_scene(pose_records)
            else:
                print("\n无有效位姿数据，跳过3D可视化")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        # 停止视频流
        self.stop_video_stream()
        
        # 停止云台控制器
        if hasattr(self, 'gimbal'):
            self.gimbal.stop()
            print("  ✓ 云台控制器已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='云台位姿估计')
    parser.add_argument('--gimbal_ip', type=str, default='192.168.1.99',
                       help='云台IP地址 (默认: 192.168.1.99)')
    parser.add_argument('--gimbal_port', type=int, default=6001,
                       help='云台控制端口 (默认: 6001)')
    parser.add_argument('--qr_size', type=float, default=0.10,
                       help='二维码物理尺寸（米） (默认: 0.1)')
    parser.add_argument('--fov_x', type=float, default=60.0,
                       help='相机水平视场角（度） (默认: 60.0)')
    parser.add_argument('--model_size', type=str, default='s',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8模型大小 (默认: s)')
    parser.add_argument('--detector', type=str, default='opencv',
                       choices=['opencv', 'pyzbar'],
                       help='二维码检测器类型 (默认: opencv)')
    parser.add_argument('--square_size', type=int, default=800,
                       help='矫正后正方形图像边长 (默认: 800)')
    parser.add_argument('--conf_threshold', type=float, default=0.2,
                       help='YOLOv8置信度阈值 (默认: 0.2)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='YOLOv8 NMS阈值 (默认: 0.5)')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'interactive', 'video'],
                       help='运行模式: single=单次估计, interactive=交互模式, video=实时视频流 (默认: single)')
    parser.add_argument('--image', type=str, default=None,
                       help='直接使用指定图像进行位姿估计（跳过抓取步骤）')
    parser.add_argument('--video', action='store_true',
                       help='启动时自动打开实时视频流')
    parser.add_argument('--display_pose', action='store_true',
                       help='在视频上显示位姿信息')
    
    args = parser.parse_args()
    
    # 创建位姿估计器
    try:
        estimator = GimbalPoseEstimator(
            gimbal_ip=args.gimbal_ip,
            gimbal_port=args.gimbal_port,
            qr_size=args.qr_size,
            fov_x=args.fov_x,
            model_size=args.model_size,
            detector_type=args.detector,
            square_size=args.square_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        return
    
    try:
        if args.image is not None:
            # 使用指定图像进行位姿估计
            print(f"\n使用指定图像: {args.image}")
            success, pose_data, result_image = estimator.estimate_pose_from_image(args.image)
            
            if success:
                estimator.print_pose_info(pose_data)
                estimator.save_result(result_image, pose_data)
                print("\n✓ 位姿估计完成")
            else:
                print("\n✗ 位姿估计失败")
        
        elif args.mode == 'single':
            # 单次估计模式
            success, pose_data = estimator.run_single_estimation(save_result=True)
            if not success:
                print("\n✗ 位姿估计失败")
        
        elif args.mode == 'interactive':
            # 交互模式
            estimator.run_interactive_mode()
        
        elif args.mode == 'video':
            # 实时视频流模式
            print("\n启动实时视频流模式...")
            estimator.start_video_stream(display_pose=args.display_pose)
            
            # 等待视频流结束
            try:
                while estimator.video_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n用户中断")
            finally:
                estimator.stop_video_stream()
    
    finally:
        # 清理资源
        estimator.cleanup()


if __name__ == "__main__":
    main()
