#!/usr/bin/env python3
"""AR 位姿 → 机器狗导航 (X30 Motion SDK)
============================================
调用 gimbal_pose_estimation_ascend 完成 QR 检测 + 位姿估计，
将 QR 坐标系中的 camera_position 转换为狗体相对位移，
通过 robot_interface.Dog.move() 下发相对运动指令。

坐标系转换 (二维码竖直安装):
  QR 坐标系: X=右, Y=上, Z=向板内(相机方向)
  Dog 坐标系: X=前, Y=右, Z=上
  转换:  Dog_X =  QR_Z   (前进距离)
         Dog_Y = -QR_X   (右侧距离)
         Dog_Z = -QR_Y   (高度，忽略)

依赖:
  - robot_interface.Dog: UDP 运动控制 (192.168.1.103:43893)
  - 机器狗需开机并连接同一局域网

用法:
  python robot_dog_bridge.py --image ./test.jpg --model ./qr_yolov8.om
  python robot_dog_bridge.py --mode single --model ./qr_yolov8.om
  python robot_dog_bridge.py --mode loop --model ./qr_yolov8.om
  python robot_dog_bridge.py --sim
"""

import sys
import time
import math
import os
import re
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gimbal_pose_estimation_ascend import GimbalPoseEstimator
from robot_interface import Dog


# ============================================================
# 读取 camera_intrinsics.txt
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
        Dog_X =  QR_Z    (前进距离: QR 在狗前方多远)
        Dog_Y = -QR_X    (右侧距离: QR 在狗右侧多远)
        Dog_Z = -QR_Y    (高度差)
    """
    if camera_position is None or len(camera_position) != 3:
        raise ValueError("camera_position 必须为 [x, y, z]")

    x_qr, y_qr, z_qr = (float(v) for v in camera_position)
    return z_qr, -x_qr, -y_qr


def compute_dog_move(x_dog: float, y_dog: float, z_dog: float,
                     stop_distance: float = 0.8) -> Optional[Tuple[float, float, float]]:
    """将狗体相对坐标转换为 Dog.move(x, y, z) 参数。

    Dog.move() 参数:
        x: 前进距离(m)，正=前进
        y: 侧移距离(m)，正=左移，负=右移
        z: 旋转角度(°)，正=右转

    返回 (dx, dy, dz) 或 None (已在安全距离内)。

    注意: 移动距离会减去 stop_distance，确保停在 QR 前方而非撞上去。
    """
    horizontal_dist = math.hypot(x_dog, y_dog)
    if horizontal_dist <= stop_distance:
        return None  # 已在安全距离内

    # 按比例缩减，使终点距 QR 正好 stop_distance
    ratio = (horizontal_dist - stop_distance) / horizontal_dist

    # 前进距离 = 狗体 X * ratio
    dx = x_dog * ratio
    # 侧移: 狗体 Y(右+) → SDK y(左+)，符号取反
    dy = -y_dog * ratio
    # 旋转角度: 朝向 QR 方向 (arctan2: 右/前)
    dz = math.degrees(math.atan2(y_dog, x_dog))

    return dx, dy, dz


# ============================================================
# X30 机器狗导航器
# ============================================================
class RobotDogNavigator:
    """X30 机器狗导航控制器

    使用 robot_interface.Dog 的 move() 相对运动指令，
    不需要 TDRobot / ROS2 / 地图导航。
    """

    def __init__(self, ip: str = '192.168.1.103', port: int = 43893,
                 auto_stand: bool = True, stop_distance: float = 0.8):
        self.stop_distance = stop_distance
        self._dog = Dog(ip=ip, port=port)
        self._ready = False
        self._auto_stand = auto_stand
        self._sim = False

    def connect(self) -> bool:
        """起立并准备运动 (约 3 秒)"""
        try:
            self._dog.init()
            self._ready = True
            print("[Dog] 已起立，运动就绪")
            return True
        except Exception as e:
            print(f"[Dog] 初始化失败: {e}")
            self._sim = True
            self._ready = True
            print("[Dog] 切换到模拟模式")
            return True

    def disconnect(self):
        """停止运动，关闭连接"""
        try:
            self._dog.close()
        except Exception:
            pass
        self._ready = False
        print("[Dog] 已断开")

    @property
    def ready(self) -> bool:
        return self._ready

    def go_to(self, x: float, y: float, z: float) -> bool:
        """相对移动: Dog.move(x=前进m, y=左移m, z=右转°)"""
        if not self._ready:
            print("[Dog] 未就绪")
            return False

        print(f"[Dog] move(x={x:.3f}m, y={y:.3f}m, z={z:.2f}°)")

        if self._sim:
            return self._sim_move(x, y, z)

        try:
            self._dog.move(x=x, y=y, z=z)
            print("[Dog] 到达 ✓")
            return True
        except Exception as e:
            print(f"[Dog] 移动失败: {e}")
            return False

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
    estimator: GimbalPoseEstimator,
    navigator: RobotDogNavigator,
    image_path: Optional[str] = None,
    debug: bool = False,
    camera_matrix: Optional[np.ndarray] = None,
) -> bool:
    """单次: 位姿估计 → 机器狗导航"""
    if image_path:
        print(f"\n[Pipeline] 图片模式: {image_path}")
        if camera_matrix is not None:
            estimator._override_camera_matrix = camera_matrix
        success, pose, result_img = estimator.estimate_pose_from_image(
            image_path, debug=debug)
    else:
        print("\n[Pipeline] 云台抓图模式 ...")
        success, pose = estimator.run_single_estimation(save_result=True)

    if not success or pose is None:
        print("[Pipeline] 位姿估计失败，跳过导航")
        return False

    estimator.print_pose_info(pose)
    return navigator.go_to_from_pose(pose)


# ============================================================
def main():
    import argparse

    p = argparse.ArgumentParser(description="AR位姿 → X30机器狗导航")
    # ── 云台 / 模型参数 ──
    p.add_argument("--gimbal-ip", default="192.168.1.99")
    p.add_argument("--gimbal-port", type=int, default=6001)
    p.add_argument("--qr-size", type=float, default=0.10, help="QR物理尺寸(m)")
    p.add_argument("--fov-x", type=float, default=60.0)
    p.add_argument("--model", default="./qr_yolov8.om", help="OM模型路径")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--detector", default="opencv", choices=["opencv", "pyzbar"])
    p.add_argument("--square-size", type=int, default=800)
    p.add_argument("--conf", type=float, default=0.2, help="YOLO置信度")
    p.add_argument("--nms", type=float, default=0.5, help="NMS阈值")
    # ── 输入源 ──
    p.add_argument("--image", default=None, help="图片路径(跳过抓图)")
    p.add_argument("--mode", default="single", choices=["single", "loop"],
                   help="single=单次, loop=循环")
    # ── 机器狗 (X30 Motion SDK) ──
    p.add_argument("--dog-ip", default="192.168.1.103", help="机器狗 IP")
    p.add_argument("--dog-port", type=int, default=43893, help="机器狗 UDP 端口")
    p.add_argument("--sim", action="store_true", help="模拟模式(不连真狗)")
    p.add_argument("--stop-distance", type=float, default=0.8,
                   help="机器狗与二维码的水平安全距离(m)，默认0.8")
    p.add_argument("--no-stand", action="store_true", help="不自动起立")
    p.add_argument("--debug", action="store_true", help="保存调试图像")
    p.add_argument("--camera-intrinsics", default=None,
                   help="camera_intrinsics.txt 路径（读取内参和QR参数）")
    args = p.parse_args()

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
    print("AR Pose → X30 Dog Pipeline")
    print("=" * 55)
    estimator = GimbalPoseEstimator(
        gimbal_ip=args.gimbal_ip, gimbal_port=args.gimbal_port,
        qr_size=args.qr_size, fov_x=args.fov_x,
        model_path=args.model, device_id=args.device_id,
        detector_type=args.detector, square_size=args.square_size,
        conf_threshold=args.conf, nms_threshold=args.nms,
    )

    # 2) 初始化机器狗导航
    if args.sim:
        navigator = RobotDogNavigator(
            ip=args.dog_ip, port=args.dog_port,
            auto_stand=False, stop_distance=args.stop_distance)
        navigator._ready = True
        navigator._sim = True
        print("[Dog] 模拟模式")
    else:
        navigator = RobotDogNavigator(
            ip=args.dog_ip, port=args.dog_port,
            auto_stand=not args.no_stand,
            stop_distance=args.stop_distance)
        if not navigator.connect():
            print("[P] 机器狗连接失败")
            return

    # 提取 camera_matrix（如果有）
    cam_mat = getattr(args, '_camera_matrix', None)

    try:
        if args.mode == "loop":
            print("\n[Loop] 按 Ctrl+C 停止\n")
            count = 0
            while True:
                count += 1
                print(f"\n{'─'*40}\n 第 {count} 轮\n{'─'*40}")
                pipeline(estimator, navigator, args.image, args.debug,
                         camera_matrix=cam_mat)
                time.sleep(2.0)
        else:
            pipeline(estimator, navigator, args.image, args.debug,
                     camera_matrix=cam_mat)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        navigator.disconnect()
        estimator.cleanup()


if __name__ == "__main__":
    main()
