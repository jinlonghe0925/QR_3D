#!/usr/bin/env python3
"""
云台控制+图像抓取Demo
- RTSP流抓图: ch0/stream0(主视频)、ch1/stream0(可见光)、ch2/stream0(红外)
- 云台角度控制: 输入yaw/pitch改变朝向
"""

import logging
import socket
import struct
import subprocess
import threading
import time
import os
from typing import Optional

import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# RTSP流地址配置
RTSP_TV = "rtsp://192.168.1.99:554/ch1/stream0"   # 可见光
RTSP_IR = "rtsp://192.168.1.99:554/ch2/stream0"   # 红外
RTSP_MAIN = "rtsp://192.168.1.99:554/ch0/stream0" # 主视频


class GimbalController:
    """云台控制器 (精简版)"""

    LEN_MODE = 0x1D
    LEN_CMD = 0x0A
    LOCAL_IP = ""
    LOCAL_PORT = 6000

    def __init__(self, ip="192.168.1.99", port=6001):
        self.target_ip = ip
        self.target_port = port
        self._running = True
        self._enabled = False

        # 图像抓取同步
        self._capture_lock = threading.Lock()
        self._capture_event = threading.Event()
        self._capture_buffer = b""
        self._capture_received = 0

        # 焦距查询同步
        self._focal_length_lock = threading.Lock()
        self._focal_length_event = threading.Event()
        self._focal_length_value = None  # 存储查询到的焦距值（mm）
        self._fov_x_value = None  # 存储查询到的水平视场角（度）
        self._fov_y_value = None  # 存储查询到的垂直视场角（度）

        # 伺服参数查询同步
        self._servo_params_lock = threading.Lock()
        self._servo_params_event = threading.Event()
        self._servo_params_data = None  # 存储查询到的伺服参数原始数据

        self._open_socket()

        # 存储当前云台角度 (yaw: 偏航, pitch: 俯仰, roll: 横滚)
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0

        # 使能云台并回零
        self._send_mode(b"\xaa")
        time.sleep(0.3)

        # 解锁所有轴（确保方位/俯仰/横滚都可控制）
        self.set_axis_lock(yaw_lock=False, pitch_lock=False, roll_lock=False)
        time.sleep(0.1)

        # 所有轴切换为闭环模式（电流环，可响应角度控制）
        self.set_loop_mode(axis=0x04, closed_loop=True)
        time.sleep(0.1)

        self.follow(0.0, 0.0, 0.0)
        time.sleep(0.3)
        self._enabled = True

        self._recv_thread = threading.Thread(target=self._recv_task, daemon=True)
        self._recv_thread.start()
        logger.info(f"云台控制器初始化，目标: {ip}:{port}")

    def _open_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((self.LOCAL_IP, self.LOCAL_PORT))
        except OSError:
            self.sock.bind(('', 0))
        logger.info(f"UDP已绑定: {self.sock.getsockname()}")

    @staticmethod
    def _calculate_crc(length: int, fid: int, payload: bytes) -> int:
        crc = 0x55 ^ 0xAA ^ length ^ fid
        for b in payload:
            crc ^= b
        return crc & 0xFF

    def _build_packet(self, length: int, fid: int, body: bytes) -> bytes:
        body_data = body.ljust(length - 1, b"\x00")[: length - 1]
        check = self._calculate_crc(length, fid, body_data)
        return struct.pack("!BBBB", 0x55, 0xAA, length, fid) + body_data + struct.pack("!B", check)

    def _send_mode(self, body: bytes):
        """发送模式帧 (fid=0x00, 固定长度 0x1D)"""
        self._sock_send(0x00, self.LEN_MODE, body)

    def _send_cmd(self, fid: int, body: bytes):
        """发送命令帧 (固定长度 0x0A)"""
        self._sock_send(fid, self.LEN_CMD, body)

    def _sock_send(self, fid: int, length: int, body: bytes):
        """发送协议包"""
        try:
            self.sock.sendto(self._build_packet(length, fid, body),
                             (self.target_ip, self.target_port))
        except Exception as e:
            logger.error(f"发送失败: {e}")

    # ==== 云台角度控制 ====

    def follow(self, yaw: float, pitch: float, roll: float):
        """
        云台角度控制: yaw偏航角(°), pitch俯仰角(°), roll横滚角(°)

        协议格式:
            dat1=0x03  角度控制模式
            7字节保留位
            yaw (float, 大端序, 4字节)
            pitch (float, 大端序, 4字节)
            roll (float, 大端序, 4字节)

        参数:
            yaw:   偏航角 (水平旋转)
            pitch: 俯仰角 (上下方向)
            roll:  横滚角 (翻滚，默认 0.0°)
        """
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_roll = roll

        # 发送角度控制命令 (dat1=0x03 + 保留位 + yaw + pitch + roll)
        self._send_mode(b"\x03" + b"\x00" * 7 + struct.pack("!fff", yaw, pitch, roll))

        logger.info(f"云台指向: yaw={yaw}°, pitch={pitch}°, roll={roll}°")

    def set_axis_lock(self, yaw_lock: bool = False, pitch_lock: bool = False, roll_lock: bool = False):
        """
        轴系锁定开关控制 (控制字 0x43)

        协议格式:
            dat1 = 0x43                  控制字 = 轴系锁定开关
            dat2-5 (unsigned int, 大端序) 锁定参数: 0x00000cba
                bit0 (a): 方位轴锁定, 1=锁定, 0=解锁
                bit1 (b): 俯仰轴锁定, 1=锁定, 0=解锁
                bit2 (c): 横滚轴锁定, 1=锁定, 0=解锁

        参数:
            yaw_lock:   True=锁定方位轴, False=解锁
            pitch_lock: True=锁定俯仰轴, False=解锁
            roll_lock:  True=锁定横滚轴, False=解锁
        """
        lock_value = 0x00000000
        if yaw_lock:
            lock_value |= 0x01    # bit0
        if pitch_lock:
            lock_value |= 0x02    # bit1
        if roll_lock:
            lock_value |= 0x04    # bit2

        self._send_mode(b"\x43" + struct.pack("!I", lock_value))

        status = []
        if yaw_lock:   status.append("方位锁定")
        else:          status.append("方位解锁")
        if pitch_lock: status.append("俯仰锁定")
        else:          status.append("俯仰解锁")
        if roll_lock:  status.append("横滚锁定")
        else:          status.append("横滚解锁")

        logger.info(f"轴系锁定设置: {', '.join(status)} (0x{lock_value:08X})")

    def set_loop_mode(self, axis: int = 0x04, closed_loop: bool = True):
        """
        开环闭环切换控制 (控制字 0x54)

        协议格式:
            dat1 = 0x54              控制字 = 开环闭环切换
            dat2 = axis              轴系: 0x01方位, 0x02俯仰, 0x03横滚, 0x04全部
            dat3 = mode              0x00=电流环(闭环, 默认), 0x01=开环

        参数:
            axis:        轴系选择 (1=方位, 2=俯仰, 3=横滚, 4=全部)
            closed_loop: True=电流环(闭环, 可响应角度控制), False=开环
        """
        mode = 0x00 if closed_loop else 0x01
        self._send_mode(b"\x54" + bytes([axis, mode]))

        axis_name = {0x01: "方位", 0x02: "俯仰", 0x03: "横滚", 0x04: "全部"}.get(axis, "未知")
        mode_name = "电流环(闭环)" if closed_loop else "开环"
        logger.info(f"轴系模式设置: {axis_name}轴 → {mode_name}")

    def diagnose(self):
        """
        云台轴系诊断：解锁所有轴、切换闭环模式、读取横滚安装参数
        解决 roll/yaw/pitch 某个轴不响应角度控制的问题
        """
        print("\n" + "=" * 50)
        print("云台轴系诊断")
        print("=" * 50)

        # 1. 解锁所有轴
        print("\n[1/4] 解锁所有轴...")
        self.set_axis_lock(yaw_lock=False, pitch_lock=False, roll_lock=False)
        time.sleep(0.2)

        # 2. 所有轴切换为闭环模式（电流环）
        print("[2/4] 所有轴切换为闭环模式...")
        self.set_loop_mode(axis=0x04, closed_loop=True)  # 0x04=全部轴
        time.sleep(0.2)

        # 3. 读取横滚安装参数
        print("[3/4] 读取横滚轴安装参数...")
        self.read_roll_install_params(timeout=2.0)
        time.sleep(0.2)

        # 4. 发送一个小的角度变化测试各轴
        print("[4/4] 发送测试角度 (yaw=0, pitch=0, roll=0)...")
        self.follow(0.0, 0.0, 0.0)
        time.sleep(0.5)

        print("\n✓ 诊断完成")
        print("  - 所有轴已解锁")
        print("  - 所有轴已切换为闭环模式")
        print("  - 如果某个轴仍不响应，可能是硬件不支持该轴")
        print("=" * 50)

    def get_current_angles(self):
        """
        获取当前云台角度

        返回:
            (yaw, pitch, roll) 元组，单位：度
        """
        return self.current_yaw, self.current_pitch, self.current_roll

    def read_roll_install_params(self, timeout: float = 2.0):
        """
        读取横滚轴安装参数 (控制字 0x55, 帧ID 0x0A)

        协议格式:
            帧头: 0x55 0xAA
            长度: 0x17 (23)
            帧ID: 0x0A
            body: 0x55 (控制字) + 0x00 * 21 (填充)

        返回数据包含:
            - 限位值 (正/负限位)
            - 编码器方向
            - 电机旋转方向
            - 陀螺安装方向和安装轴
            - 电机极对数
        """
        print("\n" + "=" * 50)
        print("读取横滚轴安装参数...")
        print("=" * 50)

        # 重置同步事件
        with self._servo_params_lock:
            self._servo_params_data = None
            self._servo_params_event.clear()

        # 发送读取命令 (帧ID=0x0A, 长度=0x17, 控制字=0x55)
        body = bytes([0x55]) + b"\x00" * 21  # 22字节body
        self._sock_send(fid=0x0A, length=0x17, body=body)
        print(f"  已发送: fid=0x0A, length=0x17, ctrl=0x55")

        # 等待响应
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._servo_params_event.wait(timeout=min(0.5, timeout - (time.time() - start_time))):
                with self._servo_params_lock:
                    data = self._servo_params_data
                if data:
                    print(f"  ✓ 收到响应 ({len(data)} 字节)")
                    # 打印原始十六进制
                    hex_str = ' '.join(f'{b:02X}' for b in data[:min(40, len(data))])
                    print(f"  原始数据: [{hex_str}]")

                    # 检查数据有效性（body第5字节开始是参数数据）
                    # data[4]=控制字, data[5:]=参数
                    if len(data) > 5:
                        params = data[5:]
                        non_zero = sum(1 for b in params if b != 0)
                        print(f"  参数字节: {len(params)} 字节, 非零字节: {non_zero} 个")

                        if non_zero > 0:
                            print("  ✓ 横滚轴安装参数有效（已配置）")
                        else:
                            print("  ✗ 横滚轴安装参数全为零（未配置）")
                    else:
                        print("  ✗ 响应数据太短")
                    print("=" * 50)
                    return True
            if time.time() - start_time >= timeout:
                break

        print("  ✗ 读取横滚安装参数超时")
        print("=" * 50)
        return False

    # ==== 可见光变倍/变焦/调焦控制 ====

    def zoom_in(self):
        """变倍+ (控制字 0x01)"""
        body = bytes([0x01]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("变倍+")

    def zoom_out(self):
        """变倍- (控制字 0x02)"""
        body = bytes([0x02]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("变倍-")

    def zoom_stop(self):
        """变倍停 (控制字 0x03)"""
        body = bytes([0x03]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("变倍停")

    def focus_wide(self):
        """广视场/短焦 (控制字 0x04)"""
        body = bytes([0x04]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("广视场(短焦)")

    def focus_narrow(self):
        """窄视场/长焦 (控制字 0x05)"""
        body = bytes([0x05]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("窄视场(长焦)")

    def zoom_to_focal_length(self, focal_length: float):
        """变焦到指定焦距 (控制字 0x06)
        参数:
            focal_length: 焦距值 (mm)
        """
        body = bytes([0x06]) + struct.pack("!f", focal_length) + b"\x00" * 5
        self._send_cmd(fid=0x01, body=body)
        logger.info(f"变焦到指定焦距: {focal_length}mm")

    def focus_near(self):
        """调焦+ (控制字 0x07)"""
        body = bytes([0x07]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("调焦+")

    def focus_far(self):
        """调焦- (控制字 0x08)"""
        body = bytes([0x08]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("调焦-")

    def focus_stop(self):
        """调焦停 (控制字 0x09)"""
        body = bytes([0x09]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("调焦停")

    def zoom_to_fov(self, fov: float):
        """变焦到指定视场角 (控制字 0x0A)
        参数:
            fov: 视场角 (度)
        """
        body = bytes([0x0A]) + struct.pack("!f", fov) + b"\x00" * 5
        self._send_cmd(fid=0x01, body=body)
        logger.info(f"变焦到指定视场角: {fov}°")

    def zoom_to_magnification(self, mag: float):
        """变倍到指定倍率 (控制字 0x0B)
        参数:
            mag: 倍率
        """
        body = bytes([0x0B]) + struct.pack("!f", mag) + b"\x00" * 5
        self._send_cmd(fid=0x01, body=body)
        logger.info(f"变倍到指定倍率: {mag}x")

    def auto_focus(self):
        """自动聚焦模式 (控制字 0x80)"""
        body = bytes([0x80]) + b"\x00" * 9
        self._send_cmd(fid=0x01, body=body)
        logger.info("自动对焦")

    def query_focal_length(self, timeout: float = 2.0) -> tuple:
        """
        查询当前焦距、视场角
        返回: (fov_x, fov_y, focal_length_x, focal_length_y)，如果查询失败返回(None, None, None, None)
        协议: 发送可见光状态查询命令 (帧ID=0x01, 长度=0x10, 控制字=0x10)
        返回数据格式:
            字节5-6: 视场H/L (有效值*10, unsigned short, MSB在前, 单位:度) - 水平视场角
            字节9-10: 焦距H/L (有效值*10, 单位mm)
        """
        print("查询当前焦距和视场角...")

        # 重置事件和值
        with self._focal_length_lock:
            self._focal_length_value = None
            self._fov_x_value = None
            self._fov_y_value = None
            self._focal_length_event.clear()

        # 发送可见光状态查询命令 (帧ID=0x01, 控制字=0x10)
        # 长度固定为0x10
        query_body = bytes([0x10]) + b"\x00" * 15  # 控制字 + 15字节填充
        self._send_cmd(fid=0x01, body=query_body)
        print("已发送状态查询命令 (控制字=0x10)")

        # 等待响应
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._focal_length_event.wait(timeout=min(0.5, timeout - (time.time() - start_time))):
                with self._focal_length_lock:
                    focal_length = self._focal_length_value
                    fov_x = self._fov_x_value
                    fov_y = self._fov_y_value
                print(f"✓ fov_x: {fov_x:.1f}°, fov_y: {fov_y:.1f}°")
                print(f"✓ focal_length_x: {focal_length:.1f}mm, focal_length_y: {focal_length:.1f}mm")
                return fov_x, fov_y, focal_length, focal_length
            if time.time() - start_time >= timeout:
                break

        print("✗ 查询焦距和视场角超时")
        return None, None, None, None

    def get_zoom_info(self):
        """读取焦距、变倍、调焦信息 (需要扩展协议，当前为占位方法)"""
        # 注意：协议中没有明确说明如何读取这些信息
        # 可能需要发送查询命令并解析返回数据
        # 这里先打印提示信息
        logger.info("读取焦距、变倍、调焦信息 (需要确认协议具体格式)")
        print("提示: 读取焦距、变倍、调焦信息功能需要根据具体协议实现")
        print("当前协议未明确说明读取命令格式，请参考设备文档")

    # ==== 图像抓取 ====

    def capture_image(self, channel: int = 0, output_path: str = "capture.jpg", timeout: float = 10.0) -> bool:
        """
        单幅图像抓取 (帧ID 0x04, 控制字 0x17)
        channel: 0=可见光TV, 1=红外IR
        """
        with self._capture_lock:
            self._capture_buffer = b""
            self._capture_received = 0
            self._capture_event.clear()

        body = bytes([0x17, channel])
        self._send_cmd(fid=0x04, body=body)
        logger.info(f"发送图像抓取命令, 通道={channel}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            remaining = timeout - (time.time() - start_time)
            if self._capture_event.wait(timeout=min(1.0, remaining)):
                with self._capture_lock:
                    data = self._capture_buffer
                if data:
                    jpg_start = data.find(b"\xff\xd8")
                    jpg_end = data.rfind(b"\xff\xd9")
                    if jpg_start >= 0 and jpg_end > jpg_start:
                        jpg_data = data[jpg_start:jpg_end + 2]
                        with open(output_path, 'wb') as f:
                            f.write(jpg_data)
                        logger.info(f"[成功] 图像已保存: {output_path} ({len(jpg_data)} bytes)")
                        return True
                    else:
                        with self._capture_lock:
                            self._capture_event.clear()
        logger.error("图像抓取超时")
        return False

    # ==== 后台接收 ====

    def _recv_task(self):
        """后台接收线程 - 处理图像抓取数据和查询响应"""
        self.sock.settimeout(0.5)
        while self._running:
            try:
                data, addr = self.sock.recvfrom(65535)
                if addr[0] != self.target_ip:
                    continue

                fid = -1
                if data[:2] == b"\x55\xAA" and len(data) >= 6:
                    fid = data[3]

                # 处理可见光状态信息包 (帧ID 0x01)
                # 协议格式: 信息头(0x55 0xAA), 长度(0x10), 帧ID(0x01), 状态字, 数据...
                # 字节4: 状态字
                # 字节5-6: 视场H/L (有效值*10, unsigned short, MSB在前, 单位:度) - 水平视场角
                # 字节7-8: 倍率H/L (有效值*10)
                # 字节9-10: 焦距H/L (有效值*10, 单位mm)
                if fid == 0x01 and len(data) >= 11:
                    with self._focal_length_lock:
                        status_word = data[4]
                        # 解析水平视场角 (字节5-6)
                        fov_h = data[5]
                        fov_l = data[6]
                        fov_x_raw = (fov_h << 8) | fov_l
                        self._fov_x_value = fov_x_raw / 10.0  # 水平视场角（度）
                        # 计算垂直视场角 (假设图像宽高比为16:9)
                        # fov_y = 2 * arctan(tan(fov_x/2) * (height/width))
                        import math
                        fov_x_rad = math.radians(self._fov_x_value)
                        aspect_ratio = 9.0 / 16.0  # height/width for 16:9
                        fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * aspect_ratio)
                        self._fov_y_value = math.degrees(fov_y_rad)  # 垂直视场角（度）
                        # 解析焦距 (字节9-10)
                        focal_h = data[9]
                        focal_l = data[10]
                        focal_raw = (focal_h << 8) | focal_l
                        focal_length_mm = focal_raw / 10.0  # 焦距（mm）
                        self._focal_length_value = focal_length_mm
                        # 计算 fx, fy (像素单位，假设图像分辨率为1920x1080)
                        # fx = fov_x 对应的焦距（像素）, fy = fov_y 对应的焦距（像素）
                        img_width = 1920  # 假设分辨率
                        img_height = 1080
                        self._focal_length_x = focal_length_mm  # 物理焦距 x（mm）
                        self._focal_length_y = focal_length_mm  # 物理焦距 y（mm），通常相同
                        self._focal_length_event.set()
                        logger.debug(f"收到可见光状态: 状态字=0x{status_word:02X}, fov_x={self._fov_x_value:.1f}°, fov_y={self._fov_y_value:.1f}°, 焦距={focal_length_mm:.1f}mm")

                if not self._capture_event.is_set():
                    is_image_frame = (fid == 0x04) or (len(data) > 200 and b"\xff\xd8" in data)
                    if is_image_frame:
                        with self._capture_lock:
                            self._capture_buffer += data
                            self._capture_received += 1
                        jpg_end = self._capture_buffer.rfind(b"\xff\xd9")
                        if jpg_end >= 0:
                            self._capture_event.set()
                            logger.info(f"检测到完整JPEG帧 (buffer={len(self._capture_buffer)}B)")

                # 处理伺服参数响应 (帧ID 0x0A)
                if fid == 0x0A and len(data) >= 5:
                    with self._servo_params_lock:
                        self._servo_params_data = data
                        self._servo_params_event.set()
                    ctrl_word = data[4]
                    logger.info(f"收到伺服参数响应: fid=0x{fid:02X}, ctrl=0x{ctrl_word:02X}, len={len(data)}")

                # 调试打印：未知小数据包（可能是控制响应）
                if len(data) <= 50 and data[:2] == b"\x55\xAA":
                    ctrl_word = data[4] if len(data) > 4 else -1
                    hex_str = ' '.join(f'{b:02X}' for b in data[:16])
                    logger.debug(f"收到控制响应: fid=0x{fid:02X}, ctrl=0x{ctrl_word:02X}, data=[{hex_str}...]")

            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"接收异常: {e}")

    def stop(self):
        """停止控制器"""
        self._running = False
        try:
            self.sock.close()
        except Exception:
            pass


# ==== RTSP流抓图 ====

def capture_from_rtsp(rtsp_url: str, output_path: str, timeout: int = 5) -> bool:
    """从RTSP/UDP流抓取一帧图像"""
    if rtsp_url.startswith("udp://") or rtsp_url.startswith("rtp://"):
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-protocol_whitelist", "file,udp,rtp",
                    "-i", rtsp_url,
                    "-vframes", "1",
                    "-y", output_path,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"[成功] 图像已保存: {output_path}")
                return True
            else:
                logger.error("抓图失败: ffmpeg 未收到数据")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"抓图超时 ({timeout}s)")
            return False
        except FileNotFoundError:
            logger.error("ffmpeg 未安装")
            return False

    cap = cv2.VideoCapture(
        rtsp_url, cv2.CAP_FFMPEG,
        params=[
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ],
    )

    if not cap.isOpened():
        logger.error(f"无法打开RTSP流: {rtsp_url}")
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_time = time.time()
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite(output_path, frame)
            cap.release()
            logger.info(f"[成功] 图像已保存: {output_path}")
            return True
        time.sleep(0.05)

    cap.release()
    logger.error(f"抓图超时 ({timeout}s)")
    return False


# ==== 主程序 ====

if __name__ == "__main__":
    import sys

    gimbal_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.99"
    gimbal_port = int(sys.argv[2]) if len(sys.argv) > 2 else 6001

    # 根据IP更新RTSP地址
    tv_url = f"rtsp://{gimbal_ip}:554/ch1/stream0"
    ir_url = f"rtsp://{gimbal_ip}:554/ch2/stream0"
    main_url = f"rtsp://{gimbal_ip}:554/ch0/stream0"

    print("=" * 50)
    print("  云台控制+图像抓取 Demo")
    print("=" * 50)
    print(f"  云台地址: {gimbal_ip}:{gimbal_port}")
    print(f"  可见光: {tv_url}")
    print(f"  红外:   {ir_url}")
    print(f"  主视频: {main_url}")
    print("=" * 50)

    gimbal = GimbalController(gimbal_ip, gimbal_port)

    try:
        print("\n" + "=" * 50)
        print("  交互菜单")
        print("=" * 50)
        print("  === 图像抓取 ===")
        print("  1. [RTSP] 抓取可见光图像 (TV)")
        print("  2. [RTSP] 抓取红外图像 (IR)")
        print("  3. [RTSP] 抓取主视频图像 (Main)")
        print("")
        print("  === 云台控制 ===")
        print("  g. 输入角度控制云台朝向 (yaw/pitch)")
        print("")
        print("  q. 退出")
        print("=" * 50)

        while True:
            cmd = input("\n请选择: ").strip()

            if cmd == 'q':
                break
            elif cmd == '1':
                print(f"\n[RTSP] 抓取可见光图像...")
                capture_from_rtsp(tv_url, f"capture_tv_{int(time.time())}.jpg")
            elif cmd == '2':
                print(f"\n[RTSP] 抓取红外图像...")
                capture_from_rtsp(ir_url, f"capture_ir_{int(time.time())}.jpg")
            elif cmd == '3':
                print(f"\n[RTSP] 抓取主视频图像...")
                capture_from_rtsp(main_url, f"capture_main_{int(time.time())}.jpg")
            elif cmd == 'g':
                print("\n=== 云台角度控制 ===")
                print("输入目标角度 (单位: 度)")
                print("  yaw:   偏航角 (水平旋转), 范围通常 -180~180")
                print("  pitch: 俯仰角 (上下方向), 范围通常 -90~90")
                print("  roll:  横滚角 (翻滚), 范围通常 -180~180 (回车=0)")
                try:
                    yaw_input = input("  yaw (°)   = ").strip()
                    pitch_input = input("  pitch (°) = ").strip()
                    roll_input = input("  roll (°)  = ").strip()
                    yaw = float(yaw_input)
                    pitch = float(pitch_input)
                    #roll = float(roll_input) if roll_input else 0.0
                    roll = float(roll_input)

                    gimbal.follow(yaw, pitch, roll)
                except ValueError:
                    print("输入无效，请输入数字")
            else:
                print("无效选择")

    finally:
        gimbal.stop()

    print("\n完成！")
