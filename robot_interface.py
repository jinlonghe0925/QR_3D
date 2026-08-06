"""
绝影 X30 运动接口模块 —— 供算法直接 import 调用
纯标准库，无任何第三方依赖

用法:
    from robot_interface import Dog

    dog = Dog(ip='192.168.1.103')
    dog.init()                     # 站立→手动模式→开始运动 (约3秒)
    dog.move(x=1.0, y=0, z=0)     # 前进 1 米
    dog.move(x=0, y=-0.5, z=45)   # 右移 0.5 米 + 右转 45°
    dog.close()                    # 停止并断开
"""

import socket
import struct
from time import sleep

# ======================= 配置（按需修改） =======================
ROBOT_IP   = '192.168.1.103'
ROBOT_PORT = 43893

# 满速标定参数（实测校准）
VMAX_X = 0.5    # 前进满速 (m/s)
VMAX_Y = 0.3    # 侧移满速 (m/s)
VMAX_Z = 60.0   # 旋转满速 (°/s)

# ======================= UDP 指令码 =======================
_HEARTBEAT    = 0x21040001
_STAND_UP     = 0x21010202
_MANUAL_MODE  = 0x21010C02
_START_MOTION = 0x21010201
_GAIT_WALK    = 0x21010300
_AXIS_LY = 0x21010130   # 前进/后退
_AXIS_LX = 0x21010131   # 左移/右移
_AXIS_RX = 0x21010135   # 左转/右转


def _u32(v):
    return int(v) & 0xFFFFFFFF


class Dog:
    """机器狗运动控制接口"""

    def __init__(self, ip=ROBOT_IP, port=ROBOT_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)

    def _send(self, code, value):
        self._sock.sendto(struct.pack('<3I', code, _u32(value), 0), self._addr)

    def init(self):
        """初始化狗到可运动状态（阻塞约 3 秒）"""
        self._send(_STAND_UP, 0);     sleep(1.5)
        self._send(_MANUAL_MODE, 0);   sleep(0.5)
        self._send(_START_MOTION, 0);  sleep(0.5)
        self._send(_GAIT_WALK, 0);     sleep(0.5)

    def move(self, x=0.0, y=0.0, z=0.0):
        """按距离/角度移动，三轴等比例调速同时到达

        Args:
            x: 前进距离 (米)，正=前进，负=后退
            y: 侧移距离 (米)，正=左移，负=右移
            z: 旋转角度 (度)，正=右转，负=左转

        Note:
            move 是阻塞调用，完成移动后才返回。
            精度取决于 VMAX_X/Y/Z 标定，实测后可校准。
        """
        tx = abs(x) / VMAX_X if abs(x) > 1e-6 else 0
        ty = abs(y) / VMAX_Y if abs(y) > 1e-6 else 0
        tz = abs(z) / VMAX_Z if abs(z) > 1e-6 else 0
        t = max(tx, ty, tz)
        if t < 0.01:
            return

        vx = max(-1, min(1, (x / VMAX_X) / t)) if tx > 0 else 0
        vy = max(-1, min(1, (y / VMAX_Y) / t)) if ty > 0 else 0
        vz = max(-1, min(1, (z / VMAX_Z) / t)) if tz > 0 else 0

        self._send(_AXIS_LY, int(vx * 32767))
        self._send(_AXIS_LX, int(vy * 32767))
        self._send(_AXIS_RX, int(vz * 32767))

        sleep(t)

        self._send(_AXIS_LY, 0)
        self._send(_AXIS_LX, 0)
        self._send(_AXIS_RX, 0)

    def close(self):
        """停止运动，关闭 UDP 连接"""
        self._send(_AXIS_LY, 0)
        self._send(_AXIS_LX, 0)
        self._send(_AXIS_RX, 0)
        self._sock.close()
