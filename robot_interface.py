"""
绝影 X30 运动接口模块 —— 供算法直接 import 调用
纯标准库，无任何第三方依赖

用法:
    from robot_interface import Dog

    dog = Dog(ip='192.168.1.103')
    dog.init()                     # 起立→手动模式→★力控→步态 (约3秒)
    dog.force_control()            # 仅补发「力控」使能 (对应遥控器同名按钮)
    dog.move(x=1.0, y=0, z=0)     # 前进 1 米 (内部 20Hz 流式下发, 像手柄)
    dog.move(x=0, y=-0.5, z=45)   # 右移 0.5 米 + 右转 45°
    dog.close()                    # 停止并断开

注意:
    狗必须先「起立」并「力控」使能后才能响应移动指令, 否则移动指令会被运控丢弃
    (网页按钮看似"失效"即此原因)。力控对应实体遥控器上的「力控」开关。
    dog.ready 属性可查询当前是否已使能。
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

# 流式下发参数（关键!）
# X30 运控要求轴指令像手柄一样「持续下发」，单发一帧会被安全看门狗丢弃，
# 表现就是"起立/趴下有效，移动无效"。默认 20Hz 持续下发。
STREAM_HZ     = 20.0     # 轴指令下发频率 (Hz)
AXIS_MAX      = 32767    # 摇杆满量程 (±32767)
STOP_FRAMES   = 8        # 松手后补发零速帧数 (确保被接收)
# ★ Bug 修复: 流内必须带心跳!
#   之前 HEARTBEAT_HZ=0.0 → stream 20Hz 持续 0.67s 全部被运控看门狗丢弃 → 狗不动
#   现在改为 5.0Hz → 每 4 帧 (200ms) 发一次 heartbeat → 运控保活
#   (短流如 13 帧也能保证至少 3-4 次心跳)
HEARTBEAT_HZ  = 5.0      # 流内附带心跳频率 (0=不发; 5.0=每 200ms 1 次)

# ======================= UDP 指令码 =======================
_HEARTBEAT    = 0x21040001
_STAND_UP     = 0x21010202   # 起立 (已验证)
_MANUAL_MODE  = 0x21010C02   # 手动模式
_FORCE_CONTROL = 0x21010205  # ★力控 (实体遥控器同名按钮, 移动前必发使能)
_START_MOTION = 0x21010201   # 备用使能 (若 FORCE_CONTROL 无效可换此码)
_GAIT_DEFAULT = 0x21010300   # 默认 gait (可能是「踏步」原地不动; 走路用 0x21010301/0x21010302 试)
_GAIT_TROT    = 0x21010301   # 备选 gait: 小跑/行走 (原 0x21010300 踏步时换此)
_GAIT_RUN     = 0x21010302   # 备选 gait: 奔跑
_AXIS_LY = 0x21010130   # 前进/后退 (虚拟摇杆 LY; 需 20Hz 流式)
_AXIS_LX = 0x21010131   # 左移/右移 (虚拟摇杆 LX)
_AXIS_RX = 0x21010135   # 左转/右转 (虚拟摇杆 RX)
_LIE_DOWN = 0x21010203  # 趴下复位 (已验证)
_DAMP     = 0x21010204  # 阻尼模式 (软趴, 备选)


def _u32(v):
    return int(v) & 0xFFFFFFFF


class Dog:
    """机器狗运动控制接口"""

    def __init__(self, ip=ROBOT_IP, port=ROBOT_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)
        self._ready = False     # 是否已起立且「开始运动」已使能

    def _send(self, code, value, debug=False):
        try:
            self._sock.sendto(struct.pack('<3I', code, _u32(value), 0), self._addr)
            if debug:
                # ★ 诊断模式: 打印每个发出去的 UDP 包
                print(f"[Dog] → UDP code=0x{code:08X} value={value} target={self._addr}")
        except OSError as e:
            # UDP sendto 失败 (网络不通/狗掉线) → 立即打印, 不吞掉
            print(f"[Dog] ✗ UDP 发送失败: code=0x{code:08X} value={value} → {e}")
            print(f"[Dog]   目标: {self._addr}, 请检查: 1)网线 2)狗 IP 3)防火墙")
            raise

    @property
    def ready(self):
        """是否已起立且「开始运动」已使能（可接收移动指令）"""
        return self._ready

    def init(self, gait=_GAIT_DEFAULT):
        """初始化狗到可运动状态（阻塞约 4 秒）

        顺序: 心跳注册 → 起立 → 手动模式 → ★力控 → ★开始运动 → ★gait。
        其中「力控 + 开始运动」即遥控器上的同名两个按钮, 都缺一不可:
          - 缺「力控」 → 移动指令被运控丢弃 (按钮失效)
          - 缺「开始运动」→ 狗完全不踏步 (腿不动)
          - 两者齐全 → 狗原地踏步; 切对 gait 后才能前进
        """
        self._send(_HEARTBEAT, 0, debug=True)                     # 先"注册"客户端
        self._send(_STAND_UP, 0, debug=True);     sleep(1.5)
        self._send(_MANUAL_MODE, 0, debug=True);   sleep(0.5)
        self._send(_FORCE_CONTROL, 0, debug=True); sleep(0.5)
        self._send(_START_MOTION, 0, debug=True);  sleep(0.5)
        self._send(gait, 0, debug=True);           sleep(0.5)
        self._ready = True
        self._gait = gait

    def force_control(self):
        """「力控」使能 (对应遥控器上的「力控」按钮)。阻塞约 1 秒

        第二步 (在 start_motion 之前必须先发), 让腿进入柔性模式。
        缺它: 移动指令被运控丢弃。
        """
        self._send(_MANUAL_MODE, 0, debug=True);   sleep(0.3)
        self._send(_FORCE_CONTROL, 0, debug=True); sleep(0.4)
        self._ready = True

    def stand_quiet(self):
        """「静默起立」: 起立+手动+力控, 但**不发** 开始运动+gait (腿保持静止)

        适用场景: 拍 QR / 拍照 / 视频采集时, 狗需要站着但**不能踏步**
        (踏步会让身体微抖, 拍出来的图像模糊 / 位姿抖动)。
        之后调用 start_motion() + set_gait(行走) 才会让狗真的能移动。
        """
        self._send(_HEARTBEAT, 0, debug=True)                     # 先"注册"客户端
        self._send(_STAND_UP, 0, debug=True);     sleep(1.5)
        self._send(_MANUAL_MODE, 0, debug=True);   sleep(0.5)
        self._send(_FORCE_CONTROL, 0, debug=True); sleep(0.5)
        # ★ 不发 START_MOTION 也不发 gait → 狗站着腿不动
        self._ready = True
        self._gait = None  # 标记未发 gait, 后续需主动调 start_motion + set_gait

    def start_motion(self):
        """「开始运动」使能 (对应遥控器上的同名按钮)。阻塞约 1 秒

        第三步 (在 force_control 之后发), 启动腿部步态循环。
        缺它: 狗完全不踏步; 两者齐全后狗才会「原地踏步」,
        此时再切对 gait 才能前进。

        ★ Bug 修复: 补发 MANUAL_MODE + FORCE_CONTROL + HEARTBEAT
          因为 stand_quiet 后 _ready=True, 但有些运控版本会"忘记"
          之前的力控状态, 在收到 AXIS 指令时仍当作"未力控"拒收。
          显式重发确保 100% 生效。
        """
        self._send(_HEARTBEAT, 0, debug=True)
        self._send(_MANUAL_MODE, 0, debug=True);   sleep(0.2)
        self._send(_FORCE_CONTROL, 0, debug=True); sleep(0.3)
        self._send(_START_MOTION, 0, debug=True);  sleep(0.5)
        gait = getattr(self, '_gait', _GAIT_DEFAULT)
        if gait is None:
            gait = _GAIT_DEFAULT
        self._send(gait, 0, debug=True);           sleep(0.3)
        self._ready = True
        self._gait = gait

    def set_gait(self, gait=_GAIT_DEFAULT):
        """切换步态 (解决「原地踏步不前进」)。

        备选码:
          0x21010300  踏步 (默认, 原地腿动 body 不前进)
          0x21010301  小跑/行走 (body 平移)
          0x21010302  奔跑
        """
        self._send(gait, 0, debug=True)
        sleep(0.3)
        self._gait = gait
        print(f"[Dog] 步态已切换: 0x{gait:08X}")

    def move(self, x=0.0, y=0.0, z=0.0, max_move_m=None, max_turn_deg=None, hz=STREAM_HZ):
        """按距离/角度移动，三轴等比例调速同时到达

        Args:
            x: 前进距离 (米)，正=前进，负=后退
            y: 侧移距离 (米)，正=左移，负=右移
            z: 旋转角度 (度)，正=右转，负=左转
            max_move_m:  单次平移安全上限 (米), None=不限。超出直接拒绝 (返回 False)
            max_turn_deg: 单次旋转安全上限 (度), None=不限
            hz: 轴指令流式下发频率 (默认 20Hz)

        Returns:
            True=已执行, False=被安全锁拒绝 (未执行任何动作)

        Note:
            move 是阻塞调用，完成移动后才返回。
            关键: 轴指令必须「持续下发」(类似手柄), 单发一帧会被运控看门狗丢弃,
            因此这里在 t 秒内以 hz 频率反复发送, 结束后再补发若干零速帧。
            精度取决于 VMAX_X/Y/Z 标定，实测后可校准。
        """
        # ── 安全锁: 限制单次移动量 ──
        if max_move_m is not None and max(abs(x), abs(y)) > max_move_m + 1e-9:
            print(f"[Dog] ✗ 平移量超限: max(|x|,|y|)={max(abs(x),abs(y)):.3f} > {max_move_m} m")
            return False
        if max_turn_deg is not None and abs(z) > max_turn_deg + 1e-9:
            print(f"[Dog] ✗ 旋转量超限: |z|={abs(z):.1f} > {max_turn_deg}°")
            return False

        tx = abs(x) / VMAX_X if abs(x) > 1e-6 else 0
        ty = abs(y) / VMAX_Y if abs(y) > 1e-6 else 0
        tz = abs(z) / VMAX_Z if abs(z) > 1e-6 else 0
        t = max(tx, ty, tz)
        if t < 0.01:
            return True

        vx = max(-1, min(1, (x / VMAX_X) / t)) if tx > 0 else 0
        vy = max(-1, min(1, (y / VMAX_Y) / t)) if ty > 0 else 0
        vz = max(-1, min(1, (z / VMAX_Z) / t)) if tz > 0 else 0

        self.stream(vx=vx, vy=vy, vz=vz, duration=t, hz=hz)
        return True

    def stream(self, vx=0.0, vy=0.0, vz=0.0, duration=1.0, hz=STREAM_HZ,
               stop_after=True, hb_hz=HEARTBEAT_HZ):
        """以 hz 频率持续下发三轴速度指令 duration 秒 (模拟手柄按住)。

        Args:
            vx/vy/vz: 归一化速度, 范围 [-1, 1]
            duration: 持续秒数
            hz:       下发频率 (Hz)
            stop_after: 结束后是否补发零速帧 (松手)
            hb_hz:    流内附带心跳频率 (0=不发)
        """
        hz = max(1.0, float(hz))
        period = 1.0 / hz
        n = max(1, int(round(float(duration) * hz)))
        hb_every = int(round(hz / hb_hz)) if hb_hz and hb_hz > 0 else 0
        a = int(max(-1.0, min(1.0, vx)) * AXIS_MAX)
        b = int(max(-1.0, min(1.0, vy)) * AXIS_MAX)
        c = int(max(-1.0, min(1.0, vz)) * AXIS_MAX)
        # ★ Bug 修复: 流开始先发 1 次 heartbeat, 运控收到心跳才"开门"接收控制流
        self._send(_HEARTBEAT, 0, debug=True)
        # ★ 诊断: 打印归一化速度 + 实际字节值
        print(f"[Dog] stream: n={n} frames, vx={vx:+.2f}→a={a}, vy={vy:+.2f}→b={b}, vz={vz:+.2f}→c={c}")
        for i in range(n):
            if i == 0 or i == n-1:  # 只 print 首末帧, 避免刷屏
                self._send(_AXIS_LY, a, debug=True)
                self._send(_AXIS_LX, b, debug=True)
                self._send(_AXIS_RX, c, debug=True)
            else:
                self._send(_AXIS_LY, a)
                self._send(_AXIS_LX, b)
                self._send(_AXIS_RX, c)
            if hb_every and i % hb_every == 0:
                self._send(_HEARTBEAT, 0, debug=True)
            sleep(period)
        # 流结束再发 1 次 heartbeat, 让运控识别"流结束"
        self._send(_HEARTBEAT, 0, debug=True)
        if stop_after:
            self.stop_move(hz=hz)

    def stop_move(self, hz=STREAM_HZ):
        """停止移动: 补发若干零速帧 (松手), 确保运控收到并刹住"""
        period = 1.0 / max(1.0, float(hz))
        # 先发 heartbeat 让运控"开门", 再发零速帧
        self._send(_HEARTBEAT, 0, debug=True)
        print(f"[Dog] stop_move: 发 {STOP_FRAMES} 帧零速 (LY/LX/RX = 0/0/0)")
        for _ in range(STOP_FRAMES):
            self._send(_AXIS_LY, 0)
            self._send(_AXIS_LX, 0)
            self._send(_AXIS_RX, 0)
            sleep(period)
        self._send(_HEARTBEAT, 0)

    def raw(self, code, value=0, n=1, period=0.05):
        """裸指令发送 (排查用): 把 code/value 重复发送 n 次, 间隔 period 秒"""
        for _ in range(max(1, int(n))):
            self._send(code, value)
            if n > 1:
                sleep(period)

    def stand_down(self):
        """趴下复位 (站立 → 卧姿)。阻塞约 2 秒"""
        self.stop_move()
        self._send(_LIE_DOWN, 0)
        self._ready = False
        sleep(2.0)

    def damp(self):
        """阻尼模式 (关节软掉, 可直接搬动)。阻塞约 1 秒"""
        self.stop_move()
        self._send(_DAMP, 0)
        self._ready = False
        sleep(1.0)

    def close(self, stand_down=True):
        """停止运动并断开。

        Args:
            stand_down: True=先趴下复位再断开 (默认); False=只停运动
        """
        self.stop_move()
        if stand_down:
            self.stand_down()
        self._ready = False
        self._sock.close()
