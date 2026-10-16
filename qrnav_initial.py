"""页面2模块1：捕获二维码平面姿态，并规划/执行水平朝向调整。"""
import copy
import math
import threading
import time

import cv2
import numpy as np


def compute_initial_heading(pose, yaw_sign=1.0, turn_sign=1.0):
    """PnP的R将二维码坐标变到相机系，R[:,2]是二维码平面法线。

    相机/导航方向约定：X右、Y下、Z前，绕Y正向为右转。
    视平面法线为相机Z轴；将二维码法线投影到相机XZ平面，
    atan2(nx, nz)即绕相机Y轴的有符号夹角。
    云台相对狗身变换仅用于诊断，不参与该夹角计算。
    初始云台与狗同向；保持云台角度，通过转狗调整偏航。
    不使用相机位置X/Z的视线方位来代替平面朝向。
    """
    if pose.get('rmat') is not None:
        rotation = np.asarray(pose['rmat'], dtype=float).reshape(3, 3)
    else:
        rotation, _ = cv2.Rodrigues(np.asarray(pose['rvec'], dtype=float).reshape(3, 1))
    if not np.isfinite(rotation).all():
        raise ValueError('Pose旋转矩阵包含非有限数值')
    normal = rotation[:, 2].copy()
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        raise ValueError('二维码平面法线无效')
    normal /= norm
    if pose.get('tvec') is not None:
        target = np.asarray(pose['tvec'], dtype=float).reshape(3)
    elif pose.get('camera_position') is not None:
        target = -rotation @ np.asarray(pose['camera_position'], dtype=float).reshape(3)
    else:
        target = np.array([0., 0., 1.])
    if not np.isfinite(target).all():
        raise ValueError('Pose位置包含非有限数值')
    # 平面法线有正反两种表示，选取朝向二维码所在一侧的方向。
    if float(np.dot(normal, target)) < 0:
        normal = -normal
    if math.hypot(normal[0], normal[2]) < 1e-6:
        raise ValueError('二维码法线接近相机Y轴，无法确定绕Y轴偏角')
    correction = math.degrees(math.atan2(normal[0], normal[2]))
    angles = [float(pose.get('gimbal_' + axis + '_deg', 0.0))
              for axis in ('yaw', 'pitch', 'roll')]
    if not all(math.isfinite(x) for x in angles):
        raise ValueError('捕获时的云台角度无效')
    yaw, pitch, roll = np.radians([angles[0] * yaw_sign, angles[1], angles[2]])
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch),
                             math.sin(pitch), math.cos(roll), math.sin(roll))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    camera_to_body = ry @ rx @ rz
    body_normal = camera_to_body @ normal
    normal_yaw = math.degrees(math.atan2(body_normal[0], body_normal[2]))
    return {
        'yaw_correction_deg': correction,
        'dog_turn_deg': correction * float(turn_sign),
        'direction': '右' if correction > 0 else ('左' if correction < 0 else '无需转向'),
        'plane_yaw_camera_deg': correction,
        'plane_yaw_body_deg': normal_yaw,
        'normal_camera': normal.tolist(), 'normal_body': body_normal.tolist(),
        'gimbal_yaw_deg': angles[0], 'gimbal_pitch_deg': angles[1],
        'gimbal_roll_deg': angles[2],
        'distance_m': float(np.linalg.norm(target)),
    }


class InitialLocalization:
    """捕获和执行分开；每份捕获结果只执行一次，转向后重新捕获验证。"""
    def __init__(self, capture_provider, estimator_provider, config_provider,
                 invalidate_navigation, sleeper=time.sleep):
        self.capture_provider = capture_provider
        self.estimator_provider = estimator_provider
        self.config_provider = config_provider
        self.invalidate_navigation = invalidate_navigation
        self.sleep = sleeper
        self._lock = threading.Lock()
        self._pose = None
        self._state = {'stage': 'idle', 'plan': None, 'message': '等待初始定位捕获',
                       'capture_ts': 0.0, 'can_execute': False}

    def state(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def _set(self, **values):
        with self._lock:
            self._state.update(values)

    def initialize_gimbal(self):
        self._pose = None
        self._set(stage='initializing', plan=None, can_execute=False,
                  capture_ts=0.0, message='正在初始化云台到 0° / 0° / 0°')
        try:
            estimator = self.estimator_provider()
            if estimator is None or estimator.gimbal is None:
                raise ValueError('云台未连接')
            estimator._gimbal_rotate(0.0, 0.0, 0.0)
            message = '已发送云台初始化指令：Yaw=0°、Pitch=0°、Roll=0°；待云台停稳后重新捕获Pose'
            self._set(stage='idle', message=message)
            return '✓ ' + message
        except Exception as exc:
            self._set(stage='failed', message=str(exc))
            return f'✗ 云台初始化失败: {exc}'

    def capture(self):
        self._set(stage='capturing', can_execute=False, plan=None, message='正在捕获二维码姿态')
        try:
            ok, pose, message, _ = self.capture_provider('模块1 初始定位')
            if not ok:
                raise ValueError(message)
            cfg = self.config_provider()
            # 用户约定初始云台和狗同向；保持云台角度，仅转狗做水平对齐。
            plan = compute_initial_heading(pose, cfg.get('YAW_SIGN', 1),
                                           cfg.get('INITIAL_TURN_SIGN', 1))
            tolerance = float(cfg.get('INITIAL_YAW_TOL_DEG', 2.0))
            plan['tolerance_deg'] = tolerance
            plan['needs_turn'] = abs(plan['yaw_correction_deg']) > tolerance
            self._pose = copy.deepcopy(pose)
            action = (f"计划向{plan['direction']}转" if plan['needs_turn'] else
                      f"在{tolerance:g}°容差内，无需转向")
            message = (f"初始定位捕获完成：视平面与二维码平面绕相机Y轴偏角 {plan['yaw_correction_deg']:+.2f}°，"
                       f"{action}；云台偏航 {plan['gimbal_yaw_deg']:+.2f}°")
            self._set(stage='captured', plan=plan, message=message,
                      capture_ts=time.time(), can_execute=True)
            return '✓ ' + message
        except Exception as exc:
            self._pose = None
            self._set(stage='failed', message=str(exc), can_execute=False)
            return f'✗ 初始定位捕获失败: {exc}'

    def execute(self):
        with self._lock:
            if not self._state['can_execute']:
                return '✗ 请先捕获模块1的二维码姿态；执行后需重新捕获'
            plan = copy.deepcopy(self._state['plan'])
            self._state.update(stage='executing', can_execute=False, message='正在调整水平朝向')
        dog = None
        try:
            estimator = self.estimator_provider()
            if estimator is None:
                raise ValueError('相机控制实例未就绪')
            # 不从capture1/2或实时Pose重新取值，严格执行页面展示的同一份计划。
            turn = float(plan['dog_turn_deg']) if plan['needs_turn'] else 0.0
            self.invalidate_navigation()
            if turn:
                dog = estimator._get_dog()
                dog.prepare_rl_walk()
                self.sleep(4.0)
                max_step = float(self.config_provider().get('INITIAL_TURN_STEP_DEG', 30.0))
                if max_step <= 0:
                    raise ValueError('单段转角必须大于0')
                segments = max(1, int(math.ceil(abs(turn) / max_step)))
                try:
                    for _ in range(segments):
                        if not dog.move(z=turn / segments, max_turn_deg=max_step):
                            raise RuntimeError('转向指令未完成发送')
                finally:
                    dog.stop_move()
            message = ('已发送水平转向并归零，请重新捕获复测偏角' if turn else
                       '水平偏角已在容差内，无需转动机器狗')
            self._set(stage='executed', message=message)
            return '✓ ' + message
        except Exception as exc:
            if dog is not None:
                try:
                    dog.stop_move()
                except Exception:
                    pass
            message = f'初始定位执行失败: {exc}；请重新捕获'
            self._set(stage='failed', message=message)
            return '✗ ' + message
