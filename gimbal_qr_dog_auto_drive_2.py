#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两段式 QR 位姿到机器狗自动导航的启动入口。"""

import argparse
import threading
import time

import gimbal_pose_estimation_ascend as _gpea

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_dog_port_holder, _qrnav_estimator_holder,
)
from qrnav_stream import _qrnav_pose_sync_loop, _qrnav_live_pose_loop
from qrnav_web import start_combined_http_stream


def _build_parser():
    parser = argparse.ArgumentParser(
        description="两段式 QR 位姿 → 机器狗自动导航 (V2)")
    parser.add_argument("--gimbal_ip", type=str, default="192.168.1.99")
    parser.add_argument("--gimbal_port", type=int, default=6001)
    parser.add_argument("--qr_size", type=float, default=0.10)
    parser.add_argument("--fov_x", type=float, default=60.0)
    parser.add_argument("--model", type=str, default="./qr_yolov8.om")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--detector", type=str, default="opencv",
                        choices=["opencv", "pyzbar"])
    parser.add_argument("--square_size", type=int, default=800)
    parser.add_argument("--conf_threshold", type=float, default=0.2)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="video",
                        choices=["single", "interactive", "video"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--display_pose", action="store_true")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--dog-ip", type=str, default="192.168.1.103")
    parser.add_argument("--dog-port", type=int, default=43893)
    parser.add_argument("--source", type=str, default="gimbal",
                        choices=["gimbal", "cam"])
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _QR_NAV_CFG["FOV_X"] = args.fov_x
    _qrnav_dog_port_holder[0] = args.port

    print("=" * 60, flush=True)
    print("QR 位姿 → 机器狗 两段式自动导航 (V2)", flush=True)
    print("=" * 60, flush=True)
    print(f"  mode={args.mode}  model={args.model}", flush=True)
    print(f"  gimbal={args.gimbal_ip}:{args.gimbal_port}  "
          f"dog={args.dog_ip}:{args.dog_port}  HTTP={args.port}", flush=True)
    print("=" * 60, flush=True)

    try:
        estimator = _gpea.GimbalPoseEstimator(
            gimbal_ip=args.gimbal_ip,
            gimbal_port=args.gimbal_port,
            qr_size=args.qr_size,
            fov_x=args.fov_x,
            model_path=args.model,
            device_id=args.device_id,
            detector_type=args.detector,
            square_size=args.square_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
        )
    except Exception as exc:
        print(f"\n✗ 初始化 estimator 失败: {exc}", flush=True)
        return 1

    # 在启动 HTTP 前注入共享对象，消除页面刚打开时 provider/estimator 为空的窗口。
    _qrnav_estimator_holder[0] = estimator
    _gpea._http_dog_provider = estimator._handle_dog_action

    print("\n[DOG] 创建实例并静默起立 ...", flush=True)
    try:
        dog = estimator._get_dog()
        print(f"  ✓ dog={getattr(dog, '_addr', '?')}", flush=True)
        dog.stand_quiet()
        print("  ✓ 静默起立完成，可捕获 QR；执行横移会自动开始并在到位后停止", flush=True)
    except Exception as exc:
        # 狗离线不能阻止相机和网页启动，页面仍可用于诊断并手动重试。
        print(f"  ⚠ 静默起立失败: {exc}", flush=True)

    print("\n[HTTP] 启动合并服务器 ...", flush=True)
    try:
        start_combined_http_stream(port=args.port, owner=estimator)
    except Exception as exc:
        print(f"✗ HTTP 服务启动失败: {exc}", flush=True)
        try:
            estimator.cleanup()
        except Exception:
            pass
        return 1

    estimator._http_stream_started = True
    estimator._http_port = args.port
    try:
        estimator.qr_size = float(_QR_NAV_CFG.get("QR_SIZE_M", 0.12))
    except Exception as exc:
        print(f"  ⚠ QR 尺寸设置失败: {exc}", flush=True)

    print(f"  云台控制台: http://0.0.0.0:{args.port}/", flush=True)
    print(f"  页面2:      http://0.0.0.0:{args.port}/qr_nav2", flush=True)

    stop_evt = threading.Event()
    threading.Thread(
        target=_qrnav_pose_sync_loop,
        args=(estimator, stop_evt),
        daemon=True,
    ).start()

    if float(_QR_NAV_CFG.get("LIVE_POSE_FPS", 0.0) or 0.0) > 0:
        threading.Thread(
            target=_qrnav_live_pose_loop,
            args=(estimator, stop_evt),
            daemon=True,
        ).start()

    try:
        if args.mode == "video" or args.video or args.display_pose:
            print("\n[VIDEO] 启动视频流 ...", flush=True)
            estimator.start_video_stream(display_pose=True)
            while estimator.video_running:
                time.sleep(0.1)
        else:
            print(f"[VIDEO] mode={args.mode}，HTTP 服务保持运行", flush=True)
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n用户中断", flush=True)
    except Exception as exc:
        print(f"\n✗ 运行异常: {exc}", flush=True)
        return_code = 1
    else:
        return_code = 0
    finally:
        stop_evt.set()
        try:
            estimator.stop_video_stream()
        except Exception:
            pass
        try:
            estimator.cleanup()
        except Exception:
            pass
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
