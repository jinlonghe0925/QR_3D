#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO QR 检测器封装接口
================================================================
统一封装 ONNX / Ascend 两种后端的 YOLO QR 检测器，
为 robot_dog_bridge_win_v2.py 提供简洁的调用接口。

底层依赖: yolo_detection_ascend.py
    - .om 后缀 → Ascend NPU (ACL)
    - .onnx 后缀 → ONNX Runtime (CPU)

用法:
    from yolo_detector import YoloDetector

    # 从配置初始化
    detector = YoloDetector.from_config(app_config)

    # 或直接传参
    detector = YoloDetector(model_path="models/qr_yolov8.om",
                            conf_threshold=0.2, nms_threshold=0.5)

    # 检测
    detections = detector.detect(image_bgr)
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── 路径注入 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yolo_detection_ascend as _backend


class YoloDetector:
    """YOLO QR 检测器封装。

    根据模型后缀自动选择后端:
        .om   → Ascend NPU (ACL, Linux only)
        .onnx → ONNX Runtime (CPU, cross-platform)
    """

    # ── 默认值 ──
    DEFAULT_MODEL_PATH = "models/qr_yolov8.om"
    DEFAULT_CONF = 0.2
    DEFAULT_NMS = 0.5
    DEFAULT_INPUT_SIZE = (640, 640)

    def __init__(self,
                 model_path: Optional[str] = None,
                 conf_threshold: float = DEFAULT_CONF,
                 nms_threshold: float = DEFAULT_NMS,
                 device_id: int = 0,
                 input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
                 input_dtype: str = "float32"):
        """初始化 YOLO 检测器。

        参数:
            model_path:    模型路径 (.om 或 .onnx)
            conf_threshold: 置信度阈值
            nms_threshold:  NMS IoU 阈值
            device_id:      Ascend 设备 ID (仅 .om)
            input_size:     模型输入尺寸 (W, H)
            input_dtype:    输入数据类型 ('float32' / 'uint8')
        """
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._conf = conf_threshold
        self._nms = nms_threshold
        self._device_id = device_id
        self._input_size = input_size
        self._input_dtype = input_dtype
        self._backend_name = ""
        self._detector = None

        # ── 加载检测器 ──
        self._load()

    # ═══════════════════════════════════════════════════════════
    # 工厂方法
    # ═══════════════════════════════════════════════════════════
    @classmethod
    def from_config(cls, config: dict) -> "YoloDetector":
        """从 app_config.json 的 model 节创建检测器。

        参数:
            config: load_app_config() 返回的配置字典

        配置键:
            model.model_path    → 模型路径
            model.conf_threshold → 置信度阈值
            model.nms_threshold  → NMS 阈值
            model.device_id      → Ascend 设备 ID (可选)
        """
        model_cfg = config.get("model", {})
        input_size_raw = model_cfg.get("input_size", cls.DEFAULT_INPUT_SIZE)
        if isinstance(input_size_raw, list):
            input_size_raw = tuple(input_size_raw)
        return cls(
            model_path=model_cfg.get("model_path", cls.DEFAULT_MODEL_PATH),
            conf_threshold=model_cfg.get("conf_threshold", cls.DEFAULT_CONF),
            nms_threshold=model_cfg.get("nms_threshold", cls.DEFAULT_NMS),
            device_id=model_cfg.get("device_id", 0),
            input_size=input_size_raw,
            input_dtype=model_cfg.get("input_dtype", "float32"),
        )

    # ═══════════════════════════════════════════════════════════
    # 核心 API
    # ═══════════════════════════════════════════════════════════
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测图像中的二维码候选区域。

        参数:
            image: BGR 格式的 numpy 数组 (H, W, 3)

        返回:
            detections 列表，每项含:
                - polygon_xy:   (N, 2) 角点坐标
                - confidence:   置信度
                - bbox_xyxy:    (4,) 边界框 [x1, y1, x2, y2]
        """
        if self._detector is None:
            return []
        return self._detector.detect(image=image, is_bgr=True)

    # ═══════════════════════════════════════════════════════════
    # 属性
    # ═══════════════════════════════════════════════════════════
    @property
    def backend_name(self) -> str:
        """当前后端名称: 'Ascend (ACL)' 或 'ONNX Runtime (CPU)'。"""
        return self._backend_name

    @property
    def model_path(self) -> str:
        """模型文件路径。"""
        return self._model_path

    @property
    def conf_threshold(self) -> float:
        """置信度阈值。"""
        return self._conf

    @property
    def nms_threshold(self) -> float:
        """NMS 阈值。"""
        return self._nms

    @property
    def is_ready(self) -> bool:
        """检测器是否已加载成功。"""
        return self._detector is not None

    # ═══════════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════════
    def _load(self):
        """加载底层检测器实例。"""
        resolved = self._model_path
        ext = os.path.splitext(resolved)[1].lower()

        try:
            if ext == ".onnx":
                self._backend_name = "ONNX Runtime (CPU)"
                self._detector = _backend.OnnxYoloQRDetector(
                    model_path=resolved,
                    conf_threshold=self._conf,
                    nms_threshold=self._nms,
                    input_size=self._input_size,
                    input_dtype=self._input_dtype,
                )
            else:
                self._backend_name = "Ascend (ACL)"
                self._detector = _backend.AscendYoloQRDetector(
                    model_path=resolved,
                    conf_threshold=self._conf,
                    nms_threshold=self._nms,
                    device_id=self._device_id,
                    input_size=self._input_size,
                    input_dtype=self._input_dtype,
                )
            print(f"[{self._backend_name}] detector loaded: model={resolved} "
                  f"conf={self._conf} nms={self._nms}")
        except Exception as exc:
            self._detector = None
            self._backend_name = "FAILED"
            print(f"[ERROR] Failed to load detector: {exc}")

    def release(self):
        """释放检测器资源。"""
        if self._detector is not None:
            try:
                self._detector.release()
            except Exception:
                pass
            self._detector = None
