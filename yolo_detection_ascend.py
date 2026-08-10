#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend OM backend for YOLO QR detection.
=============================================================================
模型转换步骤:
  1. 下载 qrdet 的 .pt 模型 (运行一次 QRDetector('s') 自动缓存到本地)
  2. pt → onnx (在 PC 上执行):
       from ultralytics import YOLO
       model = YOLO("qrdet-s.pt")
       model.export(format="onnx", imgsz=640, opset=11, simplify=True, batch=1, dynamic=False)
  3. onnx → om (在 Ascend 开发环境执行):
       atc --model=qr_yolov8.onnx --framework=5 --output=qr_yolov8 \\
           --input_format=NCHW --input_shape="images:1,3,640,640" \\
           --soc_version=Ascend310P3 --log=error
  4. 将生成的 qr_yolov8.om 放到 models/ 目录下
=============================================================================

Public API:
    - load_yolov8_detector(...)
    - detect_qr_yolov8(image, detector)

Default OM location:
    ./models/qr_yolov8.om

Override with:
    - model_path argument
    - ASCEND_QR_MODEL_PATH environment variable
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "qr_yolov8.om"


# ── 工具函数 ─────────────────────────────────────────────────────────
def _check_acl(ret: int, action: str) -> None:
    if ret != 0:
        raise RuntimeError(f"ACL failed during {action}, ret={ret}")


def _letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = new_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    resize_w = int(round(src_w * scale))
    resize_h = int(round(src_h * scale))
    resized = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    pad_w = dst_w - resize_w
    pad_h = dst_h - resize_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (float(left), float(top))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


# ── 共享后处理函数（ONNX 和 Ascend 共用）────────────────────────────
def _postprocess_seg_impl(
    squeezed: np.ndarray, outputs: "List",
    original_hw: Tuple[int, int], meta: Dict[str, Any],
    input_size: Tuple[int, int], conf_threshold: float, nms_threshold: float,
) -> "List[Dict[str, Any]]":
    """处理原始 seg 输出: (channels, preds), 含掩码系数"""
    channels = squeezed.shape[0]
    if channels < 5:
        return []

    full_pred = squeezed.T.astype(np.float32, copy=False)
    scores = full_pred[:, 4].copy()
    conf_mask = scores >= conf_threshold
    indices = np.where(conf_mask)[0]
    if len(indices) == 0:
        return []

    filtered = full_pred[indices]
    scores = scores[indices]

    box = filtered[:, :4].copy()
    input_w, input_h = input_size
    if np.nanmax(box) <= 2.0:
        box[:, [0, 2]] *= input_w
        box[:, [1, 3]] *= input_h
    cx, cy, bw, bh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    scale = float(meta["scale"])
    pad_x, pad_y = meta["pad"]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    orig_h, orig_w = original_hw
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h - 1)

    keep = _nms(xyxy, scores, nms_threshold)
    if len(keep) == 0:
        return []

    proto = None
    has_mask = (channels > 6 and len(outputs) > 1)
    if has_mask:
        proto = np.asarray(outputs[1], dtype=np.float32)
        if proto.ndim == 4:
            proto = proto[0]

    detections: List[Dict[str, Any]] = []
    for k in keep:
        x1, y1, x2, y2 = xyxy[k]
        if x2 <= x1 or y2 <= y1:
            continue
        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        if has_mask and proto is not None and filtered.shape[1] >= 37:
            mask_coeffs = filtered[k, 5:37].astype(np.float32)
            bbox_model = np.array([
                filtered[k, 0], filtered[k, 1], filtered[k, 2], filtered[k, 3],
            ], dtype=np.float32)
            if np.nanmax(bbox_model) <= 2.0:
                bbox_model[[0, 2]] *= input_w
                bbox_model[[1, 3]] *= input_h
            cx_m, cy_m, bw_m, bh_m = bbox_model
            bbox_nms = np.array([
                cx_m - bw_m / 2, cy_m - bh_m / 2,
                cx_m + bw_m / 2, cy_m + bh_m / 2,
            ], dtype=np.float32)
            rotated = _compute_rotated_polygon_impl(
                bbox_nms, mask_coeffs, proto, original_hw, meta, input_size,
            )
            if rotated is not None and len(rotated) == 4:
                polygon = rotated.astype(np.float32)

        detections.append({
            "polygon_xy": polygon,
            "confidence": float(scores[k]),
            "bbox_xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
        })
    return detections


def _compute_rotated_polygon_impl(
    bbox_nms: np.ndarray, mask_coeffs: np.ndarray,
    proto: np.ndarray, original_hw: Tuple[int, int], meta: Dict[str, Any],
    input_size: Tuple[int, int],
) -> "Optional[np.ndarray]":
    """利用分割输出计算旋转四边形角点"""
    try:
        x1, y1, x2, y2 = bbox_nms.astype(np.float32)
        if x2 <= x1 or y2 <= y1:
            return None

        mask = np.tensordot(mask_coeffs, proto, axes=([0], [0]))
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -20, 20)))

        scale_m = 160.0 / input_size[0]
        mx1 = int(np.clip(x1 * scale_m, 0, 159))
        my1 = int(np.clip(y1 * scale_m, 0, 159))
        mx2 = int(np.clip(x2 * scale_m, 0, 160))
        my2 = int(np.clip(y2 * scale_m, 0, 160))
        if mx2 <= mx1 + 2 or my2 <= my1 + 2:
            return None
        crop = mask[my1:my2, mx1:mx2]
        if crop.size == 0:
            return None

        scale_orig = float(meta["scale"])
        pad_x, pad_y = meta["pad"]
        ow = max(int((x2 - x1) / scale_orig), 32)
        oh = max(int((y2 - y1) / scale_orig), 32)
        crop_r = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
        crop_bin = (crop_r > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) < 4:
            hull = cv2.convexHull(largest)
            if len(hull) >= 4:
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            else:
                return None

        corners = approx.reshape(-1, 2).astype(np.float32)
        bbox_x1_orig = (x1 - pad_x) / scale_orig
        bbox_y1_orig = (y1 - pad_y) / scale_orig
        corners[:, 0] = bbox_x1_orig + corners[:, 0] / ow * ((x2 - x1) / scale_orig)
        corners[:, 1] = bbox_y1_orig + corners[:, 1] / oh * ((y2 - y1) / scale_orig)
        orig_h, orig_w = original_hw
        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h - 1)

        if len(corners) > 4:
            hull_pts = cv2.convexHull(corners.astype(np.int32)).reshape(-1, 2).astype(np.float32)
            if len(hull_pts) >= 4:
                s = hull_pts[:, 0] + hull_pts[:, 1]
                d = hull_pts[:, 0] - hull_pts[:, 1]
                corners = np.array([
                    hull_pts[np.argmin(s)], hull_pts[np.argmax(d)],
                    hull_pts[np.argmax(s)], hull_pts[np.argmin(d)],
                ], dtype=np.float32)
        if len(corners) >= 4:
            return corners[:4]
    except Exception:
        pass
    return None


class OnnxYoloQRDetector:
    """基于 ONNX Runtime 的 YOLO 二维码检测器（CPU 对比测试用）。

    使用与 AscendYoloQRDetector 完全相同的前后处理流水线:
        binary threshold → letterbox → normalize → NCHW tensor
    ONNX 推理 → 相同的 reshape/decode/NMS 后处理。

    用于对比验证: OM 模型无效果时，先跑 ONNX 确认预处理管线是否正确。
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.2,
        nms_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        input_dtype: str = "float32",
        apply_binary_preprocess: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = input_size
        self.input_dtype = input_dtype
        self.apply_binary_preprocess = apply_binary_preprocess

        self._session = None
        self._input_name = None
        self._output_names = None

    def _ensure_session(self):
        if self._session is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("需要 onnxruntime: pip install onnxruntime")
        self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._output_shapes = [o.shape for o in self._session.get_outputs()]
        print(f"[ONNX] loaded: {self.model_path}  inputs={[self._input_name]}  "
              f"input_shape={self._session.get_inputs()[0].shape}")
        for name, shape in zip(self._output_names, self._output_shapes):
            print(f"       output '{name}': shape={shape}")

    # ── 与 AscendYoloQRDetector 完全相同的处理管线 ───────────────────
    def detect(self, image: np.ndarray, is_bgr: bool = True) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []
        self._ensure_session()
        input_tensor, meta = self._preprocess(image, is_bgr=is_bgr)
        outputs = self._infer(input_tensor)
        return self._postprocess(outputs, image.shape[:2], meta)

    def _preprocess(
        self, image: np.ndarray, is_bgr: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        working = np.ascontiguousarray(image)  # 确保内存连续 (多线程安全)
        if self.apply_binary_preprocess:
            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            working = np.stack([binary] * 3, axis=-1)
        elif not is_bgr:
            working = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)

        letterboxed, scale, pad = _letterbox(working, self.input_size)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        if self.input_dtype == "uint8":
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...].astype(np.uint8))
        else:
            tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            tensor = np.ascontiguousarray(tensor)
        return tensor, {"scale": scale, "pad": pad}

    def _infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        feeds = {self._input_name: input_tensor}
        results = self._session.run(self._output_names, feeds)
        return [np.asarray(r, dtype=np.float32) for r in results]

    def _postprocess(
        self, outputs, original_hw, meta
    ) -> List[Dict[str, Any]]:
        if not outputs:
            return []
        raw = outputs[0].astype(np.float32, copy=False)
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2 and squeezed.shape[1] in (5, 6, 7, 38):
            return self._postprocess_nms(squeezed, outputs, original_hw, meta)
        return _postprocess_seg_impl(squeezed, outputs, original_hw, meta,
                                     self.input_size, self.conf_threshold, self.nms_threshold)

    def _compute_rotated_polygon(
        self, bbox_nms, mask_coeffs, proto, original_hw, meta,
    ) -> Optional[np.ndarray]:
        return _compute_rotated_polygon_impl(
            bbox_nms, mask_coeffs, proto, original_hw, meta, self.input_size)

    def _postprocess_nms(
        self, pred: np.ndarray, outputs: List[np.ndarray],
        original_hw: Tuple[int, int], meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """处理 nms=True 导出的 ONNX 输出"""
        # 过滤零行
        mask_valid = (pred[:, :4].sum(axis=1) > 0) & (pred[:, 4] >= self.conf_threshold)
        pred = pred[mask_valid]
        if len(pred) == 0:
            return []
        boxes = pred[:, :4].copy()
        scores = pred[:, 4].copy()
        has_mask = (pred.shape[1] > 6)

        # 获取原型掩码（如果存在）
        proto = None
        if has_mask and len(outputs) > 1:
            proto = np.asarray(outputs[1], dtype=np.float32)
            if proto.ndim == 4:
                proto = proto[0]  # (32, 160, 160)

        # 坐标映射：模型空间 → 原图
        scale = float(meta["scale"])
        pad_x, pad_y = meta["pad"]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        orig_h, orig_w = original_hw
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        detections: List[Dict[str, Any]] = []
        for i in range(len(pred)):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue

            # 默认 polygon = 轴对齐 bbox
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            # 尝试从分割输出计算旋转四边形（匹配 qrdet 原始接口）
            if has_mask and proto is not None:
                # pred[i, 0:4] 已在模型空间 (0~640)，无需转换
                bbox_nms = pred[i, :4].astype(np.float32)
                rotated = self._compute_rotated_polygon(
                    bbox_nms, pred[i, 6:38].astype(np.float32),
                    proto, original_hw, meta,
                )
                if rotated is not None and len(rotated) == 4:
                    polygon = rotated

            detections.append({
                "polygon_xy": polygon.astype(np.float32),
                "confidence": float(scores[i]),
                "bbox_xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
            })
        return detections

    def _reshape_yolo_output(self, raw: np.ndarray) -> np.ndarray:
        if raw.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2:
            channels, num = squeezed.shape
            if channels in (5, 6, 84, 85) and num > channels:
                return squeezed.T
            if channels > 6 and num > channels:
                return squeezed[:5, :].T
            return squeezed
        values = raw.reshape(-1)
        for width in (6, 5, 85, 84):
            if values.size % width == 0:
                pred = values.reshape(-1, width)
                if pred.shape[0] > 0:
                    return pred
        return values.reshape(1, -1)

    def _decode_predictions(self, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pred.shape[1] == 5:
            scores = pred[:, 4]
        else:
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            scores = obj if cls_scores.size == 0 else obj * cls_scores.max(axis=1)
        mask = scores >= self.conf_threshold
        pred = pred[mask]
        scores = scores[mask]
        if len(pred) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        box = pred[:, :4].astype(np.float32)
        input_w, input_h = self.input_size
        if np.nanmax(box) <= 2.0:
            box[:, [0, 2]] *= input_w
            box[:, [1, 3]] *= input_h
        cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        xyxy_direct = box.copy()
        direct_valid = np.mean(
            (xyxy_direct[:, 2] > xyxy_direct[:, 0]) & (xyxy_direct[:, 3] > xyxy_direct[:, 1])
        )
        if direct_valid > 0.8 and np.nanmedian(box[:, 2] - box[:, 0]) > 2:
            xyxy = xyxy_direct
        return xyxy.astype(np.float32), scores.astype(np.float32)

    def release(self) -> None:
        self._session = None

    def __del__(self) -> None:
        self.release()


# ── Ascend OM 检测器 ──────────────────────────────────────────────────
class AscendYoloQRDetector:
    """基于 Ascend ACL 的 YOLO 二维码检测器。

    加载 .om 模型，在 Ascend NPU 上执行推理。
    需要 Ascend CANN 运行时环境 (Python acl 包)。
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.2,
        nms_threshold: float = 0.5,
        device_id: int = 0,
        input_size: Tuple[int, int] = (640, 640),
        input_dtype: str = "float32",
        apply_binary_preprocess: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.device_id = int(device_id)
        self.input_size = input_size
        self.input_dtype = input_dtype
        self.apply_binary_preprocess = apply_binary_preprocess

        self.acl = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers: List[Tuple[int, int]] = []
        self.output_buffers: List[Tuple[int, int]] = []
        self.output_sizes: List[int] = []
        self.output_shapes: List[Optional[Tuple[int, ...]]] = []

        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Ascend OM model not found: {self.model_path}. "
                "Put your .om file there or pass model_path."
            )
        try:
            import acl  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Python package 'acl' is required on Ascend runtime. "
                "Make sure CANN is installed and acl is in PYTHONPATH."
            ) from exc

        self.acl = acl
        _check_acl(acl.init(), "acl.init")
        _check_acl(acl.rt.set_device(self.device_id), "acl.rt.set_device")

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        _check_acl(ret, "acl.mdl.load_from_file")

        self.model_desc = acl.mdl.create_desc()
        _check_acl(acl.mdl.get_desc(self.model_desc, self.model_id), "acl.mdl.get_desc")
        self._create_datasets()

    def _create_datasets(self) -> None:
        acl = self.acl
        if acl is None or self.model_desc is None:
            raise RuntimeError("ACL model is not initialized.")

        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        input_count = acl.mdl.get_num_inputs(self.model_desc)
        for idx in range(input_count):
            size = acl.mdl.get_input_size_by_index(self.model_desc, idx)
            ptr, ret = acl.rt.malloc(size, 0)
            _check_acl(ret, f"acl.rt.malloc input {idx}")
            data_buffer = acl.create_data_buffer(ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            _check_acl(ret, f"acl.mdl.add_dataset_buffer input {idx}")
            self.input_buffers.append((ptr, size))

        output_count = acl.mdl.get_num_outputs(self.model_desc)
        for idx in range(output_count):
            size = acl.mdl.get_output_size_by_index(self.model_desc, idx)
            shape = self._get_acl_tensor_shape(is_input=False, index=idx)
            ptr, ret = acl.rt.malloc(size, 0)
            _check_acl(ret, f"acl.rt.malloc output {idx}")
            data_buffer = acl.create_data_buffer(ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            _check_acl(ret, f"acl.mdl.add_dataset_buffer output {idx}")
            self.output_buffers.append((ptr, size))
            self.output_sizes.append(size)
            self.output_shapes.append(shape)

    def _get_acl_tensor_shape(self, is_input: bool, index: int) -> Optional[Tuple[int, ...]]:
        acl = self.acl
        if acl is None or self.model_desc is None:
            return None
        getter_name = "get_input_dims" if is_input else "get_output_dims"
        getter = getattr(acl.mdl, getter_name, None)
        if getter is None:
            return None
        try:
            dims_info = getter(self.model_desc, index)
        except Exception:
            return None
        if isinstance(dims_info, tuple):
            dims_info = dims_info[0]
        if isinstance(dims_info, dict):
            dims = dims_info.get("dims")
        else:
            dims = dims_info
        if not dims:
            return None
        try:
            return tuple(int(x) for x in dims if int(x) > 0)
        except Exception:
            return None

    def detect(self, image: np.ndarray, is_bgr: bool = True) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []
        input_tensor, meta = self._preprocess(image, is_bgr=is_bgr)
        outputs = self._infer(input_tensor)
        return self._postprocess(outputs, image.shape[:2], meta)

    def _preprocess(
        self, image: np.ndarray, is_bgr: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        working = np.ascontiguousarray(image)  # 确保内存连续 (多线程安全)
        if self.apply_binary_preprocess:
            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # 用 np.stack 替代 cv2.cvtColor(..., GRAY2BGR)，更可靠
            working = np.stack([binary] * 3, axis=-1)
        elif not is_bgr:
            working = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)

        letterboxed, scale, pad = _letterbox(working, self.input_size)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        if self.input_dtype == "uint8":
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...].astype(np.uint8))
        else:
            tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            tensor = np.ascontiguousarray(tensor)
        return tensor, {"scale": scale, "pad": pad}

    def _infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        acl = self.acl
        if acl is None or self.model_id is None:
            raise RuntimeError("ACL model is not initialized.")
        if not self.input_buffers:
            raise RuntimeError("No ACL input buffer was allocated.")

        input_ptr, input_size = self.input_buffers[0]
        input_tensor = np.ascontiguousarray(input_tensor)
        if input_tensor.nbytes > input_size:
            raise ValueError(
                f"Input tensor is larger than model input buffer: "
                f"{input_tensor.nbytes} > {input_size}"
            )

        src_ptr = acl.util.numpy_to_ptr(input_tensor)
        _check_acl(
            acl.rt.memcpy(input_ptr, input_size, src_ptr, input_tensor.nbytes, 1),
            "copy input host to device",
        )
        _check_acl(
            acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
            "acl.mdl.execute",
        )

        outputs: List[np.ndarray] = []
        for idx, (output_ptr, output_size) in enumerate(self.output_buffers):
            host = np.empty(output_size, dtype=np.uint8)
            host_ptr = acl.util.numpy_to_ptr(host)
            _check_acl(
                acl.rt.memcpy(host_ptr, output_size, output_ptr, output_size, 2),
                f"copy output {idx} device to host",
            )
            if output_size % 4 == 0:
                output = host.view(np.float32).copy()
                shape = self.output_shapes[idx] if idx < len(self.output_shapes) else None
                if shape is not None and int(np.prod(shape)) == output.size:
                    output = output.reshape(shape)
                outputs.append(output)
            else:
                outputs.append(host.copy())
        return outputs

    def _postprocess(
        self,
        outputs: Sequence[np.ndarray],
        original_hw: Tuple[int, int],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not outputs:
            return []
        raw = outputs[0].astype(np.float32, copy=False)
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2 and squeezed.shape[1] in (5, 6, 7, 38):
            return self._postprocess_nms(squeezed, outputs, original_hw, meta)
        return _postprocess_seg_impl(squeezed, list(outputs), original_hw, meta,
                                     self.input_size, self.conf_threshold, self.nms_threshold)

    def _postprocess_nms(
        self, pred: np.ndarray, outputs, original_hw, meta,
    ) -> List[Dict[str, Any]]:
        """处理 nms 格式: [N, 6/38]"""
        mask_valid = (pred[:, :4].sum(axis=1) > 0) & (pred[:, 4] >= self.conf_threshold)
        pred = pred[mask_valid]
        if len(pred) == 0:
            return []
        boxes = pred[:, :4].copy()
        scores = pred[:, 4].copy()
        has_mask = (pred.shape[1] > 6)
        proto = None
        if has_mask and len(outputs) > 1:
            proto = np.asarray(outputs[1], dtype=np.float32)
            if proto.ndim == 4:
                proto = proto[0]
        scale = float(meta["scale"])
        pad_x, pad_y = meta["pad"]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        orig_h, orig_w = original_hw
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
        detections: List[Dict[str, Any]] = []
        for i in range(len(pred)):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            if has_mask and proto is not None and pred.shape[1] >= 38:
                bbox_nms = pred[i, :4].astype(np.float32)
                rotated = _compute_rotated_polygon_impl(
                    bbox_nms, pred[i, 6:38].astype(np.float32),
                    proto, original_hw, meta, self.input_size,
                )
                if rotated is not None and len(rotated) == 4:
                    polygon = rotated.astype(np.float32)
            detections.append({
                "polygon_xy": polygon,
                "confidence": float(scores[i]),
                "bbox_xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
            })
        return detections

    def _reshape_yolo_output(self, raw: np.ndarray) -> np.ndarray:
        if raw.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2:
            channels, num = squeezed.shape
            # 检测模型: (5,N) 或 (6,N) 或 (85,N) — 直接转置用
            if channels in (5, 6, 84, 85) and num > channels:
                return squeezed.T
            # 分割模型: (37,N), (116,N) 等 — 只取前 5 列 (cx,cy,w,h,conf)
            if channels > 6 and num > channels:
                # 取 bbox 4 + obj_conf 1 = 前 5 列
                return squeezed[:5, :].T
            return squeezed
        # 兜底: 尝试常见宽度
        values = raw.reshape(-1)
        for width in (6, 5, 85, 84):
            if values.size % width == 0:
                pred = values.reshape(-1, width)
                if pred.shape[0] > 0:
                    return pred
        return values.reshape(1, -1)

    def _decode_predictions(self, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pred.shape[1] == 5:
            scores = pred[:, 4]
        else:
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            scores = obj if cls_scores.size == 0 else obj * cls_scores.max(axis=1)

        mask = scores >= self.conf_threshold
        pred = pred[mask]
        scores = scores[mask]
        if len(pred) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        box = pred[:, :4].astype(np.float32)
        input_w, input_h = self.input_size

        if np.nanmax(box) <= 2.0:
            box[:, [0, 2]] *= input_w
            box[:, [1, 3]] *= input_h

        cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xyxy = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
        )

        xyxy_direct = box.copy()
        direct_valid = np.mean(
            (xyxy_direct[:, 2] > xyxy_direct[:, 0]) & (xyxy_direct[:, 3] > xyxy_direct[:, 1])
        )
        if direct_valid > 0.8 and np.nanmedian(box[:, 2] - box[:, 0]) > 2:
            xyxy = xyxy_direct

        return xyxy.astype(np.float32), scores.astype(np.float32)

    def release(self) -> None:
        acl = self.acl
        if acl is None:
            return
        for ptr, _ in self.input_buffers:
            acl.rt.free(ptr)
        for ptr, _ in self.output_buffers:
            acl.rt.free(ptr)
        if self.input_dataset is not None:
            acl.mdl.destroy_dataset(self.input_dataset)
        if self.output_dataset is not None:
            acl.mdl.destroy_dataset(self.output_dataset)
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self.acl = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


# ── 公共 API ─────────────────────────────────────────────────────────
def load_yolov8_detector(
    conf_threshold: float = 0.2,
    nms_threshold: float = 0.5,
    model_path: Optional[str] = None,
    device_id: int = 0,
    input_size: Tuple[int, int] = (640, 640),
    input_dtype: str = "float32",
    apply_binary_preprocess: bool = False,
):
    """加载 YOLO QR 检测器，自动识别 .om / .onnx 后端。

    参数:
        conf_threshold: 置信度阈值 (默认 0.2)
        nms_threshold: NMS IoU 阈值 (默认 0.5)
        model_path: .om 或 .onnx 模型路径
        device_id: Ascend 设备 ID (仅 .om 后端)
        input_size: 模型输入尺寸
        input_dtype: 'float32' 或 'uint8'
        apply_binary_preprocess: 二值化前处理

    返回:
        检测器实例，失败返回 None
    """
    resolved = model_path or os.environ.get("ASCEND_QR_MODEL_PATH") or str(DEFAULT_MODEL_PATH)
    ext = os.path.splitext(resolved)[1].lower()

    try:
        if ext == ".onnx":
            backend = "ONNX Runtime (CPU)"
            detector = OnnxYoloQRDetector(
                model_path=resolved,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                input_size=input_size,
                input_dtype=input_dtype,
                apply_binary_preprocess=apply_binary_preprocess,
            )
        else:
            backend = "Ascend (ACL)"
            detector = AscendYoloQRDetector(
                model_path=resolved,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                device_id=device_id,
                input_size=input_size,
                input_dtype=input_dtype,
                apply_binary_preprocess=apply_binary_preprocess,
            )
        print(
            f"[{backend}] detector loaded: model={resolved} "
            f"conf={conf_threshold} nms={nms_threshold}"
        )
        return detector
    except Exception as exc:
        print(f"[ERROR] Failed to load detector: {exc}")
        return None


def detect_qr_yolov8(image: np.ndarray, detector) -> List[Dict[str, Any]]:
    """检测图像中的二维码候选区域。

    参数:
        image: BGR 格式的 numpy 数组 (H, W, 3)
        detector: AscendYoloQRDetector 实例

    返回:
        detections 列表，每项含 polygon_xy, confidence, bbox_xyxy
    """
    if detector is None:
        return []
    return detector.detect(image=image, is_bgr=True)


def detect_image(
    image_or_path, detector,
    output_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """检测图片（支持文件路径或 numpy 数组）。

    参数:
        image_or_path: 图片文件路径 或 BGR numpy 数组
        detector: 检测器实例
        output_path: 若指定，将可视化结果写入该路径

    返回:
        (detections, result_image)
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required.")

    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise TypeError(f"Expected str/Path or np.ndarray, got {type(image_or_path)}")

    detections = detect_qr_yolov8(image, detector)

    result_img = None
    if output_path is not None:
        result_img = _draw_detections(image, detections)
        cv2.imwrite(str(output_path), result_img)

    return detections, result_img


# ── 辅助 ─────────────────────────────────────────────────────────────
def _parse_input_size(value: str) -> Tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError("input size must be like 640 or 640,640")


def _draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    result = image.copy()
    for det in detections:
        # 优先用 polygon_xy（旋转四边形）；没有则用 bbox_xyxy
        polygon = det.get("polygon_xy")
        if polygon is not None and len(polygon) == 4:
            corners = polygon.astype(np.int32)
        else:
            bbox = det.get("bbox_xyxy")
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                continue

        confidence = float(det.get("confidence", 0.0))
        cv2.polylines(result, [corners], True, (0, 255, 0), 2)
        for pt in corners[:4]:
            cv2.circle(result, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        cv2.putText(
            result, f"{confidence:.3f}",
            (int(corners[0][0]), max(20, int(corners[0][1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return result


# ── CLI / 测试入口 ───────────────────────────────────────────────────
def main() -> int:
    import time

    parser = argparse.ArgumentParser(description="Ascend OM YOLO QR detector.")
    parser.add_argument("--image", default=None, help="单张测试图片路径")
    parser.add_argument("--dir", default=None, help="批量测试目录")
    parser.add_argument("--camera", type=int, default=None, help="摄像头设备号 (如 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                        help=f"模型路径 (.om 或 .onnx)，默认: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--onnx", default=None,
                        help="ONNX 模型路径，用于对比测试")
    parser.add_argument("--device-id", type=int, default=0, help="NPU 设备 ID")
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--input-size", type=_parse_input_size, default=(640, 640))
    parser.add_argument("--input-dtype", choices=["float32", "uint8"], default="float32")
    parser.add_argument("--binary-preprocess", action="store_true",
                        help="启用二值化前处理（默认关闭）")
    parser.add_argument("--output", default="result", help="输出文件名前缀")
    args = parser.parse_args()

    if cv2 is None:
        print("[ERROR] opencv-python 未安装。")
        return 2

    # ── 收集图片列表 ──
    image_paths: List[str] = []

    if args.image:
        image_paths = [args.image]
    elif args.dir:
        p = Path(args.dir)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(str(x) for x in p.glob(ext))
            image_paths = sorted(set(image_paths))
    elif args.camera is not None:
        image_paths = []  # 相机模式不走文件列表
    else:
        # 无参数: 找 test_data/ 或 qr_template.jpg
        root = Path(__file__).resolve().parent
        td = root / "test_data"
        qt = root / "qr_template.jpg"
        if td.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(str(x) for x in td.glob(ext))
            image_paths = sorted(set(image_paths))
        elif qt.exists():
            image_paths = [str(qt)]

    if args.camera is None and not image_paths:
        print("[ERROR] 未找到图片。用 --image, --dir 或 --camera 指定输入。")
        return 2

    # ── 加载检测器 ──
    detector = load_yolov8_detector(
        conf_threshold=args.conf, nms_threshold=args.nms,
        model_path=args.model, device_id=args.device_id,
        input_size=args.input_size, input_dtype=args.input_dtype,
        apply_binary_preprocess=args.binary_preprocess,
    )
    if detector is None:
        return 3

    # ── 对比模式: 同时加载 ONNX 做对比 ──
    onnx_detector = None
    if args.onnx:
        onnx_detector = load_yolov8_detector(
            conf_threshold=args.conf, nms_threshold=args.nms,
            model_path=args.onnx, input_size=args.input_size,
            input_dtype=args.input_dtype,
            apply_binary_preprocess=args.binary_preprocess,
        )
        if onnx_detector is None:
            print("[WARN] ONNX 模型加载失败，仅使用主模型测试。")

    try:
        # ── 相机模式 ──
        if args.camera is not None:
            print(f"[INFO] 打开摄像头 {args.camera}，按 'q' 退出。")
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                print(f"[ERROR] 无法打开摄像头 {args.camera}")
                return 4
            frame_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                t0 = time.perf_counter()
                detections = detect_qr_yolov8(frame, detector)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                vis = _draw_detections(frame, detections)
                cv2.putText(vis, f"QR:{len(detections)} {elapsed_ms:.0f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Ascend QR Detect", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] 相机模式结束，共处理 {frame_count} 帧。")
            return 0

        # ── 图片批量模式 ──
        print(f"[INFO] 测试图片数: {len(image_paths)}")
        total_elapsed = 0.0
        total_dets = 0
        success = 0

        for idx, img_path in enumerate(image_paths):
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [{idx+1}/{len(image_paths)}] SKIP: {img_path}")
                continue

            fname = os.path.splitext(os.path.basename(img_path))[0]

            # ── 主模型推理 ──
            t0 = time.perf_counter()
            detections = detect_qr_yolov8(image, detector)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_elapsed += elapsed_ms
            total_dets += len(detections)
            success += 1

            # ── ONNX 对比推理 ──
            onnx_dets = None
            onnx_ms = 0.0
            if onnx_detector:
                t1 = time.perf_counter()
                onnx_dets = detect_qr_yolov8(image, onnx_detector)
                onnx_ms = (time.perf_counter() - t1) * 1000

            # ── 打印结果 ──
            print(f"  [{idx+1}/{len(image_paths)}] {fname}")
            print(f"       [OM]  {len(detections)} QR  {elapsed_ms:.1f}ms  " +
                  (f"vs  [ONNX] {len(onnx_dets)} QR  {onnx_ms:.1f}ms" if onnx_detector else ""))
            for di, det in enumerate(detections):
                print(f"       [OM-{di}] conf={det['confidence']:.4f}")
            if onnx_dets:
                for di, det in enumerate(onnx_dets):
                    print(f"       [ONNX-{di}] conf={det['confidence']:.4f}")

            # ── 保存可视化 ──
            result = _draw_detections(image, detections)
            cv2.imwrite(f"{args.output}_{fname}.jpg", result)
            if onnx_detector and onnx_dets:
                result_onnx = _draw_detections(image.copy(), onnx_dets)
                cv2.imwrite(f"{args.output}_onnx_{fname}.jpg", result_onnx)

        if success > 0:
            avg_ms = total_elapsed / success
            print(f"\n=== 完成: {success} 张, {total_dets} 个QR, 均{avg_ms:.1f}ms/张 ===")
        return 0
    finally:
        if hasattr(detector, "release"):
            detector.release()
        if onnx_detector and hasattr(onnx_detector, "release"):
            onnx_detector.release()


if __name__ == "__main__":
    raise SystemExit(main())
