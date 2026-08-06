#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置加载模块
- 从 camera_intrinsics.txt 读取相机内参
- 从 app_config.json 读取应用参数 (统一配置入口)
"""

import json
import re
import os
from pathlib import Path
from typing import Optional
import numpy as np


# ── 默认配置 (app_config.json 的基线值) ──
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent


def load_camera_intrinsics(txt_path: str) -> dict:
    """从 camera_intrinsics.txt 读取相机内参和 QR 参数。

    参数:
        txt_path: camera_intrinsics.txt 文件路径

    返回:
        dict: 包含 camera_matrix (3x3), dist_coeffs (5x1), fov_x, fov_y, qr_size 等字段
    """
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

    # 字段别名兼容
    if "fov_x" not in info and "fov_x_deg" in info:
        info["fov_x"] = info["fov_x_deg"]
    if "fov_y" not in info and "fov_y_deg" in info:
        info["fov_y"] = info["fov_y_deg"]

    return info


def load_app_config(config_path: Optional[str] = None) -> dict:
    """加载应用配置文件 (JSON 格式)。

    加载优先级: config_path > app_config.json (同目录) > 内建默认值

    参数:
        config_path: 自定义配置文件路径, None 则使用默认 app_config.json

    返回:
        dict: 完整配置字典 (嵌套结构，与 app_config.json 一致)
    """
    # ── 确定配置文件路径 ──
    if config_path is not None:
        json_path = config_path
    else:
        json_path = os.path.join(str(_DEFAULT_CONFIG_DIR), "app_config.json")

    config = {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # 过滤掉以 _ 开头的注释字段
        config = {k: v for k, v in config.items() if not k.startswith("_")}
    else:
        print(f"[Config] 配置文件不存在: {json_path}, 使用内建默认值")

    return config


def get_config_value(config: dict, *keys, default=None):
    """从嵌套配置字典中安全取值。

    用法:
        val = get_config_value(config, "gimbal", "rotation_step_deg", default=5.0)
    """
    d = config
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d
