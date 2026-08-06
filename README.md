# QR_ASCEND — Ascend NPU 二维码位姿估计 + 机器狗导航

## 文件说明

| 文件 | 功能 |
|------|------|
| `gimbal_pose_estimation_ascend.py` | 位姿估计主程序（云台抓图 → QR 检测 → PnP 位姿） |
| `robot_dog_bridge.py` | 位姿 → 机器狗导航管线 |
| `robot_interface.py` | X30 机器狗运动接口（UDP 直连，纯标准库，无依赖） |
| `yolo_detection_ascend.py` | YOLO QR 检测（Ascend OM / ONNX） |
| `qr_detection.py` | QR 几何处理（四边形近似、单应变换） |
| `pose_estimation.py` | 位姿估计（PnP 解算、欧拉角） |
| `gimbal_demo.py` | 云台控制器 |
| `main_demo.py` | 调试图像保存 |
| `config.json` | 配置文件（qr_size 等） |

## 环境要求

```bash
# Ascend CANN 环境
# robot_interface.py 为纯标准库，无需安装额外依赖
```

## 启动命令

### 1. YOLO 检测测试（单独测试 OM 模型）

```bash
# 单张图片
python yolo_detection_ascend.py --model ./qr_yolov8.om --image ./test.jpg

# 批量目录
python yolo_detection_ascend.py --model ./qr_yolov8.om --dir ./testimgs/
```

### 2. 位姿估计（云台抓图）

```bash
# 单次估计
python gimbal_pose_estimation_ascend.py --mode single --model ./qr_yolov8.om

# 从本地图片
python gimbal_pose_estimation_ascend.py --image ./test.jpg --model ./qr_yolov8.om

# 交互模式
python gimbal_pose_estimation_ascend.py --mode interactive --model ./qr_yolov8.om
```

### 3. 位姿估计 + 机器狗导航

```bash
# 从云台抓图 → 位姿估计 → 机器狗移动
python robot_dog_bridge.py --mode single --model ./qr_yolov8.om

# 从本地图片测试
python robot_dog_bridge.py --image ./test.jpg --model ./qr_yolov8.om

# 循环模式（持续采集 + 导航）
python robot_dog_bridge.py --mode loop --model ./qr_yolov8.om

# 模拟模式（不连真狗，只打印坐标）
python robot_dog_bridge.py --image ./test.jpg --sim
```

### 完整参数

```bash
python robot_dog_bridge.py \
    --mode single \                  # single / loop
    --model ./qr_yolov8.om \         # OM 模型路径
    --device-id 0 \                  # NPU 设备
    --gimbal-ip 192.168.1.99 \       # 云台 IP
    --qr-size 0.10 \                 # QR 物理尺寸 (m)
    --fov-x 60.0 \                   # 相机视场角
    --conf 0.2 \                     # YOLO 置信度
    --nms 0.5 \                      # NMS 阈值
    --dog-ip 192.168.1.103 \         # 机器狗 IP
    --dog-port 43893 \               # 机器狗 UDP 端口
    --stop-distance 0.8 \            # 距二维码水平安全距离 (m)
    --no-stand \                     # 不自动起立
    --debug                          # 保存调试图像
```

机器狗导航使用 `robot_interface.Dog.move(x, y, z)` 相对运动指令:
- `x`: 前进距离(m)，直接使用 QR_depth 转换后的狗体前进量
- `y`: 侧移距离(m)，符号按狗体右侧距离取反适配
- `z`: 旋转角度(°)，由 arctan2 计算朝向 QR 的方向

三轴等比例调速同时到达，默认停在二维码前方 `0.8 m`。不需要 TDRobot / ROS2 / 地图导航。

### 板子上部署步骤

```bash
# 1. 拷贝 QR_ASCEND 到板子
scp -r QR_ASCEND/ user@board:/home/hjl/

# 2. robot_interface.py 为纯标准库，无需安装额外依赖

# 3. 机器狗需开机并在同一局域网
# UDP 运动控制: 192.168.1.103:43893

# 4. 启动
cd /home/hjl/QR_ASCEND
python robot_dog_bridge.py --mode single --model ./qr_yolov8.om
```
