#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云台位姿估计接口
功能：
1. 从云台抓取可见光图片
2. 使用实际相机估计接口进行位姿估计（非仿真）
3. 打印估计的位姿信息，用于手动调整云台朝向
4. 实时显示视频流

使用说明：
- 确保云台已连接并上电
- 确保二维码在可见光相机视野内
- 运行后，程序会自动抓取图像并估计位姿
- 根据输出的位姿信息，手动调整云台角度
- 按 'v' 键可以打开/关闭实时视频流显示
"""

import sys
import time
import os
import queue
import threading
import math
import numpy as np
import cv2


# ========== HTTP MJPEG 视频流（替代 cv2.imshow，板子无 GUI 时使用）==========
_http_scale = 0.35                  # 缩放比例（1920x1080 → 672x378，减少带宽）
_http_jpeg_quality = 35             # JPEG 压缩质量（0-100，越低越流畅）
_http_fps = 10.0                    # 推流帧率上限 (只推一个原图窗口, 已大幅省带宽)


class _JpegBuffer:
    """最新 JPEG 帧缓冲 (线程安全, 条件变量零空转)。

    编码只在 update 端做一次, Flask 端所有客户端共享同一份 JPEG,
    避免每个浏览器连接各自编码导致 CPU 打满。
    """
    def __init__(self):
        self._cond = threading.Condition()
        self._jpg = b""

    def update(self, jpg: bytes):
        with self._cond:
            self._jpg = jpg
            self._cond.notify_all()

    def wait(self, timeout: float = 0.5) -> bytes:
        with self._cond:
            self._cond.wait(timeout)
            return self._jpg


_http_buf_raw = _JpegBuffer()        # 原图窗口 (MJPEG 实时流, 唯一推流窗口)
_http_key_queue = []                 # 模拟键盘输入队列
_http_key_lock = threading.Lock()

# ── 检测结果 / 对齐预览 (静态图片 + 前端轮询, 带宽极低) ──
_http_detect_lock = threading.Lock()
_http_detect_jpg = b""               # 检测结果 JPEG (按 [c] 后更新)
_http_detect_ver = 0                 # 版本号 (前端用于判断是否需要刷新)
_http_detect_msg = "尚未捕获: 按 [c] 进行位姿估计"

_http_aligned_lock = threading.Lock()
_http_aligned_jpg = b""              # 对齐预览 JPEG (旋转云台后预期画面)
_http_aligned_ver = 0
_http_aligned_msg = "尚未生成: 等待 [c] 捕获完成"


def _set_panel_status(text: str):
    """立即刷新网页面板提示文字。

    按键送达后马上可见 (不等位姿估计跑完), 用于区分
    『按键没送到』和『位姿估计正在跑』两种情况。
    """
    global _http_detect_msg, _http_aligned_msg
    with _http_detect_lock:
        _http_detect_msg = text
    with _http_aligned_lock:
        _http_aligned_msg = text


def _make_placeholder_jpg(text: str, w: int = 640, h: int = 360) -> bytes:
    """生成提示占位图 (避免空图破图, 节省带宽)"""
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
    cv2.putText(img, text, (24, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(img, "Press [c] to capture pose",
                (24, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    ok, buf = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


_http_placeholder_detect = _make_placeholder_jpg("No detection yet")
_http_placeholder_aligned = _make_placeholder_jpg("Aligned preview not generated")

# ── 截图 / 状态 / 3D 显示 (由 estimator 注入或后台线程更新) ──
_http_snap_provider = None           # () -> 高清截图 JPEG bytes
_http_status_provider = None         # () -> 状态文本
_http_3d_lock = threading.Lock()
_http_3d_png = b""                   # 3D 场景 PNG (matplotlib Agg 离屏渲染)
_http_3d_ver = 0                     # 版本号 (前端轮询用, 与 detect/aligned 一致)
_http_3d_msg = "尚未生成: 先按 [c] 捕获位姿, 再点 [3D 显示]"

# ── 机器狗网页控制 (由 estimator 注入) ──
_http_dog_provider = None            # (action:str, value:float) -> 结果文本
_http_dog_lock = threading.Lock()
_http_dog_msg = "机器狗未连接: 先「① 起立」→「② 力控」→「③ 开始运动」"
_http_dog_busy = threading.Lock()    # 串行化: 一次只执行一个动作 (防连点变成多步)

# ── 机器狗安全锁: 限制单次移动量 (防止网页误触让狗跑太远) ──
DOG_MAX_MOVE_M = 0.3                 # 单次平移硬上限 (米)
DOG_MAX_TURN_DEG = 30.0              # 单次旋转硬上限 (度)
DOG_DEFAULT_MOVE_M = 0.2             # 默认平移步长 (米) —— 每按一次只走这一步
DOG_DEFAULT_TURN_DEG = 15.0          # 默认旋转步长 (度)
DOG_AUTO_INIT = True                 # 未使能运动时, 移动前自动「起立+力控」


def _set_3d_png(png: bytes = b"", msg: str = ""):
    """3D 渲染线程回写结果"""
    global _http_3d_png, _http_3d_msg, _http_3d_ver
    with _http_3d_lock:
        if png:
            _http_3d_png = png
            _http_3d_ver += 1
        if msg:
            _http_3d_msg = msg


def _encode_stream_jpeg(frame):
    """缩放 + 画中心十字 + JPEG 编码 (只做一次, 供所有客户端共享)"""
    h, w = frame.shape[:2]
    nw, nh = int(w * _http_scale), int(h * _http_scale)
    if nw > 0 and nh > 0 and (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
    # 在图像中心画黄色 X 十字 (auto_aim 瞄准参考标线)
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    L = max(12, frame.shape[1] // 60)   # 臂长, 跟图宽自适应
    color = (0, 255, 255)                # BGR: 黄色
    thick = max(1, frame.shape[1] // 480)
    cv2.line(frame, (cx - L, cy - L), (cx + L, cy + L), color, thick, cv2.LINE_AA)
    cv2.line(frame, (cx - L, cy + L), (cx + L, cy - L), color, thick, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _http_jpeg_quality])
    return jpg.tobytes() if ok else None


def _make_flask_app(owner=None):
    """创建 Flask 应用 (控制台页面: 视频/检测/截图/3D + 全套按钮)

    参数:
        owner: GimbalPoseEstimator 实例, 注入到 app.config['_owner_instance']
               供 /3d_view 等路由访问 self._3d_html / self._viz3d 等
    """
    from flask import Flask, Response, request
    app = Flask(__name__)
    if owner is not None:
        app.config["_owner_instance"] = owner

    @app.route("/raw")
    def _stream_raw():
        def gen():
            while True:
                jpg = _http_buf_raw.wait(0.5)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── 检测结果 / 对齐预览: 静态图片 + 前端轮询 (带宽极低, 不再 MJPEG 推流) ──
    @app.route("/detect.png")
    def _stream_detect_png():
        with _http_detect_lock:
            jpg = _http_detect_jpg
        if not jpg:
            return Response(_http_placeholder_detect, mimetype="image/jpeg")
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/detect_ver")
    def _stream_detect_ver():
        with _http_detect_lock:
            return Response(str(_http_detect_ver), mimetype="text/plain; charset=utf-8")

    @app.route("/detect_msg")
    def _stream_detect_msg():
        with _http_detect_lock:
            msg = _http_detect_msg
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/aligned.png")
    def _stream_aligned_png():
        with _http_aligned_lock:
            jpg = _http_aligned_jpg
        if not jpg:
            return Response(_http_placeholder_aligned, mimetype="image/jpeg")
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/aligned_ver")
    def _stream_aligned_ver():
        with _http_aligned_lock:
            return Response(str(_http_aligned_ver), mimetype="text/plain; charset=utf-8")

    @app.route("/aligned_msg")
    def _stream_aligned_msg():
        with _http_aligned_lock:
            msg = _http_aligned_msg
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/")
    def _stream_index():
        return """<html><head><meta charset="utf-8"><title>QR 位姿估计控制台</title>
        <style>
        body{font-family:Arial;margin:10px;background:#1a1a1a;color:#fff}
        h2{color:#0f0;text-align:center}
        img{max-width:100%;border:1px solid #333;margin:5px 0}
        .container{display:flex;flex-wrap:wrap;gap:10px}
        .panel{flex:1;min-width:420px;text-align:center}
        .msg{color:#0cf;font-size:13px;margin-top:4px;min-height:18px}
        button{padding:8px 14px;margin:3px;font-size:14px;cursor:pointer;border:none;border-radius:4px}
        .btn-c{background:#0a0;color:#fff}.btn-p{background:#08c;color:#fff}
        .btn-s{background:#606;color:#fff}.btn-q{background:#c00;color:#fff}
        .btn-yn{background:#444;color:#fff}.btn-dir{background:#555;color:#fff}
        .btn-zoom{background:#c80;color:#fff}.btn-f{background:#626;color:#fff}
        .btn-snap{background:#088;color:#fff}.btn-3d{background:#c0c;color:#fff}
        .btn-dog{background:#e60;color:#fff}.btn-dogq{background:#a00;color:#fff}
        .btn-dogm{background:#e60;color:#fff}
        .btn-dogm:disabled{background:#555;color:#999;cursor:not-allowed}
        .status{margin-top:8px;font-size:14px;color:#ff0;text-align:center;min-height:20px}
        </style></head><body>
        <h2>QR 位姿估计 控制台</h2>
        <div class="container">
        <div class="panel"><h3>原图 (实时)</h3><img src="/raw"></div>
        <div class="panel"><h3>检测结果 ([c] 后)</h3>
          <img id="detect" data-ver="0" src="/detect.png?0"><div class="msg" id="detect_msg"></div></div>
        <div class="panel"><h3>3D 坐标场景 (Plotly, 红/蓝视锥)</h3>
          <iframe id="three_d" data-ver="0" src="/3d_view" style="width:100%;height:480px;border:1px solid #333;background:#222"></iframe>
          <div class="msg" id="three_d_msg"></div></div>
        </div>
        <div class="panel"><img id="snap" src="" style="display:none"></div>
        <div style="text-align:center">
        <button class="btn-c" onclick="send('c')">[c] 捕获位姿</button>
        <button class="btn-p" onclick="send('p')">[p] 暂停/继续</button>
        <button class="btn-s" onclick="send('s')">[s] 位姿信息开关</button>
        <button class="btn-yn" onclick="send('y')">[y] 重拍</button>
        <button class="btn-yn" onclick="send('n')">[n] 继续</button>
        <button class="btn-q" onclick="send('q')">[q] 退出</button>
        <br>
        <button class="btn-dir" onclick="send('left')">← Yaw-</button>
        <button class="btn-dir" onclick="send('right')">→ Yaw+</button>
        <button class="btn-dir" onclick="send('up')">↑ Pitch+</button>
        <button class="btn-dir" onclick="send('down')">↓ Pitch-</button>
        <button class="btn-yn" onclick="send('+')">+ 步长</button>
        <button class="btn-yn" onclick="send('-')">- 步长</button>
        <br>
        <button class="btn-zoom" onclick="send('a')">[A] 变倍+</button>
        <button class="btn-zoom" onclick="send('d')">[D] 变倍-</button>
        <button class="btn-f" onclick="send('f')">[f] 自动对焦</button>
        <button class="btn-c" onclick="send('r')">[r] 云台回中</button>
        <br>
        <button class="btn-snap" onclick="snap()">📷 截图</button>
        <button class="btn-3d" onclick="window.open('/3d_view','_blank')">🔮 3D 显示</button>
        <button class="btn-3d" onclick="send('g')">♻ 重新生成 3D</button>
        </div>
        <hr style="border-color:#333;margin:12px 0">
        <h3 style="color:#f0a;text-align:center">🐕 机器狗控制 (UDP)</h3>
        <div style="text-align:center;margin-bottom:12px">
        <div style="color:#888;font-size:12px;margin-bottom:6px">
          顺序: ① 起立 → ② 力控 → ③ 开始运动 → ④ 单步移动 (每按一次 = 一步)</div>
        <button class="btn-dog" onclick="dog('stand')">① 🦿 起立连接</button>
        <button class="btn-dog" onclick="dog('force')">② 🖐️ 力控</button>
        <button class="btn-dog" onclick="dog('motion')">③ 🚀 开始运动</button>
        <button class="btn-dog" onclick="dog('down')">🐾 趴下复位</button>
        <button class="btn-dog" onclick="dog('damp')">💧 阻尼</button>
        <button class="btn-dogq" onclick="dog('stop')">⏹ 停止并趴下</button>
        <br>
        <button class="btn-dog" onclick="dogstep(-1)">− 步长</button>
        <span id="dog_step" style="color:#0cf;font-size:14px;margin:0 6px">0.2m / 15°</span>
        <button class="btn-dog" onclick="dogstep(1)">+ 步长</button>
        <span style="color:#888;font-size:12px">(安全锁: 单次 ≤0.3m / ≤30°)</span>
        <br>
        <button class="btn-dogm" onclick="dog('fwd')">⬆ 前进</button>
        <button class="btn-dogm" onclick="dog('back')">⬇ 后退</button>
        <button class="btn-dogm" onclick="dog('left')">⬅ 左移</button>
        <button class="btn-dogm" onclick="dog('right')">➡ 右移</button>
        <button class="btn-dogm" onclick="dog('tl')">↺ 左转</button>
        <button class="btn-dogm" onclick="dog('tr')">↻ 右转</button>
        <div class="msg" id="dog_msg">机器狗未连接: 先「① 起立」→「② 力控」→「③ 开始运动」</div>
        <div style="color:#888;font-size:11px;margin-top:4px">
          ⚠ 「② 力控 + ③ 开始运动」即实体遥控器上的同名两个按钮, 缺一腿不动</div>
        </div>
        <div class="status" id="status"></div>
        <script>
        function send(k){fetch('/key/'+k).then(r=>r.text()).then(t=>{document.getElementById('status').innerText=t})}
        function snap(){var s=document.getElementById('snap');s.src='/snap?'+Date.now();s.style.display='block'}
        // 机器狗控制: /dog/<action>?v=<步长>  → 返回结果文本
        // 单步模式: 每按一次只走固定一步; 执行期间禁用按钮, 连点无效 (防多步)
        // 步长上限与后端安全锁一致 (0.3m / 30°)
        var DOG_LOCK_M=0.3, DOG_LOCK_DEG=30;
        var dogStepM=0.2, dogStepDeg=15;
        var dogBusy=false;
        function dog(a){
          if(dogBusy)return;                       // 上一步未完成, 忽略本次点击
          var v=(a=='tl'||a=='tr')?dogStepDeg:(a=='stand'||a=='force'||a=='motion'||a=='start'||a=='stop'||a=='down'||a=='damp')?0:dogStepM;
          dogBusy=true;
          var ms=document.querySelectorAll('.btn-dogm');
          ms.forEach(function(b){b.disabled=true;});
          var e=document.getElementById('dog_msg'); if(e)e.innerText='⏳ 执行中: '+a+' ...';
          fetch('/dog/'+a+'?v='+v)
          .then(r=>r.text()).then(t=>{if(t&&e)e.innerText=t;})
          .catch(()=>{})
          .then(function(){dogBusy=false;ms.forEach(function(b){b.disabled=false;});});
        }
        function dogstep(d){if(d>0){dogStepM=Math.min(DOG_LOCK_M,+(dogStepM+0.1).toFixed(1));dogStepDeg=Math.min(DOG_LOCK_DEG,dogStepDeg+5);}
          else{dogStepM=Math.max(0.1,+(dogStepM-0.1).toFixed(1));dogStepDeg=Math.max(5,dogStepDeg-5);}
          document.getElementById('dog_step').innerText=dogStepM+'m / '+dogStepDeg+'°';}
        // 检测 / 对齐: 仅当服务端版本号变化时才下载新图, 带宽极低
        function pollPanel(imgId, msgId, verUrl, imgUrl, msgUrl){
          fetch(verUrl).then(r=>r.text()).then(v=>{
            var img=document.getElementById(imgId);
            if(img.getAttribute('data-ver')!==v){img.src=imgUrl+'?'+v;img.setAttribute('data-ver',v);}
          }).catch(()=>{});
          fetch(msgUrl).then(r=>r.text()).then(t=>{var e=document.getElementById(msgId);if(t&&e.innerText!==t)e.innerText=t;}).catch(()=>{});
        }
        // 3D Plotly iframe: 仅当版本号变化时强制刷新 src (带 cache-buster)
        function pollIframe(){
          fetch('/3d_view_ver').then(r=>r.text()).then(v=>{
            var f=document.getElementById('three_d');
            if(f.getAttribute('data-ver')!==v){
              f.setAttribute('data-ver',v);
              f.src='/3d_view?_v='+v;   // cache-buster 让 iframe 重新加载
            }
          }).catch(()=>{});
          fetch('/3d_msg').then(r=>r.text()).then(t=>{var e=document.getElementById('three_d_msg');if(t&&e.innerText!==t)e.innerText=t;}).catch(()=>{});
        }
        setInterval(()=>{
          pollPanel('detect','detect_msg','/detect_ver','/detect.png','/detect_msg');
          pollIframe();
          fetch('/dog_msg').then(r=>r.text()).then(t=>{var e=document.getElementById('dog_msg');if(t&&e.innerText!==t)e.innerText=t;}).catch(()=>{});
        },800);
        setInterval(()=>{fetch('/status').then(r=>r.text()).then(t=>{document.getElementById('status').innerText=t}).catch(()=>{})},2000);
        var km={'c':'c','p':'p','s':'s','q':'q','y':'y','n':'n','a':'a','d':'d','f':'f','r':'r','g':'g',
        '+':'+','=':'+','-':'-','_':'-','ArrowLeft':'left','ArrowRight':'right','ArrowUp':'up','ArrowDown':'down'};
        document.onkeydown=function(e){var k=km[e.key];if(k){e.preventDefault();send(k)}}
        </script></body></html>"""

    @app.route("/snap")
    def _stream_snap():
        """高清截图 (原始分辨率, 按需编码)"""
        jpg = b""
        if _http_snap_provider is not None:
            try:
                jpg = _http_snap_provider() or b""
            except Exception:
                jpg = b""
        if not jpg:
            jpg = _http_buf_raw.wait(0.05)
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/3d")
    def _stream_3d_page():
        """3D 场景页面 (兼容老链接, 嵌入 /3d_view iframe)"""
        return """<html><head><meta charset="utf-8"><title>3D 场景 (兼容页)</title>
        <style>body{margin:0;background:#111;color:#eee;font-family:Arial}
        .hdr{padding:10px;text-align:center;background:#222;border-bottom:1px solid #444}
        iframe{border:0;width:100%;height:calc(100vh - 50px)}
        </style></head><body>
        <div class="hdr">QR 与相机 3D 场景 (Plotly 交互式) | 主页第三窗口也同步显示</div>
        <iframe src="/3d_view"></iframe>
        </body></html>"""

    @app.route("/3d.png")
    def _stream_3d_png():
        with _http_3d_lock:
            png = _http_3d_png
        return Response(png, mimetype="image/png")

    @app.route("/3d_ver")
    def _stream_3d_ver():
        with _http_3d_lock:
            return Response(str(_http_3d_ver), mimetype="text/plain; charset=utf-8")

    @app.route("/3d_msg")
    def _stream_3d_msg():
        with _http_3d_lock:
            msg = _http_3d_msg
        return Response(msg, mimetype="text/plain; charset=utf-8")

    # ═══════════════════════════════════════════════════════════
    # Plotly HTML 3D 可视化 (板端无 GUI, FrustumVisualizer.render_html)
    #   /3d_view      → 返回最近一次生成的 Plotly HTML (供 iframe 嵌入)
    #   /3d_view_ver  → 返回版本号 (前端轮询用, 与 detect/aligned 一致)
    # ═══════════════════════════════════════════════════════════
    @app.route("/3d_view")
    def _stream_3d_view():
        # 用闭包捕获 self (Flask 装饰器场景下需要传实例)
        # 但 _make_flask_app 不持有 self 实例, 改用 _http_status_provider 的间接方式:
        # 简单做法: 把 html 暂存到全局, 通过 _http_buf 风格的全局变量传递
        # 实际上 _make_flask_app 是被 create_http_stream(self) 调用, self 通过 _http_status_provider 注入
        # 这里采用最直接的方式: 用一个全局可变对象
        try:
            from flask import current_app
            # 通过 Flask config 间接拿 (创建 app 时已注入 self)
            inst = current_app.config.get("_owner_instance")
            if inst is None:
                return Response("<html><body><h3>未初始化</h3></body></html>",
                                mimetype="text/html; charset=utf-8")
            with inst._3d_html_lock:
                html = inst._3d_html
                ver = inst._3d_html_ver
            if not html:
                with inst._3d_html_lock:
                    msg = inst._3d_html_msg
                return Response(
                    f"<html><body style='background:#111;color:#ff0;font-family:Arial;"
                    f"display:flex;align-items:center;justify-content:center;height:100vh'>"
                    f"<h3>3D 尚未生成: {msg}</h3></body></html>?" + str(ver),
                    mimetype="text/html; charset=utf-8")
            # 用版本号作 cache-buster, 避免 iframe 缓存
            sep = '&' if '?' in html else '?'
            return Response(html + f"{sep}_v={ver}",
                            mimetype="text/html; charset=utf-8")
        except Exception as e:
            return Response(
                f"<html><body><h3>3D 渲染异常: {e}</h3></body></html>",
                mimetype="text/html; charset=utf-8")

    @app.route("/3d_view_ver")
    def _stream_3d_view_ver():
        try:
            from flask import current_app
            inst = current_app.config.get("_owner_instance")
            if inst is None:
                return Response("0", mimetype="text/plain; charset=utf-8")
            with inst._3d_html_lock:
                return Response(str(inst._3d_html_ver), mimetype="text/plain; charset=utf-8")
        except Exception:
            return Response("0", mimetype="text/plain; charset=utf-8")

    @app.route("/status")
    def _stream_status():
        try:
            msg = _http_status_provider() if _http_status_provider else ""
        except Exception:
            msg = ""
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/key/<k>")
    def _http_key(k):
        """浏览器发送按键: 按钮或键盘 → /key/c 等"""
        with _http_key_lock:
            _http_key_queue.append(k)
        try:
            msg = _http_status_provider() if _http_status_provider else f"OK: {k}"
        except Exception:
            msg = f"OK: {k}"
        return Response(msg, mimetype="text/plain; charset=utf-8")

    # ═══════════════════════════════════════════════════════════
    # 机器狗控制 (robot_interface.Dog, UDP)
    #   /dog/<action>?v=<步长>   action: stand/start/down/damp/stop
    #                                    fwd/back/left/right/tl/tr
    #   /dog_msg                 返回最近一次操作结果 (前端轮询)
    # ═══════════════════════════════════════════════════════════
    @app.route("/dog/<action>")
    def _http_dog(action):
        global _http_dog_msg
        value = request.args.get("v", default=0.0, type=float)
        # ?gait=0x21010301 可选: 同时切步态 (力控动作下有效, 解决「原地踏步不前进」)
        gait_str = request.args.get("gait", default="", type=str)
        gc = None
        if gait_str and gait_str not in ("0", "None", "default"):
            try:
                gc = int(gait_str, 0)
            except Exception:
                gc = None
        # 串行化: 一次只执行一个动作, 连点直接拒绝 (保证「单步」)
        if not _http_dog_busy.acquire(blocking=False):
            msg = "⏳ 上一个动作还在执行, 请稍候再按 (单步模式)"
            with _http_dog_lock:
                _http_dog_msg = msg
            return Response(msg, mimetype="text/plain; charset=utf-8")
        try:
            with _http_dog_lock:
                _http_dog_msg = f"⏳ 执行中: {action} ..."
            if _http_dog_provider:
                msg = _http_dog_provider(action, value, gait=gc)
            else:
                msg = "机器狗未连接"
        except Exception as e:
            msg = f"狗控制异常: {e}"
        finally:
            _http_dog_busy.release()
        with _http_dog_lock:
            _http_dog_msg = msg
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/dog_msg")
    def _http_dog_msg_route():
        with _http_dog_lock:
            msg = _http_dog_msg
        return Response(msg, mimetype="text/plain; charset=utf-8")

    return app


def start_http_stream(port=8888, owner=None):
    """启动 HTTP 视频流服务（后台守护线程）

    参数:
        owner: 持有 _3d_html/_viz3d 等状态的实例, 供 /3d_view 路由访问
    """
    # 禁止 werkzeug 日志刷屏
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    app = _make_flask_app(owner=owner)
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True
    )
    t.start()
    print(f"  HTTP 视频流: http://0.0.0.0:{port}/")
    return t


def update_stream_frame(frame_raw, frame_detect=None):
    """更新视频帧: 缩放+编码一次 → 共享缓冲 (速率受 _http_fps 限制, 不积压)

    参数 frame_detect 已弃用 (检测窗口改为静态图 + 轮询, 节省带宽)。
    """
    if not hasattr(update_stream_frame, '_next_t'):
        update_stream_frame._next_t = 0.0
    now = time.time()
    if now < update_stream_frame._next_t:
        return  # 限速: 编码跟不上就丢帧
    update_stream_frame._next_t = now + 1.0 / _http_fps

    if frame_raw is not None:
        jpg = _encode_stream_jpeg(frame_raw)
        if jpg:
            _http_buf_raw.update(jpg)
    # frame_detect 已弃用, 不再写入 MJPEG 缓冲


def _get_http_key():
    """获取浏览器端发送的按键（模拟 cv2.waitKey）。

    单字符 → ord() 兼容原有 key == ord('x') 比较;
    多字符 (left/right/up/down 等网页按钮) → 原样返回字符串。
    """
    with _http_key_lock:
        if _http_key_queue:
            key = _http_key_queue.pop(0)
            return ord(key) if len(key) == 1 else key
    return -1


# 导入云台控制模块
try:
    from gimbal_demo import GimbalController, capture_from_rtsp
except ImportError:
    print("错误: 无法导入 gimbal_demo 模块")
    print("请确保 gimbal_demo.py 在同一目录下")
    sys.exit(1)

# 导入 main_demo 模块以使用其 save_debug_image 函数
try:
    import main_demo
except ImportError:
    print("警告: 无法导入 main_demo 模块，将使用内置的保存函数")
    main_demo = None

# 导入二维码检测和位姿估计模块
import yolo_detection_ascend as yolo_detection
import qr_detection
import pose_estimation
import pose_pipeline            # QR_3D 兼容 (本工程不再使用 compute_gimbal_target 迭代对齐)
from pose_pipeline import compute_gimbal_target, build_aligned_pose

# Plotly 3D 可视化器 — 板端无 GUI, 用 Plotly HTML。
# 延迟导入：模块顶层不强制要求 plotly, 在第一次需要时再装/再 import。
#   - PC 端没装 plotly → 构造函数里尝试 `pip install plotly`, 装好就用 Plotly
#   - 装失败 / 无网络 → fallback 到 matplotlib PNG (跟原始 QR_3D 一致)
FrustumVisualizer = None
_HAS_PLOTLY = False

# ── 诊断: 启动时记录当前 python / plotly 状态 ──
import sys as _sys
print(f"[DIAG] python: {_sys.executable}")
try:
    import plotly as _pl
    print(f"[DIAG] plotly: {_pl.__version__} ({_pl.__file__})")
except Exception as _e:
    print(f"[DIAG] plotly 不可用: {_e}")


class GimbalPoseEstimator:
    """云台位姿估计器"""
    
    def __init__(self, 
                 gimbal_ip="192.168.1.99", 
                 gimbal_port=6001,
                 config_file="config.json",
                 qr_size=None,
                 fov_x=60.0,
                 model_path="./qr_yolov8.om",
                 device_id=0,
                 detector_type='opencv',
                 square_size=800,
                 conf_threshold=0.2,
                 nms_threshold=0.5):
        """
        初始化云台位姿估计器
        
        参数:
            gimbal_ip: 云台IP地址
            gimbal_port: 云台控制端口
            config_file: 配置文件路径（用于读取qr_size等参数）
            qr_size: 二维码物理尺寸（米），如果为None则从config.json读取
            fov_x: 相机水平视场角（度）
            model_size: YOLOv8模型大小
            detector_type: 二维码检测器类型
            square_size: 矫正后正方形图像边长
            conf_threshold: YOLOv8置信度阈值
            nms_threshold: YOLOv8非极大值抑制阈值
        """
        print("=" * 60)
        print("初始化云台位姿估计器")
        print("=" * 60)
        
        # 从配置文件读取参数
        self.config_file = config_file
        config_data = self._load_config(config_file)
        self._cfg = config_data  # 缓存, 给 FrustumVisualizer 用
        
        # 如果未提供qr_size，从配置文件读取
        if qr_size is None:
            qr_size = config_data.get("estimate", {}).get("qr_size", 0.1)
            print(f"  从配置文件读取 qr_size -------------------------: {qr_size} 米")
        #qr_size = 0.05
        qr_size = 0.12

        print(f" qr_size -------------------------: {qr_size} 米")

        # 保存配置参数
        self.gimbal_ip = gimbal_ip
        self.qr_size = qr_size
        self.fov_x = fov_x
        self.detector_type = detector_type
        self.square_size = square_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 初始化云台控制器
        print(f"\n1. 连接云台: {gimbal_ip}:{gimbal_port}")
        try:
            self.gimbal = GimbalController(ip=gimbal_ip, port=gimbal_port)
            print("   ✓ 云台连接成功")
        except Exception as e:
            print(f"   ✗ 云台连接失败: {e}")
            raise

        # ── 变倍步进状态 ──
        # 协议字 0x01/0x02 (zoom_in/zoom_out) 是"持续变倍"指令——云台收到后
        # 会一直放到物理极限, 直到收到 0x03 (zoom_stop) 才停. 视频流按键
        # 若用 zoom_in/zoom_out, 按一下会无限持续下去, 必须配套发 zoom_stop.
        # 因此 [A]/[D] 改用 0x0B (zoom_to_magnification): 发目标倍率,
        # 云台走到该倍率就停, 一次按键 = 一步变倍. 与 gimbal_operator.py 一致.
        self.current_zoom_mag = 1.0   # 启动时默认 1x, 首次按键后用云台回读对齐
        self.zoom_step_val   = 1.0    # 每按一下步进的倍率
        self._zoom_min       = 1.0
        self._zoom_max       = 30.0
        
        # 构建RTSP地址
        self.rtsp_tv = f"rtsp://{gimbal_ip}:554/ch1/stream0"  # 可见光
        self.rtsp_ir = f"rtsp://{gimbal_ip}:554/ch2/stream0"  # 红外
        print(f"   可见光地址: {self.rtsp_tv}")
        print(f"   红外地址:   {self.rtsp_ir}")
        
        # ── YOLOv8 检测器 (Ascend OM) ──
        # 关键: Ascend ACL 的 device/context 与**线程**绑定。在 A 线程加载的
        #       .om 模型放到 B 线程执行时, acl.rt.memcpy 会直接失败:
        #       "ACL failed during copy input host to device, ret=107002"。
        #       所以必须保证「加载线程 == 推理线程」, 检测器按线程各持一份。
        #       此处先为当前(主)线程加载: auto_aim / 单图流程都在主线程执行;
        #       视频流的位姿估计走常驻推理线程, 由该线程自行懒加载。
        self._model_path = model_path
        self._device_id = device_id
        self._detector_local = threading.local()   # 每线程一份检测器
        self._detector_lock = threading.Lock()     # 串行化 ACL 模型加载

        print(f"\n2. 加载 YOLO 检测器 (Ascend OM: {model_path})")
        self.detector = yolo_detection.load_yolov8_detector(
            model_path=model_path, device_id=device_id,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
        if self.detector is None:
            raise RuntimeError("YOLOv8检测器加载失败")
        self._detector_local.det = self.detector   # 主线程复用这一份, 不重复加载
        print("   ✓ 检测器加载成功")
        
        # 创建输出目录
        self.output_dir = "gimbal_pose_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n3. 输出目录: {self.output_dir}")
        
        # 视频流相关属性
        self.video_capture = None           # OpenCV VideoCapture对象
        self.video_thread = None            # 视频推流线程
        self._reader_thread = None          # RTSP 读帧线程 (读/处理分离)
        self._latest_frame = None           # 最新帧 (读帧线程覆盖式写入, 旧帧丢弃)
        self._est_busy = False              # 位姿估计进行中标志 (防重入)
        self._inf_queue = None              # 常驻推理线程的帧队列
        self._inf_thread = None             # 常驻推理线程 (ACL 同线程执行)
        self.video_running = False          # 视频流运行标志
        self.video_paused = False           # 视频暂停标志
        self.current_frame = None           # 当前帧
        self.frame_lock = threading.Lock()  # 帧锁
        self.latest_pose_data = None       # 最新的位姿数据 (EMA 平均值)
        self._first_pose_data = None        # 首帧原始位姿 (单帧, 用于 [c] 旋转, 与 QR_3D auto_aim 一致)
        self.pose_records = []             # 位姿记录 (网页 3D 显示用)
        self.gimbal_step = 5.0             # 云台网页微调步长 (度)
        self.show_pose_on_video = False    # 是否在视频上显示位姿信息

        # 3D 可视化 (Plotly HTML 板端渲染, 红色视锥=旋转前, 蓝色视锥=旋转后)
        self._viz3d = None
        self._3d_html = ""                 # 最近一次 Plotly HTML
        self._3d_html_ver = 0              # 版本号 (前端轮询用)
        self._3d_html_lock = threading.Lock()
        self._3d_html_msg = "尚未生成: 先按 [c] 捕获位姿"
        # 同步初始化 viz3d (启动时检测, 失败则后台尝试装 plotly)
        self._init_viz3d_sync()
        
        # Review 模式相关属性
        # 注意: 初始必须是 False。若为 True, 视频循环里 [c] 的处理条件
        #       `not self.review_mode` 会被挡住 → 开机后按 [c] 完全无反应。
        #       review 模式只在 [c] 捕获完成且检测到 YOLO 框后才进入。
        self.review_mode = False          # 是否处于 review 模式
        self.review_detections = []        # 保存YOLO检测结果供循环绘制
        self.need_recapture = False        # 是否需要在下一轮循环重新捕获
        
        # 多帧平滑属性
        self._pose_history: list = []      # 滑动窗口: [(pose_data, weight), ...]
        self._pose_window_size = 10        # 窗口大小
        self._pose_ema_alpha = 0.35        # EMA 平滑系数 (0=完全依赖历史, 1=完全依赖新帧)
        
        print("\n" + "=" * 60)
        print("初始化完成")
        print("=" * 60)

    def _load_config(self, config_file):
        """
        从JSON配置文件读取参数

        参数:
            config_file: 配置文件路径

        返回:
            配置字典，如果文件不存在则返回空字典
        """
        import json
        config_data = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                print(f"  已加载配置文件: {config_file}")
            except Exception as e:
                print(f"  警告: 无法读取配置文件 {config_file}: {e}")
        else:
            print(f"  警告: 配置文件不存在: {config_file}，使用默认参数")

        return config_data

    def capture_visible_image(self, output_path=None, use_rtsp=True):
        """
        步骤1: 从云台抓取可见光图片
        
        参数:
            output_path: 输出图像路径（如果为None，则自动生成）
            use_rtsp: 是否使用RTSP流（True）或直接抓取（False）
        
        返回:
            success: 是否成功
            image_path: 保存的图像路径
        """
        print("\n" + "-" * 60)
        print("步骤1: 抓取可见光图像")
        print("-" * 60)
        
        # 生成输出文件路径
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"capture_tv_{timestamp}.jpg")
        
        print(f"  输出路径: {output_path}")
        
        # 方法1: 使用RTSP流抓取（推荐）
        if use_rtsp:
            print(f"  方法: RTSP流 ({self.rtsp_tv})")
            success = capture_from_rtsp(self.rtsp_tv, output_path, timeout=5)
        
        # 方法2: 使用云台命令直接抓取
        else:
            print(f"  方法: 云台命令 (通道0=可见光)")
            success = self.gimbal.capture_image(
                channel=0,  # 0=可见光
                output_path=output_path,
                timeout=10.0
            )
        
        if success:
            print(f"  ✓ 图像抓取成功: {output_path}")
            # 验证图像可读性
            img = cv2.imread(output_path)
            if img is not None:
                print(f"  图像尺寸: {img.shape[1]}x{img.shape[0]}")
                return True, output_path
            else:
                print(f"  ✗ 图像保存成功但无法读取")
                return False, None
        else:
            print(f"  ✗ 图像抓取失败")
            return False, None
    
    def _get_detector(self):
        """取『当前线程』的 YOLOv8 检测器 (懒加载 + 每线程一份)。

        为什么不能直接用 self.detector:
            Ascend ACL 的 device/context 是线程私有的。在别的线程加载的 .om
            模型在本线程执行会报
            "ACL failed during copy input host to device, ret=107002"。
            所以检测器必须"在哪个线程推理, 就在哪个线程加载"。

        返回:
            AscendYoloQRDetector 实例
        """
        det = getattr(self._detector_local, "det", None)
        if det is not None:
            return det

        with self._detector_lock:              # ACL 模型加载不适合并发
            det = getattr(self._detector_local, "det", None)
            if det is None:
                print(f"  [ACL] 线程 {threading.get_ident()} 首次加载 YOLO 检测器...")
                det = yolo_detection.load_yolov8_detector(
                    model_path=self._model_path,
                    device_id=self._device_id,
                    conf_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                )
                if det is None:
                    raise RuntimeError("YOLOv8检测器加载失败")
                self._detector_local.det = det
                print(f"  [ACL] ✓ 检测器已绑定线程 {threading.get_ident()} "
                      f"(该线程后续所有推理复用此模型)")
        return det

    def estimate_pose_from_image(self, image_path, debug=True, debug_dir=None):
        """
        步骤2: 使用实际相机估计接口进行位姿估计
        
        注意：这不是仿真，而是实际相机的位姿估计
        严格按照main_demo.py的process_image_for_pose函数调用方式
        
        参数:
            image_path: 输入图像路径
            debug: 是否保存调试图像
            debug_dir: 调试图像保存目录
        
        返回:
            success: 是否成功
            pose_data: 位姿数据字典
            result_image: 结果图像（带标注）
        """
        print("\n" + "-" * 60)
        print("步骤2: 位姿估计（实际相机）")
        print("-" * 60)
        
        # 初始化调试目录（按照main_demo.py的方式）
        if debug:
            if debug_dir is None:
                timestamp = int(time.time())
                debug_dir = f"debug_output_{timestamp}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"\n  调试模式已启用，中间图片将保存到: {debug_dir}")
        
        # 读取图像
        print(f"\n  读取图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ 无法读取图像")
            return False, None, None
        
        print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 1. 保存原始图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(image, "01_original_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "01_original_image.jpg")
                cv2.imwrite(filepath, image)
                print(f"  调试: 图像已保存到 {filepath}")
        
        # 使用YOLOv8检测二维码 (检测器取当前线程的那一份, ACL 同线程)
        print(f"\n  2.1 YOLOv8检测二维码...")
        detections = yolo_detection.detect_qr_yolov8(image, self._get_detector())
        print(f"  检测到 {len(detections)} 个二维码")
        
        if len(detections) == 0:
            print(f"  ✗ 未检测到二维码")
            return False, None, None
        
        # 只处理第一个检测到的二维码
        detection = detections[0]
        polygon_xy = detection['polygon_xy'].astype(np.float32)
        confidence = detection['confidence']
        print(f"  使用检测结果 1 (置信度: {confidence:.2f})")
        
        # 2. 保存YOLO初始检测结果图（带边界框）（按照main_demo.py）
        # 绘制YOLO框到图像上
        image_init_detect = qr_detection.draw_qr_boundary(image.copy(), 
                                                        qr_detection.order_points_clockwise2(
                                                            qr_detection.approximate_quadrilateral(polygon_xy)
                                                        ))
        
        # 保存带YOLO框的图像，用于detection窗口显示
        self.review_frame_with_yolo = image_init_detect.copy()
        
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(image_init_detect, "02_yolo_initial_detect", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "02_yolo_initial_detect.jpg")
                cv2.imwrite(filepath, image_init_detect)
                print(f"  调试: 图像已保存到 {filepath}")
        

        
        # 近似为四边形
        print(f"\n  2.2 近似四边形...")
        try:
            quad_points = qr_detection.approximate_quadrilateral(polygon_xy)
            quad_points = qr_detection.order_points_clockwise2(quad_points)
            print(f"  ✓ 四边形角点: {quad_points.shape}")
        except ValueError as e:
            print(f"  ✗ 四边形近似失败: {e}")
            return False, None, None



        #--根据yolo增加遮罩，方便后续alpha合成 -------------------------------------------

        # 生成二值图像（YOLO框内为255，框外为0）
        # 计算YOLO框的边界
        x_coords = quad_points[:, 0]
        y_coords = quad_points[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # 计算中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 计算最大边长（最大y差和最大x差的较大值）
        max_x_diff = x_max - x_min
        max_y_diff = y_max - y_min
        max_side = max(max_x_diff, max_y_diff)
        
        print(f"  YOLO框中心: ({center_x}, {center_y})")
        print(f"  YOLO框最大边长: {max_side} pixels")
        
        # 创建二值图像（与image_init_detect同分辨率）
        alpha_mask = np.zeros(image_init_detect.shape[:2], dtype=np.uint8)
        
        # 在YOLO框内填充255（使用多边形填充）
        pts = quad_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(alpha_mask, [pts], 255)
        
        # 保存二值图像
        if debug:
            alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_alpha.jpg")
        else:
            alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_alpha.jpg")
        cv2.imwrite(alpha_filepath, alpha_mask)
        print(f"  调试: 二值图像已保存到 {alpha_filepath}")
        
        # 对二值图像进行膨胀操作，让白色部分变大
        # 定义膨胀核（5x5的矩形核）
        kernel = np.ones((55, 55), np.uint8)
        alpha_mask_big = cv2.dilate(alpha_mask, kernel, iterations=1)
        
        # 保存膨胀后的二值图像
        if debug:
            big_alpha_filepath = os.path.join(debug_dir, "02_yolo_initial_detect_big_alpha.jpg")
        else:
            big_alpha_filepath = os.path.join(self.output_dir, "02_yolo_initial_detect_big_alpha.jpg")
        cv2.imwrite(big_alpha_filepath, alpha_mask_big)
        print(f"  调试: 膨胀后二值图像已保存到 {big_alpha_filepath}")
        
        #--------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------

        # 透视变换矫正 - 按照main_demo.py的方式
        print(f"\n  2.3 透视变换矫正...")
        
        # 按照main_demo.py的方式计算dst_square
        margin_qr =1500  # 边距
        shift_qr = margin_qr / 2.0  # 偏移量
        
        dst_square = np.array([
            [shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, shift_qr],
            [self.square_size - 1 + shift_qr, self.square_size - 1 + shift_qr],
            [shift_qr, self.square_size - 1 + shift_qr]
        ], dtype=np.float32)
        
        H = qr_detection.compute_homography(quad_points, dst_square)
        if H is None:
            print(f"  ✗ 单应矩阵计算失败")
            return False, None, None
        
        # 透视变换目标尺寸按照main_demo.py的方式
        warped = qr_detection.warp_perspective_with_homography(
            image, H, (self.square_size + margin_qr, self.square_size + margin_qr)
        )
        #--------------------------------------------------------------------------------------------

        warped_alpha = qr_detection.warp_perspective_with_homography(
            alpha_mask_big, H, (self.square_size + margin_qr, self.square_size + margin_qr)
        )
        #3. 保存透视变换后的alpha图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped_alpha, "03_warped_alpha_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_alpha_image.jpg")
                cv2.imwrite(filepath, warped_alpha)
                print(f"  调试: 图像已保存到 {filepath}")


        #----------------------------------------------------------------------------------------------


        print(f"  ✓ 透视变换完成")
        print(f"  透视变换尺寸: {self.square_size + margin_qr}x{self.square_size + margin_qr}")
        
        # 3. 保存透视变换后的矫正图像（按照main_demo.py）
        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped, "03_warped_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_image.jpg")
                cv2.imwrite(filepath, warped)
                print(f"  调试: 图像已保存到 {filepath}")
        


        # alpha合成
        warped_alpha = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR)
        warped_alpha = warped_alpha / 255.0
        warped_01 = warped / 255.0
        warped_comp = warped_alpha * warped_01 
        warped_comp = warped_comp * 255.0
        warped_comp = warped_comp.astype(np.uint8)

        #step2 : 补白---------------------------------------------------------------------
        border_image =(1-warped_alpha)* 255.0
        border_image = border_image.astype(np.uint8)
        warped_comp = cv2.add(warped_comp, border_image)


        if debug:
            if main_demo is not None:
                main_demo.save_debug_image(warped_comp, "03_warped_image_comp", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "03_warped_image_comp.jpg")
                cv2.imwrite(filepath, warped_comp)
                print(f"  调试: 图像已保存到 {filepath}")
        
        # 在矫正图像上检测二维码角点
        print(f"\n  2.4 检测二维码角点...")
        # success, points_warped, decoded_info = qr_detection.detect_qr_in_warped_image(
        #     warped, self.detector_type
        # )

        success, points_warped, decoded_info = qr_detection.detect_qr_in_warped_image(
            warped, self.detector_type
        )
        
        if success:
            print(f"  ✓ 检测到二维码: {decoded_info}")
            # 映射回原始图像
            H_inv = np.linalg.inv(H)
            points_corrected = qr_detection.transform_points_back(points_warped, H_inv)


            points_corrected = points_corrected[:, :2]
            #---------------由于相机缺失一个自由度，不能绕Z轴旋转，所以需要将Z轴旋转的角度去掉-----------------
            points_corrected = qr_detection.order_points_clockwise2(points_corrected)
            #---------------
            print(f"  ✓ 角点已映射回原始图像")
            
            # 4. 保存矫正图像上的检测结果（带边界框）（按照main_demo.py）
            if debug:
                image_warped_detect = qr_detection.draw_qr_boundary(warped.copy(), points_warped)
                if main_demo is not None:
                    main_demo.save_debug_image(image_warped_detect, "04_warped_detect", debug_dir)
                else:
                    os.makedirs(debug_dir, exist_ok=True)
                    filepath = os.path.join(debug_dir, "04_warped_detect.jpg")
                    cv2.imwrite(filepath, image_warped_detect)
                    print(f"  调试: 图像已保存到 {filepath}")
        else:
            print(f"  ⚠ 矫正图像未检测到二维码，使用YOLO检测的角点")
            points_corrected = quad_points
        
        
        # 定义二维码的3D角点（世界坐标系）
        print(f"\n  2.5 位姿估计...")
        print(f"二维码实际尺寸...", self.qr_size)
        object_points = pose_estimation.define_qr_object_points(qr_size=self.qr_size)
        
        # 查询当前焦距和视场角，用于计算相机内参矩阵
        # 如果有 _override_camera_matrix（来自 camera_intrinsics.txt），直接使用
        if hasattr(self, '_override_camera_matrix') and self._override_camera_matrix is not None:
            H, W = image.shape[:2]
            camera_matrix = self._override_camera_matrix
            fov_x, fov_y, focal_x, focal_y = None, None, None, None
            print(f"  使用 camera_intrinsics.txt 中的内参矩阵（跳过云台查询）")
            # 保存相机内参到实例，供 auto_aim 等后续步骤从 fx 反算 FOV
            self.camera_matrix = camera_matrix
        else:
            print(f"\n  2.5.1 查询当前焦距和视场角...")
            fov_x, fov_y, focal_x, focal_y = self.gimbal.query_focal_length()
            if fov_x is not None and fov_y is not None:
                print(f"  ✓ 查询成功: fov_x={fov_x:.1f}°, fov_y={fov_y:.1f}°, focal_x={focal_x:.1f}mm, focal_y={focal_y:.1f}mm")
                # 使用查询到的视场角创建相机内参矩阵
                H, W = image.shape[:2]
                camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=fov_x, fov_y=fov_y)
            else:
                # 查询失败: 按最初版本回退到默认 fov_x (云台未返回真实数据)
                print(f"  ⚠ 查询失败，使用默认fov_x={self.fov_x}°")
                print(f"  ⚠⚠ 焦距不准确！当前为默认视场角，非云台真实数据，变倍后角度计算会有较大误差")
                H, W = image.shape[:2]
                camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=self.fov_x)
        # 保存相机内参到实例，供 auto_aim 等后续步骤从 fx 反算 FOV
        self.camera_matrix = camera_matrix
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        print(f"  相机内参矩阵:\n{camera_matrix}")
        
        # 保存相机内参到 txt（方便后续离线测试）
        if debug and debug_dir is not None:
            cam_txt_path = os.path.join(debug_dir, "camera_intrinsics.txt")
            with open(cam_txt_path, "w") as f:
                f.write("# 相机内参矩阵 camera_matrix (3x3)\n")
                f.write("# 格式: fx,  0, cx\n")
                f.write("#        0, fy, cy\n")
                f.write("#        0,  0,  1\n\n")
                f.write(f"image_width = {W}\n")
                f.write(f"image_height = {H}\n")
                f.write(f"fov_x = {fov_x if fov_x is not None else 'N/A'}\n")
                f.write(f"fov_y = {fov_y if fov_y is not None else 'N/A'}\n")
                f.write(f"focal_x = {focal_x if focal_x is not None else 'N/A'}\n")
                f.write(f"focal_y = {focal_y if focal_y is not None else 'N/A'}\n")
                f.write(f"qr_size = {self.qr_size:.4f}\n\n")
                f.write("camera_matrix = [\n")
                for row in camera_matrix:
                    f.write(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],\n")
                f.write("]\n\n")
                f.write("dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]\n")
            print(f"  ✓ 相机内参已保存到: {cam_txt_path}")
        
        # 估计位姿
        success_pose, rvec, tvec, rmat = pose_estimation.estimate_pose(
            object_points, points_corrected, camera_matrix, dist_coeffs
        )
        
        if not success_pose:
            print(f"  ✗ 位姿估计失败")
            return False, None, None
        
        print(f"  ✓ 位姿估计成功")
        
        # 计算欧拉角
        yaw, pitch, roll = pose_estimation.rotation_matrix_to_euler_angles(rmat)
        
        # 计算相机位置（世界坐标系）
        camera_position = -rmat.T @ tvec
        
        # 整理位姿数据
        pose_data = {
            'rvec': rvec.flatten(),                    # 旋转向量
            'tvec': tvec.flatten(),                    # 平移向量
            'rmat': rmat,                             # 旋转矩阵
            'yaw': np.degrees(yaw),                   # 偏航角（度）
            'pitch': np.degrees(pitch),                # 俯仰角（度）
            'roll': np.degrees(roll),                  # 横滚角（度）
            'camera_position': camera_position.flatten(),  # 相机位置（米）
            'distance': np.linalg.norm(camera_position),   # 距离（米）
            'qr_size': self.qr_size,                  # 二维码尺寸（米）
            'image_points': points_corrected,          # 图像角点
            'object_points': object_points,            # 二维码3D角点（世界坐标系）
            'camera_matrix': camera_matrix             # 相机内参矩阵
        }
        
        # 绘制结果图像
        print(f"\n  2.6 绘制结果...")
        result_image = image.copy()
        
        # 绘制二维码边界
        result_image = qr_detection.draw_qr_boundary(result_image, points_corrected)
        
        # 绘制坐标轴
        result_image = pose_estimation.draw_axes(
            result_image, camera_matrix, dist_coeffs, rvec, tvec
        )
        
        print(f"  ✓ 结果图像已绘制")
        
        # 5. 保存最终结果图（带正确角点顺序和坐标系）（按照main_demo.py）
        if debug and points_corrected is not None:
            if main_demo is not None:
                main_demo.save_debug_image(result_image, "05_result_image", debug_dir)
            else:
                os.makedirs(debug_dir, exist_ok=True)
                filepath = os.path.join(debug_dir, "05_result_image.jpg")
                cv2.imwrite(filepath, result_image)
                print(f"  调试: 图像已保存到 {filepath}")
            print(f"  调试: 中间图片已保存到 {debug_dir}")
        
        return True, pose_data, result_image
    
    def print_pose_info(self, pose_data):
        """
        步骤3: 打印位姿信息
        
        这些信息可以手动输入到云台控制系统中
        
        参数:
            pose_data: 位姿数据字典
        """
        print("\n" + "=" * 60)
        print("位姿估计结果")
        print("=" * 60)
        
        print(f"\n【旋转向量】 (rvec)")
        print(f"  [{pose_data['rvec'][0]:.6f}, {pose_data['rvec'][1]:.6f}, {pose_data['rvec'][2]:.6f}]")
        
        print(f"\n【平移向量】 (tvec) - 相机在二维码坐标系中的位置")
        print(f"  X: {pose_data['tvec'][0]:.6f} 米")
        print(f"  Y: {pose_data['tvec'][1]:.6f} 米")
        print(f"  Z: {pose_data['tvec'][2]:.6f} 米")
        
        print(f"\n【欧拉角】 (ZYX顺序, 度)")
        print(f"  Yaw   (绕Z轴, 偏航):   {pose_data['yaw']:.2f}°")
        print(f"  Pitch (绕Y轴, 俯仰):   {pose_data['pitch']:.2f}°")
        print(f"  Roll  (绕X轴, 横滚):   {pose_data['roll']:.2f}°")
        
        print(f"\n【相机位置】 (世界坐标系)")
        print(f"  X: {pose_data['camera_position'][0]:.6f} 米")
        print(f"  Y: {pose_data['camera_position'][1]:.6f} 米")
        print(f"  Z: {pose_data['camera_position'][2]:.6f} 米")
        print(f"  距离: {pose_data['distance']:.6f} 米")
        
        print(f"\n【其他信息】")
        print(f"  二维码尺寸: {pose_data['qr_size']} 米")
        print(f"  图像角点坐标:")
        for i, pt in enumerate(pose_data['image_points']):
            print(f"    角点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")


        #------云台坐标转换------------------------------------------------

        yam_camera_input = pose_data['pitch']
        pitch_camera_input = pose_data['roll'] - 180.0

        if(abs(pitch_camera_input)>=180):
            pitch_camera_input_temp = (360.0 + pitch_camera_input)% 360.0
            pitch_camera_input = pitch_camera_input_temp



        #-----获知当前云台朝向，并计算云台需要调整的角度-------------------

        # 获取当前云台角度并加上相机角度
        try:
            #gimbal_yaw, gimbal_pitch = self.gimbal.get_current_angles()
            gimbal_yaw, gimbal_pitch, gimbal_roll = self.gimbal.get_current_angles()

            print(f"\n【变量角度】")
            print(f"  add  Yaw:   {yam_camera_input:.2f}°")
            print(f"  add Pitch: {pitch_camera_input:.2f}°")


            print(f"\n【云台当前角度】")
            print(f"  Gimbal Yaw:   {gimbal_yaw:.2f}°")
            print(f"  Gimbal Pitch: {gimbal_pitch:.2f}°")
            print(f"  Gimbal Roll: {gimbal_roll:.2f}°")

            # 加上云台当前角度
            yam_camera_input += gimbal_yaw
            pitch_camera_input += gimbal_pitch

            print(f"\n【叠加后角度】 (位姿角度 + 云台当前角度)")
        except Exception as e:
            print(f"\n警告: 无法获取云台当前角度: {e}")
            print(f"\n【位姿角度】 (未叠加云台角度)")

        #-----------------------------------------------------------------
        
        print("\n" + "=" * 60)
        print("手动调整云台朝向 - 参考值")
        print("=" * 60)
        print(f"\n根据位姿估计结果，建议的云台角度（仅供参考）:")
        print(f"  Yaw   (水平旋转): {yam_camera_input:.2f}°")
        print(f"  Pitch (上下俯仰): {pitch_camera_input:.2f}°")
        print(f"\n注意: 实际调整时需要考虑坐标系定义和安装方式")
        print("=" * 60)
    
    def save_result(self, result_image, pose_data, output_prefix=None):
        """
        保存结果图像和位姿数据
        
        参数:
            result_image: 结果图像
            pose_data: 位姿数据
            output_prefix: 输出文件前缀
        """
        if output_prefix is None:
            output_prefix = os.path.join(self.output_dir, f"pose_result_{int(time.time())}")
        
        # 保存结果图像
        result_image_path = f"{output_prefix}.jpg"
        cv2.imwrite(result_image_path, result_image)
        print(f"\n结果图像已保存: {result_image_path}")
        
        # 保存位姿数据（NPY格式）
        pose_data_path = f"{output_prefix}_pose.npy"
        np.save(pose_data_path, pose_data)
        print(f"位姿数据已保存: {pose_data_path}")
        
        # 保存位姿数据（文本格式，方便查看）
        pose_text_path = f"{output_prefix}_pose.txt"
        with open(pose_text_path, 'w', encoding='utf-8') as f:
            f.write("位姿估计结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"旋转向量 (rvec):\n")
            f.write(f"  {pose_data['rvec']}\n\n")
            f.write(f"平移向量 (tvec):\n")
            f.write(f"  {pose_data['tvec']}\n\n")
            f.write(f"欧拉角 (度):\n")
            f.write(f"  Yaw:   {pose_data['yaw']:.2f}°\n")
            f.write(f"  Pitch: {pose_data['pitch']:.2f}°\n")
            f.write(f"  Roll:  {pose_data['roll']:.2f}°\n\n")
            f.write(f"相机位置 (米):\n")
            f.write(f"  X: {pose_data['camera_position'][0]:.6f}\n")
            f.write(f"  Y: {pose_data['camera_position'][1]:.6f}\n")
            f.write(f"  Z: {pose_data['camera_position'][2]:.6f}\n")
            f.write(f"  距离: {pose_data['distance']:.6f}\n")
        print(f"位姿数据(文本)已保存: {pose_text_path}")
    
    def run_single_estimation(self, save_result=True):
        """
        运行单次位姿估计
        
        流程:
        1. 抓取可见光图像
        2. 位姿估计
        3. 打印结果
        4. 可选：保存结果
        
        返回:
            success: 是否成功
            pose_data: 位姿数据
        """
        print("\n" + "=" * 60)
        print("开始位姿估计流程")
        print("=" * 60)
        
        # 步骤1: 抓取图像
        success, image_path = self.capture_visible_image(use_rtsp=True)
        if not success:
            print("\n✗ 流程终止: 图像抓取失败")
            return False, None
        
        # 步骤2: 位姿估计
        success, pose_data, result_image = self.estimate_pose_from_image(image_path)
        if not success:
            print("\n✗ 流程终止: 位姿估计失败")
            return False, None
        
        # 步骤3: 打印位姿信息
        self.print_pose_info(pose_data)
        
        # 步骤4: 保存结果（可选）
        if save_result and result_image is not None:
            self.save_result(result_image, pose_data)
        
        print("\n" + "=" * 60)
        print("位姿估计流程完成")
        print("=" * 60)
        
        return True, pose_data
    
    def run_interactive_mode(self):
        """
        运行交互模式
        
        用户可以多次抓取图像并估计位姿，
        根据结果手动调整云台角度
        可以打开实时视频流查看画面
        """
        print("\n" + "=" * 60)
        print("交互模式")
        print("=" * 60)
        print("\n命令说明:")
        print("  c - 抓取图像并估计位姿")
        print("  v - 打开/关闭实时视频流")
        print("  s - 调整云台角度")
        print("  z - 变倍/变焦/调焦控制")
        print("  a - 自动瞄准模式（等间隔识别+自动旋转）")
        print("  q - 退出")
        print("=" * 60)
        print("\n视频流窗口快捷键:")
        print("  q - 关闭视频")
        print("  p - 暂停/继续")
        print("  c - 捕获当前帧并估计位姿")
        print("  s - 显示/隐藏位姿信息")
        print("=" * 60)
        
        while True:
            cmd = input("\n请选择命令 (c/v/s/q): ").strip().lower()
            
            if cmd == 'q':
                print("\n退出程序")
                # 停止视频流
                if self.video_running:
                    self.stop_video_stream()
                break
            
            elif cmd == 'c':
                # 抓取并估计
                success, pose_data = self.run_single_estimation(save_result=True)
                
                if success:
                    print("\n✓ 位姿估计完成，请根据上面的信息调整云台")
            
            elif cmd == 'v':
                # 打开/关闭视频流
                if self.video_running:
                    print("\n关闭视频流...")
                    self.stop_video_stream()
                else:
                    print("\n打开视频流...")
                    self.start_video_stream(display_pose=True)
            
            elif cmd == 's':
                # 调整云台角度
                print("\n调整云台角度:")
                print("(输入角度，单位: 度)")
                try:
                    yaw = float(input("  Yaw (偏航角): "))
                    pitch = float(input("  Pitch (俯仰角): "))
                    roll = float(input("  roll (横滚角): "))

                    print(f"\n发送云台控制命令...")
                    self._gimbal_rotate(yaw, pitch, roll)
                    print(f"✓ 云台已调整到: Yaw={yaw}°, Pitch={pitch}°, Roll={roll}°")
                
                except ValueError:
                    print("✗ 输入无效，请输入数字")
                except Exception as e:
                    print(f"✗ 云台控制失败: {e}")
            
            elif cmd == 'z':
                # 变倍/变焦/调焦控制
                print("\n=== 变倍/变焦/调焦控制 ===")
                print("  1. 变倍+ (Zoom In)")
                print("  2. 变倍- (Zoom Out)")
                print("  3. 变倍停 (Zoom Stop)")
                print("  4. 广视场/短焦 (Wide)")
                print("  5. 窄视场/长焦 (Narrow)")
                print("  6. 变焦到指定焦距 (Zoom to Focal Length)")
                print("  7. 调焦+ (Focus Near)")
                print("  8. 调焦- (Focus Far)")
                print("  9. 调焦停 (Focus Stop)")
                print(" 10. 变焦到指定视场角 (Zoom to FOV)")
                print(" 11. 变倍到指定倍率 (Zoom to Magnification)")
                print(" 12. 变倍级别控制 (1-50级)")
                print(" 13. 自动对焦 (Auto Focus)")
                print(" 14. 查询当前焦距 (Query Focal Length)")
                print("  r. 读取焦距、变倍、调焦信息")
                print("  0. 返回")
                print("=" * 30)
                
                sub_cmd = input("\n请选择 (1-11/0/r): ").strip().lower()
                
                try:
                    if sub_cmd == '1':
                        self.gimbal.zoom_in()
                        print("✓ 变倍+")
                    elif sub_cmd == '2':
                        self.gimbal.zoom_out()
                        print("✓ 变倍-")
                    elif sub_cmd == '3':
                        self.gimbal.zoom_stop()
                        print("✓ 变倍停")
                    elif sub_cmd == '4':
                        self.gimbal.focus_wide()
                        print("✓ 广视场(短焦)")
                    elif sub_cmd == '5':
                        self.gimbal.focus_narrow()
                        print("✓ 窄视场(长焦)")
                    elif sub_cmd == '6':
                        focal_length = float(input("  输入焦距 (mm): "))
                        self.gimbal.zoom_to_focal_length(focal_length)
                        print(f"✓ 变焦到指定焦距: {focal_length}mm")
                    elif sub_cmd == '7':
                        self.gimbal.focus_near()
                        print("✓ 调焦+")
                    elif sub_cmd == '8':
                        self.gimbal.focus_far()
                        print("✓ 调焦-")
                    elif sub_cmd == '9':
                        self.gimbal.focus_stop()
                        print("✓ 调焦停")
                    elif sub_cmd == '10':
                        fov = float(input("  输入视场角 (度): "))
                        self.gimbal.zoom_to_fov(fov)
                        print(f"✓ 变焦到指定视场角: {fov}°")
                    elif sub_cmd == '11':
                        mag = float(input("  输入倍率: "))
                        self.gimbal.zoom_to_magnification(mag)
                        print(f"✓ 变倍到指定倍率: {mag}x")
                    elif sub_cmd == '12':
                        # 变倍级别控制 (1-50级，1=最小倍率，50=最大倍率)
                        print("\n  变倍级别控制 (1-50级)")
                        print("  1 = 最小倍率, 50 = 最大倍率")
                        try:
                            level = int(input("  输入变倍级别 (1-50): "))
                            if 1 <= level <= 50:
                                # 假设变倍范围是1x到50x，将级别转换为倍率
                                mag = float(level)  # 1级=1x, 50级=50x
                                self.gimbal.zoom_to_magnification(mag)
                                print(f"✓ 变倍到级别 {level} (倍率: {mag}x)")
                            else:
                                print("✗ 级别必须在1-50之间")
                        except ValueError:
                            print("✗ 输入无效，请输入1-50的整数")
                    elif sub_cmd == '13':
                        # 自动对焦
                        self.gimbal.auto_focus()
                        print("✓ 自动对焦已触发")
                    elif sub_cmd == '14':
                        # 查询当前焦距和视场角
                        print("\n查询当前焦距和视场角...")
                        fov_x, fov_y, focal_x, focal_y = self.gimbal.query_focal_length()
                        if fov_x is not None and fov_y is not None:
                            print(f"✓ fov_x: {fov_x:.1f}°, fov_y: {fov_y:.1f}°")
                            print(f"✓ focal_length_x: {focal_x:.1f}mm, focal_length_y: {focal_y:.1f}mm")
                        else:
                            print("✗ 查询失败或超时")
                    elif sub_cmd == 'r':
                        self.gimbal.get_zoom_info()
                    elif sub_cmd == '0':
                        print("返回主菜单")
                    else:
                        print("✗ 无效选择")
                except ValueError:
                    print("✗ 输入无效，请输入数字")
                except Exception as e:
                    print(f"✗ 控制失败: {e}")
            
            elif cmd == 'a':
                # 自动瞄准模式
                print("\n" + "=" * 60)
                print("自动瞄准模式")
                print("=" * 60)
                try:
                    interval_str = input("  识别间隔（秒，回车=1.0）: ").strip()
                    interval = float(interval_str) if interval_str else 1.0

                    max_retries_str = input("  最大连续失败次数（回车=3）: ").strip()
                    max_retries = int(max_retries_str) if max_retries_str else 3
                except ValueError:
                    print("✗ 输入无效，使用默认值")
                    interval = 1.0
                    max_retries = 3

                print(f"\n启动自动瞄准: 间隔={interval}秒, 最大失败={max_retries}次")
                print("  按 Ctrl+C 停止自动瞄准，返回主菜单")
                print("=" * 60)

                # 调用自动瞄准（内部捕获 KeyboardInterrupt，返回交互模式）
                self.auto_aim(interval=interval, max_retries=max_retries)

                print("\n已退出自动瞄准模式，返回主菜单")
                print("=" * 60)

            else:
                print("✗ 无效命令")
    
    def start_video_stream(self, display_pose=False):
        """
        启动实时视频流显示
        
        参数:
            display_pose: 是否在视频上显示位姿信息
        """
        if self.video_running:
            print("⚠ 视频流已经在运行")
            return
        
        print("\n" + "-" * 60)
        print("启动实时视频流")
        print("-" * 60)
        
        # 打开RTSP流
        rtsp_url = self.rtsp_tv
        print(f"  连接RTSP流: {rtsp_url}")
        
        # 使用TCP协议 + 低延迟参数 (max_delay 限制 ffmpeg 内部缓冲, 防积压)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
            'rtsp_transport;tcp|max_delay;500000|probesize;32768|analyzeduration;0'
        self.video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # 设置缓冲区大小（减少延迟）
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.video_capture.isOpened():
            print(f"  ✗ 无法打开RTSP流")
            self.video_capture = None
            return
        
        print(f"  ✓ 视频流已打开")
        print(f"  分辨率: {self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  帧率: {self.video_capture.get(cv2.CAP_PROP_FPS)} FPS")
        
        # 设置标志
        self.video_running = True
        self.video_paused = False
        self.show_pose_on_video = display_pose
        # 复位推理状态 (支持 stop → start 重开后 [c] 仍可用)
        self._est_busy = False
        self._inf_thread = None
        self._inf_queue = None
        
        # 读帧线程 (只读 RTSP 不编码, 最新帧覆盖 → 永不积压)
        self._latest_frame = None
        self._latest_frame_ts = 0.0   # 最新帧到达时间戳 (云台对齐时判断是否为新帧)
        # 云台旋转开始时间戳 (>0 时推流跳过 ts < 该值的旧帧, 防止旋转前 RTSP 缓冲里的旧帧
        # 被当成实时图推上第一窗口, 与第三窗口 aligned.png 实拍画面不一致 → "误差太大")
        self._rotation_pending_ts = 0.0
        self._reader_thread = threading.Thread(target=self._video_reader_loop, daemon=True)
        self._reader_thread.start()

        # 推流线程 (按固定帧率编码最新帧 + 处理按键)
        self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
        self.video_thread.start()
        
        # 注入网页 截图/状态 provider
        global _http_snap_provider, _http_status_provider

        def _snap_provider():
            with self.frame_lock:
                f = self._latest_frame
            if f is None:
                return b""
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return jpg.tobytes() if ok else b""

        def _status_provider():
            try:
                gy, gp, _ = self.gimbal.get_current_angles()
            except Exception:
                gy, gp = 0.0, 0.0
            return (f"Yaw={gy:.1f}° Pitch={gp:.1f}° | 步长={self.gimbal_step:.0f}° | "
                    f"位姿记录={len(getattr(self, 'pose_records', []))} | "
                    f"{'暂停' if self.video_paused else '运行'}")

        _http_snap_provider = _snap_provider
        _http_status_provider = _status_provider

        # 注入 机器狗控制 provider (网页按钮 → /dog/<action>)
        global _http_dog_provider
        _http_dog_provider = self._handle_dog_action

        # 启动 HTTP 视频流（替代 cv2.imshow，板子无 GUI 时使用）
        if not hasattr(self, '_http_stream_started'):
            self._http_stream_started = True
            self._http_port = 8888
            start_http_stream(port=self._http_port, owner=self)
            print(f"\n  HTTP 控制台已启动")
            print(f"  浏览器打开: http://<板子IP>:{self._http_port}/")
            print(f"  原图流: http://<板子IP>:{self._http_port}/raw")
            print(f"  检测流: http://<板子IP>:{self._http_port}/detect")
            print(f"  截图:   http://<板子IP>:{self._http_port}/snap")
            print(f"  3D:     http://<板子IP>:{self._http_port}/3d")
        print(f"\n  视频流已启动（HTTP MJPEG）")
        print(f"  网页/键盘按键: q=退出, p=暂停, c=捕获位姿, s=位姿显示, g=生成3D")
        print(f"  云台: 方向键=微调, +/-=步长, a/d=变倍, f=对焦, r=回中")
        print("-" * 60)

    # ── 云台快捷控制 (网页按钮) ──
    def _gimbal_manual_adjust(self, dyaw=0.0, dpitch=0.0):
        """网页方向按钮: 按当前步长微调云台"""
        try:
            gy, gp, _ = self.gimbal.get_current_angles()
            ny = max(-180.0, min(180.0, gy + dyaw))
            np_ = max(-90.0, min(90.0, gp + dpitch))
            self._gimbal_rotate(ny, np_, 0.0)
            print(f"[云台] 微调 → Yaw={ny:.1f}° Pitch={np_:.1f}° (步长 {self.gimbal_step:.0f}°)")
        except Exception as e:
            print(f"  ✗ 云台微调失败: {e}")

    def _gimbal_home(self):
        """云台回中 (0, 0)"""
        try:
            self._gimbal_rotate(0.0, 0.0, 0.0)
            print("[云台] 回中 (0, 0)")
        except Exception as e:
            print(f"  ✗ 云台回中失败: {e}")

    # ═══════════════════════════════════════════════════════════
    # 机器狗网页控制 (robot_interface.Dog, UDP → 192.168.1.103:43893)
    # ═══════════════════════════════════════════════════════════
    def _get_dog(self):
        """懒加载机器狗实例 (首次调用时连接)"""
        if getattr(self, '_dog', None) is None:
            from robot_interface import Dog
            ip = '192.168.1.103'
            try:
                cfg = getattr(self, '_cfg', None) or {}
                ip = cfg.get('dog_navigator', {}).get('ip', ip)
            except Exception:
                pass
            self._dog = Dog(ip=ip)
            self._dog_ip = ip
            print(f"[狗] 实例已创建: {ip}")
        return self._dog

    def _handle_dog_action(self, action, value=0.0, gait=None):
        """网页机器狗控制入口

        action: stand=起立连接 (心跳+起立+手动+力控+开始运动+gait)
                force=②力控 (单独补发, 缺它移动指令被运控丢弃)
                motion=③开始运动 (单独补发, 缺它狗腿不动)
                down=趴下复位 / damp=阻尼 / stop=停止并断开
                fwd=前进 / back=后退 / left=左移 / right=右移 / tl=左转 / tr=右转
        value:  步长 (平移单位 m, 旋转单位 °)

        安全说明:
          - 单步模式: 每个移动按钮 = 一次固定步长 (默认 0.2m / 15°), 不会连续运动
          - 安全锁: 单次平移 ≤ DOG_MAX_MOVE_M, 单次旋转 ≤ DOG_MAX_TURN_DEG, 超限拒绝
          - 力控使能: 狗需先「起立」并「力控 + 开始运动」(对应遥控器同名两个按钮) 才能移动;
                      未使能时若 DOG_AUTO_INIT=True, 会自动起立+力控+开始运动后再执行这一步
        """
        act = str(action).lower().strip()
        v = float(value or 0.0)
        try:
            dog = self._get_dog()
            if act in ('stand', 'init', 'connect'):
                dog.init()                      # 起立→手动→力控→步态 (约3s)
                print("[狗] 起立+力控完成, 运动就绪")
                return f"✓ 已起立并力控 ({getattr(self, '_dog_ip', '')})"
            if act in ('force', 'force_control'):
                dog.force_control()              # 单独补发「力控」(第②步, 缺它移动指令被丢)
                print("[狗] 力控 (第②步)")
                return "✓ 已发送「力控」, 还需点「③ 开始运动」腿才会动"
            if act in ('motion', 'start', 'start_motion', 'enable'):
                # 可选: 同时切步态 (解决原地踏步不前进), 由 route 把 ?gait= 解析后传入
                _gc = gait
                dog.start_motion()                # 单独补发「开始运动」(第③步, 启动步态循环)
                if _gc:
                    dog.set_gait(_gc)             # 顺便切行走 gait
                print(f"[狗] 开始运动 (第③步), gait={_gc}")
                msg = "✓ 已发送「开始运动」, 狗应原地踏步"
                if _gc:
                    msg += f" (gait=0x{_gc:08X})"
                return msg
            if act in ('down', 'liedown', 'crouch'):
                dog.stand_down()                # 趴下复位 (保持连接)
                print("[狗] 已趴下复位")
                return "✓ 已趴下复位"
            if act in ('damp', 'soft'):
                dog.damp()                      # 阻尼模式
                print("[狗] 已进入阻尼模式")
                return "✓ 已进入阻尼模式 (可搬动)"
            if act in ('stop', 'close', 'halt'):
                dog.close()                     # 三轴归零 + 趴下复位 + 关闭 socket
                self._dog = None
                print("[狗] 已停止, 趴下复位并断开")
                return "✓ 已停止并趴下复位"
            # 平移 / 旋转 (带安全锁)
            m = v if v > 1e-6 else DOG_DEFAULT_MOVE_M
            deg = v if v > 1e-6 else DOG_DEFAULT_TURN_DEG
            lock_m = DOG_MAX_MOVE_M
            lock_deg = DOG_MAX_TURN_DEG

            # ── 运动使能检查: 未起立/未力控/未开始运动时, 自动初始化 ──
            prefix = ""
            if not getattr(dog, 'ready', True):
                if DOG_AUTO_INIT:
                    print("[狗] 未使能运动, 自动起立 + 力控 + 开始运动 ...")
                    dog.init()
                    prefix = "✓ 自动起立+力控+开始运动; "
                else:
                    return "✗ 尚未开始运动: 请先按「① 起立」→「② 力控」→「③ 开始运动」"

            def _do(x=0.0, y=0.0, z=0.0, label=""):
                ok = dog.move(x=x, y=y, z=z, max_move_m=lock_m, max_turn_deg=lock_deg)
                if not ok:
                    return f"✗ {label} 被安全锁拒绝 (上限 {lock_m}m / {lock_deg}°)"
                return f"✓ {label}"

            if act in ('fwd', 'forward'):
                return prefix + _do(x=m, label=f"前进一步 {m:.2f} m")
            if act in ('back', 'backward'):
                return prefix + _do(x=-m, label=f"后退一步 {m:.2f} m")
            if act == 'left':
                return prefix + _do(y=m, label=f"左移一步 {m:.2f} m")
            if act == 'right':
                return prefix + _do(y=-m, label=f"右移一步 {m:.2f} m")
            if act in ('tl', 'turn_left'):
                return prefix + _do(z=-deg, label=f"左转 {deg:.0f}°")
            if act in ('tr', 'turn_right'):
                return prefix + _do(z=deg, label=f"右转 {deg:.0f}°")
            return f"未知动作: {act}"
        except ImportError as e:
            print(f"[狗] robot_interface 导入失败: {e}")
            return f"✗ 缺少 robot_interface: {e}"
        except Exception as e:
            print(f"[狗] 动作 {act} 失败: {e}")
            return f"✗ {act} 失败: {e}"

    def _gimbal_rotate(self, yaw: float, pitch: float, roll: float = 0.0):
        """发送云台角度命令, 并标记旋转开始时间 → 第一窗口推流线程过滤旋转前旧帧。

        所有云台旋转的『唯一入口』。调用后:
          - self._rotation_pending_ts = time.time()  (推流线程见此标记会跳过 ts 更早的帧)
          - 待 RTSP 缓冲排空 (通常 0.5~0.7s), 推流线程拿到 ts >= _rotation_pending_ts 的新帧
            后自动清标记, 恢复实时推流。
        """
        try:
            self.gimbal.follow(yaw, pitch, roll)
            self._rotation_pending_ts = time.time()
        except Exception as e:
            print(f"  ✗ 云台旋转失败: {e}")

    def _init_viz3d_sync(self):
        """同步初始化 Plotly 3D 可视化器 (主程序启动时调用)。

        - 若已装 plotly → 直接 import visualizer_3d → _viz3d 就绪
        - 若没装 → 启动后台线程自动 `pip install plotly` (用清华源, 避免 SSL),
                    本方法直接返回, _viz3d 暂时 None → render 走 matplotlib fallback。
                    装好后后台线程会自动重试 import, _viz3d 立即可用。

        关键: 同步只做"快速检测", 不阻塞主流程。
        """
        global _HAS_PLOTLY, FrustumVisualizer
        import sys as _sys2
        print(f"[3D-INIT] sys.executable = {_sys2.executable}")
        try:
            import plotly as _pl
            print(f"[3D-INIT] plotly available: {_pl.__version__} @ {_pl.__file__}")
        except Exception as _e:
            print(f"[3D-INIT] plotly NOT available: {_e}")
        # 1) 快速尝试直接 import visualizer_3d
        try:
            from visualizer_3d import FrustumVisualizer as _FV
            FrustumVisualizer = _FV
            _HAS_PLOTLY = True
            self._viz3d = _FV(config=self._cfg)
            print("[3D] ✓ Plotly 已加载 (视觉器 FrustumVisualizer 就绪)")
            return
        except ImportError as _e:
            print(f"[3D] 初次 import 失败: {_e}")
        # 2) plotly 缺失 → 后台线程尝试安装 + 重新初始化 (不阻塞主程序)
        print(f"[3D] 启动后台线程自动安装 plotly ...")
        threading.Thread(target=self._bg_install_plotly, daemon=True).start()

    def _bg_install_plotly(self):
        """后台线程: pip install plotly (清华源) → 重新初始化 viz3d。

        安装成功后, _viz3d 会被赋值, 之后按 [c] 就能用 Plotly。
        """
        global _HAS_PLOTLY, FrustumVisualizer
        import subprocess, sys
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                 "--trusted-host", "pypi.tuna.tsinghua.edu.cn",
                 "plotly"],
                capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                print(f"[3D-BG] 清华源失败, 尝试默认源 ...")
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", "plotly"],
                    capture_output=True, text=True, timeout=180,
                )
            if r.returncode != 0:
                print(f"[3D-BG] ✗ 安装 plotly 失败:\n{r.stderr or r.stdout}")
                print(f"[3D-BG]   手动安装: {sys.executable} -m pip install plotly")
                return
            print(f"[3D-BG] ✓ plotly 安装完成, 重新初始化 viz3d ...")
            from visualizer_3d import FrustumVisualizer as _FV
            FrustumVisualizer = _FV
            _HAS_PLOTLY = True
            self._viz3d = _FV(config=self._cfg)
            print(f"[3D-BG] ✓ viz3d 就绪, 之后按 [c] 会走 Plotly HTML 路径")
        except Exception as _e:
            print(f"[3D-BG] ✗ 安装/初始化失败: {_e}")

    def _start_3d_render_async(self):
        """后台线程渲染 3D 场景

        严格按 QR_3D/robot_dog_bridge_win_v2.py 流程:
          pose_before = pd (PnP 原始, 红视锥)
            ↓ compute_gimbal_target(dx,dy,fov,W,H,gimbal)
          (dyaw, dpitch, target_yaw, target_pitch)
            ↓ build_aligned_pose(pd, dyaw, dpitch)
          aligned_pose (蓝视锥, rmat 已重建)
            ↓ self._viz3d.render_html(pd, aligned_pose, fov_x, W, H)
          Plotly HTML

        Plotly 不可用时 fallback 到 matplotlib PNG。
        """
        pd = getattr(self, '_first_pose_data', None) or getattr(self, 'latest_pose_data', None)
        if pd is None:
            self._set_3d_html_msg("无位姿数据: 先按 [c] 捕获位姿")
            print("[3D] 无位姿数据")
            return

        # ── 严格按 QR_3D 流程计算 FOV / 图像尺寸 ──
        cam = pd.get('camera_matrix')
        if cam is not None:
            fx = float(cam[0, 0])
            # PnP 时记录过 img_w/img_h; 兜底用 1280x720
            Ww = int(pd.get('img_w', 1280))
            Hh = int(pd.get('img_h', 720))
            # FOV 用内参反算 (避免默认 60° 在 zoom 时不准)
            fov_x_actual = 2.0 * math.degrees(math.atan(Ww / (2.0 * fx)))
        else:
            Ww, Hh = 1280, 720
            fov_x_actual = self.fov_x

        # ── 路径 1: Plotly HTML (有 viz3d) ──
        if self._viz3d is not None:
            self._set_3d_html_msg(f"Plotly 3D 生成中... ({time.strftime('%H:%M:%S')})")
            print(f"[3D] 开始 Plotly 渲染 (FOV={fov_x_actual:.1f}°, {Ww}x{Hh})...")

            def _run():
                try:
                    # 严格按 robot_dog_bridge_win_v2.py 第 1097-1102 行的流程:
                    #   1) compute_gimbal_target → (dyaw, dpitch, _, _)
                    #   2) build_aligned_pose    → aligned_pose (蓝视锥, rmat 已重建)
                    #   3) render_html(pd, aligned_pose, fov_x, W, H)
                    img_pts = pd.get('image_points')
                    if img_pts is None or len(img_pts) < 4:
                        raise RuntimeError("pd 中无 image_points")
                    qr_cx = float(np.mean(img_pts[:, 0]))
                    qr_cy = float(np.mean(img_pts[:, 1]))
                    dx = qr_cx - Ww / 2.0
                    dy = qr_cy - Hh / 2.0

                    dyaw, dpitch, target_yaw, target_pitch = compute_gimbal_target(
                        dx, dy, fov_x_actual, Ww, Hh,
                        gimbal=self.gimbal,
                        angle_dead_zone_deg=0.1,
                    )
                    aligned_pose = build_aligned_pose(pd, dyaw, dpitch)
                    print(f"[3D]   pose_before: Yaw={pd.get('yaw', 0):.1f}° "
                          f"Pitch={pd.get('pitch', 0):.1f}°")
                    print(f"[3D]   中心补偿: ΔYaw={dyaw:+.2f}° ΔPitch={dpitch:+.2f}°")
                    print(f"[3D]   aligned_pose: Yaw={aligned_pose.get('yaw', 0):.1f}° "
                          f"Pitch={aligned_pose.get('pitch', 0):.1f}° "
                          f"Roll={aligned_pose.get('roll', 0):.1f}°")

                    html = self._viz3d.render_html(pd, aligned_pose, fov_x_actual, Ww, Hh)
                    with self._3d_html_lock:
                        self._3d_html = html
                        self._3d_html_ver += 1
                    self._set_3d_html_msg(
                        f"已生成 ({time.strftime('%H:%M:%S')}) | FOV={fov_x_actual:.1f}°")
                    print(f"[3D] ✓ Plotly HTML 已生成 ({len(html)} chars)")
                except Exception as e:
                    self._set_3d_html_msg(f"渲染失败: {e}")
                    print(f"[3D] ✗ Plotly 渲染失败: {e}")
                    import traceback
                    traceback.print_exc()
            threading.Thread(target=_run, daemon=True).start()
            return

        # ── 路径 2: matplotlib PNG fallback (PC 端无 plotly 时) ──
        _set_3d_png(msg=f"matplotlib 3D 生成中 ({time.strftime('%H:%M:%S')})...")
        print(f"[3D] Plotly 不可用, 用 matplotlib fallback...")

        def _run_mpl():
            try:
                png = self.visualize_3d_scene(
                    list(getattr(self, 'pose_records', [])) or [pd],
                    headless=True,
                )
                if png:
                    _set_3d_png(png, "已生成: matplotlib fallback")
                    print(f"[3D] ✓ matplotlib PNG 已生成 ({len(png)} bytes)")
                else:
                    _set_3d_png(msg="渲染失败 (matplotlib 也失败)")
            except Exception as e:
                _set_3d_png(msg=f"渲染失败: {e}")
                print(f"[3D] ✗ matplotlib 失败: {e}")
        threading.Thread(target=_run_mpl, daemon=True).start()

    def _set_3d_html_msg(self, msg: str):
        """更新 3D HTML 提示消息 (供 /3d_msg 端点轮询)。"""
        with self._3d_html_lock:
            self._3d_html_msg = msg

    def _video_reader_loop(self):
        """读帧线程: 只负责读 RTSP, 覆盖式存最新帧 (读/处理分离 → 永不积压)。"""
        fail = 0
        while self.video_running:
            if self.video_capture is None:
                time.sleep(0.1)
                continue
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                fail += 1
                time.sleep(0.05)
                if fail == 20:
                    print("⚠ RTSP 无帧 (中断?)")
                continue
            fail = 0
            with self.frame_lock:
                self.current_frame = frame
                self._latest_frame = frame   # 覆盖旧帧 (丢帧保实时)
                self._latest_frame_ts = time.time()   # 该帧『到达』时间戳 (用于判断新帧)

    def _video_display_loop(self):
        """推流线程: 按固定帧率取『最新帧』→ 叠加 → 编码推流 → 处理按键。

        位姿估计(耗时数秒)放后台线程, 不阻塞推流和按键 → 网页不卡、功能不失灵。
        """
        frame_count = 0
        interval = 1.0 / max(1.0, _http_fps)
        next_t = time.time()

        try:
            while self.video_running:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                    continue
                next_t = time.time() + interval

                # ── 按键 (暂停时也处理, 保证 p/q 始终有效) ──
                key = _get_http_key()

                # 方向键 (网页按钮, 多字符按键) → 云台微调
                if isinstance(key, str):
                    _dir = {'left': (-1, 0), 'right': (1, 0),
                            'up': (0, 1), 'down': (0, -1)}.get(key)
                    if _dir:
                        self._gimbal_manual_adjust(_dir[0] * self.gimbal_step,
                                                   _dir[1] * self.gimbal_step)
                    key = -1   # 已处理

                if self.video_paused:
                    if key == ord('p') or key == ord('P'):
                        self.video_paused = False
                        print("\n继续视频")
                    elif key == ord('q') or key == ord('Q'):
                        break
                    else:
                        # 暂停只暂停"捕获/检测", 画面仍取最新帧实时推流,
                        # 否则云台转动时原图窗口会冻结不刷新
                        with self.frame_lock:
                            pf = self._latest_frame
                        if pf is not None:
                            pf = pf.copy()
                            cv2.putText(pf, "PAUSED - [p] resume", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            update_stream_frame(pf, None)
                    continue

                # ── 取最新帧 (旧帧已被覆盖, 不积压) ──
                with self.frame_lock:
                    frame = self._latest_frame
                    frame_ts = self._latest_frame_ts
                if frame is None:
                    continue

                # ── 旋转恢复期过滤 ──
                # 云台旋转后, RTSP 缓冲里还残留 ~0.5s 的旧画面, 推流线程若直接用就
                # 会把"旋转前"的旧帧当成实时图推上第一窗口, 与第三窗口 aligned.png
                # 实拍不一致 → "误差太大"。_gimbal_rotate() 已记下旋转开始时间戳,
                # 此处跳过 ts 更早的帧, 50ms 后再试; 等 RTSP 排空拿到 ts >= 标记
                # 的新帧后再恢复正常推流 (第一窗口直接显示旋转后画面, 与第三窗口对齐)。
                if self._rotation_pending_ts > 0:
                    if frame_ts < self._rotation_pending_ts:
                        next_t = time.time() + 0.05
                        continue
                    # 新帧已到 → 清除标记, 恢复实时推流
                    dt_ms = (time.time() - self._rotation_pending_ts) * 1000
                    print(f"[推流] 旋转后新帧到位 (恢复期 {dt_ms:.0f}ms), "
                          f"恢复实时窗口第一窗口")
                    self._rotation_pending_ts = 0.0

                frame_count += 1
                display_frame = frame.copy()

                # 在视频上显示位姿信息（仅在原窗口显示文本）
                if self.show_pose_on_video and self.latest_pose_data is not None:
                    pose_data = self.latest_pose_data

                    # 显示位姿文本信息
                    info_y = 30
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(display_frame, f"Yaw: {pose_data.get('yaw', 0):.1f}°",
                               (10, info_y), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Pitch: {pose_data.get('pitch', 0):.1f}°",
                               (10, info_y + 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Dist: {pose_data.get('distance', 0):.2f}m",
                               (10, info_y + 60), font, 0.7, (0, 255, 0), 2)

                # 实时云台角度 (每帧刷新, 云台转动立即可见, 不依赖位姿捕获)
                try:
                    gy, gp, _ = self.gimbal.get_current_angles()
                    cv2.putText(display_frame, f"Gimbal: Yaw={gy:.1f} Pitch={gp:.1f}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except Exception:
                    pass

                # 显示帧计数和状态
                cv2.putText(display_frame, f"Frame: {frame_count}",
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 先确定 detection_frame
                if self.review_mode and hasattr(self, 'review_frame_with_yolo'):
                    # 使用带YOLO框的图像（从image_init_detect保存）
                    detection_frame = self.review_frame_with_yolo.copy()

                    # 在 detection 窗口上绘制坐标轴
                    if self.show_pose_on_video and self.latest_pose_data is not None:
                        pose_data = self.latest_pose_data
                        try:
                            H, W = detection_frame.shape[:2]
                            # 关键: 必须用 PnP 时用的同一份 camera_matrix, 否则
                            #   世界原点 (QR 中心) 投影会偏移, 看起来"轴不在 QR 中心"
                            # 只有 pose_data 里没存 camera_matrix 才退回 fov_x 默认
                            if 'camera_matrix' in pose_data and pose_data['camera_matrix'] is not None:
                                camera_matrix = np.asarray(pose_data['camera_matrix'], dtype=np.float64)
                            else:
                                camera_matrix = pose_estimation.create_camera_matrix((W, H), fov_x=self.fov_x)
                            dist_coeffs = np.zeros((5, 1), dtype=np.float32)

                            # 使用最新的位姿数据绘制坐标轴
                            if 'rvec' in pose_data and 'tvec' in pose_data:
                                rvec = pose_data['rvec'].reshape(3, 1)
                                tvec = pose_data['tvec'].reshape(3, 1)
                                detection_frame = pose_estimation.draw_axes(
                                    detection_frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05
                                )
                        except Exception:
                            pass  # 忽略绘制错误

                    # 显示提示信息
                    cv2.putText(detection_frame, "Press 'y' to reshoot, 'n' to continue",
                               (10, detection_frame.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # 正常模式：复制推流数据
                    detection_frame = display_frame.copy()

                # HTTP 流推送 (只推原图 MJPEG; 检测/对齐窗口改为静态图 + 轮询, 减带宽)
                update_stream_frame(display_frame, None)

                # ── 按键处理 ──
                if key == ord('q') or key == ord('Q'):
                    # 退出视频
                    break
                elif key == ord('p') or key == ord('P'):
                    # 暂停
                    self.video_paused = True
                    print("\n暂停视频")
                elif key == ord('s') or key == ord('S'):
                    # 显示/隐藏位姿信息
                    self.show_pose_on_video = not self.show_pose_on_video
                    print(f"\n位姿信息显示: {'开启' if self.show_pose_on_video else '关闭'}")
                elif key == ord('a') or key == ord('A'):
                    # 变倍+ (步进: 走到目标倍率就停, 不会无限持续)
                    try:
                        self.current_zoom_mag = min(self._zoom_max,
                                                    self.current_zoom_mag + self.zoom_step_val)
                        self.gimbal.zoom_to_magnification(self.current_zoom_mag)
                        print(f"[云台] 变倍+ → {self.current_zoom_mag:.1f}x")
                    except Exception as e:
                        print(f"  ✗ 变倍失败: {e}")
                elif key == ord('d') or key == ord('D'):
                    # 变倍- (步进: 走到目标倍率就停, 不会无限持续)
                    try:
                        self.current_zoom_mag = max(self._zoom_min,
                                                    self.current_zoom_mag - self.zoom_step_val)
                        self.gimbal.zoom_to_magnification(self.current_zoom_mag)
                        print(f"[云台] 变倍- → {self.current_zoom_mag:.1f}x")
                    except Exception as e:
                        print(f"  ✗ 变倍失败: {e}")
                elif key == ord('f') or key == ord('F'):
                    # 自动对焦
                    try:
                        self.gimbal.auto_focus(); print("[云台] 自动对焦")
                    except Exception as e:
                        print(f"  ✗ 对焦失败: {e}")
                elif key == ord('r') or key == ord('R'):
                    # 云台回中
                    self._gimbal_home()
                elif key == ord('g') or key == ord('G'):
                    # 生成 3D 场景 (后台线程)
                    self._start_3d_render_async()
                elif key == ord('+') or key == ord('='):
                    # 步长+
                    self.gimbal_step = min(45.0, self.gimbal_step + 5.0)
                    print(f"[云台] 步长 = {self.gimbal_step:.0f}°")
                elif key == ord('-') or key == ord('_'):
                    # 步长-
                    self.gimbal_step = max(1.0, self.gimbal_step - 5.0)
                    print(f"[云台] 步长 = {self.gimbal_step:.0f}°")
                elif self.review_mode:
                    # 在 review 模式下，检查用户按键
                    # [c] 在此等价于 [y] 重拍, 避免 review 模式下按 [c] "无反应"死锁
                    if key == ord('y') or key == ord('Y') or key == ord('c') or key == ord('C'):
                        # 重新截图
                        print("\n  用户选择重新截图...")
                        self.review_mode = False
                        if hasattr(self, 'review_detections'):
                            delattr(self, 'review_detections')
                        # 设置标志，在下一轮循环中重新捕获
                        self.need_recapture = True
                    elif key == ord('n') or key == ord('N'):
                        # 继续
                        print("\n  用户选择继续...")
                        self.review_mode = False
                        if hasattr(self, 'review_detections'):
                            delattr(self, 'review_detections')

                # 检测放后台线程 (多帧采集+YOLO+PnP 耗时数秒, 不能阻塞推流)
                if self.need_recapture:
                    self.need_recapture = False
                    print(f"\n[视频] 重新捕获当前帧进行位姿估计...")
                    _set_panel_status("重新捕获中: 多帧采集 + YOLO + PnP ...")
                    self._start_estimate_async(frame)
                elif not self.review_mode and (key == ord('c') or key == ord('C')):
                    # 捕获当前帧并估计位姿
                    print(f"\n[视频] 捕获当前帧进行位姿估计...")
                    _set_panel_status("位姿估计中: 多帧采集 + YOLO + PnP (约 3~5 秒) ...")
                    self._start_estimate_async(frame)

        except Exception as e:
            print(f"✗ 视频显示错误: {e}")

        finally:
            print("\n视频流已停止")

    def _start_estimate_async(self, frame):
        """把位姿估计投递到『常驻推理线程』执行 (不阻塞视频推流和按键处理)。

        为什么用常驻线程, 而不是每次 [c] 都新起一个线程:
          1. Ascend ACL 的 device/context 与线程绑定 —— 加载 .om 的线程必须
             就是执行推理的线程, 否则 acl.rt.memcpy 直接报 107002
             (copy input host to device failed);
          2. 若每次 [c] 新建线程, 每个线程都会各自加载一份 ACL 模型并占用
             设备显存 (不会自动释放), 按几次就把 NPU 显存耗光;
             常驻线程只加载一次并复用。
        """
        if self._est_busy:
            print("  ⚠ 位姿估计正在进行中, 忽略重复触发")
            _set_panel_status("位姿估计进行中, 请等待本次完成...")
            return

        # 首次调用时拉起常驻推理线程
        if self._inf_thread is None or not self._inf_thread.is_alive():
            self._inf_queue = queue.Queue(maxsize=1)
            self._inf_thread = threading.Thread(
                target=self._inference_loop, daemon=True,
                name="pose-inference")
            self._inf_thread.start()

        self._est_busy = True
        try:
            self._inf_queue.put(frame.copy(), timeout=1.0)
        except Exception as e:
            self._est_busy = False
            print(f"  ⚠ 位姿估计任务投递失败: {e}")
            _set_panel_status("位姿估计任务投递失败, 请重新按 [c]")

    def _inference_loop(self):
        """常驻推理线程主体: 串行执行位姿估计 (ACL 加载与推理始终同线程)。

        注意: 该线程第一次调用 _get_detector() 时才会真正加载 .om 模型,
              加载后的模型通过 threading.local 只属于本线程, 后续复用。
        """
        while True:
            try:
                f = self._inf_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.video_running:
                    self._est_busy = False   # 退出前复位, 避免下次 [c] 被误判为"忙"
                    break
                continue
            if f is None:
                self._est_busy = False
                break
            try:
                self._capture_and_estimate(f)
            except Exception as e:
                print(f"✗ 位姿估计异常: {e}")
                _set_panel_status(f"位姿估计失败: {e}")
            finally:
                self._est_busy = False
    
    def _capture_and_estimate(self, frame):
        """多帧滑动窗口 + EMA 平滑位姿估计

        方案:
          - 连续采集 N 帧 (默认 5 帧)，逐帧做 QR 检测 + PnP 估计
          - 对 camera_position 做 EMA (指数移动平均)
          - 对 yaw/pitch/roll 做加权平均
          - 置信度低的帧降权
          - 只有当前窗口超过半数帧成功才输出结果
        """
        window_size = self._pose_window_size
        alpha = self._pose_ema_alpha

        print(f"\n{'='*50}")
        print(f"多帧位姿采集 (窗口={window_size}帧, EMA α={alpha})")
        print(f"{'='*50}")

        raw_poses = []  # [(pose_data, confidence, is_success), ...]

        # ── 阶段 1: 连续采集 window_size 帧 (从读帧线程取最新帧, 不抢 RTSP) ──
        for i in range(window_size):
            if i == 0:
                frame_i = frame.copy()
            else:
                print(f"    等待下一帧 ({i+1}/{window_size}) ...")
                time.sleep(0.15)  # 帧间隔 ~150ms
                with self.frame_lock:
                    latest = self._latest_frame
                frame_i = latest.copy() if latest is not None else None
                if frame_i is None:
                    print(f"    ✗ 帧 {i+1} 读取失败，跳过")
                    raw_poses.append((None, 0.0, False))
                    continue

            temp_path = os.path.join(self.output_dir, f"temp_capture_{i}.jpg")
            cv2.imwrite(temp_path, frame_i)

            success, pose_data, _ = self.estimate_pose_from_image(temp_path, debug=False)
            if success and pose_data is not None:
                conf = float(pose_data.get("confidence", 0.5))
                raw_poses.append((pose_data, conf, True))
                print(f"    帧 {i+1}: ✓ conf={conf:.3f} "
                      f"pos=({pose_data['camera_position'][0]:.2f},{pose_data['camera_position'][1]:.2f},{pose_data['camera_position'][2]:.2f})")
            else:
                raw_poses.append((None, 0.0, False))
                print(f"    帧 {i+1}: ✗ 位姿估计失败")

        # ── 阶段 2: 过滤 & 平滑 ──
        valid = [(p, c) for p, c, ok in raw_poses if ok]
        success_count = len(valid)

        if success_count < max(1, window_size // 2):
            print(f"\n✗ 有效帧不足 ({success_count}/{window_size})，位姿估计失败")
            self._pose_history.clear()
            return

        # 置信度归一化作为权重
        weights = np.array([c for _, c in valid])
        weights = weights / (weights.sum() + 1e-8)

        # EMA 平滑 camera_position
        if self._pose_history:
            prev_pos = np.array(self._pose_history[-1][0]["camera_position"])
            prev_yaw = float(self._pose_history[-1][0].get("yaw", 0))
            prev_pitch = float(self._pose_history[-1][0].get("pitch", 0))
            prev_roll = float(self._pose_history[-1][0].get("roll", 0))
        else:
            prev_pos = None

        # 当前窗口加权平均
        positions = np.array([p["camera_position"] for p, _ in valid])
        yaws = np.array([p.get("yaw", 0) for p, _ in valid])
        pitches = np.array([p.get("pitch", 0) for p, _ in valid])
        rolls = np.array([p.get("roll", 0) for p, _ in valid])

        weighted_pos = np.average(positions, axis=0, weights=weights)
        weighted_yaw = np.average(yaws, weights=weights)
        weighted_pitch = np.average(pitches, weights=weights)
        weighted_roll = np.average(rolls, weights=weights)

        # EMA 融合历史
        if prev_pos is not None:
            fused_pos = alpha * weighted_pos + (1 - alpha) * prev_pos
            fused_yaw = alpha * weighted_yaw + (1 - alpha) * prev_yaw
            fused_pitch = alpha * weighted_pitch + (1 - alpha) * prev_pitch
            fused_roll = alpha * weighted_roll + (1 - alpha) * prev_roll
        else:
            fused_pos = weighted_pos
            fused_yaw = weighted_yaw
            fused_pitch = weighted_pitch
            fused_roll = weighted_roll

        # 构建平滑后的 pose_data（使用第一个有效帧的其他字段）
        base_pose = valid[0][0].copy()
        base_pose["camera_position"] = fused_pos.tolist()
        base_pose["yaw"] = float(fused_yaw)
        base_pose["pitch"] = float(fused_pitch)
        base_pose["roll"] = float(fused_roll)
        base_pose["distance"] = float(np.linalg.norm(fused_pos))
        base_pose["raw_count"] = success_count
        base_pose["window_size"] = window_size
        base_pose["ema_alpha"] = alpha

        # 保存历史
        self._pose_history.append((base_pose, float(np.mean([c for _, c in valid]))))
        if len(self._pose_history) > 10:
            self._pose_history.pop(0)

        # 更新最新数据
        self.latest_pose_data = base_pose

        # 同时保存『首帧原始位姿』(等同 QR_3D auto_aim 的单帧估计)
        # 用于 [c] 流程的云台旋转计算 → 与 QR_3D 源代码逻辑严格一致
        #   - 不取 10 帧 EMA 平均值 (会把小角度"平滑掉", 旋转量偏小)
        #   - 不取最后一帧 (gimbal 旋转过程中可能引入滞后帧)
        #   - 只取第一帧, 相当于『对原始帧做单帧 PnP』
        try:
            self._first_pose_data = valid[0][0].copy()
        except Exception:
            pass

        # 收集位姿记录 → 网页 3D 显示 (点击 [3D 显示] / 按 g 渲染)
        try:
            self.pose_records.append(base_pose)
            if len(self.pose_records) > 300:   # 上限, 防内存增长
                self.pose_records.pop(0)
        except Exception:
            pass

        # ── 自动重渲 3D 窗口 (节流: 1.5s 内最多一次, 不阻塞 [c] 流程) ──
        try:
            now_ts = time.time()
            last = getattr(self, '_last_3d_render_ts', 0.0)
            if now_ts - last >= 1.5:
                self._last_3d_render_ts = now_ts
                self._start_3d_render_async()
        except Exception:
            pass

        # ── 阶段 3: 打印 & 保存 ──
        std_pos = np.std(positions, axis=0) if len(positions) > 1 else np.zeros(3)
        print(f"\n  窗口统计 ({success_count}/{window_size} 有效):")
        print(f"    位置 STD: X={std_pos[0]:.4f} Y={std_pos[1]:.4f} Z={std_pos[2]:.4f} m")
        print(f"    融合后位置: X={fused_pos[0]:.4f} Y={fused_pos[1]:.4f} Z={fused_pos[2]:.4f} m")
        print(f"    融合后角度: Yaw={fused_yaw:.2f}° Pitch={fused_pitch:.2f}° Roll={fused_roll:.2f}°")

        self.print_pose_info(base_pose)

        timestamp = int(time.time())
        output_prefix = os.path.join(self.output_dir, f"video_pose_{timestamp}")
        # 用第一帧画结果图
        temp_first = os.path.join(self.output_dir, "temp_capture_0.jpg")
        result_image = cv2.imread(temp_first)
        if result_image is not None:
            self.save_result(result_image, base_pose, output_prefix)

        print(f"\n✓ 多帧位姿估计完成 ({success_count}/{window_size} 帧有效)")

        # 启动 review 模式
        saved_frame = cv2.imread(temp_first)
        if saved_frame is not None:
            self._start_review_mode(saved_frame)
    
    def _grab_fresh_frame(self, settle: float = 0.30, timeout: float = 3.0, min_ts: float = None):
        """等云台停稳后从读帧线程取一帧『新鲜』画面。

        参数:
            settle:  先等待的稳定时间 (秒)
            timeout: 最长等待时间 (秒)
            min_ts:  只接受到达时间戳 >= min_ts 的帧。云台旋转后必须用它,
                     否则 RTSP 缓冲里的"旋转前"旧帧会被当成复测结果 →
                     同一偏差被重复补偿, 角度越转越大。
        """
        time.sleep(settle)
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.frame_lock:
                f = self._latest_frame
                ts = self._latest_frame_ts
            if f is not None and (min_ts is None or ts >= min_ts):
                return f.copy()
            time.sleep(0.05)
        return None

    # 注: 原 _align_gimbal_center_loop 已删除。
    # 该函数采用 "像素偏移 / FOV → 角度" 的迭代闭环 (最多 3 轮), 在 RTSP 缓冲
    # 滞后场景下会反复补偿同一偏差 → 云台过冲/回弹, 且第一窗口与 aligned.png
    # 第三窗口明显不一致。严格按 QR_3D 方案改为单次旋转: 直接使用 PnP 给出的
    # 相机 pitch/roll 作为云台目标 (见 _start_review_mode 内调用处)。

    def _start_review_mode(self, frame):
        """
        启动 review 模式，在视频流上显示 YOLO 检测框
        
        参数:
            frame: 捕获的帧
        """
        print("\n" + "=" * 60)
        print("启动 Review 模式 - 请在视频窗口查看 YOLO 检测框")
        print("=" * 60)
        
        # 使用 YOLOv8 检测二维码 (检测器取当前线程的那一份, ACL 同线程)
        try:
            detections = yolo_detection.detect_qr_yolov8(frame.copy(), self._get_detector())
            # 保存检测结果，供视频循环绘制使用
            self.review_detections = detections
            print(f"  检测到 {len(detections)} 个二维码")
            
            # 保存带YOLO框的图像，用于detection窗口显示
            if len(detections) > 0:
                # 在帧上绘制YOLO框
                frame_with_yolo = frame.copy()
                for det in detections:
                    polygon_xy = det['polygon_xy'].astype(np.float32)
                    conf = det['confidence']
                    
                    # 绘制多边形边界（红色）
                    pts = polygon_xy.reshape((-1, 1, 2))
                    cv2.polylines(frame_with_yolo, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                    
                    # 绘制角点（红色）
                    for pt in polygon_xy:
                        cv2.circle(frame_with_yolo, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                    
                    # 显示置信度（红色）
                    cv2.putText(frame_with_yolo, f"{conf:.2f}", 
                               (int(polygon_xy[0][0]), int(polygon_xy[0][1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 保存带YOLO框的图像
                self.review_frame_with_yolo = frame_with_yolo
                print(f"  ✓ 已保存带YOLO框的图像")

                # ── 生成网页静态图: 检测结果 + 对齐预览 (供 /detect.png /aligned.png 轮询) ──
                try:
                    global _http_detect_jpg, _http_detect_ver, _http_detect_msg
                    global _http_aligned_jpg, _http_aligned_ver, _http_aligned_msg

                    Hh, Ww = frame_with_yolo.shape[:2]
                    pd = getattr(self, 'latest_pose_data', None)

                    # (1) 检测结果图 = YOLO 框 + 位姿坐标轴
                    detect_img = frame_with_yolo.copy()
                    if pd is not None and 'rvec' in pd and 'tvec' in pd:
                        try:
                            # 同上: 必须用 PnP 时的 camera_matrix, 否则投影会偏移
                            if 'camera_matrix' in pd and pd['camera_matrix'] is not None:
                                cam = np.asarray(pd['camera_matrix'], dtype=np.float64)
                            else:
                                cam = pose_estimation.create_camera_matrix((Ww, Hh), fov_x=self.fov_x)
                            dist = np.zeros((5, 1), dtype=np.float32)
                            detect_img = pose_estimation.draw_axes(
                                detect_img, cam, dist,
                                pd['rvec'].reshape(3, 1),
                                pd['tvec'].reshape(3, 1),
                                length=0.05)
                        except Exception:
                            pass
                    cv2.putText(detect_img, "Detection (after [c])",
                                (10, Hh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                    ok, buf = cv2.imencode('.jpg', detect_img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        ts = time.strftime('%H:%M:%S')
                        if pd is not None:
                            dmsg = (f"捕获 {ts}  Yaw={pd.get('yaw', 0):.1f}° "
                                    f"Pitch={pd.get('pitch', 0):.1f}° "
                                    f"Dist={pd.get('distance', 0):.2f}m")
                        else:
                            dmsg = f"捕获 {ts}"
                        with _http_detect_lock:
                            _http_detect_jpg = buf.tobytes()
                            _http_detect_ver += 1
                            _http_detect_msg = dmsg

                    # (2) 对齐结果 = 真实旋转云台 → 等停稳 + RTSP 排空 → 重拍一帧
                    #     严格按 QR_3D 单次旋转方案: 不迭代, 不二次补偿,
                    #     否则 RTSP 缓冲滞后会反复补偿同一偏差 → 云台过冲/回弹。
                    #     角度算法: 相机 pitch → 云台 yaw; 相机 roll → 云台 pitch
                    #     (与 QR_3D/gimbal_pose_estimation_ascend.py 一致)
                    with _http_aligned_lock:
                        _http_aligned_ver += 1
                        _http_aligned_msg = "云台旋转对齐中, 请稍候 ..."

                    aligned, amsg = frame, "未旋转 (无位姿数据)"
                    # [关键] 用『首帧原始位姿』做旋转, 而非 10 帧 EMA 平均
                    #   - QR_3D auto_aim 用单帧位姿增量对齐
                    #   - 10 帧 EMA 平均会"吃掉"小幅偏差, 旋转量过小 → 二维码移不到位
                    #   - 与 QR_3D 源代码逻辑严格保持一致
                    pd_rotate = getattr(self, '_first_pose_data', None) or pd
                    if pd_rotate is not None and self.gimbal is not None:
                        try:
                            # ── QR_3D 增量式对齐 (与 pose_pipeline.compute_gimbal_target 一致) ──
                            #   不直接用 PnP 的 pitch/roll 当绝对目标 (避免 roll-180 又叠加 gimbal_pitch
                            #   → pitch 方向过头 170°+)。改用 image_points 算 QR 像素中心, 转 FOV→角度。
                            Hh, Ww = frame.shape[:2]
                            img_pts = pd_rotate.get('image_points')
                            if img_pts is None or len(img_pts) < 4:
                                raise RuntimeError("无 image_points, 无法算像素偏移")
                            qr_cx = float(np.mean(img_pts[:, 0]))
                            qr_cy = float(np.mean(img_pts[:, 1]))
                            dx = qr_cx - Ww / 2.0
                            dy = qr_cy - Hh / 2.0

                            # FOV: 优先用相机内参反算 (PnP 时已记录 camera_matrix)
                            #   避免云台 query_focal_length() 超时带来的不准确值
                            cam = pd_rotate.get('camera_matrix')
                            if cam is not None:
                                fx = float(cam[0, 0])
                                fov_x_actual = 2.0 * math.degrees(math.atan(Ww / (2.0 * fx)))
                            else:
                                fov_x_actual = self.fov_x

                            # compute_gimbal_target 内部: dyaw, dpitch ← 像素偏移 → 角度,
                            #                          target = current + offset (增量式)
                            dyaw, dpitch, yaw_target, pitch_target = compute_gimbal_target(
                                dx, dy, fov_x_actual, Ww, Hh,
                                gimbal=self.gimbal,
                                angle_dead_zone_deg=0.1,
                            )

                            print(f"\n  [对齐] QR_3D 增量式 (首帧位姿) → "
                                  f"Yaw={yaw_target:.2f}° Pitch={pitch_target:.2f}°  "
                                  f"(ΔYaw={dyaw:+.2f}° ΔPitch={dpitch:+.2f}°  "
                                  f"FOV={fov_x_actual:.1f}°)")
                            # 单次旋转 (走 _gimbal_rotate 标记旋转时间戳, 第一窗口将过滤旧帧)
                            self._gimbal_rotate(yaw_target, pitch_target, 0.0)

                            # ── 等云台停稳 (0.5s) + RTSP 缓冲排空, 抓一帧新图 ──
                            rtsp_latency = 0.6
                            min_ts = time.time() + rtsp_latency
                            shot = self._grab_fresh_frame(settle=0.5, min_ts=min_ts, timeout=2.5)
                            if shot is None:
                                print("  ⚠ 单帧快照超时, 用原图兜底")
                                shot = frame
                            aligned = shot
                            amsg = (f"云台已对齐 (增量, 首帧位姿)  ΔYaw={dyaw:+.2f}° "
                                    f"ΔPitch={dpitch:+.2f}°  FOV={fov_x_actual:.1f}°")
                        except Exception as e:
                            print(f"  ⚠ 对齐失败, 兜底用原图: {e}")
                            amsg = f"对齐异常: {e}"
                            aligned = frame
                    if pd is not None:
                        amsg = (f"{amsg}  Yaw={pd.get('yaw', 0):.1f}° "
                                f"Pitch={pd.get('pitch', 0):.1f}°")

                    ok, buf = cv2.imencode('.jpg', aligned,
                                           [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        with _http_aligned_lock:
                            _http_aligned_jpg = buf.tobytes()
                            _http_aligned_ver += 1
                            _http_aligned_msg = amsg
                except Exception as _e:
                    print(f"  ⚠ 生成网页静态图失败: {_e}")
        except Exception as e:
            print(f"  ⚠ YOLO 检测失败: {e}")
            self.review_detections = []
        
        # 只有当检测到YOLO框时，才进入review模式
        if hasattr(self, 'review_detections') and len(self.review_detections) > 0:
            self.review_mode = True
            print("\n  提示: 在视频窗口按 'y' 重新截图，按 'n' 继续")
        else:
            # 没有检测到YOLO框，不进入review模式
            self.review_mode = False
            print("\n  ⚠ 未检测到YOLO框，不进入review模式")
            if hasattr(self, 'review_detections'):
                delattr(self, 'review_detections')
        
        print("=" * 60)
    
    def stop_video_stream(self):
        """停止视频流"""
        if not self.video_running:
            return
        
        print("\n停止视频流...")
        self.video_running = False

        # 等待线程结束
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self.video_thread is not None:
            self.video_thread.join(timeout=2.0)
            self.video_thread = None
        
        # 释放VideoCapture
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        print("  ✓ 视频流已停止")
    
    def visualize_3d_scene(self, pose_records, headless=False):
        """
        将等间隔的位姿数据、二维码及世界坐标系画在三维场景中

        参数:
            pose_records: list of pose_data dicts，每条记录包含:
                - object_points: 二维码3D角点
                - qr_size: 二维码物理尺寸（米）
                - camera_position: 相机在世界坐标系中的位置 [x, y, z]
                - rmat: 旋转矩阵 (3x3)
                - distance: 相机到原点距离（米）
                - yaw, pitch, roll: 欧拉角（度）
            headless: True 时用 Agg 后端离屏渲染, 返回 PNG bytes
                      (板端无 GUI / 网页显示用); False 时弹窗显示
        """
        try:
            import matplotlib
            if headless:
                matplotlib.use('Agg')   # 板端无 GUI, 必须离屏渲染
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("错误: 需要matplotlib库进行3D可视化")
            print("请安装: pip install matplotlib")
            return None
        
        if not pose_records:
            print("没有位姿数据可供可视化")
            return
        
        print(f"\n正在生成3D场景可视化（共 {len(pose_records)} 个位姿）...")
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=45)
        
        # 从第一条记录获取二维码信息（所有记录应该一致）
        first_pose = pose_records[0]
        object_points_raw = np.array(first_pose['object_points'])
        qr_size = first_pose['qr_size']
        
        # 放大二维码平面使其在3D场景中更明显（保持世界坐标系比例不变）
        viz_scale = 5.0
        object_points = object_points_raw * viz_scale
        qr_center = np.mean(object_points, axis=0)
        
        # ======= 1. 绘制二维码平面（放大版） =======
        # 二维码四边
        vertices = object_points[[0, 1, 2, 3, 0]]
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                'k-', linewidth=2.5, label='QR Code Border')
        
        # 半透明填充二维码平面
        poly = Poly3DCollection([object_points], alpha=0.12, color='gray')
        ax.add_collection3d(poly)
        
        # 二维码中心标签
        ax.text(qr_center[0], qr_center[1], qr_center[2],
                'QR Code', fontsize=12, ha='center', color='black', weight='bold')
        
        # 标记四个角点（1=左上, 2=右上, 3=右下, 4=左下）
        corner_labels = ['1 (TL)', '2 (TR)', '3 (BR)', '4 (BL)']
        for i, pt in enumerate(object_points):
            ax.scatter(pt[0], pt[1], pt[2], c='black', s=50, marker='s')
            ax.text(pt[0], pt[1], pt[2] + object_points_raw.max() * 0.3, corner_labels[i],
                    fontsize=9, ha='center', color='black')
        
        # ======= 2. 绘制世界坐标系轴（二维码中心） =======
        world_axis_length = qr_size * viz_scale * 1.5
        ax.quiver(0, 0, 0, world_axis_length, 0, 0,
                  color='r', arrow_length_ratio=0.15, linewidth=1.5, label='World X')
        ax.quiver(0, 0, 0, 0, world_axis_length, 0,
                  color='g', arrow_length_ratio=0.15, linewidth=1.5, label='World Y')
        ax.quiver(0, 0, 0, 0, 0, -world_axis_length,
                  color='b', arrow_length_ratio=0.15, linewidth=1.5, label='World Z')
        
        # ======= 3. 绘制所有相机位置（渐变色轨迹） =======
        colors = plt.cm.jet(np.linspace(0.1, 0.9, len(pose_records)))
        all_positions = []
        
        for i, pose in enumerate(pose_records):
            cam_pos = np.array(pose['camera_position'])
            all_positions.append(cam_pos)
            rmat = np.array(pose['rmat'])
            color = colors[i]
            
            # 相机位置点（渐变颜色）
            ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                       c=[color], s=40, marker='o', alpha=0.8, zorder=5)
            
            # 标注位姿顺序编号（1, 2, 3...）
            avg_cam_pos = np.mean(np.abs(cam_pos))
            label_offset = max(object_points_raw.max(), avg_cam_pos) * 0.04
            ax.text(cam_pos[0] + label_offset, cam_pos[1] + label_offset, 
                    cam_pos[2] + label_offset,
                    str(i + 1), fontsize=7, ha='left', va='bottom',
                    color='black', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # 首尾两个位置画相机坐标系轴
            if i == 0 or i == len(pose_records) - 1:
                axis_len = qr_size * 0.3
                cam_x = rmat.T @ np.array([[1], [0], [0]])
                cam_y = rmat.T @ np.array([[0], [1], [0]])
                cam_z = rmat.T @ np.array([[0], [0], [1]])
                
                # 相机坐标系（X红=右, Y绿=下, Z蓝=前）
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_x[0, 0], cam_x[1, 0], cam_x[2, 0],
                          length=axis_len, color='r', arrow_length_ratio=0.25, alpha=0.6)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_y[0, 0], cam_y[1, 0], cam_y[2, 0],
                          length=axis_len, color='g', arrow_length_ratio=0.25, alpha=0.6)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                          cam_z[0, 0], cam_z[1, 0], cam_z[2, 0],
                          length=axis_len, color='b', arrow_length_ratio=0.25, alpha=0.6)
                
                label_text = 'Start' if i == 0 else 'End'
                marker = '^' if i == 0 else 's'
                ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                           c=[color], s=100, marker=marker, edgecolors='black', linewidth=1,
                           label=f'Camera ({label_text})', zorder=10)
                ax.text(cam_pos[0] + 0.01, cam_pos[1] + 0.01, cam_pos[2] + 0.01,
                        label_text, fontsize=9, ha='left', color=color)
        
        all_positions = np.array(all_positions)
        
        # ======= 4. 绘制相机移动轨迹连线 =======
        if len(all_positions) > 1:
            ax.plot(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                    'y--', linewidth=1.5, alpha=0.7, label='Camera Path')
        
        # ======= 5. 绘制视线（首帧和末帧到二维码中心的连线） =======
        ax.plot([all_positions[0, 0], qr_center[0]],
                [all_positions[0, 1], qr_center[1]],
                [all_positions[0, 2], qr_center[2]],
                'c:', alpha=0.4, linewidth=1, label='Start LOS')
        ax.plot([all_positions[-1, 0], qr_center[0]],
                [all_positions[-1, 1], qr_center[1]],
                [all_positions[-1, 2], qr_center[2]],
                'm:', alpha=0.4, linewidth=1, label='End LOS')
        
        # ======= 6. 设置图形属性 =======
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'QR Code & Camera 3D Scene ({len(pose_records)} poses)')
        
        # 设置坐标轴范围
        max_range = max(qr_size * 2, np.max(np.abs(all_positions))) * 1.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        
        # 信息文本
        first_dist = pose_records[0]['distance']
        last_dist = pose_records[-1]['distance']
        avg_dist = np.mean([p['distance'] for p in pose_records])
        
        info_text = (
            f"Total Poses: {len(pose_records)}\n"
            f"First Distance: {first_dist:.3f} m\n"
            f"Last Distance:  {last_dist:.3f} m\n"
            f"Avg Distance:   {avg_dist:.3f} m\n"
            f"QR Code Size:   {qr_size} m\n"
            f"Interval:       auto_aim sampling"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))
        
        plt.tight_layout()

        if headless:
            # 离屏渲染 → PNG bytes (供网页 /3d 显示, 不需要 GUI)
            # 注意: 不要用 bbox_inches='tight' — Agg + 3D 时它按 2D 算 bbox
            #       会裁掉 legend / 轴标签 / 标题的边缘部分
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, pad_inches=0.3,
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            png = buf.getvalue()
            buf.close()
            return png

        # 非 headless (PC 端弹窗) — 但如果之前 _start_3d_render_async 已切到 Agg
        # 后端, plt.show() 不响应且不报错。需要检测后端并 fallback。
        try:
            import matplotlib as _mpl
            backend = _mpl.get_backend().lower()
        except Exception:
            backend = ""
        if "agg" in backend:
            # 当前后端是 Agg (无 GUI): plt.show() 不会弹窗
            # 自动 fallback: 把 PNG 写到当前目录供用户查看
            out_path = os.path.join(os.getcwd(), f"gimbal_3d_scene_{time.strftime('%H%M%S')}.png")
            try:
                fig.savefig(out_path, dpi=100, pad_inches=0.3,
                             facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"[3D] 当前后端为 Agg (无 GUI), 已保存到 {out_path}")
                return None
            except Exception as _e:
                print(f"[3D] ⚠ 保存失败: {_e}")
                plt.close(fig)
                return None

        print("3D场景已生成，关闭窗口后程序继续...")
        plt.show()
        return None
    
    def auto_aim(self, interval=1.0, max_retries=3):
        """
        自动瞄准循环：等时间间隔识别二维码，估计位姿后自动旋转云台到对应位置
        
        参数:
            interval: 识别间隔时间（秒），默认 1.0 秒
            max_retries: 连续失败最大重试次数，超过则自动退出
        
        控制逻辑:
            1. 抓取可见光图像
            2. 二维码检测 + 位姿估计
            3. 计算云台目标角度:
               yam_camera_input  = pose_data['pitch']        (Yaw 目标)
               pitch_camera_input = pose_data['roll'] - 180.0  (Pitch 目标)
            4. 将目标角度直接发送给云台（不叠加当前角度）
            5. 等待 interval 秒后重复
        
        退出方式:
            按 Ctrl+C 停止循环
        """
        print("\n" + "=" * 60)
        print("自动瞄准模式")
        print("=" * 60)
        print(f"  识别间隔: {interval} 秒")
        print(f"  最大连续失败次数: {max_retries}")
        print(f"  提示: 按 Ctrl+C 停止")
        print("=" * 60)
        
        retry_count = 0
        pose_records = []  # 收集每次成功的位姿数据，用于退出时3D可视化
        
        try:
            while True:
                loop_start = time.time()
                
                # ===== 步骤1: 抓取图像 =====
                print(f"\n[{time.strftime('%H:%M:%S')}] 抓取图像...")
                success, image_path = self.capture_visible_image(use_rtsp=True)
                
                if not success:
                    retry_count += 1
                    print(f"  ✗ 图像抓取失败 ({retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        print(f"  ✗ 连续失败 {max_retries} 次，自动退出")
                        break
                    time.sleep(interval)
                    continue
                
                # ===== 步骤2: 位姿估计 =====
                print(f"  估计位姿...")
                success, pose_data, result_image = self.estimate_pose_from_image(
                    image_path, debug=False, debug_dir=None
                )
                
                if not success or pose_data is None:
                    retry_count += 1
                    print(f"  ✗ 位姿估计失败 ({retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        print(f"  ✗ 连续失败 {max_retries} 次，自动退出")
                        break
                    time.sleep(interval)
                    continue
                
                # 成功，重置失败计数，并记录位姿数据
                retry_count = 0
                pose_records.append(pose_data)  # 收集等间隔位姿数据，用于3D可视化

                #----------------------------------------

                cam_pos = pose_data['camera_position']
                print("\n")
                print("╔" + "═" * 56 + "╗")
                print("║" + " " * 16 + "📷  相 机 坐 标" + " " * 25 + "║")
                print("╠" + "═" * 56 + "╣")
                print(f"║  X (水平方向):  {cam_pos[0]:>10.4f}  m" + " " * 24 + "║")
                print(f"║  Y (垂直方向):  {cam_pos[1]:>10.4f}  m" + " " * 24 + "║")
                print(f"║  Z (深度方向):  {cam_pos[2]:>10.4f}  m" + " " * 24 + "║")
                print("╚" + "═" * 56 + "╝")
                print("")
                
                # ===== 步骤3: 计算云台目标角度 (增量式: 让QR中心对齐图像中心) =====
                # 原始算法逻辑 (robot_dog_bridge_win_v2.py 已验证):
                #   1) 用 pose_data['image_points'] 算出 QR 中心像素坐标
                #   2) 与图像中心相减, 得到 (dx, dy) 像素偏移
                #   3) 由相机内参 fx 反算 FOV, 再换算像素→角度 (dyaw, dpitch)
                #   4) target = current + offset, 发送到云台
                # 关键修复: 不再依赖云台 query_focal_length() (会超时),
                #          直接用步骤2保存的 camera_matrix 算 FOV。
                img = cv2.imread(image_path)
                if img is None:
                    print(f"  ✗ 无法读取图像以获取尺寸: {image_path}")
                    time.sleep(interval)
                    continue
                img_h, img_w = img.shape[:2]
                img_center_x = img_w / 2.0
                img_center_y = img_h / 2.0

                # QR 中心 = 四个角点的几何中心
                img_pts = pose_data['image_points']
                qr_cx = float(np.mean(img_pts[:, 0]))
                qr_cy = float(np.mean(img_pts[:, 1]))
                dx = qr_cx - img_center_x
                dy = qr_cy - img_center_y

                # 从相机内参矩阵反算 FOV (避免云台查询超时带来的不准确焦距)
                #   fx = img_w / (2 * tan(fov_x/2))  ⇒  fov_x = 2 * atan(img_w / (2*fx))
                if hasattr(self, 'camera_matrix') and self.camera_matrix is not None:
                    fx = float(self.camera_matrix[0, 0])
                    fy = float(self.camera_matrix[1, 1])
                    fov_x_from_intrinsic = 2.0 * math.degrees(math.atan(img_w / (2.0 * fx)))
                    fov_y_from_intrinsic = 2.0 * math.degrees(math.atan(img_h / (2.0 * fy)))
                    print(f"  相机内参: fx={fx:.1f}px fy={fy:.1f}px → FOV_x={fov_x_from_intrinsic:.2f}° FOV_y={fov_y_from_intrinsic:.2f}°")
                    fov_x_actual = fov_x_from_intrinsic
                else:
                    # 兜底: 用默认 60°
                    fov_x_actual = 60.0
                    print(f"  ⚠ 未找到 camera_matrix, 使用默认 fov_x=60°")

                # 计算目标角度 = 当前云台角度 + 像素偏移对应角度
                dyaw, dpitch, yaw_target, pitch_target = compute_gimbal_target(
                    dx, dy, fov_x_actual, img_w, img_h,
                    gimbal=self.gimbal,
                    angle_dead_zone_deg=0.1,
                )

                # 把目标角度限制到云台物理可达范围 (yaw ±180°, pitch ±85°)
                yaw_target = max(-180.0, min(180.0, yaw_target))
                pitch_target = max(-85.0, min(85.0, pitch_target))

                # ===== 步骤4: 发送云台控制命令 =====
                print(f"  像素偏移: ΔX={int(dx):+d}px ΔY={int(dy):+d}px")
                print(f"  角度偏移: ΔYaw={dyaw:+.2f}° ΔPitch={dpitch:+.2f}°")
                print(f"  目标角度: Yaw={yaw_target:.2f}°, Pitch={pitch_target:.2f}°")
                try:
                    self._gimbal_rotate(yaw_target, pitch_target, 0.0)
                    print(f"  ✓ 云台控制命令已发送")
                except Exception as e:
                    print(f"  ✗ 云台控制失败: {e}")
                
                # ===== 步骤5: 等待到下一个周期 =====
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n自动瞄准已停止（用户中断）")
        
        finally:
            print("=" * 60)
            
            # ===== 退出时绘制3D场景可视化 =====
            if pose_records:
                print(f"\n共收集到 {len(pose_records)} 个等间隔位姿数据，正在绘制3D场景...")
                self.visualize_3d_scene(pose_records)
            else:
                print("\n无有效位姿数据，跳过3D可视化")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        # 停止视频流
        self.stop_video_stream()
        
        # 停止云台控制器
        if hasattr(self, 'gimbal'):
            self.gimbal.stop()
            print("  ✓ 云台控制器已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='云台位姿估计')
    parser.add_argument('--gimbal_ip', type=str, default='192.168.1.99',
                       help='云台IP地址 (默认: 192.168.1.99)')
    parser.add_argument('--gimbal_port', type=int, default=6001,
                       help='云台控制端口 (默认: 6001)')
    parser.add_argument('--qr_size', type=float, default=0.10,
                       help='二维码物理尺寸（米） (默认: 0.1)')
    parser.add_argument('--fov_x', type=float, default=60.0,
                       help='相机水平视场角（度） (默认: 60.0)')
    parser.add_argument('--model', type=str, default='./qr_yolov8.om',
                       help='OM 模型路径 (默认: ./qr_yolov8.om)')
    parser.add_argument('--device-id', type=int, default=0,
                       help='Ascend NPU 设备 ID (默认: 0)')
    parser.add_argument('--detector', type=str, default='opencv',
                       choices=['opencv', 'pyzbar'],
                       help='二维码检测器类型 (默认: opencv)')
    parser.add_argument('--square_size', type=int, default=800,
                       help='矫正后正方形图像边长 (默认: 800)')
    parser.add_argument('--conf_threshold', type=float, default=0.2,
                       help='YOLOv8置信度阈值 (默认: 0.2)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='YOLOv8 NMS阈值 (默认: 0.5)')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'interactive', 'video'],
                       help='运行模式: single=单次估计, interactive=交互模式, video=实时视频流 (默认: single)')
    parser.add_argument('--image', type=str, default=None,
                       help='直接使用指定图像进行位姿估计（跳过抓取步骤）')
    parser.add_argument('--video', action='store_true',
                       help='启动时自动打开实时视频流')
    parser.add_argument('--display_pose', action='store_true',
                       help='在视频上显示位姿信息')
    
    args = parser.parse_args()
    
    # 创建位姿估计器
    try:
        estimator = GimbalPoseEstimator(
            gimbal_ip=args.gimbal_ip,
            gimbal_port=args.gimbal_port,
            qr_size=args.qr_size,
            fov_x=args.fov_x,
            model_path=args.model,
            device_id=args.device_id,
            detector_type=args.detector,
            square_size=args.square_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        return
    
    try:
        if args.image is not None:
            # 使用指定图像进行位姿估计
            print(f"\n使用指定图像: {args.image}")
            success, pose_data, result_image = estimator.estimate_pose_from_image(args.image)
            
            if success:
                estimator.print_pose_info(pose_data)
                estimator.save_result(result_image, pose_data)
                print("\n✓ 位姿估计完成")
            else:
                print("\n✗ 位姿估计失败")
        
        elif args.mode == 'single':
            # 单次估计模式
            success, pose_data = estimator.run_single_estimation(save_result=True)
            if not success:
                print("\n✗ 位姿估计失败")
        
        elif args.mode == 'interactive':
            # 交互模式
            estimator.run_interactive_mode()
        
        elif args.mode == 'video':
            # 实时视频流模式
            print("\n启动实时视频流模式...")
            estimator.start_video_stream(display_pose=args.display_pose)
            
            # 等待视频流结束
            try:
                while estimator.video_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n用户中断")
            finally:
                estimator.stop_video_stream()
    
    finally:
        # 清理资源
        estimator.cleanup()


if __name__ == "__main__":
    main()
