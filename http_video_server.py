#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP MJPEG 视频流服务器
================================================================
用于 Ascend 板端 (无 GUI) 将视频帧推送到 PC 浏览器显示。

功能:
    - MJPEG 流:  http://<ip>:8080/video       (实时视频 + 叠加标注)
    - ROI 图:    http://<ip>:8080/roi          (最新识别结果)
    - 控制 API:  http://<ip>:8080/cmd?key=<c/r/t/a/d/f/+/-/方向键>
    - 状态页:    http://<ip>:8080/             (带控件的 HTML 页面)

用法:
    from http_video_server import HttpVideoServer

    server = HttpVideoServer(port=8080)
    server.start()

    # 在视频循环中更新帧
    server.update_frame(display_frame_bgr)
    server.update_roi(roi_frame_bgr)

    # 读取 PC 端发送的按键
    key = server.get_pending_key()
"""

import threading
import time
import queue
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from io import BytesIO
from typing import Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════
# MJPEG 帧缓冲 (线程安全)
# ═══════════════════════════════════════════════════════════
class _FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._video_jpg = b""
        self._roi_jpg = b""
        self._snap_jpg = b""
        self._video_cond = threading.Condition(self._lock)
        self._roi_cond = threading.Condition(self._lock)

    def update_video(self, frame_bgr: np.ndarray, quality: int = 70):
        _, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self._video_cond:
            self._video_jpg = jpg.tobytes()
            self._video_cond.notify_all()

    def update_roi(self, frame_bgr: np.ndarray, quality: int = 85):
        if frame_bgr is None:
            return
        _, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self._roi_cond:
            self._roi_jpg = jpg.tobytes()
            self._roi_cond.notify_all()

    def snap(self, frame_bgr: np.ndarray, quality: int = 90):
        _, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self._lock:
            self._snap_jpg = jpg.tobytes()

    def get_video(self) -> bytes:
        with self._lock:
            return self._video_jpg

    def wait_video(self, timeout: float = 0.5) -> bytes:
        """等待新视频帧 (条件变量，零 CPU 空转)"""
        with self._video_cond:
            self._video_cond.wait(timeout)
            return self._video_jpg

    def wait_roi(self, timeout: float = 0.5) -> bytes:
        with self._roi_cond:
            self._roi_cond.wait(timeout)
            return self._roi_jpg

    def get_roi(self) -> bytes:
        with self._lock:
            return self._roi_jpg

    def get_snap(self) -> bytes:
        with self._lock:
            return self._snap_jpg


# ═══════════════════════════════════════════════════════════
# HTTP 请求处理器
# ═══════════════════════════════════════════════════════════
_HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QR Pose Control</title>
<style>
  body{font-family:Arial;background:#111;color:#eee;margin:0;padding:10px;}
  h2{text-align:center;color:#0f0;}
  .row{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;}
  img{border:2px solid #333;height:auto;}
  .vid{max-width:55%}.roi{max-width:40%}.snap{max-width:40%}
  .ctrl{margin:12px auto;text-align:center;}
  button{padding:8px 14px;margin:3px;font-size:14px;cursor:pointer;border:none;border-radius:4px;}
  .btn-c{background:#0a0;color:#fff}.btn-r{background:#08c;color:#fff}
  .btn-dir{background:#555;color:#fff}.btn-zoom{background:#c80;color:#fff}
  .btn-f{background:#606;color:#fff}.btn-q{background:#c00;color:#fff}
  .btn-step{background:#444;color:#fff}.btn-snap{background:#088;color:#fff}
  .status{margin-top:8px;font-size:13px;color:#aaa;text-align:center;}
</style></head><body>
<h2>QR Pose Control Panel</h2>
<div class="row">
  <img class="vid" id="video" src="/video" alt="Video">
  <img class="roi" id="roi" src="/roi" alt="ROI">
</div>
<div class="row">
  <img class="snap" id="snap" src="" alt="Snapshot" style="display:none">
</div>
<div class="ctrl">
  <button class="btn-c" onclick="send('c')">[c] Detect QR</button>
  <button class="btn-r" onclick="send('r')">[r] Rotate</button>
  <button class="btn-snap" onclick="snap()">📷 Snap</button>
  <button class="btn-snap" onclick="window.open('/3d','_blank')">🔮 3D</button>
  <br>
  <button class="btn-zoom" onclick="send('a')">[A] Zoom+</button>
  <button class="btn-zoom" onclick="send('d')">[D] Zoom-</button>
  <button class="btn-f" onclick="send('f')">[f] Focus</button>
  <br>
  <button class="btn-dir" onclick="send('left')">← Yaw-</button>
  <button class="btn-dir" onclick="send('right')">→ Yaw+</button>
  <button class="btn-dir" onclick="send('up')">↑ Pitch+</button>
  <button class="btn-dir" onclick="send('down')">↓ Pitch-</button>
  <button class="btn-step" onclick="send('+')">+ Step</button>
  <button class="btn-step" onclick="send('-')">- Step</button>
  <button class="btn-q" onclick="send('q')">[q] Quit</button>
</div>
<div class="status" id="status"></div>
<script>
  function send(k){fetch('/cmd?key='+k).then(r=>r.text()).then(t=>{document.getElementById('status').innerText=t})}
  setInterval(()=>{document.getElementById('video').src='/video?'+Date.now()},1000)
  setInterval(()=>{document.getElementById('roi').src='/roi?'+Date.now()},1500)
  function snap(){
    var s=document.getElementById('snap');
    s.src='/snap?'+Date.now();s.style.display='block';
  }
  // 电脑键盘映射
  var km={'c':'c','r':'r','a':'a','d':'d','f':'f','t':'t','q':'q','+':'+','=':'+','-':'-','_':'-','ArrowLeft':'left','ArrowRight':'right','ArrowUp':'up','ArrowDown':'down'};
  document.onkeydown=function(e){var k=km[e.key];if(k){e.preventDefault();send(k)}}
</script></body></html>"""


class _VideoHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器 (每个请求在独立线程中处理)"""
    frame_buf: _FrameBuffer = None   # 类变量，由服务器注入
    key_queue: queue.Queue = None    # 按键队列
    html_3d: str = ""                # 3D 可视化 HTML

    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._serve_html()
        elif path == "/video":
            self._serve_mjpeg(self.frame_buf.wait_video)
        elif path == "/roi":
            self._serve_mjpeg(self.frame_buf.wait_roi)
        elif path == "/snap":
            self._serve_single(self.frame_buf.get_snap)
        elif path == "/3d":
            self._serve_3d()
        elif path.startswith("/cmd"):
            self._handle_cmd()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_HTML_PAGE.encode("utf-8"))

    def _serve_mjpeg(self, getter):
        """MJPEG 流: 持续发送帧 (无新帧时保持连接)"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                jpg = getter()
                if jpg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def _serve_single(self, getter):
        """单帧图片"""
        jpg = getter()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)) if jpg else "0")
        self.end_headers()
        if jpg:
            self.wfile.write(jpg)

    def _serve_3d(self):
        """Plotly 3D 可视化 HTML"""
        html = self.html_3d or "<html><body><h3>3D 可视化尚未生成，请先执行检测</h3></body></html>"
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_cmd(self):
        """解析 /cmd?key=xxx 并推入按键队列"""
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        key = qs.get("key", [""])[0]
        if key and self.key_queue is not None:
            self.key_queue.put(key)
            print(f"[HTTP] 按键: {key}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(f"OK: {key}".encode())


# ═══════════════════════════════════════════════════════════
# HTTP 视频服务器
# ═══════════════════════════════════════════════════════════
class HttpVideoServer:
    """HTTP MJPEG 视频服务器。

    用法:
        server = HttpVideoServer(port=8080)
        server.start()
        # 在视频循环中:
        server.update_frame(frame)   # 更新 MJPEG 视频帧
        server.update_roi(roi_img)   # 更新 ROI 结果图
        key = server.get_pending_key()  # 非阻塞读取按键
        server.stop()
    """

    def __init__(self, port: int = 8080, jpg_quality: int = 70):
        self._port = port
        self._jpg_quality = jpg_quality
        self._frame_buf = _FrameBuffer()
        self._key_queue = queue.Queue()
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动 HTTP 服务器 (后台线程)。"""
        _VideoHandler.frame_buf = self._frame_buf
        _VideoHandler.key_queue = self._key_queue

        # 多线程服务器: SO_REUSEADDR + ThreadingMixIn (避免MJPEG长连接阻塞其他请求)
        import socket as _socket
        class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            allow_reuse_address = True
            daemon_threads = True
            def server_bind(self):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                super().server_bind()

        self._httpd = _ThreadedHTTPServer(("0.0.0.0", self._port), _VideoHandler)
        self._running = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        # 打印本机 IP
        ip = self._get_local_ip()
        print(f"\n[HTTP Video] 视频流已启动:")
        print(f"  控制页面: http://{ip}:{self._port}/")
        print(f"  (视频+ROI合并为单画面, 640px, JPEG质量50)")
        print(f"  在 PC 浏览器打开控制页面即可操作")

    def stop(self):
        """停止 HTTP 服务器。"""
        self._running = False
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    def update_frame(self, frame_bgr: np.ndarray):
        """更新 MJPEG 视频帧 (带标注的显示帧)。

        参数:
            frame_bgr: BGR 格式的 numpy 数组 (H, W, 3)
        """
        self._frame_buf.update_video(frame_bgr, quality=self._jpg_quality)

    def update_roi(self, frame_bgr: np.ndarray):
        """更新 ROI 识别结果帧。

        参数:
            frame_bgr: BGR 格式的 numpy 数组
        """
        self._frame_buf.update_roi(frame_bgr, quality=85)

    def snap(self, frame_bgr: np.ndarray):
        """保存高分辨率截图。

        参数:
            frame_bgr: BGR 格式的 numpy 数组 (原始分辨率)
        """
        self._frame_buf.snap(frame_bgr, quality=90)

    def set_3d_html(self, html: str):
        """设置 3D 可视化 HTML。

        参数:
            html: Plotly 生成的完整 HTML 字符串
        """
        _VideoHandler.html_3d = html

    def get_pending_key(self, timeout: float = 0.0) -> Optional[str]:
        """非阻塞获取待处理的按键。

        返回: 按键字符串 ('c', 'r', 'a', 'd', 'f', '+', '-',
               'left', 'right', 'up', 'down', 'q') 或 None

        PC 端通过控制页面的按钮发送 HTTP 请求，
        按键被推入队列，在此处消费。
        """
        try:
            return self._key_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_pending_keys(self):
        """清空所有待处理按键。"""
        while True:
            try:
                self._key_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
