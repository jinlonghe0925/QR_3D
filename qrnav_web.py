# -*- coding: utf-8 -*-
"""qrnav_web.py — V2 HTML 页面 + V1/V2 全部路由 + HTTP 启动

依赖: 全部兄弟模块 (最上层, 无人依赖它除了主入口)
"""
import json
import time
import threading
import logging

import gimbal_pose_estimation_ascend as _gpea

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_pose_lock, _qrnav_pose_box, _qrnav_poll_busy,
    _qrnav_jpg_buf, _qrnav_estimator_holder, _qrnav_pose_ver,
)
from qrnav_geometry import pose_to_command
from qrnav_dogctl import (
    _ensure_dog_ready, _get_dog_step_state,
)
from qrnav_phases import (
    _phase_get, first_capture_once, second_capture_once,
    exec_phase1_lateral, exec_phase2_gimbal_zoom,
    mark_dog_ready, reset_phase,
    _is_af_running, _set_af_status, _auto_full_worker,
    _qrnav_auto_full_status,
    _do_capture_once,
)
from qrnav_initial import InitialLocalization
from qrnav_steps import (
    lateral_step_only, zoom_step_only, auto_step_nav,
    align_step_nav, exec_plan_step,
)


def _invalidate_navigation_after_initial_turn():
    reset_phase()
    with _qrnav_pose_lock:
        _qrnav_pose_box[0] = None
        _qrnav_pose_ver[0] += 1


_initial_localization = InitialLocalization(
    lambda name: _do_capture_once(name, align_gimbal=False),
    lambda: _qrnav_estimator_holder[0], lambda: _QR_NAV_CFG,
    _invalidate_navigation_after_initial_turn)

# ============================================================
#  HTML — V2 页面 (5 步走, 状态条 + 阶段指示)
# ============================================================
HTML_QR_NAV2_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>QR → 狗 两段式自动导航 (V2)</title>
<style>
  body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#1a1a1f;color:#eee;padding:8px}
  h3{margin:4px 0;color:#0cf;font-size:14px}
  a{color:#0cf}
  .row{display:flex;flex-wrap:wrap;gap:8px}
  .panel{background:#26262e;border-radius:5px;padding:8px;flex:1;min-width:300px}
  .vid{position:relative;background:#000;border:1px solid #444}
  .vid img{display:block;width:480px;height:auto}
  .vid .overlay{position:absolute;top:4px;left:4px;background:rgba(0,0,0,0.6);
                padding:3px 6px;font:11px monospace;color:#0f0;border-radius:3px}
  .stepbar{display:flex;gap:0;margin:8px 0;align-items:center}
  .step{flex:1;background:#333;color:#aaa;padding:10px 6px;text-align:center;
        border-right:1px solid #111;font-size:12px}
  .step:last-child{border-right:0}
  .step.active{background:#0af;color:#fff;font-weight:bold}
  .step.done{background:#0a0;color:#000}
  .step.cur{background:#0cf;color:#000;font-weight:bold}
  .arrow{color:#888;font-size:18px;padding:0 4px}
  .pose{font-family:Consolas,monospace;font-size:12px;line-height:1.6}
  .pose b{color:#0cf;display:inline-block;min-width:90px}
  .ok{color:#0f0}.warn{color:#f80}.err{color:#f44}
  button{padding:7px 12px;margin:2px;border:none;border-radius:3px;cursor:pointer;
         background:#0af;color:#fff;font-size:12px;font-weight:bold}
  button:hover{background:#08d}
  button:disabled{background:#555;cursor:not-allowed;color:#aaa}
  button.danger{background:#c33}
  button.success{background:#0a0;color:#000}
  button.warn{background:#f80;color:#222}
  .btn-block{width:100%;font-size:13px;padding:10px}
  .row2{margin:3px 0}
  .grid{display:grid;grid-template-columns:repeat(3,46px);gap:3px;margin:4px 0}
  .hint{color:#bbb;font-size:11px;margin:2px 0;padding:4px;background:#333;border-radius:3px}
  .summary{background:#00263a;border:1px solid #0ff;padding:6px;margin:4px 0;
           font-family:Consolas,monospace;font-size:12px;line-height:1.5}
  .log{background:#1a1a1a;border:1px solid #444;padding:6px;margin-top:6px;
       font-family:Consolas,monospace;font-size:11px;line-height:1.5;
       max-height:200px;overflow-y:auto;color:#0f0}
  .log .err{color:#f44}
  .log .warn{color:#f80}
  .log .ok{color:#0f0}
  /* ── 机器狗控制盘 (与参照主页面同款) ── */
  .btn-dog{background:#555;color:#fff}
  .btn-dog:hover{background:#777}
  .btn-dogm{background:#0a6;color:#fff;font-size:13px}
  .btn-dogm:hover{background:#0c8}
  .btn-dogm:disabled{background:#444;color:#888}
  .msg{background:#111;border:1px solid #555;padding:5px;margin-top:5px;
       font-family:Consolas,monospace;font-size:11px;line-height:1.4;
       color:#ff0;min-height:32px;border-radius:3px;white-space:pre-wrap}
</style>
</head>
<body>
<h3>🐕📷 QR 位姿 → 机器狗 两段式自动导航 <span style="color:#888">(V2 策略)</span>
   <a href="/qr_nav" style="font-size:11px">← V1 页面</a></h3>

<!-- ============= ✨ ▶ 一键完成 (主入口, 单按钮跑完整 6 段) ============= -->
<div class="panel" style="border:2px solid #0cf;margin-bottom:10px">
  <h3 style="color:#0cf">模块1 · 初始定位 / 朝向对齐</h3>
  <div class="hint">初始云台与狗同向。捕获Pose后，将二维码平面法线投影到相机XZ平面，与视平面法线（相机Z轴）比较，计算绕相机Y轴的偏角；只转狗，不平移、不调整云台。俯仰方向的倾斜不在本步骤调整范围内。</div>
  <div class="row2">
    <button id="b_initial_gimbal" class="btn-block" style="width:100%;background:#7652ad"
      onclick="initialAction('gimbal_init')">初始化云台（0° / 0° / 0°）</button>
  </div>
  <div class="row2">
    <button id="b_initial_capture" class="btn-block" style="width:48%;background:#0a6"
      onclick="initialAction('capture')">📸 捕获初始Pose</button>
    <button id="b_initial_execute" class="btn-block" style="width:48%;background:#08f" disabled
      onclick="initialAction('execute')">↻ 执行朝向对齐</button>
  </div>
  <div id="initial_plan" class="summary">尚未捕获初始Pose</div>
  <div id="initial_msg" class="msg">执行后请重新捕获复测；每份Pose只执行一次。安全前移将在后续加入。</div>
  <div id="initial_result" class="hint"></div>
</div>

<div class="panel" style="background:linear-gradient(90deg,#001a33,#003366);
                          border:2px solid #4af;margin-bottom:8px;
                          padding:8px;border-radius:4px">
  <div class="row2" style="text-align:center">
    <button class="btn-block" id="b_autofull"
            style="background:#4af;color:#000;font-size:18px;font-weight:bold;
                   padding:14px;letter-spacing:2px"
            onclick="autoFullStart()">▶ 一键完成 (10~15 秒)</button>
  </div>
  <!-- ★ 手动分步: 主流程区醒目两列按钮, 不再折叠在 details 里 -->
  <div class="row2" style="text-align:center;margin-top:8px">
    <button class="btn-block" id="b_cap1"
            style="width:48%;background:#0a0;color:#000;font-size:14px;font-weight:bold;
                   padding:12px;display:inline-block"
            onclick="call('/qr_nav2/capture1')">📸 ① 第一次捕获<br><span style="font-size:10px">(算 3D pose, 不动云台)</span></button>
    <button class="btn-block" id="b_exec1"
            style="width:48%;background:#08f;color:#fff;font-size:14px;font-weight:bold;
                   padding:12px;display:inline-block;margin-left:4px"
            onclick="call('/qr_nav2/exec_lateral')">↔️ ① 执行横移<br><span style="font-size:10px">(自动开始运动 → 按相机 X 横移 → 零速停止)</span></button>
  </div>
  <div id="autofull_prog" style="margin-top:8px;display:none">
    <div style="background:#000;border:1px solid #4af;height:22px;border-radius:3px;overflow:hidden">
      <div id="autofull_bar" style="background:linear-gradient(90deg,#4af,#0f8);
                                    height:100%;width:0%;transition:width .4s"></div>
    </div>
    <div id="autofull_msg" style="color:#4af;font-size:13px;margin-top:5px;
                                  font-family:Consolas,monospace;font-weight:bold">就绪</div>
  </div>
  <div style="color:#8cf;font-size:11px;margin-top:6px;line-height:1.5">
    ✨ 自动: 狗初始化 → ① 捕获 → ① 横移 → 狗就位(自动) → ② 捕获 → ② 云台+变倍<br>
    ⚠ 任何阶段失败自动停止, 可手动点「↺ 重置」再重试, 或展开「📋 手动分步」单步调试
  </div>
</div>

<!-- 步骤条: 6 段 (idle → done) -->
<div class="stepbar" id="stepbar">
  <div class="step" data-phase="idle">0 idle</div>
  <div class="arrow">→</div>
  <div class="step" data-phase="first_captured">① 捕获</div>
  <div class="arrow">→</div>
  <div class="step" data-phase="first_executed">① 执行横移</div>
  <div class="arrow">→</div>
  <div class="step" data-phase="dog_ready">狗就位</div>
  <div class="arrow">→</div>
  <div class="step" data-phase="second_captured">② 捕获</div>
  <div class="arrow">→</div>
  <div class="step" data-phase="done">② 执行云台+变倍</div>
</div>

<div class="row">
  <div class="panel">
    <div class="vid">
      <img id="qr_img" src="/qr_nav/stream" alt="QR">
      <div class="overlay" id="overlay">无 QR</div>
    </div>
    <div class="summary" id="live_summary">
      <b>📊 实时位姿</b> <span id="ph_label" style="color:#0ff">— 阶段: idle —</span><br>
      ↔ 相机位置 X <span id="live_lat">--</span> m &nbsp;
      ⤵ 位置 Z <span id="live_fwd">--</span> m &nbsp;
      🔍 占比 <span id="live_fill">--</span> &nbsp;
      yaw <span id="live_yaw">--</span>°<br>
      <span id="live_hint" style="color:#888">--</span>
      <br><span style="color:#8cf">云台位置作为狗的位置；执行横移距离 = |相机位置 X|</span>
    </div>
  </div>

  <div class="panel" style="min-width:380px">
    <details style="background:#3a2600;border:2px solid #f80;padding:8px;border-radius:4px">
      <summary style="color:#f80;cursor:pointer;font-weight:bold;font-size:13px">
        📋 手动分步 (备选 / 出错时单步调试) — 点展开
      </summary>
      <table style="width:100%;margin-top:6px;font-size:12px">
        <tr><td style="color:#0f0">① 第一次捕获</td>
            <td><button class="btn-block" id="b_cap1"
                onclick="call('/qr_nav2/capture1')">📸① 第一次捕获 (算 3D, 不动云台)</button></td></tr>
        <tr><td style="color:#0f0">① 执行横移</td>
            <td><button class="btn-block" id="b_exec1" style="background:#08f"
                onclick="call('/qr_nav2/exec_lateral')">↔️① 自动开始 → 按相机 X 横移 → 停止</button></td></tr>
        <tr><td style="color:#888">✓ 狗就位 (自动)</td>
            <td><button class="btn-block" id="b_dogready" style="background:#0c0;color:#000"
                onclick="call('/qr_nav2/dog_ready')">✓ 手动标狗就位 (一般无须点)</button></td></tr>
        <tr><td style="color:#0f0">② 第二次捕获</td>
            <td><button class="btn-block" id="b_cap2"
                onclick="call('/qr_nav2/capture2')">📸② 第二次捕获 (底层算法同①)</button></td></tr>
        <tr><td style="color:#0f0">② 执行云台+变倍</td>
            <td><button class="btn-block" id="b_exec2" style="background:#f0f;color:#000"
                onclick="call('/qr_nav2/exec_gimbal_zoom')">🔄② 执行云台+变倍 (转回图像中心 + 变倍)</button></td></tr>
        <tr><td style="color:#f44">↺ 重置</td>
            <td><button class="btn-block danger" id="b_reset"
                onclick="call('/qr_nav2/reset')">↺ 重置 → 回到阶段 0</button></td></tr>
      </table>
    </details>

    <div class="row2" style="margin-top:8px">
      <button class="btn-block warn" id="b_init"
              onclick="dog('stand_quiet')" style="width:48%;background:#0a6">🦿 静默起立</button>
      <button class="btn-block danger" id="b_stop"
              onclick="dog('stop')" style="width:48%;background:#a30">⏹ 复位 (停止+趴下)</button>
    </div>
    <div style="color:#888;font-size:10px;text-align:center;margin-top:2px">
      <b>🦿 静默起立</b> = 起立+手动+力控，腿保持静止；<b>⏹ 复位</b> = 停止+趴下复位+断开。<br>
      两个独立按钮, 文字和动作一一对应, 不会反。<br>
      走的是和主页面同一个 <b>/dog/stand</b> · <b>/dog/stop</b> 链路, 不再经服务端自回调。
    </div>
    <div class="row2" style="font-size:11px;color:#aaa">
      状态: <span id="init_state" style="color:#f80">未知</span> &nbsp;
      <span id="step_stand">①</span>·
      <span id="step_force">②</span>·
      <span id="step_motion">③</span>
    </div>

    <div class="summary">
      <b>📦 第一次捕获结果 (first_pose)</b><br>
      <span id="cap1_summary" style="color:#888">未捕获</span>
    </div>
    <div class="summary">
      <b>📦 第二次捕获结果 (second_pose)</b><br>
      <span id="cap2_summary" style="color:#888">未捕获</span>
    </div>
    <div class="log" id="log">
      <div>⏳ 等待操作…</div>
    </div>
    <div class="hint" id="hint">提示: ① 捕获 → ① 横移 → ✓ 狗就位 → ② 捕获 → ② 云台+变倍</div>
  </div>

  <div class="panel" style="min-width:300px">
    <details>
      <summary style="color:#0cf;cursor:pointer;font-weight:bold;padding:4px 0">
        🤖 机器狗驱动 (14 个按钮) — 点展开高级控制
      </summary>
      <!-- ===== 🦿 机器狗驱动: 与参照主页面完全同款 ===== -->
    <div class="row2">
      <b style="color:#0cf">🦿 机器狗驱动 (与主页面同款)</b>
    </div>
    <!-- ① ② ③ 三步使能 (关键: 缺力控/开始运动腿不动) -->
    <div class="row2">
      <button class="btn-block" style="background:#0c0;color:#000"
              onclick="dog('stand')">① 起立 (连接+力控+开始运动)</button>
    </div>
    <div class="row2">
      <button style="background:#08f;width:48%" onclick="dog('force')">② 力控</button>
      <button style="background:#08f;width:48%" onclick="dog('motion')">③ 开始运动</button>
    </div>
    <div style="color:#888;font-size:10px;text-align:center">
      ⚠ 缺「力控」移动指令被丢, 缺「开始运动」狗腿不动
    </div>
    <!-- 步长调节 (与主页面同款: 0.2m / 15°, 上限 0.3m / 30°) -->
    <div class="row2" style="text-align:center;margin-top:6px">
      <button class="btn-dog" onclick="dogstep(-1)">− 步长</button>
      <span id="dog_step" style="color:#0cf;font-size:13px;margin:0 8px">0.2m / 15°</span>
      <button class="btn-dog" onclick="dogstep(1)">+ 步长</button>
    </div>
    <!-- 方向按钮: 带 ?v=<步长> (与主页面一致, 不带 v 后端会用默认步长) -->
    <div class="grid" style="margin:auto;width:fit-content;grid-template-columns:repeat(3,72px)">
      <div></div>
      <button class="btn-dogm" onclick="dog('fwd')">⬆ 前进</button>
      <div></div>
      <button class="btn-dogm" onclick="dog('left')">⬅ 左移</button>
      <button class="btn-dogm" style="background:#c33" onclick="dog('down')">⏹ 趴下</button>
      <button class="btn-dogm" onclick="dog('right')">➡ 右移</button>
      <div></div>
      <button class="btn-dogm" onclick="dog('back')">⬇ 后退</button>
      <div></div>
    </div>
    <div class="row2" style="margin-top:4px;text-align:center">
      <button class="btn-dogm" style="width:48%" onclick="dog('tl')">↺ 左转</button>
      <button class="btn-dogm" style="width:48%" onclick="dog('tr')">↻ 右转</button>
    </div>
    <div class="row2" style="margin-top:4px;text-align:center">
      <button class="danger" style="width:48%" onclick="dog('damp')">💧 阻尼</button>
      <button class="danger" style="width:48%" onclick="dog('stop')">✖ 停止并断开</button>
    </div>
    <div class="msg" id="dog_msg">机器狗未连接: 先「① 起立」→「② 力控」→「③ 开始运动」</div>
    <div class="row2" style="margin-top:4px">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="checkDogConn()">🔍 检查狗连接 (GET /dog_msg)</button>
    </div>
    </details>

    <div class="row2" style="margin-top:8px">
      <b style="color:#0cf">🔍 镜头 (变倍+对焦)</b>
    </div>
    <div class="row2">
      <button style="background:#08f;width:30%" onclick="call('/qr_nav/zoom/out')">🔍-</button>
      <button style="background:#08f;width:33%" onclick="call('/qr_nav/zoom/auto')">🎯自动</button>
      <button style="background:#08f;width:30%" onclick="call('/qr_nav/zoom/in')">🔍+</button>
    </div>
    <div class="row2">
      <button style="background:#08f;width:100%" onclick="call('/qr_nav/zoom/focus')">🔲 自动对焦</button>
    </div>
    <div class="row2" style="margin-top:8px">
      <b style="color:#888;font-size:11px">🔧 中心换算</b>
    </div>
    <div class="row2" style="font-size:11px;color:#aaa">
      横偏死区 <span id="lat_dead">5</span>mm / 中心死区
      <span id="ctr_dead">4</span>px<br>
      云台单次增量上限 = FOV/2
    </div>
    <div class="row2">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="call('/qr_nav2/toggle_center_yaw')">↔ 翻 CENTER_YAW_SIGN</button>
    </div>
    <div class="row2">
      <button style="background:#444;font-size:10px;width:100%"
              onclick="call('/qr_nav2/toggle_center_pitch')">↕ 翻 CENTER_PITCH_SIGN</button>
    </div>
  </div>
</div>

<!-- ============= 🤖 机器人运动 (V1 模式一键) ============= -->
<!-- 把 V1 全部"机器人运动"按钮合并到这里, 一个页面搞定, 不必切回 /qr_nav -->
<details style="margin-top:8px">
  <summary style="background:#3a0010;border:2px solid #f44;padding:8px;cursor:pointer;
                   color:#f44;font-weight:bold">
    ⚙ 高级 / V1 模式 (5 个一键) — 点展开
  </summary>
  <div class="panel" style="background:#3a0010;border:2px solid #f44;margin-top:2px">
  <b style="color:#f44">🤖 机器人运动 (一键, V1 模式)</b>
  <span style="color:#888;font-size:11px;margin-left:6px">
    ← V1 的「自动一步 / 仅横向对齐 / 一键执行 + 方向翻转」, V2 页面内可直接用
  </span>
  <table style="width:100%;margin-top:6px;font-size:12px">
    <tr>
      <td style="color:#0f0;width:130px">🎯 自动一步 (横移+变倍)</td>
      <td>
        <button class="btn-block" style="background:#0a0;color:#000"
                onclick="call('/qr_nav2/auto')">🎯 自动一步 (按最近 pose 决策 + 串行执行)</button>
      </td>
      <td style="color:#888;font-size:10px">单步按钮, 不重新捕获</td>
    </tr>
    <tr>
      <td style="color:#0f0">↔️ 仅横向对齐</td>
      <td>
        <button class="btn-block" style="background:#08f"
                onclick="call('/qr_nav2/align')">↔️ 仅横向对齐 (只横移, 不变倍)</button>
      </td>
      <td style="color:#888;font-size:10px">横移后变倍由用户控</td>
    </tr>
    <tr>
      <td style="color:#0f0">🚀 一键执行</td>
      <td>
        <button class="btn-block" id="b_execplan" style="background:#f0f;color:#000"
                onclick="execPlanLoop()">🚀 一键执行 (循环 横移+变倍, 直至对齐)</button>
      </td>
      <td style="color:#888;font-size:10px">前端 JS 循环调, 50 步硬限</td>
    </tr>
    <tr>
      <td style="color:#f80">↔ 翻转左右方向</td>
      <td>
        <button class="btn-block" style="background:#f80;color:#222"
                onclick="call('/qr_nav2/lateral_sign')">↔ 翻转 LATERAL_SIGN (左右反了点这个)</button>
      </td>
      <td style="color:#888;font-size:10px">单独翻横移方向</td>
    </tr>
    <tr>
      <td style="color:#f80">🔄 反转 yaw</td>
      <td>
        <button class="btn-block" style="background:#f80;color:#222"
                onclick="call('/qr_nav2/yaw_sign')">🔄 反转 YAW_SIGN (云台左右反了用这个)</button>
      </td>
      <td style="color:#888;font-size:10px">会同时翻前后/左右, 慎用</td>
    </tr>
  </table>
  <div style="color:#888;font-size:10px;margin-top:6px">
    ⚠ 上述 5 个按钮在 V2 页面里直接生效 (使用最近一次捕获的 pose),
      不走 V2 的 6 段式状态机. V2 的两段式流程 (① 捕获→横移 → ② 捕获→云台+变倍) 仍在顶部.
  </div>
  </div>
</details>

<script>
// ══════════════════════════════════════════════════════════
//  🦿 机器狗驱动 (与参照主页面 gimbal_pose_estimation_ascend.py 同款)
//    关键: 必须带 ?v=<步长>; 单步模式每次只走固定一步
// ══════════════════════════════════════════════════════════
var DOG_LOCK_M = 0.3, DOG_LOCK_DEG = 30;
var dogStepM = 0.2, dogStepDeg = 15;
var dogBusy = false;
var dogReadyCache = false;      // 由 /qr_nav/dog_ready 轮询刷新

// ── 🦿 一键初始化 / 停止 切换 ──
//    与参照主页面完全同路径: 浏览器直接打 /dog/stand 或 /dog/stop,
//    不再经服务端 /qr_nav/init_all 自回调 (那条链路曾被代理/冷却卡死)
//
//    关键时序 (防止"刚点完初始化立刻点停止"卡死):
//      • 后端 dog.init() 串行下发 心跳→起立→手动→力控→开始运动→gait ≈3.5s
//      • 但 fetch 返回时间 ≈ 后端把指令塞进发送队列的时间, 远早于狗硬件完成
//      • 旧逻辑 dogBusy 在 fetch 返回瞬间就释放 → 用户在过渡期点 stop,
//        dog.close() 撞上未完成的 motion 指令 → 后端阻塞 / 狗卡顿
//    新逻辑: 锁延长到"狗硬件稳定", 走 DOG_LOCK_INIT_MS / DOG_LOCK_STOP_MS
//
//    按钮拆成两个独立按钮 (#b_init 调 stand, #b_stop 调 stop),
//    文字和动作一一对应, 杜绝"反了"歧义
function syncInitBtn(){
  // 仅更新指示灯 (🦿 静默起立 / ⏹ 复位 按钮文字固定, 不再切换)
  // V2 的「静默起立」可重复点 (没副作用, idempotent), 永远高亮
  // 「⏹ 复位」狗未就绪时灰显
  const initBtn = document.getElementById('b_init');
  const stopBtn = document.getElementById('b_stop');
  if (initBtn) initBtn.style.opacity = '1.0';
  if (stopBtn) {
    if (dogReadyCache) stopBtn.style.opacity = '1.0';
    else               stopBtn.style.opacity = '0.5';
  }
}

// ★ 锁时长配置 —— 经验值, 与 robot_interface.Dog 实测对齐
//   init (stand/force/motion): 后端 3.5s + 1s 硬件稳定 = 4.5s
//   stop  (stop/down/damp)  : 后端 ~1s + 2.5s 完全停下 = 3.5s
//   移动 (fwd/left/...)     : 0.6m/s × 0.2m ≈ 0.4s + 0.5s 余量 = 0.9s
//   转向 (tl/tr)           : 慢一些, 1.2s
var DOG_LOCK_INIT_MS  = 4500;
var DOG_LOCK_STOP_MS  = 3500;
var DOG_LOCK_MOVE_MS  = 900;
var DOG_LOCK_TURN_MS  = 1200;
function _lock_ms_for(a){
  if (a=='stand'||a=='force'||a=='motion'||a=='start'||a=='stand_quiet') return DOG_LOCK_INIT_MS;
  if (a=='stop' ||a=='down' ||a=='damp')               return DOG_LOCK_STOP_MS;
  if (a=='tl'   ||a=='tr')                              return DOG_LOCK_TURN_MS;
  return DOG_LOCK_MOVE_MS;
}

function dog(a){
  if (dogBusy) return;                       // 上一步未完成, 忽略本次点击
  // stand/force/motion/stop/down/damp/stand_quiet 不需要步长; tl/tr 用角度; 其余用平移步长
  var v = (a=='tl' || a=='tr') ? dogStepDeg
        : (a=='stand'||a=='force'||a=='motion'||a=='start'||a=='stop'||a=='down'||a=='damp'||a=='stand_quiet') ? 0
        : dogStepM;
  var lockMs    = _lock_ms_for(a);           // ★ 硬件完成才释放
  var isInitAct = (a=='stand'||a=='force'||a=='motion'||a=='start'||a=='stand_quiet');
  var isStopAct = (a=='stop' ||a=='down' ||a=='damp');
  // 模式准备由后端统一处理，前端每次点击只发一个请求。
  var gaitQuery = (a=='motion'||a=='start'||a=='start_motion'||a=='enable')
                ? '&gait=0x21010420' : '';
  dogBusy = true;
  var ms = document.querySelectorAll('.btn-dogm');
  ms.forEach(function(b){ b.disabled = true; });
  var e = document.getElementById('dog_msg');
  if (e) e.innerText = '⏳ 执行中: /dog/' + a + '?v=' + v + gaitQuery + ' ...';
  log('→ /dog/' + a + '?v=' + v + gaitQuery + '  (锁 ' + (lockMs/1000).toFixed(1) + 's)', 'info');
  fetch('/dog/' + a + '?v=' + v + gaitQuery)
    .then(function(r){ return r.text(); })
    .then(function(t){
      log('← /dog/' + a + '  ' + t,
          t.indexOf('✗')>=0 || t.indexOf('ERR')>=0 ? 'err' :
          (t.indexOf('⚠')>=0 ? 'warn' : 'ok'));
      if (t && (t.indexOf('✗') >= 0 || t.indexOf('ERR') >= 0)) {
        if (e) e.innerText = t;
        return { ok:false, t:t };
      }
      if (e) e.innerText = t + '  ⏳ 硬件稳定 ' + (lockMs/1000).toFixed(1) + 's';
      pollDogState();
      return { ok:true, t:t };
    })
    .catch(function(err){
      if (e) e.innerText = '✗ 请求失败: ' + err + '  (立即释放锁)';
      log('✗ /dog/' + a + '  ' + err, 'err');
      // ★ 失败时立即释放, 不让用户干等
      lockMs = 0;
      return { ok:false, t:'请求失败' };
    })
    .then(function(res){
      if (lockMs === 0) {
        // 失败 / 普通移动: 立即释放
        if (res && res.ok && isInitAct) { dogReadyCache = true;  syncInitBtn(); }
        if (res && res.ok && isStopAct) { dogReadyCache = false; syncInitBtn(); }
        if (!res || !res.ok)            { dogReadyCache = false; syncInitBtn(); }
        dogBusy = false;
        ms.forEach(function(b){ b.disabled = false; });
        return;
      }
      // ★ 关键: 延后释放锁, 防止 stop 撞上未完成的 init
      setTimeout(function(){
        if (res && res.ok && isInitAct) { dogReadyCache = true;  syncInitBtn(); }
        if (res && res.ok && isStopAct) { dogReadyCache = false; syncInitBtn(); }
        if (!res || !res.ok)            { dogReadyCache = false; syncInitBtn(); }
        pollDogState();
        dogBusy = false;
        ms.forEach(function(b){ b.disabled = false; });
        if (e) e.innerText += '  ✓ 稳定, 可操作';
        log('✓ /dog/' + a + ' 锁已释放, dogReady=' + dogReadyCache, 'ok');
      }, lockMs);
    });
}
function dogstep(d){
  if (d > 0){
    dogStepM   = Math.min(DOG_LOCK_M,   +(dogStepM + 0.1).toFixed(1));
    dogStepDeg = Math.min(DOG_LOCK_DEG, dogStepDeg + 5);
  } else {
    dogStepM   = Math.max(0.1, +(dogStepM - 0.1).toFixed(1));
    dogStepDeg = Math.max(5,   dogStepDeg - 5);
  }
  var el = document.getElementById('dog_step');
  if (el) el.innerText = dogStepM + 'm / ' + dogStepDeg + '°';
}
function checkDogConn(){
  var e = document.getElementById('dog_msg');
  fetch('/dog_msg').then(function(r){ return r.text(); }).then(function(t){
    if (e) e.innerText = t || '(空)';
    log('← /dog_msg  ' + t, 'ok');
  }).catch(function(err){
    if (e) e.innerText = '✗ /dog_msg 请求失败: ' + err;
  });
}

var initialBusy = false;
async function pollInitial(){
  try{
    const r=await fetch('/qr_nav2/initial/state');
    if(!r.ok)throw new Error('HTTP '+r.status);
    const s=await r.json(), p=s.plan;
    document.getElementById('initial_msg').textContent=s.message;
    document.getElementById('b_initial_gimbal').disabled=initialBusy||['executing','capturing','initializing'].includes(s.stage);
    document.getElementById('b_initial_capture').disabled=initialBusy||s.stage==='executing'||s.stage==='capturing';
    document.getElementById('b_initial_execute').disabled=initialBusy||!s.can_execute;
    const e=document.getElementById('initial_plan');
    e.textContent=p ? '捕获 '+new Date(s.capture_ts*1000).toLocaleTimeString()+
      ' ｜ 绕相机Y轴偏角 '+p.yaw_correction_deg.toFixed(2)+'°'+
      ' ｜ '+(p.needs_turn?'向'+p.direction+'转 '+Math.abs(p.yaw_correction_deg).toFixed(2)+'°':'在 '+p.tolerance_deg+'° 容差内')+
      ' ｜ 相机X/Z平面法线 ('+p.normal_camera[0].toFixed(3)+', '+p.normal_camera[2].toFixed(3)+')'+
      ' ｜ 云台Yaw '+p.gimbal_yaw_deg.toFixed(2)+'°' : '尚无可执行的初始Pose';
  }catch(e){}
}
async function initialAction(action){
  if(initialBusy)return;
  initialBusy=true;
  document.getElementById('b_initial_gimbal').disabled=true;
  document.getElementById('b_initial_capture').disabled=true;
  document.getElementById('b_initial_execute').disabled=true;
  const result=document.getElementById('initial_result');
  result.textContent=action==='gimbal_init'?'正在初始化云台到 0° / 0° / 0°…':(action==='capture'?'正在捕获初始Pose…':'正在执行水平转向…');
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),60000);
  try{
    const r=await fetch('/qr_nav2/initial/'+action,{signal:controller.signal});
    if(!r.ok)throw new Error('HTTP '+r.status);
    const text=await r.text();
    result.textContent=text;
    log('模块1: '+text,text.includes('✗')?'err':'ok');
  }catch(e){result.textContent=e.name==='AbortError'?'请求超时，请检查终端执行状态后重新捕获':'请求失败: '+e.message;}
  finally{clearTimeout(timer);initialBusy=false;document.getElementById('b_initial_capture').disabled=false;await pollInitial();pollPhase();}
}
pollInitial();
setInterval(pollInitial,800);

var callBusy = false;
async function call(path){
  if (callBusy) return;
  callBusy = true;
  var e = document.getElementById('hint');
  e.textContent = '⏳ ' + path + ' ...';
  log('→ ' + path, 'info');
  try {
    const r = await fetch(path);
    const t = await r.text();
    e.textContent = path + ' → ' + t;
    log('← ' + path + '  ' + t, t.indexOf('✗')>=0 || t.indexOf('ERR')>=0 ? 'err' :
                         (t.indexOf('⚠')>=0 || t.indexOf('⏳')>=0 ? 'warn' : 'ok'));
  } catch(err) {
    e.textContent = '✗ ' + path + ' 请求失败: ' + err;
    log('✗ ' + path + '  ' + err, 'err');
  } finally {
    callBusy = false;
    pollPhase();
  }
}
function log(msg, level){
  var div = document.getElementById('log');
  var c = level==='err' ? 'err' : (level==='warn' ? 'warn' : 'ok');
  div.innerHTML = '<div class="' + c + '">[' + new Date().toLocaleTimeString() + '] ' + msg + '</div>' + div.innerHTML;
  // 限长
  while (div.children.length > 30) div.removeChild(div.lastChild);
}

async function pollPhase(){
  try {
    const r = await fetch('/qr_nav2/phase');
    const s = await r.json();
    // 1) 步骤条
    const order = ['idle','first_captured','first_executed','dog_ready','second_captured','done'];
    const steps = document.querySelectorAll('.stepbar .step');
    let curIdx = order.indexOf(s.phase);
    if (curIdx < 0) curIdx = 0;
    steps.forEach((el, i) => {
      el.classList.remove('active', 'done', 'cur');
      if (i < curIdx) el.classList.add('done');
      else if (i === curIdx) el.classList.add('cur');
    });
    document.getElementById('ph_label').textContent = '— 阶段: ' + s.phase + ' —';
    // 2) 按钮启用 / 禁用 (按阶段)
    //    ★ b_exec1 永远可点 (任何 phase 包括 idle), exec_phase1_lateral 内部会
    //      自动先跑 first_capture_once 兜底, 解决「QR 偶尔检测不到,
    //      phase 卡在 idle, b_exec1 永远 disabled」的问题
    const allow = {
      'b_cap1':     ['idle','first_captured'],
      'b_exec1':    ['idle','first_captured','first_executed','dog_ready','second_captured','done'],
      'b_dogready': ['first_executed','dog_ready','second_captured','done'],
      'b_cap2':     ['dog_ready','first_executed','second_captured','done'],
      'b_exec2':    ['second_captured','done'],
      'b_reset':    ['first_captured','first_executed','dog_ready','second_captured','done'],
    };
    for (const id in allow){
      const el = document.getElementById(id);
      if (el) el.disabled = !allow[id].includes(s.phase);
    }
    // 3) 摘要
    const fmt = (sm) => {
      if (!sm) return '—';
      return (sm.lateral_x!==null ? ('相机位置 X=' + (sm.lateral_x>=0?'+':'') + sm.lateral_x.toFixed(3) + 'm') : '相机位置 X=--')
           + '  位置 Z=' + (sm.camera_z!==null?sm.camera_z.toFixed(2):'--') + 'm'
           + '  占比 ' + (sm.fill_ratio!==null?(sm.fill_ratio*100).toFixed(1):'--') + '%'
           + '  yaw ' + (sm.gimbal_yaw_deg!==null?sm.gimbal_yaw_deg.toFixed(1):'--') + '°';
    };
    const c1 = document.getElementById('cap1_summary');
    const c2 = document.getElementById('cap2_summary');
    // ★ 需求: 捕获结果面板追加「云台→二维码 位置+方向」报告
    const rpt = (t) => t ? ('<br><span style="color:#8cf;font-size:11px;white-space:pre-line">' + String(t) + '</span>') : '';
    if (c1) c1.innerHTML = s.first_capture_msg ? (s.first_capture_msg + '<br>' + fmt(s.first_pose_summary) + rpt(s.first_pose_report)) : '— 未捕获 —';
    if (c2) c2.innerHTML = s.second_capture_msg ? (s.second_capture_msg + '<br>' + fmt(s.second_pose_summary) + rpt(s.second_pose_report)) : '— 未捕获 —';
    // 4) cfg 提示
    if (s.cfg) {
      const ld = document.getElementById('lat_dead');
      const cd = document.getElementById('ctr_dead');
      if (ld) ld.textContent = (s.cfg.exec_min_lat*1000).toFixed(0);
      if (cd) cd.textContent = s.cfg.center_dead_px.toFixed(0);
    }
  } catch(e) {}
}

async function poll(){
  const S = (id, fn) => { try { const el=document.getElementById(id); if(el) el.textContent=fn(); } catch(e){} };
  try {
    const r = await fetch('/qr_nav/pose');
    const j = await r.json();
    if (!j.ready){
      ['live_lat','live_fwd','live_fill','live_yaw'].forEach(id=>S(id,()=>'--'));
      S('live_hint', ()=>j.hint||'等待 QR…');
      S('overlay', ()=>'无 QR');
    } else {
      S('live_lat',  ()=> (j.lateral_x>=0?'+':'') + j.lateral_x.toFixed(3));
      S('live_fwd',  ()=> j.camera_z.toFixed(2));
      S('live_fill', ()=> (j.fill_ratio*100).toFixed(1) + '%');
      S('live_yaw',  ()=> j.gimbal_yaw_deg.toFixed(1));
      S('live_hint', ()=> j.hint);
      S('overlay', ()=>'相机X '+(j.lateral_x>=0?'+':'')+j.lateral_x.toFixed(3)+'m  Z '+j.camera_z.toFixed(2)+'m');
    }
  } catch(e) {}
  setTimeout(poll, 400);
}
poll();

async function pollDogState(){
  try {
    const r = await fetch('/qr_nav/dog_ready');
    const j = await r.json();
    // 同步「一键初始化 / 停止」切换状态
    dogReadyCache = !!(j.motion || j.ready);
    syncInitBtn();
    const setLight = (id, ok) => {
      const el = document.getElementById(id);
      if (el) el.textContent = ok ? '🟢' : '⚪';
    };
    setLight('step_stand',  j.stand);
    setLight('step_force',  j.force);
    setLight('step_motion', j.motion);
    const s = document.getElementById('init_state');
    if (s) {
      if (j.motion)      { s.textContent='✅ 已就绪, 可运动'; s.style.color='#0f0'; }
      else if (j.force)  { s.textContent='⚠ 已力控, 还差开始运动'; s.style.color='#f80'; }
      else if (j.stand)  { s.textContent='⚠ 已起立, 还差力控'; s.style.color='#f80'; }
      else               { s.textContent='❌ 未初始化'; s.style.color='#f44'; }
    }
  } catch(e) {}
}
pollDogState();
setInterval(pollDogState, 3000);
pollPhase();
setInterval(pollPhase, 800);

// ── 🚀 一键执行: 循环调 /qr_nav2/exec_plan 直到对齐 ──
var execBusy = false;
async function execPlanLoop(){
  if (execBusy) return;
  execBusy = true;
  const btn = document.getElementById('b_execplan');
  const hint = document.getElementById('hint');
  const oldTxt = btn ? btn.innerText : '';
  if (btn) btn.disabled = true;
  let i = 0;
  try {
    while (i < 50) {
      i++;
      if (btn) btn.innerText = `🚀 第 ${i} 步...`;
      const r = await fetch('/qr_nav2/exec_plan');
      const t = await r.text();
      log('[' + i + '] /qr_nav2/exec_plan → ' + t, t.indexOf('✗')>=0?'err':'ok');
      if (t.indexOf('已对齐')>=0 && t.indexOf('变倍 OK')>=0) {
        if (hint) hint.textContent = '✓ 一键执行完成 (横偏对齐 + 变倍 OK)';
        break;
      }
      if (t.indexOf('无 QR')>=0) {
        if (hint) hint.textContent = '✗ 无 QR, 中断';
        break;
      }
      await new Promise(r => setTimeout(r, 600));
    }
    if (i >= 50 && hint) hint.textContent = '⏹ 超过 50 步, 自动停止';
  } catch(e) {
    log('一键执行异常: ' + e, 'err');
  } finally {
    if (btn) { btn.innerText = oldTxt || '🚀 一键执行 (循环 横移+变倍, 直至对齐)'; btn.disabled = false; }
    execBusy = false;
  }
}

// ══════════════════════════════════════════════════════════
//  ✨ ▶ 一键完成 (后台线程跑 6 段, 500ms 轮询状态)
//    - 点一次按钮 → 自动: 狗初始化 → ① 捕获 → ① 横移 →
//      狗就位(自动) → ② 捕获 → ② 云台+变倍
//    - 实时进度条 + 步骤文字
//    - 失败自动停止, 保留手动分步兜底
// ══════════════════════════════════════════════════════════
var autoFullBusy = false;
var autoFullTimer = null;
async function autoFullStart(){
  if (autoFullBusy) return;
  autoFullBusy = true;
  const btn  = document.getElementById('b_autofull');
  const prog = document.getElementById('autofull_prog');
  const bar  = document.getElementById('autofull_bar');
  const msg  = document.getElementById('autofull_msg');
  if (btn)  btn.disabled = true;
  if (prog) prog.style.display = 'block';
  if (bar)  bar.style.width = '0%';
  if (msg) { msg.style.color = '#4af'; msg.textContent = '⏳ 启动后台线程 ...'; }
  try {
    const r = await fetch('/qr_nav2/auto_full');
    const t = await r.text();
    log('▶ 一键: ' + t, 'info');
    if (t.indexOf('已在进行中') >= 0) {
      if (msg) msg.textContent = '⏳ ' + t;
    } else {
      autoFullTimer = setInterval(pollAutoFull, 500);
    }
  } catch(e) {
    if (msg) { msg.style.color = '#f44'; msg.textContent = '✗ 启动失败: ' + e; }
    autoFullBusy = false;
    if (btn)  btn.disabled = false;
  }
}
async function pollAutoFull(){
  try {
    const r = await fetch('/qr_nav2/auto_full_status');
    const s = await r.json();
    const bar = document.getElementById('autofull_bar');
    const msg = document.getElementById('autofull_msg');
    const btn = document.getElementById('b_autofull');
    if (bar) bar.style.width = Math.max(0, s.pct) + '%';
    if (msg) msg.textContent = s.msg;
    if (!s.running) {
      clearInterval(autoFullTimer);
      autoFullTimer = null;
      autoFullBusy = false;
      if (btn) btn.disabled = false;
      if (msg) msg.style.color = (s.stage === 'error' || s.pct < 0) ? '#f44' : '#0f0';
      // 阶段条和狗状态都刷一次
      pollPhase();
      pollDogState();
      // 3 秒后清空错误色, 方便下次启动
      setTimeout(function(){
        if (msg) msg.style.color = '#4af';
      }, 3000);
    }
  } catch(e) {}
}
</script>
</body>
</html>"""


# ============================================================
#  注册 V1 + V2 路由
# ============================================================
def _register_qr_nav_routes(app):
    """同时挂 V1 (/qr_nav/*) 和 V2 (/qr_nav2/*)"""
    from flask import Response

    # ── V1 路由 (与 gimbal_qr_dog_auto_drive.py 同款) ──
    @app.route("/qr_nav/stream")
    def _qrnav_stream():
        def gen():
            while True:
                jpg = _qrnav_jpg_buf.wait(0.5)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/qr_nav/pose")
    def _qrnav_pose():
        with _qrnav_pose_lock:
            pose = _qrnav_pose_box[0]
        data = pose_to_command(pose)
        try:
            data["lat_max_step"] = float(
                _QR_NAV_CFG.get("DOG_MAX_STEP_M", 0.30))
        except Exception:
            pass
        return Response(json.dumps(data, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav/capture")
    def _qrnav_capture():
        est = _qrnav_estimator_holder[0]
        if est is None:
            return Response("✗ estimator 未就绪", mimetype="text/plain; charset=utf-8")
        frame = getattr(est, "current_frame", None)
        if frame is None:
            return Response("✗ 无视频帧", mimetype="text/plain; charset=utf-8")
        if getattr(est, "_est_busy", False):
            return Response("⏳ 上一步还没跑完", mimetype="text/plain; charset=utf-8")
        try:
            est._start_estimate_async(frame)
        except Exception as e:
            return Response(f"✗ 触发失败: {e}", mimetype="text/plain; charset=utf-8")
        time.sleep(0.3)
        t0 = time.time()
        while time.time() - t0 < 15.0:
            if not getattr(est, "_est_busy", False):
                break
            time.sleep(0.1)
        pd = getattr(est, "latest_pose_data", None)
        if pd is None:
            return Response("✗ 估计失败", mimetype="text/plain; charset=utf-8")
        with _qrnav_pose_lock:
            _qrnav_pose_box[0] = pd
            _qrnav_pose_ver[0] += 1
        return Response("✓ V1 捕获完成", mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/do_lateral")
    def _qrnav_do_lateral():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = lateral_step_only()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/do_zoom")
    def _qrnav_do_zoom():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            msg = zoom_step_only()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/init_all")
    def _qrnav_init_all():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return "⏳ 上一步还没跑完"
        try:
            ok, msg = _ensure_dog_ready(force_init=True)
            if not ok and "冷却" not in msg:
                msg = f"初始化失败: {msg}"
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav/dog_ready")
    def _qrnav_dog_ready():
        try:
            state = _get_dog_step_state()
        except Exception as e:
            return Response(json.dumps({"stand": False, "force": False,
                                        "motion": False, "err": str(e)}),
                            mimetype="application/json; charset=utf-8")
        return Response(json.dumps(state, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav/zoom/<action>")
    def _qrnav_zoom(action):
        est = _qrnav_estimator_holder[0]
        gimbal = getattr(est, "gimbal", None)
        if est is None or gimbal is None:
            return Response("✗ 云台未就绪", mimetype="text/plain; charset=utf-8")
        try:
            act = str(action).lower().strip()
            zmin = getattr(est, "_zoom_min", 1.0)
            zmax = getattr(est, "_zoom_max", 30.0)
            zstep = getattr(est, "zoom_step_val", 1.0)
            cur = getattr(est, "current_zoom_mag", 1.0)
            if act in ("in", "+"):
                target = min(zmax, cur + zstep)
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = f"✓ 变倍+ → {target:.1f}x"
            elif act in ("out", "-"):
                target = max(zmin, cur - zstep)
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = f"✓ 变倍- → {target:.1f}x"
            elif act == "auto":
                with _qrnav_pose_lock:
                    pose = _qrnav_pose_box[0]
                if pose is None:
                    return Response("✗ 无 QR 数据", mimetype="text/plain; charset=utf-8")
                cmd = pose_to_command(pose)
                fill = cmd.get("fill_ratio") or 0.0
                if fill <= 1e-4:
                    return Response("✗ QR 占比无效", mimetype="text/plain; charset=utf-8")
                k = (_QR_NAV_CFG["ZOOM_TARGET"] / fill) ** 0.5
                target = max(zmin, min(zmax, cur * k))
                gimbal.zoom_to_magnification(target)
                est.current_zoom_mag = target
                msg = f"✓ 自动变倍 → {target:.1f}x"
            elif act == "focus":
                gimbal.auto_focus()
                msg = "✓ 自动对焦已触发"
            else:
                msg = f"未知动作: {act}"
            return Response(msg, mimetype="text/plain; charset=utf-8")
        except Exception as e:
            return Response(f"✗ 变倍失败: {e}",
                            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav")
    def _qrnav_page():
        # V2 文件不提供 V1 HTML (避免文件过大), 让用户跳回 V1 页面
        return Response(
            """<!DOCTYPE html><html><head><meta charset="utf-8">
            <title>V1 已迁移</title></head><body style="background:#1a1a1f;color:#eee;
            font-family:Microsoft YaHei;padding:30px">
            <h2>V1 页面请运行 <code>gimbal_qr_dog_auto_drive.py</code> 访问</h2>
            <p>当前 V2 文件仅提供 <a style="color:#0cf" href="/qr_nav2">两段式 V2 页面</a></p>
            </body></html>""",
            mimetype="text/html; charset=utf-8")

    # ── V2 路由 (核心) ──

    @app.route('/qr_nav2/initial/state')
    def _qrnav2_initial_state():
        return Response(json.dumps(_initial_localization.state(), ensure_ascii=False),
                        mimetype='application/json; charset=utf-8')

    @app.route('/qr_nav2/initial/capture')
    def _qrnav2_initial_capture():
        if _is_af_running() or not _qrnav_poll_busy.acquire(blocking=False):
            return Response('⏳ 导航或捕获正在执行，请稍后再试', mimetype='text/plain; charset=utf-8')
        try:
            return Response(_initial_localization.capture(), mimetype='text/plain; charset=utf-8')
        finally:
            _qrnav_poll_busy.release()

    @app.route('/qr_nav2/initial/gimbal_init')
    def _qrnav2_initial_gimbal_init():
        if _is_af_running() or not _qrnav_poll_busy.acquire(blocking=False):
            return Response('⏳ 导航或捕获正在执行，请稍后再试', mimetype='text/plain; charset=utf-8')
        try:
            return Response(_initial_localization.initialize_gimbal(), mimetype='text/plain; charset=utf-8')
        finally:
            _qrnav_poll_busy.release()

    @app.route('/qr_nav2/initial/execute')
    def _qrnav2_initial_execute():
        if _is_af_running() or not _qrnav_poll_busy.acquire(blocking=False):
            return Response('⏳ 导航或捕获正在执行，请稍后再试', mimetype='text/plain; charset=utf-8')
        dog_lock = getattr(_gpea, '_http_dog_busy', None)
        acquired = False
        try:
            if dog_lock is not None:
                acquired = dog_lock.acquire(blocking=False)
                if not acquired:
                    return Response('⏳ 机器狗控制正在执行', mimetype='text/plain; charset=utf-8')
            return Response(_initial_localization.execute(), mimetype='text/plain; charset=utf-8')
        finally:
            if acquired:
                dog_lock.release()
            _qrnav_poll_busy.release()

    @app.route("/qr_nav2")
    def _qrnav2_page():
        resp = Response(HTML_QR_NAV2_PAGE,
                        mimetype="text/html; charset=utf-8")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/qr_nav2/phase")
    def _qrnav2_phase():
        s = _phase_get()
        # 附带 cfg 给前端显示
        s["cfg"] = {
            "exec_min_lat":    float(_QR_NAV_CFG.get("EXEC_MIN_LAT", 0.005)),
            "center_dead_px":  float(_QR_NAV_CFG.get("CENTER_DEAD_PX", 4.0)),
            "dog_max_step_m":  float(_QR_NAV_CFG.get("DOG_MAX_STEP_M", 0.30)),
        }
        return Response(json.dumps(s, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav2/capture1")
    def _qrnav2_capture1():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            # 捕获本身不依赖狗, 直接拍就行
            # (狗就绪检查放到 exec_lateral 里, 那里 _move_lateral_steps 会自动兜底)
            msg = first_capture_once()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/capture2")
    def _qrnav2_capture2():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            msg = second_capture_once()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/exec_lateral")
    def _qrnav2_exec_lateral():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        dog_lock = getattr(_gpea, "_http_dog_busy", None)
        dog_lock_acquired = False
        try:
            # 执行横移内部串行完成运动使能、横移和零速停止。继续共用机器狗锁，
            # 避免高级控制区的手动命令与横移同时向同一 UDP socket 写数据。
            if dog_lock is not None:
                dog_lock_acquired = dog_lock.acquire(timeout=8.0)
                if not dog_lock_acquired:
                    msg = "✗ 机器狗控制仍在执行，横移未发送，请稍后再点"
                    return Response(msg, mimetype="text/plain; charset=utf-8")
            # ★ 关键修复: 不再前置调 _ensure_dog_ready()
            #   - V2 用 stand_quiet (motion 永远 False), _ensure_dog_ready 会误判
            #   - _init_dog_3steps 走 HTTP 链路, 受 _http_dog_busy 锁影响
            #   - 改让 exec_phase1_lateral → _move_lateral_steps 内部兜底
            #     (主动创建 dog 实例, 未就绪时调 init, gait=None 时调 start_motion+set_gait)
            msg = exec_phase1_lateral()
        except Exception as e:
            import traceback
            msg = f"ERR {e}\n{traceback.format_exc()}"
        finally:
            if dog_lock_acquired:
                dog_lock.release()
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/exec_gimbal_zoom")
    def _qrnav2_exec_gimbal_zoom():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            msg = exec_phase2_gimbal_zoom()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/dog_ready")
    def _qrnav2_dog_ready():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            msg = mark_dog_ready()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/reset")
    def _qrnav2_reset():
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            msg = reset_phase()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/toggle_center_yaw")
    def _qrnav2_toggle_yaw():
        cur = _QR_NAV_CFG.get("CENTER_YAW_SIGN", +1)
        new = -cur
        _QR_NAV_CFG["CENTER_YAW_SIGN"] = new
        return Response(f"✓ CENTER_YAW_SIGN: {cur:+d} → {new:+d}",
                        mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/toggle_center_pitch")
    def _qrnav2_toggle_pitch():
        cur = _QR_NAV_CFG.get("CENTER_PITCH_SIGN", +1)
        new = -cur
        _QR_NAV_CFG["CENTER_PITCH_SIGN"] = new
        return Response(f"✓ CENTER_PITCH_SIGN: {cur:+d} → {new:+d}",
                        mimetype="text/plain; charset=utf-8")

    # ── V2 路由: 把 V1 全部"机器人运动"按钮并进来, 不必切回 V1 页面 ──

    @app.route("/qr_nav2/auto")
    def _qrnav2_auto():
        """V1 兼容: 自动一步 (横移+变倍, 用最近一次 pose, 不重新捕获)"""
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = auto_step_nav()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/align")
    def _qrnav2_align():
        """V1 兼容: 仅横向对齐 (横移, 不动变倍)"""
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = align_step_nav()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/exec_plan")
    def _qrnav2_exec_plan():
        """V1 兼容: 一键执行 (横移+变倍 串行一次) — V2 单步等效"""
        if not _qrnav_poll_busy.acquire(blocking=False):
            return Response("⏳ 上一步还没跑完",
                            mimetype="text/plain; charset=utf-8")
        try:
            ok, imsg = _ensure_dog_ready()
            if not ok:
                return Response(imsg, mimetype="text/plain; charset=utf-8")
            msg = exec_plan_step()
        except Exception as e:
            msg = f"ERR {e}"
        finally:
            _qrnav_poll_busy.release()
        return Response(msg, mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/lateral_sign")
    def _qrnav2_lateral_sign():
        """V1 兼容: 切换 LATERAL_SIGN (横移方向翻转)"""
        cur = _QR_NAV_CFG.get("LATERAL_SIGN", +1)
        new = -cur
        _QR_NAV_CFG["LATERAL_SIGN"] = new
        return Response(
            f"✓ LATERAL_SIGN: {cur:+d} → {new:+d} "
            f"(横移方向{'正常' if new>0 else '已翻转'})",
            mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/yaw_sign")
    def _qrnav2_yaw_sign():
        """V1 兼容: 切换 YAW_SIGN (云台 yaw 方向翻转)"""
        cur = _QR_NAV_CFG.get("YAW_SIGN", +1)
        new = -cur
        _QR_NAV_CFG["YAW_SIGN"] = new
        return Response(
            f"✓ YAW_SIGN: {cur:+d} → {new:+d} "
            f"({'右转正' if new>0 else '左转正'})",
            mimetype="text/plain; charset=utf-8")

    # ── 一键全流程: 前端轮询 /qr_nav2/auto_full_status 看进度 ──

    @app.route("/qr_nav2/auto_full")
    def _qrnav2_auto_full():
        """启动后台线程跑完整 6 段 (init → ① → 横移 → 狗就位 → ② → 云台+变倍)"""
        if _is_af_running():
            return Response("⏳ 一键流程已在进行中", mimetype="text/plain; charset=utf-8")
        _set_af_status(True, "starting", "🚀 启动中 ...", 1)
        threading.Thread(target=_auto_full_worker, daemon=True).start()
        return Response("🚀 一键全流程已启动", mimetype="text/plain; charset=utf-8")

    @app.route("/qr_nav2/auto_full_status")
    def _qrnav2_auto_full_status():
        """前端每 500ms 轮询, 取当前步骤 / 进度 / 状态文本"""
        s = _qrnav_auto_full_status[0]
        if s is None:
            s = {"running": False, "stage": "idle", "msg": "(未启动, 点 ▶ 一键完成)", "pct": 0}
        return Response(json.dumps(s, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8")

    @app.route("/qr_nav2_index")
    def _qrnav2_index():
        return Response(
            """<!DOCTYPE html><html><head><meta charset="utf-8">
            <title>QR→狗 V2 + V1 合并</title>
            <style>body{margin:0;background:#111}iframe{border:0;display:block}
            #cloud{width:100%;height:620px;border-bottom:1px solid #333}
            #qnav{width:100%;height:580px}</style></head>
            <body>
              <iframe id="cloud" src="/"></iframe>
              <iframe id="qnav" src="/qr_nav2"></iframe>
            </body></html>""",
            mimetype="text/html; charset=utf-8")

    return app


# ============================================================
#  启动合并 HTTP
# ============================================================
def start_combined_http_stream(port, owner=None):
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = _gpea._make_flask_app(owner=owner)
    _register_qr_nav_routes(app)
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               debug=False, use_reloader=False),
        daemon=True)
    t.start()
    return t
