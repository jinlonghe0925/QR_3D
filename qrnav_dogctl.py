# -*- coding: utf-8 -*-
"""qrnav_dogctl.py — 机器狗控制 (UDP 直调 + HTTP 回调 + 初始化/就绪判定)

依赖: qrnav_config
"""
import time

from qrnav_config import (
    _QR_NAV_CFG, _qrnav_dog_port_holder, _qrnav_estimator_holder,
    _qrnav_last_init_ts, _QR_NAV_INIT_COOLDOWN,
)

# ============================================================
#  自动驾驶 (与 V1 保持一致, 给 V1 路由用)
# ============================================================
def _call_dog_http(action, value=0.0, gait=None, timeout=15.0):
    """本机 HTTP 回调 /dog/<action> (与参照主页面按钮同一个路由)

    注意: 必须绕过 http_proxy —— 板子上若配了代理, urllib 会把 127.0.0.1
    也丢给代理, 直接返回「ERR 调狗失败」, 表现就是「一键初始化按了没反应」。
    """
    import urllib.request
    import urllib.parse
    qs = {"v": value}
    if gait is not None:
        qs["gait"] = gait
    url = f"http://127.0.0.1:{_qrnav_dog_port_holder[0]}/dog/{action}?" + \
          urllib.parse.urlencode(qs)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"ERR 调狗失败: {e}"


def _dog_msg_bad(msg):
    """判断 /dog/* 返回文本是否为失败 (供三步初始化判断是否继续下一步)"""
    return any(k in (msg or "") for k in ("ERR", "✗", "拒绝", "异常", "未连接"))


def _move_lateral_steps(lat, max_step=None):
    """按相机位置 X 决定左右方向，自动使能、分段横移并零速停止。

    页面 2 不再要求先点“手动开始运动”。每次执行都重新发送
    力控/开始运动/行走 gait，并等待运控完成状态切换，不能依赖可能已经
    失效的本地 ``_gait`` 缓存。横移结束或中途失败时都补发零速帧。
    实际左右移动优先复用页面一的
    ``_handle_dog_action(left/right)``，避免自动导航与手动按钮走两套协议。
    """
    cfg = _QR_NAV_CFG
    step = abs(float(lat))
    lateral_sign = float(cfg.get("LATERAL_SIGN", +1))
    lat_eff = float(lat) * lateral_sign
    direction = "right" if lat_eff > 0 else "left"
    sign = +1 if direction == "left" else -1
    mx = float(max_step if max_step else
               cfg.get("DOG_MAX_MOVE_M", 0.30) or 0.30)
    if mx <= 0:
        mx = step or 0.30
    n = int(step // mx)
    if n * mx < step - 1e-9:
        n += 1
    n = max(1, n)
    per = step / n
    lock_deg = float(cfg.get("DOG_MAX_TURN_DEG", 30.0) or 30.0)

    est = _qrnav_estimator_holder[0]
    if est is None:
        return (direction, 0.0, 0, 0.0, "✗ estimator 未注入, 没法调狗")
    dog = getattr(est, "_dog", None)
    if dog is None:
        owner = getattr(est, "owner", None) or est
        if not hasattr(owner, "_get_dog"):
            return (direction, 0.0, 0, 0.0,
                    "✗ dog 实例不存在, 且无 _get_dog() 方法可创建")
        try:
            dog = owner._get_dog()
            est._dog = dog
            print(f"[V2.move] dog 实例未存在, 已通过 owner._get_dog() 主动创建 "
                  f"(ip={getattr(owner, '_dog_ip', '?')})")
        except Exception as e:
            return (direction, 0.0, 0, 0.0, f"✗ 创建 dog 实例失败: {e}")

    head = []
    # 日志393..401：RLWalk从接收到kRunPolicy约3.1s；等待期间Dog心跳持续运行。
    settle_s = max(4.0, float(cfg.get("DOG_MOTION_SETTLE_S", 4.0) or 0.0))
    try:
        print("[V2.move] 执行横移: 复用页面1模式 → RLWalk → 等待切换完成")
        walk_gait = dog.prepare_rl_walk()
        if settle_s > 0.0:
            print(f"[V2.move] 等待运控状态稳定 {settle_s:.1f}s，再发送横移")
            time.sleep(settle_s)
        head.append(
            f"✓ 共用RLWalk指令0x{walk_gait:08X}，稳定等待 {settle_s:.1f}s")
    except Exception as e:
        return (direction, 0.0, 0, 0.0,
                f"✗ 横移中止: 开始运动/行走 gait 准备失败: {e}")

    move_handler = getattr(est, "_handle_dog_action", None)
    move_path = ("页面手动 left/right 控制入口" if callable(move_handler)
                 else "兼容 dog.move 入口")
    diag = [
        f"诊断: dog={type(dog).__name__}, addr={getattr(dog, '_addr', '?')}",
        f"诊断: ready={getattr(dog, 'ready', '?')}, "
        f"_gait=0x{getattr(dog, '_gait', 0) or 0:08X}",
        f"诊断: 相机X={float(lat):+.3f}m × LATERAL_SIGN={lateral_sign:+g} "
        f"→ direction={direction}",
        f"诊断: per={per:.3f}m × {n}段, path={move_path}",
    ]
    diag_text = " | ".join(diag)
    print("[V2.move] ========== 横移诊断 ==========")
    for item in diag:
        print(f"[V2.move] {item}")
    print("[V2.move] ===================================")

    move_rets = []
    for i in range(n):
        try:
            if callable(move_handler):
                print(f"[V2.move] [{i + 1}/{n}] 调页面控制入口: "
                      f"_handle_dog_action('{direction}', value={per:.3f})")
                ret = move_handler(direction, value=per)
                print(f"[V2.move] [{i + 1}/{n}] → {ret}")
                if _dog_msg_bad(ret) or not str(ret).lstrip().startswith("✓"):
                    move_rets.append(f"[{i + 1}/{n}] ✗ 页面控制入口返回: {ret}")
                    break
                move_rets.append(f"[{i + 1}/{n}] {ret}")
            else:
                print(f"[V2.move] [{i + 1}/{n}] 兼容调 dog.move(" 
                      f"y={sign * per:+.3f}, max_move_m={mx}, "
                      f"max_turn_deg={lock_deg})")
                # 只传旧版 robot_interface.py 已有的参数，避免部署文件版本不一致时
                # 因新增关键字参数不存在而完全不移动。
                ok = dog.move(y=sign * per, max_move_m=mx,
                              max_turn_deg=lock_deg)
                print(f"[V2.move] [{i + 1}/{n}] → {ok}")
                if not ok:
                    move_rets.append(
                        f"[{i + 1}/{n}] ✗ 被安全锁拒绝 (|y|={per:.2f}m > {mx}m)")
                    break
                move_rets.append(f"[{i + 1}/{n}] ✓ {per:.2f}m 兼容直调")
        except Exception as e:
            import traceback
            print(f"[V2.move] [{i + 1}/{n}] ✗ 横移异常:\n"
                  f"{traceback.format_exc()}")
            move_rets.append(f"[{i + 1}/{n}] ✗ 横移异常: {e}")
            break

    # move() 默认会在段末发送停止帧；这里无条件再补一次，覆盖分段结束、
    # 安全锁拒绝及移动中途异常，保证按钮返回前机器狗已经收到零速命令。
    try:
        dog.stop_move()
        move_rets.append("✓ 已完成计划横移并发送零速（非实测里程到位反馈）")
    except Exception as e:
        move_rets.append(f"✗ 停止运动失败: {e}")

    msg = " | ".join(move_rets)
    if head:
        msg = " | ".join(head) + " || " + msg
    return direction, step, n, per, msg + "\n" + diag_text


# ============================================================
#  V1 用的辅助 (一键初始化 / 狗状态) — V2 同样用
# ============================================================
def _get_dog_obj():
    est = _qrnav_estimator_holder[0]
    if est is None:
        return None
    return getattr(est, "_dog", None)


def _init_dog_3steps():
    """起立 + 力控 + 开始运动 —— 与参照主页面 ① 起立 → ② 力控 → ③ 开始运动 等价

    关键: 三步必须都真的下发, 否则「狗腿不动」或「移动指令被运控丢弃」。
    旧实现用 _is_dog_fully_ready() 一读 True 就提前 return (只发了 stand),
    这是 V2 页面点「一键初始化」后狗不原地踏步的直接原因。
    """
    now = time.time()
    elapsed = now - _qrnav_last_init_ts[0]
    if elapsed < _QR_NAV_INIT_COOLDOWN:
        return (f"⏳ 冷却中 (距上次初始化 {elapsed:.1f}s < "
                f"{_QR_NAV_INIT_COOLDOWN:.1f}s), 请稍候再按")
    _qrnav_last_init_ts[0] = now
    # stand入口统一调用prepare_rl_walk，已包含所需的起立/力控/启停/RLWalk。
    # 不能再追加force和motion：会中断RL或再次切换启停。
    r = _call_dog_http("stand", value=0.0)
    return r


def _get_dog_step_state():
    """狗的三步使能状态 (供页面指示灯 + 「一键初始化/停止」切换判断)

    判定优先级:
      1) estimator 里的 Dog 实例 (ready + gait 区分静默起立与开始运动)
      2) 退回读 /dog_msg 文本 —— 注意「✓ 已停止并趴下复位 / 趴下 / 阻尼」
         必须判为「未就绪」, 否则停止后会误显示为「已就绪」, 切换按钮卡死
    """
    state = {"stand": False, "force": False, "motion": False, "ready": False}
    dog = _get_dog_obj()
    if dog is not None:
        state["stand"] = bool(getattr(dog, '_stand_requested', False))
        state["force"] = bool(getattr(dog, '_force_requested', False))
        state["motion"] = bool(getattr(dog, '_motion_requested', False))
        state["ready"] = (state["motion"] and getattr(dog, '_gait', None) == 0x21010420)
        return state
    try:
        import urllib.request
        url = f"http://127.0.0.1:{_qrnav_dog_port_holder[0]}/dog_msg"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1) as r:
            msg = r.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return state
    if msg == "" or "未连接" in msg:
        return state
    # 停止 / 趴下 / 阻尼 / 断开 之后一律视为「未就绪」
    if any(k in msg for k in ("停止", "趴下", "阻尼", "断开")):
        return state
    state["stand"] = state["force"] = state["motion"] = True
    state["ready"] = True
    return state


def _is_dog_fully_ready():
    try:
        return _get_dog_step_state().get("ready", False)
    except Exception:
        return False


def _ensure_dog_ready(force_init=False):
    if not force_init and _is_dog_fully_ready():
        return True, "狗已就绪"
    msg = _init_dog_3steps()
    if "ERR" in msg or "失败" in msg:
        return False, msg
    if "冷却" in msg:
        return _is_dog_fully_ready(), msg
    return True, msg
