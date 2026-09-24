"""
rate_limiter.py — 分级限速 + 熔断器
管理各搜索源的请求间隔、串行锁、每小时限额、熔断状态。
状态维护在模块级 dict，不进 registry._state（LLM 不需要感知）。

使用方式（在 search.runner 里）：
    ok, err = rate_limiter.acquire(source, cfg)
    if not ok:
        return {"error": err}
    try:
        result = call_with_retry(...)
    finally:
        rate_limiter.release(source, success=result["ok"] if result else False, cfg=cfg)
"""

import random
import threading
import time

# session 级状态：{source: {...}}
_rs: dict = {}
_rs_lock = threading.Lock()


# ── 内部工具 ─────────────────────────────────────────────────────────────────

def _risk_cfg(source: str, cfg) -> dict:
    """返回该源的限速参数 dict；未命中任何等级时按 risk_none 处理。"""
    rl = getattr(cfg, "RATE_LIMITS", {})
    for level_params in rl.values():
        if source in level_params.get("sources", []):
            return level_params
    return {
        "min_interval": 1, "max_interval": 1, "concurrency": 5,
        "qph": None, "circuit_break_after": 5, "circuit_break_minutes": 10,
    }


def _get(source: str, concurrency: int) -> dict:
    """懒初始化并返回该源的状态 dict。"""
    with _rs_lock:
        if source not in _rs:
            _rs[source] = {
                "sem":                threading.Semaphore(concurrency),
                "last_call_at":       0.0,
                "fail_streak":        0,
                "circuit_open_until": 0.0,
                "hour_calls":         [],   # list[float]，记录调用时间戳
                "data_lock":          threading.Lock(),  # 保护 last_call_at / hour_calls
            }
        return _rs[source]


# ── 公开接口 ─────────────────────────────────────────────────────────────────

def acquire(source: str, cfg) -> tuple[bool, str]:
    """
    尝试获取调用许可：
    1. 检查熔断器
    2. 检查 QPH
    3. 获取并发槽位（concurrency=1 时串行等待）
    4. 执行 jitter sleep

    返回 (True, "") 表示可以调用；(False, error_msg) 表示被限制。
    获取成功后必须调用 release() 释放槽位。
    """
    rc = _risk_cfg(source, cfg)
    st = _get(source, rc.get("concurrency", 5))
    now = time.time()

    # 1. 熔断器
    if now < st["circuit_open_until"]:
        remaining = int(st["circuit_open_until"] - now)
        return False, (
            f"搜索源 '{source}' 熔断中（连续失败 {rc.get('circuit_break_after', 3)} 次），"
            f"约 {remaining}s 后自动恢复，期间请换用其他搜索源"
        )

    # 2. QPH
    qph = rc.get("qph")
    if qph is not None:
        with st["data_lock"]:
            cutoff = now - 3600
            st["hour_calls"] = [t for t in st["hour_calls"] if t > cutoff]
            if len(st["hour_calls"]) >= qph:
                return False, (
                    f"搜索源 '{source}' 每小时限额 {qph} 次已达上限，"
                    "请等待下一个小时窗口或换用其他搜索源"
                )

    # 3. 并发槽（concurrency=1 时串行阻塞）
    st["sem"].acquire()

    # 4. Jitter sleep（在持有槽位时执行，保证串行情况下请求间距）
    with st["data_lock"]:
        last = st["last_call_at"]
    elapsed = time.time() - last
    interval = random.uniform(rc.get("min_interval", 1), rc.get("max_interval", 1))
    gap = interval - elapsed
    if gap > 0:
        time.sleep(gap)

    return True, ""


def release(source: str, success: bool, cfg) -> None:
    """
    记录调用结果，更新 last_call_at / fail_streak / 熔断状态，释放并发槽。
    必须在 acquire() 成功后的 finally 块中调用。
    """
    rc = _risk_cfg(source, cfg)
    st = _rs.get(source)
    if st is None:
        return

    now = time.time()
    with st["data_lock"]:
        st["last_call_at"] = now
        st["hour_calls"].append(now)
        if success:
            st["fail_streak"] = 0
        else:
            st["fail_streak"] += 1
            threshold = rc.get("circuit_break_after", 3)
            if st["fail_streak"] >= threshold:
                minutes = rc.get("circuit_break_minutes", 60)
                st["circuit_open_until"] = now + minutes * 60
                st["fail_streak"] = 0  # 熔断后重置，下次恢复后重新计数

    st["sem"].release()


def reset(source: str = None) -> None:
    """清除限速状态，测试用。"""
    with _rs_lock:
        if source:
            _rs.pop(source, None)
        else:
            _rs.clear()


def status(source: str, cfg) -> dict:
    """返回该源当前限速状态摘要，供调试用。"""
    rc = _risk_cfg(source, cfg)
    st = _rs.get(source)
    if st is None:
        return {"source": source, "status": "未初始化"}
    now = time.time()
    cutoff = now - 3600
    with st["data_lock"]:
        hour_count = sum(1 for t in st["hour_calls"] if t > cutoff)
        circuit_open = now < st["circuit_open_until"]
    return {
        "source":          source,
        "circuit_open":    circuit_open,
        "circuit_until":   int(st["circuit_open_until"] - now) if circuit_open else 0,
        "fail_streak":     st["fail_streak"],
        "hour_calls":      hour_count,
        "qph_limit":       rc.get("qph"),
        "concurrency":     rc.get("concurrency"),
        "interval_range":  (rc.get("min_interval"), rc.get("max_interval")),
    }
