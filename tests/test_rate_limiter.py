"""
test_rate_limiter.py
分级限速器单元测试：熔断、QPH、jitter、串行锁、成功重置 fail_streak。
全部 mock time.sleep，不实际等待。
"""

import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import rate_limiter

# ── mock config ───────────────────────────────────────────────────────────────

class _Cfg:
    RATE_LIMITS = {
        "risk_high": {
            "sources":               ["xhs", "zhihu"],
            "min_interval":          3,
            "max_interval":          8,
            "concurrency":           1,
            "qph":                   5,
            "circuit_break_after":   3,
            "circuit_break_minutes": 1,   # 1 分钟，便于测试
        },
        "risk_none": {
            "sources":               ["tavily"],
            "min_interval":          0,
            "max_interval":          0,
            "concurrency":           5,
            "qph":                   None,
            "circuit_break_after":   5,
            "circuit_break_minutes": 10,
        },
    }

cfg = _Cfg()


def setup():
    rate_limiter.reset()


# ── helpers ───────────────────────────────────────────────────────────────────

def _acquire_release(source, success=True):
    ok, err = rate_limiter.acquire(source, cfg)
    if ok:
        rate_limiter.release(source, success=success, cfg=cfg)
    return ok, err


# ── 测试 ─────────────────────────────────────────────────────────────────────

def test_acquire_release_normal(monkeypatch_sleep=True):
    """正常 acquire/release 不报错。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep"):
        ok, err = _acquire_release("xhs", success=True)
    assert ok, f"应成功：{err}"
    print("[PASS] acquire/release 正常")


def test_circuit_breaker_opens_after_n_failures():
    """连续失败 circuit_break_after 次后熔断。"""
    setup()
    import unittest.mock as mock
    threshold = cfg.RATE_LIMITS["risk_high"]["circuit_break_after"]
    with mock.patch("tools.rate_limiter.time.sleep"):
        for _ in range(threshold):
            ok, _ = rate_limiter.acquire("xhs", cfg)
            assert ok
            rate_limiter.release("xhs", success=False, cfg=cfg)

        # 下一次应被熔断
        ok, err = rate_limiter.acquire("xhs", cfg)
    assert not ok, "熔断后应拒绝"
    assert "熔断" in err
    print(f"[PASS] 连续 {threshold} 次失败后熔断: {err[:60]}")


def test_circuit_breaker_recovers_after_timeout():
    """熔断时间到期后恢复。"""
    setup()
    import unittest.mock as mock
    threshold = cfg.RATE_LIMITS["risk_high"]["circuit_break_after"]
    with mock.patch("tools.rate_limiter.time.sleep"):
        for _ in range(threshold):
            ok, _ = rate_limiter.acquire("xhs", cfg)
            assert ok
            rate_limiter.release("xhs", success=False, cfg=cfg)

    # 手动将 circuit_open_until 设为过去
    st = rate_limiter._rs["xhs"]
    st["circuit_open_until"] = time.time() - 1

    with mock.patch("tools.rate_limiter.time.sleep"):
        ok, err = rate_limiter.acquire("xhs", cfg)
        if ok:
            rate_limiter.release("xhs", success=True, cfg=cfg)
    assert ok, f"熔断超时后应恢复：{err}"
    print("[PASS] 熔断时间过期后恢复")


def test_success_resets_fail_streak():
    """成功调用将 fail_streak 归零。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep"):
        # 2 次失败
        for _ in range(2):
            ok, _ = rate_limiter.acquire("xhs", cfg)
            assert ok
            rate_limiter.release("xhs", success=False, cfg=cfg)

        # 1 次成功
        ok, _ = rate_limiter.acquire("xhs", cfg)
        assert ok
        rate_limiter.release("xhs", success=True, cfg=cfg)

    assert rate_limiter._rs["xhs"]["fail_streak"] == 0
    print("[PASS] 成功后 fail_streak 归零")


def test_qph_limit():
    """QPH 达到上限后拒绝。"""
    setup()
    import unittest.mock as mock
    limit = cfg.RATE_LIMITS["risk_high"]["qph"]
    with mock.patch("tools.rate_limiter.time.sleep"):
        for _ in range(limit):
            ok, _ = _acquire_release("xhs", success=True)
            assert ok

        ok, err = rate_limiter.acquire("xhs", cfg)
    assert not ok, "QPH 达限后应拒绝"
    assert "每小时限额" in err
    print(f"[PASS] QPH 限额 {limit} 次后拒绝: {err[:60]}")


def test_qph_window_slides():
    """超过 1 小时的调用记录应从计数中移除。"""
    setup()
    import unittest.mock as mock
    limit = cfg.RATE_LIMITS["risk_high"]["qph"]

    # 注入 limit 条过期记录（2 小时前）
    rate_limiter._get("xhs", 1)
    with rate_limiter._rs["xhs"]["data_lock"]:
        old_ts = time.time() - 7200
        rate_limiter._rs["xhs"]["hour_calls"] = [old_ts] * limit

    with mock.patch("tools.rate_limiter.time.sleep"):
        ok, err = rate_limiter.acquire("xhs", cfg)
        if ok:
            rate_limiter.release("xhs", success=True, cfg=cfg)
    assert ok, f"过期记录应被清除，本次应允许：{err}"
    print("[PASS] QPH 窗口滑动，过期记录清除后允许调用")


def test_risk_none_no_qph():
    """risk_none 源不受 QPH 限制。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep"):
        for _ in range(100):
            ok, err = _acquire_release("tavily", success=True)
            assert ok, f"risk_none 源不应被 QPH 拦截: {err}"
    print("[PASS] risk_none 无 QPH 限制，100 次全通过")


def test_jitter_sleep_called():
    """高风险源调用时必须触发 sleep。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep") as mock_sleep:
        # 先让 last_call_at = now，强制触发间隔
        rate_limiter._get("xhs", 1)
        rate_limiter._rs["xhs"]["last_call_at"] = time.time()
        ok, _ = rate_limiter.acquire("xhs", cfg)
        if ok:
            rate_limiter.release("xhs", success=True, cfg=cfg)
    assert mock_sleep.called, "高风险源应调用 time.sleep"
    print(f"[PASS] jitter sleep 被调用，sleep({mock_sleep.call_args})")


def test_concurrency_1_serializes():
    """concurrency=1 时第二个线程阻塞直到第一个释放。"""
    setup()
    import unittest.mock as mock
    results = []
    barrier = threading.Event()

    def worker(n):
        with mock.patch("tools.rate_limiter.time.sleep"):
            ok, _ = rate_limiter.acquire("xhs", cfg)
        if ok:
            results.append(f"start-{n}")
            barrier.wait(timeout=2)
            rate_limiter.release("xhs", success=True, cfg=cfg)
            results.append(f"end-{n}")

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    time.sleep(0.05)   # 确保 t1 先拿到锁
    t2.start()
    time.sleep(0.1)

    # t2 此时应阻塞在 sem.acquire()，results 只有 t1 的 start
    assert results == ["start-1"], f"t2 不应在 t1 释放前开始: {results}"

    barrier.set()      # 让 t1 释放
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert len(results) == 4
    assert results[0] == "start-1"
    assert results[1] == "end-1"
    assert results[2] == "start-2"
    print(f"[PASS] concurrency=1 串行执行: {results}")


def test_unknown_source_defaults_to_risk_none():
    """未知源走 risk_none 默认参数，不报错。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep"):
        ok, err = _acquire_release("unknown_source", success=True)
    assert ok, f"未知源应走 risk_none 默认值: {err}"
    print("[PASS] 未知源默认 risk_none，正常放行")


def test_status_returns_dict():
    """status() 返回包含关键字段的 dict。"""
    setup()
    import unittest.mock as mock
    with mock.patch("tools.rate_limiter.time.sleep"):
        _acquire_release("xhs", success=True)
    s = rate_limiter.status("xhs", cfg)
    assert s["source"] == "xhs"
    assert "circuit_open" in s
    assert "hour_calls" in s
    assert s["hour_calls"] == 1
    print(f"[PASS] status(): {s}")


if __name__ == "__main__":
    tests = [
        test_acquire_release_normal,
        test_circuit_breaker_opens_after_n_failures,
        test_circuit_breaker_recovers_after_timeout,
        test_success_resets_fail_streak,
        test_qph_limit,
        test_qph_window_slides,
        test_risk_none_no_qph,
        test_jitter_sleep_called,
        test_concurrency_1_serializes,
        test_unknown_source_defaults_to_risk_none,
        test_status_returns_dict,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            import traceback
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
