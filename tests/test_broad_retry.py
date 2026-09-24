"""
test_broad_retry.py
宽泛搜索失败条目的重试豁免：
- 宽泛批次中失败的 query 记入 broad_failed_queries
- broad_locked 后失败条目的重试放行（错误处理不受主 workflow 锁定限制）
- 豁免一次性（消费制）：重试成功后同 query 不再放行；重试再失败也不再次放行
- 锁定后新 query 仍被拦截
"""

import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.search as search_mod
from tools.registry import reset_state, get_state
from tools._client import set_client
from tools import rate_limiter


class _Cfg:
    TAVILY_API_KEY = "fake"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_test_output_br", "reports")
    MAX_BROAD_SEARCHES = 8
    MAX_TARGETED_PER_SECTION = 5
    RATE_LIMITS = {
        "risk_none": {
            "sources": ["tavily", "arxiv", "github"],
            "min_interval": 0, "max_interval": 0, "concurrency": 10,
            "qph": None, "circuit_break_after": 100, "circuit_break_minutes": 1,
        },
    }

class _FakeClient:
    pass

_cfg = _Cfg()
set_client(_FakeClient(), _cfg)

_FAKE_RESULTS = [
    {"title": "R1", "url": "https://r1.com", "snippet": "snippet1"},
]


def _setup(caller_behavior):
    """caller_behavior: dict query -> 'ok' | 'fail' | ['fail','ok',...]（行为序列）"""
    reset_state()
    rate_limiter.reset()
    set_client(_FakeClient(), _cfg)
    _calls = {}

    def caller(q, n, cfg):
        beh = caller_behavior.get(q)
        if isinstance(beh, list):
            idx = _calls.get(q, 0)
            _calls[q] = idx + 1
            beh = beh[min(idx, len(beh) - 1)]
        if beh == "fail":
            raise RuntimeError("模拟网络失败")
        return _FAKE_RESULTS

    search_mod._CALLERS["tavily"] = caller


def teardown():
    out_root = os.path.join(os.path.dirname(__file__), "_test_output_br")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)


def _lock_broad():
    state = get_state()
    state["broad_locked"] = True
    state["broad_pending_lock"] = False


def test_failed_broad_query_recorded_and_retry_exempted():
    # 宽泛批次：q1 失败 → 记入集合 → 锁定后重试 q1 放行（豁免消费）
    _setup({"q1": ["fail", "ok"]})
    state = get_state()

    r1 = search_mod.runner("q1", "tavily", section_title="")
    assert "error" in r1
    assert "q1" in state["broad_failed_queries"]

    _lock_broad()
    r2 = search_mod.runner("q1", "tavily", section_title="")
    assert "error" not in r2, r2
    assert r2["source_id"]
    # 豁免已消费
    assert "q1" not in state["broad_failed_queries"]
    print("[PASS] 失败条目记入集合，锁定后重试放行")


def test_retry_exemption_consumed_once():
    # 豁免一次性：重试成功后再搜同 query 被锁定拦截
    _setup({"q1": ["fail", "ok"]})
    search_mod.runner("q1", "tavily", section_title="")
    _lock_broad()
    search_mod.runner("q1", "tavily", section_title="")   # 豁免重试，成功
    r3 = search_mod.runner("q1", "tavily", section_title="")
    assert "error" in r3
    assert "锁定" in r3["error"]
    print("[PASS] 豁免一次性，重试成功后同 query 恢复拦截")


def test_new_query_still_blocked_when_locked():
    # 锁定后，非失败条目（新 query）仍被拦截
    _setup({"q1": "fail"})
    search_mod.runner("q1", "tavily", section_title="")
    _lock_broad()
    r = search_mod.runner("q2", "tavily", section_title="")
    assert "error" in r
    assert "锁定" in r["error"]
    print("[PASS] 锁定后新 query 仍被拦截")


def test_retry_fail_no_second_chance():
    # 豁免重试再失败 → 不重新记入集合 → 第三次被锁定拦截（每个失败条目只有一次重试）
    _setup({"q1": "fail"})
    state = get_state()
    search_mod.runner("q1", "tavily", section_title="")
    _lock_broad()
    r2 = search_mod.runner("q1", "tavily", section_title="")   # 豁免重试，仍失败
    assert "error" in r2
    assert "q1" not in state["broad_failed_queries"], "豁免重试再失败不应重新记入（防无限豁免）"
    r3 = search_mod.runner("q1", "tavily", section_title="")
    assert "锁定" in r3["error"]
    print("[PASS] 豁免重试再失败不二次豁免")


if __name__ == "__main__":
    tests = [
        test_failed_broad_query_recorded_and_retry_exempted,
        test_retry_exemption_consumed_once,
        test_new_query_still_blocked_when_locked,
        test_retry_fail_no_second_chance,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    teardown()
    print(f"\n{passed}/{len(tests)} passed")
