"""
测试 search.runner 的落盘、state 更新、tool_result 格式、多源 dispatch。
mock _CALLERS["tavily"]，不发真实请求。
"""

import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.search as search_mod
from tools.registry import get_state, reset_state, _make_section_entry, _next_sec_key, state_lock
from tools._client import set_client
from tools import rate_limiter as _rl

# ─── 准备：mock config + client ──────────────────────────────────────────────

class _FakeCfg:
    TAVILY_API_KEY = "fake"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_test_output", "reports")
    MAX_BROAD_SEARCHES = 8
    MAX_TARGETED_PER_SECTION = 5
    GH_BIN = "gh"
    XHS_BACKEND = "opencli"
    ZHIHU_COOKIE = ""
    SEMANTIC_SCHOLAR_API_KEY = ""
    RATE_LIMITS = {
        "risk_none": {
            "sources": ["tavily", "arxiv", "semantic_scholar", "github",
                        "xiaohongshu", "zhihu", "weixin", "weibo", "google_scholar"],
            "min_interval": 0, "max_interval": 0, "concurrency": 10,
            "qph": None, "circuit_break_after": 100, "circuit_break_minutes": 1,
        },
    }

class _FakeClient:
    pass

_cfg = _FakeCfg()
set_client(_FakeClient(), _cfg)

_SEARCH_DIR = os.path.join(os.path.dirname(_cfg.OUTPUT_DIR), "search_results")

_FAKE_RESULTS = [
    {"title": "标题A", "url": "https://a.com", "snippet": "内容A"},
    {"title": "标题B", "url": "https://b.com", "snippet": "内容B"},
    {"title": "标题C", "url": "https://c.com", "snippet": "内容C"},
]

_orig_callers = dict(search_mod._CALLERS)


def _fake_caller(q, n, cfg):
    return _FAKE_RESULTS


def _add_section(title: str):
    """向 state 中注入一个测试章节，供定向补搜测试使用。"""
    state = get_state()
    with state_lock:
        key = _next_sec_key()
        state["sections"][key] = _make_section_entry(title=title)


def setup():
    reset_state()
    _rl.reset()
    set_client(_FakeClient(), _cfg)
    search_mod._CALLERS["tavily"] = _fake_caller
    if os.path.exists(_SEARCH_DIR):
        shutil.rmtree(_SEARCH_DIR)


def teardown():
    search_mod._CALLERS.update(_orig_callers)
    out_root = os.path.join(os.path.dirname(__file__), "_test_output")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)


# ─── 测试 ─────────────────────────────────────────────────────────────────────

def test_tool_result_is_compact():
    setup()
    result = search_mod.runner("剧本杀市场规模")  # 宽泛搜索，不需要章节存在
    assert "source_id" in result
    assert "summary" in result
    assert "result_count" in result
    assert "results" not in result, "tool_result 不应包含原始 results"
    assert result["result_count"] == 3
    assert result["source"] == "tavily"
    print(f"[PASS] tool_result 紧凑: {result}")


def test_raw_results_written_to_disk():
    setup()
    result = search_mod.runner("剧本杀市场规模")
    src_id = result["source_id"]
    file_path = get_state()["search_sources"][src_id]["file"]
    assert os.path.exists(file_path), f"文件不存在: {file_path}"

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["query"] == "剧本杀市场规模"
    assert data["source"] == "tavily"
    assert len(data["results"]) == 3
    assert data["results"][0]["title"] == "标题A"
    print(f"[PASS] 原始结果落盘（含 source 字段）: {file_path}")


def test_state_updated():
    setup()
    result = search_mod.runner("剧本杀市场规模")
    src_id = result["source_id"]
    state = get_state()
    assert src_id in state["search_sources"]
    entry = state["search_sources"][src_id]
    assert entry["query"] == "剧本杀市场规模"
    assert entry["source"] == "tavily"
    assert os.path.exists(entry["file"])
    print(f"[PASS] state 更新: src_id={src_id}, source={entry['source']}")


def test_src_id_includes_source():
    """src_id 格式为 src_<source>_<md5>。"""
    setup()
    result = search_mod.runner("测试source前缀")
    assert result["source_id"].startswith("src_tavily_"), f"src_id 应含 source 前缀: {result['source_id']}"
    print(f"[PASS] src_id 含 source 前缀: {result['source_id']}")


def test_same_query_same_source_same_id():
    """相同 query + 相同 source → 相同 src_id。"""
    setup()
    id1 = search_mod._make_src_id("同一个 query", "tavily")
    id2 = search_mod._make_src_id("同一个 query", "tavily")
    assert id1 == id2
    print(f"[PASS] 相同 query+source → 相同 src_id: {id1}")


def test_different_source_different_id():
    """相同 query + 不同 source → 不同 src_id。"""
    id_t = search_mod._make_src_id("同一个 query", "tavily")
    id_a = search_mod._make_src_id("同一个 query", "arxiv")
    assert id_t != id_a
    print(f"[PASS] 不同 source → 不同 src_id: {id_t} vs {id_a}")


def test_summary_content():
    setup()
    result = search_mod.runner("测试摘要")
    assert "标题A" in result["summary"]
    print(f"[PASS] summary 包含标题: {result['summary']}")


def test_unknown_source_returns_error():
    setup()
    result = search_mod.runner("测试", source="unknown_source")
    assert "error" in result
    assert "不支持" in result["error"]
    print(f"[PASS] 未知 source 返回 error: {result['error'][:60]}")


def test_search_failure_handled():
    setup()
    search_mod._CALLERS["tavily"] = lambda q, n, cfg: [{"title": "搜索失败", "url": "", "snippet": "错误: timeout"}]
    result = search_mod.runner("失败查询")
    assert "source_id" in result
    assert result["summary"] == "搜索失败"
    print(f"[PASS] 搜索失败也落盘并返回摘要: {result}")


def test_broad_locked_after_first_batch():
    """第一次宽泛搜索完成后 broad_pending_lock=True，broad_locked 仍为 False（等下一 turn 提升）。"""
    setup()
    r1 = search_mod.runner("宽泛搜索1")
    assert "error" not in r1, f"第一次宽泛搜索应成功: {r1}"

    state = get_state()
    assert state["broad_pending_lock"] is True
    assert state["broad_locked"] is False

    r2 = search_mod.runner("宽泛搜索2")
    assert "error" not in r2, f"同批第二次宽泛搜索应成功: {r2}"

    # 模拟 agent.py 在下一 turn 提升锁
    state["broad_locked"] = True
    state["broad_pending_lock"] = False

    r3 = search_mod.runner("宽泛搜索3")
    assert "error" in r3
    assert "锁定" in r3["error"]
    print(f"[PASS] broad_pending_lock: r1={r1['broad_searches_done']}, r3 error={r3['error'][:40]}")


def test_targeted_per_section_limit():
    """定向补搜达到 MAX_TARGETED_PER_SECTION 后被拒。"""
    setup()
    _add_section("某章节")
    _cfg.MAX_TARGETED_PER_SECTION = 2

    r1 = search_mod.runner("定向1", section_title="某章节")
    # 第一次会置 pending_save，需重置状态以允许继续搜索
    state = get_state()
    _, sec = __import__("tools.registry", fromlist=["get_section_by_title"]).get_section_by_title("某章节")
    sec["status"] = "draft"

    r2 = search_mod.runner("定向2", section_title="某章节")
    assert "error" not in r1 and "error" not in r2

    sec["status"] = "draft"
    r3 = search_mod.runner("定向3", section_title="某章节")
    assert "error" in r3
    assert "上限" in r3["error"]

    _cfg.MAX_TARGETED_PER_SECTION = 5
    print(f"[PASS] targeted limit: r3 error={r3['error'][:50]}")


def test_targeted_returns_remaining():
    """定向补搜返回 targeted_done / targeted_remaining。"""
    setup()
    _add_section("某章节")

    r = search_mod.runner("定向查询", section_title="某章节")
    assert "targeted_done" in r
    assert r["targeted_done"] == 1
    assert r["targeted_remaining"] == 4
    assert r["source"] == "tavily"
    print(f"[PASS] targeted_remaining: {r['targeted_remaining']}")


def test_multi_source_dispatch():
    """不同 source 走不同 caller，结果都落盘。"""
    setup()
    for src in ("arxiv", "semantic_scholar", "github"):
        search_mod._CALLERS[src] = _fake_caller

    for src in ("arxiv", "semantic_scholar", "github"):
        r = search_mod.runner("Python LLM", source=src)
        assert "error" not in r, f"source={src} 应成功: {r}"
        assert r["source"] == src
        assert r["source_id"].startswith(f"src_{src}_")
        assert get_state()["search_sources"][r["source_id"]]["source"] == src

    print("[PASS] 多源 dispatch 和 src_id 前缀均正确")


if __name__ == "__main__":
    tests = [
        test_tool_result_is_compact,
        test_raw_results_written_to_disk,
        test_state_updated,
        test_src_id_includes_source,
        test_same_query_same_source_same_id,
        test_different_source_different_id,
        test_summary_content,
        test_unknown_source_returns_error,
        test_search_failure_handled,
        test_broad_locked_after_first_batch,
        test_targeted_per_section_limit,
        test_targeted_returns_remaining,
        test_multi_source_dispatch,
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
