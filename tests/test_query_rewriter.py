"""
测试 rewrite_query 工具：
- broad 模式：返回 3 条候选词，包含 note 字段
- targeted 模式：读取 critique gaps，返回 3 条候选词
- targeted 无 section_title → 报错
- targeted 无 critique 结果 → 报错
- 历史搜索词传入 avoided 字段
"""

import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import get_state, reset_state, _make_section_entry
from tools._client import set_client
import tools.query_rewriter as qr_mod

_BROAD_RESULT = (
    '{"queries": ['
    + ", ".join(
        f'{{"query": "检索词{i}", "intent": "维度{i}"}}'
        for i in range(1, 9)
    )
    + "]}"
)
_TARGETED_RESULT = '{"queries": [{"query": "Generali 2025 operating profit", "intent": "填补利润缺口"}, {"query": "Generali 2025 annual report dividend", "intent": "分红数据"}, {"query": "site:generali.com 2025 results", "intent": "官方来源"}]}'


class _FakeCfg:
    CRITIQUE_MODEL    = "fake"
    OUTPUT_DIR        = "/tmp/test_qr"
    MAX_BROAD_SEARCHES = 8


class _FakeMsg:
    def __init__(self, text):
        self.stop_reason = "end_turn"
        self.content = [type("T", (), {"type": "text", "text": text})()]


class _FakeStream:
    def __init__(self, text):
        self._text = text
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def get_final_message(self): return _FakeMsg(self._text)


def _make_client(text):
    class M:
        def stream(self_, **kw): return _FakeStream(text)
    class C:
        messages = M()
    return C()


def _setup(broad_response=_BROAD_RESULT):
    reset_state()
    set_client(_make_client(broad_response), _FakeCfg())
    state = get_state()
    state["original_query"] = "中国金融行业未来趋势分析"
    state["run_dir"] = ""
    return state


def _add_section(state, title, with_review=False):
    sec = _make_section_entry(title=title, order=1)
    if with_review:
        sec["review"] = {
            "is_sufficient": False,
            "needs_search": True,
            "quality_score": 0.5,
            "gaps": ["缺少 operating profit 数据", "缺少 2025 年分红信息"],
            "improvement_suggestions": ["搜索 2025 年报", "查官方披露"],
        }
        sec["final"] = "现有章节内容"
    state["sections"]["sec_001"] = sec
    return sec


def test_broad_returns_max_broad_searches_queries():
    state = _setup()
    result = qr_mod.runner(mode="broad")
    assert "error" not in result, f"不应报错: {result.get('error')}"
    assert result["mode"] == "broad"
    assert len(result["queries"]) == 8   # 对齐 MAX_BROAD_SEARCHES
    assert "全部并行" in result["note"]
    for q in result["queries"]:
        assert "query" in q and "intent" in q
    print(f"[PASS] broad 返回 {len(result['queries'])} 条（对齐 MAX_BROAD_SEARCHES）")


def test_targeted_reads_critique_gaps():
    state = _setup(broad_response=_TARGETED_RESULT)
    _add_section(state, "融资情况横向比较", with_review=True)
    result = qr_mod.runner(mode="targeted", section_title="融资情况横向比较")
    assert "error" not in result, f"不应报错: {result.get('error')}"
    assert result["mode"] == "targeted"
    assert result["section_title"] == "融资情况横向比较"
    assert len(result["queries"]) == 3
    print(f"[PASS] targeted 读 critique: {[q['query'] for q in result['queries']]}")


def test_targeted_no_section_title_returns_error():
    _setup()
    result = qr_mod.runner(mode="targeted")
    assert "error" in result
    assert "section_title" in result["error"]
    print(f"[PASS] targeted 无 section_title 报错: {result['error']}")


def test_targeted_no_critique_returns_error():
    state = _setup()
    _add_section(state, "融资情况", with_review=False)
    result = qr_mod.runner(mode="targeted", section_title="融资情况")
    assert "error" in result
    assert "critique" in result["error"]
    print(f"[PASS] targeted 无 critique 报错: {result['error']}")


def test_invalid_mode_returns_error():
    _setup()
    result = qr_mod.runner(mode="random")
    assert "error" in result
    print(f"[PASS] 非法 mode 报错: {result['error']}")


def test_avoided_contains_past_queries():
    state = _setup(broad_response=_TARGETED_RESULT)
    _add_section(state, "融资情况", with_review=True)
    # 添加历史搜索记录
    state["search_sources"]["src_001"] = {
        "query": "Generali 2025 annual results",
        "section_title": "融资情况",
        "file": "", "summary": "", "result_count": 3,
    }
    result = qr_mod.runner(mode="targeted", section_title="融资情况")
    assert "error" not in result
    assert "Generali 2025 annual results" in result["avoided"]
    print(f"[PASS] avoided 包含历史 query: {result['avoided']}")


if __name__ == "__main__":
    tests = [
        test_broad_returns_max_broad_searches_queries,
        test_targeted_reads_critique_gaps,
        test_targeted_no_section_title_returns_error,
        test_targeted_no_critique_returns_error,
        test_invalid_mode_returns_error,
        test_avoided_contains_past_queries,
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
