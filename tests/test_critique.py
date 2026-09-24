"""
测试 critique_section.runner：
- 正常评审草稿内容
- 不足评审
- 无内容时报错
- 达到最大反思次数强制通过
- reflect_count 自增
- _try_parse_result 各分支
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import get_state, reset_state, _make_section_entry
from tools._client import set_client
import tools.reflect as reflect_mod


class _FakeCfg:
    CRITIQUE_MODEL = "fake"
    MAX_REFLECT_PER_SECTION = 3

class _FakeResponse:
    def __init__(self, text):
        self.stop_reason = "end_turn"
        self.content = [type("T", (), {"type": "text", "text": text})()]

class _FakeStream:
    """client.messages.stream 的假实现（子 LLM 走流式调用）。"""
    def __init__(self, resp):
        self._resp = resp
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def get_final_message(self):
        return self._resp

def _make_client(response_text):
    resp = _FakeResponse(response_text)
    return type("C", (), {
        "messages": type("M", (), {
            "stream": staticmethod(lambda **kw: _FakeStream(resp)),
            "create": staticmethod(lambda **kw: resp),
        })()
    })()

_GOOD_JSON = '{"is_sufficient": true, "quality_score": 0.9, "gaps": [], "improvement_suggestions": [], "stop_reason": "sufficient"}'
_BAD_JSON  = '{"is_sufficient": false, "quality_score": 0.4, "gaps": ["缺少年度数据"], "improvement_suggestions": ["搜索年度报告"], "stop_reason": "needs_more_search"}'

_SAMPLE_CONTENT = "2020年市场规模达100亿元，同比增长30%。头部品牌xxx占市场30%份额，包括若干知名公司。"


def setup(response_text=_GOOD_JSON):
    reset_state()
    set_client(_make_client(response_text), _FakeCfg())


def _add_section(title, content):
    state = get_state()
    state["sections"]["sec_001"] = _make_section_entry(
        description="", initial_query="", order=1, title=title)
    state["sections"]["sec_001"]["draft"] = content
    return state


def test_normal_evaluation():
    setup(_GOOD_JSON)
    state = _add_section("市场概况", _SAMPLE_CONTENT)
    state["sections"]["sec_001"]["criteria"] = "需要包含市场规模数据和品牌分析"

    result = reflect_mod.runner("市场概况", iteration=1)

    assert "error" not in result
    assert result["is_sufficient"] is True
    assert result["quality_score"] == 0.9
    assert result["iteration"] == 1
    print(f"[PASS] 正常评审: is_sufficient={result['is_sufficient']}, score={result['quality_score']}")


def test_insufficient_evaluation():
    setup(_BAD_JSON)
    _add_section("市场概况", _SAMPLE_CONTENT)

    result = reflect_mod.runner("市场概况", iteration=1)

    assert result["is_sufficient"] is False
    assert len(result["gaps"]) > 0
    assert len(result["improvement_suggestions"]) > 0
    print(f"[PASS] 不足评审: gaps={result['gaps']}")


def test_no_content_returns_error():
    setup()
    state = get_state()
    state["sections"]["sec_001"] = _make_section_entry(
        description="", initial_query="", order=1, title="空章节")
    result = reflect_mod.runner("空章节", iteration=1)
    assert "error" in result
    assert "save_section" in result["error"]
    print(f"[PASS] 无已生成内容报错: {result['error']}")


def test_max_reflect_force_pass():
    setup()
    _add_section("测试章节", _SAMPLE_CONTENT)
    for i in range(3):
        reflect_mod.runner("测试章节", iteration=i+1)

    result = reflect_mod.runner("测试章节", iteration=4)
    assert result["is_sufficient"] is True
    assert "note" in result
    assert "强制通过" in result["note"]
    print(f"[PASS] 超限强制通过: {result['note']}")


def test_reflect_count_increments():
    setup()
    state = _add_section("章节X", _SAMPLE_CONTENT)

    reflect_mod.runner("章节X", iteration=1)
    reflect_mod.runner("章节X", iteration=2)

    assert state["sections"]["sec_001"]["reflect_count"] == 2
    print(f"[PASS] reflect_count 自增: {state['sections']['sec_001']['reflect_count']}")


def test_parse_result_fenced():
    result = reflect_mod._try_parse_result(f"```json\n{_GOOD_JSON}\n```")
    assert result["is_sufficient"] is True
    print(f"[PASS] fenced JSON 解析正常")


def test_parse_result_unparseable_returns_none():
    # 坏 JSON → None；runner 层走保守 fallback（判不足 + 触发重新评审）
    text = 'Content is insufficient. "quality_score": 0.35, some broken json'
    assert reflect_mod._try_parse_result(text) is None
    print(f"[PASS] 坏 JSON 返回 None（由 runner 保守兜底）")


if __name__ == "__main__":
    tests = [
        test_normal_evaluation,
        test_insufficient_evaluation,
        test_no_content_returns_error,
        test_max_reflect_force_pass,
        test_reflect_count_increments,
        test_parse_result_fenced,
        test_parse_result_unparseable_returns_none,
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
    print(f"\n{passed}/{len(tests)} passed")
