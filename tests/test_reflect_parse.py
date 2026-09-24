"""
测试 reflect._try_parse_result 的 JSON 解析：
- 正常 JSON 输出
- 代码块包裹的 JSON
- 花括号提取
- 无法解析 → None（runner 层走保守 fallback：判不足并触发重新评审）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.reflect import _try_parse_result


def test_clean_json():
    text = '{"is_sufficient": true, "quality_score": 0.9, "gaps": [], "improvement_suggestions": [], "stop_reason": "sufficient"}'
    result = _try_parse_result(text)
    assert result["is_sufficient"] is True
    assert result["quality_score"] == 0.9


def test_fenced_json():
    text = '```json\n{"is_sufficient": false, "quality_score": 0.4, "gaps": ["缺少数据"], "improvement_suggestions": ["搜索年度报告"], "stop_reason": "needs_more_search"}\n```'
    result = _try_parse_result(text)
    assert result["is_sufficient"] is False
    assert result["quality_score"] == 0.4


def test_brace_extraction():
    text = '前缀文字 {"is_sufficient": true, "quality_score": 0.75} 后缀文字'
    result = _try_parse_result(text)
    assert result is not None
    assert result["quality_score"] == 0.75


def test_unparseable_returns_none():
    # 坏 JSON → None；runner 层会走保守 fallback（0.3 + 重新评审）
    text = 'The content is insufficient. "quality_score": 0.35, some extra text that breaks JSON'
    assert _try_parse_result(text) is None


if __name__ == "__main__":
    tests = [
        test_clean_json,
        test_fenced_json,
        test_brace_extraction,
        test_unparseable_returns_none,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
