"""
测试 registry.dispatch 的参数校验逻辑：
- 缺少必填字段
- 字段类型不符合 schema
- 正常调用
- 未知工具
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import register, dispatch, _validate_types

# ─── 构造一个 dummy 工具用于测试 ─────────────────────────────────────────────

_DUMMY_SCHEMA = {
    "name": "dummy_tool",
    "description": "测试用工具",
    "input_schema": {
        "type": "object",
        "properties": {
            "name":    {"type": "string",  "description": "名称"},
            "count":   {"type": "integer", "description": "数量"},
            "ratio":   {"type": "number",  "description": "比例"},
            "enabled": {"type": "boolean", "description": "开关"},
            "tags":    {"type": "array",   "description": "标签列表"},
            "meta":    {"type": "object",  "description": "元数据"},
        },
        "required": ["name", "count"]
    }
}

def _dummy_runner(name: str, count: int, ratio=None, enabled=None, tags=None, meta=None) -> dict:
    return {"ok": True, "name": name, "count": count}

register(_DUMMY_SCHEMA, _dummy_runner)

# ─── 测试 ─────────────────────────────────────────────────────────────────────

def test_missing_required():
    result = dispatch("dummy_tool", {"name": "hello"})  # 缺 count
    assert "error" in result, f"期望 error，得到: {result}"
    assert "count" in result["error"]
    print(f"[PASS] 缺少必填字段: {result['error']}")


def test_type_mismatch_string():
    result = dispatch("dummy_tool", {"name": 123, "count": 5})  # name 应为 string
    assert "error" in result, f"期望 error，得到: {result}"
    assert "name" in result["error"]
    print(f"[PASS] 字段类型错误(string): {result['error']}")


def test_type_mismatch_integer():
    result = dispatch("dummy_tool", {"name": "ok", "count": "five"})  # count 应为 integer
    assert "error" in result, f"期望 error，得到: {result}"
    assert "count" in result["error"]
    print(f"[PASS] 字段类型错误(integer): {result['error']}")


def test_type_mismatch_array():
    result = dispatch("dummy_tool", {"name": "ok", "count": 1, "tags": "not-a-list"})
    assert "error" in result, f"期望 error，得到: {result}"
    assert "tags" in result["error"]
    print(f"[PASS] 字段类型错误(array): {result['error']}")


def test_number_accepts_int():
    # number 类型应同时接受 int 和 float
    result = dispatch("dummy_tool", {"name": "ok", "count": 1, "ratio": 1})
    assert "error" not in result, f"number 应接受 int，得到: {result}"
    result2 = dispatch("dummy_tool", {"name": "ok", "count": 1, "ratio": 0.5})
    assert "error" not in result2, f"number 应接受 float，得到: {result2}"
    print("[PASS] number 类型同时接受 int 和 float")


def test_valid_call():
    result = dispatch("dummy_tool", {"name": "hello", "count": 3})
    assert result == {"ok": True, "name": "hello", "count": 3}, f"期望正常返回，得到: {result}"
    print(f"[PASS] 正常调用: {result}")


def test_unknown_tool():
    result = dispatch("nonexistent_tool", {})
    assert "error" in result
    print(f"[PASS] 未知工具: {result['error']}")


if __name__ == "__main__":
    tests = [
        test_missing_required,
        test_type_mismatch_string,
        test_type_mismatch_integer,
        test_type_mismatch_array,
        test_number_accepts_int,
        test_valid_call,
        test_unknown_tool,
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
