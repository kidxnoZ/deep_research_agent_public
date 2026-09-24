"""
test_fix_paths.py
-----------------
测试 save_section 的 fix 模式两个触发场景（对应"问题二"的修复验证）：

场景 1：定向补搜达上限
    source_ids 满 MAX_TARGETED_PER_SECTION 后，search 返回"达上限"错误；
    此时 save_section 在 pending 为空但有 critique_feedback 时，应走 fix 模式，
    而不是报"无新搜索结果（pending 为空）"。

场景 2：needs_search=false 且 is_sufficient=false（needs_fix 路径）
    critique 返回 can_fix_directly（无需搜索，直接修复）；
    save_section 自动读 review 的 gaps，无 pending 也应走 fix 模式。

不改源代码，只通过 dispatch() 调用工具，mock 掉 LLM 外部依赖。

运行：
    cd searchAgent
    python -m pytest tests/test_fix_paths.py -v -s
    或
    python tests/test_fix_paths.py
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from tools.registry import dispatch, reset_state, get_state, get_section_by_title
from tools._client import set_client
from tools.errors import set_error_log_path


def _sec(title: str) -> dict:
    """按标题取章节 entry。"""
    _, entry = get_section_by_title(title)
    return entry


def _llm_ok(text: str) -> dict:
    """构造 call_llm_with_retry 的成功返回值（含一个 text block）。"""
    block = MagicMock(); block.type = "text"; block.text = text
    resp = MagicMock(); resp.content = [block]; resp.stop_reason = "end_turn"
    return {"ok": True, "data": resp}


# critique 返回"无需搜索，直接修复"
CRITIQUE_NO_SEARCH = json.dumps({
    "is_sufficient": False,
    "needs_search": False,
    "quality_score": 0.7,
    "gaps": ["缺少 2025 年分红数据"],
    "improvement_suggestions": ["补充各公司 2025 年实际分红金额"],
    "stop_reason": "can_fix_directly",
}, ensure_ascii=False)


def _setup() -> str:
    """重置 state、注入 fake client/config、隔离输出目录。"""
    reset_state()
    tmpdir = tempfile.mkdtemp(prefix="fix_paths_")
    get_state()["run_dir"] = tmpdir
    set_error_log_path(os.path.join(tmpdir, "errors.jsonl"))
    set_client(MagicMock(), cfg)
    return tmpdir


def _plan(title: str = "测试章节") -> dict:
    r = dispatch("plan_sections", {
        "query": "测试研究问题",
        "sections": [{"title": title, "description": "d", "initial_query": "q",
                      "draft_content": "训练知识初稿"}],
    })
    assert r["confirmed"], f"plan_sections failed: {r}"
    return _sec(title)


def test_targeted_search_limit_then_fix():
    """场景 1：补搜达上限 → search 拦截 → save 走 fix 模式（不报 pending 为空）。"""
    _setup()
    sec = _plan()
    # 模拟已 save 过一次：final 非空、source_ids 满上限、pending 空
    sec["final"] = "已有章节内容（第一次 save 的结果）"
    sec["status"] = "needs_search"   # 非 pending_save，让 search 走到达上限检查
    sec["source_ids"] = [f"src_{i}" for i in range(cfg.MAX_TARGETED_PER_SECTION)]

    # search 应返回"达上限"错误
    r = dispatch("search", {"query": "补充搜索", "section_title": "测试章节"})
    assert "error" in r, f"search 应报错，实际 {r}"
    assert "上限" in r["error"], f"应提示达上限，实际 {r['error']}"

    # 模拟 critique 判不足（补搜耗尽场景）
    sec["review"] = {
        "is_sufficient": False, "needs_search": True, "quality_score": 0.5,
        "gaps": ["缺少某公司数据"], "improvement_suggestions": ["补充该数据"],
    }

    # save_section 不传 critique_feedback，自动读 review → 走 fix 模式
    with patch("tools.report.call_llm_with_retry", return_value=_llm_ok("修订后的章节内容")):
        r = dispatch("save_section", {"title": "测试章节", "order": 1})

    assert "error" not in r, f"save 不应报 pending 为空，实际 {r}"
    assert r.get("mode") == "fix", f"应走 fix 模式，实际 mode={r.get('mode')}"
    assert sec["final"] == "修订后的章节内容"
    assert sec["pending_src_ids"] == []
    assert sec["status"] == "saved"
    print(f"[PASS] 达上限 → fix：mode={r['mode']}，final 已更新")


def test_needs_fix_path_then_save_fix():
    """场景 2：critique 返回 needs_search=false + is_sufficient=false → save 走 fix。"""
    _setup()
    sec = _plan()
    sec["final"] = "已有章节内容"
    sec["status"] = "saved"

    # critique 返回 can_fix_directly
    with patch("tools.reflect.call_llm_with_retry", return_value=_llm_ok(CRITIQUE_NO_SEARCH)):
        r = dispatch("critique_section", {"section_title": "测试章节", "iteration": 1})

    assert r["is_sufficient"] is False
    assert r["needs_search"] is False
    assert sec["status"] == "needs_fix", f"应置为 needs_fix，实际 {sec['status']}"

    # save_section 自动读 review（needs_fix、无 pending）→ 走 fix 模式
    with patch("tools.report.call_llm_with_retry", return_value=_llm_ok("修复后的章节内容")):
        r = dispatch("save_section", {"title": "测试章节", "order": 1})

    assert "error" not in r, f"save 不应报错，实际 {r}"
    assert r.get("mode") == "fix", f"应走 fix 模式，实际 mode={r.get('mode')}"
    assert sec["final"] == "修复后的章节内容"
    assert sec["status"] == "saved"
    print(f"[PASS] needs_fix → fix：mode={r['mode']}，final 已更新")


if __name__ == "__main__":
    tests = [test_targeted_search_limit_then_fix, test_needs_fix_path_then_save_fix]
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
