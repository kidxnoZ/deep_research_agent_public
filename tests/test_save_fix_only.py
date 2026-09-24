"""
测试 save_runner is_fix_only 路径（v2.7.6）：
- patch 模式 pending 为空 + critique_feedback 非空 → 成功，mode="fix"
- patch 模式 pending 为空 + critique_feedback 为空 → 报错（原有保护保留）
- patch 模式 pending 非空 + critique_feedback 非空 → 正常 patch 模式
- full 模式 pending 为空 + 无初稿 → 报错（原有行为不变）
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import get_state, reset_state, _make_section_entry
from tools._client import set_client
import tools.report as report_mod

_GENERATED = "修订后的章节正文，已根据审核反馈修复了排序问题，内容更加准确。" * 3
_ORIGINAL  = "原始章节内容，角色发布顺序有误，需要修正。" * 3

_TEST_DIR = os.path.join(os.path.dirname(__file__), "_test_output_fix")


class _FakeCfg:
    CRITIQUE_MODEL = "fake"
    OUTPUT_DIR     = os.path.join(_TEST_DIR, "reports")


# ── fake streaming client ─────────────────────────────────────────────────────

class _FakeMsg:
    def __init__(self, text):
        self.stop_reason = "end_turn"
        self.content = [type("T", (), {"type": "text", "text": text})()]

class _FakeStream:
    def __init__(self, text):
        self._text = text
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def get_final_message(self):
        return _FakeMsg(self._text)

def _make_client(text=_GENERATED):
    class M:
        def stream(self_, **kw):
            return _FakeStream(text)
    class C:
        messages = M()
    return C()


_cfg = _FakeCfg()


def _setup(with_final=False):
    reset_state()
    set_client(_make_client(), _cfg)
    state = get_state()
    state["run_dir"] = _TEST_DIR
    sec = _make_section_entry(title="测试章节", order=1)
    sec["draft"] = "初稿内容"
    if with_final:
        sec["final"]  = _ORIGINAL
        sec["status"] = "saved"
    state["sections"]["sec_001"] = sec
    return state


def _add_search(state, src_id="src_001", section_title="测试章节"):
    path = os.path.join(_TEST_DIR, f"{src_id}.json")
    os.makedirs(_TEST_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "query": "测试", "source_id": src_id,
            "results": [{"title": "T", "url": "u", "snippet": "内容片段"}]
        }, f)
    state["search_sources"][src_id] = {
        "file": path, "query": "测试",
        "section_title": section_title, "summary": "摘要", "result_count": 1,
    }
    state["sections"]["sec_001"]["pending_src_ids"].append(src_id)


def teardown():
    if os.path.exists(_TEST_DIR):
        shutil.rmtree(_TEST_DIR)


# ── 测试用例 ──────────────────────────────────────────────────────────────────

def test_fix_only_succeeds():
    """patch 模式 + pending 为空 + 有 critique_feedback → mode=fix，保存修订内容"""
    state = _setup(with_final=True)

    result = report_mod.save_runner("测试章节", order=1,
                                    critique_feedback="角色发布顺序排列有误，请按官方时间重新排序")

    assert "error" not in result, f"不应报错: {result.get('error')}"
    assert result["saved"] is True
    assert result["mode"] == "fix"
    sec = state["sections"]["sec_001"]
    assert sec["final"] == _GENERATED
    assert sec["pending_src_ids"] == []
    print(f"[PASS] fix_only 成功: mode={result['mode']} preview={result['preview'][:40]}…")


def test_fix_only_no_feedback_returns_error():
    """patch 模式 + pending 为空 + 无 critique_feedback → 报错（原有保护保留）"""
    _setup(with_final=True)

    result = report_mod.save_runner("测试章节", order=1)

    assert "error" in result
    assert "无需 save" in result["error"]
    print(f"[PASS] 无 feedback 仍报错: {result['error']}")


def test_patch_mode_with_search_and_feedback():
    """patch 模式 + pending 非空 + 有 feedback → 正常 patch，mode=patch"""
    state = _setup(with_final=True)
    _add_search(state, "src_002")

    result = report_mod.save_runner("测试章节", order=1,
                                    critique_feedback="补充 2024 年最新数据")

    assert "error" not in result, f"不应报错: {result.get('error')}"
    assert result["mode"] == "patch"
    assert state["sections"]["sec_001"]["pending_src_ids"] == []
    assert "src_002" in state["sections"]["sec_001"]["source_ids"]
    print(f"[PASS] 正常 patch 模式: mode={result['mode']}")


def test_full_mode_no_search_no_draft_returns_error():
    """full 模式（首次 save）+ pending 为空 + 无初稿 → 报错（原有行为不变）"""
    state = _setup(with_final=False)
    state["sections"]["sec_001"]["draft"] = ""

    result = report_mod.save_runner("测试章节", order=1)

    assert "error" in result
    assert "search" in result["error"]
    print(f"[PASS] full 模式无搜索无初稿仍报错: {result['error']}")


def test_full_mode_no_search_with_draft_saves_draft():
    """full 模式（首次 save）+ pending 为空 + 有初稿 → 保存初稿（原有行为不变）"""
    _setup(with_final=False)

    result = report_mod.save_runner("测试章节", order=1)

    assert "error" not in result, f"不应报错: {result.get('error')}"
    assert result["saved"] is True
    assert "初稿" in result.get("note", "")
    print(f"[PASS] full 模式无搜索有初稿保存初稿: note={result.get('note')}")


if __name__ == "__main__":
    tests = [
        test_fix_only_succeeds,
        test_fix_only_no_feedback_returns_error,
        test_patch_mode_with_search_and_feedback,
        test_full_mode_no_search_no_draft_returns_error,
        test_full_mode_no_search_with_draft_saves_draft,
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
    teardown()
    print(f"\n{passed}/{len(tests)} passed")
