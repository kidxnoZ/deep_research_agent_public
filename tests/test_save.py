"""
测试 save_section.runner：
- 正常从搜索文件生成内容并保存
- 无搜索结果时报错
- 章节不在规划列表时报错
- 重复保存覆盖旧内容
- 落盘 checkpoint
- tool_result 包含 preview 而非完整 content
"""

import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.registry import get_state, reset_state, _make_section_entry, get_section_by_title
from tools._client import set_client
import tools.report as report_mod


class _FakeCfg:
    CRITIQUE_MODEL = "fake"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_test_output", "reports")

_GENERATED_CONTENT = "这是由子 LLM 根据搜索结果生成的章节正文，包含了具体数据和分析内容。" * 5


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

def _make_client(response_text=_GENERATED_CONTENT):
    resp = _FakeResponse(response_text)
    return type("C", (), {
        "messages": type("M", (), {
            "stream": staticmethod(lambda **kw: _FakeStream(resp)),
            "create": staticmethod(lambda **kw: resp),
        })()
    })()

_cfg = _FakeCfg()
_SEARCH_DIR = os.path.join(os.path.dirname(_cfg.OUTPUT_DIR), "search_results")


def _setup_search_file(title: str, src_id: str = "src_aaa") -> str:
    os.makedirs(_SEARCH_DIR, exist_ok=True)
    path = os.path.join(_SEARCH_DIR, f"{src_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "source_id": src_id,
            "query": "测试查询",
            "section_title": title,
            "result_count": 2,
            "results": [
                {"title": "标题1", "url": "https://a.com", "snippet": "内容A"},
                {"title": "标题2", "url": "https://b.com", "snippet": "内容B"},
            ]
        }, f, ensure_ascii=False)
    return path


def setup():
    reset_state()
    set_client(_make_client(), _cfg)
    if os.path.exists(os.path.dirname(_cfg.OUTPUT_DIR)):
        shutil.rmtree(os.path.dirname(_cfg.OUTPUT_DIR))


def _add_section(title: str):
    """在 state 中创建章节 entry（sec_001），返回 state。"""
    state = get_state()
    state["sections"]["sec_001"] = _make_section_entry(
        description="", initial_query="", order=1, title=title)
    return state


def _add_search_source(title: str, src_id: str = "src_aaa"):
    """写入搜索文件 + search_sources 索引 + 章节 pending_src_ids。"""
    path = _setup_search_file(title, src_id)
    state = get_state()
    state["search_sources"][src_id] = {
        "file": path, "query": "测试查询",
        "section_title": title, "summary": "测试摘要", "result_count": 2
    }
    _, sec = get_section_by_title(title)
    if sec is not None:
        sec["pending_src_ids"].append(src_id)


def teardown():
    out_root = os.path.join(os.path.dirname(__file__), "_test_output")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)


def test_saves_from_search_results():
    setup()
    state = _add_section("市场概况")
    _add_search_source("市场概况")

    result = report_mod.save_runner("市场概况", order=1)

    assert "error" not in result
    assert result["saved"] is True
    assert len(state["sections"]) == 1
    assert state["sections"]["sec_001"]["final"] == _GENERATED_CONTENT
    print(f"[PASS] 从搜索结果生成并保存: sections={result['sections_completed']}")


def test_tool_result_has_preview_not_content():
    setup()
    _add_section("市场概况")
    _add_search_source("市场概况")

    result = report_mod.save_runner("市场概况", order=1)

    assert "preview" in result, "tool_result 应包含 preview"
    assert "content" not in result, "tool_result 不应暴露完整 content"
    assert len(result["preview"]) <= 150
    print(f"[PASS] tool_result 只含 preview（{len(result['preview'])} 字符），无完整 content")


def test_no_search_results_returns_error():
    setup()
    _add_section("空章节")

    result = report_mod.save_runner("空章节", order=1)
    assert "error" in result
    assert "search" in result["error"]
    print(f"[PASS] 无搜索结果报错: {result['error']}")


def test_title_not_in_plan_returns_error():
    setup()
    _add_section("已规划章节")
    _add_search_source("不存在的章节")

    result = report_mod.save_runner("不存在的章节", order=1)
    assert "error" in result
    print(f"[PASS] 不在规划列表报错: {result['error'][:60]}")


def test_overwrite_existing():
    setup()
    state = _add_section("市场概况")
    _add_search_source("市场概况")

    report_mod.save_runner("市场概况", order=1)
    report_mod.save_runner("市场概况", order=1)

    assert len(state["sections"]) == 1, "重复保存不应新增条目"
    print(f"[PASS] 重复保存覆盖而非追加")


def test_checkpoint_written():
    setup()
    _add_section("市场概况")
    _add_search_source("市场概况")

    report_mod.save_runner("市场概况", order=1)

    checkpoint = os.path.join(_cfg.OUTPUT_DIR, "sections_checkpoint.json")
    assert os.path.exists(checkpoint), "sections_checkpoint.json 未写盘"
    with open(checkpoint, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["sections"]) == 1
    print(f"[PASS] checkpoint 写盘: {checkpoint}")


if __name__ == "__main__":
    tests = [
        test_saves_from_search_results,
        test_tool_result_has_preview_not_content,
        test_no_search_results_returns_error,
        test_title_not_in_plan_returns_error,
        test_overwrite_existing,
        test_checkpoint_written,
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
