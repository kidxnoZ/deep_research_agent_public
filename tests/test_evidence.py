"""
测试 extract_evidence.runner：
- 正常 claim 提取和落盘
- 多次调用累积 claims
- source_id 不存在时报错
- JSON 解析两阶段（fenced / bare）
- _parse_claims 直接单元测试
"""

import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.extract_evidence as ev_mod
from tools.registry import get_state, reset_state
from tools._client import set_client

# ─── mock config + client ─────────────────────────────────────────────────────

class _FakeCfg:
    TAVILY_API_KEY = "fake"
    CRITIQUE_MODEL = "fake-model"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_test_output", "reports")

class _FakeResponse:
    def __init__(self, text):
        self.content = [type("T", (), {"type": "text", "text": text})()]

class _FakeClient:
    def __init__(self, response_text):
        self._text = response_text
    def messages(self):
        pass

_cfg = _FakeCfg()
set_client(None, _cfg)

_EVIDENCE_DIR = os.path.join(os.path.dirname(_cfg.OUTPUT_DIR), "evidence")
_SEARCH_DIR   = os.path.join(os.path.dirname(_cfg.OUTPUT_DIR), "search_results")

_SAMPLE_CLAIMS_JSON = '{"claims": [{"claim": "2020年剧本杀市场规模约100亿", "detail": "同比增长30%", "source_ids": ["src_aaa"]}, {"claim": "头部品牌包括xxx", "detail": "占市场30%份额", "source_ids": ["src_aaa", "src_bbb"]}]}'


def _setup_search_file(src_id: str, query: str = "test query",
                       section: str = "测试章节") -> str:
    os.makedirs(_SEARCH_DIR, exist_ok=True)
    path = os.path.join(_SEARCH_DIR, f"{src_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "source_id": src_id,
            "query": query,
            "section_title": section,
            "result_count": 2,
            "results": [
                {"title": "标题1", "url": "https://a.com", "snippet": "内容1"},
                {"title": "标题2", "url": "https://b.com", "snippet": "内容2"},
            ]
        }, f, ensure_ascii=False)
    return path


def setup(llm_response: str = _SAMPLE_CLAIMS_JSON):
    reset_state()
    if os.path.exists(os.path.dirname(_cfg.OUTPUT_DIR)):
        shutil.rmtree(os.path.dirname(_cfg.OUTPUT_DIR))

    fake_client = type("C", (), {
        "messages": type("M", (), {
            "create": staticmethod(lambda **kw: _FakeResponse(llm_response))
        })()
    })()
    set_client(fake_client, _cfg)


def teardown():
    out_root = os.path.join(os.path.dirname(__file__), "_test_output")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)


# ─── 测试 ─────────────────────────────────────────────────────────────────────

def test_claims_extracted_and_stored():
    setup()
    state = get_state()
    _setup_search_file("src_aaa")
    state["search_sources"]["src_aaa"] = {
        "file": os.path.join(_SEARCH_DIR, "src_aaa.json"),
        "query": "test", "section_title": "测试章节", "summary": "x", "result_count": 2
    }

    result = ev_mod.runner("测试章节", ["src_aaa"])

    assert "error" not in result
    assert result["new_claims"] == 2
    assert result["total_claims"] == 2
    assert len(result["top_claims"]) <= 3
    assert len(state["evidence"]["测试章节"]) == 2
    print(f"[PASS] claims 提取: new={result['new_claims']}, top={result['top_claims']}")


def test_evidence_written_to_disk():
    setup()
    state = get_state()
    _setup_search_file("src_aaa")
    state["search_sources"]["src_aaa"] = {
        "file": os.path.join(_SEARCH_DIR, "src_aaa.json"),
        "query": "test", "section_title": "测试章节", "summary": "x", "result_count": 2
    }

    ev_mod.runner("测试章节", ["src_aaa"])

    evidence_files = os.listdir(_EVIDENCE_DIR)
    assert len(evidence_files) == 1
    with open(os.path.join(_EVIDENCE_DIR, evidence_files[0]), encoding="utf-8") as f:
        data = json.load(f)
    assert data["section_title"] == "测试章节"
    assert len(data["claims"]) == 2
    print(f"[PASS] evidence 落盘: {evidence_files[0]}, claims={len(data['claims'])}")


def test_multiple_calls_accumulate():
    setup()
    state = get_state()
    for sid in ["src_aaa", "src_bbb"]:
        _setup_search_file(sid)
        state["search_sources"][sid] = {
            "file": os.path.join(_SEARCH_DIR, f"{sid}.json"),
            "query": "test", "section_title": "测试章节", "summary": "x", "result_count": 2
        }

    ev_mod.runner("测试章节", ["src_aaa"])
    ev_mod.runner("测试章节", ["src_bbb"])

    total = len(state["evidence"]["测试章节"])
    assert total == 4, f"两次调用应累积 4 条，实际: {total}"
    print(f"[PASS] 多次调用累积: total_claims={total}")


def test_missing_source_id_returns_error():
    setup()
    result = ev_mod.runner("测试章节", ["src_nonexistent"])
    assert "error" in result
    print(f"[PASS] 不存在的 source_id 报错: {result['error']}")


def test_parse_claims_fenced():
    text = f"```json\n{_SAMPLE_CLAIMS_JSON}\n```"
    claims = ev_mod._parse_claims(text)
    assert len(claims) == 2
    assert claims[0]["claim"] == "2020年剧本杀市场规模约100亿"
    print(f"[PASS] fenced JSON 解析: {len(claims)} claims")


def test_parse_claims_bare():
    claims = ev_mod._parse_claims(_SAMPLE_CLAIMS_JSON)
    assert len(claims) == 2
    print(f"[PASS] 裸 JSON 解析: {len(claims)} claims")


def test_parse_claims_invalid_returns_empty():
    claims = ev_mod._parse_claims("这不是 JSON，模型输出了乱七八糟的东西")
    assert claims == []
    print(f"[PASS] 无效 JSON → 空列表")


if __name__ == "__main__":
    tests = [
        test_claims_extracted_and_stored,
        test_evidence_written_to_disk,
        test_multiple_calls_accumulate,
        test_missing_source_id_returns_error,
        test_parse_claims_fenced,
        test_parse_claims_bare,
        test_parse_claims_invalid_returns_empty,
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
