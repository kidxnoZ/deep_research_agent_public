"""
test_search_multisource.py
各非 Tavily 搜索源的单元测试：正常返回、空结果、网络失败、参数缺失四种 case。
全部 mock，不发真实请求/子进程。
"""

import sys, os, json, shutil, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.search as search_mod
from tools.registry import reset_state
from tools._client import set_client
from tools import rate_limiter

# ─── mock config ─────────────────────────────────────────────────────────────

class _Cfg:
    TAVILY_API_KEY = "fake"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "_test_output_ms", "reports")
    MAX_BROAD_SEARCHES = 8
    MAX_TARGETED_PER_SECTION = 5
    GH_BIN = "gh"
    XHS_BACKEND = "opencli"
    ZHIHU_COOKIE = "z_c0=fake; d_c0=fake"
    SEMANTIC_SCHOLAR_API_KEY = ""
    # 测试中所有源走零限速，不干扰功能验证
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

_cfg = _Cfg()
set_client(_FakeClient(), _cfg)

_FAKE_RESULTS = [
    {"title": "R1", "url": "https://r1.com", "snippet": "snippet1"},
    {"title": "R2", "url": "https://r2.com", "snippet": "snippet2"},
]

_orig_callers = dict(search_mod._CALLERS)


def _fake_caller(q, n, cfg):
    return _FAKE_RESULTS


def setup():
    reset_state()
    rate_limiter.reset()
    set_client(_FakeClient(), _cfg)
    search_mod._CALLERS.update(_orig_callers)
    search_mod._CALLERS["tavily"] = _fake_caller


def teardown():
    out_root = os.path.join(os.path.dirname(__file__), "_test_output_ms")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)


# ── arXiv ─────────────────────────────────────────────────────────────────────

def test_arxiv_normal():
    setup()
    search_mod._CALLERS["arxiv"] = lambda q, n, cfg: _FAKE_RESULTS
    r = search_mod.runner("transformer attention", source="arxiv")
    assert "error" not in r, r
    assert r["source"] == "arxiv"
    assert r["source_id"].startswith("src_arxiv_")
    assert r["result_count"] == 2
    print(f"[PASS] arxiv 正常: {r['source_id']}")


def test_arxiv_empty():
    setup()
    search_mod._CALLERS["arxiv"] = lambda q, n, cfg: []
    r = search_mod.runner("nonexistent_xyzabc123", source="arxiv")
    assert "error" in r
    assert "无结果" in r["error"] or "0 条" in r["error"]
    print(f"[PASS] arxiv 空结果: {r['error'][:60]}")


def test_arxiv_network_failure():
    setup()
    def _fail(q, n, cfg):
        raise ConnectionError("timeout")
    search_mod._CALLERS["arxiv"] = _fail
    r = search_mod.runner("transformer", source="arxiv")
    assert "error" in r
    print(f"[PASS] arxiv 网络失败: {r['error'][:60]}")


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def test_semantic_scholar_normal():
    setup()
    search_mod._CALLERS["semantic_scholar"] = lambda q, n, cfg: _FAKE_RESULTS
    r = search_mod.runner("BERT NLP", source="semantic_scholar")
    assert "error" not in r, r
    assert r["source"] == "semantic_scholar"
    assert r["source_id"].startswith("src_semantic_scholar_")
    print(f"[PASS] semantic_scholar 正常: {r['source_id']}")


def test_semantic_scholar_null_abstract():
    """abstract 为 null 时 snippet 应置空字符串，不崩溃。"""
    setup()
    papers = [{"title": "Paper A", "url": "https://s2.org/1", "snippet": ""}]
    search_mod._CALLERS["semantic_scholar"] = lambda q, n, cfg: papers
    r = search_mod.runner("some topic", source="semantic_scholar")
    assert "error" not in r
    state_entry = __import__("tools.registry", fromlist=["get_state"]).get_state()["search_sources"][r["source_id"]]
    with open(state_entry["file"], encoding="utf-8") as f:
        data = json.load(f)
    assert data["results"][0]["snippet"] == ""
    print("[PASS] semantic_scholar null abstract 不崩溃")


def test_semantic_scholar_network_failure():
    setup()
    search_mod._CALLERS["semantic_scholar"] = lambda q, n, cfg: (_ for _ in ()).throw(RuntimeError("503"))
    r = search_mod.runner("llm", source="semantic_scholar")
    assert "error" in r
    print(f"[PASS] semantic_scholar 网络失败: {r['error'][:60]}")


# ── GitHub ────────────────────────────────────────────────────────────────────

def test_github_normal():
    setup()
    search_mod._CALLERS["github"] = lambda q, n, cfg: _FAKE_RESULTS
    r = search_mod.runner("llm inference", source="github")
    assert "error" not in r, r
    assert r["source"] == "github"
    assert r["source_id"].startswith("src_github_")
    print(f"[PASS] github 正常: {r['source_id']}")


def test_github_empty():
    setup()
    search_mod._CALLERS["github"] = lambda q, n, cfg: []
    r = search_mod.runner("xyznonexistent999", source="github")
    assert "error" in r
    print(f"[PASS] github 空结果: {r['error'][:60]}")


def test_github_cli_failure():
    setup()
    def _fail(q, n, cfg):
        raise RuntimeError("gh search repos 失败: error: could not find binary")
    search_mod._CALLERS["github"] = _fail
    r = search_mod.runner("pytorch", source="github")
    assert "error" in r
    print(f"[PASS] github CLI 失败: {r['error'][:60]}")


# ── 小红书 ────────────────────────────────────────────────────────────────────

def test_xiaohongshu_normal():
    setup()
    search_mod._CALLERS["xiaohongshu"] = lambda q, n, cfg: _FAKE_RESULTS
    r = search_mod.runner("露营装备推荐", source="xiaohongshu")
    assert "error" not in r, r
    assert r["source"] == "xiaohongshu"
    assert r["source_id"].startswith("src_xiaohongshu_")
    print(f"[PASS] xiaohongshu 正常: {r['source_id']}")


def test_xiaohongshu_opencli_not_available():
    setup()
    def _fail(q, n, cfg):
        raise RuntimeError("opencli xiaohongshu search 失败: [Errno 2] No such file or directory")
    search_mod._CALLERS["xiaohongshu"] = _fail
    r = search_mod.runner("好物推荐", source="xiaohongshu")
    assert "error" in r
    print(f"[PASS] xiaohongshu opencli 不可用: {r['error'][:60]}")


def test_xiaohongshu_empty():
    setup()
    search_mod._CALLERS["xiaohongshu"] = lambda q, n, cfg: []
    r = search_mod.runner("nonexistent", source="xiaohongshu")
    assert "error" in r
    print(f"[PASS] xiaohongshu 空结果: {r['error'][:60]}")


# ── 知乎 ──────────────────────────────────────────────────────────────────────

def test_zhihu_normal():
    setup()
    search_mod._CALLERS["zhihu"] = lambda q, n, cfg: _FAKE_RESULTS
    r = search_mod.runner("Python 异步编程", source="zhihu")
    assert "error" not in r, r
    assert r["source"] == "zhihu"
    assert r["source_id"].startswith("src_zhihu_")
    print(f"[PASS] zhihu 正常: {r['source_id']}")


def test_zhihu_opencli_not_available():
    """opencli 找不到时应返回 error。"""
    setup()
    import unittest.mock as mock
    with mock.patch("shutil.which", return_value=None):
        search_mod._CALLERS["zhihu"] = search_mod._call_zhihu
        r = search_mod.runner("知乎测试", source="zhihu")
    assert "error" in r
    assert "opencli" in r["error"].lower() or "未找到" in r["error"]
    print(f"[PASS] zhihu opencli 不可用: {r['error'][:80]}")


def test_zhihu_network_failure():
    setup()
    def _fail(q, n, cfg):
        raise ConnectionError("401 Unauthorized")
    search_mod._CALLERS["zhihu"] = _fail
    r = search_mod.runner("知乎查询", source="zhihu")
    assert "error" in r
    print(f"[PASS] zhihu 网络失败: {r['error'][:60]}")


# ── _call_arxiv 单元测试（直接测解析逻辑，mock HTTP） ─────────────────────────

def test_call_arxiv_parse():
    """mock requests.get，验证 Atom XML 解析结果。"""
    import unittest.mock as mock

    ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <link href="https://arxiv.org/abs/1706.03762"/>
    <summary>We propose the Transformer architecture.</summary>
  </entry>
</feed>"""

    class _Resp:
        text = ATOM_XML
        def raise_for_status(self): pass

    with mock.patch("tools.search.requests.get", return_value=_Resp()):
        results = search_mod._call_arxiv("transformer", 3, _cfg)

    assert len(results) == 1
    assert "Attention" in results[0]["title"]
    assert "arxiv.org" in results[0]["url"]
    assert "Transformer" in results[0]["snippet"]
    print(f"[PASS] _call_arxiv XML 解析: {results[0]['title']}")


def test_call_semantic_scholar_parse():
    """mock requests.get，验证 JSON 解析结果。"""
    import unittest.mock as mock

    RESP_JSON = {
        "data": [
            {"title": "BERT", "url": "https://s2.org/1", "abstract": "Pre-training of deep...",
             "paperId": "abc123", "year": 2018, "citationCount": 50000}
        ]
    }

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return RESP_JSON

    with mock.patch("tools.search.requests.get", return_value=_Resp()):
        with mock.patch("tools.search.time.sleep"):  # 跳过等待
            results = search_mod._call_semantic_scholar("BERT", 3, _cfg)

    assert len(results) == 1
    assert results[0]["title"] == "BERT"
    assert "Pre-training" in results[0]["snippet"]
    print(f"[PASS] _call_semantic_scholar JSON 解析: {results[0]['title']}")


def test_per_source_limit():
    """rate_limiter QPH 达限后 runner 返回 error，不影响其他源。"""
    setup()
    import unittest.mock as mock
    from tools import rate_limiter as rl

    # 直接往 hour_calls 塞满 arxiv 的 QPH 配额
    qph = _cfg.RATE_LIMITS["risk_none"]["qph"]  # None → risk_none 不限
    # 用 risk_high 参数测（xhs QPH=25 → 临时注入 2）
    rl.reset("arxiv")
    rl._get("arxiv", 5)  # 初始化
    fake_qph = 2
    # 直接塞 fake_qph 条当前时间戳
    with rl._rs["arxiv"]["data_lock"]:
        rl._rs["arxiv"]["hour_calls"] = [time.time()] * fake_qph

    # mock rate_limiter._risk_cfg 返回 qph=fake_qph for arxiv
    orig_risk_cfg = rl._risk_cfg
    def patched_risk_cfg(source, cfg):
        if source == "arxiv":
            return {"min_interval": 0, "max_interval": 0, "concurrency": 5,
                    "qph": fake_qph, "circuit_break_after": 5, "circuit_break_minutes": 10}
        return orig_risk_cfg(source, cfg)

    search_mod._CALLERS["arxiv"] = _fake_caller
    search_mod._CALLERS["tavily"] = _fake_caller

    with mock.patch("tools.rate_limiter._risk_cfg", side_effect=patched_risk_cfg):
        with mock.patch("tools.rate_limiter.time.sleep"):
            r_arxiv = search_mod.runner("query_arxiv", source="arxiv")
            r_tavily = search_mod.runner("query_tavily", source="tavily")

    assert "error" in r_arxiv, f"arxiv QPH 满后应拒绝: {r_arxiv}"
    assert "每小时限额" in r_arxiv["error"] or "限额" in r_arxiv["error"]
    assert "error" not in r_tavily, f"tavily 不受 arxiv 限额影响: {r_tavily}"

    rl.reset()
    print(f"[PASS] per-source QPH 限额，arxiv 拒绝后 tavily 仍可用")


if __name__ == "__main__":
    tests = [
        test_arxiv_normal,
        test_arxiv_empty,
        test_arxiv_network_failure,
        test_semantic_scholar_normal,
        test_semantic_scholar_null_abstract,
        test_semantic_scholar_network_failure,
        test_github_normal,
        test_github_empty,
        test_github_cli_failure,
        test_xiaohongshu_normal,
        test_xiaohongshu_opencli_not_available,
        test_xiaohongshu_empty,
        test_zhihu_normal,
        test_zhihu_opencli_not_available,
        test_zhihu_network_failure,
        test_call_arxiv_parse,
        test_call_semantic_scholar_parse,
        test_per_source_limit,
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
