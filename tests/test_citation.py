"""
test_citation.py — 引用溯源功能测试

覆盖：
- ref_idx 分配与 URL 去重
- reset_state 清零计数器
- search 结果写入时带 ref_idx
- save_section 正文解析 [n] 存 citations
- compile_report 生成参考文献表
- extract_evidence 停用（不再注册）
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from tools.registry import assign_ref_idx, get_ref_url_map, reset_state, get_state


# ── TestRefIdx ───────────────────────────────────────────────────────────────

class TestRefIdx:
    def setup_method(self):
        reset_state()

    def test_assign_increments(self):
        idx1 = assign_ref_idx("https://a.com/page1")
        idx2 = assign_ref_idx("https://b.com/page2")
        assert idx1 == 1
        assert idx2 == 2

    def test_same_url_returns_same_idx(self):
        url = "https://example.com/report.pdf"
        idx1 = assign_ref_idx(url)
        idx2 = assign_ref_idx(url)
        assert idx1 == idx2

    def test_empty_url_returns_zero(self):
        assert assign_ref_idx("") == 0

    def test_get_ref_url_map(self):
        assign_ref_idx("https://x.com/a")
        assign_ref_idx("https://x.com/b")
        m = get_ref_url_map()
        assert m["https://x.com/a"] == 1
        assert m["https://x.com/b"] == 2

    def test_reset_clears_counter(self):
        assign_ref_idx("https://x.com/first")
        reset_state()
        idx = assign_ref_idx("https://x.com/second")
        assert idx == 1  # 重新从 1 开始

    def test_reset_clears_map(self):
        assign_ref_idx("https://x.com/page")
        reset_state()
        assert get_ref_url_map() == {}

    def test_section_entry_has_citations_field(self):
        from tools.registry import _make_section_entry
        entry = _make_section_entry(title="测试章节")
        assert "citations" in entry
        assert entry["citations"] == {}


# ── TestSearchRefIdx ──────────────────────────────────────────────────────────

class TestSearchRefIdx:
    """验证 search runner 写入磁盘文件时包含 ref_idx 字段。"""

    def setup_method(self):
        reset_state()

    def _make_search_result_file(self):
        """模拟一次 search 调用，返回写入的结果文件路径。"""
        from tools.registry import get_state
        import tools.search as search_mod

        results = [
            {"title": "报告A", "url": "https://example.com/a", "snippet": "内容A"},
            {"title": "报告B", "url": "https://sec.gov/b.pdf", "snippet": "内容B"},
        ]

        mock_cfg = MagicMock()
        mock_cfg.TAVILY_API_KEY = "fake"
        mock_cfg.MAX_BROAD_SEARCHES = 10
        mock_cfg.MAX_TARGETED_PER_SECTION = 5
        mock_cfg.AUTHORITY_DOMAINS = []

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cfg.OUTPUT_DIR = os.path.join(tmpdir, "output")
            state = get_state()
            state["run_dir"] = tmpdir

            with (
                patch.object(search_mod, "_call_tavily", return_value=results),
                patch("tools.search.get_config", return_value=mock_cfg),
                patch("tools.search.rate_limiter.acquire", return_value=(True, None)),
                patch("tools.search.rate_limiter.release"),
                patch("tools.search.call_with_retry",
                      side_effect=lambda fn, q, mr, cfg, **kw: {"ok": True, "data": results}),
            ):
                ret = search_mod.runner(query="test query", source="tavily")

            src_id = ret.get("source_id")
            assert src_id is not None

            file_path = state["search_sources"][src_id]["file"]
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data["results"]

    def test_results_have_ref_idx(self):
        results = self._make_search_result_file()
        for r in results:
            assert "ref_idx" in r
            assert isinstance(r["ref_idx"], int)
            assert r["ref_idx"] > 0

    def test_ref_idx_unique_per_url(self):
        results = self._make_search_result_file()
        urls = [r["url"] for r in results]
        idxs = [r["ref_idx"] for r in results]
        assert len(set(urls)) == len(set(idxs))


# ── TestSaveCitations ─────────────────────────────────────────────────────────

class TestSaveCitations:
    """验证 save_section 解析 [n] 并写入 sec['citations']。"""

    def setup_method(self):
        reset_state()

    def _setup_section_with_search(self, content_with_refs: str):
        """建一个带 pending_src_id 的章节，模拟 save 调用后返回含 [n] 的正文。"""
        from tools.registry import get_state, _make_section_entry
        from tools.registry import assign_ref_idx

        state = get_state()

        # 手动分配两个 ref_idx
        url_a = "https://alpha.com/report"
        url_b = "https://beta.com/data"
        idx_a = assign_ref_idx(url_a)
        idx_b = assign_ref_idx(url_b)

        # 构造 section
        sec_key = "sec_001"
        state["sections"][sec_key] = _make_section_entry(title="市场分析", order=1)
        sec = state["sections"][sec_key]

        # 构造 search_sources + 磁盘文件
        with tempfile.TemporaryDirectory() as tmpdir:
            state["run_dir"] = tmpdir
            sr_dir = os.path.join(tmpdir, "search_results")
            os.makedirs(sr_dir)
            file_path = os.path.join(sr_dir, "src_tavily_abc.json")
            payload = {
                "query": "市场分析",
                "source": "tavily",
                "results": [
                    {"title": "Alpha报告", "url": url_a, "snippet": "...", "ref_idx": idx_a},
                    {"title": "Beta数据", "url": url_b, "snippet": "...", "ref_idx": idx_b},
                ],
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            src_id = "src_tavily_abc"
            state["search_sources"][src_id] = {
                "file": file_path, "query": "市场分析",
                "source": "tavily", "section_title": "市场分析",
                "summary": "", "result_count": 2,
            }
            sec["pending_src_ids"] = [src_id]
            sec["draft"] = "初稿内容"

            mock_cfg = MagicMock()
            mock_cfg.OUTPUT_DIR = os.path.join(tmpdir, "output")
            mock_cfg.CRITIQUE_MODEL = "test-model"

            mock_llm_result = {"ok": True, "data": MagicMock()}

            import tools.report as report_mod
            with (
                patch("tools.report.get_config", return_value=mock_cfg),
                patch("tools.report.get_client"),
                patch("tools.report.call_llm_with_retry", return_value=mock_llm_result),
                patch("tools.report._extract_text", return_value=content_with_refs),
            ):
                ret = report_mod.save_runner(title="市场分析", order=1)

            return ret, sec

    def test_citations_populated_from_content(self):
        content = "市场规模达 500 亿 [1]，同比增长 30% [2]。"
        ret, sec = self._setup_section_with_search(content)
        assert ret.get("saved") is True
        assert 1 in sec["citations"]
        assert 2 in sec["citations"]

    def test_citations_maps_to_correct_url(self):
        content = "数据来源 [1]，另见 [2]。"
        ret, sec = self._setup_section_with_search(content)
        assert sec["citations"][1] == "https://alpha.com/report"
        assert sec["citations"][2] == "https://beta.com/data"

    def test_no_citations_if_no_brackets(self):
        content = "没有任何引用标注的内容。"
        ret, sec = self._setup_section_with_search(content)
        assert sec["citations"] == {}

    def test_unknown_ref_idx_not_in_citations(self):
        # [99] 没有分配过
        content = "引用了一个不存在的来源 [99]。"
        ret, sec = self._setup_section_with_search(content)
        assert 99 not in sec["citations"]


# ── TestCompileReferences ─────────────────────────────────────────────────────

class TestCompileReferences:
    """验证 compile_report 在报告末尾生成参考文献表。"""

    def setup_method(self):
        reset_state()

    def test_reference_table_appended(self):
        from tools.registry import get_state, _make_section_entry, assign_ref_idx
        import tools.report as report_mod

        state = get_state()

        url1 = "https://example.com/source1"
        url2 = "https://example.com/source2"
        idx1 = assign_ref_idx(url1)
        idx2 = assign_ref_idx(url2)

        # 建两个完成的章节
        for i, (title, url, idx) in enumerate([
            ("第一章", url1, idx1), ("第二章", url2, idx2)
        ], start=1):
            key = f"sec_{i:03d}"
            state["sections"][key] = _make_section_entry(title=title, order=i)
            sec = state["sections"][key]
            sec["final"] = f"内容 [{idx}]。"
            sec["status"] = "saved"
            sec["order"] = i
            sec["citations"] = {idx: url}

        mock_cfg = MagicMock()
        mock_cfg.CRITIQUE_MODEL = "test-model"

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cfg.OUTPUT_DIR = os.path.join(tmpdir, "output")
            state["run_dir"] = tmpdir

            mock_llm = {"ok": True, "data": MagicMock()}
            mock_parse = {"ok": True, "data": {"transitions": {}, "summary": ""}}

            with (
                patch("tools.report.get_config", return_value=mock_cfg),
                patch("tools.report.get_client"),
                patch("tools.report.call_llm_with_retry", return_value=mock_llm),
                patch("tools.report.parse_json_with_retry", return_value=mock_parse),
            ):
                ret = report_mod.compile_runner(report_title="测试报告")

            assert "saved_path" in ret
            with open(ret["saved_path"], encoding="utf-8") as f:
                report_text = f.read()

        assert "## 参考文献" in report_text
        assert f"[{idx1}]" in report_text
        assert url1 in report_text
        assert f"[{idx2}]" in report_text
        assert url2 in report_text

    def test_no_reference_table_if_no_citations(self):
        from tools.registry import get_state, _make_section_entry
        import tools.report as report_mod

        state = get_state()
        key = "sec_001"
        state["sections"][key] = _make_section_entry(title="章节一", order=1)
        sec = state["sections"][key]
        sec["final"] = "没有任何引用的内容。"
        sec["status"] = "saved"
        sec["order"] = 1
        sec["citations"] = {}

        mock_cfg = MagicMock()
        mock_cfg.CRITIQUE_MODEL = "test-model"

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cfg.OUTPUT_DIR = os.path.join(tmpdir, "output")
            state["run_dir"] = tmpdir

            mock_llm = {"ok": True, "data": MagicMock()}
            mock_parse = {"ok": True, "data": {"transitions": {}, "summary": ""}}

            with (
                patch("tools.report.get_config", return_value=mock_cfg),
                patch("tools.report.get_client"),
                patch("tools.report.call_llm_with_retry", return_value=mock_llm),
                patch("tools.report.parse_json_with_retry", return_value=mock_parse),
            ):
                ret = report_mod.compile_runner(report_title="无引用报告")

            with open(ret["saved_path"], encoding="utf-8") as f:
                report_text = f.read()

        assert "## 参考文献" not in report_text


# ── TestExtractEvidenceDisabled ───────────────────────────────────────────────

class TestExtractEvidenceDisabled:
    """验证 extract_evidence 工具已停用（不再注册）。"""

    def test_not_registered(self):
        from tools.registry import get_schemas
        names = [s["name"] for s in get_schemas()]
        assert "extract_evidence" not in names
