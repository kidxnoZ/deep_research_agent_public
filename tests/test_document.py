"""
test_document.py — ingest / outline / retrieve 单元测试（全部 mock 网络）
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# 确保根目录在 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── mock config / state，避免依赖真实 .env ─────────────────────────────────────

def _make_cfg():
    cfg = types.SimpleNamespace(
        OUTPUT_DIR="/tmp/search_agent_test/reports",
        MAX_DEEP_READS_PER_RUN=5,
        DEEP_READ_MAX_CHARS=4000,
        DEEP_READ_CHUNK_SIZE=1500,
        AUTHORITY_DOMAINS=["arxiv.org", "sec.gov", ".gov", ".edu"],
    )
    return cfg


_fake_state = {
    "documents": {},
    "run_dir": "/tmp/search_agent_test",
}


def _reset_state():
    _fake_state["documents"] = {}
    _fake_state["run_dir"] = "/tmp/search_agent_test"


# ── helper: 直接 patch get_config / get_state 再 import document ───────────────

def _load_document_module():
    """每次用同一模块（已 import 则复用）。"""
    import tools.document as doc_mod
    return doc_mod


# ── 工具函数单测 ───────────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools.document
        importlib.reload(tools.document)   # 确保干净模块
        self.doc = tools.document

    def test_resolve_arxiv_abs(self):
        url = "https://arxiv.org/abs/2305.12345"
        self.assertIn("/html/", self.doc._resolve_arxiv_url(url))

    def test_resolve_arxiv_pdf(self):
        # /pdf/ 尊重原地址，不再强制转 html
        url = "https://arxiv.org/pdf/2305.12345v2"
        self.assertEqual(url, self.doc._resolve_arxiv_url(url))

    def test_resolve_arxiv_html_kept(self):
        url = "https://arxiv.org/html/2305.12345v2"
        self.assertEqual(url, self.doc._resolve_arxiv_url(url))

    def test_resolve_non_arxiv(self):
        url = "https://example.com/paper.pdf"
        self.assertEqual(url, self.doc._resolve_arxiv_url(url))

    def test_guess_title_from_url(self):
        title = self.doc._guess_title("https://example.com/annual_report_2024.pdf")
        self.assertIn("annual", title.lower())

    def test_chunk_text_basic(self):
        text = "# Section A\n" + "word " * 400 + "\n# Section B\n" + "word " * 200
        chunks = self.doc._chunk_text(text, chunk_size=1500)
        self.assertTrue(len(chunks) >= 2)
        sections = {c["section"] for c in chunks}
        self.assertIn("Section A", sections)
        self.assertIn("Section B", sections)

    def test_chunk_text_no_headings(self):
        text = "line\n" * 200
        chunks = self.doc._chunk_text(text, chunk_size=1500)
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            self.assertFalse(c["is_table"])

    def test_extract_outline(self):
        text = "## Intro\nintro text\n## Methods\nmethod text"
        outline = self.doc._extract_outline(text)
        self.assertEqual(len(outline), 2)
        self.assertEqual(outline[0]["title"], "Intro")
        self.assertEqual(outline[1]["title"], "Methods")

    def test_extract_outline_tree(self):
        # 嵌套层级；标题后紧跟子标题（无正文）也不丢失
        text = ("# Paper Title\n## 1 Intro\nbody\n"
                "## 2 Methods\n### 2.1 Data\n### 2.2 Model\n"
                "## 3 Results\n")
        outline = self.doc._extract_outline(text)
        self.assertEqual(len(outline), 1)
        self.assertEqual(outline[0]["title"], "Paper Title")
        children = outline[0]["children"]
        self.assertEqual([c["title"] for c in children], ["1 Intro", "2 Methods", "3 Results"])
        self.assertEqual([c["title"] for c in children[1]["children"]],
                         ["2.1 Data", "2.2 Model"])

    def test_extract_html_tables(self):
        html = """
        <table>
          <tr><th>Model</th><th>Score</th></tr>
          <tr><td>GPT-4</td><td>90</td></tr>
          <tr><td>Llama</td><td>82</td></tr>
        </table>
        """
        tables = self.doc._extract_html_tables(html)
        self.assertEqual(len(tables), 1)
        self.assertIn("GPT-4", tables[0])
        self.assertIn("90", tables[0])

    def test_parse_html_basic(self):
        html = """<html><head><title>Test Page</title></head>
        <body><h1>Introduction</h1><p>Hello world.</p></body></html>"""
        result = self.doc._parse_html(html, "https://example.com/test")
        self.assertTrue(result["ok"])
        self.assertEqual(result["source_type"], "html")
        self.assertIn("Introduction", result["text"])
        self.assertEqual(result["title"], "Test Page")

    def test_parse_html_empty(self):
        html = "<html><body></body></html>"
        result = self.doc._parse_html(html, "https://example.com")
        self.assertFalse(result["ok"])

    def test_parse_html_headings_preserved(self):
        # 回归：</h2> 不再被提前替换，标题应转成 ## 前缀（outline bug 修复）
        html = ("<html><head><title>T</title></head><body>"
                "<h2>1 Introduction</h2><p>Hello world.</p>"
                "<h3>2.1 Methods</h3><p>Detail here.</p>"
                "</body></html>")
        result = self.doc._parse_html(html, "https://example.com/x")
        self.assertTrue(result["ok"])
        self.assertIn("## 1 Introduction", result["text"])
        self.assertIn("### 2.1 Methods", result["text"])

    def test_parse_html_modal_banner_cleaned(self):
        # arxiv 弹窗（form 内）与公告 banner 应被清理
        html = ("<html><head><title>T</title></head><body>"
                '<div class="ds-announcement" role="region">'
                "arXiv is now an independent nonprofit! Learn more</div>"
                "<form>Report GitHub Issue<p>Content selection saved.</p></form>"
                "<h2>1 Intro</h2><p>Real content here.</p>"
                "</body></html>")
        result = self.doc._parse_html(html, "https://arxiv.org/html/x")
        self.assertNotIn("independent nonprofit", result["text"])
        self.assertNotIn("Content selection saved", result["text"])
        self.assertIn("Real content here.", result["text"])

    def test_chunk_text_levels(self):
        text = ("# Title\nintro text\n## Section A\nbody text\n"
                "### Sub A\nsub body\n###### Deep\nx")
        chunks = self.doc._chunk_text(text, chunk_size=1500)
        levels = {(c["section"], c["level"]) for c in chunks}
        self.assertIn(("Title", 1), levels)
        self.assertIn(("Section A", 2), levels)
        self.assertIn(("Sub A", 3), levels)
        self.assertIn(("Deep", 6), levels)

    def test_decode_html_utf8(self):
        raw = "中文内容测试".encode("utf-8")
        self.assertEqual(self.doc._decode_html(raw), "中文内容测试")

    def test_decode_html_gbk(self):
        raw = "中文内容测试".encode("gbk")
        self.assertEqual(self.doc._decode_html(raw), "中文内容测试")

    def test_decode_html_declared_charset(self):
        # 页面声明 gb2312 且确实 GBK 编码 → utf-8 失败后按声明解
        raw = ('<meta charset="gb2312">' + "中文内容").encode("gbk")
        self.assertIn("中文内容", self.doc._decode_html(raw))

    def test_decode_html_utf8_with_wrong_declaration(self):
        # utf-8 优先：即使声明 gb2312，utf-8 能解就用 utf-8（北大新闻网场景）
        raw = ('<meta charset="gb2312">' + "中文内容").encode("utf-8")
        self.assertIn("中文内容", self.doc._decode_html(raw))


class TestBM25(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools.document
        importlib.reload(tools.document)
        self.doc = tools.document

    def test_tokenize_chinese(self):
        tokens = self.doc._tokenize("深度学习模型 accuracy 90%")
        self.assertIn("accuracy", tokens)
        self.assertIn("90", tokens)
        # 中文单字
        self.assertIn("度", tokens)

    def test_bm25_rank_basic(self):
        chunks = [
            {"text": "learning rate batch size optimizer adam", "section": "Training", "is_table": False},
            {"text": "the weather today is sunny and warm", "section": "Other", "is_table": False},
            {"text": "model architecture layers transformer attention", "section": "Arch", "is_table": False},
        ]
        ranked = self.doc._bm25_rank("learning rate optimizer", chunks)
        # 第一个 chunk 应排最高
        self.assertEqual(ranked[0][0]["section"], "Training")

    def test_bm25_table_boost(self):
        chunks = [
            {"text": "accuracy precision recall f1", "section": "Results", "is_table": False},
            {"text": "accuracy precision recall f1", "section": "Table", "is_table": True},
        ]
        ranked = self.doc._bm25_rank("accuracy f1", chunks)
        # 表格块应得分更高（1.5x）
        self.assertTrue(ranked[0][0]["is_table"])

    def test_bm25_empty_query(self):
        chunks = [{"text": "hello world", "section": "", "is_table": False}]
        ranked = self.doc._bm25_rank("", chunks)
        self.assertEqual(len(ranked), 1)


# ── doc_candidate 标记测试 ─────────────────────────────────────────────────────

class TestDocCandidate(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools.search as s
        importlib.reload(s)
        self.search = s

    def test_pdf_suffix(self):
        self.assertTrue(self.search._is_doc_candidate("https://example.com/paper.pdf"))

    def test_pdf_endpoint(self):
        self.assertTrue(self.search._is_doc_candidate("https://chinamoney.org.cn/fileDownLoad.do?mode=save"))

    def test_arxiv_domain(self):
        self.assertTrue(self.search._is_doc_candidate("https://arxiv.org/abs/2305.12345"))

    def test_sec_gov(self):
        self.assertTrue(self.search._is_doc_candidate("https://sec.gov/filing/annual.htm"))

    def test_ordinary_url(self):
        self.assertFalse(self.search._is_doc_candidate("https://wikipedia.org/wiki/Python"))

    def test_empty_url(self):
        self.assertFalse(self.search._is_doc_candidate(""))

    def test_mark_doc_candidates(self):
        results = [
            {"title": "arXiv paper", "url": "https://arxiv.org/abs/1234.5678", "snippet": "..."},
            {"title": "Wiki page",   "url": "https://wikipedia.org/wiki/ML",   "snippet": "..."},
            {"title": "SEC filing",  "url": "https://sec.gov/filing.pdf",       "snippet": "..."},
        ]
        marked, candidates = self.search._mark_doc_candidates(results)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(any(r.get("doc_candidate") for r in marked))
        wiki = next(r for r in marked if "wikipedia" in r["url"])
        self.assertFalse(wiki.get("doc_candidate", False))


# ── ingest runner（mock HTTP）─────────────────────────────────────────────────

SAMPLE_HTML = """<html>
<head><title>Deep Learning Survey</title></head>
<body>
<h1>Introduction</h1>
<p>Deep learning has transformed many fields.</p>
<h2>Methods</h2>
<p>We use transformer architecture with attention mechanism.</p>
<table>
  <tr><th>Model</th><th>Params</th><th>Accuracy</th></tr>
  <tr><td>BERT</td><td>110M</td><td>92.1</td></tr>
  <tr><td>GPT-4</td><td>1.8T</td><td>96.3</td></tr>
</table>
<h2>Results</h2>
<p>Our model achieves state-of-the-art performance.</p>
</body>
</html>"""


class TestIngestRunner(unittest.TestCase):

    def setUp(self):
        _reset_state()

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    @patch("tools.document.requests.get")
    def test_ingest_html(self, mock_get, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = _fake_state

        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.text = SAMPLE_HTML
        resp.content = SAMPLE_HTML.encode("utf-8")
        resp.raise_for_status = lambda: None
        mock_get.return_value = resp

        import tools.document as doc_mod
        result = doc_mod.ingest_runner("https://example.com/survey.html")

        self.assertNotIn("error", result, result.get("error"))
        self.assertIn("doc_id", result)
        self.assertEqual(result["source_type"], "html")
        self.assertTrue(result["has_tables"])
        self.assertIn(result["doc_id"], _fake_state["documents"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    @patch("tools.document.requests.get")
    def test_ingest_idempotent(self, mock_get, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = _fake_state

        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.text = SAMPLE_HTML
        resp.content = SAMPLE_HTML.encode("utf-8")
        resp.raise_for_status = lambda: None
        mock_get.return_value = resp

        import tools.document as doc_mod
        r1 = doc_mod.ingest_runner("https://example.com/survey.html")
        r2 = doc_mod.ingest_runner("https://example.com/survey.html")
        self.assertEqual(r1["doc_id"], r2["doc_id"])
        self.assertIn("note", r2)  # 第二次应返回"已存在"note

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_ingest_limit(self, mock_gs, mock_gc):
        cfg = _make_cfg()
        cfg.MAX_DEEP_READS_PER_RUN = 2
        mock_gc.return_value = cfg
        state = {"documents": {"doc_1": {}, "doc_2": {}}, "run_dir": "/tmp"}
        mock_gs.return_value = state

        import tools.document as doc_mod
        result = doc_mod.ingest_runner("https://example.com/new.html")
        self.assertIn("error", result)
        self.assertIn("上限", result["error"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    @patch("tools.document.requests.get")
    def test_ingest_empty_page(self, mock_get, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = _fake_state

        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<html><body></body></html>"
        resp.content = b"<html><body></body></html>"
        resp.raise_for_status = lambda: None
        mock_get.return_value = resp

        import tools.document as doc_mod
        result = doc_mod.ingest_runner("https://example.com/empty.html")
        self.assertIn("error", result)


# ── outline runner ────────────────────────────────────────────────────────────

class TestOutlineRunner(unittest.TestCase):

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_outline_ok(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        doc_entry = {
            "source": "https://x.com/paper.html",
            "source_type": "html",
            "title": "My Paper",
            "path": "/tmp/doc_abc",
            "chunks": [
                {"text": "intro text", "section": "Introduction", "is_table": False},
                {"text": "method text", "section": "Methods", "is_table": False},
            ],
            "outline": [
                {"title": "Introduction", "level": 2, "children": []},
                {"title": "Methods", "level": 2, "children": []},
            ],
            "has_tables": False,
        }
        mock_gs.return_value = {"documents": {"doc_abc": doc_entry}}

        import tools.document as doc_mod
        result = doc_mod.outline_runner("doc_abc")
        self.assertEqual(result["doc_id"], "doc_abc")
        self.assertEqual([n["title"] for n in result["outline"]],
                         ["Introduction", "Methods"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_outline_missing(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = {"documents": {}}

        import tools.document as doc_mod
        result = doc_mod.outline_runner("doc_nonexistent")
        self.assertIn("error", result)


# ── retrieve runner ───────────────────────────────────────────────────────────

class TestRetrieveRunner(unittest.TestCase):

    def _make_doc_state(self):
        chunks = [
            {"text": "learning rate 0.001 batch size 32 adam optimizer weight decay",
             "section": "Training Setup", "is_table": False},
            {"text": "transformer architecture self-attention multi-head layers 12",
             "section": "Architecture", "is_table": False},
            {"text": "MMLU score 89.2 HumanEval 72.1 GSM8K 95.0",
             "section": "Results", "is_table": True},
            {"text": "the cat sat on the mat",
             "section": "Other", "is_table": False},
        ]
        doc_entry = {
            "source": "https://x.com", "source_type": "html", "title": "Paper",
            "path": "/tmp/x", "chunks": chunks, "outline": [], "has_tables": True,
        }
        return {"documents": {"doc_test": doc_entry}}

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_relevant(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = self._make_doc_state()

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_test", "learning rate optimizer")
        self.assertNotIn("error", result)
        self.assertIn("Training Setup", result["content"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_table_prioritized(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = self._make_doc_state()

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_test", "MMLU score benchmark")
        self.assertNotIn("error", result)
        # 表格块命中率高，应出现在结果中
        self.assertIn("MMLU", result["content"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_max_chars(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = self._make_doc_state()

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_test", "learning rate", max_chars=50)
        self.assertNotIn("error", result)
        self.assertLessEqual(result["chars_returned"], 200)  # 截断后不会超很多

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_no_match(self, mock_gs, mock_gc):
        # 硬负反馈：query 与文档零重合 → 明确 error（建议转 search），不硬塞无关块
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = self._make_doc_state()

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_test", "quantum entanglement blockchain")
        self.assertIn("error", result)
        self.assertIn("search", result["error"])

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_missing_doc(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = {"documents": {}}

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_ghost", "query")
        self.assertIn("error", result)

    @patch("tools.document.get_config")
    @patch("tools.document.get_state")
    def test_retrieve_empty_query(self, mock_gs, mock_gc):
        mock_gc.return_value = _make_cfg()
        mock_gs.return_value = self._make_doc_state()

        import tools.document as doc_mod
        result = doc_mod.retrieve_runner("doc_test", "")
        self.assertIn("error", result)


# ── PDF 标题提取（字号启发式）────────────────────────────────────────────────

class TestPdfHeadings(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools.document
        importlib.reload(tools.document)
        self.doc = tools.document

    def _page(self, lines):
        # 构造 pymupdf get_text("dict") 结构
        return {"blocks": [{
            "type": 0,
            "lines": [{"spans": [{"text": t, "size": s}]} for t, s in lines],
        }]}

    def test_heading_by_size(self):
        pages = [self._page([
            ("Qwen-Audio Technical Report", 18.0),
            ("1 Introduction", 12.0),
            ("This is body text with many words to establish the base size.", 10.0),
            ("More body text here to add weight to the base size.", 10.0),
            ("2 Methods", 12.0),
            ("Another body sentence to increase the body weight.", 10.0),
        ])]
        text = self.doc._pdf_text_with_headings(pages)
        self.assertIn("# Qwen-Audio Technical Report", text)
        self.assertIn("## 1 Introduction", text)
        self.assertIn("## 2 Methods", text)
        self.assertIn("This is body text", text)
        # 正文行不得被标为标题
        self.assertNotIn("# This is body text", text)

    def test_heading_by_number_pattern(self):
        # 无大字号时，编号模式短行识别为标题（同一最深级）
        pages = [self._page([
            ("1 Introduction", 10.0),
            ("Body text of the introduction section.", 10.0),
            ("1.1 Motivation", 10.0),
            ("More body text of motivation.", 10.0),
        ])]
        text = self.doc._pdf_text_with_headings(pages)
        self.assertIn("# 1 Introduction", text)
        self.assertIn("# 1.1 Motivation", text)
        self.assertNotIn("# Body text", text)

    def test_empty_pages(self):
        self.assertEqual(self.doc._pdf_text_with_headings([]), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
