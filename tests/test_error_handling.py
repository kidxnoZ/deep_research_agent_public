"""
test_error_handling.py — 测试各 Runner 的错误处理路径
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 在导入工具前先 mock 掉 register，避免副作用
import tools.registry as _reg
_orig_register = _reg.register


# ─── Search 错误处理 ──────────────────────────────────────────────────────────

class TestSearchErrorHandling(unittest.TestCase):
    def setUp(self):
        import tools.registry as reg
        reg.reset_state()
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)

    def test_empty_query_returns_error(self):
        from tools.search import runner
        r = runner(query="")
        self.assertIn("error", r)

    def test_search_no_results_returns_error(self):
        from tools.search import runner
        with patch("tools.search.call_with_retry") as mock_retry:
            mock_retry.return_value = {"ok": True, "data": []}
            r = runner(query="something obscure")
        self.assertIn("error", r)
        self.assertIn("搜索无结果", r["error"])

    def test_search_network_error_propagates(self):
        from tools.search import runner
        with patch("tools.search.call_with_retry") as mock_retry:
            mock_retry.return_value = {
                "ok": False,
                "error": {"type": "network_timeout", "message": "timeout", "error_id": "err_x"}
            }
            r = runner(query="test query")
        self.assertIn("error", r)
        self.assertIn("network_timeout", r["error"])

    def test_broad_lock_respected(self):
        import tools.registry as reg
        reg.get_state()["broad_locked"] = True
        from tools.search import runner
        r = runner(query="test")
        self.assertIn("error", r)
        self.assertIn("锁定", r["error"])


# ─── Criteria 错误处理 ─────────────────────────────────────────────────────────

class TestCriteriaErrorHandling(unittest.TestCase):
    def setUp(self):
        import tools.registry as reg
        reg.reset_state()

    def _make_client(self, text="{}"):
        block = MagicMock(); block.type = "text"; block.text = text
        resp = MagicMock(); resp.content = [block]; resp.stop_reason = "end_turn"
        client = MagicMock()
        client.messages.create.return_value = resp
        return client

    def test_all_existing_returns_early(self):
        import tools.registry as reg
        reg.get_state()["sections"]["sec_001"] = reg._make_section_entry(order=1, title="A")
        reg.get_state()["sections"]["sec_001"]["criteria"] = "standard"
        from tools._client import set_client
        import config as cfg
        set_client(self._make_client(), cfg)
        from tools.criteria import runner
        r = runner(query="q", section_titles=["A"])
        self.assertTrue(r.get("initialized"))
        self.assertIn("无需更新", r.get("note", ""))

    def test_llm_failure_returns_error(self):
        import tools.registry as reg
        reg.get_state()["sections"]["sec_001"] = reg._make_section_entry(order=1, title="A")
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)
        from tools.criteria import runner
        with patch("tools.criteria.call_llm_with_retry") as mock_llm:
            mock_llm.return_value = {
                "ok": False,
                "error": {"type": "llm_empty_output", "message": "empty"}
            }
            r = runner(query="q", section_titles=["A"])
        self.assertIn("error", r)

    def test_json_parse_failure_uses_fallback(self):
        import tools.registry as reg
        reg.get_state()["sections"]["sec_001"] = reg._make_section_entry(order=1, title="A")
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)
        from tools.criteria import runner
        with patch("tools.criteria.call_llm_with_retry") as mock_llm, \
             patch("tools.criteria.parse_json_with_retry") as mock_parse:
            mock_llm.return_value = {"ok": True, "data": MagicMock()}
            mock_parse.return_value = {
                "ok": False,
                "error": {"type": "json_parse_error", "message": "bad json"}
            }
            r = runner(query="q", section_titles=["A"])
        # fallback：应返回默认标准而不是 error
        self.assertTrue(r.get("initialized"))
        self.assertIn("A", r["criteria"])


# ─── Critique 错误处理 ─────────────────────────────────────────────────────────

class TestCritiqueErrorHandling(unittest.TestCase):
    def setUp(self):
        import tools.registry as reg
        reg.reset_state()
        # 新结构：把章节写入 sections dict，key 为 sec_NNN，title 作为字段
        reg.get_state()["sections"]["sec_001"] = reg._make_section_entry(order=1, title="A")
        reg.get_state()["sections"]["sec_001"]["draft"]    = "content"
        reg.get_state()["sections"]["sec_001"]["criteria"] = "standard"

    def test_section_not_found(self):
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)
        from tools.reflect import runner
        r = runner(section_title="NonExistent", iteration=1)
        self.assertIn("error", r)

    def test_max_reflect_forces_sufficient(self):
        import tools.registry as reg
        import config as cfg
        reg.get_state()["sections"]["sec_001"]["reflect_count"] = cfg.MAX_REFLECT_PER_SECTION
        from tools._client import set_client
        set_client(MagicMock(), cfg)
        from tools.reflect import runner
        r = runner(section_title="A", iteration=99)
        self.assertTrue(r.get("is_sufficient"))
        self.assertIn("强制通过", r.get("note", ""))

    def test_llm_failure_returns_error(self):
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)
        from tools.reflect import runner
        with patch("tools.reflect.call_llm_with_retry") as mock_llm:
            mock_llm.return_value = {
                "ok": False,
                "error": {"type": "server_error", "message": "500"}
            }
            r = runner(section_title="A", iteration=1)
        self.assertIn("error", r)

    def test_json_parse_fallback(self):
        from tools._client import set_client
        import config as cfg
        set_client(MagicMock(), cfg)
        from tools.reflect import runner
        with patch("tools.reflect.call_llm_with_retry") as mock_llm, \
             patch("tools.reflect.parse_json_with_retry") as mock_parse:
            mock_llm.return_value = {"ok": True, "data": MagicMock()}
            mock_parse.return_value = {
                "ok": False,
                "error": {"type": "json_parse_error", "message": "bad", "error_id": "err_x"}
            }
            r = runner(section_title="A", iteration=1)
        # fallback：保守判为不足
        self.assertFalse(r.get("is_sufficient"))
        self.assertTrue(r.get("parse_error"))


# ─── Registry param_error 计数 ────────────────────────────────────────────────

class TestRegistryParamRetry(unittest.TestCase):
    def setUp(self):
        import tools.registry as reg
        reg.reset_state()

    def test_missing_required_field(self):
        import tools.registry as reg
        r = reg.dispatch("search", {})  # 缺少 query
        self.assertIn("error", r)
        self.assertIn("必填字段", r["error"])

    def test_unknown_tool(self):
        import tools.registry as reg
        r = reg.dispatch("nonexistent_tool", {})
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
