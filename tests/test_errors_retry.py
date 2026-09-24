"""
test_errors_retry.py — 测试 errors.py 和 retry.py 的核心行为
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import anthropic

from tools.errors import (
    ErrorType, ok, err, is_fatal, unwrap,
    set_error_log_path, log_error, read_error_log,
)
from tools.retry import call_with_retry, call_llm_with_retry, parse_json_with_retry


class TestResultFactories(unittest.TestCase):
    def test_ok(self):
        r = ok({"x": 1})
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"]["x"], 1)

    def test_err(self):
        r = err(ErrorType.NETWORK_TIMEOUT, "timeout", error_id="err_abc")
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "network_timeout")
        self.assertEqual(r["error"]["error_id"], "err_abc")

    def test_is_fatal_auth(self):
        r = err(ErrorType.AUTH_FAILURE, "401")
        self.assertTrue(is_fatal(r))

    def test_is_fatal_ok(self):
        self.assertFalse(is_fatal(ok({})))

    def test_is_fatal_non_auth(self):
        r = err(ErrorType.NETWORK_TIMEOUT, "timeout")
        self.assertFalse(is_fatal(r))

    def test_unwrap_ok(self):
        self.assertEqual(unwrap(ok({"a": 1})), {"a": 1})

    def test_unwrap_err_raises(self):
        with self.assertRaises(RuntimeError):
            unwrap(err(ErrorType.PARAM_ERROR, "bad"))


class TestErrorLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.tmp.close()
        set_error_log_path(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)
        set_error_log_path("")

    def test_log_and_read(self):
        eid = log_error("search", ErrorType.NETWORK_TIMEOUT, "timed out", retried=1)
        self.assertTrue(eid.startswith("err_"))
        entries = read_error_log()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "network_timeout")
        self.assertEqual(entries[0]["tool"], "search")
        self.assertFalse(entries[0]["resolved"])

    def test_read_empty(self):
        self.assertEqual(read_error_log(), [])


class TestCallWithRetry(unittest.TestCase):
    def test_success_first_try(self):
        fn = MagicMock(return_value=42)
        r = call_with_retry(fn, tool_name="t")
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], 42)
        fn.assert_called_once()

    def test_retry_on_timeout(self):
        fn = MagicMock(side_effect=[
            requests.exceptions.Timeout("slow"),
            42,
        ])
        with patch("tools.retry.time.sleep"):
            r = call_with_retry(fn, tool_name="t", max_retries=2)
        self.assertTrue(r["ok"])
        self.assertEqual(fn.call_count, 2)

    def test_no_retry_on_auth_failure(self):
        err_resp = MagicMock()
        err_resp.status_code = 401
        fn = MagicMock(side_effect=anthropic.AuthenticationError(
            "bad key", response=MagicMock(), body={}
        ))
        r = call_with_retry(fn, tool_name="t", max_retries=2)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "auth_failure")
        fn.assert_called_once()  # 不重试

    def test_no_retry_on_param_error(self):
        resp = MagicMock()
        resp.status_code = 400
        fn = MagicMock(side_effect=requests.exceptions.HTTPError(response=resp))
        r = call_with_retry(fn, tool_name="t", max_retries=2)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "param_error")
        fn.assert_called_once()

    def test_exhaust_retries_returns_error(self):
        fn = MagicMock(side_effect=requests.exceptions.Timeout("slow"))
        with patch("tools.retry.time.sleep"):
            r = call_with_retry(fn, tool_name="t", max_retries=2)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "network_timeout")
        self.assertEqual(fn.call_count, 3)  # 初始 + 2 次重试


class TestCallLlmWithRetry(unittest.TestCase):
    def _make_response(self, text="hello", stop_reason="end_turn"):
        block = MagicMock()
        block.type = "text"
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = stop_reason
        return resp

    def _make_stream_client(self, responses):
        """子 LLM 走 client.messages.stream；responses 按调用顺序消费。"""
        it = iter(responses)

        class _Stream:
            def __init__(self, resp):
                self._resp = resp
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def get_final_message(self):
                return self._resp

        client = MagicMock()
        client.messages.stream.side_effect = lambda **kw: _Stream(next(it))
        client.messages.create.side_effect = lambda **kw: next(it)
        return client

    def test_success(self):
        client = self._make_stream_client([self._make_response("ok")])
        r = call_llm_with_retry(
            client, tool_name="t",
            max_tokens=1000, max_tokens_ceiling=2000,
            model="m", system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertTrue(r["ok"])

    def test_max_tokens_increases_budget(self):
        client = self._make_stream_client([
            self._make_response("", stop_reason="max_tokens"),
            self._make_response("full output"),
        ])
        with patch("tools.retry.time.sleep"):
            r = call_llm_with_retry(
                client, tool_name="t",
                max_tokens=1000, max_tokens_ceiling=2000,
                model="m", system="s",
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertTrue(r["ok"])
        # 第二次调用应用了更大的 max_tokens
        second_call_kwargs = client.messages.stream.call_args_list[1][1]
        self.assertGreater(second_call_kwargs["max_tokens"], 1000)

    def test_max_tokens_at_ceiling_returns_error(self):
        client = self._make_stream_client([self._make_response("", stop_reason="max_tokens")])
        r = call_llm_with_retry(
            client, tool_name="t",
            max_tokens=2000, max_tokens_ceiling=2000,  # 已在上限
            model="m", system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "llm_max_tokens")


class TestParseJsonWithRetry(unittest.TestCase):
    def _parse(self, text):
        import json, re
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    def test_success_no_retry(self):
        r = parse_json_with_retry('{"a": 1}', self._parse, tool_name="t")
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"]["a"], 1)

    def test_fail_no_client(self):
        r = parse_json_with_retry("not json", self._parse, tool_name="t")
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["type"], "json_parse_error")

    def test_repair_retry_success(self):
        block = MagicMock(); block.type = "text"; block.text = '{"repaired": true}'
        resp = MagicMock(); resp.content = [block]; resp.stop_reason = "end_turn"
        client = MagicMock()
        client.messages.create.return_value = resp

        r = parse_json_with_retry(
            "not json", self._parse,
            tool_name="t", client=client,
            model="m", system="s",
            original_messages=[{"role": "user", "content": "q"}],
        )
        self.assertTrue(r["ok"])
        self.assertTrue(r["data"]["repaired"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
