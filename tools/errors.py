"""
errors.py
---------
统一错误类型、Result 工厂函数、错误日志写入。

设计原则：
- 基础设施可恢复错误（网络/429/5xx/LLM截断）由 retry 层处理，Agent 不感知
- 参数错误 / 业务错误返回 Agent，Agent 决定策略（最多重试 2 次）
- 认证失败直接终止 loop，Agent 无法修复
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ErrorType(str, Enum):
    # 基础设施 — retry 层内部处理，Agent 不感知（除非耗尽）
    NETWORK_TIMEOUT   = "network_timeout"
    CONNECTION_ERROR  = "connection_error"
    RATE_LIMIT        = "rate_limit"       # HTTP 429
    SERVER_ERROR      = "server_error"     # HTTP 5xx
    LLM_MAX_TOKENS    = "llm_max_tokens"   # stop_reason=max_tokens
    LLM_EMPTY_OUTPUT  = "llm_empty_output" # extract_text 返回空
    JSON_PARSE_ERROR  = "json_parse_error" # LLM 输出 JSON 解析失败

    # Agent 决策 — 直接返回，Agent 决定策略
    PARAM_ERROR       = "param_error"      # 400 / schema 不合法 / 必填字段缺失
    SEARCH_NO_RESULTS = "search_no_results"
    BUSINESS_ERROR    = "business_error"   # 超限、章节不存在等

    # 致命 — 直接终止 loop
    AUTH_FAILURE      = "auth_failure"     # 401 / 403


# 哪些类型在 retry 耗尽后可以传给 Agent 决策（不直接致命）
AGENT_VISIBLE_AFTER_RETRY = {
    ErrorType.NETWORK_TIMEOUT,
    ErrorType.CONNECTION_ERROR,
    ErrorType.RATE_LIMIT,
    ErrorType.SERVER_ERROR,
    ErrorType.LLM_MAX_TOKENS,
    ErrorType.LLM_EMPTY_OUTPUT,
    ErrorType.JSON_PARSE_ERROR,
}

# 哪些类型直接致命（不经过 Agent）
FATAL_TYPES = {ErrorType.AUTH_FAILURE}


# ---------------------------------------------------------------------------
# Result factories
# ---------------------------------------------------------------------------

def ok(data: dict) -> dict:
    """成功结果。"""
    return {"ok": True, "data": data}


def err(
    error_type: ErrorType,
    message: str,
    *,
    retryable: bool = False,
    error_id: str | None = None,
    extra: dict | None = None,
) -> dict:
    """失败结果（已进入 Agent 可见阶段，retryable=False 表示重试已耗尽或本就不可重试）。"""
    payload: dict[str, Any] = {
        "type": error_type.value,
        "message": message,
        "retryable": retryable,
    }
    if error_id:
        payload["error_id"] = error_id
    if extra:
        payload.update(extra)
    return {"ok": False, "error": payload}


def is_fatal(result: dict) -> bool:
    """判断 result 是否为致命错误（需终止 loop）。"""
    if result.get("ok"):
        return False
    etype = result.get("error", {}).get("type")
    return etype == ErrorType.AUTH_FAILURE.value


def unwrap(result: dict) -> dict:
    """从 ok result 中取出 data；若非 ok 则抛 RuntimeError（仅用于测试/内部断言）。"""
    if not result.get("ok"):
        raise RuntimeError(f"unwrap failed: {result}")
    return result["data"]


# ---------------------------------------------------------------------------
# Error log
# ---------------------------------------------------------------------------

_log_path: str = ""


def set_error_log_path(path: str) -> None:
    """agent.py 在创建 run_dir 后调用，设置本次运行的错误日志路径。"""
    global _log_path
    _log_path = path


def log_error(
    tool_name: str,
    error_type: ErrorType,
    message: str,
    *,
    exc: BaseException | None = None,
    retried: int = 0,
    resolved: bool = False,
) -> str:
    """
    将错误写入 errors.jsonl，返回 error_id。
    traceback 只在日志里，不塞给 LLM 上下文。
    """
    error_id = f"err_{uuid.uuid4().hex[:8]}"
    entry = {
        "error_id":  error_id,
        "timestamp": datetime.now().isoformat(),
        "tool":      tool_name,
        "type":      error_type.value,
        "message":   message,
        "traceback": traceback.format_exc() if exc else "",
        "retried":   retried,
        "resolved":  resolved,
    }
    if _log_path:
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return error_id


def read_error_log() -> list[dict]:
    """读取本次运行的所有错误记录（用于摘要统计）。"""
    if not _log_path or not os.path.exists(_log_path):
        return []
    entries = []
    with open(_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
