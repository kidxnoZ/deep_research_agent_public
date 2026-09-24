"""
retry.py
--------
通用重试装饰器和 LLM 调用包装器。

使用方式：
    result = call_with_retry(fn, *args, tool_name="search", **kwargs)

返回 Result dict（ok/error 格式），不抛异常。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import anthropic
import requests

from .errors import ErrorType, err, log_error
from ._client import get_llm_semaphore

# 全局默认重试次数：最多 3 次重试，共 4 次尝试
MAX_RETRIES = 3
# 初始退避时间（秒），每次翻倍：3s → 6s → 12s
# 3s 起步是对齐 arxiv 官方 3s/请求门槛（对超时同样适用：arxiv 超时常为限流软表现）
BASE_BACKOFF = 3.0


# ---------------------------------------------------------------------------
# HTTP 状态码 → ErrorType
# ---------------------------------------------------------------------------

def _classify_http(status: int) -> ErrorType:
    if status in (401, 403):
        return ErrorType.AUTH_FAILURE
    if status == 429:
        return ErrorType.RATE_LIMIT
    if status == 400:
        return ErrorType.PARAM_ERROR
    if status >= 500:
        return ErrorType.SERVER_ERROR
    return ErrorType.BUSINESS_ERROR


def _is_retryable(etype: ErrorType) -> bool:
    return etype in {
        ErrorType.NETWORK_TIMEOUT,
        ErrorType.CONNECTION_ERROR,
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER_ERROR,
    }


def _get_semaphore():
    """获取全局 LLM 并发信号量，未初始化时返回无限制的占位对象。"""
    try:
        return get_llm_semaphore()
    except RuntimeError:
        import threading
        s = threading.Semaphore(999)
        return s


# ---------------------------------------------------------------------------
# 通用 HTTP/requests 重试
# ---------------------------------------------------------------------------

def call_with_retry(
    fn: Callable,
    *args,
    tool_name: str = "unknown",
    max_retries: int = MAX_RETRIES,
    **kwargs,
) -> dict:
    """
    执行 fn(*args, **kwargs)，自动重试可恢复的基础设施错误。
    fn 应返回任意值（不是 Result dict）；本函数将成功结果包装为 {"ok": True, "data": <返回值>}。
    fn 内部抛出的异常被分类后决定是否重试。
    """
    last_error_id = None
    for attempt in range(max_retries + 1):
        retry_after = None
        try:
            result = fn(*args, **kwargs)
            # 若之前有重试失败，标记为 resolved
            if last_error_id:
                _mark_resolved(last_error_id)
            return {"ok": True, "data": result}

        except requests.exceptions.Timeout as e:
            etype = ErrorType.NETWORK_TIMEOUT
            msg = f"请求超时: {e}"
        except requests.exceptions.ConnectionError as e:
            etype = ErrorType.CONNECTION_ERROR
            msg = f"连接失败: {e}"
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            etype = _classify_http(status)
            retry_after = _parse_retry_after(e.response) if e.response is not None else None
            msg = f"HTTP {status}: {e}"
        except requests.exceptions.RequestException as e:
            etype = ErrorType.CONNECTION_ERROR
            msg = f"请求异常: {e}"
            retry_after = None
        except anthropic.AuthenticationError as e:
            etype = ErrorType.AUTH_FAILURE
            msg = f"认证失败: {e}"
        except anthropic.RateLimitError as e:
            etype = ErrorType.RATE_LIMIT
            msg = f"限流: {e}"
            retry_after = None
        except anthropic.APIStatusError as e:
            etype = _classify_http(e.status_code)
            msg = f"API 错误 HTTP {e.status_code}: {e}"
            retry_after = None
        except anthropic.APIConnectionError as e:
            etype = ErrorType.CONNECTION_ERROR
            msg = f"API 连接失败: {e}"
            retry_after = None
        except anthropic.APITimeoutError as e:
            etype = ErrorType.NETWORK_TIMEOUT
            msg = f"API 超时: {e}"
            retry_after = None
        except Exception as e:
            etype = ErrorType.BUSINESS_ERROR
            msg = f"未知异常: {type(e).__name__}: {e}"
            retry_after = None

        last_error_id = log_error(
            tool_name, etype, msg,
            exc=None,  # traceback 由 format_exc 在 log_error 内部捕获
            retried=attempt,
            resolved=False,
        )

        # 致命错误立即返回
        if etype == ErrorType.AUTH_FAILURE:
            return err(etype, msg, error_id=last_error_id)

        # 不可重试的参数/业务错误立即返回 Agent
        if not _is_retryable(etype):
            return err(etype, msg, error_id=last_error_id)

        # 可重试：未到上限则等待后重试
        if attempt < max_retries:
            # 429 优先 Retry-After；否则统一指数退避 BASE_BACKOFF×2^attempt（3s/6s/12s）
            wait = retry_after if (etype == ErrorType.RATE_LIMIT and retry_after) else BASE_BACKOFF * (2 ** attempt)
            print(f"   [retry] {tool_name} {etype.value}，{wait:.1f}s 后重试（{attempt+1}/{max_retries}）")
            time.sleep(wait)
        else:
            return err(etype, msg, retryable=False, error_id=last_error_id)

    # should not reach
    return err(ErrorType.BUSINESS_ERROR, "retry loop 异常退出", retryable=False)


# ---------------------------------------------------------------------------
# LLM 调用包装（支持 max_tokens 自增重试 + JSON 修复重试）
# ---------------------------------------------------------------------------

def _stream_create(client, **kwargs):
    with client.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


def call_llm_with_retry(
    client,
    *,
    tool_name: str,
    max_tokens: int,
    max_tokens_ceiling: int,
    model: str,
    system: str,
    messages: list,
    max_retries: int = MAX_RETRIES,
    repair_prompt: str | None = None,
) -> dict:
    """
    调用 client.messages.stream（流式，规避 SDK 10 分钟非流式保护），处理：
    - 网络/429/5xx：等待后重试（原参数）
    - max_tokens 截断：增加 budget（×2，不超 ceiling）后重试
    - 空输出：原参数重试
    返回 {"ok": True, "data": response} 或 {"ok": False, "error": {...}}
    """
    current_max_tokens = max_tokens
    current_messages = list(messages)
    last_error_id = None

    for attempt in range(max_retries + 1):
        # 用通用 retry 包装单次调用，信号量限制并发
        sem = _get_semaphore()
        sem.acquire()
        try:
            result = call_with_retry(
                _stream_create,
                client,
                tool_name=tool_name,
                max_retries=0,
                model=model,
                max_tokens=current_max_tokens,
                system=system,
                messages=current_messages,
            )
        finally:
            sem.release()

        if not result["ok"]:
            etype = result["error"]["type"]
            # 认证失败 / 参数错误 / 不可重试 → 直接返回
            if etype in (ErrorType.AUTH_FAILURE.value, ErrorType.PARAM_ERROR.value) or \
               not _is_retryable(ErrorType(etype)):
                return result
            # 其他可重试错误
            if attempt < max_retries:
                wait = BASE_BACKOFF * (2 ** attempt)
                print(f"   [retry] {tool_name} LLM 错误，{wait:.1f}s 后重试（{attempt+1}/{max_retries}）")
                time.sleep(wait)
                continue
            return result

        response = result["data"]

        # stop_reason=max_tokens：尝试增大 budget
        if response.stop_reason == "max_tokens":
            new_tokens = min(current_max_tokens * 2, max_tokens_ceiling)
            if new_tokens > current_max_tokens and attempt < max_retries:
                last_error_id = log_error(
                    tool_name, ErrorType.LLM_MAX_TOKENS,
                    f"max_tokens={current_max_tokens} 截断，增至 {new_tokens} 重试",
                    retried=attempt, resolved=False,
                )
                print(f"   [retry] {tool_name} max_tokens 截断，{current_max_tokens}→{new_tokens}，重试（{attempt+1}/{max_retries}）")
                current_max_tokens = new_tokens
                continue
            # budget 已到上限或重试耗尽
            last_error_id = log_error(
                tool_name, ErrorType.LLM_MAX_TOKENS,
                f"max_tokens 截断，budget ceiling={max_tokens_ceiling} 已达上限",
                retried=attempt, resolved=False,
            )
            return {**err(ErrorType.LLM_MAX_TOKENS,
                          f"输出被截断且已达 budget 上限 {max_tokens_ceiling}",
                          error_id=last_error_id),
                    "partial_response": response}

        # 空输出：原参数重试；若携带 repair_prompt 则追加修复消息
        text = _extract_text(response)
        if not text:
            if attempt < max_retries:
                last_error_id = log_error(
                    tool_name, ErrorType.LLM_EMPTY_OUTPUT,
                    "子 LLM 输出为空，重试",
                    retried=attempt, resolved=False,
                )
                print(f"   [retry] {tool_name} 空输出，重试（{attempt+1}/{max_retries}）")
                if repair_prompt:
                    current_messages = list(messages) + [
                        {"role": "assistant", "content": ""},
                        {"role": "user",      "content": repair_prompt},
                    ]
                continue
            last_error_id = log_error(
                tool_name, ErrorType.LLM_EMPTY_OUTPUT,
                "子 LLM 持续输出为空，重试耗尽",
                retried=attempt, resolved=False,
            )
            return err(ErrorType.LLM_EMPTY_OUTPUT,
                       "子 LLM 输出为空且重试耗尽",
                       error_id=last_error_id)

        # 成功
        if last_error_id:
            _mark_resolved(last_error_id)
        return {"ok": True, "data": response}

    return err(ErrorType.BUSINESS_ERROR, "LLM retry loop 异常退出", retryable=False)


# ---------------------------------------------------------------------------
# JSON 解析重试
# ---------------------------------------------------------------------------

def parse_json_with_retry(
    raw: str,
    parse_fn: Callable[[str], dict | None],
    *,
    tool_name: str,
    client=None,
    model: str = "",
    system: str = "",
    original_messages: list | None = None,
    max_tokens: int = 4000,
    max_retries: int = 1,
) -> dict:
    """
    先用 parse_fn 尝试解析 raw。
    失败后若提供了 client，则向子 LLM 发送修复 prompt 重新生成，最多 max_retries 次。
    返回 {"ok": True, "data": parsed_dict} 或 {"ok": False, "error": {...}}
    """
    parsed = parse_fn(raw)
    if parsed is not None:
        return {"ok": True, "data": parsed}

    if not client or not original_messages:
        error_id = log_error(tool_name, ErrorType.JSON_PARSE_ERROR,
                             f"JSON 解析失败且无修复 client，原文前300字: {raw[:300]}")
        return err(ErrorType.JSON_PARSE_ERROR,
                   f"JSON 解析失败，原文前300字: {raw[:300]}",
                   error_id=error_id)

    repair_msg = (
        "你的上一条回复不是合法的 JSON。请只输出一个合法的 JSON 对象，"
        "不加任何前缀、后缀或代码块标记，直接以 { 开头。"
    )

    for attempt in range(max_retries):
        error_id = log_error(tool_name, ErrorType.JSON_PARSE_ERROR,
                             f"JSON 解析失败，发送修复 prompt 重试（{attempt+1}/{max_retries}）",
                             retried=attempt, resolved=False)
        print(f"   [retry] {tool_name} JSON 解析失败，发送修复 prompt（{attempt+1}/{max_retries}）")

        repair_messages = list(original_messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user",      "content": repair_msg},
        ]
        sem = _get_semaphore()
        sem.acquire()
        try:
            result = call_with_retry(
                client.messages.create,
                tool_name=tool_name,
                max_retries=0,
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=repair_messages,
            )
        finally:
            sem.release()
        if not result["ok"]:
            return result

        new_raw = _extract_text(result["data"])
        parsed = parse_fn(new_raw)
        if parsed is not None:
            _mark_resolved(error_id)
            return {"ok": True, "data": parsed}
        raw = new_raw  # 下一次修复用新原文

    error_id = log_error(tool_name, ErrorType.JSON_PARSE_ERROR,
                         f"JSON 修复重试耗尽，原文前300字: {raw[:300]}",
                         retried=max_retries, resolved=False)
    return err(ErrorType.JSON_PARSE_ERROR,
               f"JSON 解析失败且修复重试耗尽",
               error_id=error_id)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    return ""


def _parse_retry_after(response) -> float | None:
    """解析 Retry-After header；无 header 或非法值返回 None。"""
    if response is None:
        return None
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _mark_resolved(error_id: str) -> None:
    """在 error_log 中标记该 error_id 为 resolved（追加一条修正记录）。"""
    from .errors import _log_path, log_error as _le
    if not _log_path:
        return
    try:
        with open(_log_path, "a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps({"error_id": error_id, "resolved": True}) + "\n")
    except OSError:
        pass
