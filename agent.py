"""
agent.py
--------
主 agentic loop。
维护完整的 messages 对话链，每轮让 LLM 决定调用哪个工具，直到 stop_reason=end_turn。
所有文件（搜索结果、checkpoint、最终报告、trace）写入同一 run_dir：
    traces/{query}_{timestamp}/
"""

import json
import io
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic

import config as cfg
from tools import get_schemas, dispatch, reset_state, get_state, set_client
from context.system_prompt import build as build_system_prompt
from tools.errors import set_error_log_path, read_error_log, ErrorType, is_fatal
from tools.retry import call_with_retry
from tools._client import get_llm_semaphore

# API 调用超时（秒）：超过此时间无响应则抛出 TimeoutError
API_TIMEOUT = 180

# 可以在同一 turn 中并行执行的工具（无全局独占 state 写入）
_PARALLEL_SAFE = {
    "search", "save_section", "critique_section",
    "initialize_criteria", "get_current_date", "rewrite_query",
    "ingest", "outline", "retrieve",
}


def run(query: str) -> str:
    """运行研究任务，并保证异常路径也会恢复进程级 stdout。"""
    original_stdout = sys.stdout
    try:
        return _run_impl(query)
    finally:
        redirected_stdout = sys.stdout
        sys.stdout = original_stdout
        if redirected_stdout is not original_stdout:
            log_file = getattr(redirected_stdout, "_fh", None)
            if log_file is not None and not log_file.closed:
                log_file.close()


def _run_impl(query: str) -> str:
    """
    执行一次完整的深度研究。
    返回 agent 的最终文本回复。
    """
    reset_state()

    # 创建本次运行的工作目录，所有文件都落在这里
    run_dir = _make_run_dir(query)
    os.makedirs(run_dir, exist_ok=True)
    state = get_state()
    state["run_dir"] = run_dir

    # 初始化错误日志路径
    set_error_log_path(os.path.join(run_dir, "errors.jsonl"))

    client = anthropic.Anthropic(
        base_url=cfg.ANTHROPIC_BASE_URL,
        api_key=cfg.API_KEY,
    )
    set_client(client, cfg)

    system = build_system_prompt(cfg.MAX_SECTIONS, cfg.MAX_REFLECT_PER_SECTION)
    tools = get_schemas()
    messages = [{"role": "user", "content": query}]
    trace_path = os.path.join(run_dir, "trace.json")

    print(f"\n{'='*60}")
    print(f"研究问题：{query}")
    print(f"模型：{cfg.MODEL}  最大步骤：{cfg.MAX_AGENT_STEPS}")
    print(f"运行目录：{run_dir}")
    print('='*60)

    # 终端输出同步写入日志文件
    log_path = os.path.join(run_dir, "run.log")
    _log_fh = open(log_path, "w", encoding="utf-8")

    class _Tee(io.TextIOBase):
        def __init__(self, orig, fh):
            self._orig, self._fh = orig, fh
        def write(self, s):
            self._orig.write(s)
            self._orig.flush()
            self._fh.write(s)
            self._fh.flush()
            return len(s)
        def flush(self):
            self._orig.flush()
            self._fh.flush()

    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_fh)

    step = 0
    final_text = ""
    # 运行统计
    stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "api_calls": 0,
        "max_tokens_events": 0,
        "tool_calls": {},   # {tool_name: count}
    }
    # 截断恢复最大重试次数，防止无限循环
    max_truncation_retries = 3
    truncation_retries = 0

    while step < cfg.MAX_AGENT_STEPS:
        step += 1
        print(f"\n[Step {step}]", end=" ", flush=True)

        # 将上一轮完成的宽泛搜索批次提升为永久锁
        if state.get("broad_pending_lock"):
            state["broad_locked"] = True
            state["broad_pending_lock"] = False

        response = _call_api_with_timer(
            client=client,
            model=cfg.MODEL,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            timeout=API_TIMEOUT,
        )

        # 统计 token 用量
        if hasattr(response, "usage") and response.usage:
            stats["total_input_tokens"] += getattr(response.usage, "input_tokens", 0)
            stats["total_output_tokens"] += getattr(response.usage, "output_tokens", 0)
        stats["api_calls"] += 1

        messages.append({"role": "assistant", "content": response.content})
        # assistant 决策立刻落盘，防止 tool dispatch 途中崩溃丢失记录
        _write_trace(trace_path, messages)

        for block in response.content:
            if block.type == "tool_use":
                brief_input = {
                    k: (str(v)[:60] + "..." if isinstance(v, str) and len(str(v)) > 60 else v)
                    for k, v in block.input.items()
                    if k != "sections"
                }
                print(f"→ {block.name}({json.dumps(brief_input, ensure_ascii=False)})")
            elif block.type == "text" and block.text.strip():
                print(f"→ [text]\n{block.text}")
                final_text = block.text

        if response.stop_reason == "end_turn":
            print("\n[完成] stop_reason=end_turn")
            _write_trace(trace_path, messages)
            break

        # ---- max_tokens 截断恢复 ----
        if response.stop_reason == "max_tokens":
            stats["max_tokens_events"] += 1
            if truncation_retries < max_truncation_retries:
                truncation_retries += 1
                print(
                    f"\n[截断] stop_reason=max_tokens（第 {truncation_retries}/{max_truncation_retries} 次恢复）"
                    " → 追加续写提示"
                )
                # 先处理本轮已有的 tool_use（如果有），再追加续写提示
                tool_results, fatal = _collect_tool_results(response, stats)
                if fatal:
                    _write_trace(trace_path, messages)
                    break
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                    _write_trace(trace_path, messages)
                # 追加续写提示，让 agent 在下一 turn 继续
                messages.append({
                    "role": "user",
                    "content": (
                        "你的上一条回复因输出过长被截断（stop_reason=max_tokens）。"
                        "请继续未完成的工具调用或分析，不要重复已完成的步骤。"
                    ),
                })
                _write_trace(trace_path, messages)
                continue
            else:
                print(
                    f"\n[停止] stop_reason=max_tokens 且已达最大恢复次数 {max_truncation_retries}，终止"
                )
                _write_trace(trace_path, messages)
                break

        if response.stop_reason != "tool_use":
            print(f"\n[停止] stop_reason={response.stop_reason}")
            _write_trace(trace_path, messages)
            break

        # 重置截断计数（本轮正常完成，非截断）
        truncation_retries = 0

        tool_results, fatal = _collect_tool_results(response, stats)
        if fatal:
            _write_trace(trace_path, messages)
            break
        messages.append({"role": "user", "content": tool_results})
        _write_trace(trace_path, messages)

    else:
        print(f"\n[警告] 已达最大步骤数 {cfg.MAX_AGENT_STEPS}，强制终止")
        _write_trace(trace_path, messages)

    _print_summary(stats, state)

    # 恢复 stdout，关闭日志文件
    sys.stdout = _orig_stdout
    _log_fh.close()

    return final_text


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _collect_tool_results(response, stats: dict) -> tuple[list, bool]:
    """
    遍历 response.content 中的 tool_use block，dispatch 并收集 tool_result。
    白名单内的工具并行执行（ThreadPoolExecutor），其余串行。
    返回 (tool_results, fatal)：fatal=True 表示有认证失败，需终止 loop。
    """
    blocks = [b for b in response.content if b.type == "tool_use"]
    if not blocks:
        return [], False

    parallel_blocks = [b for b in blocks if b.name in _PARALLEL_SAFE]
    serial_blocks   = [b for b in blocks if b.name not in _PARALLEL_SAFE]

    # results 按原始 block 顺序存放（保证 tool_result 顺序与 tool_use 一致）
    results_map: dict[str, dict] = {}

    # ── 并行 dispatch ────────────────────────────────────────────────────────
    if parallel_blocks:
        with ThreadPoolExecutor(max_workers=len(parallel_blocks)) as executor:
            future_to_block = {
                executor.submit(dispatch, b.name, b.input): b
                for b in parallel_blocks
            }
            for future in as_completed(future_to_block):
                b = future_to_block[future]
                try:
                    results_map[b.id] = future.result()
                except Exception as e:
                    results_map[b.id] = {"error": f"dispatch 异常: {type(e).__name__}: {e}"}

    # ── 串行 dispatch ────────────────────────────────────────────────────────
    for b in serial_blocks:
        results_map[b.id] = dispatch(b.name, b.input)

    # ── 按原始顺序组装 tool_results，主线程统一更新 stats ───────────────────
    tool_results = []
    fatal = False
    stats_lock = threading.Lock()

    for b in blocks:
        result = results_map[b.id]

        if isinstance(result, dict) and not result.get("ok", True):
            error_info = result.get("error", {})
            if isinstance(error_info, dict) and error_info.get("type") == ErrorType.AUTH_FAILURE.value:
                print(f"\n[致命] 认证失败（{b.name}），终止 loop")
                fatal = True

        brief_result = {
            k: (str(v)[:80] + "..." if isinstance(v, str) and len(str(v)) > 80 else v)
            for k, v in (result.items() if isinstance(result, dict) else {}.items())
            if k not in ("results", "criteria", "sections", "report_preview")
        }
        print(f"   ← [{b.name}] {json.dumps(brief_result, ensure_ascii=False)}")

        with stats_lock:
            stats["tool_calls"][b.name] = stats["tool_calls"].get(b.name, 0) + 1

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": b.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    return tool_results, fatal


def _is_retryable_api_error(e: Exception) -> bool:
    """
    主 agent API 异常是否可重试（429/5xx/连接错误），与子 LLM 的 call_with_retry 对称。
    502 等瞬时服务端错误自动重试后通常可恢复。
    """
    return isinstance(e, (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )) or (
        isinstance(e, anthropic.APIStatusError)
        and (getattr(e, "status_code", 0) == 429 or getattr(e, "status_code", 0) >= 500)
    )


def _call_api_with_timer(client, timeout: int, **kwargs):
    """
    调用 client.messages.create，并把超时交给 SDK 主动取消请求。
    超过 timeout 秒无响应则重试 1 次，仍超时才抛出异常。
    HTTP 可重试错误（429/5xx/连接错误）重试 2 次（3s/6s 指数退避）。
    """
    max_attempts = 3  # 初次 + 最多 2 次重试（超时只重试 1 次）
    timeout_retries = 0

    for attempt in range(max_attempts):
        start = time.time()
        print("思考中...", end="", flush=True)
        sem = get_llm_semaphore()
        try:
            with sem:
                response = client.messages.create(timeout=timeout, **kwargs)
        except anthropic.APITimeoutError:
            timeout_retries += 1
            if timeout_retries <= 1:
                print(f"\r[超时] API 调用超过 {timeout}s，重试（{timeout_retries}/1）", flush=True)
                continue
            print(f"\r[超时] API 调用超过 {timeout}s 且重试耗尽，终止", flush=True)
            raise
        except Exception as e:
            if attempt < max_attempts - 1 and _is_retryable_api_error(e):
                wait = 3.0 * (2 ** attempt)  # 3s / 6s 指数退避
                print(f"\r[重试] API 调用失败（{type(e).__name__}），{wait:.0f}s 后重试（{attempt + 1}/2）", flush=True)
                time.sleep(wait)
                continue
            raise e
        elapsed = time.time() - start
        print(f"\r思考完成，耗时 {elapsed:.1f}s          ", flush=True)

        # 打印思考内容（ThinkingBlock）
        for block in response.content:
            if getattr(block, "type", None) == "thinking" and getattr(block, "thinking", ""):
                thinking_text = block.thinking.strip()
                if thinking_text:
                    print(f"[思考]\n{thinking_text}\n", flush=True)

        return response


def _print_summary(stats: dict, state: dict) -> None:
    """运行结束后打印统计摘要。"""
    sections = state.get("sections", {})
    search_sources = state.get("search_sources", {})

    completed = [
        sec for sec in sections.values()
        if sec.get("final") is not None
    ]
    incomplete = [
        sec for sec in sections.values()
        if sec.get("final") is None
    ]

    print(f"\n{'='*60}")
    print("运行摘要")
    print(f"{'='*60}")
    print(f"API 调用次数   : {stats['api_calls']}")
    print(f"输入 token     : {stats['total_input_tokens']:,}")
    print(f"输出 token     : {stats['total_output_tokens']:,}")
    print(f"max_tokens 截断: {stats['max_tokens_events']} 次")
    print(f"\n搜索次数       : {len(search_sources)} 次")
    print(f"章节完成       : {len(completed)}/{len(sections)}")
    if completed:
        for sec in sorted(completed, key=lambda item: item.get("order", 0)):
            print(f"  ✓ {sec.get('title', '未命名章节')}（反思 {sec.get('reflect_count', 0)} 次）")
    if incomplete:
        print("章节未完成：")
        for sec in sorted(incomplete, key=lambda item: item.get("order", 0)):
            print(f"  ✗ {sec.get('title', '未命名章节')}")

    if stats["tool_calls"]:
        print("\n工具调用统计：")
        for name, count in sorted(stats["tool_calls"].items(), key=lambda x: -x[1]):
            print(f"  {name:<30} {count} 次")

    # 错误日志统计
    error_entries = read_error_log()
    if error_entries:
        from collections import Counter
        type_counts = Counter(e.get("type", "unknown") for e in error_entries if "type" in e)
        unresolved = [e for e in error_entries if not e.get("resolved") and "type" in e]
        print(f"\n错误日志：{len(error_entries)} 条（未解决 {len(unresolved)} 条）")
        for etype, cnt in type_counts.most_common():
            print(f"  {etype:<30} {cnt} 次")
    print('='*60)


def _make_run_dir(query: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 先压缩空白（含换行/制表符）：否则 query 带换行会生成非法目录名（WinError 123，id=37 实测）
    query_clean = re.sub(r"\s+", " ", query)[:30]
    safe = re.sub(r"[^\w\s-]", "", query_clean).strip().replace(" ", "_")
    os.makedirs(cfg.TRACE_DIR, exist_ok=True)
    return os.path.join(cfg.TRACE_DIR, f"{safe}_{timestamp}")


def _write_trace(path: str, messages: list) -> None:
    serializable = _serialize_messages(messages)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)


def _serialize_messages(messages: list) -> list:
    result = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            serialized = []
            for block in content:
                if isinstance(block, dict):
                    serialized.append(block)
                elif hasattr(block, "model_dump"):
                    serialized.append(block.model_dump())
                else:
                    serialized.append({"type": "raw", "value": str(block)})
            result.append({"role": msg["role"], "content": serialized})
        else:
            result.append(msg)
    return result
