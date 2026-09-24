"""
最小流式探测脚本：确认兼容端点是否支持 SSE 流式调用。
运行：python scripts/streaming_probe.py
"""
import os
import sys
import time

import anthropic

BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY  = os.getenv("API_KEY")
MODEL    = os.getenv("MODEL", "your-model-name")

if not BASE_URL or not API_KEY:
    print("ERROR: 请先设置 ANTHROPIC_BASE_URL 和 API_KEY 环境变量")
    sys.exit(1)

client = anthropic.Anthropic(base_url=BASE_URL, api_key=API_KEY)

print(f"=== 流式探测 ===")
print(f"base_url : {BASE_URL}")
print(f"model    : {MODEL}")
print()

# ── 测试 1：流式调用，用 get_final_message() ──────────────────────────────
print("[1] 流式 + get_final_message() ...")
t0 = time.time()
try:
    with client.messages.stream(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": "用一句话介绍自己。"}],
    ) as stream:
        msg = stream.get_final_message()
    elapsed = time.time() - t0
    text = next((b.text for b in msg.content if b.type == "text"), "")
    print(f"  OK  stop_reason={msg.stop_reason}  tokens={msg.usage.output_tokens}  elapsed={elapsed:.1f}s")
    print(f"  text: {text[:120]}")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {e}")

print()

# ── 测试 2：流式调用，逐 chunk 计数（验证 SSE 分块是否真正推送）──────────────
print("[2] 流式逐 chunk 计数 ...")
t0 = time.time()
try:
    chunk_count = 0
    accumulated = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": "用一句话介绍自己。"}],
    ) as stream:
        for text in stream.text_stream:
            chunk_count += 1
            accumulated += text
    elapsed = time.time() - t0
    print(f"  OK  chunks={chunk_count}  elapsed={elapsed:.1f}s")
    print(f"  text: {accumulated[:120]}")
    if chunk_count == 1:
        print("  WARNING: only 1 chunk - proxy may be buffering (fake streaming)")
    elif chunk_count == 0:
        print("  WARNING: 0 chunks - model may have only produced ThinkingBlocks")
    else:
        print("  OK: real SSE streaming confirmed")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {e}")

print()
print("=== 探测完毕 ===")
