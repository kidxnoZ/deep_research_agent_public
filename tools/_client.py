"""
_client.py
----------
全局共享的 Anthropic 客户端引用。
由 agent.py 在启动时通过 set_client() 注入，
criteria.py / reflect.py / report.py 通过 get_client() 获取，用于子 LLM 调用。
"""
import threading

_client = None
_config = None
_llm_semaphore: threading.Semaphore | None = None


def set_client(client, config) -> None:
    global _client, _config, _llm_semaphore
    _client = client
    _config = config
    limit = getattr(config, "MAX_CONCURRENT_LLM_CALLS", 3)
    _llm_semaphore = threading.Semaphore(limit)


def get_client():
    if _client is None:
        raise RuntimeError("LLM client 未初始化，请先调用 set_client()")
    return _client


def get_config():
    if _config is None:
        raise RuntimeError("Config 未初始化，请先调用 set_client()")
    return _config


def get_llm_semaphore() -> threading.Semaphore:
    if _llm_semaphore is None:
        raise RuntimeError("LLM semaphore 未初始化，请先调用 set_client()")
    return _llm_semaphore


def extract_text(response) -> str:
    """从 Anthropic response 中提取第一个 TextBlock 的文本。
    DeepSeek 会在 content[0] 返回 ThinkingBlock（reasoning），
    真正的文本在后续的 type='text' 块里。
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    return ""
