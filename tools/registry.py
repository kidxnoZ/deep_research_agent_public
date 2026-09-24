"""
registry.py
-----------
工具注册中心。负责：
  - 工具注册（schema + runner 绑定）
  - 入参校验
  - 工具 dispatch
  - 运行时状态管理（跨工具共享）
"""
import threading

_tools: dict = {}
state_lock = threading.Lock()   # 保护所有 state 写入，工具并行时使用

# ── 引用编号（ref_idx）全局计数器 ──────────────────────────────────────────────
_ref_url_map: dict = {}   # url → ref_idx (int)，URL 去重
_ref_counter: list = [0]  # 用列表包一层以支持函数内修改


def assign_ref_idx(url: str) -> int:
    """给 URL 分配唯一引用编号；同一 URL 多次调用返回相同编号。线程安全。"""
    if not url:
        return 0
    with state_lock:
        if url not in _ref_url_map:
            _ref_counter[0] += 1
            _ref_url_map[url] = _ref_counter[0]
        return _ref_url_map[url]


def get_ref_url_map() -> dict:
    """返回当前 url → ref_idx 映射的快照。"""
    return dict(_ref_url_map)

_TYPE_MAP = {
    "string":  str,
    "integer": int,
    "number":  (int, float),
    "boolean": bool,
    "array":   list,
    "object":  dict,
    "null":    type(None),
}

_agent_retry_counts: dict = {}
MAX_AGENT_PARAM_RETRIES = 2


def _validate_types(tool_input: dict, input_schema: dict) -> list:
    errors = []
    properties = input_schema.get("properties", {})
    for field, value in tool_input.items():
        if field not in properties:
            continue
        expected = properties[field].get("type")
        if not expected:
            continue
        py_type = _TYPE_MAP.get(expected)
        if py_type is None:
            continue
        if not isinstance(value, py_type):
            errors.append(
                f"字段 '{field}' 类型错误：期望 {expected}，"
                f"实际 {type(value).__name__}（值：{repr(value)[:60]}）"
            )
    return errors


def _make_section_entry(description: str = "", initial_query: str = "", order: int = 0, title: str = "") -> dict:
    """创建一个空白章节条目。"""
    return {
        "title":         title,   # 人类可读标题（key 为 sec_NNN，稳定不变）
        "description":   description,
        "initial_query": initial_query,
        "order":         order,
        "criteria":      "",      # initialize_criteria 写入
        "draft":         "",      # plan_sections 写入，不再覆盖
        "source_ids":    [],      # 已整合进 final 的历史 src_ids
        "pending_src_ids": [],    # 已搜索但尚未被 save 整合的 src_ids
        "review":        None,    # 最新 critique 结果，None 表示未评审
        "final":         None,    # save_section 生成内容，None 表示未保存
        "reflect_count": 0,
        "citations":     {},    # {ref_idx(int): url} — save_section 写作时解析 [n] 填入
        # draft | pending_save | saved | needs_search | needs_fix | done
        "status":        "draft",
    }


def _next_sec_key() -> str:
    """生成下一个章节 key，格式 sec_001 / sec_002 …"""
    existing = _state["sections"]
    if not existing:
        return "sec_001"
    nums = [int(k[4:]) for k in existing if k.startswith("sec_") and k[4:].isdigit()]
    return f"sec_{(max(nums) + 1 if nums else 1):03d}"


def get_section_by_title(title: str):
    """按标题查找章节，返回 (key, entry) 或 (None, None)。"""
    for key, entry in _state["sections"].items():
        if entry.get("title") == title:
            return key, entry
    return None, None


_state: dict = {
    # ── 以 section 为单位 ───────────────────────────────────────────────────
    "sections": {},               # {title: section_entry}

    # ── 全局字段 ────────────────────────────────────────────────────────────
    "original_query":     "",
    "run_dir":            "",
    "broad_locked":       False,
    "broad_pending_lock": False,
    # 宽泛搜索失败过的 query（broad_locked 后允许对失败条目重试一次，重试时消费）
    "broad_failed_queries": set(),
    # 所有搜索的元数据索引：{src_id: {file, query, section_title, summary, result_count}}
    # section_title=="" 表示宽泛搜索；有值表示定向搜索
    "search_sources": {},
    "temporal":        None,   # get_current_date 写入；None 表示无时效性要求
    # 已 ingest 的文档索引：{doc_id: {source, source_type, title, path, chunks, outline, has_tables}}
    "documents":       {},
}


def register(schema: dict, runner) -> None:
    _tools[schema["name"]] = {"schema": schema, "runner": runner}


def get_schemas() -> list:
    return [t["schema"] for t in _tools.values()]


def dispatch(tool_name: str, tool_input: dict) -> dict:
    if tool_name not in _tools:
        return {"error": f"未知工具: {tool_name}。已注册工具: {list(_tools.keys())}"}

    tool = _tools[tool_name]
    input_schema = tool["schema"].get("input_schema", {})

    required = input_schema.get("required", [])
    missing = [f for f in required if f not in tool_input]
    if missing:
        return {"error": f"缺少必填字段: {missing}"}

    type_errors = _validate_types(tool_input, input_schema)
    if type_errors:
        return {"error": f"参数类型不合法: {type_errors}"}

    try:
        result = tool["runner"](**tool_input)
    except TypeError as e:
        return {"error": f"入参错误: {e}"}
    except Exception as e:
        return {"error": f"工具执行失败: {type(e).__name__}: {e}"}

    if isinstance(result, dict) and not result.get("ok", True):
        error_info = result.get("error", {})
        if isinstance(error_info, dict) and error_info.get("type") == "param_error":
            sig = f"{tool_name}:{sorted(tool_input.keys())}"
            count = _agent_retry_counts.setdefault(tool_name, {})
            count[sig] = count.get(sig, 0) + 1
            if count[sig] > MAX_AGENT_PARAM_RETRIES:
                error_info["agent_retries_exhausted"] = True
            return result

    return result


def get_state() -> dict:
    return _state


def get_section(title: str) -> dict | None:
    """获取指定章节的 entry（按 title 查找），不存在返回 None。"""
    return get_section_by_title(title)[1]


def reset_state() -> None:
    _state["sections"]            = {}
    _state["original_query"]      = ""
    _state["run_dir"]             = ""
    _state["broad_locked"]        = False
    _state["broad_pending_lock"]  = False
    _state["broad_failed_queries"] = set()
    _state["search_sources"]      = {}
    _state["temporal"]            = None
    _state["documents"]           = {}
    _state.pop("evidence", None)
    _agent_retry_counts.clear()
    # 清空引用编号计数器
    _ref_url_map.clear()
    _ref_counter[0] = 0
