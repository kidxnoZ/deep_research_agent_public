"""
criteria.py — initialize_criteria 工具
子 LLM 调用：根据用户 query 和规划章节，制定每个章节的质量评估标准。
结果写入 sections[title].criteria。
"""

import json
import re

from .registry import register, get_state, get_section_by_title, state_lock
from ._client import get_client, get_config
from .errors import ErrorType, err
from .retry import call_llm_with_retry, parse_json_with_retry, _extract_text

SCHEMA = {
    "name": "initialize_criteria",
    "description": (
        "根据研究问题和章节规划，制定每个章节的质量评估标准。"
        "在 plan_sections 之后、第一次 search 之前调用一次。"
        "标准将被 critique_section 用于评估每章节内容是否充分。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query":          {"type": "string", "description": "原始研究问题"},
            "section_titles": {"type": "array", "items": {"type": "string"}, "description": "章节标题列表"}
        },
        "required": ["query", "section_titles"]
    }
}

_SYSTEM = """你是一个研究质量专家。
给定一个研究问题和报告章节列表，为每个章节制定具体的质量评估标准。

重要原则：
- 标准必须严格对齐用户的研究问题，不要添加用户没有提及的维度
- 用户问的是什么，标准就衡量什么；不要变成百科全书式的过度要求
- 数据呈现原则：若章节涉及多主体对比、数值序列或量化指标（价格、份额、排名、增长率、金额、占比等），
  该章节标准中必须包含"关键数据用 Markdown 表格呈现，并标注年份与口径，且对数据有分析和解读"——这是对用户已要求内容的呈现形式约束，不新增维度

时效性原则（优先级最高）：
- 涉及"最新版本"、"当前"、"截止"等时效性词汇时，必须加入：**"需通过搜索确认当前最新版本"**
- 时效性信息不能依靠 draft_content 中的估算，必须由搜索结果核实

输出规则（严格遵守）：
- 只输出一个 JSON 对象，不加任何前缀、后缀、解释或代码块标记
- 键为章节标题，值为该章节的评估标准字符串"""


def runner(query: str, section_titles: list) -> dict:
    client = get_client()
    cfg = get_config()
    state = get_state()
    sections = state["sections"]

    missing_titles = []
    for t in section_titles:
        _, entry = get_section_by_title(t)
        if entry is not None and not entry["criteria"]:
            missing_titles.append(t)

    if not missing_titles:
        criteria_out = {}
        for t in section_titles:
            _, entry = get_section_by_title(t)
            if entry is not None:
                criteria_out[t] = entry["criteria"]
        return {"initialized": True, "criteria": criteria_out, "note": "所有章节已有标准，无需更新"}

    sections_text = "\n".join(f"- {t}" for t in missing_titles)
    temporal = state.get("temporal")
    if temporal:
        d = temporal["current_date"]
        temporal_ctx = (
            f"\n\n当前真实日期：{d}。"
            "请为每个章节独立判断：该章节内容是否需要覆盖到此日期"
            "（例如最新版本章节需要，1.0首发角色章节不需要）。"
            f"若需要，在该章节的 criteria 中明确写入：该章节需确认内容已覆盖至 {d}。"
        )
    else:
        temporal_ctx = ""
    messages = [{"role": "user", "content": (
        f"用户研究问题：{query}\n\n"
        f"需要制定标准的章节：\n{sections_text}\n\n"
        f"请为每个章节制定评估标准，以 JSON 对象返回。{temporal_ctx}"
    )}]

    llm_result = call_llm_with_retry(
        client, tool_name="initialize_criteria",
        max_tokens=10000, max_tokens_ceiling=20000,
        model=cfg.CRITIQUE_MODEL, system=_SYSTEM, messages=messages,
    )
    if not llm_result["ok"]:
        e = llm_result["error"]
        if e.get("type") == ErrorType.LLM_MAX_TOKENS.value:
            criteria_map = {t: f"需要全面覆盖'{t}'的核心内容，包含具体数据、案例或时间线，信息来源可靠"
                            for t in missing_titles}
            with state_lock:
                for title, criteria_text in criteria_map.items():
                    _, entry = get_section_by_title(title)
                    if entry is not None:
                        entry["criteria"] = criteria_text
            criteria_out = {t: criteria_map[t] for t in section_titles if t in criteria_map}
            existing = {t: get_section_by_title(t)[1]["criteria"]
                        for t in section_titles if get_section_by_title(t)[1] and get_section_by_title(t)[1].get("criteria")}
            criteria_out.update({k: v for k, v in existing.items() if k not in criteria_out})
            return {"initialized": True, "criteria": criteria_out,
                    "note": "子LLM输出超限，已使用默认标准"}
        return {"error": f"criteria 子LLM失败（{e['type']}）: {e['message']}"}

    raw = _extract_text(llm_result["data"])

    parse_result = parse_json_with_retry(
        raw, _try_parse_criteria,
        tool_name="initialize_criteria", client=client,
        model=cfg.CRITIQUE_MODEL, system=_SYSTEM,
        original_messages=messages, max_tokens=12000,
    )
    if not parse_result["ok"]:
        criteria_map = {t: f"需要全面覆盖'{t}'的核心内容，包含具体数据、案例或时间线，信息来源可靠"
                        for t in missing_titles}
    else:
        criteria_map = parse_result["data"]

    with state_lock:
        for title, criteria_text in criteria_map.items():
            _, entry = get_section_by_title(title)
            if entry is not None:
                entry["criteria"] = criteria_text

    criteria_out = {}
    for t in section_titles:
        _, entry = get_section_by_title(t)
        if entry is not None:
            criteria_out[t] = entry["criteria"]
    return {
        "initialized": True,
        "criteria": criteria_out,
        "next_step": "请对每个章节并行调用 critique_section（iteration=1）评估当前草稿，找出信息缺口"
    }


def _try_parse_criteria(text: str) -> dict | None:
    fenced = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r'\{[\s\S]*\}', text)
    if brace:
        try:
            return json.loads(brace.group())
        except json.JSONDecodeError:
            pass
    return None


register(SCHEMA, runner)
