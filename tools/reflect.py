"""
reflect.py — critique_section 工具
子 LLM 评审章节内容；结果写入 sections[title].review 和 status。
读取 final（若存在）否则读 draft 作为评审内容。
"""

import json
import re

from .registry import register, get_state, get_section_by_title, state_lock
from ._client import get_client, get_config
from .errors import ErrorType, err
from .retry import call_llm_with_retry, parse_json_with_retry, _extract_text

SCHEMA = {
    "name": "critique_section",
    "description": (
        "评审已生成章节内容的质量和完整性，给出结构化反馈。"
        "必须在每轮 save_section 之后立即调用（search→save→critique 是一个完整循环，不得跳过）。"
        "若返回 is_sufficient=false，根据 gaps 进行下一轮补搜，然后再次 save_section，再次 critique。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section_title": {
                "type": "string",
                "description": "章节标题"
            },
            "iteration": {
                "type": "integer",
                "description": "当前第几次反思，从 1 开始"
            }
        },
        "required": ["section_title", "iteration"]
    }
}

_SYSTEM = """你是一个研究内容评审专家。
给定章节的质量标准和已生成的章节内容，评审内容是否充分：
1. 章节内容是否覆盖了标准要求的所有维度
2. 是否存在可通过定向搜索补充的信息空白
3. 关键数字/事实/结论是否都已用 [n] 标注来源；若存在重要声明缺少引用，列入 gaps

重要原则：
- 只在用户实际研究问题的范围内评判——不要要求用户没有提问的信息维度
- 如果内容已经回答了用户的问题，即便不够全面也应判为 is_sufficient=true
- 不要因"缺少某百科信息"而反复标记 needs_more_search
- 这是多章节报告中的一章，评审时只看本章标题所划定的范围内是否充分；不要要求本章说明相邻章节的内容、版本是否过时、或向读者提示其他章节

needs_search 字段说明（is_sufficient=false 时必填）：
- true：gaps 中包含需要从外部搜索才能获取的信息（如最新数据、具体事件、尚未知晓的细节）
- false：gaps 都是可以基于现有搜索结果和模型知识直接修复的问题（如内容组织不当、细节遗漏、排序错误、逻辑不清晰）

输出规则（严格遵守）：
- 只输出一个 JSON 对象，不加任何前缀、后缀或代码块标记
- 直接输出 { 开头的 JSON

JSON 字段：
  "is_sufficient": true/false
  "needs_search": true/false    is_sufficient=false 时必填；true=需搜索，false=可直接修复
  "quality_score": 0.0-1.0
  "gaps": ["缺失点1"]            is_sufficient=false 时必须非空，说明具体缺什么
  "improvement_suggestions": ["建议1"]   needs_search=true 时给出搜索方向；false 时给出直接修复建议
  "stop_reason": "sufficient" 或 "needs_more_search" 或 "can_fix_directly"

示例（需搜索）：
{"is_sufficient": false, "needs_search": true, "quality_score": 0.5, "gaps": ["缺少2023年后数据"], "improvement_suggestions": ["搜索2023-2024行业报告"], "stop_reason": "needs_more_search"}

示例（可直接修）：
{"is_sufficient": false, "needs_search": false, "quality_score": 0.75, "gaps": ["角色发布顺序排列有误"], "improvement_suggestions": ["按官方发布时间重新排序"], "stop_reason": "can_fix_directly"}

示例（充分）：
{"is_sufficient": true, "needs_search": false, "quality_score": 0.88, "gaps": [], "improvement_suggestions": [], "stop_reason": "sufficient"}"""


def runner(section_title: str, iteration: int) -> dict:
    cfg = get_config()
    state = get_state()
    sections = state["sections"]

    _, sec = get_section_by_title(section_title)
    if sec is None:
        all_titles = [e.get("title", k) for k, e in sections.items()]
        return {"error": f"章节 '{section_title}' 不在规划中，已规划章节：{all_titles}"}

    with state_lock:
        sec["reflect_count"] += 1
        count = sec["reflect_count"]

    if count > cfg.MAX_REFLECT_PER_SECTION:
        result = {
            "is_sufficient": True,
            "needs_search": False,
            "quality_score": 1.0,
            "gaps": [],
            "improvement_suggestions": [],
            "stop_reason": "sufficient",
            "note": f"已达最大反思次数 {cfg.MAX_REFLECT_PER_SECTION}，强制通过",
            "iteration": count,
        }
        sec["review"] = {k: result[k] for k in ("is_sufficient","needs_search","quality_score","gaps","improvement_suggestions")}
        with state_lock:
            sec["status"] = "done"
        return result

    # 读取内容：final 优先，否则 draft
    content = sec["final"] if sec["final"] is not None else sec.get("draft", "")
    if not content:
        return {"error": f"章节 '{section_title}' 尚无内容，请先调用 plan_sections 写入草稿或 save_section 生成内容"}

    criteria = sec.get("criteria") or f"需要全面覆盖'{section_title}'的核心内容，包含具体数据和案例"

    extra_hint = (
        "（第一次评审：请先推导该章节的具体考察维度，再对照标准评估）"
        if iteration == 1
        else f"（第 {iteration} 次评审：请重点检查上一轮指出的缺失点是否已补充）"
    )

    original_query = state.get("original_query", "")
    scope_line = f"用户原始问题：{original_query}\n\n" if original_query else ""

    prompt = (
        f"{scope_line}"
        f"章节标题：{section_title}\n\n"
        f"评估标准：\n{criteria}\n\n"
        f"已生成的章节内容：\n{content}\n\n"
        f"{extra_hint}\n\n"
        "请以 JSON 格式返回评估结果。"
    )
    messages = [{"role": "user", "content": prompt}]

    client = get_client()
    llm_result = call_llm_with_retry(
        client,
        tool_name="critique_section",
        max_tokens=4000,
        max_tokens_ceiling=8000,
        model=cfg.CRITIQUE_MODEL,
        system=_SYSTEM,
        messages=messages,
    )
    if not llm_result["ok"]:
        e = llm_result["error"]
        if e.get("type") == ErrorType.LLM_MAX_TOKENS.value:
            result = {
                "is_sufficient": False, "needs_search": True,
                "quality_score": 0.0, "gaps": ["子LLM输出超限，保守判为不足"],
                "improvement_suggestions": [], "stop_reason": "llm_max_tokens",
                "note": "子LLM输出超限，已使用保守兜底",
                "iteration": count,
            }
            with state_lock:
                sec["review"] = {k: result[k] for k in
                                  ("is_sufficient","needs_search","quality_score","gaps","improvement_suggestions")}
                sec["status"] = "needs_search"
            return result
        return {"error": f"critique 子LLM失败（{e['type']}）: {e['message']}"}

    raw = _extract_text(llm_result["data"])

    parse_result = parse_json_with_retry(
        raw,
        _try_parse_result,
        tool_name="critique_section",
        client=client,
        model=cfg.CRITIQUE_MODEL,
        system=_SYSTEM,
        original_messages=messages,
        max_tokens=4000,
    )

    if not parse_result["ok"]:
        print(f"   [critique fallback] 子LLM原始输出（前300字）：{raw[:300]!r}")
        result = {
            "is_sufficient": False,
            "needs_search": True,
            "quality_score": 0.3,
            "gaps": [f"评审反馈解析失败，请重新评审（error_id={parse_result['error'].get('error_id','')}）"],
            "improvement_suggestions": [],
            "stop_reason": "needs_more_search",
            "parse_error": True,
        }
    else:
        result = parse_result["data"]

    result["iteration"] = count

    # 写入 state
    with state_lock:
        sec["review"] = {
            "is_sufficient":          result.get("is_sufficient", False),
            "needs_search":           result.get("needs_search", True),
            "quality_score":          result.get("quality_score", 0.0),
            "gaps":                   result.get("gaps", []),
            "improvement_suggestions": result.get("improvement_suggestions", []),
        }
        if result.get("is_sufficient"):
            sec["status"] = "done"
        elif result.get("needs_search", True):
            sec["status"] = "needs_search"
        else:
            sec["status"] = "needs_fix"

    # next_step hint
    broad_done = sum(1 for m in state.get("search_sources", {}).values() if not m.get("section_title"))
    if result.get("is_sufficient"):
        result["next_step"] = "该章节已充分，调用 save_section 保存"
    elif broad_done == 0:
        result["next_step"] = "完成所有章节草稿评估后，根据 gaps 发起宽泛搜索（6-8 个并行 search，不指定 section_title）"
    else:
        result["next_step"] = f"根据 gaps 发起定向补搜：search(query=<具体缺失点>, section_title='{section_title}')"

    return result


def _try_parse_result(text: str) -> dict | None:
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
