"""
extract_evidence.py — extract_evidence 工具
子 LLM 调用：读取指定 source_ids 对应的磁盘搜索文件，去重归纳，
产出结构化 claims，落盘并更新 state["evidence"]。
"""

import json
import os
import re

from .registry import register, get_state
from ._client import get_client, get_config, extract_text

SCHEMA = {
    "name": "extract_evidence",
    "description": (
        "从已完成的搜索结果中提取结构化证据（claims）。"
        "在完成一个章节的搜索后调用，对搜索结果去重归纳，"
        "生成可索引的 claim 列表供后续 critique_section 和 save_section 使用。"
        "多次调用会累积同一章节的 evidence。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section_title": {
                "type": "string",
                "description": "章节标题，与 plan_sections 中一致"
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "本次要处理的 source_id 列表（来自 search 工具的返回值）"
            }
        },
        "required": ["section_title", "source_ids"]
    }
}

_SYSTEM = """你是一个研究信息提取专家。
给定一组搜索结果，为指定章节提取结构化证据（claims）。

要求：
- 每条 claim 是一个独立的、可核实的事实陈述
- 去除重复信息，合并来源不同但内容相同的表述
- detail 补充具体数字、时间、人名等关键细节
- source_ids 列出支持该 claim 的来源编号

输出规则（严格遵守）：
- 只输出一个 JSON 对象，不加任何前缀、后缀或代码块标记
- 格式：{"claims": [...]}
- 每条 claim 结构：{"claim": "...", "detail": "...", "source_ids": ["src_xxx"]}

示例：
{"claims": [
  {"claim": "2020年中国剧本杀市场规模约100亿元", "detail": "同比增长约30%，门店数量超3万家", "source_ids": ["src_a1b2"]},
  {"claim": "2022年受疫情影响门店数量下降至2万家", "detail": "部分头部品牌转型线上", "source_ids": ["src_c3d4", "src_e5f6"]}
]}"""


def runner(section_title: str, source_ids: list) -> dict:
    state = get_state()
    cfg = get_config()
    client = get_client()

    # 读取磁盘上的搜索原文
    search_texts = []
    missing = []
    for src_id in source_ids:
        entry = state["search_sources"].get(src_id)
        if not entry:
            missing.append(src_id)
            continue
        try:
            with open(entry["file"], encoding="utf-8") as f:
                data = json.load(f)
            snippets = "\n".join(
                f"[{r['title']}] {r['snippet']}"
                for r in data.get("results", [])
            )
            search_texts.append(
                f"--- {src_id} (query: {data['query']}) ---\n{snippets}"
            )
        except (OSError, json.JSONDecodeError) as e:
            missing.append(f"{src_id}(读取失败:{e})")

    if not search_texts:
        return {"error": f"所有 source_ids 均无法读取: {source_ids + missing}"}

    # 获取已有 evidence（避免和之前的 claim 重复）
    evidence_pool = state.get("evidence", {})
    existing = evidence_pool.get(section_title, [])
    existing_summary = ""
    if existing:
        existing_summary = "\n已有 claims（避免重复）：\n" + "\n".join(
            f"- {c['claim']}" for c in existing
        )

    prompt = (
        f"章节：{section_title}\n\n"
        f"搜索内容：\n{''.join(search_texts)}\n"
        f"{existing_summary}\n\n"
        "请提取结构化证据，以 JSON 格式返回。"
    )

    response = client.messages.create(
        model=cfg.CRITIQUE_MODEL,
        max_tokens=12000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = extract_text(response)
    if not raw:
        return {"error": f"子 LLM 输出为空（stop_reason={response.stop_reason}），可能是推理链耗尽 max_tokens，请重试"}
    new_claims = _parse_claims(raw)

    # 合并到 state（懒初始化，不依赖 reset_state 中的 evidence 字段）
    if "evidence" not in state:
        state["evidence"] = {}
    if section_title not in state["evidence"]:
        state["evidence"][section_title] = []
    state["evidence"][section_title].extend(new_claims)

    # 落盘
    _write_evidence(section_title, state["evidence"][section_title], cfg)

    top = [c["claim"] for c in new_claims[:3]]
    return {
        "section_title":  section_title,
        "new_claims":     len(new_claims),
        "total_claims":   len(state["evidence"][section_title]),
        "top_claims":     top,
        "missing_sources": missing if missing else None,
    }


def _parse_claims(text: str) -> list:
    # 1. 代码块包裹
    fenced = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if fenced:
        try:
            return json.loads(fenced.group(1)).get("claims", [])
        except json.JSONDecodeError:
            pass

    # 2. 裸 JSON 对象
    brace = re.search(r'\{[\s\S]*\}', text)
    if brace:
        try:
            return json.loads(brace.group()).get("claims", [])
        except json.JSONDecodeError:
            pass

    return []


def _write_evidence(section_title: str, claims: list, cfg) -> None:
    evidence_dir = os.path.join(os.path.dirname(cfg.OUTPUT_DIR), "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    safe = re.sub(r'[^\w\s-]', '', section_title).strip().replace(' ', '_')[:40]
    path = os.path.join(evidence_dir, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"section_title": section_title, "claims": claims},
                  f, ensure_ascii=False, indent=2)


# register(SCHEMA, runner)  # 已停用：改为 save_section 写作时顺手标 [n]，不做事后抽取
