"""
planner.py — plan_sections 工具
首次调用：记录章节规划，写每章节 draft_content（LLM 已有知识）到 sections[title].draft，初始化状态。
append=True：发现新结构后追加章节，不重置已有搜索结果和章节内容。
"""

import json
import os

from .registry import register, get_state, _make_section_entry, _next_sec_key, get_section_by_title
from ._client import get_config

SCHEMA = {
    "name": "plan_sections",
    "description": (
        "记录研究报告的章节规划。首次调用时初始化所有状态；"
        "发现新内容需要扩展结构时用 append=True 追加章节（不影响已有搜索结果和已写章节）。\n"
        "sections 中每项的 draft_content 字段：直接写出你对该章节已知内容的完整草稿——"
        "这将成为报告的第一版初稿，后续 save_section 会用搜索结果在此基础上完善。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "原始研究问题"},
            "sections": {
                "type": "array",
                "description": "章节规划列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":         {"type": "string"},
                        "description":   {"type": "string"},
                        "initial_query": {"type": "string"},
                        "draft_content": {"type": "string"}
                    },
                    "required": ["title", "description", "initial_query"]
                }
            },
            "append": {
                "type": "boolean",
                "description": (
                    "默认 false（首次调用）。"
                    "true = 追加模式：当宽泛搜索发现超出现有章节覆盖范围的内容时必须调用，"
                    "传入需要新增的章节，不影响已有章节和搜索结果。"
                )
            }
        },
        "required": ["query", "sections"]
    }
}


def runner(query: str, sections: list, append: bool = False) -> dict:
    state = get_state()
    existing = state["sections"]

    if not append:
        state["original_query"] = query
        state["sections"] = {}
        existing = state["sections"]

    draft_count = 0
    for i, s in enumerate(sections):
        title = s["title"]
        if get_section_by_title(title)[1] is not None:
            continue
        order = len(existing) + 1
        key   = _next_sec_key()
        entry = _make_section_entry(
            title=title,
            description=s.get("description", ""),
            initial_query=s.get("initial_query", ""),
            order=order,
        )
        draft = s.get("draft_content", "").strip()
        if draft:
            entry["draft"] = draft
            draft_count += 1
        existing[key] = entry

    if draft_count:
        _write_checkpoint(state)

    all_titles = [sec["title"] for sec in sorted(existing.values(), key=lambda s: s["order"])]

    if append:
        new_titles = [s["title"] for s in sections if get_section_by_title(s["title"])[1] is not None]
        return {
            "confirmed":    True,
            "new_sections": new_titles,
            "total_plan":   len(all_titles),
            "drafts_saved": draft_count,
            "next_step":    "请调用 initialize_criteria（只传新增章节标题），然后对新增章节调用 critique_section"
        }

    return {
        "confirmed":     True,
        "section_count": len(sections),
        "sections":      all_titles,
        "drafts_saved":  draft_count,
        "next_step":     "请调用 initialize_criteria，然后对每个章节并行调用 critique_section（iteration=1）评估草稿"
    }


def _write_checkpoint(state: dict) -> None:
    try:
        cfg = get_config()
        run_dir = state.get("run_dir", "")
        out_dir = run_dir if run_dir else cfg.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "sections_checkpoint.json")
        payload = {
            "sections": {
                key: {
                    "title":    sec["title"],
                    "order":    sec["order"],
                    "criteria": sec["criteria"],
                    "draft":    sec["draft"][:200] + "…" if len(sec["draft"]) > 200 else sec["draft"],
                    "final":    sec["final"],
                    "status":   sec["status"],
                }
                for key, sec in state.get("sections", {}).items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


register(SCHEMA, runner)
