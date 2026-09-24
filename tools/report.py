"""
report.py — save_section + compile_report 工具
save_section:    首次调用用全量搜索结果生成章节内容；后续调用为增量 patch 模式，
                 只整合 pending_src_ids 中的新搜索结果，在现有 final 上定点修改。
compile_report:  代码拼接所有章节 final，轻量 LLM 只生成章节间过渡句和总结，
                 由代码插入并修正标题层级，写入 Markdown 文件。
"""

import json
import os
import re

from .registry import register, get_state, get_section_by_title, state_lock, get_ref_url_map
from ._client import get_client, get_config
from .errors import ErrorType, err
from .retry import call_llm_with_retry, parse_json_with_retry, _extract_text

# ─── save_section ────────────────────────────────────────────────────────────

SAVE_SCHEMA = {
    "name": "save_section",
    "description": (
        "根据本章节的搜索结果生成或更新章节内容并保存。\n"
        "首次调用：综合全部搜索结果生成完整章节。\n"
        "后续调用：在已有内容基础上定点补充新搜索结果，修复 critique 指出的缺口。\n"
        "每轮补搜完成后立即调用，然后紧接着调用 critique_section 评审。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string",  "description": "章节标题，必须与 plan_sections 中一致"},
            "order": {"type": "integer", "description": "章节序号，从 1 开始，按规划顺序"},
            "critique_feedback": {
                "type": "string",
                "description": (
                    "（可选）覆盖自动读取的审核意见。"
                    "不填时工具自动读取 state 中最新的 critique 结果。"
                )
            }
        },
        "required": ["title", "order"]
    }
}

_HEADING_RULE = (
    "\n章节内部子标题只使用 Markdown 格式：### 为一级子标题，#### 为二级子标题。"
    "\n禁止使用中文序号（一、（一））或数字序号（1.）作为标题，正文序号列表用 Markdown 列表（- 或 1.）。"
)

_CITE_RULE = (
    "\n引用规则（严格执行，错误引用会扣分）："
    "只能给搜索结果 snippet 里明确陈述的内容标引用 [n]（n 来自搜索结果行首编号）。"
    "凭训练知识、常识或推断写的内容一律不标引用。"
    "视频转录、arxiv 等碎片化来源，只引用其明确提到的具体数字和结论，禁止据此展开成完整技术表述。"
    "不确定某内容是否来自该来源时，宁可不标。"
    "格式：...增长 30% [3]，可同时引多源：...已有记录 [1][5]。"
)

_WRITE_SYSTEM = (
    "你是一位专业研究员。"
    "根据给定的搜索结果（以及可能提供的章节初稿），撰写结构清晰、内容完整的研究章节。\n"
    "要求：\n"
    "- 有初稿时：用搜索结果更新、补充、纠正初稿；特别保留训练截止后的新信息\n"
    "- 无初稿时：综合搜索结果撰写章节内容\n"
    "- 搜索结果优先（可核实、可更新），忽略明显不相关或错误的内容\n"
    "- 保留具体数字、时间、人名等关键细节\n"
    "- 涉及多主体对比、多维度数据或数值序列（价格、份额、排名、增长率、金额、占比等）时，"
    "用 Markdown 表格呈现数据，并标注数据年份与统计口径\n"
    "- 表格只负责呈现数据，分析和论证必须用完整段落展开：每个表格前后要有解释其含义、"
    "比较结论的文字，不得用表格替代论述，也不得堆砌无解释的表格\n"
    "- 中文输出，技术术语保留英文原名\n"
    "- 只输出章节正文，不加标题行"
    + _HEADING_RULE
    + _CITE_RULE
)

_PATCH_SYSTEM = (
    "你是一位专业研究员，正在修订一篇已有内容的研究章节。\n"
    "任务：根据新增搜索结果和审核反馈，对现有章节进行定点修改。\n"
    "要求：\n"
    "- 只修改或补充审核反馈中指出的缺口，其余内容保持原样\n"
    "- 新增内容插入逻辑位置，不打乱整体结构\n"
    "- 保留具体数字、时间、人名等关键细节\n"
    "- 涉及多主体对比、多维度数据或数值序列（价格、份额、排名、增长率、金额、占比等）时，"
    "用 Markdown 表格呈现数据，并标注数据年份与统计口径\n"
    "- 表格只负责呈现数据，分析和论证必须用完整段落展开：每个表格前后要有解释其含义、"
    "比较结论的文字，不得用表格替代论述，也不得堆砌无解释的表格\n"
    "- 中文输出，技术术语保留英文原名\n"
    "- 输出完整修订后的章节正文，不加标题行"
    + _HEADING_RULE
    + _CITE_RULE
)


def save_runner(title: str, order: int, critique_feedback: str = "") -> dict:
    state = get_state()
    sections = state["sections"]

    _, sec = get_section_by_title(title)
    if sec is None:
        all_titles = [e.get("title", k) for k, e in sections.items()]
        return {"error": f"章节 '{title}' 不在规划列表中。已规划章节：{all_titles}"}

    sec["order"] = order

    # 自动从 state 读取 critique_feedback
    if not critique_feedback and sec.get("review") and not sec["review"].get("is_sufficient"):
        review = sec["review"]
        gaps = review.get("gaps", [])
        suggestions = review.get("improvement_suggestions", [])
        if gaps or suggestions:
            critique_feedback = f"gaps: {gaps}\nimprovement_suggestions: {suggestions}"

    is_first_save = sec["final"] is None

    # ── 决定本次使用哪些 src_ids ────────────────────────────────────────────
    search_sources  = state.get("search_sources", {})
    pending_ids     = list(sec.get("pending_src_ids", []))
    broad_src_ids   = [sid for sid, m in search_sources.items() if not m.get("section_title")]

    if is_first_save:
        # 首次：全量（pending + broad）
        used_ids = pending_ids + [s for s in broad_src_ids if s not in pending_ids]
    else:
        # 增量：仅用 pending（新搜索结果）
        used_ids = pending_ids

    # ── 无搜索结果时直接保存初稿 ─────────────────────────────────────────────
    # is_fix_only：patch 模式下 pending 为空但有 critique_feedback
    # 对应 needs_fix 路径（needs_search=false）或补搜达上限后直接修复
    is_fix_only = not is_first_save and not used_ids and bool(critique_feedback)

    if not used_ids:
        if is_first_save:
            existing_draft = sec.get("draft", "")
            if existing_draft:
                sec["final"]  = existing_draft
                sec["status"] = "saved"
                _write_sections_checkpoint(state)
                total = len(sections)
                done  = sum(1 for s in sections.values() if s["final"] is not None)
                return {
                    "saved": True, "preview": existing_draft[:150],
                    "sections_completed": done, "sections_total": total,
                    "remaining": total - done,
                    "note": "无搜索结果，直接保存初稿",
                }
            return {"error": f"章节 '{title}' 尚无搜索结果且无初稿，请先调用 search"}
        elif not critique_feedback:
            return {"error": f"章节 '{title}' 无新搜索结果（pending 为空）且无审核反馈，无需 save"}

    # ── 读取搜索文件 ──────────────────────────────────────────────────────────
    search_blocks = []
    for src_id in used_ids:
        entry = search_sources.get(src_id)
        if not entry:
            continue
        try:
            with open(entry["file"], encoding="utf-8") as f:
                data = json.load(f)
            snippets = "\n".join(
                (f"[{r['ref_idx']}] " if r.get("ref_idx") else "") +
                f"[{r['title']}]({r.get('url', '')}) {r['snippet']}"
                for r in data.get("results", [])
            )
            search_blocks.append(f"--- {src_id} (query: {data['query']}) ---\n{snippets}")
        except (OSError, json.JSONDecodeError):
            continue

    if not search_blocks and not is_fix_only:
        return {"error": f"章节 '{title}' 的搜索文件无法读取，src_ids={used_ids}"}

    client = get_client()
    cfg    = get_config()
    search_text = "\n\n".join(search_blocks)

    # ── 构建 prompt ───────────────────────────────────────────────────────────
    if is_first_save:
        existing_draft = sec.get("draft", "")
        draft_section = (
            f"## 章节初稿（基于训练知识，请用搜索结果完善）\n{existing_draft}\n\n"
            if existing_draft else ""
        )
        task_instruction = (
            "请结合搜索结果完善和更新初稿，特别补充截止日期后的新内容。"
            if existing_draft else "请撰写该章节内容。"
        )
        feedback_section = (
            f"\n## 审核反馈（请针对以下问题进行修复）\n{critique_feedback}\n"
            if critique_feedback else ""
        )
        prompt = (
            f"章节：{title}\n\n"
            f"{draft_section}"
            f"搜索结果（共 {len(search_blocks)} 组）：\n{search_text}\n"
            f"{feedback_section}\n"
            f"{task_instruction}"
        )
        system = _WRITE_SYSTEM
    else:
        feedback_section = (
            f"\n## 审核反馈（需修复的缺口）\n{critique_feedback}\n"
            if critique_feedback else ""
        )
        search_section = (
            f"## 新增搜索结果（共 {len(search_blocks)} 组）\n{search_text}\n\n"
            if search_blocks else ""
        )
        prompt = (
            f"章节：{title}\n\n"
            f"## 现有章节内容\n{sec['final']}\n\n"
            f"{search_section}"
            f"{feedback_section}\n"
            "请在现有内容基础上定点修改，输出完整修订后的章节正文。"
        )
        system = _PATCH_SYSTEM

    messages = [{"role": "user", "content": prompt}]
    llm_result = call_llm_with_retry(
        client, tool_name="save_section",
        max_tokens=20000, max_tokens_ceiling=40000,
        model=cfg.CRITIQUE_MODEL, system=system, messages=messages,
    )
    if not llm_result["ok"]:
        e = llm_result["error"]
        if e.get("type") == ErrorType.LLM_MAX_TOKENS.value:
            partial = _extract_text(llm_result.get("partial_response")) if llm_result.get("partial_response") else ""
            content = partial if partial else sec.get("draft", "")
            if content:
                note = "子LLM输出超限，已保存截断内容" if partial else "子LLM输出超限，已回退保存初稿"
            else:
                return {"error": f"save_section 子LLM输出超限且无可用内容"}
        else:
            return {"error": f"save_section 子LLM失败（{e['type']}）: {e['message']}"}
    else:
        content = _extract_text(llm_result["data"])
        note = None
        if not content:
            return {"error": "子 LLM 输出为空且重试耗尽，请重试"}

    with state_lock:
        sec["final"]  = content
        sec["status"] = "saved"
        sec["source_ids"].extend(pending_ids)
        sec["pending_src_ids"] = []
        # 解析正文中的 [n] 引用，建立 ref_idx → url 映射存入章节
        cited_nums = {int(m) for m in re.findall(r'\[(\d+)\]', content)}
        if cited_nums:
            ref_map = get_ref_url_map()          # url → ref_idx
            ref_to_url = {v: k for k, v in ref_map.items()}
            sec["citations"] = {n: ref_to_url[n] for n in cited_nums if n in ref_to_url}

    _write_sections_checkpoint(state)

    total = len(sections)
    done  = sum(1 for s in sections.values() if s["final"] is not None)
    return {
        "saved":              True,
        "mode":               "full" if is_first_save else ("fix" if is_fix_only else "patch"),
        "preview":            content[:150],
        "sections_completed": done,
        "sections_total":     total,
        "remaining":          total - done,
        **({"note": note} if note else {}),
    }


def _write_sections_checkpoint(state: dict) -> None:
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
                    "draft":    sec["draft"][:200] + "…" if len(sec.get("draft","")) > 200 else sec.get("draft",""),
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


# ─── compile_report ───────────────────────────────────────────────────────────

COMPILE_SCHEMA = {
    "name": "compile_report",
    "description": (
        "汇总所有已保存章节，生成最终 Markdown 报告并写入文件。"
        "代码负责拼接章节内容和修正标题层级；LLM 只生成章节间过渡句和末尾总结。"
        "只有在所有章节都已 save_section 后才能调用。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "report_title": {"type": "string", "description": "报告标题"}
        },
        "required": ["report_title"]
    }
}

_COMPILE_SYSTEM = (
    "你是研究报告编辑。根据提供的章节列表（含各章节开头摘要），完成两个任务：\n"
    "1. 为每对相邻章节生成一句自然过渡句（衔接上一章结论与下一章主题）\n"
    "2. 写一段 150-200 字的总结，联系全文各章节的核心发现\n\n"
    "只输出纯 JSON，不加任何前缀、后缀或代码块标记：\n"
    "{\"transitions\": {\"章节A→章节B\": \"过渡句\", ...}, \"summary\": \"总结段\"}"
)


def compile_runner(report_title: str) -> dict:
    state    = get_state()
    sections = state["sections"]

    if not sections:
        return {"error": "没有已保存的章节，请先完成所有章节的 search → save → critique 循环"}

    missing = [sec["title"] for sec in sections.values() if sec["final"] is None]
    if missing:
        return {
            "error":     f"还有 {len(missing)} 个章节未完成：{missing}",
            "completed": [sec["title"] for sec in sections.values() if sec["final"] is not None],
        }

    sorted_sections = sorted(sections.values(), key=lambda s: s["order"])

    # ── 轻量 LLM：只传标题 + 每章前 150 字 ──────────────────────────────────
    client = get_client()
    cfg    = get_config()

    chapter_previews = "\n".join(
        f"{i+1}. 【{sec['title']}】{(sec['final'] or '')[:150]}"
        for i, sec in enumerate(sorted_sections)
    )
    compile_messages = [{"role": "user", "content": (
        f"报告《{report_title}》共 {len(sorted_sections)} 章：\n\n{chapter_previews}\n\n"
        "请生成章节间过渡句和总结。"
    )}]

    transitions = {}
    summary = ""
    llm_result = call_llm_with_retry(
        client, tool_name="compile_report",
        max_tokens=8000, max_tokens_ceiling=16000,
        model=cfg.CRITIQUE_MODEL, system=_COMPILE_SYSTEM, messages=compile_messages,
    )
    if llm_result["ok"]:
        raw = _extract_text(llm_result["data"])
        parse_result = parse_json_with_retry(
            raw, _try_parse_compile,
            tool_name="compile_report", client=client,
            model=cfg.CRITIQUE_MODEL, system=_COMPILE_SYSTEM,
            original_messages=compile_messages, max_tokens=800,
        )
        if parse_result["ok"]:
            data = parse_result["data"]
            transitions = data.get("transitions", {})
            summary     = data.get("summary", "")

    # ── 代码拼接报告 ──────────────────────────────────────────────────────────
    report_parts = [f"# {report_title}\n"]

    for i, sec in enumerate(sorted_sections):
        # 过渡句（第一章前不加）
        if i > 0:
            prev_title = sorted_sections[i - 1]["title"]
            key = f"{prev_title}→{sec['title']}"
            transition = transitions.get(key, "")
            if transition:
                report_parts.append(f"\n{transition}")

        # 章节标题（代码统一加 ##）
        report_parts.append(f"\n## {sec['title']}\n")

        # 正文：去掉 LLM 可能在开头多写的重复标题行（与 state 标题一致时才删）
        content = sec["final"] or ""
        content = re.sub(
            r'^#{1,4}\s*' + re.escape(sec["title"]) + r'\s*\n?',
            '', content, count=1, flags=re.MULTILINE
        ).lstrip('\n')
        # 章节内部 ## → ###（LLM 误输出章节同级标题时的兜底）
        # ###/#### 保持原样：系统 prompt 已约定 ###=一级子标题、####=二级子标题
        content = re.sub(r'^##(?!#)', '###', content, flags=re.MULTILINE)
        report_parts.append(content)

    # 总结
    if summary:
        report_parts.append(f"\n## 总结\n\n{summary}\n")

    report = _add_heading_numbers("\n".join(report_parts))

    # ── 参考文献表 ────────────────────────────────────────────────────────────
    url_to_title = _build_url_title_map(state)
    all_refs: dict[int, dict] = {}
    for sec in sorted_sections:
        for ref_idx, url in sec.get("citations", {}).items():
            idx = int(ref_idx)
            if url and idx not in all_refs:
                all_refs[idx] = {"url": url, "title": url_to_title.get(url, "")}
    if all_refs:
        ref_lines = ["\n\n## 参考文献\n"]
        for idx in sorted(all_refs):
            info = all_refs[idx]
            title_part = f"{info['title']} | " if info["title"] else ""
            ref_lines.append(f"[{idx}] {title_part}{info['url']}")
        report += "\n".join(ref_lines) + "\n"

    # ── 写文件 ────────────────────────────────────────────────────────────────
    run_dir = state.get("run_dir", "")
    out_dir = run_dir if run_dir else cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r'[^\w\s-]', '', report_title).strip().replace(' ', '_')[:40]
    path = os.path.join(out_dir, f"report_{safe}.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "saved_path":     path,
        "section_count":  len(sorted_sections),
        "char_count":     len(report),
        "report_preview": report[:600],
    }


def _build_url_title_map(state: dict) -> dict:
    """从 search_sources 磁盘文件中建立 url → title 反查表，用于参考文献展示。"""
    url_to_title: dict = {}
    for entry in state.get("search_sources", {}).values():
        try:
            with open(entry["file"], encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                url = r.get("url", "")
                title = r.get("title", "")
                if url and title and url not in url_to_title:
                    url_to_title[url] = title
        except Exception:
            pass
    return url_to_title


def _add_heading_numbers(text: str) -> str:
    """为报告中 ##/###/#### 标题加层级序号（1. / 1.1. / 1.1.1.），# 标题不编号。"""
    counters = [0, 0, 0]  # 对应 ##, ###, ####
    lines = text.split('\n')
    out = []
    for line in lines:
        m = re.match(r'^(#{2,4})\s+(.*)', line)
        if m:
            hashes, title = m.group(1), m.group(2)
            level = len(hashes) - 2  # 0=##, 1=###, 2=####
            # 标题发生跳级时向上提升到最近的有效父级，避免 1.0.1 之类的编号。
            while level > 0 and counters[level - 1] == 0:
                level -= 1
                hashes = "#" * (level + 2)
            counters[level] += 1
            for i in range(level + 1, 3):
                counters[i] = 0
            num = '.'.join(str(counters[i]) for i in range(level + 1)) + '.'
            line = f"{hashes} {num} {title}"
        out.append(line)
    return '\n'.join(out)


def _try_parse_compile(text: str):
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


register(SAVE_SCHEMA, save_runner)
register(COMPILE_SCHEMA, compile_runner)
