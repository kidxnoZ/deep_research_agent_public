"""
query_rewriter.py — rewrite_query 工具
根据研究问题或 critique 缺口，生成优化后的候选检索词供主 agent 选择。
只读 state，不写 state。
"""

import json
import re

from .registry import register, get_state, get_section_by_title
from ._client import get_client, get_config
from .retry import call_llm_with_retry, parse_json_with_retry, _extract_text
SCHEMA = {
    "name": "rewrite_query",
    "description": (
        "在调用 search 之前，生成优化后的候选检索词，每条自带 source 字段（搜索源）。\n"
        "broad 模式（宽泛探索，调用一次）：生成与 MAX_BROAD_SEARCHES 等量的候选词，"
        "agent 将全部候选词并行调用 search，不需要挑选。\n"
        "targeted 模式（定向补搜，每章节每轮调一次）：根据 critique 信息缺口生成 3 条候选词，"
        "agent 从中选择 1-2 条最相关的调用 search。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": '"broad"（宽泛探索，broad_locked=False 时调用一次）或 "targeted"（定向补搜）',
            },
            "section_title": {
                "type": "string",
                "description": "targeted 模式必填，指定要补搜的章节标题",
            },
        },
        "required": ["mode"],
    },
}

_SYSTEM = """你是 Research Agent 的 Search Query Rewriter。

## 任务
根据输入信息设计检索词，并为每条检索词指定搜索源。

## 检索词设计原则
- broad 模式：从原始研究问题提炼，覆盖面最大化、用于探索，不过早针对单一缺口；输出条数由 prompt 指定
- targeted 模式：根据 critique 信息缺口，精准针对缺口设计，把泛化问题拆解为多个 focused sub-query，固定输出 3 条

1. 每条 query 应适合搜索引擎：保留真正影响结果的实体、指标、时间、专业术语
2. 涉及专业术语优先使用标准术语；必要时保留英文术语、缩写
3. 使用当前主题最合适的语言（中英文均可）
4. "最新"类需求用 latest/current + 时间范围；仅在需要比较明确版本时才枚举版本号
5. 各条候选词覆盖不同检索意图，不生成高度相似的 query

## 第一原则：换检索目标，而不是换词
已有：Generali 2025 annual results → Gap：缺 operating profit 和 dividend
✓ Generali 2025 operating profit dividend
✓ Generali 2025 annual report financial statements
✗ Generali 2025 full year results（仅换词，意图相同）

## 搜索源指定规则
tavily 是默认主源（通用网页搜索，覆盖事实/新闻/百科/政策等绝大多数场景）。
非 tavily 的垂直源是**限量辅助源**：本批最多标 1-3 条，只对最精准命中垂直域、信息增益最大的 query 标非 tavily 源，其余一律 tavily。
只有某条 query 明确落进以下垂直域时，才把该条的 source 指定为对应垂直源，否则一律 tavily：

| 垂直域 | 判定特征 | source |
|---|---|---|
| 生活经验 | 怎么做/避坑/口碑/消费决策/家长实操/个人体验 | xiaohongshu 或 zhihu |
| 学术研究 | 论文/研究进展/技术综述/学术观点 | arxiv / google_scholar（google_scholar 仅元数据无摘要，arxiv 有摘要） |
| 代码 | 开源项目/代码实现/框架用法/技术方案 | github |
| 实时舆情 | 热点事件/公众讨论/舆论走向 | weibo |
| 公众号文章 | 深度长文/公众号内容 | weixin |

注意：宁可 source=tavily 也不要硬套垂直源——query 不匹配任何垂直域时，强行用垂直源只会引入噪音。

## 各源 query 设计规则（按 source 定制语法）
不同搜索源对 query 的解析规则不同，指定 source 时必须按该源语法设计 query：
- github：gh search repos 把空格分隔的每个词按 AND 严格匹配（name/description 需同时含全部词）。
  只写 2-3 个核心词（如 "convex sets"、"graphs convex sets"），禁止自然语言长句（"xxx implementation examples" 必然 0 结果）。
- arxiv：API all 字段多词 AND。用 2-4 个核心学术术语（如 "graph of convex sets motion planning"），
  去掉 technical report、paper 等填充词。
- google_scholar：论文标题或 2-4 个核心术语。
- tavily 及社媒源（xiaohongshu/zhihu/weixin/weibo）：搜索引擎语义，自然语言或关键词短语均可。

## 输出格式
只输出纯 JSON，不加任何前缀、后缀或代码块：
{"queries": [{"query": "检索词", "intent": "该词针对的缺口或覆盖维度", "source": "tavily|xiaohongshu|zhihu|arxiv|google_scholar|github|weibo|weixin"}]}"""


def _parse_queries(text: str):
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


def runner(mode: str, section_title: str = "") -> dict:
    if mode not in ("broad", "targeted"):
        return {"error": f"mode 必须是 'broad' 或 'targeted'，收到：{mode!r}"}

    state = get_state()
    search_sources = state.get("search_sources", {})
    original_query = state.get("original_query", "")
    cfg = get_config()

    if mode == "targeted":
        if not section_title:
            return {"error": "targeted 模式必须传 section_title"}
        _, sec = get_section_by_title(section_title)
        if sec is None:
            return {"error": f"章节 '{section_title}' 不在规划列表中"}

        review = sec.get("review") or {}
        gaps = review.get("gaps", [])
        suggestions = review.get("improvement_suggestions", [])
        if not gaps and not suggestions:
            return {"error": f"章节 '{section_title}' 尚无 critique 结果，请先调用 critique_section"}

        past_queries = [
            m["query"] for m in search_sources.values()
            if m.get("section_title") == section_title
        ]

        prompt = (
            f"研究问题：{original_query}\n\n"
            f"目标章节：{section_title}\n\n"
            f"Critique 信息缺口：\n" +
            "\n".join(f"- {g}" for g in gaps) +
            (
                "\n\n改进建议：\n" + "\n".join(f"- {s}" for s in suggestions)
                if suggestions else ""
            ) +
            (
                "\n\n已有定向搜索词（避免重复或同义替换，换检索目标而非换词）：\n" +
                "\n".join(f"- {q}" for q in past_queries)
                if past_queries else ""
            ) +
            "\n\n请输出 3 条 targeted 检索词，每条针对不同缺口。"
        )
    else:
        # broad 模式：只用 original_query，返回条数对齐 MAX_BROAD_SEARCHES
        broad_count = getattr(cfg, "MAX_BROAD_SEARCHES", 8)
        past_broad = [
            m["query"] for m in search_sources.values()
            if not m.get("section_title")
        ]

        prompt = (
            f"研究问题：{original_query}\n\n"
            f"请输出 {broad_count} 条宽泛探索检索词，从不同维度覆盖整个研究主题。" +
            (
                "\n\n已有宽泛搜索词（避免重复或同义替换）：\n" +
                "\n".join(f"- {q}" for q in past_broad)
                if past_broad else ""
            )
        )

    client = get_client()
    messages = [{"role": "user", "content": prompt}]

    llm_result = call_llm_with_retry(
        client, tool_name="rewrite_query",
        max_tokens=5000, max_tokens_ceiling=10000,
        model=cfg.CRITIQUE_MODEL, system=_SYSTEM, messages=messages,
    )
    if not llm_result["ok"]:
        e = llm_result["error"]
        return {"error": f"rewrite_query 子LLM失败（{e['type']}）: {e['message']}"}

    raw = _extract_text(llm_result["data"])
    parse_result = parse_json_with_retry(
        raw, _parse_queries,
        tool_name="rewrite_query", client=client,
        model=cfg.CRITIQUE_MODEL, system=_SYSTEM,
        original_messages=messages, max_tokens=500,
    )
    if not parse_result["ok"]:
        return {"error": f"rewrite_query 输出解析失败，原文：{raw[:200]}"}

    data = parse_result["data"]
    queries = data.get("queries", [])
    if not queries:
        return {"error": "rewrite_query 返回空候选词列表"}

    # 规范化：每条 query 都必须有合法 source，缺省或非法补 tavily
    _valid_sources = {"tavily", "arxiv", "github",
                      "xiaohongshu", "zhihu", "weixin", "google_scholar", "weibo"}
    for q in queries:
        if isinstance(q, dict) and q.get("source") not in _valid_sources:
            q["source"] = "tavily"

    # 配额兜底：非 tavily 辅助源超限时，按输出顺序把多余的改回 tavily
    aux_limit = (getattr(cfg, "MAX_AUX_SOURCES_BROAD", 3) if mode == "broad"
                 else getattr(cfg, "MAX_AUX_SOURCES_TARGETED", 1))
    aux_count = 0
    for q in queries:
        if not isinstance(q, dict):
            continue
        if q.get("source") != "tavily":
            if aux_count < aux_limit:
                aux_count += 1
            else:
                q["source"] = "tavily"

    if mode == "broad":
        return {
            "mode":    "broad",
            "queries": queries,
            "avoided": past_broad if past_broad else [],
            "note":    (
                "将以上候选词全部并行调用 search（不传 section_title）；"
                "每条 query 按各自的 source 字段指定搜索源"
            ),
        }
    else:
        return {
            "mode":          "targeted",
            "section_title": section_title,
            "queries":       queries[:3],
            "avoided":       past_queries,
            "note":          (
                "从以上候选词中选择 1-2 条最相关的调用 search，不得全部使用；"
                "每条 query 按各自的 source 字段指定搜索源"
            ),
        }


register(SCHEMA, runner)
