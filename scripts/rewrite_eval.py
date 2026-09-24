"""
test_rewrite_eval.py — rewrite_query 工具效果评估（golden 集）

适配 tools/query_rewriter.py 的 rewrite_query 工具（targeted 模式）：
- 每条用例构造独立 state：研究问题 + 章节 + review.gaps + 历史 query
- 通过 dispatch 真实调用 rewrite_query（真实调 LLM，生成 3 条候选）
- 从 3 条候选中取与 golden 相似度最高的一条，算 recall / jaccard
- 同时输出 baseline（原 query 不 rewrite 直接对比 golden），量化 rewrite 的提升

需先配置 .env / 环境变量（真实调 LLM）。

运行：
    cd searchAgent
    python scripts/rewrite_eval.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
import config as cfg
from tools.registry import dispatch, reset_state, get_state, _make_section_entry
from tools._client import set_client

# ── 测试集（硬编码 golden）────────────────────────────────────────────────────
# (研究问题, 章节标题, critique gap, 原 query, 期望 rewrite query)
CASES = [
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "财务画像",
        "缺 Generali operating result",
        "Generali Group 2025 full year results operating result net result total assets premiums",
        "Generali 2025 operating result",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "财务画像",
        "缺 Zurich dividend",
        "Zurich Insurance Group 2025 annual results total assets revenue net income market capitalization",
        "Zurich 2025 dividend annual report",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "财务画像",
        "缺 Munich Re 净利润增速",
        "Munich Re 2025 annual results total assets revenue net income market cap",
        "Munich Re 2025 net profit growth",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "财务画像",
        "缺 AIA 新业务价值",
        "AIA Group 2025 full year results VONB net profit revenue growth",
        "AIA 2025 value of new business VONB",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "分红",
        "缺 Allianz 每股派息",
        "Allianz 2025 dividend per share EUR total dividend payout record",
        "Allianz 2025 dividend per share",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "融资",
        "缺 AXA 偿付能力比率",
        "AXA Generali Solvency II ratio 2025 SFCR own funds SCR",
        "AXA 2025 Solvency II ratio",
    ),
    (
        "收集整理国际综合实力前十的保险公司相关资料横向比较",
        "评级",
        "缺 UnitedHealth 评级展望",
        "UnitedHealth Group S&P Moody's credit rating 2025 2026 outlook",
        "UnitedHealth credit rating S&P Moody's Fitch 2025",
    ),
    (
        "原神全角色发布版本按照版本顺序整理",
        "5.0 及以后角色",
        "缺最新版本角色",
        "原神 5.5 5.6 5.7 5.8 6.0 6.1 6.2 6.3 6.4 6.5 版本 新角色 上线 官方 汇总",
        "原神 最新版本 新角色 上线",
    ),
    (
        "收集整理中国9阶层实际收入和财务状况",
        "各阶层收入水平与实际分布",
        "缺权威量化来源",
        "中国社会九大阶层 梁晓声 收入区间 各阶层 人数 占比 划分标准",
        "CFPS 中国家庭金融调查 收入分层 人口占比",
    ),
    (
        "中国金融未来发展趋势，哪个细分领域更有前景",
        "债券承销",
        "缺债券承销规模",
        "2025 2026 中国IPO节奏 并购重组 券商投行收入 债券承销",
        "2025 中国债券承销规模 券商",
    ),
]


# ── 相似度计算 ───────────────────────────────────────────────────────────────

def _tokenize(query: str) -> set:
    q = (query or "").lower()
    tokens = set()
    tokens.update(re.findall(r"[a-z]+|\d+", q))
    for seg in re.findall(r"[一-鿿]+", q):
        tokens.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def _recall(golden: set, predict: set) -> float:
    if not golden:
        return 1.0
    return len(golden & predict) / len(golden)


def _jaccard(golden: set, predict: set) -> float:
    if not golden and not predict:
        return 1.0
    if not golden or not predict:
        return 0.0
    return len(golden & predict) / len(golden | predict)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    client = anthropic.Anthropic(
        base_url=cfg.ANTHROPIC_BASE_URL,
        api_key=cfg.API_KEY,
    )
    set_client(client, cfg)

    print("=" * 76)
    print("rewrite_query 工具效果评估（真实调 LLM）")
    print("=" * 76)

    base_recs, base_jacs = [], []
    rw_recs, rw_jacs = [], []

    for i, (question, section_title, gap, original_query, golden) in enumerate(CASES, 1):
        # ── 构造独立 state ────────────────────────────────────────────────
        reset_state()
        state = get_state()
        state["original_query"] = question
        sec = _make_section_entry(title=section_title, order=1)
        sec["review"] = {
            "is_sufficient": False, "needs_search": True, "quality_score": 0.5,
            "gaps": [gap], "improvement_suggestions": [],
        }
        sec["final"] = "现有章节内容"
        state["sections"]["sec_001"] = sec
        state["search_sources"]["src_past"] = {
            "query": original_query, "section_title": section_title,
            "file": "", "summary": "", "result_count": 5,
        }

        g_tok = _tokenize(golden)

        # baseline：原 query 不 rewrite
        base_tok = _tokenize(original_query)
        base_rec, base_jac = _recall(g_tok, base_tok), _jaccard(g_tok, base_tok)

        # rewrite：真实调工具
        try:
            r = dispatch("rewrite_query", {"mode": "targeted", "section_title": section_title})
        except Exception as e:
            print(f"[{i:2d}] 异常：{type(e).__name__}: {e}")
            continue

        if "error" in r:
            print(f"[{i:2d}] rewrite 报错：{r['error'][:120]}")
            continue

        queries = [q.get("query", "") for q in r.get("queries", [])]
        # 从 3 条候选取与 golden 相似度最高的一条
        best = max(
            (q for q in queries if q),
            key=lambda q: _jaccard(g_tok, _tokenize(q)),
            default="",
        )
        rw_tok = _tokenize(best)
        rw_rec, rw_jac = _recall(g_tok, rw_tok), _jaccard(g_tok, rw_tok)

        base_recs.append(base_rec); base_jacs.append(base_jac)
        rw_recs.append(rw_rec);   rw_jacs.append(rw_jac)

        print(f"[{i:2d}] {section_title} | gap: {gap}")
        print(f"     golden  : {golden}")
        print(f"     baseline: {original_query}")
        print(f"     rewrite : {best}   （候选 {len(queries)} 条，取最佳）")
        print(f"     baseline jaccard={base_jac:.3f}  →  rewrite jaccard={rw_jac:.3f}  (recall {rw_rec:.3f})")
        print("-" * 76)

    n = len(rw_recs)
    if n:
        print(f"\n总样本数：{n}")
        print(f"baseline 平均 jaccard : {sum(base_jacs) / n:.3f}")
        print(f"rewrite  平均 jaccard : {sum(rw_jacs) / n:.3f}")
        print(f"rewrite  平均 recall  : {sum(rw_recs) / n:.3f}")
    else:
        print("\n无有效样本")


if __name__ == "__main__":
    main()
