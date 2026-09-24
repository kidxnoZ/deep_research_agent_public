"""
test_flow_state_v2.py
---------------------
全流程状态追踪测试 v2：验证重构后以 section 为单位的 state 转移。
mock 序列严格按实际 LLM 调用顺序排列，不改源代码，只调用 dispatch()。

运行：
    cd searchAgent
    python -m pytest tests/test_flow_state_v2.py -v -s
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from tools.registry import dispatch, reset_state, get_state, get_section_by_title
from tools._client import set_client
from tools.errors import set_error_log_path

# ── 显示参数 ────────────────────────────────────────────────────────────────

COMPRESS = 120
SEP  = "─" * 64
SEP2 = "═" * 64

# ── 辅助：按 title 取章节 entry ───────────────────────────────────────────────

def _sec(title: str) -> dict:
    """按标题查找章节 entry（断言辅助）。"""
    _, entry = get_section_by_title(title)
    return entry

# ── 辅助：state 打印 ─────────────────────────────────────────────────────────

def print_state(label: str):
    state = get_state()
    sections = state.get("sections", {})
    print(f"\n{SEP}")
    print(f"[STATE] {label}")
    print(SEP)
    # 全局字段
    print(f"  original_query  : {state.get('original_query','')!r}")
    print(f"  broad_locked    : {state.get('broad_locked')}  broad_pending_lock: {state.get('broad_pending_lock')}")
    ss = state.get("search_sources", {})
    broad_ids = [sid for sid, m in ss.items() if not m.get("section_title")]
    print(f"  search_sources ({len(ss)} 条，宽泛 {len(broad_ids)} 条):")
    for sid, meta in ss.items():
        kind = "broad" if not meta.get("section_title") else f"→ {meta['section_title']!r}"
        print(f"    {sid} [{kind}]: query={meta['query']!r}  file={meta['file']}")
    # 各章节
    print(f"  sections ({len(sections)} 章节):")
    for key, sec in sorted(sections.items(), key=lambda kv: kv[1]["order"]):
        draft_len  = len(sec.get("draft",""))
        final_len  = len(sec.get("final") or "")
        source_ids = sec.get("source_ids", [])
        review     = sec.get("review")
        print(f"    [{sec['order']}] {key}  title={sec.get('title','')!r}")
        print(f"         status={sec['status']}  reflect_count={sec['reflect_count']}")
        print(f"         criteria={_short(sec.get('criteria',''), 60)!r}")
        print(f"         draft={draft_len}chars  final={'None' if sec['final'] is None else f'{final_len}chars'}")
        print(f"         source_ids={source_ids}")
        if review:
            print(f"         review: is_sufficient={review['is_sufficient']} "
                  f"needs_search={review['needs_search']} "
                  f"score={review['quality_score']}")
            if review.get("gaps"):
                print(f"           gaps={review['gaps'][:2]}")
            if review.get("improvement_suggestions"):
                print(f"           suggestions={review['improvement_suggestions'][:2]}")
        else:
            print(f"         review: None")


def _short(v, limit=COMPRESS):
    if isinstance(v, str) and len(v) > limit:
        return f"{v[:limit]}…"
    return v


# ── 辅助：messages trace 打印 ────────────────────────────────────────────────

def print_trace(messages: list, label: str):
    print(f"\n{SEP}")
    print(f"[TRACE] {label}  ({len(messages)} messages)")
    print(SEP)
    for i, msg in enumerate(messages):
        role    = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "?")
                if btype == "tool_use":
                    inp = {k: v for k, v in block.get("input", {}).items()
                           if k not in ("sections", "draft_content")}
                    print(f"  [{i}][{j}] {role} → {block.get('name')}({json.dumps(inp, ensure_ascii=False)[:100]})")
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if len(c) > COMPRESS:
                        print(f"  [{i}][{j}] {role} ← {c[:COMPRESS]!r}…  [压缩 {len(c)}chars]")
                    else:
                        print(f"  [{i}][{j}] {role} ← {c!r}")


# ── Mock 工厂 ────────────────────────────────────────────────────────────────

def _mock_llm(text: str):
    block = MagicMock(); block.type = "text"; block.text = text
    resp  = MagicMock()
    resp.content = [block]; resp.stop_reason = "end_turn"
    resp.usage   = MagicMock(); resp.usage.input_tokens = 100; resp.usage.output_tokens = 50
    return resp


CRITERIA_JSON = json.dumps({
    "版本 1.0 首发角色": "需列出 1.0 版本全部首发角色及其元素/武器，数据来源须可核实",
    "版本 2.0+ 角色":    "需列出 2.0 版本起每版本新增角色，时效性内容需搜索核实",
}, ensure_ascii=False)

CRITIQUE_FAIL = json.dumps({
    "is_sufficient": False, "needs_search": True, "quality_score": 0.55,
    "gaps": ["缺少 1.0 版本完整角色列表"],
    "improvement_suggestions": ["搜索原神 1.0 首发角色完整列表"],
    "stop_reason": "needs_more_search",
}, ensure_ascii=False)

CRITIQUE_PASS = json.dumps({
    "is_sufficient": True, "needs_search": False, "quality_score": 0.92,
    "gaps": [], "improvement_suggestions": [], "stop_reason": "sufficient",
}, ensure_ascii=False)

SAVE_CH1 = "## 版本 1.0 首发角色\n旅行者、安柏、凯亚、丽莎……（搜索结果已补全）"
SAVE_CH2 = "## 版本 2.0+ 角色\n2.0稻妻：雷电将军……（搜索结果已补全）"
COMPILE  = "# 原神角色研究报告\n\n## 摘要\n本报告梳理各版本角色……\n\n## 版本 1.0\n...\n\n## 2.0+\n..."

TAVILY_RESULTS = {
    "results": [
        {"title": "原神1.0角色", "url": "https://example.com/1", "content": "旅行者、安柏、凯亚……"},
        {"title": "原神版本时间线", "url": "https://example.com/2", "content": "1.0: 2020-09-28 发布……"},
    ]
}

# ── LLM 调用顺序（精确） ──────────────────────────────────────────────────────
#
# 1. initialize_criteria          → CRITERIA_JSON
# 2. critique ch1 iter=1          → CRITIQUE_FAIL
# 3. critique ch2 iter=1          → CRITIQUE_PASS
# 4. save_section ch1 (宽泛搜索后) → SAVE_CH1
# 5. save_section ch2 (宽泛搜索后) → SAVE_CH2
# 6. critique ch1 iter=2          → CRITIQUE_PASS
# 7. compile_report               → COMPILE


# ── 主测试 ───────────────────────────────────────────────────────────────────

def test_full_flow_v2():
    reset_state()
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    set_error_log_path(tmp.name)
    get_state()["run_dir"] = tempfile.mkdtemp(prefix="flow_v2_")

    seq = [
        _mock_llm(CRITERIA_JSON),  # 1
        _mock_llm(CRITIQUE_FAIL),  # 2
        _mock_llm(CRITIQUE_PASS),  # 3
        _mock_llm(SAVE_CH1),       # 4
        _mock_llm(SAVE_CH2),       # 5
        _mock_llm(CRITIQUE_PASS),  # 6
        _mock_llm(COMPILE),        # 7
    ]
    llm_iter = iter(seq)

    mock_client = MagicMock()

    def _stream(**kw):
        # 子 LLM 走 client.messages.stream
        class _S:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def get_final_message(self):
                return next(llm_iter)
        return _S()

    mock_client.messages.stream.side_effect = _stream
    mock_client.messages.create.side_effect = lambda **kw: next(llm_iter)
    set_client(mock_client, cfg)

    messages = []
    _id = [0]

    def call(name, inp):
        result = dispatch(name, inp)
        _id[0] += 1
        tid = f"t{_id[0]:02d}"
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": name, "input": inp}
        ]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid,
             "content": json.dumps(result, ensure_ascii=False)}
        ]})
        return result

    print(f"\n{SEP2}\n  v2 全流程 state 追踪\n{SEP2}")

    # ══ 1. plan_sections ══════════════════════════════════════════════════════
    r = call("plan_sections", {
        "query": "原神各版本角色发布顺序",
        "sections": [
            {"title": "版本 1.0 首发角色", "description": "1.0版本初始角色",
             "initial_query": "原神1.0首发角色",
             "draft_content": "旅行者等1.0角色（训练知识，需搜索核实）"},
            {"title": "版本 2.0+ 角色", "description": "2.0版本起各新角色",
             "initial_query": "原神2.0角色",
             "draft_content": "雷电将军等2.0角色（训练知识，需搜索核实）"},
        ]
    })
    assert r["confirmed"], f"plan_sections failed: {r}"
    print_state("① plan_sections 后")
    assert _sec("版本 1.0 首发角色") is not None
    assert _sec("版本 1.0 首发角色")["draft"] != ""
    assert _sec("版本 1.0 首发角色")["status"] == "draft"
    assert _sec("版本 1.0 首发角色")["final"] is None
    # 验证 key 格式为 sec_NNN
    keys = list(get_state()["sections"].keys())
    assert all(k.startswith("sec_") for k in keys), f"期望 sec_NNN key，实际：{keys}"
    print_trace(messages, "plan_sections 后")

    # ══ 2. initialize_criteria ════════════════════════════════════════════════
    r = call("initialize_criteria", {
        "query": "原神各版本角色发布顺序",
        "section_titles": ["版本 1.0 首发角色", "版本 2.0+ 角色"]
    })
    assert r["initialized"], f"criteria failed: {r}"
    print_state("② initialize_criteria 后")
    assert _sec("版本 1.0 首发角色")["criteria"] != ""
    assert _sec("版本 2.0+ 角色")["criteria"]    != ""
    print_trace(messages, "criteria 后")

    # ══ 3. critique × 2（iter=1，读 draft）════════════════════════════════════
    r1 = call("critique_section", {"section_title": "版本 1.0 首发角色", "iteration": 1})
    r2 = call("critique_section", {"section_title": "版本 2.0+ 角色",    "iteration": 1})
    print_state("③ critique iter=1 后")
    print(f"\n  [DECISION] ch1: is_sufficient={r1['is_sufficient']} needs_search={r1.get('needs_search')}")
    print(f"  [DECISION] ch2: is_sufficient={r2['is_sufficient']} needs_search={r2.get('needs_search')}")
    # 验证 review 写入 state
    assert _sec("版本 1.0 首发角色")["review"]["is_sufficient"] == False
    assert _sec("版本 1.0 首发角色")["review"]["needs_search"]  == True
    assert _sec("版本 1.0 首发角色")["status"] == "needs_search"
    assert _sec("版本 2.0+ 角色")["review"]["is_sufficient"]    == True
    assert _sec("版本 2.0+ 角色")["status"]    == "done"
    print_trace(messages, "critique iter=1 后")

    # ══ 4. 宽泛搜索 ═══════════════════════════════════════════════════════════
    with patch("tools.search.requests.post") as mp:
        mp.return_value = MagicMock(
            json=MagicMock(return_value=TAVILY_RESULTS),
            raise_for_status=MagicMock()
        )
        r = call("search", {"query": "原神1.0版本首发角色完整列表", "max_results": 5})
    print_state("④ 宽泛 search 后")
    assert r.get("broad_searches_done") == 1
    assert len(get_state()["search_sources"]) == 1
    # 宽泛搜索后，未完成章节 status → pending_save
    # ch1 was needs_search → pending_save; ch2 was done → stays done
    assert _sec("版本 1.0 首发角色")["status"] == "pending_save"
    assert _sec("版本 2.0+ 角色")["status"]    == "done"   # done 不被覆盖

    # 模拟 turn 切换：broad_pending_lock → broad_locked
    state = get_state()
    if state.get("broad_pending_lock"):
        state["broad_locked"]      = True
        state["broad_pending_lock"] = False
        print("\n  [TURN] broad_pending_lock → broad_locked=True")

    # ══ 5. pending_save 拦截验证 ═══════════════════════════════════════════════
    print(f"\n  [TEST] 对 ch1（status=pending_save）再发定向搜索，应被拦截：")
    with patch("tools.search.requests.post"):
        blocked = dispatch("search", {
            "query": "凯亚详情", "section_title": "版本 1.0 首发角色"
        })
    print(f"    → {blocked}")
    assert "error" in blocked and "pending_save" in blocked["error"]
    print("    [OK] 正确拦截")

    # ══ 6. save_section ch1（自动读取 review 作为 critique_feedback）══════════
    print(f"\n  [NOTE] save_section ch1 不传 critique_feedback，工具自动从 state 读取")
    r = call("save_section", {"title": "版本 1.0 首发角色", "order": 1})
    print_state("⑤ save_section ch1 后")
    assert r["saved"]
    assert _sec("版本 1.0 首发角色")["final"] is not None
    assert _sec("版本 1.0 首发角色")["status"] == "saved"
    print_trace(messages, "save ch1 后")

    # ══ 7. save_section ch2（也有宽泛搜索，调用 LLM）══════════════════════════
    r = call("save_section", {"title": "版本 2.0+ 角色", "order": 2})
    print_state("⑥ save_section ch2 后")
    assert r["saved"]
    assert _sec("版本 2.0+ 角色")["final"] is not None
    assert SAVE_CH2[:30] in _sec("版本 2.0+ 角色")["final"]
    assert _sec("版本 2.0+ 角色")["status"] == "saved"

    # ══ 8. critique ch1 iter=2（读 final）══════════════════════════════════════
    r = call("critique_section", {"section_title": "版本 1.0 首发角色", "iteration": 2})
    print_state("⑦ critique iter=2 后")
    print(f"\n  [DECISION] ch1 iter=2: is_sufficient={r['is_sufficient']}")
    assert r["is_sufficient"]
    assert _sec("版本 1.0 首发角色")["status"] == "done"
    assert _sec("版本 1.0 首发角色")["review"]["is_sufficient"] == True
    print_trace(messages, "critique iter=2 后")

    # ══ 9. compile_report ══════════════════════════════════════════════════════
    r = call("compile_report", {"report_title": "原神角色版本研究报告"})
    print_state("⑧ compile_report 后（最终）")
    assert "saved_path" in r, f"compile failed: {r}"
    print(f"\n  报告路径：{r['saved_path']}")
    print_trace(messages, "最终 trace")

    # ══ 最终断言 ══════════════════════════════════════════════════════════════
    final_state = get_state()
    assert all(sec["final"] is not None for sec in final_state["sections"].values()), "所有章节应有 final"
    assert all(sec["status"] in ("done","saved") for sec in final_state["sections"].values())
    assert all(sec["criteria"] != "" for sec in final_state["sections"].values())
    ss = final_state["search_sources"]
    assert len(ss) >= 1
    broad_ids = [sid for sid, m in ss.items() if not m.get("section_title")]
    assert len(broad_ids) >= 1

    print(f"\n{SEP2}")
    print("  [PASS] 全部断言通过，v2 state 结构正确")
    print(f"{SEP2}\n")
    os.unlink(tmp.name)


if __name__ == "__main__":
    test_full_flow_v2()
