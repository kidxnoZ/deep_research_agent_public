"""
测试 compile_report 标题层级修正 + 编号（v2.7.7）：
- LLM 输出 ### 一级子标题 保持 ###，不被降级为 ####
- LLM 输出 #### 二级子标题 保持 ####
- LLM 误输出 ## 章节同级标题 降级为 ###（兜底）
- _add_heading_numbers 编号正确：##→1. / ###→1.1. / ####→1.1.1.

背景 bug：旧代码把 ### 错误降级为 ####，导致正文里没有 ### 这一层，
编号出现 "1.0.1." 的畸形结果（counter[1] 恒为 0）。
"""

import re, os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.report import _add_heading_numbers


def _fix_levels(content: str) -> str:
    """复刻 compile_runner 修复后的层级修正逻辑。"""
    return re.sub(r'^##(?!#)', '###', content, flags=re.MULTILINE)


def _compile(content_by_title: dict) -> str:
    """模拟 compile_runner 的代码拼接 + 层级修正 + 编号（不含 LLM 过渡句）。"""
    parts = ["# 测试报告\n"]
    for title, final in content_by_title.items():
        parts.append(f"\n## {title}\n")
        # 去重复标题（与 state 标题一致才删）
        content = re.sub(
            r'^#{1,4}\s*' + re.escape(title) + r'\s*\n?',
            '', final, count=1, flags=re.MULTILINE
        ).lstrip('\n')
        content = _fix_levels(content)
        parts.append(content)
    return _add_heading_numbers("\n".join(parts))


def _headings(text: str) -> list:
    return [l for l in text.split('\n') if l.startswith('#')]


def test_llm_h3_stays_h3():
    """LLM 输出 ### 一级子标题，应保持 ###，编号 1.1. 而非 1.0.1."""
    final = "### 宏观政策定调\n内容A\n### 利率下行\n内容B"
    report = _compile({"中国金融业": final})
    heads = _headings(report)
    assert any(l.startswith("### 1.1.") for l in heads), f"应含 1.1. 编号: {heads}"
    assert not any("1.0." in l for l in heads), f"不应出现 1.0.x 畸形编号: {heads}"
    print(f"[PASS] ### 保持 ###，编号正确: {heads}")


def test_llm_h4_stays_h4():
    """LLM 输出 #### 二级子标题，应保持 ####，编号 1.1.1."""
    final = "### 宏观政策定调\n内容A\n#### 监管框架细节\n内容B"
    report = _compile({"中国金融业": final})
    heads = _headings(report)
    assert any(l.startswith("#### 1.1.1.") for l in heads), f"应含 1.1.1. 编号: {heads}"
    assert not any("1.0." in l for l in heads), f"不应出现 1.0.x: {heads}"
    print(f"[PASS] #### 保持 ####，编号正确: {heads}")


def test_llm_h2_downgraded_to_h3():
    """LLM 误输出 ## 章节同级标题，应降级为 ###，编号 1.1."""
    final = "## 宏观政策定调\n内容A\n## 利率下行\n内容B"
    report = _compile({"中国金融业": final})
    heads = _headings(report)
    assert any(l.startswith("### 1.1.") for l in heads), f"应降级为 ### 1.1.: {heads}"
    assert not any(l.startswith("## 1.") and "中国金融业" not in l for l in heads), \
        f"正文不应保留 ## 标题: {heads}"
    print(f"[PASS] ## 兜底降级为 ###: {heads}")


def test_multi_section_numbering_resets():
    """多章节编号独立递增，第二章从 2.1. 重新开始。"""
    final = "### 甲\n内容\n### 乙\n内容"
    report = _compile({"第一章": final, "第二章": final})
    heads = _headings(report)
    assert any(l.startswith("### 1.1.") for l in heads)
    assert any(l.startswith("### 1.2.") for l in heads)
    assert any(l.startswith("### 2.1.") for l in heads)
    assert any(l.startswith("### 2.2.") for l in heads)
    print(f"[PASS] 多章节编号独立递增: {heads}")


def test_mixed_h3_h4_numbering():
    """混用 ### 和 #### 时，#### 编号挂在最近的 ### 下。"""
    final = "### 一级A\n内容\n#### 二级a\n内容\n#### 二级b\n内容\n### 一级B\n内容\n#### 二级c\n内容"
    report = _compile({"章节": final})
    heads = _headings(report)
    expected = [
        "### 1.1. 一级A",
        "#### 1.1.1. 二级a",
        "#### 1.1.2. 二级b",
        "### 1.2. 一级B",
        "#### 1.2.1. 二级c",
    ]
    # 只取章节内子标题（跳过 # 报告标题和 ## 章节标题）
    got = [l for l in heads if l.startswith('###') or l.startswith('####')]
    for e in expected:
        assert e in got, f"缺少预期标题 '{e}'，实际: {got}"
    print(f"[PASS] 混用层级编号正确: {got}")


def test_skipped_heading_level_is_promoted():
    """直接出现 #### 时应提升为 ###，不能产生 1.0.1 编号。"""
    report = _add_heading_numbers("# 报告\n\n## 章节\n\n#### 跳级标题\n内容")
    heads = _headings(report)
    assert "### 1.1. 跳级标题" in heads
    assert not any(".0." in heading for heading in heads)


if __name__ == "__main__":
    tests = [
        test_llm_h3_stays_h3,
        test_llm_h4_stays_h4,
        test_llm_h2_downgraded_to_h3,
        test_multi_section_numbering_resets,
        test_mixed_h3_h4_numbering,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
