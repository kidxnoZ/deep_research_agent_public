import agent


def test_summary_reads_current_section_state(capsys, monkeypatch):
    monkeypatch.setattr(agent, "read_error_log", lambda: [])
    stats = {
        "api_calls": 3,
        "total_input_tokens": 100,
        "total_output_tokens": 20,
        "max_tokens_events": 0,
        "tool_calls": {"search": 2},
    }
    state = {
        "sections": {
            "sec_001": {
                "title": "已完成章节",
                "order": 1,
                "final": "正文",
                "reflect_count": 2,
            },
            "sec_002": {
                "title": "待完成章节",
                "order": 2,
                "final": None,
                "reflect_count": 1,
            },
        },
        "search_sources": {"src_001": {}},
    }

    agent._print_summary(stats, state)
    output = capsys.readouterr().out

    assert "章节完成       : 1/2" in output
    assert "✓ 已完成章节（反思 2 次）" in output
    assert "✗ 待完成章节" in output
