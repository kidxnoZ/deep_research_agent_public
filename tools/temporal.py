"""
temporal.py — get_current_date 工具
获取真实当前日期，写入 state["temporal"]，供 initialize_criteria 等工具读取。
"""

from datetime import datetime
from .registry import register, get_state

SCHEMA = {
    "name": "get_current_date",
    "description": (
        "获取当前真实日期，写入 state 供后续工具使用。\n"
        "当 query 含时效性关键词（最新、当前、截至目前、现在、到今天、XXXX年至今、开服到现在）时，"
        "在 plan_sections 之前调用一次。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


def runner() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    get_state()["temporal"] = {"current_date": today}
    return {"current_date": today}


register(SCHEMA, runner)
