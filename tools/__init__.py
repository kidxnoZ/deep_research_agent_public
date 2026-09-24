"""
tools/__init__.py
导入所有工具模块（触发 register 调用），对外暴露 registry 的核心函数。
"""

from . import planner          # noqa: F401 — 触发 register
from . import criteria         # noqa: F401
from . import search           # noqa: F401
from . import reflect          # noqa: F401
from . import report           # noqa: F401
from . import temporal         # noqa: F401
from . import query_rewriter   # noqa: F401
from . import document         # noqa: F401

from .registry import get_schemas, dispatch, reset_state, get_state, state_lock
from ._client import set_client

__all__ = ["get_schemas", "dispatch", "reset_state", "get_state", "set_client"]
