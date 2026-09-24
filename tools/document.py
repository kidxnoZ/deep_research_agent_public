"""
document.py — 文档检索三件套：ingest / outline / retrieve

ingest(source)          抓取 + 分块 + 建索引，返回 doc_id
outline(doc_id)         返回章节层级结构
retrieve(doc_id, query) BM25 关键词打分，返回 top-k 相关块
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from urllib.parse import urljoin

import requests

from .registry import register, get_state, state_lock
from ._client import get_config
from .errors import ErrorType, ok, err, log_error

# ── ingest ────────────────────────────────────────────────────────────────────

INGEST_SCHEMA = {
    "name": "ingest",
    "description": (
        "抓取并索引一份文档（论文 / 财报 / 技术报告 / 官方网页），使其可被 outline / retrieve 查询。\n"
        "支持：HTTP/HTTPS URL（HTML 或 PDF）、本地文件路径（.pdf / .txt / .md）。\n"
        "arXiv 链接自动优先抓 HTML 版，其他 URL 按 Content-Type 自动判断 HTML/PDF。\n"
        "调用时机：search 结果含 doc_candidates 且满足 A/B 档触发条件时调用；"
        "或用户显式指定文档路径/URL 时调用。\n"
        "正文永不进 tool_result，只返回 doc_id 句柄 + 摘要（标题、块数、是否含表格）。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "文档 URL 或本地文件路径"},
        },
        "required": ["source"],
    },
}


def ingest_runner(source: str) -> dict:
    cfg = get_config()
    state = get_state()

    # 上限检查
    docs = state.get("documents", {})
    max_reads = getattr(cfg, "MAX_DEEP_READS_PER_RUN", 5)
    if len(docs) >= max_reads:
        return {"error": f"已 ingest {len(docs)} 份文档，达上限 {max_reads}，不再接受新文档"}

    source = source.strip()
    if not source:
        return {"error": "source 不能为空"}

    # 生成 doc_id
    doc_id = "doc_" + hashlib.md5(source.encode()).hexdigest()[:8]
    if doc_id in docs:
        d = docs[doc_id]
        return {
            "doc_id": doc_id, "title": d["title"],
            "chunks": len(d["chunks"]), "has_tables": d["has_tables"],
            "note": "文档已存在，直接使用",
        }

    # 确定存储路径
    run_dir = state.get("run_dir", "")
    doc_dir = os.path.join(run_dir if run_dir else os.path.dirname(cfg.OUTPUT_DIR),
                           "documents", doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    # 抓取内容
    is_local = not source.startswith("http://") and not source.startswith("https://")
    if is_local:
        result = _load_local(source)
    else:
        source_url = _resolve_arxiv_url(source)
        result = _fetch_url(source_url)

    if not result["ok"]:
        return {"error": result["error"]}

    source_type = result["source_type"]   # "html" | "pdf" | "text"
    raw_text    = result["text"]
    title       = result.get("title", "") or _guess_title(source)
    tables      = result.get("tables", [])

    # 分块
    chunk_size = getattr(cfg, "DEEP_READ_CHUNK_SIZE", 1500)
    text_chunks  = _chunk_text(raw_text, chunk_size)
    table_chunks = [{"text": t, "section": "表格", "level": 0, "is_table": True} for t in tables]
    chunks = text_chunks + table_chunks

    if not chunks:
        return {"error": "文档内容为空，无法建立索引"}

    # 提取 outline（从文本标题行直接建树，标题后无正文也不丢失）
    outline = _extract_outline(raw_text)

    # 写盘
    meta = {
        "source": source, "source_type": source_type, "title": title,
        "path": doc_dir, "has_tables": bool(tables),
        "chunk_count": len(chunks), "outline": outline,
    }
    with open(os.path.join(doc_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 写 state
    with state_lock:
        state["documents"][doc_id] = {
            "source": source, "source_type": source_type, "title": title,
            "path": doc_dir, "chunks": chunks, "outline": outline,
            "has_tables": bool(tables),
        }

    return {
        "doc_id":     doc_id,
        "title":      title,
        "source_type": source_type,
        "chunks":     len(chunks),
        "has_tables": bool(tables),
        "outline_sections": len(outline),
    }


# ── outline ───────────────────────────────────────────────────────────────────

OUTLINE_SCHEMA = {
    "name": "outline",
    "description": (
        "返回已 ingest 文档的树状章节结构（宏观视图）：\n"
        "outline 为嵌套列表 [{title, level, children}]，level 为标题层级（数字越小层级越高）。\n"
        "在 plan_sections 之前调用，了解文档真实结构，让报告框架对齐文档而非套模板。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "ingest 返回的 doc_id"},
        },
        "required": ["doc_id"],
    },
}


def outline_runner(doc_id: str) -> dict:
    state = get_state()
    doc = state.get("documents", {}).get(doc_id)
    if doc is None:
        return {"error": f"doc_id '{doc_id}' 不存在，请先调用 ingest"}

    return {
        "doc_id":  doc_id,
        "title":   doc["title"],
        "outline": doc["outline"],
        "chunks":  len(doc["chunks"]),
        "has_tables": doc["has_tables"],
    }


# ── retrieve ──────────────────────────────────────────────────────────────────

RETRIEVE_SCHEMA = {
    "name": "retrieve",
    "description": (
        "从已 ingest 的文档中，按 query 检索最相关的文本块/表格块（BM25 打分）。\n"
        "用途：替代 search，从文档内部定位具体段落/数字/表格。\n"
        "与 search 的区别：search 从互联网取 snippet；retrieve 从本地文档取原文块。\n"
        "调用时机：critique 指出信息缺口，且该缺口预期在已 ingest 文档中存在时。\n"
        "返回 error（未找到相关内容）说明文档中确无此主题，应改用 search 补充。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id":    {"type": "string",  "description": "ingest 返回的 doc_id"},
            "query":     {"type": "string",  "description": "检索问题或关键词"},
            "max_chars": {"type": "integer", "description": "返回内容字符上限，默认 4000"},
        },
        "required": ["doc_id", "query"],
    },
}


def retrieve_runner(doc_id: str, query: str, max_chars: int = 0) -> dict:
    cfg = get_config()
    state = get_state()

    if not query or not query.strip():
        return {"error": "query 不能为空"}

    doc = state.get("documents", {}).get(doc_id)
    if doc is None:
        return {"error": f"doc_id '{doc_id}' 不存在，请先调用 ingest"}

    if max_chars <= 0:
        max_chars = getattr(cfg, "DEEP_READ_MAX_CHARS", 4000)

    chunks = doc["chunks"]
    if not chunks:
        return {"error": "文档无内容块"}

    # BM25 打分，取 top-k
    scored = _bm25_rank(query, chunks)

    # 硬负反馈：query 与文档零重合（所有块得分 0）时明确报错，不硬塞无关块
    if not scored or all(s <= 0 for _, s in scored):
        return {"error": "文档中未找到与 query 相关的内容，建议改用 search 补充"}

    selected = []
    total_chars = 0
    for chunk, score in scored:
        text = chunk["text"]
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            min_fragment = min(200, max_chars // 2)
            if remaining > min_fragment:
                selected.append({"text": text[:remaining] + "…[截断]",
                                  "section": chunk.get("section", ""),
                                  "is_table": chunk.get("is_table", False),
                                  "score": round(score, 3)})
            break
        selected.append({"text": text, "section": chunk.get("section", ""),
                          "is_table": chunk.get("is_table", False),
                          "score": round(score, 3)})
        total_chars += len(text)

    if not selected:
        return {"error": "未找到相关内容，请换关键词"}

    # 拼接返回（保留 section 标注）
    parts = []
    for item in selected:
        prefix = f"[{item['section']}]\n" if item.get("section") else ""
        parts.append(prefix + item["text"])
    content = "\n\n---\n\n".join(parts)

    return {
        "doc_id":        doc_id,
        "query":         query,
        "blocks_returned": len(selected),
        "chars_returned": len(content),
        "content":       content,
    }


# ── 内部工具函数 ───────────────────────────────────────────────────────────────

def _resolve_arxiv_url(url: str) -> str:
    """只有 /abs/ 抽象页映射到 /html/ 全文；/pdf/ 与 /html/ 尊重原地址。"""
    # https://arxiv.org/abs/2305.12345 → https://arxiv.org/html/2305.12345
    m = re.match(r"https?://arxiv\.org/abs/(\d+\.\d+)(v\d+)?", url)
    if m:
        paper_id = m.group(1) + (m.group(2) or "")
        return f"https://arxiv.org/html/{paper_id}"
    return url


def _fetch_url(url: str) -> dict:
    """HTTP 抓取，按 Content-Type 分派 HTML / PDF 解析。"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; searchAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"ok": False, "error": f"请求超时: {url}"}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"请求失败: {e}"}

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _parse_pdf_bytes(resp.content, url)
    else:
        # 字节解码：requests.text 对缺失/错误 charset 声明的老中文网站会产生 mojibake
        html_text = _decode_html(resp.content)
        # HTML：先检测页面里的全文 PDF 直链（摘要页 → PDF 全文）
        pdf_url = _find_pdf_link(html_text, url)
        if pdf_url:
            try:
                pdf_resp = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
                pdf_resp.raise_for_status()
                if "pdf" in pdf_resp.headers.get("Content-Type", "").lower():
                    return _parse_pdf_bytes(pdf_resp.content, pdf_url)
            except requests.exceptions.RequestException:
                pass  # PDF 抓取失败，退回 HTML 摘要解析
        return _parse_html(html_text, url)


def _decode_html(raw: bytes) -> str:
    """
    按字节解码 HTML：依次尝试 utf-8 → meta 声明 charset → gbk，兜底 replacement。
    requests.text 对缺失/错误 charset 声明的老中文网站会产生 Latin-1 mojibake
    （实测北大新闻网 UTF-8 页面被解成乱码，chunks 中文全毁）。
    """
    head = raw[:2000].decode("ascii", errors="ignore")
    m = re.search(r'charset\s*=\s*["\']?\s*([a-zA-Z0-9_-]+)', head, re.IGNORECASE)
    declared = m.group(1).lower().replace("_", "-") if m else ""
    candidates = ["utf-8"]
    if declared and declared not in ("utf-8", "utf8") and declared not in candidates:
        candidates.append(declared)
    candidates.append("gbk")
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _find_pdf_link(html: str, base_url: str) -> str:
    """
    从 HTML 摘要页里找全文 PDF 直链（优先 citation_pdf_url meta，退而同域 .pdf 链接）。
    找不到返回空串——付费墙页面（如 Springer）无 PDF 直链，自然降级为摘要。
    """
    # 优先 citation_pdf_url meta（学术页面标准）
    m = re.search(
        r'<meta[^>]*name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if m:
        return urljoin(base_url, m.group(1))
    # 退而求其次：页面内 .pdf 链接
    for href in re.findall(r'<a[^>]*href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE):
        return urljoin(base_url, href)
    return ""


def _load_local(path: str) -> dict:
    """加载本地文件（.pdf / .txt / .md）。"""
    if not os.path.exists(path):
        return {"ok": False, "error": f"文件不存在: {path}"}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with open(path, "rb") as f:
            return _parse_pdf_bytes(f.read(), path)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return {"ok": True, "source_type": "text", "text": text, "title": os.path.basename(path), "tables": []}


def _parse_html(html: str, url: str) -> dict:
    """HTML → 正文文本 + 标题提取（无外部依赖，正则清洗）。"""
    # 提取 <title>
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""

    # 提取 <table> 并结构化为 markdown
    tables = _extract_html_tables(html)

    # 去除 script / style / nav / header / footer / aside / form
    # （form：arxiv "报告问题"弹窗等常驻 UI）
    cleaned = re.sub(r"<(script|style|nav|header|footer|aside|form)[^>]*>.*?</\1>",
                     "", html, flags=re.IGNORECASE | re.DOTALL)
    # 去除公告 banner（如 arxiv 的 ds-announcement）
    cleaned = re.sub(r'<div[^>]*class="[^"]*ds-announcement[^"]*"[^>]*>.*?</div>',
                     "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # 段落换行（h[1-6] 闭合标签必须保留，供下方标题转换匹配）
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(p|div|li|tr)>", "\n", cleaned, flags=re.IGNORECASE)
    # 标题 → markdown # 标记（level 1-6，供 _chunk_text 识别）
    for level in range(1, 7):
        cleaned = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda m, l=level: f"\n{'#' * l} {re.sub(r'<[^>]+>', '', m.group(1)).strip()}\n",
            cleaned, flags=re.IGNORECASE | re.DOTALL,
        )
    # 去除剩余 tag
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # 折叠多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if len(cleaned) < 30:
        return {"ok": False, "error": "页面正文内容过少，可能为登录墙或空页"}

    return {"ok": True, "source_type": "html", "text": cleaned, "title": title, "tables": tables}


def _extract_html_tables(html: str) -> list[str]:
    """从 HTML 中提取 <table>，转成 markdown 格式。"""
    tables = []
    for table_html in re.findall(r"<table[^>]*>.*?</table>", html, re.IGNORECASE | re.DOTALL):
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.IGNORECASE | re.DOTALL)
        md_rows = []
        for i, row in enumerate(rows[:30]):  # 最多 30 行
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)
            cells = [re.sub(r"<[^>]+>", "", c).strip().replace("\n", " ") for c in cells]
            if not any(cells):
                continue
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
        if len(md_rows) > 2:
            tables.append("\n".join(md_rows))
    return tables


def _parse_pdf_bytes(data: bytes, source: str) -> dict:
    """PDF 字节 → 文本 + 表格（优先 pymupdf，降级 pypdf）。"""
    # 尝试 pymupdf
    try:
        import fitz  # pymupdf
        return _parse_pdf_fitz(data, source)
    except ImportError:
        pass
    # 降级 pypdf
    try:
        import pypdf
        return _parse_pdf_pypdf(data, source)
    except ImportError:
        pass
    return {"ok": False, "error": "PDF 解析需要 pymupdf 或 pypdf，请 pip install pymupdf 或 pypdf"}


def _parse_pdf_fitz(data: bytes, source: str) -> dict:
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    title = doc.metadata.get("title", "") or _guess_title(source)
    tables = []
    pages = []
    for page in doc:
        # 表格抽取
        for tab in page.find_tables():
            rows = tab.extract()
            if not rows:
                continue
            md = []
            for i, row in enumerate(rows[:30]):
                cells = [str(c or "").strip().replace("\n", " ") for c in row]
                md.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md.append("|" + "|".join(["---"] * len(cells)) + "|")
            if len(md) > 2:
                tables.append("\n".join(md))
        pages.append(page.get_text("dict"))
    full_text = _pdf_text_with_headings(pages)
    return {"ok": True, "source_type": "pdf", "text": full_text, "title": title, "tables": tables}


def _pdf_text_with_headings(pages: list[dict]) -> str:
    """
    从 pymupdf get_text("dict") 结构提取文本，用字号启发式把标题行转成 # 前缀。
    正文字号 = 按字符数加权的众数；字号显著更大的行按字号档位映射 1-4 级标题；
    与正文同字号但匹配编号模式（如 "2.1 xxx"）或独立全大写的行识别为下级标题。
    输出带 # 前缀的文本，由 _chunk_text 统一消费。
    """
    # 1. 收集所有行 (text, size)
    lines: list[tuple[str, float]] = []
    for page in pages:
        for block in page.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = max(s.get("size", 0) or 0 for s in spans)
                lines.append((text, size))

    if not lines:
        return ""

    # 2. 正文字号 = 按字符数加权的众数
    size_weight = Counter()
    for text, size in lines:
        size_weight[round(size, 1)] += len(text)
    if not size_weight:
        return "\n".join(t for t, _ in lines)
    base_size = max(size_weight.items(), key=lambda kv: kv[1])[0]

    # 3. 显著大于正文的字号档位 → 标题 level（大→小 = 1→4）
    heading_sizes = sorted(
        {round(s, 1) for _, s in lines if s > base_size + 0.5 and s > base_size * 1.15},
        reverse=True,
    )[:4]
    size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

    # 4. 输出：标题行加 # 前缀；同字号但编号模式/全大写的短行标为最深级标题
    max_level = len(heading_sizes) + 1
    out = []
    for text, size in lines:
        lvl = size_to_level.get(round(size, 1))
        if lvl is None and len(text) <= 150:
            if re.match(r"^\d+(\.\d+)*\s+\S", text) or (len(text) > 4 and text.isupper()):
                lvl = max_level
        if lvl:
            out.append(f"{'#' * min(lvl, 6)} {text}")
        else:
            out.append(text)
    return "\n".join(out)


def _parse_pdf_pypdf(data: bytes, source: str) -> dict:
    import io
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    title = (reader.metadata or {}).get("/Title", "") or _guess_title(source)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(t for t in pages_text if t.strip())
    return {"ok": True, "source_type": "pdf", "text": full_text, "title": title, "tables": []}


def _guess_title(source: str) -> str:
    """从 URL 或文件路径猜标题。"""
    basename = source.rstrip("/").split("/")[-1].split("?")[0]
    return re.sub(r"[_\-]+", " ", basename).strip() or source[:60]


def _chunk_text(text: str, chunk_size: int) -> list[dict]:
    """
    按 markdown 标题（# 至 ######）和段落切块，每块 ≤ chunk_size 字符。
    每块带 section 字段（所属最近一个标题）与 level 字段（标题层级，0 表示无标题）。
    """
    lines = text.split("\n")
    chunks = []
    current_section = ""
    current_level = 0
    current_buf: list[str] = []

    def _flush(buf: list[str], section: str, level: int):
        text_block = "\n".join(buf).strip()
        if not text_block:
            return
        # 超大块按 chunk_size 切分
        while len(text_block) > chunk_size:
            cut = text_block[:chunk_size].rfind("\n")
            if cut < chunk_size // 2:
                cut = chunk_size
            chunks.append({"text": text_block[:cut].strip(), "section": section,
                           "level": level, "is_table": False})
            text_block = text_block[cut:].strip()
        if text_block:
            chunks.append({"text": text_block, "section": section,
                           "level": level, "is_table": False})

    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            _flush(current_buf, current_section, current_level)
            current_buf = []
            current_section = heading.group(2).strip()
            current_level = len(heading.group(1))
        else:
            current_buf.append(line)
            if sum(len(l) for l in current_buf) >= chunk_size:
                _flush(current_buf, current_section, current_level)
                current_buf = []

    _flush(current_buf, current_section, current_level)
    return chunks


def _extract_outline(text: str) -> list[dict]:
    """
    从文本中的 # 标题行构建树状大纲（栈法）。
    直接从标题行建树而非从 chunk 的 section 推导——标题后无正文时该标题仍保留在树中。
    返回 [{title, level, children:[...]}] 嵌套结构。
    """
    root: list[dict] = []
    stack: list[tuple[int, dict]] = []   # (level, node)
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        node = {"title": m.group(2).strip(), "level": level, "children": []}
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root.append(node)
        stack.append((level, node))
    return root


# ── BM25 实现 ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """简单 tokenize：小写 + 中文单字 + 英文词，去标点。"""
    text = text.lower()
    # 拆出中文字符和英文单词
    tokens = re.findall(r"[一-鿿]|[a-z0-9]+", text)
    return tokens


def _bm25_rank(query: str, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> list[tuple]:
    """返回 (chunk, score) 列表，按 score 降序，表格块得分加权 1.5×。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [(c, 0.0) for c in chunks]

    # 文档统计
    n = len(chunks)
    doc_tokens = [_tokenize(c["text"]) for c in chunks]
    doc_lens   = [len(t) for t in doc_tokens]
    avg_dl     = sum(doc_lens) / n if n else 1

    # IDF：每个 query token 在多少 chunk 中出现
    df = Counter()
    for tokens in doc_tokens:
        for tok in set(tokens):
            df[tok] += 1

    scores = []
    for i, (chunk, tokens) in enumerate(zip(chunks, doc_tokens)):
        tf_map = Counter(tokens)
        dl     = doc_lens[i] or 1
        score  = 0.0
        for tok in q_tokens:
            if tok not in tf_map:
                continue
            tf  = tf_map[tok]
            idf = math.log((n - df[tok] + 0.5) / (df[tok] + 0.5) + 1)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm
        # 表格块加权（数字/指标密集，命中价值高）
        if chunk.get("is_table"):
            score *= 1.5
        scores.append((chunk, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ── 注册 ──────────────────────────────────────────────────────────────────────

register(INGEST_SCHEMA, ingest_runner)
register(OUTLINE_SCHEMA, outline_runner)
register(RETRIEVE_SCHEMA, retrieve_runner)
