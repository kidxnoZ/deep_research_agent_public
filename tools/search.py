"""
search.py — search 工具
支持多搜索源：tavily（默认）、arxiv、github、xiaohongshu、zhihu、weixin、weibo、google_scholar。
所有源归一化为 {title, url, snippet}，下游 save_section/report 零改动。
结果落盘，tool_result 只返回紧凑引用。
"""

import hashlib
import json
import os
import re
import subprocess
import time
import requests

from .registry import register, get_state, get_section_by_title, state_lock, assign_ref_idx
from . import rate_limiter
from ._client import get_config
from .errors import ErrorType, ok, err, log_error
from .retry import call_with_retry

SCHEMA = {
    "name": "search",
    "description": (
        "执行网络搜索，获取相关网页内容。结果自动落盘，返回 source_id 供后续使用。\n"
        "两种使用场景：\n"
        "1. 宽泛搜索阶段（section_title 可不填）：对整个研究主题并行发起多个不同角度的搜索\n"
        "2. 定向补搜阶段（指定 section_title）：针对 critique_section 指出的具体信息空白\n\n"
        "source 可选值（默认 tavily）：\n"
        "- tavily：通用网页搜索（默认主源，覆盖事实/新闻/百科/政策等绝大多数场景）\n"
        "- arxiv：学术预印本论文\n"
        "- github：开源代码仓库\n"
        "- xiaohongshu：生活经验/口碑/消费决策/避坑\n"
        "- zhihu：问答/深度讨论/专业见解\n"
        "- weixin：微信公众号文章\n"
        "- google_scholar：Google Scholar 学术论文（仅标题/作者/引用数等元数据，无摘要正文）\n"
        "- weibo：微博实时讨论/舆情"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query":         {"type": "string",  "description": "搜索关键词或问题"},
            "source":        {"type": "string",  "description": "搜索源，默认 tavily"},
            "section_title": {"type": "string",  "description": "归属章节（可选）"},
            "max_results":   {"type": "integer", "description": "最大返回结果数，默认 3"}
        },
        "required": ["query"]
    }
}


def runner(query: str, source: str = "tavily", section_title: str = "", max_results: int = 3) -> dict:
    cfg = get_config()
    state = get_state()
    sections = state["sections"]

    if not query or not query.strip():
        return err(ErrorType.PARAM_ERROR, "query 不能为空")

    if source not in _CALLERS:
        return {"error": f"不支持的搜索源 '{source}'，可用源：{list(_CALLERS.keys())}"}

    broad_retry = False
    if not section_title:
        failed = state.get("broad_failed_queries", set())
        if query in failed:
            # 失败条目的重试豁免：宽泛批次中失败过的 query 允许重试一次
            # （错误处理路径，不计为新一批宽泛搜索）；豁免一次性，消费后不再放行
            failed.remove(query)
            broad_retry = True
        elif state.get("broad_locked"):
            return {"error": "宽泛搜索已完成一批并锁定，请使用定向补搜（指定 section_title）"}
        else:
            broad_done = sum(1 for m in state.get("search_sources", {}).values() if not m.get("section_title"))
            if broad_done >= cfg.MAX_BROAD_SEARCHES:
                return {"error": f"宽泛搜索已达上限 {cfg.MAX_BROAD_SEARCHES} 次，请使用定向补搜"}
    else:
        _, sec = get_section_by_title(section_title)
        if sec is None:
            return {"error": f"章节「{section_title}」不在规划中，已规划章节：{list(sections.keys())}"}
        if sec["status"] == "pending_save":
            return {"error": (
                f"章节「{section_title}」已有未保存的搜索结果（status=pending_save），"
                "必须先调用 save_section，再 critique_section，然后才能继续搜索。"
                "这是 search→save→critique 强制循环。"
            )}
        targeted_done = len(sec.get("source_ids", [])) + len(sec.get("pending_src_ids", []))
        if targeted_done >= cfg.MAX_TARGETED_PER_SECTION:
            return {"error": (
                f"章节「{section_title}」定向补搜已达上限 {cfg.MAX_TARGETED_PER_SECTION} 次，"
                "请调用 save_section(critique_feedback=<gaps>) 直接修复"
            )}

    # 分级限速：熔断检查 → QPH 检查 → 获取并发槽 → jitter sleep
    ok, rate_err = rate_limiter.acquire(source, cfg)
    if not ok:
        return {"error": rate_err}

    call_ok = False
    try:
        result = call_with_retry(_CALLERS[source], query, max_results, cfg, tool_name="search")
        call_ok = result["ok"]
    finally:
        rate_limiter.release(source, success=call_ok, cfg=cfg)

    if not result["ok"]:
        e = result["error"]
        if not section_title and not broad_retry:
            # 记录宽泛搜索失败条目，供 broad_locked 后重试豁免
            # （豁免重试再失败不再记入，避免无限豁免）
            state.setdefault("broad_failed_queries", set()).add(query)
        return {"error": f"搜索失败（{e['type']}）: {e['message']}，error_id={e.get('error_id','')}"}

    results = result["data"]

    if not results:
        error_id = log_error("search", ErrorType.SEARCH_NO_RESULTS,
                             f"query='{query}' source='{source}' 返回 0 条结果")
        if not section_title and not broad_retry:
            state.setdefault("broad_failed_queries", set()).add(query)
        hint = ("gh search 按 AND 严格匹配每个词，长 query 易 0 结果，请用 2-3 个核心词重试"
                if source == "github" else "请换关键词重试")
        return {"error": f"搜索无结果（query='{query}'，source='{source}'），{hint}，error_id={error_id}"}

    results, doc_candidates = _mark_doc_candidates(results, source)
    # 给每条结果 URL 分配全局唯一引用编号，写入磁盘文件供 save_section 读取
    results = [dict(r, ref_idx=assign_ref_idx(r.get("url", ""))) for r in results]
    src_id = _make_src_id(query, source)
    run_dir = state.get("run_dir", "")
    file_path = _write_results(src_id, query, source, section_title, results, cfg, run_dir)
    summary = _make_summary(results)

    if not section_title:
        with state_lock:
            for _key, s in sections.items():
                if s["status"] not in ("done",):
                    s["status"] = "pending_save"
            state["search_sources"][src_id] = {
                "file": file_path, "query": query, "source": source,
                "section_title": section_title, "summary": summary, "result_count": len(results),
            }
            state["broad_pending_lock"] = True

        broad_done = sum(1 for m in state["search_sources"].values() if not m.get("section_title"))
        ret = {
            "source_id":                src_id,
            "source":                   source,
            "result_count":             len(results),
            "summary":                  summary,
            "broad_searches_done":      broad_done,
            "broad_searches_remaining": max(0, cfg.MAX_BROAD_SEARCHES - broad_done),
        }
        if doc_candidates:
            ret["doc_candidates"] = doc_candidates
            ret["doc_candidates_note"] = f"{len(doc_candidates)} 条结果为权威文档候选，可按需调用 ingest 精读"
        return ret
    else:
        with state_lock:
            if src_id not in sec["source_ids"] and src_id not in sec.get("pending_src_ids", []):
                sec.setdefault("pending_src_ids", []).append(src_id)
            state["search_sources"][src_id] = {
                "file": file_path, "query": query, "source": source,
                "section_title": section_title, "summary": summary, "result_count": len(results),
            }
            sec["status"] = "pending_save"

        targeted_done = len(sec.get("source_ids", [])) + len(sec.get("pending_src_ids", []))
        ret = {
            "source_id":         src_id,
            "source":            source,
            "result_count":      len(results),
            "summary":           summary,
            "targeted_done":     targeted_done,
            "targeted_remaining": max(0, cfg.MAX_TARGETED_PER_SECTION - targeted_done),
        }
        if doc_candidates:
            ret["doc_candidates"] = doc_candidates
            ret["doc_candidates_note"] = f"{len(doc_candidates)} 条结果为权威文档候选，可按需调用 ingest 精读"
        return ret


# ── 各源实现（签名统一：query, max_results, cfg → list[{title,url,snippet}]） ──

def _call_tavily(query: str, max_results: int, cfg) -> list:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": cfg.TAVILY_API_KEY, "query": query, "max_results": max_results,
              "include_answer": False, "include_raw_content": False},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])]


def _call_arxiv(query: str, max_results: int, cfg) -> list:
    import feedparser
    quoted = requests.utils.quote(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{quoted}&start=0&max_results={max_results}")
    # arxiv API 官方要求带描述性 User-Agent；不带 UA 的脚本流量是限流重点对象
    headers = {"User-Agent": "Mozilla/5.0 (compatible; searchAgent/1.0)"}
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return [
        {
            "title":   entry.get("title", "").replace("\n", " ").strip(),
            "url":     entry.get("link", ""),
            "snippet": entry.get("summary", "").replace("\n", " ").strip(),
        }
        for entry in feed.entries[:max_results]
    ]


def _call_semantic_scholar(query: str, max_results: int, cfg) -> list:
    headers = {}
    api_key = getattr(cfg, "SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(3):
        wait = (attempt + 1) * 3  # 3s, 6s, 9s
        time.sleep(wait)
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": max_results,
                    "fields": "title,abstract,tldr,openAccessPdf,url,year,citationCount"},
            headers=headers, timeout=30,
        )
        if resp.status_code == 429:
            continue
        resp.raise_for_status()
        data = resp.json()
        results = []
        for p in data.get("data", [])[:max_results]:
            abstract = p.get("abstract") or ""
            if not abstract:
                # abstract 缺失时回落 tldr（S2 一句话摘要）
                tldr = p.get("tldr") or {}
                abstract = tldr.get("text", "") if isinstance(tldr, dict) else ""
            # 附上开放获取 PDF 链接（若有），供下游深挖
            oa = p.get("openAccessPdf") or {}
            oa_url = oa.get("url", "") if isinstance(oa, dict) else ""
            snippet = abstract
            if oa_url:
                snippet = (abstract + f"\n[开放获取PDF] {oa_url}") if abstract else f"[开放获取PDF] {oa_url}"
            results.append({
                "title":   p.get("title", ""),
                "url":     (p.get("url")
                            or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}"),
                "snippet": snippet,
            })
        return results
    raise RuntimeError("Semantic Scholar 429 rate limit，请配置 SEMANTIC_SCHOLAR_API_KEY 以提升限额")


def _call_github(query: str, max_results: int, cfg) -> list:
    gh_bin = getattr(cfg, "GH_BIN", "gh")
    # gh search repos 的多词 query 必须拆开作为独立 token，不能整体加引号
    query_tokens = query.split()
    proc = subprocess.run(
        [gh_bin, "search", "repos"] + query_tokens +
        ["--sort", "stars", "--limit", str(max_results),
         "--json", "fullName,url,description,stargazersCount"],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh search repos 失败: {proc.stderr.strip()}")
    items = json.loads(proc.stdout or "[]")
    results = []
    for i, item in enumerate(items):
        full_name = item.get("fullName", "")
        readme = ""
        if i < 2 and full_name:  # top 2 抓 README 全文
            readme = _fetch_gh_readme(gh_bin, full_name)
        results.append({
            "title":   full_name,
            "url":     item.get("url", ""),
            "snippet": readme or item.get("description") or "",
        })
    return results


def _call_xiaohongshu(query: str, max_results: int, cfg) -> list:
    import shutil
    backend = getattr(cfg, "XHS_BACKEND", "opencli")
    if backend != "opencli":
        raise RuntimeError(f"XHS_BACKEND='{backend}' 暂不支持，仅支持 opencli")
    opencli_bin = shutil.which("opencli")
    if not opencli_bin:
        raise RuntimeError("opencli 未找到，请确认已安装（npm install -g opencli）并在 PATH 中")
    proc = subprocess.run(
        [opencli_bin, "xiaohongshu", "search", query, "-f", "yaml"],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli xiaohongshu search 失败: {proc.stderr.strip()}")
    import yaml
    items = yaml.safe_load(proc.stdout) or []
    results = []
    for i, item in enumerate(items[:max_results]):
        if not isinstance(item, dict):
            continue
        url = item.get("url", item.get("note_url", ""))
        content = ""
        if i < 2 and url:  # top 2 抓笔记正文
            content = _fetch_xhs_note(opencli_bin, url)
        results.append({
            "title":   item.get("title", ""),
            "url":     url,
            "snippet": content or item.get("desc", ""),
        })
    return results


def _call_zhihu(query: str, max_results: int, cfg) -> list:
    import shutil, yaml
    opencli_bin = shutil.which("opencli")
    if not opencli_bin:
        raise RuntimeError("opencli 未找到，请确认已安装（npm install -g opencli）并在 PATH 中")
    proc = subprocess.run(
        [opencli_bin, "zhihu", "search", query, "--limit", str(max_results), "-f", "yaml"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli zhihu search 失败: {proc.stderr.strip()[:200]}")
    items = yaml.safe_load(proc.stdout) or []
    results = []
    fetched = 0
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        content = ""
        if item.get("type") == "answer" and fetched < 2:  # top 2 回答抓全文
            m = re.search(r"/answer/(\d+)", url)
            if m:
                content = _fetch_zhihu_answer(opencli_bin, m.group(1))
                fetched += 1
        results.append({
            "title":   item.get("title", ""),
            "url":     url,
            "snippet": content or f"[{item.get('type','')}] {item.get('author','')} · 赞{item.get('votes',0)}",
        })
    return results


def _call_weixin(query: str, max_results: int, cfg) -> list:
    import shutil, yaml
    opencli_bin = shutil.which("opencli")
    if not opencli_bin:
        raise RuntimeError("opencli 未找到，请确认已安装（npm install -g opencli）并在 PATH 中")
    proc = subprocess.run(
        [opencli_bin, "weixin", "search", query, "--limit", str(max_results), "-f", "yaml"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli weixin search 失败: {proc.stderr.strip()[:200]}")
    items = yaml.safe_load(proc.stdout) or []
    return [
        {
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("summary", ""),
        }
        for item in items[:max_results]
        if isinstance(item, dict)
    ]


def _call_google_scholar(query: str, max_results: int, cfg) -> list:
    import shutil, yaml
    opencli_bin = shutil.which("opencli")
    if not opencli_bin:
        raise RuntimeError("opencli 未找到，请确认已安装（npm install -g opencli）并在 PATH 中")
    proc = subprocess.run(
        [opencli_bin, "google-scholar", "search", query, "--limit", str(max_results), "-f", "yaml"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli google-scholar search 失败: {proc.stderr.strip()[:200]}")
    items = yaml.safe_load(proc.stdout) or []
    return [
        {
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": (
                f"{item.get('authors','').strip()} ({item.get('year','')}) · "
                f"引用 {item.get('cited', 0)} · {item.get('source','')}"
            ).strip(" ·"),
        }
        for item in items[:max_results]
        if isinstance(item, dict)
    ]

def _call_weibo(query: str, max_results: int, cfg) -> list:
    import shutil, yaml
    opencli_bin = shutil.which("opencli")
    if not opencli_bin:
        raise RuntimeError("opencli 未找到，请确认已安装（npm install -g opencli）并在 PATH 中")
    proc = subprocess.run(
        [opencli_bin, "weibo", "search", query, "--limit", str(max_results), "-f", "yaml"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli weibo search 失败: {proc.stderr.strip()[:200]}")
    items = yaml.safe_load(proc.stdout) or []
    return [
        {
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": f"{item.get('author','')} · {item.get('time','')}",
        }
        for item in items[:max_results]
        if isinstance(item, dict)
    ]




_CALLERS: dict = {
    "tavily":           _call_tavily,
    "arxiv":            _call_arxiv,
    # "semantic_scholar": _call_semantic_scholar,  # 已停用：无 key 共享池限流不稳定（函数保留，不注册）
    "github":           _call_github,
    "xiaohongshu":      _call_xiaohongshu,
    "zhihu":            _call_zhihu,
    "weixin":           _call_weixin,
    "google_scholar":   _call_google_scholar,
    "weibo":            _call_weibo,
}


# ── 全文抓取 helper（opencli / gh 二次调用，失败降级为元信息） ───────────────

def _fetch_xhs_note(opencli_bin: str, url: str) -> str:
    """opencli xiaohongshu note 抓笔记正文 content；失败返回空串。"""
    try:
        # opencli 是 npm 的 .CMD 包装，经 cmd.exe 执行时 URL 里的 & 会被当命令分隔符
        # （导致 -f json 参数错乱、输出退化 YAML、returncode=1）。绕开 .CMD 直调 node。
        entry = _opencli_node_entry(opencli_bin) or [opencli_bin]
        proc = subprocess.run(
            entry + ["xiaohongshu", "note", url, "-f", "json"],
            capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return ""
        # note 命令输出编码不稳定（GBK 或 UTF-8），按 bytes 读后依次尝试解码
        text = _decode_cli_output(proc.stdout or b"")
        items = json.loads(text or "[]")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("field") == "content":
                    return it.get("value", "")
        elif isinstance(items, dict):
            return items.get("content") or items.get("desc") or ""
        return ""
    except Exception:
        return ""


def _opencli_node_entry(opencli_bin: str):
    """
    opencli 是 npm 的 .CMD 包装，subprocess 经 cmd.exe 执行时参数里的 & 会被当命令分隔符。
    从 .CMD 批处理解析出 node + main.js 入口，直接调 node 绕开该问题；失败返回 None。
    """
    try:
        if not opencli_bin.lower().endswith((".cmd", ".bat")):
            return None
        with open(opencli_bin, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        m = re.search(r'["\']([^"\']*main\.js)["\']', content)
        if not m:
            return None
        main_js = m.group(1).replace("%dp0%", os.path.dirname(opencli_bin))
        if os.path.exists(main_js):
            return ["node", main_js]
    except Exception:
        pass
    return None


def _decode_cli_output(raw: bytes) -> str:
    """依次尝试 UTF-8 / GBK 解码 CLI 输出，兜底 replacement。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _fetch_zhihu_answer(opencli_bin: str, answer_id: str) -> str:
    """opencli zhihu answer-detail 抓回答全文 content；失败返回空串。"""
    try:
        proc = subprocess.run(
            [opencli_bin, "zhihu", "answer-detail", answer_id, "-f", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        if proc.returncode != 0:
            return ""
        items = json.loads(proc.stdout or "[]")
        if isinstance(items, list) and items:
            d = items[0]
            if isinstance(d, dict):
                return d.get("content", "")
        return ""
    except Exception:
        return ""


def _fetch_gh_readme(gh_bin: str, full_name: str) -> str:
    """gh repo view 抓 README 全文（截取前 500 字）；失败返回空串。"""
    try:
        proc = subprocess.run(
            [gh_bin, "repo", "view", full_name],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout[:500].strip()
    except Exception:
        return ""


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _make_src_id(query: str, source: str = "tavily") -> str:
    h = hashlib.md5(query.encode()).hexdigest()[:8]
    return f"src_{source}_{h}"


def _write_results(src_id, query, source, section_title, results, cfg, run_dir="") -> str:
    search_dir = os.path.join(
        run_dir if run_dir else os.path.dirname(cfg.OUTPUT_DIR), "search_results"
    )
    os.makedirs(search_dir, exist_ok=True)
    path = os.path.join(search_dir, f"{src_id}.json")
    payload = {
        "source_id": src_id, "query": query, "source": source,
        "section_title": section_title, "result_count": len(results), "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _make_summary(results: list) -> str:
    titles = [r["title"] for r in results if r.get("title")][:3]
    return "；".join(titles) if titles else f"{len(results)} 条结果"


# ── 权威文档候选判定 ───────────────────────────────────────────────────────────

_DOC_ENDPOINT_PATTERNS = ("fileDownLoad", "mode=save", "/pdf/", "/download/", "/getfile")

# 默认权威域名（不依赖 config，config 里的列表作为扩展）
_DEFAULT_AUTHORITY_DOMAINS = [
    "arxiv.org", "ssrn.com", "openreview.net", "soa.org",
    "sec.gov", "hkexnews.hk", "cninfo.com.cn", "chinamoney.org.cn",
    "ambest.com", "fitchratings.com", "spglobal.com", "moodys.com",
    "investor.", "q4cdn.com", "dfcfw.com",
    "allianz.com", "swissre.com", "zurich.com", "munichre.com",
    "generali.com", "axa.com", "prudential.com",
    ".gov", ".edu",
]

def _is_doc_candidate(url: str) -> bool:
    """判断 URL 是否为权威文档候选（域名权威 + 下载端点 + .pdf 后缀）。"""
    if not url:
        return False
    try:
        from ._client import get_config as _gc
        cfg = _gc()
        authority_domains = list(_DEFAULT_AUTHORITY_DOMAINS) + list(getattr(cfg, "AUTHORITY_DOMAINS", []))
    except Exception:
        authority_domains = _DEFAULT_AUTHORITY_DOMAINS

    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        return True
    for pat in _DOC_ENDPOINT_PATTERNS:
        if pat in url:
            return True
    for domain in authority_domains:
        if domain in url_lower:
            return True
    return False


def _mark_doc_candidates(results: list, source: str = "") -> tuple[list, list]:
    """给结果列表打 doc_candidate 标记，返回 (标注后的列表, 候选摘要列表)。"""
    out = []
    candidates = []
    for r in results:
        url = r.get("url", "")
        # google_scholar 是学术源，结果天然是论文原文链接，全量标候选
        if source == "google_scholar" or _is_doc_candidate(url):
            r = dict(r, doc_candidate=True)
            candidates.append(url)
        out.append(r)
    return out, candidates


register(SCHEMA, runner)
