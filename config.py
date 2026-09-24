import os


def _load_dotenv(path: str) -> None:
    """从 .env 文件加载环境变量到 os.environ（不覆盖已存在的变量，系统环境变量优先）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# DeepSeek via Anthropic-compatible API
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 多源搜索配置
SEARCH_SOURCES = ["tavily", "xiaohongshu", "zhihu", "weixin", "weibo", "github", "arxiv", "google_scholar"]
XHS_BACKEND = os.getenv("XHS_BACKEND", "opencli")          # opencli | mcp
ZHIHU_COOKIE = os.getenv("ZHIHU_COOKIE", "")               # z_c0=...; d_c0=...
GH_BIN = os.getenv("GH_BIN", "gh")                         # gh CLI 可执行路径
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # 可选，无 key 限 1 req/s


MODEL = os.getenv("MODEL", "your-model-name")
CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL", MODEL)

# Research limits
MAX_SECTIONS = 10
MAX_REFLECT_PER_SECTION = 5
MAX_AGENT_STEPS = 100
MAX_BROAD_SEARCHES = 8        # 宽泛搜索（无 section_title）最大并行次数，一批即锁
MAX_TARGETED_PER_SECTION = 5  # 每章节定向补搜上限

# 非 tavily 辅助源每轮配额上限（tavily 是唯一主源，其余源限量辅助）
MAX_AUX_SOURCES_BROAD    = 3  # 宽泛阶段：8 条候选里非 tavily 源最多 3 条
MAX_AUX_SOURCES_TARGETED = 1  # 定向补搜：3 条候选里非 tavily 源最多 1 条

# 分级限速（搜索源风险等级 + 限速参数）
_RISK_HIGH   = ["xiaohongshu", "zhihu", "weibo"]
_RISK_MID    = ["google_scholar", "weixin"]
_RISK_ARXIV  = ["arxiv"]   # arxiv 官方 API 限速 3s/请求
_RISK_GITHUB = ["github"]  # search API 认证 30 req/min（未认证 10）
_RISK_TAVILY = ["tavily"]  # dev key 100 RPM（免费）

RATE_LIMITS = {
    "risk_high": {                  # 复用浏览器登录态，bot 识别敏感
        "sources":               _RISK_HIGH,
        "min_interval":          3,     # 请求间隔下限（秒）
        "max_interval":          8,     # 请求间隔上限（秒，随机 jitter）
        "concurrency":           1,     # 同站点严格串行
        "qph":                   25,    # 每小时最大请求数
        "circuit_break_after":   3,     # 连续失败 N 次后熔断
        "circuit_break_minutes": 60,    # 熔断时长（分钟）
    },
    "risk_mid": {                   # 无登录但反爬敏感（验证码/封 IP）
        "sources":               _RISK_MID,
        "min_interval":          5,
        "max_interval":          15,
        "concurrency":           1,
        "qph":                   40,
        "circuit_break_after":   3,
        "circuit_break_minutes": 60,
    },
    "github_search": {              # GitHub Search API：认证 30 req/min，未认证 10 req/min
        "sources":               _RISK_GITHUB,
        "min_interval":          2,     # 认证 30 req/min = 1 req/2s
        "max_interval":          4,
        "concurrency":           1,     # search 端点限速严格，串行
        "qph":                   1800,  # 30 × 60
        "circuit_break_after":   3,
        "circuit_break_minutes": 15,
    },
    "tavily_api": {                 # dev key 100 RPM（免费；production 1000 RPM）
        "sources":               _RISK_TAVILY,
        "min_interval":          1,     # 100 RPM ≈ 1 req/0.6s，取 1s 保守
        "max_interval":          2,
        "concurrency":           2,     # 控制瞬时并发（原 5 过激进）
        "qph":                   6000,  # 100 × 60
        "circuit_break_after":   5,
        "circuit_break_minutes": 10,
    },
    "arxiv_api": {                  # arxiv 官方 API：3s/请求 硬限速
        "sources":               _RISK_ARXIV,
        "min_interval":          3.5,   # 3s 官方门槛 + 余量
        "max_interval":          6,
        "concurrency":           1,     # 严格串行
        "qph":                   600,   # 3s 间隔理论 1200/h，保守取半
        "circuit_break_after":   3,
        "circuit_break_minutes": 15,
    },
}

MAX_CONCURRENT_LLM_CALLS = 5  # 全局 LLM 并发上限（主+子 LLM 共用）
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")
TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")

# 文档检索配置
AUTHORITY_DOMAINS = [
    # 学术 / 预印本
    "arxiv.org", "ssrn.com", "openreview.net", "soa.org",
    # 官方 IR / 公司
    "investor.", "allianz.com", "swissre.com", "zurich.com",
    "munichre.com", "generali.com", "axa.com", "prudential.com",
    # 监管披露 / 财报
    "cninfo.com.cn", "sec.gov", "hkexnews.hk", "chinamoney.org.cn",
    "dfcfw.com", "q4cdn.com",
    # 评级机构
    "ambest.com", "fitchratings.com", "spglobal.com", "moodys.com",
    # 政府 / 高校
    ".gov", ".edu",
]
MAX_DEEP_READS_PER_RUN = 5       # 整次 run 最多 ingest 几份文档
DEEP_READ_MAX_CHARS    = 4000    # 单次 retrieve 返回正文上限（字符）
DEEP_READ_CHUNK_SIZE   = 1500    # 分块每块字符数
