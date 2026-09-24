#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单条 md 报告 → DeepResearchBench RACE 评估 pipeline

用法:
    python eval_md.py --md <report.md>            # 自动匹配 benchmark 任务
    python eval_md.py --md <report.md> --id 2     # 显式指定任务 id
    python eval_md.py --md <report.md> --name xyz # 自定义 run_name（默认取 md 文件名）

流程:
    md → 匹配 query.jsonl 任务 → 转单条 jsonl → raw_data/<run_name>.jsonl
    → 复制到 cleaned_data（绕过清洗）→ 构造单条 query 文件 → 跑 RACE → 打印分数

保证: 写入前检查目标路径不存在, 绝不覆盖已有文件, 不修改 benchmark 任何代码。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

BENCH = Path(os.environ.get("DEEP_RESEARCH_BENCH_DIR", "deep_research_bench")).expanduser().resolve()
QUERY_FILE = BENCH / "data" / "prompt_data" / "query.jsonl"
RAW_DIR = BENCH / "data" / "test_data" / "raw_data"
CLEANED_DIR = BENCH / "data" / "test_data" / "cleaned_data"
RESULTS_DIR = BENCH / "results" / "race"
TMP_DIR = BENCH / ".eval_tmp"

# 评分用的 API key 从 config 读取（config 自动加载 .env 或系统环境变量）
OPENAI_API_KEY = config.API_KEY


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def match_task(title, queries):
    """用标题连续子串在 prompt 中出现的加权次数匹配任务, 返回 [(score, task)]"""
    t = title.replace(" ", "")
    results = []
    for q in queries:
        p = q["prompt"].replace(" ", "")
        score = 0
        for n in (8, 6, 4):
            for i in range(len(t) - n + 1):
                if t[i:i + n] in p:
                    score += n * n
        results.append((score, q))
    results.sort(key=lambda x: -x[0])
    return results


def ensure_not_exists(run_name):
    paths = [
        RAW_DIR / f"{run_name}.jsonl",
        CLEANED_DIR / f"{run_name}.jsonl",
        RESULTS_DIR / run_name,
    ]
    conflicts = [str(p) for p in paths if p.exists()]
    if conflicts:
        print("[ERROR] 以下路径已存在, 拒绝覆盖:")
        for c in conflicts:
            print(f"  {c}")
        print("请换一个 --name")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="md 报告 → benchmark RACE 单条评估")
    ap.add_argument("--md", required=True, help="报告 md 文件路径")
    ap.add_argument("--id", type=int, default=None, help="显式指定 benchmark 任务 id")
    ap.add_argument("--name", default=None, help="run_name, 默认 md 文件名(去后缀)")
    ap.add_argument("--dry-run", action="store_true", help="只做匹配和转换, 不跑评估")
    args = ap.parse_args()

    md_path = Path(args.md).resolve()
    if not md_path.exists():
        print(f"[ERROR] md 文件不存在: {md_path}")
        sys.exit(1)
    with open(md_path, encoding="utf-8") as f:
        article = f.read()

    queries = load_jsonl(QUERY_FILE)

    # ---- 1. 匹配任务 ----
    if args.id is not None:
        task = next((q for q in queries if q.get("id") == args.id), None)
        if task is None:
            print(f"[ERROR] query.jsonl 里没有 id={args.id} 的任务")
            sys.exit(1)
    else:
        title = md_path.stem
        # 去掉 md 第一行标题里的"报告"等干扰词后作为匹配文本
        ranked = match_task(title, queries)
        top_score, top_task = ranked[0]
        if top_score <= 0:
            print("[ERROR] 无法匹配 benchmark 题库中的任何任务 (score=0)")
            print(f"  报告标题: {title}")
            print("  这条报告不在 100 题题库里, 无法评估。可用 --id 显式指定。")
            sys.exit(1)
        second_score = ranked[1][0] if len(ranked) > 1 else 0
        if second_score >= top_score * 0.75:
            print("[WARN] 匹配有歧义, 候选如下:")
            for s, q in ranked[:5]:
                print(f"  score={s} id={q['id']}: {q['prompt'][:50]}")
            print("请用 --id 显式指定后重跑")
            sys.exit(1)
        task = top_task

    tid, prompt, language = task["id"], task["prompt"], task["language"]
    print(f"[OK] 匹配任务 id={tid} ({language})")
    print(f"     prompt: {prompt[:60]}...")

    # ---- 2. run_name 唯一性 ----
    run_name = args.name or f"eval-id{tid}-{datetime.now():%Y%m%d_%H%M%S}"
    ensure_not_exists(run_name)
    print(f"[OK] run_name = {run_name}")

    # ---- 3. 写 raw_data 单条 jsonl ----
    raw_out = RAW_DIR / f"{run_name}.jsonl"
    with open(raw_out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": tid, "prompt": prompt, "article": article},
                           ensure_ascii=False) + "\n")
    print(f"[OK] 写入 {raw_out}")

    # ---- 4. 复制到 cleaned_data (绕过清洗; --skip_cleaning 有 bug 会跳过整个流程) ----
    cleaned_out = CLEANED_DIR / f"{run_name}.jsonl"
    shutil.copyfile(raw_out, cleaned_out)
    print(f"[OK] 复制到 {cleaned_out}")

    # ---- 5. 构造单条 query 文件 (--limit 1 按 query 顺序取第一条) ----
    TMP_DIR.mkdir(exist_ok=True)
    tmp_query = TMP_DIR / f"query_{run_name}.jsonl"
    with open(tmp_query, "w", encoding="utf-8") as f:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # ---- 6. 跑 RACE ----
    out_dir = RESULTS_DIR / run_name
    cmd = [
        sys.executable, "-u", "deepresearch_bench_race.py", run_name,
        "--raw_data_dir", str(RAW_DIR.relative_to(BENCH)),
        "--max_workers", "10",
        "--query_file", str(tmp_query),
        "--output_dir", str(out_dir),
        "--limit", "1",
        "--only_zh" if language == "zh" else "--only_en",
    ]
    env = dict(os.environ)
    env["LLM_BACKEND"] = "openai"
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = OPENAI_API_KEY
    if not env.get("OPENAI_API_KEY"):
        print("[ERROR] 未配置 OPENAI_API_KEY: 请填 eval_md.py 顶部的 OPENAI_API_KEY, 或先设置系统环境变量")
        sys.exit(1)
    print(f"[RUN] {' '.join(cmd)}")

    if args.dry_run:
        print("[DRY-RUN] 跳过评估执行")
        return

    proc = subprocess.run(cmd, cwd=str(BENCH), env=env)
    if proc.returncode != 0:
        print(f"[ERROR] benchmark 退出码 {proc.returncode}")
        sys.exit(proc.returncode)

    # ---- 7. 打印结果 ----
    result_file = out_dir / "race_result.txt"
    if result_file.exists():
        print("\n===== 评估结果 =====")
        print(result_file.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] 未生成 {result_file}, 查看 output 日志")


if __name__ == "__main__":
    main()
