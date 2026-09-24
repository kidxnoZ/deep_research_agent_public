#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量跑 DeepResearchBench 前 40 条中文题（剔除已测），串行：agent 合成 → eval RACE。

用法:
    python run_all.py

结果:
    - 每条 RACE 结果单独存 results/race/eval-id{id}-batch/
    - 总日志追加写 run_all.log
"""
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 强制 stdout 用 UTF-8：Windows 默认 GBK，print 到特殊字符（如 ²）会 UnicodeEncodeError
# （agent.run 内部的 _Tee 包装的是这个 reconfigure 后的 stdout，所以合成过程也一并覆盖）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import agent
from tools.registry import get_state

# ── 配置 ──────────────────────────────────────────────────────────────
SKIP = {1, 2, 3, 4, 5, 6, 7, 11, 18, 29}          # 之前手动测过 race 分的 id
MAX_ID = 40                            # 前 40 条中文题
BENCH_DIR = os.path.abspath(os.path.expanduser(os.environ.get("DEEP_RESEARCH_BENCH_DIR", "deep_research_bench")))
QUERY_FILE = os.path.join(BENCH_DIR, "data", "prompt_data", "query.jsonl")
RACE_DIR = os.path.join(BENCH_DIR, "results", "race")
LOG_PATH = os.path.join(PROJECT_ROOT, "run_all.log")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_queries() -> list:
    qs = []
    with open(QUERY_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q.get("language") == "zh" and q["id"] <= MAX_ID and q["id"] not in SKIP:
                qs.append(q)
    qs.sort(key=lambda x: x["id"])
    return qs


def read_race_score(run_name: str) -> dict:
    """读 results/race/<run_name>/race_result.txt，返回 {dimension: score}"""
    path = os.path.join(RACE_DIR, run_name, "race_result.txt")
    if not os.path.exists(path):
        return {}
    score = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(": ", 1)
            if len(parts) == 2:
                try:
                    score[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return score


def main() -> None:
    queries = load_queries()
    ids = [q["id"] for q in queries]
    log(f"待测 {len(queries)} 条: {ids}")
    log("=" * 60)

    for i, q in enumerate(queries, 1):
        tid = q["id"]
        run_name = f"eval-id{tid}-batch"
        # 增量跳过：已有 race 结果的题直接跳过（断点续跑，重启不重跑）
        if os.path.exists(os.path.join(RACE_DIR, run_name, "race_result.txt")):
            log(f"[{i}/{len(queries)}] id={tid} 已有 eval 结果，跳过")
            continue
        log(f"[{i}/{len(queries)}] id={tid} 开始合成 | prompt: {q['prompt'][:40]}...")

        # ── ① agent 合成 ────────────────────────────────────────────
        try:
            agent.run(q["prompt"])
        except Exception as e:
            # agent.run 内部重定向 stdout 到 _Tee，异常时手动恢复
            sys.stdout = sys.__stdout__
            log(f"[{i}/{len(queries)}] id={tid} 生成失败（异常）: {type(e).__name__}: {e}")
            continue

        run_dir = get_state().get("run_dir", "")
        reports = glob.glob(os.path.join(run_dir, "report_*.md"))
        if not reports:
            log(f"[{i}/{len(queries)}] id={tid} 生成失败（未产出报告 md，run_dir={run_dir}）")
            continue
        log(f"[{i}/{len(queries)}] id={tid} 生成成功: {os.path.basename(reports[0])}")

        # ── ② eval RACE ─────────────────────────────────────────────
        try:
            # eval_md.py 是独立子进程，不吃本进程的 stdout reconfigure，需单独设 UTF-8
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "eval_md.py"), "--md", reports[0],
                 "--id", str(tid), "--name", run_name],
                capture_output=True, text=True, encoding="utf-8",
                env=env,
            )
            # 把 eval 过程日志透出到前台（去掉末尾已由我们解析的分数段）
            if r.stdout:
                print(r.stdout.strip()[-1500:], flush=True)
        except Exception as e:
            log(f"[{i}/{len(queries)}] id={tid} eval 异常: {type(e).__name__}: {e}")
            continue

        # ── ③ 读分数并汇报 ──────────────────────────────────────────
        score = read_race_score(run_name)
        if not score:
            log(f"[{i}/{len(queries)}] id={tid} eval 失败（无 race_result.txt，exit={r.returncode}）")
            continue
        overall = score.get("Overall Score", "?")
        log(f"[{i}/{len(queries)}] id={tid} eval 成功 | Overall={overall:.4f} | "
            f"Comp={score.get('Comprehensiveness', '?')} Ins={score.get('Insight', '?')} "
            f"IF={score.get('Instruction Following', '?')} Read={score.get('Readability', '?')}")

    log("=" * 60)
    log("全部完成")


if __name__ == "__main__":
    main()
