#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FACT 评估 runner：无 Jina key 时使用评测工具的默认抓取策略。

用法:
    DEEP_RESEARCH_BENCH_DIR=/path/to/deep_research_bench \
      python fact_runner.py --raw data/test_data/raw_data/<name>.jsonl --name <run_name>
"""
import argparse
import os
import subprocess
import sys

BENCH = os.path.abspath(os.path.expanduser(os.environ.get("DEEP_RESEARCH_BENCH_DIR", "deep_research_bench")))

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="raw_data jsonl 路径（相对 benchmark 根）")
ap.add_argument("--name", required=True, help="run_name，结果写 results/fact/<name>/")
args = ap.parse_args()

api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("[ERROR] 请通过 API_KEY 或 OPENAI_API_KEY 环境变量提供评测模型凭证")
    sys.exit(1)

env = dict(os.environ)
env["LLM_BACKEND"] = "openai"
env["OPENAI_API_KEY"] = api_key
env["FACT_MODEL"] = os.environ.get("FACT_MODEL", os.environ.get("MODEL", "your-model-name"))
env.pop("JINA_API_KEY", None)  # 无 key → 免费 20 RPM 档

run_name = args.name
raw = args.raw
outdir = f"results/fact/{run_name}"
os.makedirs(os.path.join(BENCH, outdir), exist_ok=True)

steps = [
    [sys.executable, "-u", "-m", "utils.extract",
     "--raw_data_path", raw, "--output_path", f"{outdir}/extracted.jsonl",
     "--query_data_path", "data/prompt_data/query.jsonl", "--n_total_process", "1"],
    [sys.executable, "-u", "-m", "utils.deduplicate",
     "--raw_data_path", f"{outdir}/extracted.jsonl", "--output_path", f"{outdir}/deduplicated.jsonl",
     "--query_data_path", "data/prompt_data/query.jsonl", "--n_total_process", "1"],
    [sys.executable, "-u", "-m", "utils.scrape",
     "--raw_data_path", f"{outdir}/deduplicated.jsonl", "--output_path", f"{outdir}/scraped.jsonl",
     "--n_total_process", "5"],
    [sys.executable, "-u", "-m", "utils.validate",
     "--raw_data_path", f"{outdir}/scraped.jsonl", "--output_path", f"{outdir}/validated.jsonl",
     "--query_data_path", "data/prompt_data/query.jsonl", "--n_total_process", "1"],
    [sys.executable, "-u", "-m", "utils.stat",
     "--input_path", f"{outdir}/validated.jsonl", "--output_path", f"{outdir}/fact_result.txt"],
]

for s in steps:
    print(f"\n===== {' '.join(s[3:5])} =====", flush=True)
    r = subprocess.run(s, cwd=BENCH, env=env)
    if r.returncode != 0:
        print(f"[FAILED] {' '.join(s)}", flush=True)
        sys.exit(r.returncode)

res = os.path.join(BENCH, outdir, "fact_result.txt")
if os.path.exists(res):
    print("\n===== FACT 结果 =====", flush=True)
    print(open(res, encoding="utf-8").read())
else:
    print("[WARN] 未生成 fact_result.txt")
