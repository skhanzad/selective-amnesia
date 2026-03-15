#!/usr/bin/env python3
"""Run the full selective-amnesia experiment suite.

Usage:
    # Full evaluation (slow — runs all QA through Ollama)
    python -m experiments.run_all

    # Quick smoke test (2 samples, 5 QA each)
    python -m experiments.run_all --quick

    # Only LoCoMo
    python -m experiments.run_all --locomo-only

    # Only LongMemEval
    python -m experiments.run_all --longmemeval-only

    # Specific forgetting presets
    python -m experiments.run_all --presets none default aggressive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import DEFAULT_MODEL, DEFAULT_PROVIDER, RESULTS_DIR
from experiments.run_locomo import evaluate_locomo
from experiments.run_longmemeval import evaluate_longmemeval
from experiments.latex_output import generate_all_tables


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective-amnesia experiment suite")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--presets", nargs="+", default=["none", "default", "aggressive"],
                    help="Forgetting presets to evaluate")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--locomo-only", action="store_true")
    p.add_argument("--longmemeval-only", action="store_true")
    p.add_argument("--quick", action="store_true",
                    help="Smoke test: 2 LoCoMo samples (5 QA each), 10 LongMemEval instances")
    p.add_argument("--max-locomo-samples", type=int, default=None)
    p.add_argument("--max-locomo-qas", type=int, default=None)
    p.add_argument("--max-longmemeval", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.max_locomo_samples = args.max_locomo_samples or 2
        args.max_locomo_qas = args.max_locomo_qas or 5
        args.max_longmemeval = args.max_longmemeval or 10

    locomo_results = []
    longmemeval_results = []

    # ---- LoCoMo ----
    if not args.longmemeval_only:
        print("=" * 60)
        print("LoCoMo Evaluation")
        print("=" * 60)
        for preset in args.presets:
            print(f"\n--- Preset: {preset} ---")
            res = evaluate_locomo(
                forget_preset=preset,
                model=args.model,
                provider=args.provider,
                top_k=args.top_k,
                max_samples=args.max_locomo_samples,
                max_qas_per_sample=args.max_locomo_qas,
            )
            locomo_results.append(res)

    # ---- LongMemEval ----
    if not args.locomo_only:
        print("\n" + "=" * 60)
        print("LongMemEval Evaluation")
        print("=" * 60)
        for preset in args.presets:
            print(f"\n--- Preset: {preset} ---")
            res = evaluate_longmemeval(
                forget_preset=preset,
                model=args.model,
                provider=args.provider,
                top_k=args.top_k,
                max_instances=args.max_longmemeval,
            )
            longmemeval_results.append(res)

    # ---- LaTeX output ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = RESULTS_DIR / "tables.tex"
    tex = generate_all_tables(locomo_results, longmemeval_results, output_path=str(tex_path))

    print("\n" + "=" * 60)
    print("LaTeX Tables")
    print("=" * 60)
    print(tex)

    # ---- Summary JSON ----
    summary = {
        "locomo": [
            {
                "method": r["method"],
                "per_category": r["per_category"],
                "overall": r["overall"],
            }
            for r in locomo_results
        ],
        "longmemeval": [
            {
                "method": r["method"],
                "per_type": r["per_type"],
                "task_avg": r["task_avg"],
                "overall": r["overall"],
                "abstention": r["abstention"],
            }
            for r in longmemeval_results
        ],
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
