#!/usr/bin/env python3
"""Run the full benchmark suite against LoCoMo and/or LongMemEval.

Usage:
    # Run all baselines on LoCoMo (1 sample, 10 QA each for quick test)
    uv run python scripts/run_benchmark.py --dataset locomo --max-samples 1 --max-qa 10

    # Run specific baseline on LongMemEval
    uv run python scripts/run_benchmark.py --dataset longmemeval --baselines b0 b4 --max-samples 5

    # Run full suite
    uv run python scripts/run_benchmark.py --dataset all --max-samples 2 --max-qa 20

    # Quiet mode (minimal output)
    uv run python scripts/run_benchmark.py --dataset locomo --baselines b0 --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy loggers before importing anything else
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from src.evaluation.data_loaders import load_locomo, load_longmemeval
from src.evaluation.runner import run_experiment, _header, _kv, _format_duration
from src.evaluation.metrics import ExperimentResult
from src.evaluation.published_baselines import format_comparison

BASELINE_CONFIGS = {
    "b0": "configs/experiments/b0_no_memory.yaml",
    "b1": "configs/experiments/b1_flat_memory.yaml",
    "b3": "configs/experiments/b3_graph_no_forgetting.yaml",
    "b4": "configs/experiments/b4_graph_with_forgetting.yaml",
}

BASELINE_NAMES = {
    "b0": "B0_no_memory",
    "b1": "B1_flat_memory",
    "b3": "B3_graph_no_forgetting",
    "b4": "B4_graph_with_forgetting",
}

BASELINE_DESCRIPTIONS = {
    "b0": "No external memory (pure LLM)",
    "b1": "Flat memory list, recency-sorted retrieval",
    "b3": "Graph memory with neighbor traversal, no forgetting",
    "b4": "Graph memory with hybrid forgetting policy",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_suite(
    datasets: list[str],
    baselines: list[str],
    max_samples: int | None,
    max_qa: int | None,
    results_dir: str,
    verbose: bool = True,
    run_edit_delete_tests: bool = True,
    use_gpt4o_judge: bool = False,
) -> list[dict]:
    """Run all requested experiments and return summary."""
    suite_start = time.time()
    all_results = []

    total_experiments = len(datasets) * len(baselines)

    _header("SELECTIVE AMNESIA BENCHMARK SUITE", char="*", width=80)
    _kv("Datasets", ", ".join(datasets))
    _kv("Baselines", ", ".join(baselines))
    _kv("Total experiments", total_experiments)
    if max_samples:
        _kv("Max samples/dataset", max_samples)
    if max_qa:
        _kv("Max QA/sample", max_qa)
    _kv("Results directory", results_dir)

    # Describe baselines
    print()
    print("    Baselines:")
    for b in baselines:
        print(f"      {BASELINE_NAMES[b]:<30} {BASELINE_DESCRIPTIONS.get(b, '')}")

    exp_idx = 0
    for dataset in datasets:
        _header(f"DATASET: {dataset.upper()}", char="=")

        if dataset == "locomo":
            samples = load_locomo(max_samples=max_samples)
            total_turns = sum(
                sum(len(s.turns) for s in sample.sessions) for sample in samples
            )
            total_qa = sum(len(s.qa_pairs) for s in samples)
            _kv("Samples loaded", len(samples))
            _kv("Total sessions", sum(len(s.sessions) for s in samples))
            _kv("Total turns", total_turns)
            _kv("Total QA pairs", total_qa)
            if max_qa:
                _kv("QA pairs (capped)", min(total_qa, max_qa * len(samples)))
        elif dataset == "longmemeval":
            samples = load_longmemeval(max_samples=max_samples)
            total_sessions = sum(len(s.sessions) for s in samples)
            _kv("Samples loaded", len(samples))
            _kv("Total sessions", total_sessions)
            _kv("Total QA pairs", len(samples))
            # Show question type distribution
            qtypes = {}
            for s in samples:
                for q in s.qa_pairs:
                    qtypes[q.category] = qtypes.get(q.category, 0) + 1
            if qtypes:
                _kv("Question types", dict(qtypes))
        else:
            print(f"    Unknown dataset: {dataset}")
            continue

        for baseline in baselines:
            exp_idx += 1
            config_path = BASELINE_CONFIGS.get(baseline)
            if not config_path:
                print(f"    Unknown baseline: {baseline}")
                continue

            config = load_config(config_path)
            exp_name = f"{BASELINE_NAMES[baseline]}_{dataset}"

            print(f"\n    >>> Experiment {exp_idx}/{total_experiments}: {exp_name}")
            start = time.time()

            try:
                result = run_experiment(
                    experiment_name=exp_name,
                    samples=samples,
                    config=config,
                    results_dir=results_dir,
                    max_qa_per_sample=max_qa,
                    verbose=verbose,
                    run_edit_delete_tests=run_edit_delete_tests,
                    use_gpt4o_judge=use_gpt4o_judge,
                )
                elapsed = time.time() - start

                summary = {
                    "experiment": exp_name,
                    "dataset": dataset,
                    "baseline": baseline,
                    "overall_f1": round(result.overall_f1, 4),
                    "overall_exact": round(result.overall_exact, 4),
                    "overall_contains": round(result.overall_contains, 4),
                    "recall_at_5": round(result.overall_recall_at_5, 4),
                    "mrr": round(result.overall_mrr, 4),
                    "multi_hop_f1": round(result.multi_hop_f1, 4),
                    "task_success_rate": round(result.overall_task_success_rate, 4),
                    "edit_success": round(result.edit_delete_result.edit_success_rate, 4),
                    "delete_success": round(result.edit_delete_result.delete_success_rate, 4),
                    "locality_score": round(result.edit_delete_result.locality_score, 4),
                    "avg_latency": round(result.avg_latency, 3),
                    "storage_bytes": result.memory_stats.storage_bytes,
                    "f1_by_category": {k: round(v, 4) for k, v in result.f1_by_category().items()},
                    "num_questions": len(result.qa_results),
                    "memory_nodes": result.memory_stats.enabled_nodes,
                    "memory_edges": result.memory_stats.total_edges,
                    "forgotten": result.memory_stats.forgotten_count,
                    "elapsed_seconds": round(elapsed, 1),
                }
                all_results.append(summary)

            except Exception as e:
                logging.getLogger(__name__).error(
                    "Experiment %s failed: %s", exp_name, e, exc_info=True
                )
                all_results.append({"experiment": exp_name, "error": str(e)})

    suite_elapsed = time.time() - suite_start
    print_comparison_table(all_results, suite_elapsed)

    # Save summary
    summary_path = Path(results_dir) / "summary_latest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n    Summary saved to: {summary_path}")

    return all_results


def print_comparison_table(results: list[dict], total_time: float = 0) -> None:
    """Print a rich comparison table of all experiment results."""
    _header("FINAL COMPARISON", char="*", width=120)

    # Answer quality table
    print(f"    ANSWER QUALITY")
    print(
        f"    {'Experiment':<35} "
        f"{'F1':>7} {'Exact':>7} {'Cont.':>7} {'MH-F1':>7} {'TSR':>7} "
        f"{'Nodes':>6} {'Forgot':>6} "
        f"{'Time':>8}"
    )
    print(f"    {'':=<35} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>6} {'':=>6} {'':=>8}")

    for r in results:
        if "error" in r:
            print(f"    {r['experiment']:<35} {'ERROR -- ' + r['error'][:40]}")
            continue
        print(
            f"    {r['experiment']:<35} "
            f"{r['overall_f1']:>7.4f} "
            f"{r['overall_exact']:>7.4f} "
            f"{r['overall_contains']:>7.4f} "
            f"{r.get('multi_hop_f1', 0):>7.4f} "
            f"{r.get('task_success_rate', 0):>7.2%} "
            f"{r.get('memory_nodes', 0):>6} "
            f"{r.get('forgotten', 0):>6} "
            f"{_format_duration(r.get('elapsed_seconds', 0)):>8}"
        )

    # Retrieval & memory ops table
    has_retrieval = any(r.get("recall_at_5", 0) > 0 for r in results if "error" not in r)
    if has_retrieval:
        print()
        print(f"    RETRIEVAL & MEMORY OPERATIONS")
        print(
            f"    {'Experiment':<35} "
            f"{'R@5':>7} {'MRR':>7} "
            f"{'Edit%':>7} {'Del%':>7} {'Local':>7} "
            f"{'Lat(s)':>7} {'Store':>10}"
        )
        print(f"    {'':=<35} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>10}")

        for r in results:
            if "error" in r:
                continue
            storage = r.get("storage_bytes", 0)
            storage_str = f"{storage:,}" if storage else "N/A"
            print(
                f"    {r['experiment']:<35} "
                f"{r.get('recall_at_5', 0):>7.4f} "
                f"{r.get('mrr', 0):>7.4f} "
                f"{r.get('edit_success', 0):>7.2%} "
                f"{r.get('delete_success', 0):>7.2%} "
                f"{r.get('locality_score', 0):>7.4f} "
                f"{r.get('avg_latency', 0):>7.2f} "
                f"{storage_str:>10}"
            )

    # Per-category F1 breakdown
    categories = set()
    for r in results:
        categories.update(r.get("f1_by_category", {}).keys())

    if categories:
        cats = sorted(categories)
        print()
        print(f"    {'':=<90}")
        print(f"    F1 BY CATEGORY")
        print(f"    {'':=<90}")
        cat_header = "    " + f"{'Experiment':<35} " + " ".join(f"{c:>13}" for c in cats)
        print(cat_header)
        print(f"    {'':=<35} " + " ".join(f"{'':=>13}" for _ in cats))
        for r in results:
            if "error" in r:
                continue
            by_cat = r.get("f1_by_category", {})
            vals = " ".join(f"{by_cat.get(c, 0.0):>13.4f}" for c in cats)
            print(f"    {r['experiment']:<35} {vals}")

    # Delta analysis (compare each baseline against b0)
    b0_results = [r for r in results if r.get("baseline") == "b0" and "error" not in r]
    other_results = [r for r in results if r.get("baseline") != "b0" and "error" not in r]

    if b0_results and other_results:
        print()
        print(f"    {'':=<90}")
        print(f"    DELTA vs B0 (no memory)")
        print(f"    {'':=<90}")
        print(
            f"    {'Experiment':<35} "
            f"{'dF1':>8} {'dExact':>8} {'dCont.':>8} {'dTSR':>8}"
        )
        print(f"    {'':=<35} {'':=>8} {'':=>8} {'':=>8} {'':=>8}")

        for r in other_results:
            b0 = next(
                (b for b in b0_results if b.get("dataset") == r.get("dataset")),
                None,
            )
            if b0:
                df1 = r["overall_f1"] - b0["overall_f1"]
                dem = r["overall_exact"] - b0["overall_exact"]
                dcm = r["overall_contains"] - b0["overall_contains"]
                dtsr = r.get("task_success_rate", 0) - b0.get("task_success_rate", 0)
                sign_f1 = "+" if df1 >= 0 else ""
                sign_em = "+" if dem >= 0 else ""
                sign_cm = "+" if dcm >= 0 else ""
                sign_tsr = "+" if dtsr >= 0 else ""
                print(
                    f"    {r['experiment']:<35} "
                    f"{sign_f1}{df1:>7.4f} "
                    f"{sign_em}{dem:>7.4f} "
                    f"{sign_cm}{dcm:>7.4f} "
                    f"{sign_tsr}{dtsr:>7.2%}"
                )

    # Published baseline comparison
    datasets_in_results = {r.get("dataset") for r in results if "error" not in r}
    for dataset in sorted(datasets_in_results):
        our = {}
        for r in results:
            if r.get("dataset") != dataset or "error" in r:
                continue
            scores = dict(r.get("f1_by_category", {}))
            scores["overall"] = r.get("overall_f1", 0.0)
            our[r["experiment"]] = scores
        if our:
            print()
            print(f"    COMPARISON WITH PUBLISHED BASELINES ({dataset.upper()})")
            print(format_comparison(our, dataset))

    if total_time:
        print(f"\n    Total suite time: {_format_duration(total_time)}")

    print(f"    {'*' * 120}")


def main():
    parser = argparse.ArgumentParser(
        description="Selective Amnesia Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["locomo", "longmemeval", "all"],
        default="locomo",
        help="Which dataset(s) to evaluate on",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_CONFIGS.keys()),
        default=list(BASELINE_CONFIGS.keys()),
        help="Which baselines to run (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for quick testing)",
    )
    parser.add_argument(
        "--max-qa",
        type=int,
        default=None,
        help="Max QA pairs per sample (for quick testing)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save result JSONs (default: results/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (no per-question details)",
    )
    parser.add_argument(
        "--no-edit-delete",
        action="store_true",
        help="Skip edit/delete/locality tests (faster runs)",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Enable GPT-4o judge for LongMemEval accuracy",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    datasets = ["locomo", "longmemeval"] if args.dataset == "all" else [args.dataset]

    run_suite(
        datasets=datasets,
        baselines=args.baselines,
        max_samples=args.max_samples,
        max_qa=args.max_qa,
        results_dir=args.results_dir,
        verbose=not args.quiet,
        run_edit_delete_tests=not args.no_edit_delete,
        use_gpt4o_judge=args.use_judge,
    )


if __name__ == "__main__":
    main()
