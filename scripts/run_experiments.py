#!/usr/bin/env python3
"""Full experiment suite with graph caching, published baseline comparisons, and export.

Usage:
    # Quick smoke test
    uv run python scripts/run_experiments.py --dataset locomo --baselines b0 b4 --max-samples 1 --max-qa 5

    # Full LoCoMo evaluation
    uv run python scripts/run_experiments.py --dataset locomo --baselines b0 b1 b3 b4

    # LongMemEval with GPT-4o judge
    uv run python scripts/run_experiments.py --dataset longmemeval --baselines b4 --max-samples 2 --use-judge

    # Export as LaTeX
    uv run python scripts/run_experiments.py --dataset all --baselines b0 b4 --output-format latex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy loggers
for _name in ("httpx", "httpcore", "langchain", "langsmith", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from src.evaluation.data_loaders import BenchmarkSample, load_locomo, load_longmemeval
from src.evaluation.metrics import ExperimentResult
from src.evaluation.published_baselines import (
    LOCOMO_BASELINES,
    LONGMEMEVAL_BASELINES,
    format_comparison,
)
from src.evaluation.runner import (
    _format_duration,
    _header,
    _kv,
    ingest_conversations,
    evaluate_qa,
    get_memory_stats,
    evaluate_memory_ops,
    run_experiment,
)
from src.memory.graph_store import GraphStore
from src.memory.retriever import get_retriever

logger = logging.getLogger(__name__)

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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _graph_cache_path(results_dir: str, exp_name: str, sample_id: str) -> Path:
    return Path(results_dir) / "graphs" / f"{exp_name}_{sample_id}.json"


def save_graph(graph_store: GraphStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph_store.to_dict(), f, indent=2, default=str)


def load_graph(path: Path) -> GraphStore | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return GraphStore.from_dict(data)


def run_single_experiment(
    exp_name: str,
    dataset: str,
    baseline: str,
    samples: list[BenchmarkSample],
    config: dict,
    results_dir: str,
    max_qa: int | None,
    verbose: bool,
    run_edit_delete: bool,
    use_judge: bool,
    skip_ingest: bool,
    force_ingest: bool,
) -> dict:
    """Run a single experiment with optional graph caching."""
    use_memory = config.get("retrieval", {}).get("mode", "none") != "none"

    # If no caching needed, delegate to run_experiment directly
    if not use_memory or (not skip_ingest and not force_ingest):
        result = run_experiment(
            experiment_name=exp_name,
            samples=samples,
            config=config,
            results_dir=results_dir,
            max_qa_per_sample=max_qa,
            verbose=verbose,
            run_edit_delete_tests=run_edit_delete,
            use_gpt4o_judge=use_judge,
        )
        return _result_to_summary(result, exp_name, dataset, baseline)

    # With graph caching: check for cached graphs
    from src.evaluation.runner import _build_llm

    llm = _build_llm(config)
    retriever = get_retriever(config.get("retrieval", {}).get("mode", "flat"), config)
    all_qa_results = []

    for i, sample in enumerate(samples):
        cache_path = _graph_cache_path(results_dir, exp_name, sample.sample_id)

        graph_store = None
        turn_count = 0

        if skip_ingest and not force_ingest:
            graph_store = load_graph(cache_path)
            if graph_store:
                turn_count = sum(
                    sum(1 for t in s.turns if t.speaker == "assistant")
                    for s in sample.sessions
                )
                if verbose:
                    print(f"    Loaded cached graph for {sample.sample_id}: "
                          f"{graph_store.node_count()} nodes, {graph_store.edge_count()} edges")

        if graph_store is None:
            graph_store = GraphStore()
            turn_count, _ = ingest_conversations(
                sample, graph_store, llm, config,
                use_memory_extraction=True, verbose=verbose,
            )
            save_graph(graph_store, cache_path)
            if verbose:
                print(f"    Saved graph to {cache_path}")

        # Trim QA
        qa_pairs = sample.qa_pairs
        if max_qa:
            qa_pairs = qa_pairs[:max_qa]

        eval_sample = BenchmarkSample(
            sample_id=sample.sample_id,
            source=sample.source,
            sessions=sample.sessions,
            qa_pairs=qa_pairs,
        )

        qa_results, _ = evaluate_qa(
            eval_sample, graph_store, llm, retriever,
            turn_count, config, verbose=verbose,
            use_gpt4o_judge=use_judge,
        )
        all_qa_results.extend(qa_results)

    # Build ExperimentResult
    result = ExperimentResult(
        experiment_name=exp_name,
        dataset=dataset,
        retrieval_mode=config.get("retrieval", {}).get("mode", "none"),
        forgetting_policy=config.get("forgetting", {}).get("policy", "none"),
        qa_results=all_qa_results,
    )

    return _result_to_summary(result, exp_name, dataset, baseline)


def _result_to_summary(
    result: ExperimentResult, exp_name: str, dataset: str, baseline: str
) -> dict:
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
    }
    if result.overall_judge_accuracy is not None:
        summary["judge_accuracy"] = round(result.overall_judge_accuracy, 4)
    return summary


# ── Output formatting ────────────────────────────────────────────

def _build_our_results(
    all_results: list[dict], dataset: str
) -> dict[str, dict[str, float]]:
    """Convert experiment summaries to format expected by format_comparison."""
    our = {}
    for r in all_results:
        if r.get("dataset") != dataset or "error" in r:
            continue
        name = r["experiment"]
        scores = dict(r.get("f1_by_category", {}))
        scores["overall"] = r.get("overall_f1", 0.0)
        # For LongMemEval, prefer judge accuracy if available
        if dataset == "longmemeval" and "judge_accuracy" in r:
            scores["overall"] = r["judge_accuracy"]
        our[name] = scores
    return our


def print_published_comparison(all_results: list[dict], datasets: list[str]) -> None:
    for dataset in datasets:
        our = _build_our_results(all_results, dataset)
        if not our:
            continue
        _header(f"COMPARISON WITH PUBLISHED BASELINES ({dataset.upper()})")
        print(format_comparison(our, dataset))
        print()


def export_json(all_results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


def export_latex(all_results: list[dict], datasets: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for dataset in datasets:
        baselines = LOCOMO_BASELINES if dataset == "locomo" else LONGMEMEVAL_BASELINES
        our = _build_our_results(all_results, dataset)
        if not our:
            continue

        # Collect columns
        all_cats: set[str] = set()
        for v in list(baselines.values()) + list(our.values()):
            all_cats.update(k for k in v if k != "overall")
        cats = sorted(all_cats)
        columns = cats + ["overall"]
        n_cols = len(columns) + 1  # +1 for system name

        lines.append(f"% {dataset.upper()} Results")
        lines.append(f"\\begin{{tabular}}{{l{'c' * len(columns)}}}")
        lines.append("\\toprule")
        col_headers = " & ".join(c.replace("_", " ").title() for c in columns)
        lines.append(f"System & {col_headers} \\\\")
        lines.append("\\midrule")

        for name, scores in baselines.items():
            vals = " & ".join(f"{scores.get(c, 0.0):.3f}" for c in columns)
            lines.append(f"{name} & {vals} \\\\")

        lines.append("\\midrule")
        for name, scores in our.items():
            vals = " & ".join(f"{scores.get(c, 0.0):.3f}" for c in columns)
            lines.append(f"\\textbf{{{name}}} & {vals} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to: {output_path}")


def export_markdown(all_results: list[dict], datasets: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for dataset in datasets:
        baselines = LOCOMO_BASELINES if dataset == "locomo" else LONGMEMEVAL_BASELINES
        our = _build_our_results(all_results, dataset)
        if not our:
            continue

        all_cats: set[str] = set()
        for v in list(baselines.values()) + list(our.values()):
            all_cats.update(k for k in v if k != "overall")
        cats = sorted(all_cats)
        columns = cats + ["overall"]

        lines.append(f"## {dataset.upper()} Results")
        lines.append("")
        header = "| System | " + " | ".join(columns) + " |"
        sep = "| --- | " + " | ".join("---" for _ in columns) + " |"
        lines.append(header)
        lines.append(sep)

        for name, scores in baselines.items():
            vals = " | ".join(f"{scores.get(c, 0.0):.3f}" for c in columns)
            lines.append(f"| {name} | {vals} |")

        for name, scores in our.items():
            vals = " | ".join(f"{scores.get(c, 0.0):.3f}" for c in columns)
            lines.append(f"| **{name}** | {vals} |")

        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown table saved to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Selective Amnesia Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", nargs="+", default=["locomo"],
        choices=["locomo", "longmemeval", "all"],
        help="Dataset(s) to evaluate on",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=list(BASELINE_CONFIGS.keys()),
        choices=list(BASELINE_CONFIGS.keys()),
        help="Baselines to run",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-qa", type=int, default=None)
    parser.add_argument("--use-judge", action="store_true",
                        help="Enable GPT-4o judge for LongMemEval")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Load saved graphs instead of re-ingesting")
    parser.add_argument("--force-ingest", action="store_true",
                        help="Re-run ingestion even if saved graphs exist")
    parser.add_argument("--no-edit-delete", action="store_true",
                        help="Skip edit/delete/locality tests")
    parser.add_argument("--output-format", default="json",
                        choices=["json", "latex", "markdown"])
    parser.add_argument("--results-dir", default="results/experiments")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Expand "all"
    datasets = []
    for d in args.dataset:
        if d == "all":
            datasets.extend(["locomo", "longmemeval"])
        else:
            datasets.append(d)
    datasets = list(dict.fromkeys(datasets))  # dedupe preserving order

    suite_start = time.time()
    all_results: list[dict] = []
    total_experiments = len(datasets) * len(args.baselines)
    verbose = not args.quiet

    _header("SELECTIVE AMNESIA EXPERIMENT SUITE", char="*", width=80)
    _kv("Datasets", ", ".join(datasets))
    _kv("Baselines", ", ".join(args.baselines))
    _kv("Total experiments", total_experiments)
    if args.max_samples:
        _kv("Max samples/dataset", args.max_samples)
    if args.max_qa:
        _kv("Max QA/sample", args.max_qa)
    _kv("GPT-4o judge", "enabled" if args.use_judge else "disabled")
    _kv("Graph caching", "skip-ingest" if args.skip_ingest else (
        "force-ingest" if args.force_ingest else "auto"))

    exp_idx = 0
    for dataset in datasets:
        _header(f"DATASET: {dataset.upper()}", char="=")

        if dataset == "locomo":
            samples = load_locomo(max_samples=args.max_samples)
        elif dataset == "longmemeval":
            samples = load_longmemeval(max_samples=args.max_samples)
        else:
            continue

        _kv("Samples loaded", len(samples))

        for baseline in args.baselines:
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
                summary = run_single_experiment(
                    exp_name=exp_name,
                    dataset=dataset,
                    baseline=baseline,
                    samples=samples,
                    config=config,
                    results_dir=args.results_dir,
                    max_qa=args.max_qa,
                    verbose=verbose,
                    run_edit_delete=not args.no_edit_delete,
                    use_judge=args.use_judge,
                    skip_ingest=args.skip_ingest,
                    force_ingest=args.force_ingest,
                )
                summary["elapsed_seconds"] = round(time.time() - start, 1)
                all_results.append(summary)
            except Exception as e:
                logger.error("Experiment %s failed: %s", exp_name, e, exc_info=True)
                all_results.append({"experiment": exp_name, "error": str(e)})

    suite_elapsed = time.time() - suite_start

    # Print comparison with published baselines
    print_published_comparison(all_results, datasets)

    # Print summary table
    _header("EXPERIMENT SUMMARY")
    print(f"    {'Experiment':<35} {'F1':>7} {'R@5':>7} {'MRR':>7} {'TSR':>7} {'Judge':>7}")
    print(f"    {'':=<35} {'':=>7} {'':=>7} {'':=>7} {'':=>7} {'':=>7}")
    for r in all_results:
        if "error" in r:
            print(f"    {r['experiment']:<35} ERROR: {r['error'][:40]}")
            continue
        judge = f"{r['judge_accuracy']:>7.4f}" if "judge_accuracy" in r else "    N/A"
        print(
            f"    {r['experiment']:<35} "
            f"{r['overall_f1']:>7.4f} "
            f"{r.get('recall_at_5', 0):>7.4f} "
            f"{r.get('mrr', 0):>7.4f} "
            f"{r.get('task_success_rate', 0):>7.2%} "
            f"{judge}"
        )
    print(f"\n    Total suite time: {_format_duration(suite_elapsed)}")

    # Export
    out_dir = Path(args.results_dir)
    if args.output_format == "json":
        export_json(all_results, out_dir / "experiment_results.json")
    elif args.output_format == "latex":
        export_json(all_results, out_dir / "experiment_results.json")
        export_latex(all_results, datasets, out_dir / "results_table.tex")
    elif args.output_format == "markdown":
        export_json(all_results, out_dir / "experiment_results.json")
        export_markdown(all_results, datasets, out_dir / "results_table.md")


if __name__ == "__main__":
    main()
