#!/usr/bin/env python3
"""
Estimate end-to-end pipeline runtime from historical run logs.

The estimator fits a simple linear model (intercept + slope * nodes/works)
using the JSON summaries produced by `run_pipeline.sh`. It can then project
expected duration for a requested sample size or total node count.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils import get_default, repo_root


@dataclass(frozen=True)
class PipelineRun:
    path: Path
    sample_size: int
    total_seconds: float
    steps: List[Dict[str, object]]
    node_count: Optional[int] = None

    @property
    def id(self) -> str:
        return self.path.stem.replace("pipeline_run_", "")


def load_pipeline_runs() -> List[PipelineRun]:
    """Load all stored pipeline run logs."""
    runs: List[PipelineRun] = []
    log_root = repo_root() / "logs" / "pipeline_runs"
    if not log_root.exists():
        return runs

    for path in sorted(log_root.glob("pipeline_run_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        sample_size = int(data.get("sample_size") or 0)
        total_seconds = float(data.get("total_duration_seconds") or 0.0)
        steps = data.get("steps") or []
        node_count = extract_node_count(data)
        runs.append(
            PipelineRun(
                path=path,
                sample_size=sample_size,
                total_seconds=total_seconds,
                steps=steps,  # type: ignore[arg-type]
                node_count=node_count,
            )
        )
    return [run for run in runs if run.sample_size > 0 and run.total_seconds > 0]


def extract_node_count(run_payload: Dict[str, object]) -> Optional[int]:
    """Sum created nodes from the archived injection summary, when available."""
    log_files = run_payload.get("log_files") or {}
    if not isinstance(log_files, dict):
        return None
    rel_path = log_files.get("data_injection_summary")
    if not isinstance(rel_path, str):
        return None
    injection_path = repo_root() / rel_path
    if not injection_path.exists():
        return None
    try:
        data = json.loads(injection_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        return None
    total = 0
    for stats in nodes.values():
        if isinstance(stats, dict):
            created = stats.get("created")
            if isinstance(created, (int, float)):
                total += int(created)
    return total or None


def fit_linear_model(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    """
    Return (intercept, slope) for y = intercept + slope * x.
    Falls back to slope-only model when there is just one observation.
    """
    xs_list = list(xs)
    ys_list = list(ys)
    n = len(xs_list)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        x, y = xs_list[0], ys_list[0]
        if x <= 0:
            return 0.0, max(y, 0.0)
        return 0.0, max(y / x, 0.0)

    mean_x = statistics.fmean(xs_list)
    mean_y = statistics.fmean(ys_list)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_list, ys_list))
    denominator = sum((x - mean_x) ** 2 for x in xs_list)
    if denominator <= 0:
        slope = 0.0
    else:
        slope = numerator / denominator
    slope = max(slope, 0.0)
    intercept = mean_y - slope * mean_x
    intercept = max(intercept, 0.0)
    return intercept, slope


def summarise_residuals(intercept: float, slope: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    """Compute RMSE in seconds for the fitted model."""
    xs_list = list(xs)
    ys_list = list(ys)
    if not xs_list:
        return 0.0
    residuals = [y - (intercept + slope * x) for x, y in zip(xs_list, ys_list)]
    mse = sum(res ** 2 for res in residuals) / len(residuals)
    return math.sqrt(max(mse, 0.0))


def format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def estimate_runtime(runs: List[PipelineRun], target_nodes: Optional[int], target_sample: int) -> Dict[str, object]:
    """Compute runtime estimate using node-aware data when possible."""
    usable_for_nodes = [run for run in runs if run.node_count]
    if target_nodes and usable_for_nodes:
        xs = [run.node_count or 0 for run in usable_for_nodes]
        ys = [run.total_seconds for run in usable_for_nodes]
        intercept, slope = fit_linear_model(xs, ys)
        rmse = summarise_residuals(intercept, slope, xs, ys)
        estimate = intercept + slope * target_nodes
        basis = f"{len(usable_for_nodes)} run(s) with node counts"
    else:
        xs = [run.sample_size for run in runs]
        ys = [run.total_seconds for run in runs]
        intercept, slope = fit_linear_model(xs, ys)
        rmse = summarise_residuals(intercept, slope, xs, ys)
        estimate = intercept + slope * target_sample
        basis = f"{len(runs)} run(s) with sample sizes"

    return {
        "estimate_seconds": max(estimate, 0.0),
        "intercept": intercept,
        "slope": slope,
        "rmse": rmse,
        "basis": basis,
    }


def list_runs(runs: List[PipelineRun]) -> None:
    if not runs:
        print("No pipeline runs logged yet.")
        return
    print("Recorded pipeline runs:")
    for run in runs:
        duration = format_duration(run.total_seconds)
        node_info = f", nodes≈{run.node_count}" if run.node_count else ""
        print(f"  - {run.id}: works={run.sample_size}{node_info}, total={duration}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate pipeline runtime from historical logs.")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Target number of works for Step 1 (defaults to config.yaml value).",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        help="Target total nodes to be created in Neo4j (if known).",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Show recorded runs and exit.",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Display average duration per pipeline step.",
    )
    args = parser.parse_args()

    runs = load_pipeline_runs()
    if args.list_runs:
        list_runs(runs)
        return

    if not runs:
        print("No run history available yet. Execute run_pipeline.sh once before estimating.")
        return

    default_sample = int(get_default("extraction", "target_works", 2000))
    target_sample = args.sample_size or default_sample
    target_nodes = args.node_count

    estimate = estimate_runtime(runs, target_nodes, target_sample)
    seconds = estimate["estimate_seconds"]
    rmse = estimate["rmse"]
    print(f"Estimated runtime for sample_size={target_sample} (nodes={'~'+str(target_nodes) if target_nodes else 'derived'}):")
    print(f"  {format_duration(seconds)} (± {format_duration(rmse)}) based on {estimate['basis']}")

    if args.show_steps:
        print("\nAverage step durations from history:")
        step_totals: Dict[str, List[float]] = {}
        for run in runs:
            for step in run.steps:
                title = str(step.get("title", ""))
                duration = float(step.get("duration_seconds") or 0.0)
                step_totals.setdefault(title, []).append(duration)
        for title, durations in step_totals.items():
            avg = statistics.fmean(durations)
            print(f"  - {title}: {format_duration(avg)} average")


if __name__ == "__main__":
    main()
