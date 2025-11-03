#!/usr/bin/env python3
"""
Step 7 - Community Detection (Leiden)

Detect clusters of related works on the similarity graph and persist the
results to disk and/or Neo4j.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime
import json

import numpy as np
import pandas as pd
from igraph import Graph  # type: ignore[import]
from leidenalg import find_partition, RBConfigurationVertexPartition  # type: ignore[import]
from neo4j import GraphDatabase

from utils import (
    ensure_directory,
    get_default,
    neo4j_credentials,
    resolve_cli_path,
    resolve_path,
)

NEO4J_CFG = neo4j_credentials()
NEO4J_URI = NEO4J_CFG.get("uri", "neo4j://127.0.0.1:7687")
NEO4J_USER = NEO4J_CFG.get("user", "neo4j")
NEO4J_PASSWORD = NEO4J_CFG.get("password")
NEO4J_DATABASE = NEO4J_CFG.get("database", "neo4j")

DEFAULT_SIMILARITY_FILE = resolve_path("similarity_dir") / "work_similarity.parquet"
DEFAULT_OUTPUT_FILE = resolve_path("communities_dir") / "work_communities.parquet"

BASE_RESOLUTIONS: Sequence[float] = (0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
DEFAULT_RESOLUTION = float(get_default("communities", "resolution", 1.0))
DEFAULT_RESOLUTIONS: Sequence[float] = tuple(
    dict.fromkeys(list(BASE_RESOLUTIONS) + [DEFAULT_RESOLUTION])
)

def parse_args() -> argparse.Namespace:
    """Parse command-line options for community detection."""
    parser = argparse.ArgumentParser(description="Detect communities on the work similarity graph.")
    parser.add_argument(
        "--similarity-file",
        type=Path,
        default=DEFAULT_SIMILARITY_FILE,
        help="Parquet file containing similarity edges (work_a, work_b, score).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Parquet destination for cluster assignments.",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="Resolution parameters to evaluate for Leiden.",
    )
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=3,
        help="Ignore partitions producing communities smaller than this.",
    )
    parser.add_argument(
        "--min-export-size",
        type=int,
        default=1,
        help="Drop communities smaller than this when exporting assignments or writing to Neo4j.",
    )
    parser.add_argument(
        "--write-neo4j",
        action="store_true",
        help="Persist cluster assignments as properties on Work nodes.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Remove existing cluster properties before writing new ones.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    """Print a message when verbose mode is enabled."""
    if verbose:
        print(message)


def load_similarity_edges(path: Path, verbose: bool) -> pd.DataFrame:
    """Load similarity edges (work_a, work_b, score) from Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Similarity file not found: {path}")
    df = pd.read_parquet(path)
    required = {"work_a", "work_b", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Similarity parquet must contain columns: {', '.join(sorted(required))}")
    log(f"Loaded {len(df)} similarity edges from {path}", verbose)
    return df


def build_igraph_from_edges(edges_df: pd.DataFrame, verbose: bool) -> Tuple[Graph, Dict[str, int]]:
    """Construct an igraph graph and vertex index map from the similarity edge list."""
    work_ids = pd.unique(edges_df[["work_a", "work_b"]].values.ravel())
    work_ids = work_ids[~pd.isna(work_ids)]
    work_ids = work_ids.astype(str)
    idx_map = {work_id: idx for idx, work_id in enumerate(work_ids)}

    g = Graph()
    g.add_vertices(len(work_ids))
    g.vs["name"] = list(work_ids)

    edge_list: List[Tuple[int, int]] = []
    weights: List[float] = []
    seen: set[Tuple[int, int]] = set()

    for row in edges_df.itertuples(index=False):
        source = str(getattr(row, "work_a"))
        target = str(getattr(row, "work_b"))
        score = float(getattr(row, "score"))
        if source == target or math.isnan(score):
            continue
        a_idx = idx_map.get(source)
        b_idx = idx_map.get(target)
        if a_idx is None or b_idx is None:
            continue
        edge = (min(a_idx, b_idx), max(a_idx, b_idx))
        if edge in seen:
            # keep the max weight if duplicate edges exist
            existing_index = edge_list.index(edge)
            weights[existing_index] = max(weights[existing_index], score)
            continue
        seen.add(edge)
        edge_list.append(edge)
        weights.append(score)

    if not edge_list:
        raise ValueError("No similarity edges available to build the graph.")

    g.add_edges(edge_list)
    g.es["weight"] = weights

    log(f"Constructed graph with {g.vcount()} vertices and {g.ecount()} edges.", verbose)
    return g, idx_map


@dataclass
class PartitionResult:
    resolution: float
    modularity: float
    communities: int
    membership: List[int]


def evaluate_resolutions(
    graph: Graph,
    resolutions: Sequence[float],
    min_community_size: int,
    verbose: bool,
) -> List[PartitionResult]:
    """Evaluate Leiden partitions across resolution parameters, returning quality metrics."""
    results: List[PartitionResult] = []
    for resolution in resolutions:
        partition = find_partition(
            graph,
            RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
        )
        membership = partition.membership
        communities = len(partition)
        modularity = getattr(partition, "modularity", None)
        if modularity is None:
            modularity = partition.quality()

        # Filter out trivial partitions
        if communities < 2:
            log(f"Resolution {resolution:.2f}: skipped (only one community).", verbose)
            continue

        community_sizes = np.bincount(np.array(membership))
        if (community_sizes < min_community_size).all():
            log(f"Resolution {resolution:.2f}: skipped (all communities < {min_community_size}).", verbose)
            continue

        results.append(
            PartitionResult(
                resolution=resolution,
                modularity=modularity,
                communities=communities,
                membership=membership,
            )
        )
        log(
            f"Resolution {resolution:.2f}: modularity={modularity:.4f}, communities={communities}",
            verbose,
        )
    return results


def community_size_stats(membership: Sequence[int]) -> Dict[str, Any]:
    """Return descriptive statistics for cluster membership sizes."""
    sizes = np.bincount(np.array(membership, dtype=np.int32))
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return {}
    percentiles = np.percentile(sizes, [25, 50, 75, 90])
    return {
        "min": int(sizes.min()),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
    }


def write_community_summary(path: Path, summary: Dict[str, Any]) -> None:
    """Persist community-detection diagnostics to JSON."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def select_best_partition(results: List[PartitionResult]) -> PartitionResult:
    """Select the partition with the highest modularity (tie-breaking on size)."""
    if not results:
        raise ValueError("No valid partition results available.")

    # Sort by modularity descending; break ties preferring moderate community counts.
    best = max(
        results,
        key=lambda res: (res.modularity, -abs(res.communities - len(res.membership) / 20)),
    )
    return best


def build_assignment_df(
    graph: Graph,
    membership: Sequence[int],
    resolution: float,
) -> pd.DataFrame:
    """Create a DataFrame mapping work IDs to cluster metadata."""
    cluster_labels = np.array(membership, dtype=int)
    df = pd.DataFrame(
        {
            "work_id": graph.vs["name"],
            "cluster_id": cluster_labels,
            "cluster_label": [f"cluster_{c}" for c in cluster_labels],
            "cluster_resolution": resolution,
        }
    )
    community_sizes = df.groupby("cluster_id")["work_id"].count().rename("cluster_size")
    df = df.merge(community_sizes, on="cluster_id", how="left")
    return df


def write_assignments_to_neo4j(
    assignments: pd.DataFrame,
    clear_existing: bool,
    verbose: bool,
) -> None:
    """Write cluster assignments to Neo4j as node properties."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            if clear_existing:
                log("Clearing existing cluster properties on Work nodes.", verbose)
                session.run(
                    """
                    MATCH (w:Work)
                    REMOVE w.cluster_id, w.cluster_label, w.cluster_resolution, w.cluster_size
                    """
                )

            query = """
            UNWIND $rows AS row
            MATCH (w:Work {id: row.work_id})
            SET w.cluster_id = row.cluster_id,
                w.cluster_label = row.cluster_label,
                w.cluster_resolution = row.cluster_resolution,
                w.cluster_size = row.cluster_size
            """

            batch_size = 500
            rows = assignments.to_dict(orient="records")
            total = 0
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                session.run(query, rows=batch)
                total += len(batch)
            log(f"Updated {total} Work nodes with cluster assignments.", verbose)
    finally:
        driver.close()


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON content when available; otherwise return None."""
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def write_results_metrics_bundle(
    embedding_summary_path: Path,
    similarity_summary_path: Path,
    community_summary_path: Path,
    verbose: bool,
) -> None:
    """Aggregate step summaries into a single metrics JSON for notebooks."""
    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "embedding_summary": _load_json_if_exists(embedding_summary_path),
        "similarity_summary": _load_json_if_exists(similarity_summary_path),
        "community_summary": _load_json_if_exists(community_summary_path),
    }

    results: List[Dict[str, Any]] = []
    similarity_summary = payload["similarity_summary"] or {}
    community_summary = payload["community_summary"] or {}

    if similarity_summary and community_summary:
        params = similarity_summary.get("parameters", {}) or {}
        total_nodes = similarity_summary.get("total_nodes") or 0
        works_with_edges = similarity_summary.get("works_with_edges") or 0
        graph_coverage = (
            float(works_with_edges) / float(total_nodes) if total_nodes else None
        )
        modularity = (
            community_summary.get("selected", {}) or {}
        ).get("modularity")
        if modularity is None:
            evaluations = community_summary.get("evaluations") or []
            if evaluations:
                modularity = evaluations[0].get("modularity")

        results.append(
            {
                "threshold": params.get("threshold"),
                "top_k": params.get("top_k"),
                "graph_coverage": graph_coverage,
                "modularity": modularity,
                "median_keyword_purity": None,
            }
        )

    payload["results"] = results

    output_root = resolve_path("output_dir")
    destinations = [
        output_root / "results_metrics.json",
        output_root / "analysis" / "results_metrics.json",
    ]
    for target in destinations:
        ensure_directory(target.parent)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    if destinations:
        log(
            "Aggregated metrics written to: " + ", ".join(str(p) for p in destinations),
            verbose,
        )


def main() -> None:
    """Execute community detection using CLI parameters and persist the results."""
    args = parse_args()
    similarity_file = resolve_cli_path(args.similarity_file)
    similarity_df = load_similarity_edges(similarity_file, args.verbose)
    graph, index_map = build_igraph_from_edges(similarity_df, args.verbose)

    candidates = evaluate_resolutions(
        graph,
        sorted(set(args.resolutions)),
        args.min_community_size,
        args.verbose,
    )
    summary: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "graph": {
            "nodes": int(graph.vcount()),
            "edges": int(graph.ecount()),
        },
        "parameters": {
            "resolutions_requested": [float(res) for res in sorted(set(args.resolutions))],
            "min_community_size": int(args.min_community_size),
            "min_export_size": int(args.min_export_size),
        },
        "evaluations": [
            {
                "resolution": float(result.resolution),
                "modularity": float(result.modularity),
                "communities": int(result.communities),
                "size_stats": community_size_stats(result.membership),
            }
            for result in candidates
        ],
    }
    best_partition = select_best_partition(candidates)
    if args.verbose:
        print(
            f"Selected resolution {best_partition.resolution:.2f} "
            f"(modularity={best_partition.modularity:.4f}, communities={best_partition.communities})"
        )

    assignments = build_assignment_df(graph, best_partition.membership, best_partition.resolution)
    output_file = resolve_cli_path(args.output_file)
    ensure_directory(output_file.parent)
    export_assignments = assignments[assignments["cluster_size"] >= args.min_export_size].copy()
    removed_rows = int(len(assignments) - len(export_assignments))
    removed_clusters = int(
        assignments.loc[assignments["cluster_size"] < args.min_export_size, "cluster_id"].nunique()
    )
    export_assignments.to_parquet(output_file, index=False)
    csv_destination = output_file.with_suffix(".csv")
    export_assignments.to_csv(csv_destination, index=False)
    log(f"Cluster assignments saved to {output_file} and {csv_destination}", args.verbose)
    if removed_rows:
        log(
            f"Dropped {removed_rows} rows across {removed_clusters} clusters below size {args.min_export_size}.",
            args.verbose,
        )
    summary["selected"] = {
        "resolution": float(best_partition.resolution),
        "modularity": float(best_partition.modularity),
        "communities": int(best_partition.communities),
        "size_stats": community_size_stats(best_partition.membership),
    }
    summary["assignments"] = {
        "rows": int(len(export_assignments)),
        "unique_clusters": int(export_assignments["cluster_id"].nunique()),
        "removed_rows": removed_rows,
        "removed_clusters": removed_clusters,
    }
    summary_path = output_file.parent / "work_communities_summary.json"
    write_community_summary(summary_path, summary)
    write_results_metrics_bundle(
        resolve_path("embeddings_dir") / "work_embeddings_summary.json",
        resolve_path("similarity_dir") / "work_similarity_summary.json",
        summary_path,
        args.verbose,
    )

    if args.write_neo4j:
        write_assignments_to_neo4j(export_assignments, args.clear_existing, args.verbose)

    print(
        f"Community detection complete. {export_assignments['cluster_id'].nunique()} clusters "
        f"across {len(export_assignments)} works."
    )


if __name__ == "__main__":
    main()
