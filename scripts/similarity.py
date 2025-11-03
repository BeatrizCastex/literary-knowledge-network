#!/usr/bin/env python3
"""
Step 6 - Similarity Computation and Graph Creation

Loads pre-computed work embeddings, builds a similarity graph using cosine
similarity, and optionally writes weighted SIMILAR_TO relationships into Neo4j.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
from sklearn.preprocessing import normalize as sk_normalize  # type: ignore[import]

try:
    from scipy import sparse as sp  # type: ignore[import]
except ImportError:  # pragma: no cover - scipy optional for hybrid similarity
    sp = None  # type: ignore[assignment]

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
NEO4J_PASSWORD = NEO4J_CFG.get("password", "seniorthesis")
NEO4J_DATABASE = NEO4J_CFG.get("database", "neo4j")

DEFAULT_EMBEDDING_FILE = resolve_path("embeddings_dir") / "work_embeddings_sbert.parquet"
DEFAULT_OUTPUT_PARQUET = resolve_path("similarity_dir") / "work_similarity.parquet"
DEFAULT_TOP_K = int(get_default("similarity", "top_k", 10))
DEFAULT_THRESHOLD = float(get_default("similarity", "threshold", 0.75))
DEFAULT_TFIDF_FILE = resolve_path("embeddings_dir") / "work_embeddings_tfidf.npz"
DEFAULT_TFIDF_INDEX = resolve_path("embeddings_dir") / "work_embeddings_tfidf_index.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for similarity computation."""
    parser = argparse.ArgumentParser(description="Compute work similarity graph and update Neo4j.")
    parser.add_argument(
        "--embedding-file",
        type=Path,
        default=DEFAULT_EMBEDDING_FILE,
        help="Parquet file containing SBERT embeddings (work_id, embedding).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Optional parquet export of similarity edges.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retain top-k neighbours per work (set to 0 to disable and rely on threshold only).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity to keep an edge.",
    )
    parser.add_argument(
        "--tfidf-file",
        type=Path,
        default=None,
        help="Optional TF-IDF matrix (npz/npy) to blend with SBERT similarities.",
    )
    parser.add_argument(
        "--tfidf-index",
        type=Path,
        default=None,
        help="JSON file mapping TF-IDF rows to work_ids (defaults beside --tfidf-file).",
    )
    parser.add_argument(
        "--tfidf-weight",
        type=float,
        default=0.0,
        help="Weight for TF-IDF similarity contribution (0.0 disables hybrid).",
    )
    parser.add_argument(
        "--write-neo4j",
        action="store_true",
        help="Write SIMILAR_TO relationships into Neo4j.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing SIMILAR_TO relationships before writing new ones.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--export-edge-csvs",
        action="store_true",
        help="Also export SBERT, TF-IDF, and hybrid edge lists as CSV under --edges-dir.",
    )
    parser.add_argument(
        "--edges-dir",
        type=Path,
        default=resolve_path("output_dir") / "edges",
        help="Directory for CSV exports when --export-edge-csvs is enabled.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class SimilarityEdge:
    work_a: str
    work_b: str
    score: float


def log(message: str, verbose: bool) -> None:
    """Print a message only when verbose mode is requested."""
    if verbose:
        print(message)


def load_embeddings(path: Path, verbose: bool) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from Parquet, returning work IDs and the embedding matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_parquet(path)
    if "work_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Embedding parquet must contain 'work_id' and 'embedding' columns.")
    work_ids = df["work_id"].astype(str).tolist()
    embeddings_raw = df["embedding"].tolist()
    embeddings = np.array([np.asarray(vec, dtype=np.float32) for vec in embeddings_raw], dtype=np.float32)
    log(f"Loaded {len(work_ids)} embeddings with dimension {embeddings.shape[1]}", verbose)
    return work_ids, embeddings


def load_tfidf_features(
    matrix_path: Path,
    index_path: Path,
    expected_work_ids: Sequence[str],
    verbose: bool,
) -> Optional[Any]:
    """Load and align TF-IDF rows with the expected work IDs."""
    if sp is None and matrix_path.suffix == ".npz":
        raise RuntimeError(
            "SciPy is required to load sparse TF-IDF matrices saved as .npz. "
            "Install scipy or re-export embeddings with a dense .npy file."
        )
    if not matrix_path.exists():
        raise FileNotFoundError(f"TF-IDF matrix not found: {matrix_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"TF-IDF index not found: {index_path}")

    if matrix_path.suffix == ".npz":
        matrix = sp.load_npz(str(matrix_path))  # type: ignore[attr-defined]
    else:
        matrix = np.load(str(matrix_path))
    with index_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    tfidf_ids = payload.get("work_ids") or []
    if len(tfidf_ids) != matrix.shape[0]:
        raise ValueError(
            "Mismatch between TF-IDF index length and matrix rows "
            f"({len(tfidf_ids)} vs {matrix.shape[0]})."
        )
    index_map = {str(work_id): idx for idx, work_id in enumerate(tfidf_ids)}
    missing = [work_id for work_id in expected_work_ids if work_id not in index_map]
    if missing:
        raise ValueError(
            f"{len(missing)} work_ids present in SBERT embeddings are missing from TF-IDF matrix."
        )
    ordered_indices = [index_map[work_id] for work_id in expected_work_ids]
    if sp is not None and sp.issparse(matrix):
        matrix = matrix.tocsr()[ordered_indices, :]
    else:
        matrix = np.asarray(matrix)[ordered_indices, :]
    log(
        f"Loaded TF-IDF features with shape {matrix.shape} aligned to SBERT embeddings.",
        verbose,
    )
    return matrix


def normalize_vectors(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def normalize_tfidf_matrix(matrix: Any) -> Any:
    """L2-normalize TF-IDF rows (works with sparse or dense matrices)."""
    if sp is not None and sp.issparse(matrix):
        return sk_normalize(matrix, norm="l2", axis=1, copy=True)
    dense = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def build_similarity_edges(
    work_ids: Sequence[str],
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
    verbose: bool,
    tfidf_matrix: Optional[Any] = None,
    tfidf_weight: float = 0.0,
) -> List[SimilarityEdge]:
    """Compute cosine similarity edges using top-k and threshold pruning."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if len(work_ids) != embeddings.shape[0]:
        raise ValueError("Mismatch between number of work IDs and embeddings.")

    vectors = normalize_vectors(embeddings)
    sims = cosine_similarity(vectors)
    if tfidf_matrix is not None and tfidf_weight > 0:
        normalized_tfidf = normalize_tfidf_matrix(tfidf_matrix)
        if sp is not None and sp.issparse(normalized_tfidf):
            tfidf_sims = normalized_tfidf @ normalized_tfidf.T  # type: ignore[operator]
            if sp.issparse(tfidf_sims):
                tfidf_sims = tfidf_sims.toarray()
        else:
            tfidf_sims = normalized_tfidf @ normalized_tfidf.T
        sims = (1.0 - tfidf_weight) * sims + tfidf_weight * np.asarray(tfidf_sims, dtype=np.float32)
    np.fill_diagonal(sims, -1.0)  # avoid self-selection

    edges: List[SimilarityEdge] = []
    seen: Set[Tuple[int, int]] = set()
    n = len(work_ids)

    for i in range(n):
        row = sims[i]

        candidate_indices: Iterable[int]
        if top_k and top_k > 0:
            top_indices = np.argpartition(row, -top_k)[-top_k:]
            candidate_indices = top_indices
        else:
            candidate_indices = np.where(row >= threshold)[0]

        for j in candidate_indices:
            if j == i:
                continue
            score = float(row[j])
            if score < threshold:
                continue
            a_idx, b_idx = sorted((i, j))
            key = (a_idx, b_idx)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                SimilarityEdge(
                    work_a=work_ids[a_idx],
                    work_b=work_ids[b_idx],
                    score=score,
                )
            )
    log(f"Built {len(edges)} similarity edges", verbose)
    return edges


def edges_dataframe(edges: Sequence[SimilarityEdge]) -> pd.DataFrame:
    """Represent similarity edges as a DataFrame with standard columns."""
    df = pd.DataFrame(
        [
            {
                "source": edge.work_a,
                "target": edge.work_b,
                "score": float(edge.score),
            }
            for edge in edges
        ]
    )
    if not df.empty:
        df["work_a"] = df["source"]
        df["work_b"] = df["target"]
    return df


def export_edges(edges: List[SimilarityEdge], path: Path, verbose: bool) -> None:
    """Persist similarity edges to Parquet for audit/reuse."""
    if not edges:
        log("No edges to export.", verbose)
        return
    df = edges_dataframe(edges)
    ensure_directory(path.parent)
    df.to_parquet(path, index=False)
    log(f"Exported {len(edges)} edges to {path}", verbose)


def export_edge_csv(edges: Sequence[SimilarityEdge], path: Path, verbose: bool) -> None:
    """Persist similarity edges to CSV for downstream notebooks."""
    if not edges:
        log(f"No edges to export for {path.name}.", verbose)
        return
    df = edges_dataframe(edges)
    ensure_directory(path.parent)
    df.to_csv(path, index=False)
    log(f"Exported {len(edges)} edges to {path}", verbose)


def compute_graph_metrics(
    work_ids: Sequence[str],
    edges: Sequence[SimilarityEdge],
) -> Dict[str, Any]:
    """Compute structural metrics for the similarity graph."""
    adjacency: Dict[str, Set[str]] = {work_id: set() for work_id in work_ids}
    scores: List[float] = []
    for edge in edges:
        a = edge.work_a
        b = edge.work_b
        score = float(edge.score)
        scores.append(score)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    total_nodes = len(adjacency)
    total_edges = len(edges)
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    works_with_edges = sum(1 for degree in degrees if degree > 0)
    isolated = total_nodes - works_with_edges
    average_degree = (2 * total_edges / total_nodes) if total_nodes else 0.0
    density = (2 * total_edges) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0

    score_array = np.array(scores, dtype=np.float32) if scores else np.array([], dtype=np.float32)
    score_stats = None
    if score_array.size:
        score_stats = {
            "min": float(score_array.min()),
            "p25": float(np.percentile(score_array, 25)),
            "median": float(np.percentile(score_array, 50)),
            "p75": float(np.percentile(score_array, 75)),
            "max": float(score_array.max()),
            "mean": float(score_array.mean()),
        }

    degree_array = np.array(degrees, dtype=np.int32) if degrees else np.array([], dtype=np.int32)
    degree_stats = None
    if degree_array.size:
        degree_stats = {
            "min": int(degree_array.min()),
            "p25": float(np.percentile(degree_array, 25)),
            "median": float(np.percentile(degree_array, 50)),
            "p75": float(np.percentile(degree_array, 75)),
            "max": int(degree_array.max()),
            "mean": float(degree_array.mean()),
        }

    components: List[int] = []
    visited: Set[str] = set()
    for node, neighbors in adjacency.items():
        if node in visited:
            continue
        if not neighbors:
            components.append(1)
            visited.add(node)
            continue
        queue: deque[str] = deque([node])
        visited.add(node)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(size)

    component_array = np.array(components, dtype=np.int32) if components else np.array([], dtype=np.int32)
    largest_component = int(component_array.max()) if component_array.size else 0
    component_stats = {
        "count": int(len(components)),
        "largest_size": largest_component,
        "giant_component_ratio": float(largest_component / total_nodes) if total_nodes else 0.0,
    }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "works_with_edges": int(works_with_edges),
        "isolated_works": int(isolated),
        "average_degree": float(average_degree),
        "density": float(density),
        "score_stats": score_stats,
        "degree_stats": degree_stats,
        "component_stats": component_stats,
    }


def write_similarity_summary(path: Path, summary: Dict[str, Any]) -> None:
    """Write similarity graph diagnostics to JSON."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def write_edges_to_neo4j(
    edges: List[SimilarityEdge],
    clear_existing: bool,
    verbose: bool,
) -> None:
    """Merge similarity relationships into Neo4j, optionally clearing existing ones."""
    if not edges:
        log("No similarity edges to write to Neo4j.", verbose)
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            if clear_existing:
                log("Clearing existing SIMILAR_TO relationships.", verbose)
                session.run("MATCH (:Work)-[r:SIMILAR_TO]-(:Work) DELETE r")

            query = """
            UNWIND $rows AS row
            MATCH (a:Work {id: row.work_a})
            MATCH (b:Work {id: row.work_b})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = row.score
            """
            batch_size = 500
            total_created = 0
            for start in range(0, len(edges), batch_size):
                batch = edges[start : start + batch_size]
                session.run(
                    query,
                    rows=[edge.__dict__ for edge in batch],
                )
                total_created += len(batch)
            log(f"Wrote {total_created} SIMILAR_TO relationships to Neo4j.", verbose)
    finally:
        driver.close()


def main() -> None:
    """CLI entry point for generating similarity edges and updating Neo4j."""
    args = parse_args()
    embedding_file = resolve_cli_path(args.embedding_file)
    work_ids, embeddings = load_embeddings(embedding_file, args.verbose)
    tfidf_weight = float(max(0.0, min(args.tfidf_weight, 1.0)))
    tfidf_matrix: Optional[Any] = None

    def resolve_optional(path: Optional[Path]) -> Optional[Path]:
        return resolve_cli_path(path) if path is not None else None

    tfidf_file_path = resolve_optional(args.tfidf_file)
    tfidf_index_path = resolve_optional(args.tfidf_index)

    if tfidf_file_path is None and tfidf_weight > 0:
        candidates = [
            DEFAULT_TFIDF_FILE,
            DEFAULT_TFIDF_FILE.with_suffix(".npz"),
            DEFAULT_TFIDF_FILE.with_suffix(".npy"),
        ]
        for candidate in candidates:
            candidate_resolved = resolve_cli_path(candidate)
            if candidate_resolved.exists():
                tfidf_file_path = candidate_resolved
                break
    if tfidf_file_path is not None and not tfidf_file_path.exists():
        # try alternate extensions when the user provided a base path
        for extension in (".npz", ".npy"):
            candidate = tfidf_file_path.with_suffix(extension)
            if candidate.exists():
                tfidf_file_path = candidate
                break
    if tfidf_file_path is not None and tfidf_file_path.exists():
        if tfidf_index_path is None:
            tfidf_index_path = tfidf_file_path.with_name("work_embeddings_tfidf_index.json")
        if tfidf_weight <= 0.0:
            tfidf_weight = 0.3
        tfidf_matrix = load_tfidf_features(tfidf_file_path, tfidf_index_path, work_ids, args.verbose)
    elif args.tfidf_file is not None or args.tfidf_weight > 0:
        print("[warn] TF-IDF artifacts not found; proceeding with SBERT-only similarity.")
        tfidf_weight = 0.0

    edges = build_similarity_edges(
        work_ids,
        embeddings,
        args.top_k,
        args.threshold,
        args.verbose,
        tfidf_matrix=tfidf_matrix,
        tfidf_weight=tfidf_weight,
    )
    csv_exports: Dict[str, Dict[str, Any]] = {}
    if args.export_edge_csvs:
        edges_dir = resolve_cli_path(args.edges_dir)
        ensure_directory(edges_dir)

        log("Exporting hybrid edge list to CSV.", args.verbose)
        hybrid_path = edges_dir / "hybrid_edges.csv"
        export_edge_csv(edges, hybrid_path, args.verbose)
        csv_exports["hybrid"] = {"path": str(hybrid_path), "edges": len(edges)}

        log("Computing SBERT-only edge list for CSV export.", args.verbose)
        sbert_edges = build_similarity_edges(
            work_ids,
            embeddings,
            args.top_k,
            args.threshold,
            args.verbose,
            tfidf_matrix=None,
            tfidf_weight=0.0,
        )
        sbert_path = edges_dir / "sbert_edges.csv"
        export_edge_csv(sbert_edges, sbert_path, args.verbose)
        csv_exports["sbert"] = {"path": str(sbert_path), "edges": len(sbert_edges)}

        if tfidf_matrix is not None:
            log("Computing TF-IDF-only edge list for CSV export.", args.verbose)
            tfidf_edges = build_similarity_edges(
                work_ids,
                embeddings,
                args.top_k,
                args.threshold,
                args.verbose,
                tfidf_matrix=tfidf_matrix,
                tfidf_weight=1.0,
            )
            tfidf_path = edges_dir / "tfidf_edges.csv"
            export_edge_csv(tfidf_edges, tfidf_path, args.verbose)
            csv_exports["tfidf"] = {"path": str(tfidf_path), "edges": len(tfidf_edges)}
        else:
            log("TF-IDF artifacts unavailable; skipping TF-IDF edge export.", args.verbose)

    summary = compute_graph_metrics(work_ids, edges)
    summary["parameters"] = {
        "top_k": int(args.top_k),
        "threshold": float(args.threshold),
        "embedding_file": str(embedding_file),
        "tfidf_weight": float(tfidf_weight),
        "tfidf_file": str(tfidf_file_path) if tfidf_matrix is not None else None,
    }
    if csv_exports:
        summary.setdefault("exports", {})
        summary["exports"]["edge_csv"] = csv_exports

    if args.output_file:
        output_file = resolve_cli_path(args.output_file)
        export_edges(edges, output_file, args.verbose)
        summary_path = output_file.parent / "work_similarity_summary.json"
    else:
        summary_path = resolve_path("similarity_dir") / "work_similarity_summary.json"

    write_similarity_summary(summary_path, summary)

    if args.write_neo4j:
        write_edges_to_neo4j(edges, args.clear_existing, args.verbose)

    print(f"Similarity computation complete. Generated {len(edges)} edges.")


if __name__ == "__main__":
    main()
