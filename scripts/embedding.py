#!/usr/bin/env python3
"""
Step 5 - Encoding and Embedding of Metadata

This module builds numerical representations for each work so downstream tasks
such as similarity search can operate on vectors instead of raw text.

Supported strategies:
  • SBERT (default): sentence-transformers semantic embeddings.
  • TF-IDF baseline: classical sparse bag-of-words encoding.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime when absent
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from scipy import sparse as sp  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    sp = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - scikit-learn required
    raise RuntimeError("scikit-learn is required for embeddings") from exc

try:
    from tqdm.auto import tqdm  # type: ignore[import]
except ImportError:  # pragma: no cover - progress is optional
    tqdm = None  # type: ignore[assignment]

from utils import (
    ensure_directory,
    get_default,
    resolve_cli_path,
    resolve_path,
)

DEFAULT_INPUT_DIR = resolve_path("enrichment_dir")
DEFAULT_OUTPUT_DIR = resolve_path("embeddings_dir")
DEFAULT_METHOD = str(get_default("embedding", "method", "sbert"))
DEFAULT_MODEL_NAME = str(get_default("embedding", "model_name", "all-MiniLM-L6-v2"))
DEFAULT_BATCH_SIZE = int(get_default("embedding", "batch_size", 64))

# Work columns to harvest textual signal from.
TEXT_COLUMNS: Sequence[str] = (
    "original_title",
    "best_book_description",
    "keywords",
    "tags",
    "citations",
    "ddc",
    "lcc",
)


def parse_args() -> argparse.Namespace:
    """Define and parse CLI options for the embedding generator."""
    parser = argparse.ArgumentParser(description="Encode works into vector embeddings.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with enriched parquet files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store embeddings.")
    parser.add_argument(
        "--method",
        choices=("sbert", "tfidf", "both"),
        default=DEFAULT_METHOD,
        help="Embedding strategy to run.",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model identifier.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Encoding batch size for SBERT.")
    parser.add_argument("--max-features", type=int, default=25000, help="Maximum vocabulary size for TF-IDF.")
    parser.add_argument("--min-df", type=float, default=2, help="Minimum document frequency for TF-IDF terms.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency for TF-IDF terms.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram length for TF-IDF.")
    parser.add_argument("--progress", action="store_true", help="Show progress bars when supported.")
    parser.add_argument("--verbose", action="store_true", help="Emit debugging details.")
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    """Print a diagnostic message when verbose mode is enabled."""
    if verbose:
        print(message)


def load_work_frame(input_dir: Path, verbose: bool) -> pd.DataFrame:
    """Load the enriched work DataFrame from disk."""
    path = input_dir / "work_final.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected enriched work data at {path}")
    df = pd.read_parquet(path)
    log(f"Loaded work_final: {len(df)} rows", verbose)
    return df


def ensure_list(value: Any) -> List[str]:
    """Coerce values into a list of strings suitable for document assembly."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [clean_token(item) for item in value if clean_token(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [clean_token(item) for item in value if clean_token(item)]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            iterable = value.tolist()
        except Exception:  # pragma: no cover
            iterable = list(value)
        return ensure_list(iterable)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)
    token = clean_token(text)
    return [token] if token else []


def clean_token(value: Any) -> Optional[str]:
    """Return a stripped string or None when the token is empty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def assemble_document(row: pd.Series) -> str:
    """Concatenate relevant work fields into a single text document."""
    parts: List[str] = []
    for column in TEXT_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, str):
            token = clean_token(value)
            if token:
                parts.append(token)
        else:
            parts.extend(ensure_list(value))
    deduped: List[str] = []
    seen: set[str] = set()
    for token in parts:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(token)
    return " ".join(deduped)


def build_corpus(df: pd.DataFrame, verbose: bool) -> Tuple[List[str], List[str]]:
    """Build parallel lists of documents and their corresponding work IDs."""
    documents: List[str] = []
    work_ids: List[str] = []
    for row in df.itertuples(index=False):
        work_id = clean_token(getattr(row, "work_id", None))
        if not work_id:
            continue
        text = assemble_document(row._asdict())
        if not text:
            continue
        work_ids.append(work_id)
        documents.append(text)
    if verbose:
        print(f"Prepared corpus with {len(documents)} documents (from {len(df)} works).")
    return documents, work_ids


def encode_with_sbert(
    documents: List[str],
    model_name: str,
    batch_size: int,
    show_progress: bool,
    verbose: bool,
) -> np.ndarray:
    """Encode documents using a SentenceTransformer model, returning L2-normalized vectors."""
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it or run with --method tfidf."
        )
    log(f"Loading SentenceTransformer model '{model_name}'", verbose)
    model = SentenceTransformer(model_name)
    progress = show_progress and tqdm is not None
    iterator = None
    if progress:
        iterator = tqdm(range(0, len(documents), batch_size), desc="SBERT", unit="batch")
    embeddings: List[np.ndarray] = []
    for start in (iterator if iterator is not None else range(0, len(documents), batch_size)):
        if iterator is None:
            start = start  # type: ignore[assignment]
        end = start + batch_size
        batch = documents[start:end]
        if not batch:
            continue
        emb = model.encode(batch, batch_size=len(batch), show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb.astype(np.float32))
    if not embeddings:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(embeddings)


def encode_with_tfidf(
    documents: List[str],
    max_features: int,
    min_df: float,
    max_df: float,
    ngram_max: int,
    show_progress: bool,
    verbose: bool,
) -> Tuple[Any, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer and transform the corpus, returning matrix and vectorizer."""
    log("Training TF-IDF vectorizer", verbose)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, int(max(1, ngram_max))),
        norm="l2",
    )
    matrix = vectorizer.fit_transform(documents)
    if verbose:
        print(f"TF-IDF matrix shape: {matrix.shape}, nnz={matrix.nnz}")
    return matrix, vectorizer


def save_sbert_embeddings(
    output_dir: Path,
    work_ids: List[str],
    embeddings: np.ndarray,
) -> None:
    """Persist SBERT embeddings alongside work IDs as a Parquet table."""
    df = pd.DataFrame(
        {
            "work_id": work_ids,
            "embedding": embeddings.tolist(),
        }
    )
    path = output_dir / "work_embeddings_sbert.parquet"
    df.to_parquet(path, index=False)


def save_tfidf_embeddings(
    output_dir: Path,
    matrix: Any,
    vectorizer: TfidfVectorizer,
    work_ids: Sequence[str],
) -> None:
    """Persist TF-IDF artifacts (matrix and vectorizer) to disk."""
    matrix_path = output_dir / "work_embeddings_tfidf"
    vocab_path = output_dir / "tfidf_vectorizer.pkl"
    index_path = output_dir / "work_embeddings_tfidf_index.json"
    if sp is not None and sp.issparse(matrix):
        sp.save_npz(str(matrix_path) + ".npz", matrix)
    else:
        dense = matrix.toarray().astype(np.float32)  # type: ignore[attr-defined]
        np.save(str(matrix_path) + ".npy", dense)
    with vocab_path.open("wb") as fp:
        pickle.dump(vectorizer, fp)
    with index_path.open("w", encoding="utf-8") as fp:
        json.dump({"work_ids": list(work_ids)}, fp, indent=2)


def write_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    """Persist a summary JSON describing the embedding run."""
    path = output_dir / "work_embeddings_summary.json"
    ensure_directory(output_dir)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    """CLI entry point: load data, compute embeddings, and save results/metadata."""
    args = parse_args()
    input_dir = resolve_cli_path(args.input_dir)
    output_dir = ensure_directory(resolve_cli_path(args.output_dir))

    df = load_work_frame(input_dir, args.verbose)
    documents, work_ids = build_corpus(df, args.verbose)
    if not documents:
        print("No usable documents found; aborting.")
        sys.exit(1)

    run_sbert = args.method in ("sbert", "both")
    run_tfidf = args.method in ("tfidf", "both")
    summary: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "method_requested": args.method,
        "input_records": int(len(df)),
        "documents_encoded": int(len(work_ids)),
        "dropped_records": int(len(df) - len(work_ids)),
        "text_columns": list(TEXT_COLUMNS),
        "sbert": None,
        "tfidf": None,
    }

    if run_sbert:
        try:
            embeddings = encode_with_sbert(
                documents,
                args.model_name,
                args.batch_size,
                args.progress,
                args.verbose,
            )
        except RuntimeError as exc:
            if args.method == "sbert":
                raise
            print(f"[warn] SBERT skipped: {exc}")
        else:
            save_sbert_embeddings(output_dir, work_ids, embeddings)
            summary["sbert"] = {
                "model_name": args.model_name,
                "batch_size": int(args.batch_size),
                "embedding_dimension": int(embeddings.shape[1]),
                "work_count": int(len(work_ids)),
            }
            if args.verbose:
                print(f"Saved SBERT embeddings to {output_dir}")

    if run_tfidf:
        matrix, vectorizer = encode_with_tfidf(
            documents,
            args.max_features,
            args.min_df,
            args.max_df,
            args.ngram_max,
            args.progress,
            args.verbose,
        )
        save_tfidf_embeddings(output_dir, matrix, vectorizer, work_ids)
        feature_count = int(getattr(matrix, "shape", (0, 0))[1])
        if hasattr(matrix, "nnz"):
            nnz = int(getattr(matrix, "nnz"))
        else:
            nnz = int(feature_count * len(work_ids))
        summary["tfidf"] = {
            "features": feature_count,
            "nonzero_entries": nnz,
            "ngram_max": int(args.ngram_max),
            "min_df": float(args.min_df),
            "max_df": float(args.max_df),
        }
        if args.verbose:
            print(f"Saved TF-IDF artifacts to {output_dir}")

    write_summary(output_dir, summary)

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
