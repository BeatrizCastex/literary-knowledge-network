#!/usr/bin/env python3
"""
Utility to merge enriched datasets.

Given a "base" enriched directory and an incremental one (containing only new
works), this script concatenates the tables, removes duplicates based on their
natural keys, and writes the combined result to the requested output directory.

Example:
    python3 merge_enriched.py \
        --base-dir data/processed/enriched \
        --incremental-dir data/processed/enriched_incremental \
        --output-dir data/processed/enriched_combined
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

PRIMARY_KEYS: Dict[str, List[str]] = {
    "work_final": ["work_id"],
    "book_final": ["book_id"],
    "series_final": ["series_id"],
    "people_final": ["author_id"],
    "publisher_final": ["publisher_id"],
    "year_final": ["year"],
    "country_final": ["country_code"],
    "languages_final": ["language_code"],
    "tags_final": ["tag_id"],
    "extracted_keywords": ["entity_type", "entity_id", "keyword"],
    "citation_mentions": ["source_type", "source_id", "target_type", "target_id"],
}

KNOWN_TABLES: Iterable[str] = [
    "work_final",
    "book_final",
    "series_final",
    "people_final",
    "publisher_final",
    "year_final",
    "country_final",
    "languages_final",
    "tags_final",
    "extracted_keywords",
    "citation_mentions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge enriched datasets.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory containing the original enriched tables.",
    )
    parser.add_argument(
        "--incremental-dir",
        type=Path,
        required=True,
        help="Directory containing the newly enriched tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the merged tables.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="If set, tables present in only one directory are copied through instead of erroring.",
    )
    return parser.parse_args()


def resolve_table_path(directory: Path, name: str) -> Optional[Path]:
    parquet_path = directory / f"{name}.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = directory / f"{name}.csv"
    if csv_path.exists():
        return csv_path
    return None


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def deduplicate(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    keys = PRIMARY_KEYS.get(table_name)
    if keys:
        missing = [key for key in keys if key not in df.columns]
        if missing:
            raise ValueError(f"Table {table_name} is missing expected key columns: {missing}")
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def merge_tables(base_dir: Path, incremental_dir: Path, output_dir: Path, include_missing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for table in KNOWN_TABLES:
        base_path = resolve_table_path(base_dir, table)
        inc_path = resolve_table_path(incremental_dir, table)

        if base_path is None and inc_path is None:
            if include_missing:
                continue
            raise FileNotFoundError(f"Table {table} not found in either directory.")

        if base_path is None:
            if include_missing:
                df = load_table(inc_path)  # type: ignore[arg-type]
                df.to_parquet(output_dir / f"{table}.parquet", index=False)
                print(f"[info] Copied incremental-only table {table}: {len(df)} rows")
                continue
            raise FileNotFoundError(f"Table {table} missing from base directory.")

        if inc_path is None:
            if include_missing:
                df = load_table(base_path)
                df.to_parquet(output_dir / f"{table}.parquet", index=False)
                print(f"[info] Copied base-only table {table}: {len(df)} rows")
                continue
            raise FileNotFoundError(f"Table {table} missing from incremental directory.")

        base_df = load_table(base_path)
        inc_df = load_table(inc_path)
        combined = pd.concat([base_df, inc_df], ignore_index=True)
        combined = deduplicate(combined, table)
        combined.to_parquet(output_dir / f"{table}.parquet", index=False)
        print(
            f"[info] Merged {table}: base={len(base_df)} incremental={len(inc_df)} -> combined={len(combined)}"
        )


def main() -> None:
    args = parse_args()
    merge_tables(
        base_dir=args.base_dir.resolve(),
        incremental_dir=args.incremental_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        include_missing=args.include_missing,
    )
    print(f"[done] Merged enriched tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
