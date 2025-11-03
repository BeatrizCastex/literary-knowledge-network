#!/usr/bin/env python3
"""
Step 1 - Data Extraction

The script:
  * loads a random sample of works with complete publication dates
  * fetches the related books, authors, and series
  * builds pandas DataFrames for works, books, people, publishers, years, countries,
    tags, and languages
  * optionally exports the DataFrames to disk

Outputs (when `--export` is supplied):
  * `data/processed/extraction/work_final.parquet`
  * `data/processed/extraction/book_final.parquet`
  * `data/processed/extraction/series_final.parquet`
  * `data/processed/extraction/people_final.parquet`
  * `data/processed/extraction/publisher_final.parquet`
  * `data/processed/extraction/tags_final.parquet`
  * supporting tables (`year_final`, `country_final`, `languages_final`)
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd

from utils import (
    ensure_directory,
    get_default,
    raw_file_path,
    resolve_cli_path,
    resolve_path,
)

# -------------------------
# Configuration
# -------------------------

WORKS_FILE = raw_file_path("works")
BOOKS_FILE = raw_file_path("books")
AUTHORS_FILE = raw_file_path("authors")
SERIES_FILE = raw_file_path("series")

DEFAULT_TARGET_WORKS = int(get_default("extraction", "target_works", 2000))
TEST_TARGET_WORKS = int(get_default("extraction", "test_target", 200))
DEFAULT_RANDOM_SEED = int(get_default("extraction", "random_seed", 42))
DEFAULT_EXPORT_DIR = resolve_path("extraction_dir")

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "with",
    "by",
    "from",
    "about",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "that",
    "this",
    "it",
    "its",
    "which",
    "who",
    "whom",
}

ENGLISH_LANGUAGE_CODES = {
    "en",
    "eng",
    "en-us",
    "en-gb",
    "english",
}
ENGLISH_LANGUAGE_IDS = {"1"}


# -------------------------
# Helpers
# -------------------------

def jsonl_gz_reader(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a `.json.gz` file one line at a time."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to `int`, returning `None` when the conversion fails."""
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_keywords(text: Optional[str], top_k: int = 10) -> List[str]:
    """Return the top `top_k` keyword candidates from the supplied text."""
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [
        token
        for token in text.split()
        if token and token not in STOPWORDS and len(token) > 2 and not token.isdigit()
    ]
    counter = collections.Counter(tokens)
    return [word for word, _ in counter.most_common(top_k)]


def load_work_ids_from_file(path: Path) -> Set[str]:
    """Load a set of work_ids from a parquet or CSV file."""
    resolved = resolve_cli_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Existing work file not found: {resolved}")
    if resolved.suffix == ".parquet":
        df = pd.read_parquet(resolved)
    else:
        df = pd.read_csv(resolved)
    if "work_id" not in df.columns:
        raise ValueError(f"File {resolved} does not contain a 'work_id' column.")
    return set(df["work_id"].astype(str))


def _normalize_language(value: Any) -> str:
    if value in (None, "", "null"):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value != value:
            return ""
        return str(int(value))
    text = str(value).strip().lower()
    return text


def is_english_language(value: Any) -> bool:
    code = _normalize_language(value)
    if not code:
        return False
    if code in ENGLISH_LANGUAGE_IDS or code in ENGLISH_LANGUAGE_CODES:
        return True
    if code.startswith("en") or code.startswith("eng"):
        return True
    return False


def select_works_with_full_dates(
    limit: int,
    *,
    test_mode: bool = False,
    seed: Optional[int] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Sample works that contain full publication date metadata.

    Returns a tuple of (selected works, work_ids) containing at most `limit`
    randomly sampled items (or the test-mode default when enabled).
    """
    if seed is None:
        seed = DEFAULT_RANDOM_SEED
    if limit <= 0 and not test_mode:
        return [], set()
    target = TEST_TARGET_WORKS if test_mode else limit
    if target <= 0:
        return [], set()

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    selected_ids: List[str] = []
    work_ids: Set[str] = set()
    seen = 0

    for work in jsonl_gz_reader(WORKS_FILE):
        if not all(
            work.get(field)
            for field in (
                "original_title",
                "work_id",
                "original_publication_year",
                "original_publication_month",
                "original_publication_day",
            )
        ):
            continue
        work_id = str(work["work_id"])
        if exclude_ids and work_id in exclude_ids:
            continue
        original_language = work.get("original_language_id")
        if original_language and not is_english_language(original_language):
            continue
        seen += 1

        if len(selected) < target:
            selected.append(work)
            selected_ids.append(work_id)
            work_ids.add(work_id)
        else:
            idx = rng.randrange(seen)
            if idx < target:
                removed_id = selected_ids[idx]
                if removed_id in work_ids:
                    work_ids.remove(removed_id)
                selected[idx] = work
                selected_ids[idx] = work_id
                work_ids.add(work_id)

    return selected, work_ids


def enforce_english_language(
    works: List[Dict[str, Any]],
    books: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
    book_by_id: Dict[str, Dict[str, Any]] = {}
    books_by_work: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for book in books:
        book_id = str(book.get("book_id") or "")
        work_id = str(book.get("work_id") or "")
        if book_id:
            book_by_id[book_id] = book
        if work_id:
            books_by_work[work_id].append(book)

    filtered_works: List[Dict[str, Any]] = []
    filtered_work_ids: Set[str] = set()

    for work in works:
        work_id = str(work.get("work_id") or "")
        if not work_id:
            continue
        original_language = work.get("original_language_id")
        if original_language and is_english_language(original_language):
            filtered_works.append(work)
            filtered_work_ids.add(work_id)
            continue
        if original_language:
            continue

        best_book_id = str(work.get("best_book_id") or "")
        candidate_books: List[Dict[str, Any]] = []
        if best_book_id:
            best_book = book_by_id.get(best_book_id)
            if best_book is not None:
                candidate_books.append(best_book)
        if not candidate_books:
            candidate_books = books_by_work.get(work_id, [])

        english_candidate = any(
            is_english_language(book.get("language_code")) for book in candidate_books
        )
        if english_candidate:
            filtered_works.append(work)
            filtered_work_ids.add(work_id)

    filtered_books = [
        book for book in books if str(book.get("work_id") or "") in filtered_work_ids
    ]
    return filtered_works, filtered_books, filtered_work_ids


def extract_books_for_works(work_ids: Set[str]) -> List[Dict[str, Any]]:
    """Return all book entries whose `work_id` is present in `work_ids`."""
    books: List[Dict[str, Any]] = []
    for book in jsonl_gz_reader(BOOKS_FILE):
        wid = str(book.get("work_id") or "")
        if wid and wid in work_ids:
            books.append(book)
    return books


def collect_author_ids(books: Iterable[Dict[str, Any]]) -> Set[str]:
    """Accumulate unique author IDs present in the supplied book records."""
    author_ids: Set[str] = set()
    for book in books:
        for author in book.get("authors") or []:
            author_id = author.get("author_id")
            if author_id:
                author_ids.add(str(author_id))
    return author_ids


def collect_series_ids(books: Iterable[Dict[str, Any]]) -> Set[str]:
    """Collect unique series identifiers referenced across the books."""
    series_ids: Set[str] = set()
    for book in books:
        for series_id in book.get("series") or []:
            if series_id:
                series_ids.add(str(series_id))
    return series_ids


def load_authors_subset(author_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Load author records whose IDs are contained in `author_ids`."""
    authors: Dict[str, Dict[str, Any]] = {}
    if not author_ids:
        return authors
    if not author_ids:
        return authors
    for author in jsonl_gz_reader(AUTHORS_FILE):
        aid = str(author.get("author_id") or "")
        if aid in author_ids:
            authors[aid] = author
    return authors


def load_series_subset(series_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Load series metadata for the requested IDs."""
    series_map: Dict[str, Dict[str, Any]] = {}
    if not series_ids:
        return series_map
    for series in jsonl_gz_reader(SERIES_FILE):
        sid = str(series.get("series_id") or "")
        if sid in series_ids:
            series_map[sid] = series
    return series_map


# -------------------------
# DataFrame construction
# -------------------------

def build_final_dataframes(
    works: List[Dict[str, Any]],
    books: List[Dict[str, Any]],
    authors_map: Dict[str, Dict[str, Any]],
    series_map: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Assemble the final set of DataFrames (works, books, series, people, etc.)
    derived from the extracted raw JSON records.
    """
    work_rows: List[Dict[str, Any]] = []
    book_rows: List[Dict[str, Any]] = []
    series_rows: List[Dict[str, Any]] = []
    people_rows: List[Dict[str, Any]] = []

    publisher_names: Set[str] = set()
    years: Set[int] = set()
    places: Set[str] = set()
    languages: Set[str] = set()

    tag_types: Dict[str, Set[str]] = collections.defaultdict(set)
    tag_counter: collections.Counter[str] = collections.Counter()

    book_by_id: Dict[str, Dict[str, Any]] = {}
    books_by_work: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    book_tags_map: Dict[str, List[str]] = {}

    for book in books:
        book_id = str(book.get("book_id") or "")
        if not book_id:
            continue

        work_id = str(book.get("work_id") or "")
        book_by_id[book_id] = book
        if work_id:
            books_by_work[work_id].append(book)

        publication_year = safe_int(book.get("publication_year"))
        publication_month = safe_int(book.get("publication_month"))
        publication_day = safe_int(book.get("publication_day"))

        if publication_year:
            years.add(publication_year)
        if book.get("language_code"):
            languages.add(str(book["language_code"]))
        if book.get("country_code"):
            places.add(str(book["country_code"]))
        if book.get("publisher"):
            publisher_names.add(str(book["publisher"]))

        shelves = [
            shelf.get("name")
            for shelf in book.get("popular_shelves") or []
            if isinstance(shelf, dict) and shelf.get("name")
        ]
        description = book.get("description")
        keywords = extract_keywords(description)

        for shelf in shelves:
            tag_types[shelf].add("shelf")
        for keyword in keywords:
            tag_types[keyword].add("keyword")

        tag_counter.update(shelves)
        tag_counter.update(keywords)
        unique_tags = sorted({*shelves, *keywords})
        book_tags_map[book_id] = unique_tags

        book_rows.append(
            {
                "book_id": book_id,
                "title": book.get("title"),
                "title_without_series": book.get("title_without_series"),
                "image_url": book.get("image_url"),
                "goodreads_url": book.get("url") or book.get("link"),
                "work_id": work_id,
                "publication_year": publication_year,
                "publication_month": publication_month,
                "publication_day": publication_day,
                "edition_information": book.get("edition_information"),
                "isbn13": book.get("isbn13"),
                "isbn": book.get("isbn"),
                "num_pages": safe_int(book.get("num_pages")),
                "publisher": book.get("publisher"),
                "authors": book.get("authors"),
                "format": book.get("format"),
                "description": description,
                "similar_books": book.get("similar_books"),
                "kindle_asin": book.get("kindle_asin"),
                "asin": book.get("asin"),
                "popular_shelves": shelves,
                "tags": unique_tags,
                "language_code": book.get("language_code"),
                "country_code": book.get("country_code"),
                "series_ids": book.get("series") or [],
            }
        )

    for work in works:
        work_id = str(work.get("work_id") or "")
        if not work_id:
            continue

        best_book_id = str(work.get("best_book_id") or "")
        best_book = book_by_id.get(best_book_id)
        associated_books = books_by_work.get(work_id, [])
        aggregated_tags: Set[str] = set()

        for related_book in associated_books:
            related_book_id = str(related_book.get("book_id") or "")
            aggregated_tags.update(book_tags_map.get(related_book_id, []))

        original_year = safe_int(work.get("original_publication_year"))
        original_month = safe_int(work.get("original_publication_month"))
        original_day = safe_int(work.get("original_publication_day"))

        if original_year:
            years.add(original_year)
        if work.get("original_language_id"):
            languages.add(str(work["original_language_id"]))

        best_book_country = None
        best_book_description = None
        best_book_series = []
        best_book_author_ids: List[str] = []

        if best_book:
            best_book_description = best_book.get("description")
            best_book_country = best_book.get("country_code")
            best_book_series = best_book.get("series") or []
            if best_book_country:
                places.add(str(best_book_country))
            for author in best_book.get("authors") or []:
                if author.get("author_id"):
                    best_book_author_ids.append(str(author["author_id"]))
        elif associated_books:
            # fall back to the first available book if best_book is missing in the sample
            sample_book = associated_books[0]
            best_book_description = sample_book.get("description")
            best_book_country = sample_book.get("country_code")
            best_book_series = sample_book.get("series") or []
            if best_book_country:
                places.add(str(best_book_country))
            for author in sample_book.get("authors") or []:
                if author.get("author_id"):
                    best_book_author_ids.append(str(author["author_id"]))

        work_rows.append(
            {
                "work_id": work_id,
                "original_title": work.get("original_title"),
                "original_publication_year": original_year,
                "original_publication_month": original_month,
                "original_publication_day": original_day,
                "original_language_id": work.get("original_language_id"),
                "best_book_id": best_book_id or None,
                "best_book_author_ids": best_book_author_ids,
                "best_book_country_code": best_book_country,
                "best_book_series_ids": best_book_series,
                "best_book_description": best_book_description,
                "associated_book_ids": [str(b.get("book_id")) for b in associated_books if b.get("book_id")],
                "tags": sorted(aggregated_tags),
                # ddc and lcc will be obtained from wikidata
                "ddc": None,
                "lcc": None,
            }
        )

    for author_id, author in authors_map.items():
        people_rows.append(
            {
                "author_id": author_id,
                "name": author.get("name"),
                "average_rating": author.get("average_rating"),
                "ratings_count": author.get("ratings_count"),
                "text_reviews_count": author.get("text_reviews_count"),
                "author_wikidata_id": None,
                "date_of_birth": None,
                "date_of_death": None,
                "age": None,
                "year_of_birth": None,
                "year_of_death": None,
                "place_of_birth": None,
                "citizenship": None,
                "place_of_death": None,
                "gender": None,
            }
        )

    for series_id, series in series_map.items():
        series_description = series.get("description")
        series_keywords = extract_keywords(series_description)
        for keyword in series_keywords:
            tag_types[keyword].add("keyword")
        tag_counter.update(series_keywords)
        series_rows.append(
            {
                "series_id": series_id,
                "title": series.get("title"),
                "description": series_description,
                "series_works_count": safe_int(series.get("series_works_count")),
                "primary_work_count": safe_int(series.get("primary_work_count")),
                "numbered": series.get("numbered"),
                "note": series.get("note"),
                "keywords": series_keywords,
            }
        )

    publisher_rows = []
    for idx, publisher_name in enumerate(sorted(publisher_names)):
        publisher_rows.append(
            {
                "publisher_id": f"publisher_{idx+1}",
                "publisher_name": publisher_name,
                "publisher_wikidata_id": None,
                "country": None,
                "year_established": None,
            }
        )

    year_rows = [{"year": year} for year in sorted(years)]
    country_rows = [{"country_code": place} for place in sorted(filter(None, places))]
    language_rows = [{"language_code": language} for language in sorted(filter(None, languages))]

    tag_rows = []
    for idx, (tag_name, types) in enumerate(sorted(tag_types.items())):
        tag_rows.append(
            {
                "tag_id": f"tag_{idx+1}",
                "tag_name": tag_name,
                "tag_type": ";".join(sorted(types)) if types else None,
                "occurrences": tag_counter.get(tag_name, 0),
            }
        )

    return {
        "work_final": pd.DataFrame(work_rows),
        "book_final": pd.DataFrame(book_rows),
        "series_final": pd.DataFrame(series_rows),
        "people_final": pd.DataFrame(people_rows),
        "publisher_final": pd.DataFrame(publisher_rows),
        "year_final": pd.DataFrame(year_rows),
        "country_final": pd.DataFrame(country_rows),
        "languages_final": pd.DataFrame(language_rows),
        "tags_final": pd.DataFrame(tag_rows),
    }


def empty_frame_template() -> Dict[str, pd.DataFrame]:
    """Return an empty set of frames with consistent columns."""
    template = build_final_dataframes([], [], {}, {})
    return {name: df.iloc[:0].copy() for name, df in template.items()}


# -------------------------
# Orchestration
# -------------------------

def export_dataframes(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist every frame to Parquet (falling back to CSV when needed)."""
    ensure_directory(output_dir)
    parquet_supported = True
    for name, df in frames.items():
        path = output_dir / f"{name}.parquet"
        try:
            df.to_parquet(path, index=False)
        except (ImportError, ValueError):
            parquet_supported = False
            fallback_path = output_dir / f"{name}.csv"
            df.to_csv(fallback_path, index=False)
    if not parquet_supported:
        print(
            "Warning: pyarrow/fastparquet not installed. Exported CSV files instead of Parquet."
        )


def run_extraction(
    *,
    target_works: int = DEFAULT_TARGET_WORKS,
    test_mode: bool = False,
    export_dir: Optional[Path] = None,
    random_seed: Optional[int] = None,
    exclude_work_ids: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the extraction pipeline and optionally export the resulting tables.

    Returns a dictionary mapping frame names to DataFrames.
    """
    works, work_ids = select_works_with_full_dates(
        target_works, test_mode=test_mode, seed=random_seed, exclude_ids=exclude_work_ids
    )
    books = extract_books_for_works(work_ids)
    works, books, work_ids = enforce_english_language(works, books)
    if exclude_work_ids:
        works = [work for work in works if work.get("work_id") not in exclude_work_ids]
        books = [book for book in books if str(book.get("work_id") or "") not in exclude_work_ids]
        work_ids = {wid for wid in work_ids if wid not in exclude_work_ids}
    author_ids = collect_author_ids(books)
    series_ids = collect_series_ids(books)

    authors_map = load_authors_subset(author_ids)
    series_map = load_series_subset(series_ids)

    frames = build_final_dataframes(works, books, authors_map, series_map)

    if export_dir is not None:
        export_dataframes(frames, export_dir)

    return frames


def run_incremental_extraction(
    *,
    desired_new_works: int,
    exclude_work_ids: Set[str],
    export_dir: Optional[Path] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Sample works until `desired_new_works` pass filtering, excluding already known IDs.
    Returns data frames containing only the newly sampled works and related entities.
    """
    if desired_new_works <= 0:
        return empty_frame_template()

    aggregate_frames: Dict[str, List[pd.DataFrame]] = collections.defaultdict(list)
    collected_work_ids: Set[str] = set()
    attempts = 0
    seed = random_seed if random_seed is not None else DEFAULT_RANDOM_SEED

    while len(collected_work_ids) < desired_new_works:
        attempts += 1
        remaining = desired_new_works - len(collected_work_ids)
        batch_target = max(remaining * 3, 1000)
        works, candidate_ids = select_works_with_full_dates(
            batch_target,
            seed=seed + attempts,
            exclude_ids=exclude_work_ids | collected_work_ids,
        )
        if not works:
            print("[warn] Unable to locate additional candidate works.")
            break

        books = extract_books_for_works(candidate_ids)
        works, books, filtered_ids = enforce_english_language(works, books)
        filtered_ids = {wid for wid in filtered_ids if wid not in exclude_work_ids and wid not in collected_work_ids}

        if not filtered_ids:
            continue

        works = [work for work in works if work.get("work_id") in filtered_ids]
        books = [book for book in books if str(book.get("work_id") or "") in filtered_ids]

        author_ids = collect_author_ids(books)
        series_ids = collect_series_ids(books)
        authors_map = load_authors_subset(author_ids)
        series_map = load_series_subset(series_ids)

        batch_frames = build_final_dataframes(works, books, authors_map, series_map)
        for name, df in batch_frames.items():
            if df.empty:
                continue
            aggregate_frames[name].append(df)

        collected_work_ids.update(filtered_ids)

    if not collected_work_ids:
        return empty_frame_template()

    combined_frames: Dict[str, pd.DataFrame] = {}
    example_frames = empty_frame_template()
    for name in example_frames.keys():
        if aggregate_frames.get(name):
            combined_frames[name] = pd.concat(aggregate_frames[name], ignore_index=True)
        else:
            combined_frames[name] = example_frames[name].copy()

    if export_dir is not None:
        export_dataframes(combined_frames, export_dir)
    print(f"[info] Incremental sampling accumulated {len(collected_work_ids)} works across {len(aggregate_frames.get('work_final', []))} batches.")

    return combined_frames


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options for the extraction step."""
    parser = argparse.ArgumentParser(description="Goodreads data extraction (Step 1).")
    parser.add_argument(
        "--target-works",
        type=int,
        default=DEFAULT_TARGET_WORKS,
        help="Number of works to sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Alternative flag to define number of works to sample (overrides --target-works).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help=f"Enable test mode (limit {TEST_TARGET_WORKS} works).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the resulting DataFrames to parquet files.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Destination directory for exported files.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducible sampling (default: system entropy).",
    )
    parser.add_argument(
        "--existing-work-file",
        type=Path,
        help="Parquet or CSV containing existing work_ids to exclude from sampling.",
    )
    parser.add_argument(
        "--target-after-filters",
        type=int,
        help="Desired total number of works after all filters (existing + new).",
    )
    parser.add_argument(
        "--incremental-export-dir",
        type=Path,
        help="Destination for incremental sampling exports (defaults to <export-dir>/incremental).",
    )
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        help="Skip the base extraction pass; only perform incremental sampling toward the target.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the extraction script."""
    args = parse_args(argv)
    requested = args.sample_size if args.sample_size is not None else args.target_works
    effective_target = TEST_TARGET_WORKS if args.test_mode else requested

    export_dir = resolve_cli_path(args.export_dir) if args.export else None
    incremental_export_dir = (
        resolve_cli_path(args.incremental_export_dir)
        if args.incremental_export_dir is not None
        else (export_dir / "incremental" if export_dir is not None else None)
    )

    existing_work_ids: Set[str] = set()
    if args.existing_work_file is not None:
        existing_work_ids = load_work_ids_from_file(args.existing_work_file)
        print(f"[info] Loaded {len(existing_work_ids)} existing work ids from {args.existing_work_file}")

    frames: Dict[str, pd.DataFrame] = {}
    if args.incremental_only:
        print("[info] Skipping base extraction (--incremental-only).")
    else:
        frames = run_extraction(
            target_works=effective_target,
            test_mode=args.test_mode,
            export_dir=export_dir,
            random_seed=args.random_seed,
            exclude_work_ids=existing_work_ids,
        )
        if frames:
            work_df = frames.get("work_final")
            if work_df is not None and not work_df.empty:
                new_ids = set(work_df["work_id"].astype(str))
                print(f"[info] Base extraction produced {len(new_ids)} works after filtering.")
                existing_work_ids.update(new_ids)
            print("Base extraction counts:")
            for name, df in frames.items():
                print(f"  {name}: {len(df)} rows")
        if export_dir is not None:
            print(f"[info] Base extraction exported to {export_dir}")

    incremental_frames: Dict[str, pd.DataFrame] = {}
    if args.target_after_filters is not None:
        desired_total = args.target_after_filters
        current_total = len(existing_work_ids)
        remaining = desired_total - current_total
        if remaining > 0:
            if incremental_export_dir is not None:
                ensure_directory(incremental_export_dir)
            incremental_frames = run_incremental_extraction(
                desired_new_works=remaining,
                exclude_work_ids=existing_work_ids,
                export_dir=incremental_export_dir,
                random_seed=args.random_seed,
            )
            work_df = incremental_frames.get("work_final")
            if work_df is not None and not work_df.empty:
                new_ids = set(work_df["work_id"].astype(str))
                print(f"[info] Incremental sampling found {len(new_ids)} additional works (target {desired_total}).")
                existing_work_ids.update(new_ids)
                print("Incremental extraction counts:")
                for name, df in incremental_frames.items():
                    print(f"  {name}: {len(df)} rows")
            else:
                print("[warn] Incremental sampling did not yield any new works.")
            if incremental_export_dir is not None:
                print(f"[info] Incremental data exported to {incremental_export_dir}")
            else:
                print("[warn] Incremental data was not exported (no --incremental-export-dir provided).")
        else:
            print(f"[info] Existing works ({current_total}) already meet or exceed target ({desired_total}).")

    final_total = len(existing_work_ids)
    if args.target_after_filters is not None:
        print(f"[summary] Total unique works accounted for: {final_total} (target {args.target_after_filters})")
    else:
        print(f"[summary] Total unique works accounted for: {final_total}")


if __name__ == "__main__":
    main()
