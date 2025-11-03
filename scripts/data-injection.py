#!/usr/bin/env python3
"""
Step 4 - Data Injection

Loads the enriched Goodreads tables (generated in Step 3) and streams them into
Neo4j using MERGE so the script is idempotent. Constraints are created
automatically before any data is written.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
from neo4j import GraphDatabase

from utils import (
    ensure_directory,
    neo4j_credentials,
    resolve_cli_path,
    resolve_path,
)

NEO4J_CFG = neo4j_credentials()
NEO4J_URI = NEO4J_CFG.get("uri", "neo4j://127.0.0.1:7687")
NEO4J_USER = NEO4J_CFG.get("user", "neo4j")
NEO4J_PASSWORD = NEO4J_CFG.get("password")
NEO4J_DATABASE = NEO4J_CFG.get("database", "neo4j")
DEFAULT_INPUT_DIR = resolve_path("enrichment_dir")
DEFAULT_BATCH_SIZE = 500
DEFAULT_LOG_FILE = resolve_path("logs_dir") / "data_injection_summary.json"

LIST_COLUMNS: Dict[str, Sequence[str]] = {
    "work_final": [
        "best_book_author_ids",
        "best_book_series_ids",
        "associated_book_ids",
        "tags",
        "keywords",
        "citations",
    ],
    "book_final": [
        "authors",
        "tags",
        "popular_shelves",
        "series_ids",
        "similar_books",
        "citations",
        "keywords",
    ],
    "series_final": ["keywords"],
    "people_final": [
        "citizenship",
        "place_of_birth",
        "place_of_death",
        "wikidata_place_of_birth",
        "wikidata_place_of_death",
    ],
    "publisher_final": ["wikidata_country"],
    "citation_mentions": [],
    "tags_final": [],
}

LIST_TOKEN_PATTERN = re.compile(r"'([^']+)'|\"([^\"]+)\"")
EMPTY_LITERAL_MARKERS = {"", "[]", "{}", "()", "nan", "none", "null"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the data injection step."""
    parser = argparse.ArgumentParser(description="Stream enriched Goodreads data into Neo4j.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--verbose", action="store_true", help="Print debugging information")
    parser.add_argument(
        "--min-tag-occurrences",
        type=int,
        default=3,
        help="Only create Tag nodes for tags that appear at least this many times (default: 3).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Path to write a JSON summary of node/relationship counts.",
    )
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    """Print a message when verbose mode is enabled."""
    if verbose:
        print(message)


def load_table(input_dir: Path, name: str, verbose: bool) -> pd.DataFrame:
    """Load a single table from Parquet/CSV and coerce list columns when required."""
    parquet_path = input_dir / f"{name}.parquet"
    csv_path = input_dir / f"{name}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    list_columns = LIST_COLUMNS.get(name, [])
    if list_columns:
        df = ensure_list_columns(df, list_columns)

    log(f"Loaded {name}: {len(df)} rows", verbose)
    return df


def ensure_list_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure each column listed is composed of Python lists (not strings/arrays)."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(coerce_list)
    return df


def coerce_list(value: Any) -> List[Any]:
    """Convert arbitrary list-like data (arrays, strings) into a standard Python list."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, list):
        result_list: List[Any] = []
        for element in value:
            coerced = coerce_list(element)
            if isinstance(coerced, list):
                result_list.extend(coerced)
            else:
                result_list.append(coerced)
        return result_list
    if isinstance(value, tuple):
        return coerce_list(list(value))
    if isinstance(value, set):
        return coerce_list(list(value))
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        return coerce_list(converted)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        extracted = extract_quoted_tokens(stripped)
        if extracted is not None:
            return extracted
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                return coerce_list(parsed)
            except (SyntaxError, ValueError):
                pass
        if ";" in stripped and not stripped.startswith("http"):
            return [segment.strip() for segment in stripped.split(";") if segment.strip()]
        return [stripped]
    return [value]


def flatten_strings(values: Any) -> List[str]:
    """Recursively flatten nested structures into a list of unique strings."""
    result: List[str] = []
    seen: Set[str] = set()

    def _rec(item: Any) -> None:
        if item is None or (isinstance(item, float) and math.isnan(item)):
            return
        if isinstance(item, (list, tuple, set)):
            for sub in item:
                _rec(sub)
            return
        if hasattr(item, "tolist") and not isinstance(item, (str, bytes)):
            _rec(item.tolist())
            return
        text = str(item).strip()
        if text.startswith("[") and text.endswith("]"):
            extracted = extract_quoted_tokens(text)
            if extracted:
                _rec(extracted)
                return
            try:
                parsed = ast.literal_eval(text)
                _rec(parsed)
                return
            except (ValueError, SyntaxError):
                pass
        extracted = extract_quoted_tokens(text)
        if extracted:
            _rec(extracted)
            return
        if text:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                result.append(text)

    _rec(values)
    return result


def extract_quoted_tokens(text: str) -> Optional[List[str]]:
    """Extract token strings contained within quotes (single/double)."""
    tokens: List[str] = []
    for match in LIST_TOKEN_PATTERN.finditer(text):
        token = match.group(1) if match.group(1) is not None else match.group(2)
        if token:
            cleaned = token.strip()
            if cleaned and cleaned.lower() not in EMPTY_LITERAL_MARKERS:
                tokens.append(cleaned)
    if tokens:
        return tokens
    return None


def sanitize_value(value: Any) -> Optional[Any]:
    """Normalize values for Neo4j parameter passing (handling NaN, arrays, etc.)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            extracted = value.item()
            if extracted is not value:
                return sanitize_value(extracted)
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        sanitized_list: List[Any] = []
        for element in value:
            cleaned = sanitize_value(element)
            if cleaned is not None:
                sanitized_list.append(cleaned)
        return sanitized_list or None
    if isinstance(value, dict):
        sanitized_dict: Dict[str, Any] = {}
        for key, element in value.items():
            cleaned = sanitize_value(element)
            if cleaned is not None:
                sanitized_dict[key] = cleaned
        return sanitized_dict or None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in EMPTY_LITERAL_MARKERS:
            return None
        return text
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[arg-type]
        except Exception:
            pass
    return value


def sanitize_properties(record: Dict[str, Any], key_field: str) -> Dict[str, Any]:
    """Convert record values to Neo4j-safe types excluding the key field."""
    sanitized: Dict[str, Any] = {}
    for field, raw in record.items():
        if field == key_field:
            continue
        cleaned = sanitize_value(raw)
        if cleaned is not None:
            sanitized[field] = cleaned
    return sanitized


def to_year(value: Any) -> Optional[int]:
    """Normalize a value to an integer year when possible."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def normalize_country(value: Any) -> Optional[str]:
    """Return a cleaned country string or None when the value is empty."""
    text = clean_string(value)
    if not text:
        return None
    return text


def clean_string(value: Any) -> Optional[str]:
    """Trim whitespace, convert NaN to None, and leave meaningful text intact."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in EMPTY_LITERAL_MARKERS:
        return None
    return text


def create_constraints(driver) -> None:
    """Ensure uniqueness constraints exist for all primary node labels."""
    statements = [
        "CREATE CONSTRAINT work_id IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
        "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT series_id IF NOT EXISTS FOR (s:Series) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT publisher_id IF NOT EXISTS FOR (pub:Publisher) REQUIRE pub.id IS UNIQUE",
        "CREATE CONSTRAINT country_code IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE",
        "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
        "CREATE CONSTRAINT language_code IF NOT EXISTS FOR (l:Language) REQUIRE l.code IS UNIQUE",
        "CREATE CONSTRAINT tag_id IF NOT EXISTS FOR (t:Tag) REQUIRE t.id IS UNIQUE",
    ]
    with driver.session(database=NEO4J_DATABASE) as session:
        for statement in statements:
            session.run(statement)


def write_nodes(
    driver,
    label: str,
    key: str,
    rows: Iterable[Dict[str, Any]],
    batch_size: int,
    verbose: bool,
) -> Dict[str, int]:
    """MERGE node batches into Neo4j and return creation statistics."""
    filtered_rows = []
    for row in rows:
        key_value = sanitize_value(row.get(key))
        if key_value is None:
            continue
        filtered_rows.append((key_value, row))
    if not filtered_rows:
        log(f"Skipping {label}: no rows to write", verbose)
        return {"attempted": 0, "created": 0}
    log(f"Writing {len(filtered_rows)} {label} nodes", verbose)
    query = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{{key}: row.key}})
    SET n += row.props
    """
    total_created = 0
    with driver.session(database=NEO4J_DATABASE) as session:
        for batch in batched(filtered_rows, batch_size):
            payload = []
            for key_value, rec in batch:
                props = sanitize_properties(rec, key)
                payload.append({"key": key_value, "props": props})
            if not payload:
                continue
            summary = session.run(query, rows=payload).consume()
            total_created += summary.counters.nodes_created
    log(f"{label}: nodes created this run = {total_created}", verbose)
    return {"attempted": len(filtered_rows), "created": total_created}


def write_relationships(
    driver,
    start_label: str,
    start_key: str,
    end_label: str,
    end_key: str,
    rel_type: str,
    rows: Iterable[Dict[str, Any]],
    batch_size: int,
    verbose: bool,
) -> Dict[str, int]:
    """MERGE relationship batches into Neo4j and return creation statistics."""
    sanitized_rows: List[Dict[str, Any]] = []
    for row in rows:
        start_value = sanitize_value(row.get("start"))
        end_value = sanitize_value(row.get("end"))
        if start_value is None or end_value is None:
            continue
        props = sanitize_value(row.get("props", {}))
        sanitized_rows.append(
            {
                "start": start_value,
                "end": end_value,
                "props": props if isinstance(props, dict) else {},
            }
        )
    if not sanitized_rows:
        log(f"Skipping {rel_type}: no rows to write", verbose)
        return {"attempted": 0, "created": 0}
    log(f"Writing {len(sanitized_rows)} relationships {rel_type}", verbose)
    query = f"""
    UNWIND $rows AS row
    MATCH (start:{start_label} {{{start_key}: row.start}})
    MATCH (end:{end_label} {{{end_key}: row.end}})
    MERGE (start)-[rel:{rel_type}]->(end)
    SET rel += row.props
    """
    total_created = 0
    with driver.session(database=NEO4J_DATABASE) as session:
        for batch in batched(sanitized_rows, batch_size):
            if not batch:
                continue
            payload = batch
            summary = session.run(query, rows=payload).consume()
            total_created += summary.counters.relationships_created
    log(f"{rel_type}: relationships created this run = {total_created}", verbose)
    return {"attempted": len(sanitized_rows), "created": total_created}


def batched(items: List[Any], size: int) -> Iterator[List[Any]]:
    """Yield the input list in chunks of length `size`."""
    for index in range(0, len(items), size):
        yield items[index : index + size]


def load_frames(input_dir: Path, verbose: bool) -> Dict[str, pd.DataFrame]:
    """Load all enriched tables required for injection."""
    tables = {
        "work_final",
        "book_final",
        "series_final",
        "people_final",
        "publisher_final",
        "year_final",
        "country_final",
        "languages_final",
        "tags_final",
        "citation_mentions",
    }
    frames = {name: load_table(input_dir, name, verbose) for name in tables}
    return frames


def prepare_country_nodes(country_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert the country frame into dictionaries suitable for Neo4j nodes."""
    rows: List[Dict[str, Any]] = []
    if country_df is None or country_df.empty:
        return rows
    for row in country_df.itertuples(index=False):
        code = normalize_country(getattr(row, "country_code", None))
        name = clean_string(getattr(row, "country_name", None))
        if code:
            rows.append({"code": code, "name": name})
    return rows


def prepare_year_nodes(year_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert the year frame into node dictionaries (one per distinct year)."""
    rows: List[Dict[str, Any]] = []
    if year_df is None or year_df.empty:
        return rows
    for value in year_df.get("year", []):
        year = to_year(value)
        if year is not None:
            rows.append({"value": year})
    return rows


def prepare_language_nodes(frames: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Collect distinct language codes present across works/books."""
    languages: Set[str] = set()
    lang_df = frames.get("languages_final")
    if lang_df is not None and not lang_df.empty and "language_code" in lang_df.columns:
        for value in lang_df["language_code"].dropna():
            languages.add(str(value).strip())
    for table, column in (("book_final", "language_code"), ("work_final", "original_language_id")):
        df = frames.get(table)
        if df is not None and not df.empty and column in df.columns:
            for value in df[column].dropna():
                code = str(value).strip()
                if code:
                    languages.add(code)
    return [{"code": code} for code in sorted(languages)]


def prepare_tag_nodes(
    tags_df: pd.DataFrame,
    min_occurrences: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Filter and transform tag rows, returning nodes and a name→ID lookup."""
    rows: List[Dict[str, Any]] = []
    name_to_id: Dict[str, str] = {}
    if tags_df is None or tags_df.empty:
        return rows, name_to_id
    for tag in tags_df.itertuples(index=False):
        tag_id = clean_string(getattr(tag, "tag_id", None))
        tag_name = clean_string(getattr(tag, "tag_name", None))
        tag_type = clean_string(getattr(tag, "tag_type", None))
        occurrences = getattr(tag, "occurrences", None)
        occ_value = 0
        if occurrences is not None and not (isinstance(occurrences, float) and math.isnan(occurrences)):
            try:
                occ_value = int(occurrences)
            except (ValueError, TypeError):
                occ_value = 0
        if tag_id and occ_value >= max(1, min_occurrences):
            rows.append(
                {
                    "id": tag_id,
                    "name": tag_name,
                    "tag_type": tag_type,
                    "occurrences": occ_value,
                }
            )
            if tag_name:
                name_to_id[tag_name.lower()] = tag_id
    return rows, name_to_id


def prepare_publisher_nodes(publisher_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Normalize publisher rows into node dictionaries and name lookup maps."""
    rows: List[Dict[str, Any]] = []
    name_map: Dict[str, str] = {}
    if publisher_df is None or publisher_df.empty:
        return rows, name_map
    for pub in publisher_df.itertuples(index=False):
        publisher_id = clean_string(getattr(pub, "publisher_id", None))
        name = clean_string(getattr(pub, "publisher_name", None))
        wikidata_id = clean_string(getattr(pub, "publisher_wikidata_id", None))
        country = clean_string(getattr(pub, "country", None)) or clean_string(getattr(pub, "wikidata_country", None))
        year_established = to_year(getattr(pub, "year_established", None))
        wikidata_year = to_year(getattr(pub, "wikidata_year_established", None))
        inception_date = clean_string(getattr(pub, "wikidata_inception_date", None))
        if publisher_id and name:
            rows.append(
                {
                    "id": publisher_id,
                    "name": name,
                    "wikidata_id": wikidata_id,
                    "country": country,
                    "year_established": year_established,
                    "wikidata_year_established": wikidata_year,
                    "wikidata_inception_date": inception_date,
                }
            )
            name_map[name.lower()] = publisher_id
    return rows, name_map


def prepare_series_nodes(series_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Transform series records into Neo4j-ready node dictionaries."""
    rows: List[Dict[str, Any]] = []
    if series_df is None or series_df.empty:
        return rows
    for series in series_df.itertuples(index=False):
        series_id = clean_string(getattr(series, "series_id", None))
        if not series_id:
            continue
        rows.append(
            {
                "id": series_id,
                "title": clean_string(getattr(series, "title", None)),
                "description": clean_string(getattr(series, "description", None)),
                "series_works_count": getattr(series, "series_works_count", None),
                "primary_work_count": getattr(series, "primary_work_count", None),
                "numbered": clean_string(getattr(series, "numbered", None)),
                "keywords": flatten_strings(getattr(series, "keywords", [])) or None,
            }
        )
    return rows


def prepare_person_nodes(people_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Transform author/person records into node dictionaries."""
    rows: List[Dict[str, Any]] = []
    if people_df is None or people_df.empty:
        return rows
    for person in people_df.itertuples(index=False):
        person_id = clean_string(getattr(person, "author_id", None))
        if not person_id:
            continue
        rows.append(
            {
                "id": person_id,
                "name": clean_string(getattr(person, "name", None)),
                "average_rating": clean_string(getattr(person, "average_rating", None)),
                "ratings_count": getattr(person, "ratings_count", None),
                "text_reviews_count": getattr(person, "text_reviews_count", None),
                "wikidata_id": clean_string(getattr(person, "author_wikidata_id", None)),
                "date_of_birth": clean_string(getattr(person, "date_of_birth", None)),
                "date_of_death": clean_string(getattr(person, "date_of_death", None)),
                "year_of_birth": to_year(getattr(person, "year_of_birth", None)),
                "year_of_death": to_year(getattr(person, "year_of_death", None)),
                "age": getattr(person, "age", None),
                "place_of_birth": clean_string(getattr(person, "place_of_birth", None))
                or clean_string(getattr(person, "wikidata_place_of_birth", None)),
                "place_of_death": clean_string(getattr(person, "place_of_death", None))
                or clean_string(getattr(person, "wikidata_place_of_death", None)),
                "citizenship": flatten_strings(getattr(person, "citizenship", [])) or None,
                "gender": clean_string(getattr(person, "gender", None)),
            }
        )
    return rows


def prepare_work_nodes(work_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Transform work records into Neo4j node payloads."""
    rows: List[Dict[str, Any]] = []
    if work_df is None or work_df.empty:
        return rows
    for work in work_df.itertuples(index=False):
        work_id = clean_string(getattr(work, "work_id", None))
        if not work_id:
            continue
        rows.append(
            {
                "id": work_id,
                "original_title": clean_string(getattr(work, "original_title", None)),
                "description": clean_string(getattr(work, "best_book_description", None)),
                "original_language_id": clean_string(getattr(work, "original_language_id", None)),
                "original_publication_year": to_year(getattr(work, "original_publication_year", None)),
                "original_publication_month": to_year(getattr(work, "original_publication_month", None)),
                "original_publication_day": to_year(getattr(work, "original_publication_day", None)),
                "best_book_id": clean_string(getattr(work, "best_book_id", None)),
                "best_book_country_code": clean_string(getattr(work, "best_book_country_code", None)),
                "best_book_country_name": clean_string(getattr(work, "best_book_country_name", None)),
                "inferred_country_code": clean_string(getattr(work, "inferred_country_code", None)),
                "inferred_country_name": clean_string(getattr(work, "inferred_country_name", None)),
                "wikidata_id": clean_string(getattr(work, "wikidata_id", None)),
                "ddc": clean_string(getattr(work, "ddc", None)),
                "lcc": clean_string(getattr(work, "lcc", None)),
                "keywords": flatten_strings(getattr(work, "keywords", [])) or None,
                "tags": flatten_strings(getattr(work, "tags", [])) or None,
                "best_book_author_ids": flatten_strings(getattr(work, "best_book_author_ids", [])) or None,
                "best_book_series_ids": flatten_strings(getattr(work, "best_book_series_ids", [])) or None,
            }
        )
    return rows


def prepare_book_nodes(book_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return book node dictionaries and a book→work mapping."""
    rows: List[Dict[str, Any]] = []
    book_to_work: Dict[str, str] = {}
    if book_df is None or book_df.empty:
        return rows, book_to_work
    for book in book_df.itertuples(index=False):
        book_id = clean_string(getattr(book, "book_id", None))
        if not book_id:
            continue
        work_id = clean_string(getattr(book, "work_id", None))
        if work_id:
            book_to_work[book_id] = work_id
        rows.append(
            {
                "id": book_id,
                "title": clean_string(getattr(book, "title", None)),
                "title_without_series": clean_string(getattr(book, "title_without_series", None)),
                "description": clean_string(getattr(book, "description", None)),
                "image_url": clean_string(getattr(book, "image_url", None)),
                "goodreads_url": clean_string(getattr(book, "goodreads_url", None))
                or clean_string(getattr(book, "url", None)),
                "work_id": work_id,
                "publication_year": to_year(getattr(book, "publication_year", None)),
                "publication_month": to_year(getattr(book, "publication_month", None)),
                "publication_day": to_year(getattr(book, "publication_day", None)),
                "language_code": clean_string(getattr(book, "language_code", None)),
                "country_code": clean_string(getattr(book, "country_code", None))
                or clean_string(getattr(book, "inferred_country_code", None)),
                "num_pages": getattr(book, "num_pages", None),
                "isbn": clean_string(getattr(book, "isbn", None)),
                "isbn13": clean_string(getattr(book, "isbn13", None)),
                "publisher_name": clean_string(getattr(book, "publisher", None)),
                "format": clean_string(getattr(book, "format", None)),
                "edition_information": clean_string(getattr(book, "edition_information", None)),
                "authors": [
                    {
                        "author_id": clean_string(author.get("author_id")),
                        "role": clean_string(author.get("role")) or "Author",
                    }
                    for author in coerce_list(getattr(book, "authors", []))
                    if isinstance(author, dict) and clean_string(author.get("author_id"))
                ],
                "tags": flatten_strings(getattr(book, "tags", [])) or None,
                "series_ids": flatten_strings(getattr(book, "series_ids", [])) or None,
                "similar_books": flatten_strings(getattr(book, "similar_books", [])) or None,
                "keywords": flatten_strings(getattr(book, "keywords", [])) or None,
                "citations": flatten_strings(getattr(book, "citations", [])) or None,
            }
        )
    return rows, book_to_work


def build_work_author_map(work_rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build a mapping from work IDs to contributing author IDs."""
    mapping: Dict[str, List[str]] = {}
    for row in work_rows:
        work_id = row.get("id")
        if not work_id:
            continue
        authors = flatten_strings(row.get("best_book_author_ids", []))
        if authors:
            mapping[work_id] = authors
    return mapping


def resolve_book_author_roles(
    book_row: Dict[str, Any],
    work_author_map: Dict[str, List[str]],
) -> List[Tuple[str, Optional[str]]]:
    """Determine author roles for a book, falling back to work authors when missing."""
    explicit: List[Tuple[str, Optional[str]]] = []
    for author in book_row.get("authors", []) or []:
        if not isinstance(author, dict):
            continue
        author_id = sanitize_value(author.get("author_id"))
        if not author_id:
            continue
        role = sanitize_value(author.get("role")) or "Author"
        explicit.append((str(author_id), str(role)))
    if explicit:
        unique: Dict[str, str] = {}
        for author_id, role in explicit:
            if author_id not in unique:
                unique[author_id] = role
        return [(author_id, role) for author_id, role in unique.items()]

    work_id = book_row.get("work_id")
    author_ids = work_author_map.get(work_id, []) if work_id else []
    result: List[Tuple[str, Optional[str]]] = []
    for author_id in author_ids:
        cleaned = sanitize_value(author_id)
        if cleaned:
            result.append((str(cleaned), "Author"))
    return result


def build_book_relationship_data(
    book_rows: List[Dict[str, Any]],
    tag_map: Dict[str, str],
    publisher_map: Dict[str, str],
    work_author_map: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Construct all book-related relationships (edition, tags, publisher, authors)."""
    relationships: Dict[str, List[Dict[str, Any]]] = {
        "IS_EDITION": [],
        "BOOK_PUBLISHED_IN": [],
        "BOOK_YEAR": [],
        "BOOK_PUBLISHED_BY": [],
        "BOOK_HAS_TAG": [],
        "PERSON_BOOK": [],
    }
    person_book_seen: Set[Tuple[str, str]] = set()
    for row in book_rows:
        book_id = row["id"]
        work_id = row.get("work_id")
        if book_id and work_id:
            relationships["IS_EDITION"].append({"start": book_id, "end": work_id})
        country_code = row.get("country_code")
        if book_id and country_code:
            relationships["BOOK_PUBLISHED_IN"].append({"start": book_id, "end": country_code})
        year = row.get("publication_year")
        if book_id and year is not None:
            relationships["BOOK_YEAR"].append({"start": book_id, "end": year})
        publisher_name = row.get("publisher_name")
        if book_id and publisher_name:
            publisher_id = publisher_map.get(publisher_name.lower())
            if publisher_id:
                relationships["BOOK_PUBLISHED_BY"].append({"start": book_id, "end": publisher_id})
        for tag in flatten_strings(row.get("tags", [])):
            tag_id = tag_map.get(tag.lower())
            if book_id and tag_id:
                relationships["BOOK_HAS_TAG"].append({"start": book_id, "end": tag_id})
        for author_id, role in resolve_book_author_roles(row, work_author_map):
            if book_id and author_id and (author_id, book_id) not in person_book_seen:
                relationships["PERSON_BOOK"].append(
                    {
                        "start": author_id,
                        "end": book_id,
                        "props": {"role": role or "Author"},
                    }
                )
                person_book_seen.add((author_id, book_id))
    return relationships


def build_work_relationship_data(
    work_rows: List[Dict[str, Any]],
    tag_map: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Construct work relationships (series membership, tags, authors, countries)."""
    relationships: Dict[str, List[Dict[str, Any]]] = {
        "WORK_PART_OF": [],
        "WORK_HAS_TAG": [],
        "WORK_COUNTRY": [],
        "WORK_YEAR": [],
        "PERSON_WORK": [],
    }
    for row in work_rows:
        work_id = row["id"]
        for series_id in flatten_strings(row.get("best_book_series_ids", [])):
            relationships["WORK_PART_OF"].append({"start": work_id, "end": series_id})
        for tag in flatten_strings(row.get("tags", [])) + flatten_strings(row.get("keywords", [])):
            tag_id = tag_map.get(tag.lower())
            if tag_id:
                relationships["WORK_HAS_TAG"].append({"start": work_id, "end": tag_id})
        country_code = row.get("inferred_country_code") or row.get("best_book_country_code")
        if country_code:
            relationships["WORK_COUNTRY"].append({"start": work_id, "end": country_code})
        year = row.get("original_publication_year")
        if year is not None:
            relationships["WORK_YEAR"].append({"start": work_id, "end": year})
        for author_id in flatten_strings(row.get("best_book_author_ids", [])):
            relationships["PERSON_WORK"].append(
                {
                    "start": author_id,
                    "end": work_id,
                    "props": {"role": "Author"},
                }
            )
    return relationships


def build_series_relationship_data(
    series_rows: List[Dict[str, Any]],
    tag_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Create `Series` → `Tag` relationship payloads."""
    rows: List[Dict[str, Any]] = []
    for series in series_rows:
        series_id = series["id"]
        for keyword in flatten_strings(series.get("keywords", [])):
            tag_id = tag_map.get(keyword.lower())
            if tag_id:
                rows.append({"start": series_id, "end": tag_id})
    return rows


def build_publisher_relationship_data(
    publisher_rows: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble publisher relationships (LOCATED_IN, FOUNDED_IN)."""
    located_in: List[Dict[str, Any]] = []
    founded_in: List[Dict[str, Any]] = []
    for row in publisher_rows:
        publisher_id = row["id"]
        country = row.get("country")
        if country:
            located_in.append({"start": publisher_id, "end": country})
        for year in [row.get("year_established"), row.get("wikidata_year_established")]:
            year_value = to_year(year)
            if year_value is not None:
                founded_in.append({"start": publisher_id, "end": year_value})
                break
    return {"LOCATED_IN": located_in, "FOUNDED_IN": founded_in}


def build_person_relationship_data(
    people_rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    relationships: Dict[str, List[Dict[str, Any]]] = {
        "WAS_BORN": [],
        "WAS_DECEASED": [],
        "HAS_CITIZENSHIP": [],
        "BORN_IN": [],
    }
    for row in people_rows:
        person_id = row["id"]
        birth_year = row.get("year_of_birth")
        if birth_year is not None:
            relationships["WAS_BORN"].append({"start": person_id, "end": birth_year})
        death_year = row.get("year_of_death")
        if death_year is not None:
            relationships["WAS_DECEASED"].append({"start": person_id, "end": death_year})
        for country in flatten_strings(row.get("citizenship", [])):
            canonical = normalize_country(country)
            if canonical:
                relationships["HAS_CITIZENSHIP"].append({"start": person_id, "end": canonical})
        place_of_birth = normalize_country(row.get("place_of_birth"))
        if place_of_birth:
            relationships["BORN_IN"].append({"start": person_id, "end": place_of_birth})
    return relationships


def build_person_series_relationships(
    work_rows: List[Dict[str, Any]],
    work_author_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Build relationships linking people to series via the works they authored."""
    relationships: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in work_rows:
        work_id = row.get("id")
        if not work_id:
            continue
        author_ids = work_author_map.get(work_id, [])
        if not author_ids:
            continue
        for series_id in flatten_strings(row.get("best_book_series_ids", [])):
            if not series_id:
                continue
            for author_id in author_ids:
                if not author_id:
                    continue
                key = (author_id, series_id)
                if key in seen:
                    continue
                seen.add(key)
                relationships.append(
                    {"start": author_id, "end": series_id, "props": {"role": "Author", "source_work": work_id}}
                )
    return relationships


def build_person_publisher_relationships(
    book_rows: List[Dict[str, Any]],
    publisher_map: Dict[str, str],
    work_author_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Create `Person` → `Publisher` links based on the books they contributed to."""
    relationships: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in book_rows:
        publisher_name = row.get("publisher_name")
        if not publisher_name:
            continue
        publisher_id = publisher_map.get(publisher_name.lower())
        if not publisher_id:
            continue
        book_id = row.get("id")
        authors = resolve_book_author_roles(row, work_author_map)
        for author_id, _role in authors:
            if not author_id:
                continue
            key = (author_id, publisher_id)
            if key in seen:
                continue
            seen.add(key)
            relationships.append({"start": author_id, "end": publisher_id, "props": {"source_book": book_id}})
    return relationships


def build_optional_person_relationships(
    people_df: pd.DataFrame,
    tag_map: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Infer optional person relationships (related, lives_in, tag associations)."""
    relationships: Dict[str, List[Dict[str, Any]]] = {
        "IS_RELATED": [],
        "LIVES_IN": [],
        "HAS_TAG": [],
    }
    if people_df is None or people_df.empty:
        return relationships

    available = {col: col.lower() for col in people_df.columns}

    related_columns = [col for col in available if "related" in available[col]]
    lives_columns = [col for col in available if any(token in available[col] for token in ("resid", "live", "habit"))]
    tag_columns = [col for col in available if "tag" in available[col] or "keyword" in available[col]]

    seen_related: Set[Tuple[str, str]] = set()
    seen_lives: Set[Tuple[str, str]] = set()
    seen_tags: Set[Tuple[str, str]] = set()

    for person in people_df.itertuples(index=False):
        person_id = clean_string(getattr(person, "author_id", None))
        if not person_id:
            continue

        for column in related_columns:
            for target in flatten_strings(getattr(person, column, [])):
                target_id = clean_string(target)
                if target_id and target_id != person_id:
                    key = (person_id, target_id)
                    if key not in seen_related:
                        seen_related.add(key)
                        relationships["IS_RELATED"].append({"start": person_id, "end": target_id})

        for column in lives_columns:
            for place in flatten_strings(getattr(person, column, [])):
                country = normalize_country(place)
                if country:
                    key = (person_id, country)
                    if key not in seen_lives:
                        seen_lives.add(key)
                        relationships["LIVES_IN"].append({"start": person_id, "end": country})

        for column in tag_columns:
            for tag in flatten_strings(getattr(person, column, [])):
                tag_clean = clean_string(tag)
                if not tag_clean:
                    continue
                tag_id = tag_map.get(tag_clean.lower())
                if tag_id:
                    key = (person_id, tag_id)
                    if key not in seen_tags:
                        seen_tags.add(key)
                        relationships["HAS_TAG"].append({"start": person_id, "end": tag_id})

    return relationships


def build_citation_relationships(citations_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Convert citation rows into `Work`→`Work` and `Person`→`Work` relationships."""
    relationships = {"WORK_CITED_IN": [], "PERSON_NAMED_IN": []}
    if citations_df is None or citations_df.empty:
        return relationships
    for citation in citations_df.itertuples(index=False):
        source_type = clean_string(getattr(citation, "source_type", None))
        target_type = clean_string(getattr(citation, "target_type", None))
        source_id = clean_string(getattr(citation, "source_id", None))
        target_id = clean_string(getattr(citation, "target_id", None))
        if source_type == "work" and target_type == "work" and source_id and target_id:
            relationships["WORK_CITED_IN"].append({"start": source_id, "end": target_id})
        if source_type == "work" and target_type == "person" and source_id and target_id:
            relationships["PERSON_NAMED_IN"].append({"start": target_id, "end": source_id})
    return relationships


def build_work_similarity_relationships(
    book_rows: List[Dict[str, Any]],
    book_to_work: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Translate book similarity lists into `Work`→`Work` relationships."""
    relationships: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in book_rows:
        source_work = row.get("work_id")
        if not source_work:
            continue
        for similar_book_id in flatten_strings(row.get("similar_books", [])):
            target_work = book_to_work.get(similar_book_id)
            if not target_work or target_work == source_work:
                continue
            key = (source_work, target_work)
            if key in seen:
                continue
            seen.add(key)
            relationships.append({"start": source_work, "end": target_work})
    return relationships


def inject_data(args: argparse.Namespace) -> None:
    """Main driver that loads frames, prepares payloads, and writes to Neo4j."""
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD environment variable is not set. "
            "Configure credentials (e.g., via .env) before running the injection step."
        )

    input_dir = resolve_cli_path(args.input_dir)
    frames = load_frames(input_dir, args.verbose)

    country_rows = prepare_country_nodes(frames.get("country_final"))
    year_rows = prepare_year_nodes(frames.get("year_final"))
    language_rows = prepare_language_nodes(frames)
    tag_rows, tag_map = prepare_tag_nodes(frames.get("tags_final"), args.min_tag_occurrences)
    publisher_rows, publisher_name_map = prepare_publisher_nodes(frames.get("publisher_final"))
    series_rows = prepare_series_nodes(frames.get("series_final"))
    person_rows = prepare_person_nodes(frames.get("people_final"))
    work_rows = prepare_work_nodes(frames.get("work_final"))
    book_rows, book_to_work = prepare_book_nodes(frames.get("book_final"))

    work_author_map = build_work_author_map(work_rows)

    book_relationships = build_book_relationship_data(book_rows, tag_map, publisher_name_map, work_author_map)
    work_relationships = build_work_relationship_data(work_rows, tag_map)
    series_relationships = build_series_relationship_data(series_rows, tag_map)
    publisher_relationships = build_publisher_relationship_data(publisher_rows)
    person_relationships = build_person_relationship_data(person_rows)
    person_series_relationships = build_person_series_relationships(work_rows, work_author_map)
    person_publisher_relationships = build_person_publisher_relationships(
        book_rows, publisher_name_map, work_author_map
    )
    person_misc_relationships = build_optional_person_relationships(frames.get("people_final"), tag_map)
    citation_relationships = build_citation_relationships(frames.get("citation_mentions"))
    work_similarity_relationships = build_work_similarity_relationships(book_rows, book_to_work)

    node_stats: Dict[str, Dict[str, int]] = {}
    relationship_stats: Dict[str, Dict[str, int]] = {}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        create_constraints(driver)

        node_stats["Country"] = write_nodes(driver, "Country", "code", country_rows, args.batch_size, args.verbose)
        node_stats["Year"] = write_nodes(driver, "Year", "value", year_rows, args.batch_size, args.verbose)
        node_stats["Language"] = write_nodes(driver, "Language", "code", language_rows, args.batch_size, args.verbose)
        node_stats["Tag"] = write_nodes(driver, "Tag", "id", tag_rows, args.batch_size, args.verbose)
        node_stats["Publisher"] = write_nodes(driver, "Publisher", "id", publisher_rows, args.batch_size, args.verbose)
        node_stats["Series"] = write_nodes(driver, "Series", "id", series_rows, args.batch_size, args.verbose)
        node_stats["Person"] = write_nodes(driver, "Person", "id", person_rows, args.batch_size, args.verbose)
        node_stats["Work"] = write_nodes(driver, "Work", "id", work_rows, args.batch_size, args.verbose)
        node_stats["Book"] = write_nodes(driver, "Book", "id", book_rows, args.batch_size, args.verbose)

        relationship_stats["BOOK_IS_EDITION"] = write_relationships(
            driver,
            "Book",
            "id",
            "Work",
            "id",
            "IS_EDITION",
            book_relationships["IS_EDITION"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["BOOK_PUBLISHED_IN"] = write_relationships(
            driver,
            "Book",
            "id",
            "Country",
            "code",
            "PUBLISHED_IN",
            book_relationships["BOOK_PUBLISHED_IN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["BOOK_YEAR_PUBLISHED"] = write_relationships(
            driver,
            "Book",
            "id",
            "Year",
            "value",
            "YEAR_PUBLISHED",
            book_relationships["BOOK_YEAR"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["BOOK_PUBLISHED_BY"] = write_relationships(
            driver,
            "Book",
            "id",
            "Publisher",
            "id",
            "PUBLISHED_BY",
            book_relationships["BOOK_PUBLISHED_BY"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["BOOK_HAS_TAG"] = write_relationships(
            driver,
            "Book",
            "id",
            "Tag",
            "id",
            "HAS_TAG",
            book_relationships["BOOK_HAS_TAG"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_WORKED_ON_BOOK"] = write_relationships(
            driver,
            "Person",
            "id",
            "Book",
            "id",
            "WORKED_ON",
            book_relationships["PERSON_BOOK"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_WORKED_ON_SERIES"] = write_relationships(
            driver,
            "Person",
            "id",
            "Series",
            "id",
            "WORKED_ON",
            person_series_relationships,
            args.batch_size,
            args.verbose,
        )

        relationship_stats["WORK_PART_OF"] = write_relationships(
            driver,
            "Work",
            "id",
            "Series",
            "id",
            "PART_OF",
            work_relationships["WORK_PART_OF"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["WORK_HAS_TAG"] = write_relationships(
            driver,
            "Work",
            "id",
            "Tag",
            "id",
            "HAS_TAG",
            work_relationships["WORK_HAS_TAG"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["WORK_PUBLISHED_IN"] = write_relationships(
            driver,
            "Work",
            "id",
            "Country",
            "code",
            "PUBLISHED_IN",
            work_relationships["WORK_COUNTRY"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["WORK_FIRST_PUBLISHED"] = write_relationships(
            driver,
            "Work",
            "id",
            "Year",
            "value",
            "FIRST_PUBLISHED",
            work_relationships["WORK_YEAR"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_WORKED_ON_WORK"] = write_relationships(
            driver,
            "Person",
            "id",
            "Work",
            "id",
            "WORKED_ON",
            work_relationships["PERSON_WORK"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["WORK_SIMILAR_TO"] = write_relationships(
            driver,
            "Work",
            "id",
            "Work",
            "id",
            "SIMILAR_TO",
            work_similarity_relationships,
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_WORKS_FOR"] = write_relationships(
            driver,
            "Person",
            "id",
            "Publisher",
            "id",
            "WORKS_FOR",
            person_publisher_relationships,
            args.batch_size,
            args.verbose,
        )

        relationship_stats["SERIES_HAS_TAG"] = write_relationships(
            driver,
            "Series",
            "id",
            "Tag",
            "id",
            "HAS_TAG",
            series_relationships,
            args.batch_size,
            args.verbose,
        )

        relationship_stats["PUBLISHER_LOCATED_IN"] = write_relationships(
            driver,
            "Publisher",
            "id",
            "Country",
            "code",
            "LOCATED_IN",
            publisher_relationships["LOCATED_IN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PUBLISHER_FOUNDED_IN"] = write_relationships(
            driver,
            "Publisher",
            "id",
            "Year",
            "value",
            "FOUNDED_IN",
            publisher_relationships["FOUNDED_IN"],
            args.batch_size,
            args.verbose,
        )

        relationship_stats["PERSON_WAS_BORN"] = write_relationships(
            driver,
            "Person",
            "id",
            "Year",
            "value",
            "WAS_BORN",
            person_relationships["WAS_BORN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_WAS_DECEASED"] = write_relationships(
            driver,
            "Person",
            "id",
            "Year",
            "value",
            "WAS_DECEASED",
            person_relationships["WAS_DECEASED"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_HAS_CITIZENSHIP"] = write_relationships(
            driver,
            "Person",
            "id",
            "Country",
            "code",
            "HAS_CITIZENSHIP",
            person_relationships["HAS_CITIZENSHIP"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_BORN_IN"] = write_relationships(
            driver,
            "Person",
            "id",
            "Country",
            "code",
            "BORN_IN",
            person_relationships["BORN_IN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_LIVES_IN"] = write_relationships(
            driver,
            "Person",
            "id",
            "Country",
            "code",
            "LIVES_IN",
            person_misc_relationships["LIVES_IN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_IS_RELATED"] = write_relationships(
            driver,
            "Person",
            "id",
            "Person",
            "id",
            "IS_RELATED",
            person_misc_relationships["IS_RELATED"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_HAS_TAG"] = write_relationships(
            driver,
            "Person",
            "id",
            "Tag",
            "id",
            "HAS_TAG",
            person_misc_relationships["HAS_TAG"],
            args.batch_size,
            args.verbose,
        )

        relationship_stats["WORK_CITED_IN"] = write_relationships(
            driver,
            "Work",
            "id",
            "Work",
            "id",
            "CITED_IN",
            citation_relationships["WORK_CITED_IN"],
            args.batch_size,
            args.verbose,
        )
        relationship_stats["PERSON_NAMED_IN"] = write_relationships(
            driver,
            "Person",
            "id",
            "Work",
            "id",
            "NAMED_IN",
            citation_relationships["PERSON_NAMED_IN"],
            args.batch_size,
            args.verbose,
        )

        print("Data injection complete.")
        if args.verbose:
            with driver.session(database=NEO4J_DATABASE) as session:
                counts = session.run(
                    """
                    CALL {
                      MATCH (n) RETURN labels(n) AS label, count(*) AS cnt
                    }
                    RETURN label, cnt ORDER BY cnt DESC
                    """
                ).data()
                print("--- Node counts ---")
                for record in counts:
                    print(f"{record['label']}: {record['cnt']}")
    finally:
        driver.close()

    summary_payload = {
        "parameters": {
            "batch_size": args.batch_size,
            "min_tag_occurrences": args.min_tag_occurrences,
        },
        "nodes": node_stats,
        "relationships": relationship_stats,
    }
    if args.log_file:
        log_file = resolve_cli_path(args.log_file)
        ensure_directory(log_file.parent)
        with log_file.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2, sort_keys=True)
    print("--- Injection summary ---")
    for label, stats in sorted(node_stats.items()):
        print(f"  Node {label}: attempted {stats['attempted']}, created {stats['created']}")
    for rel, stats in sorted(relationship_stats.items()):
        print(f"  Relationship {rel}: attempted {stats['attempted']}, created {stats['created']}")


def main() -> None:
    """CLI entry point for the injection script."""
    args = parse_args()
    inject_data(args)


if __name__ == "__main__":
    main()
