#!/usr/bin/env python3
"""
Step 2 - Data Normalization

Takes the raw extraction output from Step 1 and:
  * normalizes identifier formats and key columns
  * removes duplicate rows in each entity table
  * harmonizes publisher and tag strings for downstream enrichment
  * produces fuzzy-merge suggestions for publishers and tags
  * optionally saves the normalized frames back to disk

Usage:
    python data-normalization.py --input-dir data/processed/extraction --export
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set

import pandas as pd

try:  # optional speed-up for similarity scoring
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - rapidfuzz is optional
    fuzz = None
    process = None

from utils import (
    ensure_directory,
    get_default,
    resolve_cli_path,
    resolve_path,
)

FrameDict = Dict[str, pd.DataFrame]

EXPECTED_FRAME_NAMES = [
    "work_final",
    "book_final",
    "series_final",
    "people_final",
    "publisher_final",
    "year_final",
    "country_final",
    "languages_final",
    "tags_final",
]

LIST_COLUMNS = {
    "work_final": [
        "best_book_author_ids",
        "best_book_series_ids",
        "associated_book_ids",
        "tags",
    ],
    "book_final": ["authors", "popular_shelves", "tags", "series_ids"],
    "series_final": ["keywords"],
}

COMMON_SHELF_TAG_STOPWORDS = {
    "to-read",
    "tbr",
    "currently-reading",
    "read",
    "read-in-2024",
    "favorites",
    "favourites",
    "favorite",
    "favourite",
    "wishlist",
    "owned",
    "own",
    "school",
    "physical-tbr",
    "digital-tbr",
    "read-for-school",
    "school-books",
    "for-school",
    "my-books",
    "library",
    "all-time-favorites",
    "re-read",
    "reread",
    "to-buy",
    "5-stars",
    "favorite-books",
    "bookshelf",
    "home-library",
    "favs",
    "books-i-have",
    "want-to-buy",
    "to-reread",
    "must-read",
    "must-have",
    "i-own",
    "read-again",
    "dnf",
    "did-not-finish",
    "want-to-read",
    "books-i-own",
    "books-i-own-but-havent-read",
    "owned-books",
    "my-library",
    "ebooks",
    "kindle",
    "kindle-books",
    "audible",
    "audio",
    "audio-books",
    "audiobooks",
    "library",
    "library-books",
    "book-club",
    "bookclub",
    "shelved",
}

COMMON_SHELF_TAG_SUBSTRINGS = {
    "to-read",
    "currently_reading",
    "currently-reading",
    "favorite",
    "favourite",
    "wishlist",
    "owned",
    "own",
    "library",
    "bookclub",
    "book-club",
    "club",
    "school",
    "tbr",
    "dnf",
    "kindle",
    "audiobook",
    "audio-book",
    "audible",
    "giveaway",
    "netgalley",
    "ebook",
    "e-book",
    "digital",
    "physical",
    "signed",
    "autographed",
    "filmed",
    "movie",
    "adaptation",
}

COMMON_SHELF_TAG_PATTERNS = [
    re.compile(r"\bread[-_ ]?(?:in|during)[-_ ]?\d{4}\b"),
    re.compile(r"\bred[-_ ]?\d{4}\b"),
    re.compile(r"\b\d{4}-?reads\b"),
    re.compile(r"\b\d{4}-?books\b"),
    re.compile(r"\bowned[-_ ]?\d{1,2}\b"),
    re.compile(r"\bbooks[-_ ]?i[-_ ]?(?:own|have)\b"),
]

DEFAULT_INPUT_DIR = resolve_path("extraction_dir")
DEFAULT_OUTPUT_DIR = resolve_path("normalization_dir")
DEFAULT_PUBLISHER_THRESHOLD = float(get_default("normalization", "publisher_threshold", 90.0))
DEFAULT_TAG_THRESHOLD = float(get_default("normalization", "tag_threshold", 92.0))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_nan(value: object) -> bool:
    """Return True when the value should be treated as missing/NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _to_string_identifier(value: object) -> str:
    """Normalize identifiers to stripped strings (empty string when missing)."""
    if _is_nan(value):
        return ""
    return str(value).strip()


def _parse_list_like(value: object) -> List[object]:
    """Parse list-like strings/containers into a Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_nan(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # try JSON first, then Python literal
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
                return [parsed]
            except (SyntaxError, ValueError):
                return [stripped]
    return [value]


def _normalize_string(value: object, *, lowercase: bool = False) -> str:
    """Return a stripped string representation (optionally lower-cased)."""
    if _is_nan(value):
        return ""
    text = str(value).strip()
    return text.lower() if lowercase else text


def _unique_list(values: Iterable[object], *, key=str) -> List[object]:
    """Return a list with duplicates removed while preserving order."""
    seen = set()
    result: List[object] = []
    for value in values:
        marker = key(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _ensure_list_columns(df: pd.DataFrame, list_columns: Sequence[str]) -> pd.DataFrame:
    """Ensure selected DataFrame columns contain Python lists."""
    df = df.copy()
    for column in list_columns:
        if column not in df.columns:
            continue
        df[column] = df[column].apply(_parse_list_like)
    return df


def _sanitize_identifier_list(values: Iterable[object]) -> List[str]:
    """Normalize an iterable of identifiers into a unique list of strings."""
    cleaned: List[str] = []
    for value in values:
        identifier = _to_string_identifier(value)
        if identifier:
            cleaned.append(identifier)
    return _unique_list(cleaned)


def _sanitize_string_list(
    values: Iterable[object],
    *,
    lowercase: bool = False,
    exclude: Optional[Set[str]] = None,
    drop_numeric: bool = False,
    min_length: int = 0,
    deny_substrings: Optional[Set[str]] = None,
    deny_patterns: Optional[Sequence[re.Pattern[str]]] = None,
) -> List[str]:
    """Normalize a collection of strings with optional filters (case, stopwords, digits)."""
    cleaned: List[str] = []
    for value in values:
        text = _normalize_string(value, lowercase=lowercase)
        comparison = text if lowercase else text.lower()
        if not text:
            continue
        if min_length and len(comparison) < min_length:
            continue
        if drop_numeric and comparison.isdigit():
            continue
        if exclude is not None and comparison in exclude:
            continue
        if deny_substrings and any(chunk in comparison for chunk in deny_substrings):
            continue
        if deny_patterns and any(pattern.search(comparison) for pattern in deny_patterns):
            continue
        cleaned.append(text)
    return _unique_list(cleaned)


def _normalize_merge_mapping(mapping: Dict[str, str], *, lowercase: bool = False) -> Dict[str, str]:
    """Normalize merge mapping keys/values for consistent downstream lookups."""
    normalized: Dict[str, str] = {}
    for source, target in mapping.items():
        source_norm = _normalize_string(source, lowercase=lowercase)
        target_norm = _normalize_string(target, lowercase=lowercase)
        if source_norm and target_norm:
            normalized[source_norm] = target_norm
    return normalized


def _flatten_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Resolve chained mappings so every key points directly to its canonical target."""
    flattened: Dict[str, str] = {}

    def resolve(name: str) -> str:
        seen = set()
        current = name
        while True:
            if current in seen:
                break
            seen.add(current)
            target = mapping.get(current)
            if not target or target == current:
                return target or current
            current = target
        return current

    keys = set(mapping.keys()) | set(mapping.values())
    for key in keys:
        resolved = resolve(key)
        if key != resolved:
            flattened[key] = resolved
    return flattened


def _apply_mapping_to_list(values: Iterable[str], mapping: Dict[str, str]) -> List[str]:
    """Apply a mapping to a list of strings while removing duplicates."""
    mapped = [mapping.get(value, value) for value in values]
    return _unique_list(mapped)


def _map_scalar_value(value: object, mapping: Dict[str, str], *, lowercase: bool = False) -> str:
    """Map a single value according to the provided merge mapping."""
    key = _normalize_string(value, lowercase=lowercase)
    return mapping.get(key, key)


# ---------------------------------------------------------------------------
# Merge resolution helpers
# ---------------------------------------------------------------------------

def build_auto_merge_mapping(
    suggestions: pd.DataFrame,
    *,
    lowercase: bool = False,
    canonical_strategy: str = "shortest",
) -> Dict[str, str]:
    """Automatically build a merge mapping from fuzzy suggestions using union-find."""
    if suggestions is None or suggestions.empty:
        return {}

    parent: Dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        if parent[name] != name:
            parent[name] = find(parent[name])
        return parent[name]

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        parent[root_b] = root_a

    def normalize(value: object) -> str:
        return _normalize_string(value, lowercase=lowercase)

    for _, row in suggestions.iterrows():
        source = normalize(row["source"])
        candidate = normalize(row["candidate"])
        if not source or not candidate or source == candidate:
            continue
        union(source, candidate)

    components: Dict[str, List[str]] = {}
    for name in parent:
        root = find(name)
        components.setdefault(root, []).append(name)

    def choose_canonical(values: List[str]) -> str:
        if canonical_strategy == "alphabetical":
            return min(values)
        return min(values, key=lambda item: (len(item), item))

    mapping: Dict[str, str] = {}
    for members in components.values():
        canonical = choose_canonical(members)
        for member in members:
            if member != canonical:
                mapping[member] = canonical
    return mapping


def interactive_merge_review(
    entity_name: str,
    suggestions: pd.DataFrame,
    *,
    lowercase: bool = False,
) -> Dict[str, str]:
    """Interactively review merge suggestions and return the chosen mapping."""
    if suggestions is None or suggestions.empty:
        print(f"No {entity_name} merge suggestions to review.")
        return {}

    print(f"Interactive {entity_name} merge review ({len(suggestions)} suggestions).")
    print("Options: [1] use first, [2] use second, [m] manual canonical, [s] skip, [q] quit.")

    mapping: Dict[str, str] = {}

    def normalize(value: object) -> str:
        return _normalize_string(value, lowercase=lowercase)

    for _, row in suggestions.iterrows():
        raw_a = row["source"]
        raw_b = row["candidate"]
        similarity = row.get("similarity")
        a = normalize(raw_a)
        b = normalize(raw_b)
        if not a or not b or a == b:
            continue
        resolved_a = mapping.get(a, a)
        resolved_b = mapping.get(b, b)
        if resolved_a == resolved_b:
            continue

        prompt = f"[{entity_name}] '{raw_a}' ↔ '{raw_b}'"
        if pd.notna(similarity):
            prompt += f" (similarity {similarity:.1f})"
        prompt += " => "

        while True:
            choice = input(prompt + "[1/2/m/s/q]: ").strip().lower()
            if choice in {"1", "2", "m", "s", "q", ""}:
                break
            print("  Invalid option, please choose 1, 2, m, s, or q.")

        if choice == "q":
            break
        if choice == "s" or choice == "":
            continue
        if choice == "1":
            canonical = resolved_a
        elif choice == "2":
            canonical = resolved_b
        else:  # manual
            manual = input("  Enter canonical value: ").strip()
            canonical = normalize(manual)
            if not canonical:
                print("  Empty canonical value ignored; skipping.")
                continue

        mapping[a] = canonical
        mapping[b] = canonical

    return _flatten_mapping(mapping)


def apply_merge_mapping(
    frames: FrameDict,
    mapping: Dict[str, str],
    *,
    entity: str,
    lowercase: bool = False,
) -> Dict[str, str]:
    """Apply a merge mapping to the specified entity DataFrames."""
    normalized_mapping = _normalize_merge_mapping(mapping, lowercase=lowercase)
    flattened = _flatten_mapping(normalized_mapping)
    if not flattened:
        return {}

    if entity == "publisher":
        publisher_df = frames["publisher_final"].copy()
        publisher_df["publisher_name"] = publisher_df["publisher_name"].apply(
            lambda value: _map_scalar_value(value, flattened, lowercase=lowercase)
        )
        frames["publisher_final"] = normalize_publisher_dataframe(publisher_df)

        book_df = frames["book_final"].copy()
        book_df["publisher"] = book_df["publisher"].apply(
            lambda value: _map_scalar_value(value, flattened, lowercase=lowercase)
        )
        frames["book_final"] = normalize_book_dataframe(book_df)
    elif entity == "tag":
        tags_df = frames["tags_final"].copy()
        tags_df["tag_name"] = tags_df["tag_name"].apply(
            lambda value: _map_scalar_value(value, flattened, lowercase=lowercase)
        )
        frames["tags_final"] = normalize_tags_dataframe(tags_df)

        book_df = frames["book_final"].copy()
        book_df["tags"] = book_df["tags"].apply(lambda values: _apply_mapping_to_list(values, flattened))
        frames["book_final"] = normalize_book_dataframe(book_df)

        work_df = frames["work_final"].copy()
        work_df["tags"] = work_df["tags"].apply(lambda values: _apply_mapping_to_list(values, flattened))
        frames["work_final"] = normalize_work_dataframe(work_df)
    else:  # pragma: no cover - safeguard
        raise ValueError(f"Unknown merge entity: {entity}")

    return flattened


def refresh_merge_suggestions(
    frames: FrameDict,
    *,
    publisher_threshold: float,
    tag_threshold: float,
) -> None:
    """Recompute fuzzy merge suggestions for publishers and tags."""
    frames["publisher_merge_suggestions"] = _suggest_similar_items(
        frames["publisher_final"]["publisher_name"].tolist(), threshold=publisher_threshold
    )
    frames["tag_merge_suggestions"] = _suggest_similar_items(
        frames["tags_final"]["tag_name"].tolist(), threshold=tag_threshold
    )


def _normalized_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _serialize_list_columns_for_csv(df: pd.DataFrame, list_columns: Sequence[str]) -> pd.DataFrame:
    """Serialize list columns as JSON strings for CSV fallback output."""
    df = df.copy()
    for column in list_columns:
        if column not in df.columns:
            continue
        df[column] = df[column].apply(json.dumps)
    return df


def _suggest_similar_items(items: Iterable[str], *, threshold: float, limit: int = 5) -> pd.DataFrame:
    """Return candidate pairs of similar strings above a fuzzy similarity threshold."""
    values = list(dict.fromkeys(item for item in items if item))
    if not values:
        return pd.DataFrame(columns=["source", "candidate", "similarity"])

    suggestions: List[tuple[str, str, float]] = []

    if process and fuzz:  # pragma: no cover - depends on optional dependency
        for value in values:
            matches = process.extract(
                value,
                values,
                scorer=fuzz.token_sort_ratio,
                limit=limit,
            )
            for candidate, score, _ in matches:
                if candidate == value or score < threshold:
                    continue
                ordered_pair = tuple(sorted([value, candidate]))
                suggestions.append((ordered_pair[0], ordered_pair[1], float(score)))
    else:
        index: Dict[str, List[str]] = {}
        for value in values:
            key = value[:1].lower()
            index.setdefault(key, []).append(value)
        for bucket in index.values():
            for i, value in enumerate(bucket):
                for candidate in bucket[i + 1 :]:
                    score = similarity_ratio(value, candidate)
                    if score >= threshold:
                        ordered_pair = tuple(sorted([value, candidate]))
                        suggestions.append((ordered_pair[0], ordered_pair[1], score))

    if not suggestions:
        return pd.DataFrame(columns=["source", "candidate", "similarity"])
    df = pd.DataFrame(suggestions, columns=["source", "candidate", "similarity"])
    return df.drop_duplicates().sort_values(["source", "candidate"]).reset_index(drop=True)


def similarity_ratio(a: str, b: str) -> float:
    """Approximate fuzzy similarity between two strings on a 0-100 scale."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # simple token-sort ratio analogue without external dependencies
    tokens_a = sorted(a.lower().split())
    tokens_b = sorted(b.lower().split())
    if tokens_a == tokens_b:
        return 100.0
    joined_a = " ".join(tokens_a)
    joined_b = " ".join(tokens_b)
    # token level Jaccard-style ratio
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    base_score = 100.0 * intersection / union if union else 100.0
    # adjust using character overlap
    char_overlap = len(set(joined_a) & set(joined_b))
    char_union = len(set(joined_a) | set(joined_b))
    char_score = 100.0 * char_overlap / char_union if char_union else 100.0
    return round((base_score + char_score) / 2.0, 2)


# ---------------------------------------------------------------------------
# Normalization routines
# ---------------------------------------------------------------------------

def normalize_work_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean work-level columns and ensure list fields are normalized."""
    df = _ensure_list_columns(df, LIST_COLUMNS.get("work_final", []))
    df = df.copy()
    df["work_id"] = df["work_id"].apply(_to_string_identifier)
    df = df[df["work_id"] != ""]
    df["best_book_id"] = df["best_book_id"].apply(_to_string_identifier)
    df["original_language_id"] = df["original_language_id"].apply(_to_string_identifier)
    df["best_book_author_ids"] = df["best_book_author_ids"].apply(_sanitize_identifier_list)
    df["best_book_series_ids"] = df["best_book_series_ids"].apply(_sanitize_identifier_list)
    df["associated_book_ids"] = df["associated_book_ids"].apply(_sanitize_identifier_list)
    df["tags"] = df["tags"].apply(
        lambda values: _sanitize_string_list(
            values,
            lowercase=True,
            exclude=COMMON_SHELF_TAG_STOPWORDS,
            drop_numeric=True,
            min_length=3,
            deny_substrings=COMMON_SHELF_TAG_SUBSTRINGS,
            deny_patterns=COMMON_SHELF_TAG_PATTERNS,
        )
    )
    df = df.drop_duplicates(subset=["work_id"]).sort_values("work_id").reset_index(drop=True)
    return df


def normalize_book_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize book rows, harmonizing identifiers, tags, and author metadata."""
    df = _ensure_list_columns(df, LIST_COLUMNS.get("book_final", []))
    df = df.copy()
    df["book_id"] = df["book_id"].apply(_to_string_identifier)
    df = df[df["book_id"] != ""]
    df["work_id"] = df["work_id"].apply(_to_string_identifier)
    df["language_code"] = df["language_code"].apply(lambda x: _normalize_string(x, lowercase=True))
    df["country_code"] = df["country_code"].apply(_normalize_string).str.upper()
    df["publisher"] = df["publisher"].apply(_normalize_string)
    df["popular_shelves"] = df["popular_shelves"].apply(
        lambda values: _sanitize_string_list(
            values,
            lowercase=True,
            exclude=COMMON_SHELF_TAG_STOPWORDS,
            drop_numeric=True,
            min_length=3,
            deny_substrings=COMMON_SHELF_TAG_SUBSTRINGS,
            deny_patterns=COMMON_SHELF_TAG_PATTERNS,
        )
    )
    df["tags"] = df["tags"].apply(
        lambda values: _sanitize_string_list(
            values,
            lowercase=True,
            exclude=COMMON_SHELF_TAG_STOPWORDS,
            drop_numeric=True,
            min_length=3,
            deny_substrings=COMMON_SHELF_TAG_SUBSTRINGS,
            deny_patterns=COMMON_SHELF_TAG_PATTERNS,
        )
    )
    df["series_ids"] = df["series_ids"].apply(_sanitize_identifier_list)
    df["authors"] = df["authors"].apply(
        lambda authors: [
            {
                **{k: v for k, v in author.items() if k in ("author_id", "role")},
                "author_id": _to_string_identifier(author.get("author_id")),
                "role": _normalize_string(author.get("role")),
            }
            for author in (authors if isinstance(authors, list) else [])
            if isinstance(author, MutableMapping) and _to_string_identifier(author.get("author_id"))
        ]
    )
    df = df.drop_duplicates(subset=["book_id"]).sort_values("book_id").reset_index(drop=True)
    return df


def normalize_people_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize author/person records including derived list columns."""
    df = df.copy()
    df["author_id"] = df["author_id"].apply(_to_string_identifier)
    df = df[df["author_id"] != ""]
    df = df.drop_duplicates(subset=["author_id"]).sort_values("author_id").reset_index(drop=True)
    return df


def normalize_series_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize series rows and ensure keyword lists are sanitized."""
    df = _ensure_list_columns(df, LIST_COLUMNS.get("series_final", []))
    df = df.copy()
    df["series_id"] = df["series_id"].apply(_to_string_identifier)
    df = df[df["series_id"] != ""]
    df["keywords"] = df["keywords"].apply(lambda values: _sanitize_string_list(values, lowercase=True))
    df = df.drop_duplicates(subset=["series_id"]).sort_values("series_id").reset_index(drop=True)
    return df


def normalize_publisher_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize publisher names and Wikidata identifiers."""
    df = df.copy()
    df["publisher_id"] = df["publisher_id"].apply(_to_string_identifier)
    df = df[df["publisher_id"] != ""]
    df["publisher_name"] = df["publisher_name"].apply(_normalize_string)
    df["publisher_key"] = df["publisher_name"].apply(_normalized_key)
    df = df.groupby("publisher_key", as_index=False).agg(
        {
            "publisher_name": "first",
            "publisher_id": lambda values: sorted(set(filter(None, (_to_string_identifier(v) for v in values)))),
            "publisher_wikidata_id": "first",
            "country": "first",
            "year_established": "first",
        }
    )
    df["publisher_id"] = df["publisher_id"].apply(lambda values: values[0] if values else "")
    df = df[df["publisher_id"] != ""]
    return df.sort_values("publisher_name").reset_index(drop=True)


def normalize_tags_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tag strings and drop stopword shelves/numeric-only tags."""
    df = df.copy()
    df["tag_id"] = df["tag_id"].apply(_to_string_identifier)
    df = df[df["tag_id"] != ""]
    df["tag_name"] = df["tag_name"].apply(_normalize_string)
    df["tag_key"] = df["tag_name"].apply(_normalized_key)
    df["tag_lower"] = df["tag_name"].str.lower()
    df = df[~df["tag_lower"].isin(COMMON_SHELF_TAG_STOPWORDS)]
    df = df[~df["tag_lower"].apply(lambda value: any(sub in value for sub in COMMON_SHELF_TAG_SUBSTRINGS))]
    for pattern in COMMON_SHELF_TAG_PATTERNS:
        df = df[~df["tag_lower"].str.contains(pattern, na=False)]
    df = df[~df["tag_lower"].str.isdigit()]
    df = df.drop(columns=["tag_lower"])
    def aggregate_tag_types(values: Iterable[str]) -> str:
        normalized: List[str] = []
        for entry in values:
            if not entry or _is_nan(entry):
                continue
            if isinstance(entry, str):
                normalized.extend(part.strip() for part in entry.split(";") if part.strip())
        return ";".join(sorted(set(normalized)))

    df = df.groupby("tag_key", as_index=False).agg(
        {
            "tag_name": "first",
            "tag_id": lambda values: sorted(set(filter(None, (_to_string_identifier(v) for v in values)))),
            "tag_type": aggregate_tag_types,
            "occurrences": "sum",
        }
    )
    df["tag_id"] = df["tag_id"].apply(lambda values: values[0] if values else "")
    df = df[df["tag_id"] != ""]
    return df.sort_values("tag_name").reset_index(drop=True)


def normalize_year_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize year values to integers and drop invalid entries."""
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)
    return df


def normalize_country_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize country names and codes."""
    df = df.copy()
    df["country_code"] = df["country_code"].apply(_normalize_string).str.upper()
    df = df[df["country_code"] != ""]
    df = df.drop_duplicates(subset=["country_code"]).sort_values("country_code").reset_index(drop=True)
    return df


def normalize_language_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize language codes to lowercase ISO strings."""
    df = df.copy()
    df["language_code"] = df["language_code"].apply(_normalize_string).str.lower()
    df = df[df["language_code"] != ""]
    df = df.drop_duplicates(subset=["language_code"]).sort_values("language_code").reset_index(drop=True)
    return df


NORMALIZATION_DISPATCH = {
    "work_final": normalize_work_dataframe,
    "book_final": normalize_book_dataframe,
    "series_final": normalize_series_dataframe,
    "people_final": normalize_people_dataframe,
    "publisher_final": normalize_publisher_dataframe,
    "tags_final": normalize_tags_dataframe,
    "year_final": normalize_year_dataframe,
    "country_final": normalize_country_dataframe,
    "languages_final": normalize_language_dataframe,
}


# ---------------------------------------------------------------------------
# Loading / exporting
# ---------------------------------------------------------------------------

def load_frames(input_dir: Path) -> FrameDict:
    """Load all expected Step 1 frames (Parquet or CSV) into memory."""
    frames: FrameDict = {}
    for name in EXPECTED_FRAME_NAMES:
        frames[name] = _load_single_frame(input_dir, name)
    return frames


def _load_single_frame(input_dir: Path, name: str) -> pd.DataFrame:
    """Load a single frame from Parquet/CSV, coercing list columns when needed."""
    parquet_path = input_dir / f"{name}.parquet"
    csv_path = input_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if name in LIST_COLUMNS:
            df = _ensure_list_columns(df, LIST_COLUMNS[name])
        return df
    raise FileNotFoundError(f"Could not find {name}.parquet or {name}.csv in {input_dir}")


def export_frames(frames: FrameDict, output_dir: Path) -> None:
    """Write the normalized frames to disk (Parquet preferred, CSV fallback)."""
    ensure_directory(output_dir)
    parquet_supported = True
    for name, df in frames.items():
        target = output_dir / f"{name}.parquet"
        use_list_columns = LIST_COLUMNS.get(name, [])
        try:
            # Parquet expects lists to be Python lists; ensure they are lists before export
            df_to_write = _ensure_list_columns(df, use_list_columns)
            df_to_write.to_parquet(target, index=False)
        except (ImportError, ValueError, TypeError):
            parquet_supported = False
            csv_path = output_dir / f"{name}.csv"
            df_to_write = _serialize_list_columns_for_csv(df, use_list_columns)
            df_to_write.to_csv(csv_path, index=False)
    if not parquet_supported:
        print("Warning: pyarrow/fastparquet unavailable. Wrote CSV files instead of Parquet.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def normalize_frames(
    frames: FrameDict,
    *,
    publisher_similarity_threshold: float = 90.0,
    tag_similarity_threshold: float = 92.0,
) -> FrameDict:
    """Return normalized copies of all frames, refreshed with merge suggestions."""
    normalized: FrameDict = {}
    for name, frame in frames.items():
        if name not in NORMALIZATION_DISPATCH:
            normalized[name] = frame.copy()
            continue
        normalized[name] = NORMALIZATION_DISPATCH[name](frame)

    refresh_merge_suggestions(
        normalized,
        publisher_threshold=publisher_similarity_threshold,
        tag_threshold=tag_similarity_threshold,
    )

    return normalized


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the normalization step."""
    parser = argparse.ArgumentParser(description="Normalize Goodreads extraction tables.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Step 1 output (Parquet/CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where normalized frames will be written.",
    )
    parser.add_argument(
        "--publisher-threshold",
        type=float,
        default=DEFAULT_PUBLISHER_THRESHOLD,
        help="Similarity threshold (0-100) for publisher merge suggestions.",
    )
    parser.add_argument(
        "--tag-threshold",
        type=float,
        default=DEFAULT_TAG_THRESHOLD,
        help="Similarity threshold (0-100) for tag merge suggestions.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Persist normalized frames to disk.",
    )
    parser.add_argument(
        "--review-merges",
        action="store_true",
        help="Interactively review merge suggestions before applying them.",
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help="Automatically merge suggestions using heuristic canonical choices.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the normalization process based on CLI parameters."""
    args = parse_args(argv)
    input_dir = resolve_cli_path(args.input_dir)
    frames = load_frames(input_dir)
    normalized = normalize_frames(
        frames,
        publisher_similarity_threshold=args.publisher_threshold,
        tag_similarity_threshold=args.tag_threshold,
    )

    publisher_mapping_records: Dict[str, str] = {}
    tag_mapping_records: Dict[str, str] = {}
    merge_log_records: List[Dict[str, str]] = []

    def update_records(records: Dict[str, str], new_mapping: Dict[str, str]) -> None:
        for source, canonical in new_mapping.items():
            records.setdefault(source, canonical)

    def record_merge(entity: str, mapping: Dict[str, str], method: str) -> None:
        for source, target in mapping.items():
            merge_log_records.append(
                {
                    "entity": entity,
                    "source": source,
                    "target": target,
                    "method": method,
                }
            )

    changes_applied = False

    if args.review_merges:
        manual_pub = interactive_merge_review(
            "Publisher",
            normalized.get("publisher_merge_suggestions"),
            lowercase=False,
        )
        resolved_pub = apply_merge_mapping(normalized, manual_pub, entity="publisher", lowercase=False)
        if resolved_pub:
            update_records(publisher_mapping_records, resolved_pub)
            record_merge("publisher", resolved_pub, "manual")
            changes_applied = True
            print(f"Applied {len(resolved_pub)} publisher merges (manual).")

        manual_tag = interactive_merge_review(
            "Tag",
            normalized.get("tag_merge_suggestions"),
            lowercase=True,
        )
        resolved_tag = apply_merge_mapping(normalized, manual_tag, entity="tag", lowercase=True)
        if resolved_tag:
            update_records(tag_mapping_records, resolved_tag)
            record_merge("tag", resolved_tag, "manual")
            changes_applied = True
            print(f"Applied {len(resolved_tag)} tag merges (manual).")

        if changes_applied:
            refresh_merge_suggestions(
                normalized,
                publisher_threshold=args.publisher_threshold,
                tag_threshold=args.tag_threshold,
            )

        changes_applied = False

    if args.auto_merge:
        auto_pub = build_auto_merge_mapping(
            normalized.get("publisher_merge_suggestions"),
            lowercase=False,
        )
        resolved_pub = apply_merge_mapping(normalized, auto_pub, entity="publisher", lowercase=False)
        if resolved_pub:
            update_records(publisher_mapping_records, resolved_pub)
            record_merge("publisher", resolved_pub, "auto")
            changes_applied = True
            print(f"Applied {len(resolved_pub)} publisher merges (auto).")

        auto_tag = build_auto_merge_mapping(
            normalized.get("tag_merge_suggestions"),
            lowercase=True,
        )
        resolved_tag = apply_merge_mapping(normalized, auto_tag, entity="tag", lowercase=True)
        if resolved_tag:
            update_records(tag_mapping_records, resolved_tag)
            record_merge("tag", resolved_tag, "auto")
            changes_applied = True
            print(f"Applied {len(resolved_tag)} tag merges (auto).")

        if changes_applied:
            refresh_merge_suggestions(
                normalized,
                publisher_threshold=args.publisher_threshold,
                tag_threshold=args.tag_threshold,
            )

    if publisher_mapping_records:
        normalized["publisher_merge_mapping"] = pd.DataFrame(
            sorted(publisher_mapping_records.items()),
            columns=["source", "canonical"],
        )
    if tag_mapping_records:
        normalized["tag_merge_mapping"] = pd.DataFrame(
            sorted(tag_mapping_records.items()),
            columns=["source", "canonical"],
        )

    output_dir: Optional[Path] = None
    if args.export:
        output_dir = resolve_cli_path(args.output_dir)
        export_frames(normalized, output_dir)
        if merge_log_records:
            log_path = output_dir / "normalization_merge_log.csv"
            pd.DataFrame(merge_log_records).drop_duplicates().to_csv(log_path, index=False)

    print("Normalization complete:")
    for name in sorted(normalized.keys()):
        normalized_count = len(normalized[name])
        original_count = len(frames.get(name, []))
        delta = normalized_count - original_count
        sign = "+" if delta >= 0 else ""
        print(f"  {name}: {original_count} -> {normalized_count} ({sign}{delta})")
    if output_dir is not None:
        print(f"Normalized tables exported to {output_dir}")


if __name__ == "__main__":
    main()
