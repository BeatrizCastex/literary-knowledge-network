#!/usr/bin/env python3
"""
Step 3 – Data Enrichment

Loads the normalized Goodreads tables (Step 2), enriches them with:
  * additional country inference for works/books based on publisher info and Wikidata
  * optional Wikidata lookups (DDC/LCC for works, metadata for people/publishers)
  * enhanced keyword extraction using TF-IDF, RAKE, or a hybrid approach
  * lightweight named-entity recognition to detect citations between works and people

The script is designed to degrade gracefully when optional dependencies (requests,
scikit-learn, spaCy, etc.) are unavailable; enrichments that rely on them simply
emit warnings and skip without failing the overall pipeline.

Usage:
    python3 enritch-data.py --input-dir data/processed/normalized --output-dir data/processed/enriched --keyword-method hybrid --auto-wikidata --progress --export

spaCy model: `en_core_web_sm` (v3.x) is used for NER when available.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, cast

import pandas as pd

# Optional imports – the enrichment pipeline can operate without them,
# but certain features will be disabled.
try:
    import requests
except ImportError:  # pragma: no cover - requests should be installed but we guard anyway
    requests = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - TF-IDF extraction will be skipped
    TfidfVectorizer = None

try:
    import spacy
except ImportError:  # pragma: no cover - spaCy-based NER optional
    spacy = None

if TYPE_CHECKING:  # used solely for type hints
    from spacy.language import Language as SpacyLanguage
else:  # runtime fallback when spaCy is unavailable
    SpacyLanguage = Any  # type: ignore[assignment]

try:
    import pycountry
except ImportError:  # pragma: no cover - optional dependency
    pycountry = None

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None
    HAS_RAPIDFUZZ = False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm fallback
    tqdm = None

from utils import (
    ensure_directory,
    get_default,
    resolve_cli_path,
    resolve_path,
)
WIKIDATA_AGENT = "GoodreadsEnrichmentBot/0.1 (contact@example.com)"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

WIKIDATA_P_DEWEY = "P1036"
WIKIDATA_P_LCC = "P1149"
WIKIDATA_P_COUNTRY = "P495"
WIKIDATA_P_PLACE_OF_BIRTH = "P19"
WIKIDATA_P_PLACE_OF_DEATH = "P20"
WIKIDATA_P_GENDER = "P21"
WIKIDATA_P_CITIZENSHIP = "P27"
WIKIDATA_P_DATE_OF_BIRTH = "P569"
WIKIDATA_P_DATE_OF_DEATH = "P570"
WIKIDATA_P_INCEPTION = "P571"
WIKIDATA_P_AUTHOR = "P50"
WIKIDATA_P_PUBLISHER = "P123"
WIKIDATA_P_PUBLICATION_DATE = "P577"
WIKIDATA_P_ISBN13 = "P212"
WIKIDATA_P_ISBN10 = "P957"
WIKIDATA_P_OCCUPATION = "P106"

WIKIDATA_P_PUBLISHER_COUNTRY = "P17"
DEFAULT_LANGUAGE = "en"

DEFAULT_INPUT_DIR = resolve_path("normalization_dir")
DEFAULT_OUTPUT_DIR = resolve_path("enrichment_dir")
DEFAULT_KEYWORD_METHOD = str(get_default("enrichment", "keyword_method", "hybrid"))
DEFAULT_AUTO_WIKIDATA = bool(get_default("enrichment", "auto_wikidata", True))
DEFAULT_WIKIDATA_CACHE = resolve_path("logs_dir") / "wikidata-cache"

LIST_COLUMNS = {
    "work_final": ["tags", "best_book_author_ids", "best_book_series_ids", "associated_book_ids"],
    "book_final": ["tags", "popular_shelves", "series_ids", "authors"],
    "series_final": ["keywords"],
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "between",
    "by",
    "com",
    "con",
    "che",
    "de",
    "di",
    "el",
    "een",
    "en",
    "er",
    "for",
    "from",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "il",
    "in",
    "into",
    "is",
    "it",
    "its",
    "ji",
    "kai",
    "la",
    "las",
    "por",
    "del",
    "su",
    "los",
    "lo",
    "mn",
    "fy",
    "lm",
    "wl",
    "et",
    "des",
    "une",
    "dans",
    "est",
    "du",
    "de",
    "da",
    "ele",
    "ela",
    "dele",
    "dela",
    "tu",
    "você",
    "nostro",
    "nosso",
    "le",
    "les",
    "mas",
    "na",
    "o",
    "och",
    "of",
    "og",
    "on",
    "or",
    "som",
    "our",
    "ours",
    "over",
    "po",
    "que",
    "se",
    "she",
    "shi",
    "that",
    "the",
    "this",
    "through",
    "to",
    "ton",
    "under",
    "un",
    "una",
    "us",
    "was",
    "we",
    "were",
    "which",
    "who",
    "whom",
    "without",
    "with",
    "yours",
    "your",
    "you",
    "zi",
    "za",
}
# treat standalone digits as stopwords as well
STOPWORDS.update({str(digit) for digit in range(10)})
STOPWORDS_LIST = sorted(STOPWORDS)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_nan(value: object) -> bool:
    """Return True when the supplied value should be treated as missing."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _ensure_list(value: object) -> List[object]:
    """Coerce scalars into one-element lists and pass lists through unchanged."""
    if isinstance(value, list):
        return value
    if _is_nan(value):
        return []
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [value]
    if isinstance(value, (set, tuple)):
        return list(value)
    return [value]


def _normalize_text(text: Optional[str]) -> str:
    """Normalize optional text by replacing `None` with an empty, trimmed string."""
    return (text or "").strip()


def _normalize_token(token: str) -> str:
    """Return a simplified token (lowercase, non-word characters removed)."""
    return re.sub(r"\W+", "", token.lower())


def _standardize_country(value: Any) -> Optional[Tuple[str, str]]:
    """Attempt to resolve a country name/code into `(code, name)`."""
    if value is None or _is_nan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if pycountry:
        try:
            country = pycountry.countries.lookup(text)
            code = getattr(country, "alpha_2", getattr(country, "alpha_3", "")).upper()
            name = getattr(country, "common_name", getattr(country, "name", text))
            return code, name
        except LookupError:
            pass
    if len(text) in (2, 3) and text.isalpha():
        return text.upper(), text.upper()
    return text, text


def _split_multi_value(value: Any) -> List[str]:
    """Split semicolon-delimited values into a cleaned list of strings."""
    if value is None or _is_nan(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value)
    if not text.strip():
        return []
    parts = [segment.strip() for segment in text.split(";")]
    return [segment for segment in parts if segment]


def string_similarity(a: str, b: str) -> float:
    """Return a 0-100 fuzzy similarity score between two strings."""
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ and fuzz is not None:
        return float(fuzz.token_set_ratio(a, b))
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0


def generate_title_variants(text: str) -> Set[str]:
    """Derive a set of cleaned/shortened title variants for fuzzy matching."""
    variants: Set[str] = set()
    if not text:
        return variants
    base = text.strip()
    if not base:
        return variants
    variants.add(base)
    cleaned = re.sub(r"\s+", " ", base)
    variants.add(cleaned)
    if ":" in base:
        variants.add(base.split(":", 1)[0].strip())
    if "(" in base:
        variants.add(base.split("(", 1)[0].strip())
    simplified = re.sub(r"[^\w\s]", " ", base)
    variants.add(re.sub(r"\s+", " ", simplified).strip())
    return {variant for variant in variants if variant}


def build_work_context(
    work_row: Dict[str, Any],
    associated_books: List[Any],
    author_name_map: Dict[str, str],
) -> Dict[str, Any]:
    """Collect titles, authors, ISBNs, and other context for a work record."""
    title_variants: Set[str] = set()
    if work_row.get("original_title"):
        title_variants.update(generate_title_variants(str(work_row["original_title"])))

    publication_years: Set[int] = set()
    isbn_values: Set[str] = set()
    publisher_names: Set[str] = set()
    book_titles: Set[str] = set()

    for book in associated_books:
        title = getattr(book, "title", None)
        title_no_series = getattr(book, "title_without_series", None)
        for candidate_title in (title, title_no_series):
            if candidate_title:
                book_titles.update(generate_title_variants(str(candidate_title)))
        year = getattr(book, "publication_year", None)
        if year and not _is_nan(year):
            try:
                publication_years.add(int(year))
            except ValueError:
                pass
        publisher_value = getattr(book, "publisher", None)
        if publisher_value and not _is_nan(publisher_value):
            publisher_names.add(str(publisher_value).strip())
        for isbn_field in ("isbn13", "isbn"):
            value = getattr(book, isbn_field, None)
            if value and not _is_nan(value):
                isbn_values.add(re.sub(r"[^0-9Xx]", "", str(value)))

    title_variants.update(book_titles)
    title_variants = {variant for variant in title_variants if variant}

    author_ids = work_row.get("best_book_author_ids") or []
    author_names: List[str] = []
    for author_id in author_ids:
        name = author_name_map.get(str(author_id))
        if name:
            author_names.append(name)

    if not author_names:
        for book in associated_books:
            authors = getattr(book, "authors", None)
            if isinstance(authors, list):
                for author in authors:
                    author_id = author.get("author_id") if isinstance(author, dict) else None
                    if author_id:
                        name = author_name_map.get(str(author_id))
                        if name:
                            author_names.append(name)

    primary_title = next(iter(title_variants)) if title_variants else str(work_row.get("original_title") or "")
    primary_author = author_names[0] if author_names else ""

    return {
        "title_variants": title_variants or {primary_title},
        "primary_title": primary_title,
        "author_names": author_names,
        "primary_author": primary_author,
        "publication_years": sorted(publication_years),
        "isbn_values": [isbn for isbn in isbn_values if isbn],
        "publisher_names": [name for name in publisher_names if name],
    }


def score_work_candidate(
    context: Dict[str, Any],
    candidate: Dict[str, Any],
    entity: Dict[str, Any],
    entity_record: Dict[str, Any],
    client: WikidataClient,
) -> float:
    """Return a similarity score between a work context and a Wikidata entity."""
    candidate_titles = extract_candidate_titles(candidate, entity_record)
    if not candidate_titles:
        candidate_titles = {candidate.get("label", "")}

    base_score = 0.0
    for work_title in context["title_variants"]:
        for candidate_title in candidate_titles:
            base_score = max(base_score, string_similarity(work_title, candidate_title))

    if base_score < 35:
        return base_score

    score = base_score
    description = str(candidate.get("description") or "").lower()
    if any(keyword in description for keyword in ("novel", "book", "fiction", "literature")):
        score += 5

    author_score = 0.0
    candidate_author_ids = extract_claim_ids(entity, WIKIDATA_P_AUTHOR)
    candidate_author_names = [client.get_label(author_id) for author_id in candidate_author_ids[:5]]
    for our_author in context["author_names"]:
        for candidate_author in candidate_author_names:
            author_score = max(author_score, string_similarity(our_author, candidate_author))

    if author_score >= 85:
        score += 25
    elif author_score >= 70:
        score += 15
    elif author_score >= 60:
        score += 5
    elif context["author_names"] and not candidate_author_names:
        score -= 5

    entity_isbn_values: Set[str] = set()
    for property_id in (WIKIDATA_P_ISBN13, WIKIDATA_P_ISBN10):
        for value in WikidataClient.extract_statements(entity, property_id):
            entity_isbn_values.add(re.sub(r"[^0-9Xx]", "", str(value)))
    if entity_isbn_values and context["isbn_values"]:
        normalized_isbn = {isbn for isbn in context["isbn_values"] if isbn}
        if normalized_isbn & entity_isbn_values:
            score += 30

    candidate_publishers = extract_claim_ids(entity, WIKIDATA_P_PUBLISHER)
    candidate_publisher_names = [client.get_label(pub_id) for pub_id in candidate_publishers[:5]]
    publisher_score = 0.0
    for our_publisher in context["publisher_names"]:
        for candidate_publisher in candidate_publisher_names:
            publisher_score = max(publisher_score, string_similarity(our_publisher, candidate_publisher))
    if publisher_score >= 70:
        score += 10

    candidate_years: List[int] = []
    for value in WikidataClient.extract_statements(entity, WIKIDATA_P_PUBLICATION_DATE):
        _, year = parse_wikidata_time(value)
        if year:
            candidate_years.append(year)
    if candidate_years and context["publication_years"]:
        diff = min(abs(year - candidate_year) for year in context["publication_years"] for candidate_year in candidate_years)
        if diff <= 1:
            score += 5
        elif diff >= 10:
            score -= 5

    return score


def build_person_context(author_row: Dict[str, Any], authored_titles: List[str]) -> Dict[str, Any]:
    """Gather identifying signals (name variants, birth year, authored titles) for a person."""
    name = author_row.get("name", "") or ""
    name_variants = generate_title_variants(name)
    normalized_variants = {variant.lower() for variant in name_variants}
    birth_year = None
    if author_row.get("year_of_birth") and not _is_nan(author_row["year_of_birth"]):
        try:
            birth_year = int(author_row["year_of_birth"])
        except ValueError:
            birth_year = None
    return {
        "name": name,
        "name_variants": name_variants,
        "normalized_variants": normalized_variants,
        "birth_year": birth_year,
        "authored_titles": [title for title in authored_titles if title],
    }


def score_person_candidate(
    context: Dict[str, Any],
    candidate: Dict[str, Any],
    entity: Dict[str, Any],
    entity_record: Dict[str, Any],
    client: WikidataClient,
) -> float:
    """Score a candidate author entity against the local person context."""
    name_variants = context["name_variants"]
    candidate_titles = extract_candidate_titles(candidate, entity_record)
    base_score = 0.0
    for variant in name_variants:
        for candidate_title in candidate_titles or {candidate.get("label", "")}:
            base_score = max(base_score, string_similarity(variant, candidate_title))

    if base_score < 40:
        return base_score

    score = base_score
    description = str(candidate.get("description") or "").lower()
    if any(keyword in description for keyword in ("author", "writer", "novelist", "journalist", "poet")):
        score += 5

    occupations = [client.get_label(occ_id) for occ_id in extract_claim_ids(entity, WIKIDATA_P_OCCUPATION)[:8]]
    occ_bonus = 0.0
    for occupation in occupations:
        if not occupation:
            continue
        occupation_lower = occupation.lower()
        if any(term in occupation_lower for term in ("author", "writer", "novelist", "poet")):
            occ_bonus = max(occ_bonus, 10.0)
    score += occ_bonus

    birth_years: List[int] = []
    for value in WikidataClient.extract_statements(entity, WIKIDATA_P_DATE_OF_BIRTH):
        _, year = parse_wikidata_time(value)
        if year:
            birth_years.append(year)
    if birth_years and context.get("birth_year"):
        diff = min(abs(year - context["birth_year"]) for year in birth_years)
        if diff <= 2:
            score += 10
        elif diff > 15:
            score -= 5

    return score


def build_publisher_context(publisher_row: Dict[str, Any]) -> Dict[str, Any]:
    """Collect publisher name variants and country information for matching."""
    name = publisher_row.get("publisher_name", "") or ""
    name_variants = generate_title_variants(name)
    country_value = publisher_row.get("country")
    canonical = _standardize_country(country_value)
    return {
        "name": name,
        "name_variants": name_variants,
        "country": canonical[0] if canonical else "",
    }


def score_publisher_candidate(
    context: Dict[str, Any],
    candidate: Dict[str, Any],
    entity: Dict[str, Any],
    entity_record: Dict[str, Any],
    client: WikidataClient,
) -> float:
    """Score a candidate publisher entity using name and country similarity."""
    candidate_titles = extract_candidate_titles(candidate, entity_record)
    base_score = 0.0
    for variant in context["name_variants"]:
        for candidate_title in candidate_titles or {candidate.get("label", "")}:
            base_score = max(base_score, string_similarity(variant, candidate_title))

    if base_score < 35:
        return base_score

    score = base_score
    description = str(candidate.get("description") or "").lower()
    if any(keyword in description for keyword in ("publisher", "publishing", "imprint")):
        score += 5

    candidate_countries = [client.get_label(country_id) for country_id in extract_claim_ids(entity, WIKIDATA_P_PUBLISHER_COUNTRY)]
    if not candidate_countries:
        candidate_countries = [client.get_label(country_id) for country_id in WikidataClient.extract_statements(entity, WIKIDATA_P_COUNTRY)]
    if context["country"] and candidate_countries:
        country_score = max(string_similarity(context["country"], country) for country in candidate_countries)
        if country_score >= 70:
            score += 10

    return score


def search_candidates(client: WikidataClient, queries: Sequence[str], *, limit: int = 5) -> List[Dict[str, Any]]:
    """Search Wikidata with a set of queries and return merged candidate results."""
    seen: Set[str] = set()
    results: List[Dict[str, Any]] = []
    for query in queries:
        if not query:
            continue
        entries = client.search(query, limit=limit)
        for entry in entries:
            entity_id = entry.get("id")
            if not entity_id or entity_id in seen:
                continue
            entry_copy = dict(entry)
            entry_copy["_search_query"] = query
            results.append(entry_copy)
            seen.add(entity_id)
    return results


def get_entity_record(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a simplified entity record (labels/descriptions) from Wikidata JSON."""
    entities = entity.get("entities") or {}
    if entities:
        return next(iter(entities.values()))
    return {}


def extract_candidate_titles(candidate: Dict[str, Any], entity_record: Dict[str, Any]) -> Set[str]:
    """Return a set of candidate titles/aliases derived from search results."""
    titles: Set[str] = set()
    if candidate.get("label"):
        titles.update(generate_title_variants(str(candidate["label"])) )
    aliases = candidate.get("aliases")
    if aliases:
        for alias in aliases:
            if isinstance(alias, dict):
                alias_value = alias.get("value")
            else:
                alias_value = alias
            if alias_value:
                titles.update(generate_title_variants(str(alias_value)))
    for item in entity_record.get("labels", {}).values():
        alias_value = item.get("value") if isinstance(item, dict) else None
        if alias_value:
            titles.update(generate_title_variants(str(alias_value)))
    for alias_list in entity_record.get("aliases", {}).values():
        for alias in alias_list:
            alias_value = alias.get("value") if isinstance(alias, dict) else None
            if alias_value:
                titles.update(generate_title_variants(str(alias_value)))
    return {title for title in titles if title}


def extract_claim_ids(entity: Dict[str, Any], property_id: str) -> List[str]:
    """Pull raw claim IDs for the given Wikidata property."""
    entities = entity.get("entities") or {}
    ids: List[str] = []
    for data in entities.values():
        claims = data.get("claims", {})
        for claim in claims.get(property_id, []):
            datavalue = claim.get("mainsnak", {}).get("datavalue", {})
            value = datavalue.get("value")
            if isinstance(value, dict) and "id" in value:
                ids.append(value["id"])
    return ids


# ---------------------------------------------------------------------------
# Loading & exporting
# ---------------------------------------------------------------------------

def load_frames(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all expected normalized tables from disk."""
    frames: Dict[str, pd.DataFrame] = {}
    for name in [
        "work_final",
        "book_final",
        "series_final",
        "people_final",
        "publisher_final",
        "year_final",
        "country_final",
        "languages_final",
        "tags_final",
        "publisher_merge_mapping",
        "tag_merge_mapping",
    ]:
        frames[name] = _load_single_frame(input_dir, name)
    return frames


def _load_single_frame(input_dir: Path, name: str) -> pd.DataFrame:
    """Load a single normalized frame with list-column coercion."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    parquet_path = input_dir / f"{name}.parquet"
    csv_path = input_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if name in LIST_COLUMNS:
            for column in LIST_COLUMNS[name]:
                if column in df.columns:
                    df[column] = df[column].apply(_ensure_list)
        return df
    # Missing tables are acceptable (e.g. merge mapping may not exist yet)
    return pd.DataFrame()


def export_frames(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist enriched frames to Parquet/CSV outputs."""
    ensure_directory(output_dir)
    parquet_supported = True
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        path = output_dir / f"{name}.parquet"
        try:
            df.to_parquet(path, index=False)
        except (ImportError, ValueError, TypeError):
            parquet_supported = False
            df.to_csv(output_dir / f"{name}.csv", index=False)
    if not parquet_supported:
        print("Warning: pyarrow/fastparquet unavailable – exported CSV files instead.")


# ---------------------------------------------------------------------------
# Wikidata client
# ---------------------------------------------------------------------------

@dataclass
class WikidataClient:
    enabled: bool = field(init=False)
    cache_dir: Optional[Path] = None
    request_pause: float = 0.3
    search_pause: float = 0.1
    label_cache: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.enabled = requests is not None
        if not self.enabled:
            print("requests not available – Wikidata enrichment disabled.")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_cache = {}

    def search(self, query: str, limit: int = 5, language: str = DEFAULT_LANGUAGE) -> List[Dict[str, str]]:
        if not self.enabled or requests is None:
            return []
        http = requests
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "type": "item",
            "search": query,
            "limit": limit,
        }
        try:
            response = http.get(
                WIKIDATA_SEARCH_URL,
                params=params,
                headers={"User-Agent": WIKIDATA_AGENT},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            time.sleep(self.search_pause)
            return data.get("search", [])
        except Exception as exc:  # pragma: no cover - network call
            print(f"Wikidata search failed for '{query}': {exc}")
            return []

    def get_entity(self, entity_id: str) -> Dict:
        if not self.enabled or requests is None:
            return {}
        http = requests
        if self.cache_dir:
            cache_path = self.cache_dir / f"{entity_id}.json"
            if cache_path.exists():
                try:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        url = WIKIDATA_ENTITY_URL.format(entity_id=entity_id)
        try:
            response = http.get(url, headers={"User-Agent": WIKIDATA_AGENT}, timeout=20)
            response.raise_for_status()
            data = response.json()
            if self.cache_dir:
                cache_path = self.cache_dir / f"{entity_id}.json"
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(self.request_pause)
            return data
        except Exception as exc:  # pragma: no cover - network call
            print(f"Wikidata entity fetch failed for {entity_id}: {exc}")
            return {}

    @staticmethod
    def extract_statements(entity: Dict, property_id: str) -> List[str]:
        entities = entity.get("entities") or {}
        for data in entities.values():
            claims = data.get("claims", {})
            if property_id not in claims:
                continue
            values = []
            for claim in claims[property_id]:
                datavalue = claim.get("mainsnak", {}).get("datavalue", {})
                value = datavalue.get("value")
                if isinstance(value, dict):
                    if "id" in value:
                        values.append(value["id"])
                    elif "text" in value:
                        values.append(value["text"])
                    elif "time" in value:
                        values.append(value["time"])
                elif value:
                    values.append(str(value))
            return values
        return []

    def get_label(self, entity_id: str, language: str = DEFAULT_LANGUAGE) -> str:
        if not self.enabled or not entity_id or not entity_id.startswith("Q") or requests is None:
            return entity_id
        if entity_id in self.label_cache:
            return self.label_cache[entity_id]
        http = requests
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "labels",
            "languages": language,
        }
        try:
            response = http.get(
                WIKIDATA_SEARCH_URL,
                params=params,
                headers={"User-Agent": WIKIDATA_AGENT},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            label = data.get("entities", {}).get(entity_id, {}).get("labels", {}).get(language, {}).get("value")
            if label:
                self.label_cache[entity_id] = label
                return label
        except Exception:  # pragma: no cover - network call
            pass
        self.label_cache[entity_id] = entity_id
        return entity_id


# ---------------------------------------------------------------------------
# Country inference
# ---------------------------------------------------------------------------


def parse_wikidata_time(raw: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Parse a Wikidata time string into ISO date and integer year components."""
    if not raw or not isinstance(raw, str):
        return None, None
    match = re.match(r"^\+?(\d{4})(?:-(\d{2})-(\d{2}))?", raw)
    if not match:
        return None, None
    year, month, day = match.group(1), match.group(2), match.group(3)
    date_str = year
    if month:
        date_str += f"-{month}"
        if day:
            date_str += f"-{day}"
    return date_str, int(year)

def infer_book_and_work_countries(frames: Dict[str, pd.DataFrame]) -> None:
    """Populate inferred country codes/names for books and works when missing."""
    books = frames["book_final"].copy()
    works = frames["work_final"].copy()
    publishers = frames.get("publisher_final", pd.DataFrame())

    publisher_country: Dict[str, Tuple[str, str]] = {}
    if not publishers.empty:
        for row in publishers.itertuples(index=False):
            name_value = getattr(row, "publisher_name", None)
            country_value = getattr(row, "country", None)
            name = str(name_value).strip() if name_value and not _is_nan(name_value) else ""
            canonical = _standardize_country(country_value)
            if name and canonical:
                publisher_country[name.lower()] = canonical

    book_country_codes: List[str] = []
    book_country_names: List[str] = []
    book_inferred_codes: List[str] = []
    book_inferred_names: List[str] = []

    for row in books.itertuples(index=False):
        existing_value = getattr(row, "country_code", None)
        existing_canonical = _standardize_country(existing_value)
        existing_code = existing_name = ""
        if existing_canonical:
            existing_code, existing_name = existing_canonical

        publisher_value = getattr(row, "publisher", None)
        publisher_canonical = None
        if publisher_value and not _is_nan(publisher_value):
            publisher_canonical = publisher_country.get(str(publisher_value).strip().lower())

        if existing_code:
            final_code, final_name = existing_code, existing_name
        elif publisher_canonical:
            final_code, final_name = publisher_canonical
        else:
            raw_code = "" if existing_value is None or _is_nan(existing_value) else str(existing_value).strip()
            final_code, final_name = raw_code, ""

        if not final_name and publisher_canonical:
            final_name = publisher_canonical[1]

        book_country_codes.append(final_code)
        book_country_names.append(final_name)
        book_inferred_codes.append(final_code)
        book_inferred_names.append(final_name)

    books["country_code"] = book_country_codes
    books["country_name"] = book_country_names
    books["inferred_country_code"] = book_inferred_codes
    books["inferred_country_name"] = book_inferred_names

    books_by_work_codes: Dict[Any, List[str]] = collections.defaultdict(list)
    books_by_work_names: Dict[Any, List[str]] = collections.defaultdict(list)
    for row in books.itertuples(index=False):
        work_id = getattr(row, "work_id", None)
        if not work_id:
            continue
        for code in _split_multi_value(getattr(row, "inferred_country_code", "")):
            if code:
                books_by_work_codes[work_id].append(code)
        for name in _split_multi_value(getattr(row, "inferred_country_name", "")):
            if name:
                books_by_work_names[work_id].append(name)

    work_best_codes: List[str] = []
    work_best_names: List[str] = []
    work_inferred_codes: List[str] = []
    work_inferred_names: List[str] = []

    for row in works.itertuples(index=False):
        work_id = getattr(row, "work_id", None)
        best_value = getattr(row, "best_book_country_code", None)
        best_canonical = _standardize_country(best_value)
        best_code = best_name = ""
        if best_canonical:
            best_code, best_name = best_canonical

        wikidata_values = _split_multi_value(getattr(row, "wikidata_countries", None))
        wikidata_code = wikidata_name = ""
        if wikidata_values:
            wikidata_canonical = _standardize_country(wikidata_values[0])
            if wikidata_canonical:
                wikidata_code, wikidata_name = wikidata_canonical

        fallback_codes = books_by_work_codes.get(work_id, [])
        fallback_names = books_by_work_names.get(work_id, [])
        fallback_code = fallback_codes[0] if fallback_codes else ""
        fallback_name = fallback_names[0] if fallback_names else ""

        final_code = best_code or fallback_code or wikidata_code
        final_name = best_name or fallback_name or wikidata_name

        work_best_codes.append(best_code or fallback_code or wikidata_code)
        work_best_names.append(best_name or fallback_name or wikidata_name)
        work_inferred_codes.append(final_code)
        work_inferred_names.append(final_name)

    works["best_book_country_code"] = work_best_codes
    works["best_book_country_name"] = work_best_names
    works["inferred_country_code"] = work_inferred_codes
    works["inferred_country_name"] = work_inferred_names

    frames["book_final"] = books
    frames["work_final"] = works


def update_country_table(frames: Dict[str, pd.DataFrame]) -> None:
    """Refresh the distinct country table based on all known country values."""
    country_map: Dict[str, str] = {}

    def add_candidate(value: Any) -> None:
        normalized = _standardize_country(value)
        if not normalized:
            return
        code, name = normalized
        if not code:
            return
        existing = country_map.get(code)
        if not existing or len(name) > len(existing):
            country_map[code] = name

    country_df = frames.get("country_final")
    if country_df is not None and not country_df.empty:
        if "country_code" in country_df.columns:
            for value in country_df["country_code"].dropna():
                add_candidate(value)
        if "country_name" in country_df.columns:
            for value in country_df["country_name"].dropna():
                add_candidate(value)

    def collect_from_dataframe(df: Optional[pd.DataFrame], columns: Sequence[str]) -> None:
        if df is None or df.empty:
            return
        for column in columns:
            if column not in df.columns:
                continue
            for value in df[column].dropna():
                for entry in _split_multi_value(value):
                    add_candidate(entry)

    collect_from_dataframe(
        frames.get("book_final"),
        [
            "country_code",
            "country_name",
            "inferred_country_code",
            "inferred_country_name",
        ],
    )
    collect_from_dataframe(
        frames.get("work_final"),
        [
            "best_book_country_code",
            "best_book_country_name",
            "inferred_country_code",
            "inferred_country_name",
            "wikidata_countries",
        ],
    )
    collect_from_dataframe(
        frames.get("publisher_final"),
        ["country", "wikidata_country"],
    )
    collect_from_dataframe(
        frames.get("people_final"),
        [
            "citizenship",
            "wikidata_citizenship",
            "place_of_birth",
            "place_of_death",
            "wikidata_place_of_birth",
            "wikidata_place_of_death",
        ],
    )

    for extra in (
        frames.get("publisher_wikidata_enrichment"),
        frames.get("people_wikidata_enrichment"),
        frames.get("work_wikidata_enrichment"),
    ):
        collect_from_dataframe(extra, ["wikidata_country", "wikidata_countries"])

    if not country_map:
        frames["country_final"] = pd.DataFrame(columns=["country_code", "country_name"])
        return

    entries = [
        {"country_code": code, "country_name": name}
        for code, name in sorted(country_map.items(), key=lambda item: item[1])
        if code
    ]
    frames["country_final"] = pd.DataFrame(entries)


# ---------------------------------------------------------------------------
# Wikidata enrichment
# ---------------------------------------------------------------------------

def enrich_works_with_wikidata(
    frames: Dict[str, pd.DataFrame],
    client: WikidataClient,
    *,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Augment works with DDC/LCC, countries, and Wikidata IDs sourced from Wikidata."""
    if not client.enabled:
        return

    works = frames["work_final"]
    if works.empty:
        return

    books_df = frames.get("book_final", pd.DataFrame())
    people_df = frames.get("people_final", pd.DataFrame())

    books_by_work: Dict[Any, List[Any]] = collections.defaultdict(list)
    if not books_df.empty and "work_id" in books_df.columns:
        for book in books_df.itertuples(index=False):
            work_id = getattr(book, "work_id", None)
            if work_id:
                books_by_work[work_id].append(book)

    author_name_map: Dict[str, str] = {}
    if not people_df.empty and "author_id" in people_df.columns:
        for person in people_df.itertuples(index=False):
            author_id = getattr(person, "author_id", None)
            name = getattr(person, "name", None)
            if author_id and name:
                author_name_map[str(author_id)] = str(name)

    enriched_rows = []
    iterator = enumerate(works.itertuples(index=False), start=0)
    if verbose and tqdm:
        iterator = enumerate(
            tqdm(
                works.itertuples(index=False),
                total=len(works),
                desc="Wikidata: works",
                unit="work",
            ),
            start=0,
        )
    entity_cache: Dict[str, Dict[str, Any]] = {}
    for idx, row in iterator:
        if limit and idx >= limit:
            break
        row_dict = row._asdict() if hasattr(row, "_asdict") else row
        title = row_dict.get("original_title") or ""
        if not title:
            continue
        best_book_id = (row_dict.get("best_book_id") or "").strip()
        work_id = row_dict.get("work_id")

        context = build_work_context(row_dict, books_by_work.get(work_id, []), author_name_map)
        queries = []
        if context["primary_title"] and context["primary_author"]:
            queries.append(f"{context['primary_title']} {context['primary_author']}")
        if context["primary_title"] and context["publication_years"]:
            queries.append(f"{context['primary_title']} {context['publication_years'][0]}")
        queries.append(context["primary_title"])
        queries.extend(context["isbn_values"])
        queries = list(dict.fromkeys([query for query in queries if query]))

        candidates = search_candidates(client, queries, limit=6)
        if not candidates:
            continue

        best_score = -1.0
        best_entity: Optional[Dict[str, Any]] = None
        best_entity_record: Optional[Dict[str, Any]] = None
        best_candidate: Optional[Dict[str, Any]] = None

        for candidate in candidates[:6]:
            entity_id = candidate.get("id")
            if not entity_id:
                continue
            entity = entity_cache.get(entity_id)
            if entity is None:
                entity = client.get_entity(entity_id)
                entity_cache[entity_id] = entity
            if not entity:
                continue
            entity_record = get_entity_record(entity)
            score = score_work_candidate(context, candidate, entity, entity_record, client)
            if score > best_score:
                best_score = score
                best_entity = entity
                best_entity_record = entity_record
                best_candidate = candidate

        if not best_entity or best_entity_record is None or best_candidate is None or best_score < 45:
            continue
        best_candidate = cast(Dict[str, Any], best_candidate)

        ddc_codes = WikidataClient.extract_statements(best_entity, WIKIDATA_P_DEWEY)
        lcc_codes = WikidataClient.extract_statements(best_entity, WIKIDATA_P_LCC)
        countries = [
            client.get_label(code) if code else ""
            for code in WikidataClient.extract_statements(best_entity, WIKIDATA_P_COUNTRY)
        ]
        countries = [country for country in countries if country]

        enriched_rows.append(
            {
                "work_id": work_id,
                "best_book_id": best_book_id,
                "wikidata_id": best_candidate.get("id"),
                "wikidata_ddc": ";".join(sorted(set(ddc_codes))),
                "wikidata_lcc": ";".join(sorted(set(lcc_codes))),
                "wikidata_countries": ";".join(sorted(set(countries))),
                "match_score": best_score,
            }
        )

    if not enriched_rows:
        return

    enrichment_df = pd.DataFrame(enriched_rows)
    works = works.merge(enrichment_df, on="work_id", how="left", suffixes=("", "_enriched"))
    works["ddc"] = works.apply(
        lambda row: row["ddc"] if row.get("ddc") else row.get("wikidata_ddc"), axis=1
    )
    works["lcc"] = works.apply(
        lambda row: row["lcc"] if row.get("lcc") else row.get("wikidata_lcc"), axis=1
    )
    if "wikidata_id_enriched" in works.columns:
        works["wikidata_id"] = works["wikidata_id"].fillna(works["wikidata_id_enriched"])
    if "wikidata_countries_enriched" in works.columns:
        works["wikidata_countries"] = works["wikidata_countries"].fillna(
            works["wikidata_countries_enriched"]
        )
    if "inferred_country_code" in works.columns:
        def _fill_country_fields(row: pd.Series) -> Tuple[str, str]:
            existing_code = str(row.get("inferred_country_code") or "").strip()
            existing_name = str(row.get("inferred_country_name") or "").strip()
            if existing_code:
                return existing_code, existing_name
            for candidate in _split_multi_value(row.get("wikidata_countries")):
                canonical = _standardize_country(candidate)
                if canonical:
                    return canonical
            return "", ""

        updated = works.apply(lambda row: pd.Series(_fill_country_fields(row)), axis=1)
        works["inferred_country_code"] = updated[0]
        works["inferred_country_name"] = updated[1]
    works = works.drop(columns=[col for col in works.columns if col.endswith("_enriched")])
    wikidata_id_col = works.get("wikidata_id")
    if wikidata_id_col is not None:
        works["wikidata_enriched"] = wikidata_id_col.notna()
    else:
        works["wikidata_enriched"] = False
    frames["work_final"] = works
    frames["work_wikidata_enrichment"] = enrichment_df


def enrich_people_with_wikidata(
    frames: Dict[str, pd.DataFrame],
    client: WikidataClient,
    *,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Enrich author records with Wikidata attributes (birth data, citizenship, etc.)."""
    if not client.enabled:
        return

    people = frames["people_final"]
    if people.empty:
        return

    books_df = frames.get("book_final", pd.DataFrame())
    titles_by_author: Dict[str, List[str]] = collections.defaultdict(list)
    if not books_df.empty and "authors" in books_df.columns:
        for book in books_df.itertuples(index=False):
            authors = getattr(book, "authors", None)
            if isinstance(authors, list):
                for author in authors:
                    author_id = author.get("author_id") if isinstance(author, dict) else None
                    if author_id:
                        title = getattr(book, "title", None) or getattr(book, "title_without_series", None)
                        if title:
                            titles_by_author[str(author_id)].append(str(title))

    enriched_rows = []
    iterator = enumerate(people.itertuples(index=False), start=0)
    if verbose and tqdm:
        iterator = enumerate(
            tqdm(
                people.itertuples(index=False),
                total=len(people),
                desc="Wikidata: people",
                unit="person",
            ),
            start=0,
        )
    entity_cache: Dict[str, Dict[str, Any]] = {}
    for idx, row in iterator:
        if limit and idx >= limit:
            break
        row_dict = row._asdict() if hasattr(row, "_asdict") else row
        if row_dict.get("author_wikidata_id"):
            continue
        name = row_dict.get("name") or ""
        if not name:
            continue
        context = build_person_context(row_dict, titles_by_author.get(str(row_dict.get("author_id")), []))
        queries = [f"{name} author", name, f"{name} writer"]
        queries = list(dict.fromkeys(filter(None, queries)))
        candidates = search_candidates(client, queries, limit=6)
        if not candidates:
            continue
        best_score = -1.0
        best_entity: Optional[Dict[str, Any]] = None
        best_entity_record: Optional[Dict[str, Any]] = None
        best_candidate: Optional[Dict[str, Any]] = None

        for candidate in candidates[:6]:
            entity_id = candidate.get("id")
            if not entity_id:
                continue
            entity = entity_cache.get(entity_id)
            if entity is None:
                entity = client.get_entity(entity_id)
                entity_cache[entity_id] = entity
            if not entity:
                continue
            entity_record = get_entity_record(entity)
            score = score_person_candidate(context, candidate, entity, entity_record, client)
            if score > best_score:
                best_score = score
                best_entity = entity
                best_entity_record = entity_record
                best_candidate = candidate

        if not best_entity or best_entity_record is None or best_candidate is None or best_score < 45:
            continue
        best_candidate = cast(Dict[str, Any], best_candidate)

        entity = best_entity
        birth_places = [
            client.get_label(place) if place else ""
            for place in WikidataClient.extract_statements(entity, WIKIDATA_P_PLACE_OF_BIRTH)
        ]
        death_places = [
            client.get_label(place) if place else ""
            for place in WikidataClient.extract_statements(entity, WIKIDATA_P_PLACE_OF_DEATH)
        ]
        genders = [
            client.get_label(gender) if gender else ""
            for gender in WikidataClient.extract_statements(entity, WIKIDATA_P_GENDER)
        ]
        citizenships = [
            client.get_label(citizen) if citizen else ""
            for citizen in WikidataClient.extract_statements(entity, WIKIDATA_P_CITIZENSHIP)
        ]
        birth_date_raw = WikidataClient.extract_statements(entity, WIKIDATA_P_DATE_OF_BIRTH)
        death_date_raw = WikidataClient.extract_statements(entity, WIKIDATA_P_DATE_OF_DEATH)
        birth_date, birth_year = parse_wikidata_time(birth_date_raw[0] if birth_date_raw else None)
        death_date, death_year = parse_wikidata_time(death_date_raw[0] if death_date_raw else None)
        age = None
        if birth_year and death_year:
            age = death_year - birth_year

        enriched_rows.append(
            {
                "author_id": row_dict.get("author_id"),
                "author_wikidata_id": best_candidate.get("id"),
                "wikidata_place_of_birth": ";".join(sorted(set(birth_places))),
                "wikidata_place_of_death": ";".join(sorted(set(death_places))),
                "wikidata_gender": ";".join(sorted(set(genders))),
                "wikidata_citizenship": ";".join(sorted(set(citizenships))),
                "wikidata_date_of_birth": birth_date,
                "wikidata_date_of_death": death_date,
                "wikidata_year_of_birth": birth_year,
                "wikidata_year_of_death": death_year,
                "wikidata_age": age,
                "match_score": best_score,
            }
        )

    if not enriched_rows:
        return

    enrichment_df = pd.DataFrame(enriched_rows)
    people = people.merge(enrichment_df, on="author_id", how="left", suffixes=("", "_enriched"))
    fill_map = {
        "author_wikidata_id": "author_wikidata_id_enriched",
        "place_of_birth": "wikidata_place_of_birth",
        "place_of_death": "wikidata_place_of_death",
        "gender": "wikidata_gender",
        "citizenship": "wikidata_citizenship",
        "date_of_birth": "wikidata_date_of_birth",
        "date_of_death": "wikidata_date_of_death",
        "year_of_birth": "wikidata_year_of_birth",
        "year_of_death": "wikidata_year_of_death",
        "age": "wikidata_age",
    }
    for target_column, enriched_column in fill_map.items():
        if enriched_column not in people.columns:
            continue
        if target_column in {"year_of_birth", "year_of_death", "age"}:
            people[target_column] = pd.to_numeric(people[target_column], errors="coerce")
            enriched_numeric = pd.to_numeric(people[enriched_column], errors="coerce")
            people[target_column] = people[target_column].fillna(enriched_numeric)
        else:
            people[target_column] = people[target_column].astype("object")
            people[target_column] = people[target_column].fillna(people[enriched_column].astype("object"))

    drop_columns = [col for col in people.columns if col.endswith("_enriched")]
    if drop_columns:
        people = people.drop(columns=drop_columns)
    wikidata_id_col = people.get("author_wikidata_id")
    if wikidata_id_col is not None:
        people["wikidata_enriched"] = wikidata_id_col.notna()
    else:
        people["wikidata_enriched"] = False
    frames["people_final"] = people
    frames["people_wikidata_enrichment"] = enrichment_df


# ---------------------------------------------------------------------------
# Publisher enrichment
# ---------------------------------------------------------------------------

def enrich_publishers_with_wikidata(
    frames: Dict[str, pd.DataFrame],
    client: WikidataClient,
    *,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Enrich publisher records with Wikidata IDs, countries, and inception data."""
    if not client.enabled:
        return

    publishers = frames.get("publisher_final")
    if publishers is None or publishers.empty:
        return

    enriched_rows = []
    iterator = enumerate(publishers.itertuples(index=False), start=0)
    if verbose and tqdm:
        iterator = enumerate(
            tqdm(
                publishers.itertuples(index=False),
                total=len(publishers),
                desc="Wikidata: publishers",
                unit="publisher",
            ),
            start=0,
        )
    entity_cache: Dict[str, Dict[str, Any]] = {}
    for idx, row in iterator:
        if limit and idx >= limit:
            break
        row_dict = row._asdict() if hasattr(row, "_asdict") else row
        if row_dict.get("publisher_wikidata_id"):
            continue
        name = (row_dict.get("publisher_name") or "").strip()
        if not name:
            continue

        context = build_publisher_context(row_dict)
        queries = [f"{name} publisher", name]
        queries = list(dict.fromkeys(filter(None, queries)))
        candidates = search_candidates(client, queries, limit=6)
        if not candidates:
            continue

        best_score = -1.0
        best_entity: Optional[Dict[str, Any]] = None
        best_entity_record: Optional[Dict[str, Any]] = None
        best_candidate: Optional[Dict[str, Any]] = None

        for candidate in candidates[:6]:
            entity_id = candidate.get("id")
            if not entity_id:
                continue
            entity = entity_cache.get(entity_id)
            if entity is None:
                entity = client.get_entity(entity_id)
                entity_cache[entity_id] = entity
            if not entity:
                continue
            entity_record = get_entity_record(entity)
            score = score_publisher_candidate(context, candidate, entity, entity_record, client)
            if score > best_score:
                best_score = score
                best_entity = entity
                best_entity_record = entity_record
                best_candidate = candidate

        if not best_entity or best_entity_record is None or best_candidate is None or best_score < 45:
            continue
        best_candidate = cast(Dict[str, Any], best_candidate)

        entity = best_entity
        countries = [
            client.get_label(code) if code else ""
            for code in WikidataClient.extract_statements(entity, WIKIDATA_P_PUBLISHER_COUNTRY)
        ]
        if not countries:
            countries = [
                client.get_label(code) if code else ""
                for code in WikidataClient.extract_statements(entity, WIKIDATA_P_COUNTRY)
            ]
        countries = [country for country in countries if country]
        inception_raw = WikidataClient.extract_statements(entity, WIKIDATA_P_INCEPTION)
        inception_date, inception_year = parse_wikidata_time(inception_raw[0] if inception_raw else None)

        enriched_rows.append(
            {
                "publisher_id": row_dict.get("publisher_id"),
                "publisher_wikidata_id": best_candidate.get("id"),
                "wikidata_country": ";".join(sorted(set(countries))),
                "wikidata_year_established": inception_year,
                "wikidata_inception_date": inception_date,
                "match_score": best_score,
            }
        )

    if not enriched_rows:
        return

    enrichment_df = pd.DataFrame(enriched_rows)
    publishers = publishers.merge(enrichment_df, on="publisher_id", how="left", suffixes=("", "_enriched"))

    if "publisher_wikidata_id_enriched" in publishers.columns:
        publishers["publisher_wikidata_id"] = publishers["publisher_wikidata_id"].fillna(
            publishers["publisher_wikidata_id_enriched"]
        )
    if "country" in publishers.columns and "wikidata_country" in publishers.columns:
        publishers["country"] = publishers["country"].fillna(publishers["wikidata_country"])
    if "year_established" in publishers.columns and "wikidata_year_established" in publishers.columns:
        publishers["wikidata_year_established"] = pd.to_numeric(
            publishers["wikidata_year_established"], errors="coerce"
        )
        publishers["year_established"] = pd.to_numeric(
            publishers["year_established"], errors="coerce"
        )
        publishers["year_established"] = publishers["year_established"].fillna(
            publishers["wikidata_year_established"]
        )

    drop_columns = [col for col in publishers.columns if col.endswith("_enriched")]
    if drop_columns:
        publishers = publishers.drop(columns=drop_columns)
    wikidata_id_col = publishers.get("publisher_wikidata_id")
    if wikidata_id_col is not None:
        publishers["wikidata_enriched"] = wikidata_id_col.notna()
    else:
        publishers["wikidata_enriched"] = False
    frames["publisher_final"] = publishers
    frames["publisher_wikidata_enrichment"] = enrichment_df


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class KeywordExtractor:
    """Keyword extraction helper supporting TF-IDF, RAKE, or a hybrid of both."""

    def __init__(self, method: str = "hybrid", max_features: int = 20000):
        self.method = method
        self.max_features = max_features
        self.vectorizer: Optional[Any] = None
        self.stopwords_set = set(STOPWORDS)
        self.stopwords_list = list(STOPWORDS_LIST)
        if method in {"tfidf", "hybrid"} and TfidfVectorizer is None:
            print("scikit-learn not available – falling back to RAKE.")
            self.method = "rake"

    def fit(self, documents: Iterable[str]) -> None:
        """Fit the TF-IDF vectorizer when TF-IDF or hybrid extraction is requested."""
        if self.method in {"tfidf", "hybrid"} and TfidfVectorizer:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=self.stopwords_list,
                ngram_range=(1, 2),
                lowercase=True,
            )
            self.vectorizer.fit(list(documents))

    def extract(self, text: Optional[str], top_k: int = 10) -> List[str]:
        """Extract up to `top_k` keywords from the provided text."""
        if not text:
            return []
        candidates = []
        if self.method in {"tfidf", "hybrid"} and self.vectorizer:
            candidates.extend(self._extract_tfidf(text, top_k))
        if self.method in {"rake", "hybrid"}:
            candidates.extend(self._extract_rake(text, top_k))
        if not candidates:
            return []
        # deduplicate preserving order
        seen = set()
        ordered = []
        for candidate in candidates:
            normalized = _normalize_token(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(candidate.strip())
            if len(ordered) >= top_k:
                break
        return ordered

    def _extract_tfidf(self, text: str, top_k: int) -> List[str]:
        """Return the top TF-IDF terms for a single document."""
        if not self.vectorizer:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        response = self.vectorizer.transform([text])
        scores = zip(feature_names, response.toarray().flatten())
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        return [term for term, score in ranked[:top_k] if score > 0]

    def _extract_rake(self, text: str, top_k: int) -> List[str]:
        """Return keywords using a simple RAKE-inspired scoring strategy."""
        tokens = re.split(r"[^\w']+", text.lower())
        phrases = []
        current = []
        for token in tokens:
            if not token or token in self.stopwords_set:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            phrases.append(current)
        frequency = collections.Counter()
        degree = collections.Counter()
        for phrase in phrases:
            unique_tokens = set(phrase)
            degree.update({token: len(unique_tokens) for token in unique_tokens})
            frequency.update(phrase)
        scores = {}
        for token in frequency:
            scores[token] = degree[token] / frequency[token]
        phrase_scores = []
        for phrase in phrases:
            score = sum(scores.get(token, 0.0) for token in phrase)
            phrase_scores.append((" ".join(phrase), score))
        ranked = sorted(phrase_scores, key=lambda item: item[1], reverse=True)
        return [phrase for phrase, _ in ranked[:top_k]]


def apply_keyword_extraction(frames: Dict[str, pd.DataFrame], method: str = "hybrid") -> None:
    """Generate keywords for works/books/series and store them in-place."""
    print(f"[Step] Keyword extraction using method: {method}")
    extractor = KeywordExtractor(method=method)
    for df_name in ("work_final", "book_final", "series_final"):
        df = frames[df_name]
        if "extracted_keywords" in df.columns:
            frames[df_name] = df.drop(columns=["extracted_keywords"])

    works = frames["work_final"]
    books = frames["book_final"]
    series = frames["series_final"]

    corpus = []
    for df in (works, books, series):
        if "best_book_description" in df.columns:
            corpus.extend(filter(None, df["best_book_description"].tolist()))
        if "description" in df.columns:
            corpus.extend(filter(None, df["description"].tolist()))
    extractor.fit(corpus)

    if not works.empty:
        works_keywords = []
        iterator = works.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(works), desc="Keywords: works", unit="work")
        for _, row in iterator:
            description = row.get("best_book_description") or ""
            keywords = extractor.extract(description)
            works_keywords.append(keywords)
        works["keywords"] = works_keywords
        frames["work_final"] = works

    if not books.empty:
        books_keywords = []
        iterator = books.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(books), desc="Keywords: books", unit="book")
        for _, row in iterator:
            description = row.get("description") or ""
            keywords = extractor.extract(description)
            books_keywords.append(keywords)
        books["keywords"] = books_keywords
        frames["book_final"] = books

    if not series.empty:
        series_keywords = []
        iterator = series.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(series), desc="Keywords: series", unit="series")
        for _, row in iterator:
            description = row.get("description") or ""
            keywords = extractor.extract(description)
            series_keywords.append(keywords)
        series["keywords"] = series_keywords
        frames["series_final"] = series

    keyword_rows = []
    for _, row in works.iterrows():
        work_id = row.get("work_id")
        for rank, keyword in enumerate(row.get("keywords") or [], start=1):
            keyword_rows.append(
                {"entity_type": "work", "entity_id": work_id, "keyword": keyword, "rank": rank}
            )
    for _, row in books.iterrows():
        book_id = row.get("book_id")
        for rank, keyword in enumerate(row.get("keywords") or [], start=1):
            keyword_rows.append(
                {"entity_type": "book", "entity_id": book_id, "keyword": keyword, "rank": rank}
            )
    for _, row in series.iterrows():
        series_id = row.get("series_id")
        for rank, keyword in enumerate(row.get("keywords") or [], start=1):
            keyword_rows.append(
                {"entity_type": "series", "entity_id": series_id, "keyword": keyword, "rank": rank}
            )
    frames["extracted_keywords"] = pd.DataFrame(keyword_rows)


# ---------------------------------------------------------------------------
# Named entity extraction & citation graph
# ---------------------------------------------------------------------------

class NamedEntityExtractor:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        if spacy:
            try:
                self.model = spacy.load("en_core_web_sm")
            except Exception:
                self.model = None
                print("spaCy model 'en_core_web_sm' not available – using regex-based NER.")

    def extract(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        if self.model:
            doc = self.model(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        return self._regex_entities(text)

    @staticmethod
    def _regex_entities(text: str) -> List[Tuple[str, str]]:
        pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
        matches = []
        for match in pattern.findall(text):
            if len(match) < 3:
                continue
            matches.append((match.strip(), "PROPN"))
        return matches


def assign_citations(frames: Dict[str, pd.DataFrame]) -> None:
    """Detect entity citations via NER and attach them to the relevant frames."""
    works = frames["work_final"]
    books = frames["book_final"]
    series = frames["series_final"]
    people = frames["people_final"]

    if works.empty and books.empty and series.empty:
        return

    extractor = NamedEntityExtractor()
    print("[Step] Assigning citations based on descriptions")

    def build_index() -> dict[str, list[dict[str, str]]]:
        index: dict[str, list[dict[str, str]]] = collections.defaultdict(list)

        def register(name: Optional[str], entity_type: str, entity_id: str) -> None:
            if not name:
                return
            normalized = name.strip().lower()
            if not normalized:
                return
            index[normalized].append({"type": entity_type, "id": entity_id})

        for _, row in works.iterrows():
            work_id = str(row.get("work_id") or "")
            if not work_id:
                continue
            register(row.get("original_title"), "work", work_id)
        for _, row in books.iterrows():
            book_id = str(row.get("book_id") or "")
            if not book_id:
                continue
            register(row.get("title"), "book", book_id)
            register(row.get("title_without_series"), "book", book_id)
        for _, row in series.iterrows():
            series_id = str(row.get("series_id") or "")
            if not series_id:
                continue
            register(row.get("title"), "series", series_id)
        for _, row in people.iterrows():
            author_id = str(row.get("author_id") or "")
            if not author_id:
                continue
            register(row.get("name"), "person", author_id)

        return index

    name_index = build_index()

    citation_records: list[dict[str, str]] = []

    def collect_citations(text: str, self_refs: Set[str], source_type: str, source_id: str) -> List[str]:
        citations: List[str] = []
        seen: Set[str] = set()
        for mention, _ in extractor.extract(text):
            normalized = mention.strip().lower()
            if not normalized:
                continue
            for target in name_index.get(normalized, []):
                key = f"{target['type']}:{target['id']}"
                if key in seen or key in self_refs:
                    continue
                citations.append(key)
                citation_records.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "target_type": target["type"],
                        "target_id": target["id"],
                        "mention": mention.strip(),
                    }
                )
                seen.add(key)
        return citations

    if not works.empty:
        work_citations = []
        iterator = works.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(works), desc="Citations: works", unit="work")
        for _, row in iterator:
            work_id = str(row.get("work_id") or "")
            description = row.get("best_book_description") or ""
            self_refs = {f"work:{work_id}"} if work_id else set()
            work_citations.append(collect_citations(description, self_refs, "work", work_id))
        works["citations"] = work_citations
        frames["work_final"] = works

    if not books.empty:
        book_citations = []
        iterator = books.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(books), desc="Citations: books", unit="book")
        for _, row in iterator:
            book_id = str(row.get("book_id") or "")
            description = row.get("description") or ""
            self_refs = {f"book:{book_id}"} if book_id else set()
            book_citations.append(collect_citations(description, self_refs, "book", book_id))
        books["citations"] = book_citations
        frames["book_final"] = books

    if not series.empty:
        series_citations = []
        iterator = series.iterrows()
        if tqdm:
            iterator = tqdm(iterator, total=len(series), desc="Citations: series", unit="series")
        for _, row in iterator:
            series_id = str(row.get("series_id") or "")
            description = row.get("description") or ""
            self_refs = {f"series:{series_id}"} if series_id else set()
            series_citations.append(collect_citations(description, self_refs, "series", series_id))
        series["citations"] = series_citations
        frames["series_final"] = series

    if citation_records:
        frames["citation_mentions"] = pd.DataFrame(citation_records)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def enrich_data(
    frames: Dict[str, pd.DataFrame],
    *,
    keyword_method: str,
    wikidata_limit: Optional[int],
    wikidata_cache: Optional[Path],
    enable_wikidata: bool,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run the full enrichment pipeline and return the updated frames."""
    client = WikidataClient(cache_dir=wikidata_cache if enable_wikidata else None)

    if enable_wikidata:
        print("[Step] Enriching publishers via Wikidata")
        enrich_publishers_with_wikidata(
            frames, client, limit=wikidata_limit, verbose=verbose
        )
        print("[Step] Enriching works via Wikidata")
        enrich_works_with_wikidata(
            frames, client, limit=wikidata_limit, verbose=verbose
        )
        print("[Step] Enriching people via Wikidata")
        enrich_people_with_wikidata(
            frames, client, limit=wikidata_limit, verbose=verbose
        )

    infer_book_and_work_countries(frames)

    apply_keyword_extraction(frames, method=keyword_method)
    assign_citations(frames)
    update_country_table(frames)

    for label, id_column in (
        ("work_final", "wikidata_id"),
        ("people_final", "author_wikidata_id"),
        ("publisher_final", "publisher_wikidata_id"),
    ):
        df = frames.get(label)
        if df is None or df.empty:
            continue
        updated_df = df.copy()
        if "wikidata_enriched" in updated_df.columns:
            updated_df["wikidata_enriched"] = updated_df["wikidata_enriched"].fillna(False)
        else:
            id_series = updated_df.get(id_column)
            if id_series is not None:
                updated_df["wikidata_enriched"] = id_series.notna()
            else:
                updated_df["wikidata_enriched"] = False
        frames[label] = updated_df

    return frames


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the enrichment step."""
    parser = argparse.ArgumentParser(description="Goodreads enrichment pipeline (Step 3).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing normalized tables from Step 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where enriched tables will be written.",
    )
    parser.add_argument(
        "--keyword-method",
        choices=["tfidf", "rake", "hybrid"],
        default=DEFAULT_KEYWORD_METHOD,
        help="Keyword extraction strategy for descriptions.",
    )
    parser.add_argument(
        "--wikidata-limit",
        type=int,
        default=None,
        help="Limit the number of entities to query from Wikidata (useful for testing).",
    )
    parser.add_argument(
        "--wikidata-cache",
        type=Path,
        default=DEFAULT_WIKIDATA_CACHE,
        help="Directory to store cached Wikidata responses.",
    )
    parser.add_argument(
        "--auto-wikidata",
        action="store_true",
        help="Enable Wikidata enrichment (requires network and requests library).",
    )
    parser.add_argument(
        "--no-wikidata",
        action="store_false",
        dest="auto_wikidata",
        help="Disable Wikidata enrichment even if enabled in config.yaml.",
    )
    parser.set_defaults(auto_wikidata=DEFAULT_AUTO_WIKIDATA)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Persist enriched data to --output-dir.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars for long-running steps (requires tqdm).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point that executes the enrichment workflow."""
    args = parse_args(argv)
    input_dir = resolve_cli_path(args.input_dir)
    output_dir = resolve_cli_path(args.output_dir)
    frames = load_frames(input_dir)
    wikidata_cache: Optional[Path] = None
    if args.auto_wikidata:
        wikidata_cache = ensure_directory(resolve_cli_path(args.wikidata_cache))

    enriched = enrich_data(
        frames,
        keyword_method=args.keyword_method,
        wikidata_limit=args.wikidata_limit,
        wikidata_cache=wikidata_cache,
        enable_wikidata=args.auto_wikidata,
        verbose=args.progress,
    )

    if args.export:
        export_frames(enriched, output_dir)

    print("Enrichment complete:")
    for name, df in enriched.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"  {name}: {len(df)} rows")
    if args.export:
        print(f"Enriched tables exported to {output_dir}")


if __name__ == "__main__":
    main()
