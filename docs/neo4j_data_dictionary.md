# LiteratureNetwork Neo4j Data Dictionary

This data dictionary describes the graph schema used for the enriched literature dataset after Steps 1–7 of the pipeline. It is intended as a reference for developers and analysts who need to query or extend the Neo4j database.

---

## Overview

The graph captures literary entities (works, books, series, people, publishers), contextual metadata (countries, languages, years, tags), and the relationships connecting them. Enrichment pulls in additional details from the processing pipeline (countries, keywords, citations, Wikidata attributes).

Each section below lists the node label or relationship type, the key properties available, and notes on their provenance.

---

## Node Labels

### `Work`
Represents a unique literary work (title-level). Built from Step 1 extraction and Step 3 enrichment.

**Primary key**: `id` (string; Goodreads work ID)

**Core properties**
- `original_title` – base title as published.
- `description` – synopsis populated from associated book descriptions.
- `original_language_id` – language code where available.
- `original_publication_year`, `original_publication_month`, `original_publication_day` – first publication date components.
- `best_book_id` – Goodreads best edition identifier.
- `best_book_country_code`, `best_book_country_name` – canonicalized country drawn from Step 3 enrichment.
- `inferred_country_code`, `inferred_country_name` – final country assignment after enrichment.
- `ddc`, `lcc` – Dewey and Library of Congress classifications (when retrieved from Wikidata).
- `wikidata_id` – matched QID if enrichment succeeded.
- `keywords` – list of extracted keywords from descriptions.
- `tags` – aggregated tags inherited from associated books.
- `match_score` (optional) – enrichment confidence score.
- `cluster_id`, `cluster_label` – community assignment from Step 7 (Leiden).
- `cluster_resolution` – resolution parameter used for the detected community.
- `cluster_size` – number of works in the community.

### `Book`
Represents an edition/manifestation tied to a specific work.

**Primary key**: `id` (string; Goodreads book ID)

**Core properties**
- `title`, `title_without_series`
- `description`, `image_url`, `goodreads_url`
- `work_id` – foreign key pointing to the Work node.
- `publication_year`, `publication_month`, `publication_day`
- `language_code`
- `country_code`, `inferred_country_code`, `inferred_country_name`
- `isbn`, `isbn13`
- `publisher_name`, `publisher_id` (if matched to Publisher node)
- `format`, `edition_information`
- `num_pages`

### `Series`
Represents a book series.

**Primary key**: `id` (string; Goodreads series ID)

**Core properties**
- `title`
- `description`
- `series_works_count`, `primary_work_count`
- `numbered` – whether entries are numbered
- `keywords` – series-level keywords derived from descriptions

### `Person`
Represents an author or contributor.

**Primary key**: `id` (string; Goodreads author ID)

**Core properties**
- `name`
- `average_rating`, `ratings_count`, `text_reviews_count`
- `wikidata_id`
- `date_of_birth`, `date_of_death`, `year_of_birth`, `year_of_death`, `age`
- `place_of_birth`, `place_of_death`
- `citizenship` – free-form strings for country/region
- `gender`

### `Publisher`
Represents publishing entities.

**Primary key**: `id` (synthetic ID created during extraction)`

**Core properties**
- `name`
- `wikidata_id`
- `country`
- `year_established`
- `wikidata_year_established`
- `wikidata_inception_date`

### `Tag`
Represents normalized thematic or shelf tags.

**Primary key**: `id` (synthetic ID)

**Core properties**
- `name`
- `tag_type` – e.g., `shelf`, `keyword`
- `occurrences` – raw occurrence count (only tags meeting the minimum occurrence threshold are materialized)

### `Country`
Represents countries associated with works/people/publishers.

**Primary key**: `code` (ISO-like alpha code)

**Core properties**
- `name`

### `Language`
Represents language codes encountered in the dataset.

**Primary key**: `code`

### `Year`
Represents years used for publication and lifecycle events.

**Primary key**: `value` (integer)

---

## Relationship Types

### `(:Book)-[:IS_EDITION]->(:Work)`
Links each book edition to its parent work.

### `(:Book)-[:PUBLISHED_IN]->(:Country)`
Country of publication (either direct from dataset or inferred during enrichment).

### `(:Book)-[:YEAR_PUBLISHED]->(:Year)`
Publication year.

### `(:Book)-[:PUBLISHED_BY]->(:Publisher)`
Publisher linkage when the name could be normalized and matched.

### `(:Work)-[:PART_OF]->(:Series)`
Associates works with series memberships.

### `(:Work)-[:HAS_TAG]->(:Tag)`
All tags/keywords aggregated for the work.

### `(:Work)-[:PUBLISHED_IN]->(:Country)`
Publishing country for the work (best edition, inferred, or Wikidata fallback).

### `(:Work)-[:FIRST_PUBLISHED]->(:Year)`
Original publication year.

### `(:Work)-[:CITED_IN]->(:Work)`
Citation-style references detected in descriptions.

### `(:Series)-[:HAS_TAG]->(:Tag)`
Series-level keywords mapped to tags.

### `(:Publisher)-[:LOCATED_IN]->(:Country)`
Canonical location for the publisher.

### `(:Publisher)-[:FOUNDED_IN]->(:Year)`
Year the publisher was established (when known).

### `(:Person)-[:WORKED_ON]->(:Work)`
Primary author/contributor relationships (derived from work metadata).

### `(:Person)-[:WORKED_ON]->(:Book)`
Edition-level involvement (roles from book author lists).

### `(:Person)-[:NAMED_IN]->(:Work)`
Person mentioned in another work’s description (NER-based references).

### `(:Person)-[:WORKED_ON]->(:Series)`
Propagates contributor roles to series based on the works that belong to them.

### `(:Person)-[:WAS_BORN]->(:Year)` / `(:Person)-[:WAS_DECEASED]->(:Year)`
Lifecycle events mapped to year nodes.

### `(:Person)-[:BORN_IN]->(:Country)`
Place of birth (when available).

### `(:Person)-[:HAS_CITIZENSHIP]->(:Country)`
Citizenship affiliations.

### `(:Person)-[:HAS_TAG]->(:Tag)`
Available for future enrichment; currently unpopulated pending reliable person-level tagging.

### `(:Person)-[:WORKS_FOR]->(:Publisher)`
Publisher affiliations inferred from edition metadata (role annotations).

### `(:Person)-[:IS_RELATED]->(:Person)`
Reserved for future enrichment where relationship metadata exists.

### `(:Person)-[:LIVES_IN]->(:Country)`
Placeholder relation for future enrichment if residency data becomes available.

### `(:Work)-[:SIMILAR_TO]->(:Work)`
Weighted similarity edges computed from SBERT embeddings (property `score` holds cosine similarity). Edges are undirected conceptually; the loader stores a single directed relationship per pair.

---

## Constraints & Indexes

The loader creates uniqueness constraints (if enabled) on primary ID fields for each node label. This enforces stable merging during repeated ingestion.

---

## Notes for Consumers

* Some relationships are optional or sparsely populated depending on enrichment success (e.g., `CITED_IN`, `HAS_TAG` on series).
* Match scores recorded in the `*_wikidata_enrichment` tables can be joined back to nodes via `match_score` to gauge confidence.
* Country and language codes are normalized but may not be full ISO sets when source data is limited.
* Future enrichment steps (e.g., similarity graph or person relationships) can leverage existing placeholders (`SIMILAR_TO`, `WORKS_FOR`, `IS_RELATED`, `HAS_TAG`).

---

For questions or updates to this schema, please email me at bccastex@gmail.com
