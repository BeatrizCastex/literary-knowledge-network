from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    """Return the canonical path to config.yaml."""
    return repo_root() / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load the project configuration (cached)."""
    path = config_path()
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def config_section(name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a named section from the configuration."""
    cfg = load_config()
    return cfg.get(name, {}) if default is None else cfg.get(name, default)


def resolve_path(key: str) -> Path:
    """Resolve a configured path key under the repository root."""
    paths = config_section("paths")
    if key not in paths:
        raise KeyError(f"Path '{key}' not defined in config.yaml")
    return (repo_root() / Path(paths[key])).resolve()


def raw_file_path(key: str) -> Path:
    """Return the path to a raw Goodreads file declared in config.yaml."""
    raw_dir = resolve_path("raw_dir")
    raw_files = config_section("raw_files")
    if key not in raw_files:
        raise KeyError(f"Raw file '{key}' not defined in config.yaml")
    return (raw_dir / raw_files[key]).resolve()


def get_default(section: str, key: str, fallback: Any = None) -> Any:
    """Fetch a default value from the config, returning fallback when missing."""
    defaults = config_section("defaults")
    section_cfg = defaults.get(section, {})
    return section_cfg.get(key, fallback)


def ensure_directory(path: Path) -> Path:
    """Create the directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cli_path(path: Path) -> Path:
    """Resolve user-supplied CLI paths relative to the repository root."""
    return path if path.is_absolute() else (repo_root() / path).resolve()


def neo4j_credentials() -> Dict[str, Optional[str]]:
    """Retrieve Neo4j connection defaults, honoring environment variables."""
    neo4j_cfg = config_section("neo4j")
    uri = os.environ.get("NEO4J_URI", neo4j_cfg.get("uri"))
    user_env = neo4j_cfg.get("user_env", "NEO4J_USER")
    password_env = neo4j_cfg.get("password_env", "NEO4J_PASSWORD")
    database_env = neo4j_cfg.get("database_env", "NEO4J_DATABASE")
    return {
        "uri": uri,
        "user": os.environ.get(user_env, "neo4j"),
        "password": os.environ.get(password_env),
        "database": os.environ.get(database_env, "neo4j"),
    }
