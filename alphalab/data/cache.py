"""File-based cache for DataFrames and JSON data."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FileCache:
    """Simple file-based cache with TTL expiry.

    Stores DataFrames as parquet and dicts/lists as JSON.
    Cache keys are SHA-256 hashed to filesystem-safe names.
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _key_to_path(self, key: str, suffix: str = ".parquet") -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}{suffix}"

    def _is_expired(self, path: Path) -> bool:
        if not path.exists():
            return True
        return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) > self.ttl

    def get_df(self, key: str) -> pd.DataFrame | None:
        path = self._key_to_path(key, ".parquet")
        if self._is_expired(path):
            if path.exists():
                path.unlink()
            return None
        logger.debug("Cache hit (df): %s", key)
        return pd.read_parquet(path)

    def set_df(self, key: str, df: pd.DataFrame) -> None:
        path = self._key_to_path(key, ".parquet")
        df.to_parquet(path)

    def get_json(self, key: str) -> dict | list | None:
        path = self._key_to_path(key, ".json")
        if self._is_expired(path):
            if path.exists():
                path.unlink()
            return None
        logger.debug("Cache hit (json): %s", key)
        with open(path) as f:
            return json.load(f)

    def set_json(self, key: str, data: dict | list) -> None:
        path = self._key_to_path(key, ".json")
        with open(path, "w") as f:
            json.dump(data, f)

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.iterdir():
            if f.suffix in (".parquet", ".json"):
                f.unlink()
                count += 1
        return count
