from __future__ import annotations
from typing import Any
from pathlib import Path

from .Dataset import Dataset


class SimpleDataset(Dataset):
    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "SimpleDataset":
        return super().from_json(obj)

    @classmethod
    def from_path(cls, path: Path) -> "SimpleDataset":
        return super().from_path(path)