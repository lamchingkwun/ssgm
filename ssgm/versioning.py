from __future__ import annotations

from typing import Dict, List, Optional

from .models import MemoryRecord


class VersionIndex:
    def __init__(self) -> None:
        self.history: Dict[str, List[MemoryRecord]] = {}

    def append(self, record: MemoryRecord) -> None:
        self.history.setdefault(record.key, []).append(record)

    def versions(self, key: str) -> List[MemoryRecord]:
        return list(self.history.get(key, []))

    def latest(self, key: str) -> Optional[MemoryRecord]:
        versions = self.history.get(key, [])
        return versions[-1] if versions else None

    def rollback_target(self, key: str, version: int) -> Optional[MemoryRecord]:
        for record in self.history.get(key, []):
            if record.version == version:
                return record
        return None
