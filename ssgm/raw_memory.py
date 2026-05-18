from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ssgm.models import MemoryRecord


@dataclass
class RawMemoryCandidate:
    text: str
    tenant_id: str
    source: str
    timestamp: int
    metadata: dict[str, Any] | None = None


def to_memory_record(
    raw: RawMemoryCandidate,
    inferred_key: str,
    predicted_provenance_ok: bool,
) -> MemoryRecord:
    return MemoryRecord(
        key=inferred_key,
        value=raw.text,
        tenant_id=raw.tenant_id,
        source=raw.source,
        timestamp=raw.timestamp,
        provenance_ok=predicted_provenance_ok,
        predicted_provenance_ok=predicted_provenance_ok,
    )
