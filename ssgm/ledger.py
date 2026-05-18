from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from dataclasses import asdict, dataclass, replace
from typing import Any, List, Optional

from .models import MemoryEvent


@dataclass
class LedgerIntegrityReport:
    valid: bool
    event_count: int
    failed_index: Optional[int] = None
    reason: str = ""
    head_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "event_count": self.event_count,
            "failed_index": self.failed_index,
            "reason": self.reason,
            "head_hash": self.head_hash,
        }


class EvidenceLedger:
    GENESIS_HASH = "GENESIS"

    def __init__(self) -> None:
        self.events: List[MemoryEvent] = []

    def _coerce_event(self, event: Any) -> MemoryEvent:
        if isinstance(event, MemoryEvent):
            return replace(event, chain_index=None, prev_hash=None, event_hash=None)
        return MemoryEvent(
            op=getattr(event, "op", "write"),
            record=getattr(event, "record"),
            reason=getattr(event, "reason", ""),
            accepted=getattr(event, "accepted", True),
            mode=getattr(event, "mode", "vanilla"),
            details=dict(getattr(event, "details", {}) or {}),
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Decimal):
            integral = value.to_integral_value()
            return int(value) if value == integral else float(value)
        if isinstance(value, dict):
            return {key: EvidenceLedger._json_safe(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [EvidenceLedger._json_safe(inner) for inner in value]
        return value

    @staticmethod
    def _canonical_payload(event: MemoryEvent) -> str:
        payload = EvidenceLedger._json_safe(asdict(replace(event, event_hash=None)))
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def _hash_event(cls, event: MemoryEvent) -> str:
        return hashlib.sha256(cls._canonical_payload(event).encode("utf-8")).hexdigest()

    def append(self, event: MemoryEvent) -> None:
        base_event = self._coerce_event(event)
        prev_hash = self.events[-1].event_hash if self.events else self.GENESIS_HASH
        prepared = replace(
            base_event,
            chain_index=len(self.events),
            prev_hash=prev_hash,
            event_hash=None,
        )
        self.events.append(replace(prepared, event_hash=self._hash_event(prepared)))

    def head_hash(self) -> Optional[str]:
        return self.events[-1].event_hash if self.events else None

    def verify_chain(self) -> LedgerIntegrityReport:
        prev_hash = self.GENESIS_HASH
        for index, event in enumerate(self.events):
            if event.chain_index != index:
                return LedgerIntegrityReport(
                    valid=False,
                    event_count=len(self.events),
                    failed_index=index,
                    reason=f"chain_index mismatch at {index}",
                    head_hash=self.head_hash(),
                )
            if event.prev_hash != prev_hash:
                return LedgerIntegrityReport(
                    valid=False,
                    event_count=len(self.events),
                    failed_index=index,
                    reason=f"prev_hash mismatch at {index}",
                    head_hash=self.head_hash(),
                )
            expected_hash = self._hash_event(event)
            if event.event_hash != expected_hash:
                return LedgerIntegrityReport(
                    valid=False,
                    event_count=len(self.events),
                    failed_index=index,
                    reason=f"hash mismatch at {index}",
                    head_hash=self.head_hash(),
                )
            prev_hash = event.event_hash
        return LedgerIntegrityReport(
            valid=True,
            event_count=len(self.events),
            head_hash=self.head_hash(),
        )

    def all(self) -> List[MemoryEvent]:
        return list(self.events)

    def by_key(self, key: str) -> List[MemoryEvent]:
        return [e for e in self.events if e.record.key == key]

    def export(self):
        return [asdict(e) for e in self.events]
