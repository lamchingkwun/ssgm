from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AccessContext:
    actor_id: str
    tenant_id: str
    role: str = "user"
    now_ts: int = 0
    scopes: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    key: str
    value: str
    tenant_id: str
    source: str
    timestamp: int
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    mutable: bool = True
    # memory_class separates contradiction semantics from simple mutability.
    # - ordinary: normal mutable memory, may be updated automatically
    # - protected: important but changeable memory, conflicting updates should be conservative
    # - anchor: strongest facts / ledger-like anchors, should not be auto-overwritten
    memory_class: str = "ordinary"
    # conflict_policy refines how contradictions should be handled.
    # - auto_update: normal versioned update
    # - confirm_required: conflict should be surfaced conservatively (currently deferred)
    # - immutable: never auto-overwrite
    conflict_policy: str = "auto_update"
    version: int = 1
    parent_version: Optional[int] = None
    status: str = "active"
    provenance_ok: bool = True
    provenance_attested: Optional[bool] = None
    provenance_scope: Optional[str] = None
    predicted_provenance_ok: Optional[bool] = None
    quarantine_reason: Optional[str] = None
    access_scopes: List[str] = field(default_factory=list)
    access_relations: List[str] = field(default_factory=list)
    access_attributes: Dict[str, str] = field(default_factory=dict)
    allowed_roles: List[str] = field(default_factory=list)
    allowed_actors: List[str] = field(default_factory=list)

    @property
    def gold_provenance_ok(self) -> bool:
        return self.provenance_ok


@dataclass
class MemoryEvent:
    op: str
    record: MemoryRecord
    reason: str = ""
    accepted: bool = True
    mode: str = "vanilla"
    details: Dict[str, Any] = field(default_factory=dict)
    chain_index: Optional[int] = None
    prev_hash: Optional[str] = None
    event_hash: Optional[str] = None
