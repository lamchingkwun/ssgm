"""
memory_query.py
==============
M_core Query Interface for SSGM.

Implements the episodic memory query interface described in the SSGM paper:
    M_core ← Query(episodic_ledger, query_key, time_window)

M_core is the set of "established facts" — immutable or high-confidence
mutable records that are accepted and within the reconciliation window.

These facts serve as the grounding context for:
  1. NLI-based contradiction detection (Level 3)
  2. Anchor-guided reconciliation
  3. Semantic drift measurement (Eq.4 in the paper)

Key design decisions:
  - Only ACCEPTED records count toward M_core
  - Immutable records always qualify as M_core
  - Mutable records qualify if: confidence >= min_confidence_threshold
  - Records older than time_window are excluded
  - No additional provenance filter is applied here beyond prior acceptance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import MemoryRecord


# ---------------------------------------------------------------------------
# Query config
# ---------------------------------------------------------------------------

@dataclass
class MCoreQueryConfig:
    """
    Configuration for M_core retrieval.

    Attributes:
        min_confidence: Minimum confidence for mutable records to qualify as M_core.
                        Default: 0.75
        max_age: Maximum age (in timestamp units) for mutable records to qualify.
                 None = no limit. Default: None
        include_mutable: Whether to include mutable records in M_core.
                         If False, only immutable records are returned.
                         Default: True
        min_version: Minimum version number to qualify. Default: 1
    """
    min_confidence: float = 0.75
    max_age: Optional[int] = None
    include_mutable: bool = True
    min_version: int = 1


# ---------------------------------------------------------------------------
# M_core query
# ---------------------------------------------------------------------------

class MCoreQuery:
    """
    Query interface for retrieving M_core established facts from an episodic ledger.

    Usage:
        query = MCoreQuery()
        facts = query.get_mcore(
            ledger=engine.ledger,
            key='alice:project_goal',
            tenant_id='alice',
            now_ts=10,
            config=MCoreQueryConfig(min_confidence=0.75),
        )
        # facts: List[MemoryRecord] of established facts
    """

    def get_mcore(
        self,
        ledger,
        key: str,
        tenant_id: str,
        now_ts: int = 0,
        config: Optional[MCoreQueryConfig] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve M_core facts for a given key from the episodic ledger.

        Args:
            ledger: EvidenceLedger instance
            key: Memory key to query
            tenant_id: Tenant ID for provenance filtering
            now_ts: Current timestamp (for age filtering)
            config: MCoreQueryConfig; uses defaults if None

        Returns:
            List of MemoryRecords that qualify as M_core for the given key,
            sorted by timestamp ascending (oldest first).
        """
        config = config or MCoreQueryConfig()

        candidates: List[MemoryRecord] = []
        for event in ledger.all():
            rec = event.record

            # Key and tenant must match
            if rec.key != key or rec.tenant_id != tenant_id:
                continue

            # Only accepted write events contribute to M_core
            if getattr(event, "op", "write") != "write" or not event.accepted:
                continue

            # Minimum version
            if rec.version < config.min_version:
                continue

            # Age filter
            if config.max_age is not None:
                age = now_ts - rec.timestamp
                if age > config.max_age:
                    continue

            # Immutable records always qualify
            if not rec.mutable:
                candidates.append(rec)
                continue

            # Mutable: apply confidence threshold
            if config.include_mutable and rec.confidence >= config.min_confidence:
                candidates.append(rec)

        candidates.sort(key=lambda r: r.timestamp)
        return candidates

    def get_all_mcore(
        self,
        ledger,
        tenant_id: str,
        now_ts: int = 0,
        config: Optional[MCoreQueryConfig] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve all M_core facts for a tenant (across all keys).

        Returns a list sorted by (key, timestamp).
        """
        config = config or MCoreQueryConfig()
        all_facts: List[MemoryRecord] = []
        seen_keys: set = set()

        for event in ledger.all():
            rec = event.record
            if rec.tenant_id != tenant_id:
                continue
            if getattr(event, "op", "write") != "write" or not event.accepted:
                continue
            if rec.key in seen_keys:
                continue  # already processed this key

            mcore = self.get_mcore(ledger, rec.key, tenant_id, now_ts, config)
            if mcore:
                all_facts.extend(mcore)
                seen_keys.add(rec.key)

        return all_facts

    def get_latest_mcore(
        self,
        ledger,
        key: str,
        tenant_id: str,
        now_ts: int = 0,
        config: Optional[MCoreQueryConfig] = None,
    ) -> Optional[MemoryRecord]:
        """
        Get only the most recent M_core fact for a key (oldest-to-newest).
        """
        facts = self.get_mcore(ledger, key, tenant_id, now_ts, config)
        return facts[-1] if facts else None
