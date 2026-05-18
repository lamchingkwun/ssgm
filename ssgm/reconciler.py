"""
reconciler.py
=============
Anchor-based batch reconciler for SSGM.

Implements the paper's operational repair rule:
    M_clean(k) = R_k(V_k; Π)

via key-local replay over accepted version chains.

The key insight is that each reconciliation window bounded by N steps can keep
drift window-bounded rather than horizon-bounded. The reconciler therefore
repairs active state by replaying accepted writes against immutable anchors or,
when no anchor exists, against a lexicographic fallback priority.

Algorithm:
  1. For each active key, build its accepted version chain from the ledger.
  2. Find the immutable anchor at the root of the chain. In the current
     implementation, anchored semantics are defined primarily by
     `memory_class="anchor"` or `conflict_policy="immutable"`. The older
     `mutable=False` flag is retained only as a compatibility fallback for
     single-record chains.
     If no anchor exists, use the highest-priority accepted record, with
     recency preferred over source trust among otherwise comparable records so
     accepted benign revisions are not silently undone.
  3. Replay post-anchor mutable writes in timestamp order.
  4. If the replayed value differs from the active store value, update the
     active store to the repaired version.

Unlike the naive "pick best by priority" approach, this:
  - Explicitly respects anchor/protected semantics instead of relying only on
    the older mutability flag
  - Replays the transformation lineage rather than just snapshot voting
  - Handles lower-confidence or invalid writes by skipping them during replay
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from .models import MemoryRecord
from .store import SemanticStore
from .ledger import EvidenceLedger


# ---------------------------------------------------------------------------
# Reconciliation result
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    key: str
    action: str          # 'anchored' | 'fallback' | 'no_change' | 'rolled_back'
    anchor_version: Optional[int]
    previous_value: str
    reconciled_value: str
    num_writes_replayed: int
    num_skipped: int      # writes skipped due to INSUFFICIENT_EVIDENCE
    num_rolled_back: int  # immutable anchors restored


# ---------------------------------------------------------------------------
# BatchReconciler
# ---------------------------------------------------------------------------

class BatchReconciler:
    """
    Anchor-grounded batch reconciler.

    Key methods:
      run(): reconcile the entire mutable store against the immutable ledger.
      reconcile_key(): reconcile a single key's version chain.
    """

    TRUSTED_SOURCES = {
        "hr": 4,
        "hr_update": 4,
        "calendar": 4,
        "tool_result": 4,
        "doc": 3,
        "chat": 3,
        "manual_correction": 4,
        "summary": 1,
        "reflection": 1,
        "poisoned_copy": 0,
    }

    def __init__(self) -> None:
        self.last_run_seconds: float = 0.0
        self.results: List[ReconciliationResult] = []

    @staticmethod
    def _is_explicit_anchor(record: MemoryRecord) -> bool:
        return (
            getattr(record, "memory_class", "ordinary") == "anchor"
            or getattr(record, "conflict_policy", "auto_update") == "immutable"
        )

    # ------------------------------------------------------------------
    # Priority for fallback (when no immutable anchor exists)
    # ------------------------------------------------------------------

    @staticmethod
    def _priority(record: MemoryRecord) -> Tuple[int, float, int, int]:
        """
        Priority for picking the best record when no immutable anchor exists.
        (provenance_ok, confidence, timestamp, source_trust)

        This ordering keeps replay aligned with the write gate: when no explicit
        anchor exists, a newer accepted revision with comparable confidence
        should dominate an older ordinary record instead of being rolled back
        only because the earlier source looked more trusted.
        """
        provenance_ok = 1 if record.provenance_ok else 0
        trust = BatchReconciler.TRUSTED_SOURCES.get(record.source, 2)
        return (provenance_ok, record.confidence, record.timestamp, trust)

    # ------------------------------------------------------------------
    # Build version chain
    # ------------------------------------------------------------------

    def _build_version_chain(
        self, key: str, ledger: EvidenceLedger
    ) -> List[MemoryRecord]:
        """
        Build the ordered version chain for a key from the ledger.
        Follows parent_version links back to the root.
        Returns records in chronological order (oldest → newest).
        """
        # Collect all accepted writes for this key
        candidates = [
            event.record
            for event in ledger.all()
            if getattr(event, "op", "write") == "write" and event.accepted and event.record.key == key
        ]
        candidates.sort(key=lambda r: (r.version, r.timestamp))

        # Build version → record map
        version_map: Dict[int, MemoryRecord] = {}
        for rec in candidates:
            version_map[rec.version] = rec

        # Follow chain from latest back to root
        if not candidates:
            return []

        latest = max(candidates, key=lambda r: r.version)
        chain = []
        current_version = latest.version

        visited = set()
        while current_version is not None:
            if current_version in visited:
                break  # cycle guard
            visited.add(current_version)
            rec = version_map.get(current_version)
            if rec is None:
                break
            chain.append(rec)
            current_version = rec.parent_version

        chain.reverse()  # oldest first
        return chain

    # ------------------------------------------------------------------
    # Find immutable anchor
    # ------------------------------------------------------------------

    def _find_anchor(
        self, chain: List[MemoryRecord]
    ) -> Optional[MemoryRecord]:
        """
        Find the root anchor in a version chain.

        Priority:
          1. Explicit anchor semantics (`memory_class="anchor"` or
             `conflict_policy="immutable"`).
          2. Compatibility fallback: a single-record chain that still relies on
             the older `mutable=False` convention.

        Ordinary legacy records with `mutable=False` but without explicit
        anchor policy are no longer treated as immutable anchors once a newer
        accepted descendant exists. This keeps reconciliation aligned with the
        write gate, which allows benign time-qualified revision for that legacy
        compatibility case.
        """
        for rec in chain:
            if self._is_explicit_anchor(rec):
                return rec
        if len(chain) == 1 and not chain[0].mutable:
            return chain[0]
        return None

    # ------------------------------------------------------------------
    # Replay writes from anchor
    # ------------------------------------------------------------------

    def _replay_from_anchor(
        self,
        chain: List[MemoryRecord],
        anchor: Optional[MemoryRecord],
    ) -> Tuple[Optional[MemoryRecord], int, int, int]:
        """
        Replay the version chain starting from the immutable anchor.

        Returns:
            (reconciled_record, num_replayed, num_skipped, num_rolled_back)

        Rules during replay:
          - Explicit anchor: always the base (never overwritten)
          - Mutable with HIGHER confidence than anchor after anchor's time:
              → apply as TIME_QUALIFIED_UPDATE
          - Mutable with LOWER confidence AND older:
              → INSUFFICIENT_EVIDENCE, skip
          - Duplicate value → no-op
        """
        num_replayed = 0
        num_skipped = 0
        num_rolled_back = 0

        if not chain:
            return None, 0, 0, 0

        if anchor is None:
            # No anchor → use highest-priority record as base
            fallback = max(chain, key=self._priority)
            num_rolled_back = 1
            return fallback, 0, 0, 1

        # Start from the anchor
        current = anchor

        for rec in chain:
            if rec.version == anchor.version:
                continue  # skip anchor itself

            # Determine if this write should apply
            if rec.timestamp <= anchor.timestamp:
                # Older than or equal to anchor → skip
                num_skipped += 1
                continue

            # Time-qualified update check
            if rec.value == current.value:
                # Duplicate → no-op
                num_replayed += 1
                continue

            if self._is_explicit_anchor(rec):
                # Explicit anchor/policy write after the root anchor should not
                # replace the current replay state. Preserve the earlier anchor.
                # This should not happen if write gate worked correctly,
                # but if it does, roll back to anchor
                num_rolled_back += 1
                continue

            # Confidence comparison with anchor
            if rec.confidence >= current.confidence:
                # Higher or equal confidence → apply update
                current = rec
                num_replayed += 1
            else:
                # Lower confidence → INSUFFICIENT_EVIDENCE, skip
                num_skipped += 1

        return current, num_replayed, num_skipped, num_rolled_back

    # ------------------------------------------------------------------
    # Per-key reconciliation
    # ------------------------------------------------------------------

    def reconcile_key(
        self, key: str, store: SemanticStore, ledger: EvidenceLedger
    ) -> ReconciliationResult:
        """
        Reconcile a single key: find anchor, replay chain, update store if needed.
        """
        chain = self._build_version_chain(key, ledger)
        current_active = store.get(key)

        if not chain:
            return ReconciliationResult(
                key=key,
                action="no_change",
                anchor_version=None,
                previous_value=current_active.value if current_active else "",
                reconciled_value=current_active.value if current_active else "",
                num_writes_replayed=0,
                num_skipped=0,
                num_rolled_back=0,
            )

        anchor = self._find_anchor(chain)
        reconciled, num_replayed, num_skipped, num_rolled_back = self._replay_from_anchor(
            chain, anchor
        )

        if reconciled is None:
            return ReconciliationResult(
                key=key,
                action="no_change",
                anchor_version=None,
                previous_value=current_active.value if current_active else "",
                reconciled_value=current_active.value if current_active else "",
                num_writes_replayed=0,
                num_skipped=0,
                num_rolled_back=0,
            )

        previous_value = current_active.value if current_active else ""
        reconciled_value = reconciled.value

        # Determine action label
        if anchor and reconciled.value == anchor.value and reconciled.version > anchor.version:
            action = "anchored"
        elif num_rolled_back > 0:
            action = "rolled_back"
        elif reconciled.value != previous_value:
            action = "fallback"
        else:
            action = "no_change"

        # Update the store with the reconciled record
        if reconciled.value != previous_value:
            store.upsert(reconciled)

        return ReconciliationResult(
            key=key,
            action=action,
            anchor_version=anchor.version if anchor else None,
            previous_value=previous_value,
            reconciled_value=reconciled_value,
            num_writes_replayed=num_replayed,
            num_skipped=num_skipped,
            num_rolled_back=num_rolled_back,
        )

    # ------------------------------------------------------------------
    # Main reconcile loop
    # ------------------------------------------------------------------

    def run(self, store: SemanticStore, ledger: EvidenceLedger) -> List[ReconciliationResult]:
        """
        Reconcile all keys in the store against the immutable ledger.

        Algorithm:
          For each key in the store:
            1. Build version chain from ledger
            2. Find immutable anchor
            3. Replay chain from anchor
            4. Update store if reconciled value differs
        """
        started = perf_counter()
        self.results = []

        all_keys = set(store.records.keys())
        items_fixed = 0

        for key in all_keys:
            result = self.reconcile_key(key, store, ledger)
            self.results.append(result)
            if result.action in ("anchored", "rolled_back", "fallback"):
                items_fixed += 1

        self.last_run_seconds = perf_counter() - started
        return items_fixed
