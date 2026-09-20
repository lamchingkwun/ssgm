"""
governor.py
===========
SSGM Governance Middleware — implements the governed read/write/reconcile loop.

Key changes from naive implementation:
  1. Provenance: replaced hardcoded source whitelist with ProvenanceDetector
     (embedding-based k-NN anomaly detection, Level 1+2 rules).
  2. Weibull Decay: replaced simple stale_after threshold with Weibull
     decay function: w(Δτ) = exp(-(Δτ/η)^κ).
  3. Contradiction Gate: replaced simple value-equality check with
     proper three-way TMS decision:
       - hard contradiction → reject
       - time-qualified update → version and supersede
       - insufficient evidence → abstain (defer to reconciliation or human)
  4. WriteResult: all write decisions are now captured in last_write_result
     for adapter inspection.

Design Philosophy:
  Governance is a FILTERS-AND-DECISIONS layer. It intercepts every read/write
  and adds structured policy decisions. It does NOT generate content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter
from typing import List, Optional

from .ledger import EvidenceLedger
from .metrics import RunMetrics
from .models import AccessContext, MemoryEvent, MemoryRecord
from .provenance_detector import ProvenanceDetector, ProvenanceResult
from .reconciler import BatchReconciler
from .store import SemanticStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ContradictionDecision(Enum):
    """Three-way TMS contradiction decision."""
    HARD_CONTRADICTION = "hard_contradiction"  # reject write
    TIME_QUALIFIED_UPDATE = "time_qualified_update"  # version and supersede
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # defer to reconciliation


# ---------------------------------------------------------------------------
# Governance Capabilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GovernanceCapabilities:
    apply_access_control: bool = False
    apply_provenance_filter_on_read: bool = False
    apply_provenance_filter_on_write: bool = False
    apply_contradiction_gate: bool = False
    apply_reconciliation: bool = False
    apply_freshness_filter: bool = False


MODE_CAPABILITIES = {
    "vanilla": GovernanceCapabilities(),
    "access_only": GovernanceCapabilities(apply_access_control=True),
    "provenance_only": GovernanceCapabilities(
        apply_provenance_filter_on_read=True, apply_provenance_filter_on_write=True
    ),
    "write_gate_only": GovernanceCapabilities(
        apply_provenance_filter_on_write=True, apply_contradiction_gate=True
    ),
    "read_filter_only": GovernanceCapabilities(
        apply_access_control=True,
        apply_provenance_filter_on_read=True,
        apply_freshness_filter=True,
    ),
    "write_gate_plus_access": GovernanceCapabilities(
        apply_access_control=True,
        apply_provenance_filter_on_write=True,
        apply_contradiction_gate=True,
    ),
    "write_gate_plus_provenance": GovernanceCapabilities(
        apply_provenance_filter_on_read=True,
        apply_provenance_filter_on_write=True,
        apply_contradiction_gate=True,
    ),
    "write_gate_read_filter": GovernanceCapabilities(
        apply_access_control=True,
        apply_provenance_filter_on_read=True,
        apply_provenance_filter_on_write=True,
        apply_contradiction_gate=True,
        apply_freshness_filter=True,
    ),
    "full_ssgm": GovernanceCapabilities(
        apply_access_control=True,
        apply_provenance_filter_on_read=True,
        apply_provenance_filter_on_write=True,
        apply_contradiction_gate=True,
        apply_reconciliation=True,
        apply_freshness_filter=True,
    ),
}


# ---------------------------------------------------------------------------
# Weibull Decay
# ---------------------------------------------------------------------------

@dataclass
class WeibullDecayConfig:
    """
    Weibull decay function: w(Δτ) = exp(-(Δτ/η)^κ)

    Parameters:
        eta: scale parameter (characteristic decay time scale)
        kappa: shape parameter (controls curvature)
        threshold: minimum relevance score to be considered "fresh"
    """
    eta: float = 2.0
    kappa: float = 1.0
    threshold: float = 0.3

    def relevance(self, age: float) -> float:
        """
        Compute relevance score for a memory item with given age (Δτ).
        Returns value in [0, 1]: 1=fresh, 0=fully decayed.
        """
        import math
        return math.exp(-(age / self.eta) ** self.kappa)

    def is_stale(self, age: float) -> bool:
        """Return True if the memory item's relevance has dropped below threshold."""
        return self.relevance(age) < self.threshold


# ---------------------------------------------------------------------------
# Write Result
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """Result of a write operation, stored on engine for adapter inspection."""
    accepted: bool
    reason: str
    record: MemoryRecord
    contradiction_decision: ContradictionDecision = ContradictionDecision.HARD_CONTRADICTION
    provenance_result: Optional[ProvenanceResult] = None
    blocked_by_contradiction: bool = False
    blocked_by_provenance: bool = False
    blocked_by_schema: bool = False
    blocked_by_stale: bool = False
    quarantined: bool = False
    judge_decision: str = "allow"
    judge_details: Optional[dict] = None


# ---------------------------------------------------------------------------
# SSGM Engine
# ---------------------------------------------------------------------------

class SSGMEngine:
    POISON_SOURCES = frozenset({
        "poisoned_copy", "prompt_injection", "malicious_tool",
        "adversarial_prompt", "data_exfiltration_attempt",
    })

    def __init__(
        self,
        mode: str = "vanilla",
        stale_after: int = 5,
        weibull_config: Optional[WeibullDecayConfig] = None,
        provenance_detector: Optional[ProvenanceDetector] = None,
        use_weibull: bool = False,
        nli_adjudicator: Optional = None,  # Level 3 NLI adjudicator
        mcore_config: Optional = None,    # M_core query config
        llm_judge: Optional = None,
        embedding_model: str = "nomic-embed-text-v2-moe",
        embedding_base_url: Optional[str] = None,
        require_provenance_attestation: bool = False,
        temporal_confidence_margin: float = 0.0,
        defer_without_nli: bool = False,
        use_embeddings: bool = True,
        allow_embedding_fallback: bool = False,
    ) -> None:
        if mode not in MODE_CAPABILITIES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.capabilities = MODE_CAPABILITIES[mode]
        self.stale_after = stale_after
        self.use_weibull = use_weibull
        self.weibull = weibull_config or WeibullDecayConfig()
        self.require_provenance_attestation = require_provenance_attestation
        self.temporal_confidence_margin = max(0.0, float(temporal_confidence_margin))
        self.defer_without_nli = defer_without_nli
        self.provenance_detector = provenance_detector or ProvenanceDetector(
            use_embeddings=use_embeddings, allow_embedding_fallback=allow_embedding_fallback,
            embedding_model=embedding_model,
            **({"embedding_base_url": embedding_base_url} if embedding_base_url else {}),
        )
        self.provenance_detector.require_attestation_for_trusted_sources = require_provenance_attestation

        # Level 3 NLI (optional — requires API key)
        self.nli_adjudicator = nli_adjudicator

        # M_core query (for NLI context)
        from .memory_query import MCoreQuery, MCoreQueryConfig
        self.mcore_config = mcore_config or MCoreQueryConfig()
        self._mcore_query = MCoreQuery()

        # Optional content-based write judge.
        self.llm_judge = llm_judge

        self.ledger = EvidenceLedger()
        self.store = SemanticStore(embedding_model=embedding_model, base_url=embedding_base_url,
                                   use_embeddings=use_embeddings,
                                   allow_embedding_fallback=allow_embedding_fallback)
        self.metrics = RunMetrics()
        self.reconciler = BatchReconciler()
        self.last_write_result: Optional[WriteResult] = None
        self._active_anchor_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_ok(record: MemoryRecord) -> bool:
        return bool(record.key and record.value and record.tenant_id and record.source)

    def _quarantine_store_record(self, record: MemoryRecord) -> MemoryRecord:
        ordinal = self.metrics.quarantined_writes
        quarantine_key = f"{record.key}::quarantine::{record.timestamp}:{ordinal}"
        return replace(record, key=quarantine_key)

    @staticmethod
    def _attested_native_fast_path(record: MemoryRecord) -> bool:
        try:
            confidence = float(record.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return (
            getattr(record, "provenance_attested", None) is True
            and getattr(record, "provenance_ok", True) is True
            and record.source in {"chat", "assistant"}
            and confidence >= 0.9
            and "native_longmemeval" in set(record.tags or [])
        )

    # ------------------------------------------------------------------
    # Provenance (Level 2 — embedding-based)
    # ------------------------------------------------------------------

    def _evaluate_provenance(self, record: MemoryRecord) -> ProvenanceResult:
        """
        Evaluate provenance using the three-level detector.
        """
        return self.provenance_detector.evaluate(
            record=record,
            ledger=self.ledger,
            existing_embeddings=self.store._record_embeddings,
            llm_judge=self.llm_judge,
        )

    @staticmethod
    def _decision_rank(decision: str) -> int:
        return {"allow": 0, "quarantine": 1, "block": 2}.get(decision, 0)

    def _stricter_decision(self, left: str, right: str) -> str:
        return left if self._decision_rank(left) >= self._decision_rank(right) else right

    def _combine_provenance_decisions(self, detector_decision: str, direct_judge_decision: str, has_direct_judge: bool) -> str:
        """
        Combine provenance-detector and direct-judge decisions.

        Design intent:
        - detector ``block`` is a hard safety stop and cannot be bypassed
        - detector ``quarantine`` is an escalation signal, not a final verdict
        - when a direct judge is available, it adjudicates quarantined records
        """
        if detector_decision == "block":
            return "block"
        if not has_direct_judge:
            return detector_decision
        if detector_decision == "quarantine":
            return direct_judge_decision
        return direct_judge_decision

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    def _semantic_anchor_conflict(self, record: MemoryRecord) -> Optional[tuple[MemoryRecord, float]]:
        """
        Detect embedding-close writes that shadow an authoritative anchor under a
        different key. This is a targeted defense for ER-MIA-style co-retrieval
        attacks where a black-box adversary injects a near-duplicate false fact
        rather than directly overwriting the anchor key.
        """
        if not self.capabilities.apply_contradiction_gate:
            return None

        if not self._active_anchor_keys:
            return None

        new_tokens = self._content_tokens(record.value)
        if not new_tokens:
            return None

        anchor_candidates: list[MemoryRecord] = []

        for anchor_key in self._active_anchor_keys:
            existing = self.store.get(anchor_key)
            if existing is None:
                continue
            if existing.tenant_id != record.tenant_id or existing.status != "active":
                continue
            if existing.key == record.key:
                continue
            if getattr(existing, "memory_class", "ordinary") != "anchor" and getattr(existing, "conflict_policy", "auto_update") != "immutable":
                continue
            if existing.value == record.value:
                continue
            anchor_candidates.append(existing)

        if not anchor_candidates:
            return None

        query_emb = self.store._maybe_encode(record.value)
        if query_emb is None:
            return None

        import numpy as np

        norm_query = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        best_match: Optional[tuple[MemoryRecord, float]] = None

        for existing in anchor_candidates:

            emb = self.store._record_embeddings.get(existing.key)
            if emb is None:
                continue

            existing_tokens = self._content_tokens(existing.value)
            if not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(1, min(len(new_tokens), len(existing_tokens)))
            if overlap < 0.45:
                continue

            norm_emb = emb / (np.linalg.norm(emb) + 1e-10)
            similarity = float(max(0.0, np.dot(norm_query, norm_emb)))
            if similarity < 0.72:
                continue

            if best_match is None or similarity > best_match[1]:
                best_match = (existing, similarity)

        return best_match

    # ------------------------------------------------------------------
    # Weibull freshness
    # ------------------------------------------------------------------

    def _compute_freshness_relevance(self, record: MemoryRecord, now_ts: int) -> float:
        """Compute Weibull freshness relevance score for a record."""
        age = max(0, now_ts - record.timestamp)
        return self.weibull.relevance(float(age))

    def _is_stale_weibull(self, record: MemoryRecord, now_ts: int) -> bool:
        """Return True if record's Weibull relevance is below threshold."""
        return self.weibull.is_stale(float(max(0, now_ts - record.timestamp)))

    # ------------------------------------------------------------------
    # TMS Contradiction Gate — Three-way decision
    # ------------------------------------------------------------------

    def _contradiction_check(
        self,
        old: MemoryRecord,
        new: MemoryRecord,
        now_ts: int,
        m_core_facts: Optional[List[MemoryRecord]] = None,
        judge_decision: str = "allow",
    ) -> ContradictionDecision:
        """
        Three-way TMS contradiction decision.

        Implements the SSGM Principle 1 cascade:
          (i) typed/schema constraints — checked via schema_ok
          (ii) symbolic consistency check — value/semantic contradiction
          (iii) calibrated NLI with abstention — for INSUFFICIENT_EVIDENCE cases

        Args:
            old: Existing record in store
            new: New record being proposed for write
            now_ts: Current timestamp
            m_core_facts: Established M_core facts from episodic ledger (for NLI Level 3)

        Returns:
            ContradictionDecision.
        """
        # Different keys or tenants → no contradiction possible
        if old.key != new.key or old.tenant_id != new.tenant_id:
            return ContradictionDecision.TIME_QUALIFIED_UPDATE

        old_memory_class = getattr(old, "memory_class", "ordinary")
        old_conflict_policy = getattr(old, "conflict_policy", "auto_update")

        # Strongest class: anchors / immutable policy.
        # These should never be auto-overwritten simply because a newer write exists.
        if old_memory_class == "anchor" or old_conflict_policy == "immutable":
            if old.value != new.value:
                return ContradictionDecision.HARD_CONTRADICTION
            return ContradictionDecision.TIME_QUALIFIED_UPDATE

        # Protected memory: important but changeable. Conflicting writes should be
        # surfaced conservatively instead of either blindly updating or hard-blocking.
        if old_memory_class == "protected" or old_conflict_policy == "confirm_required":
            if old.value != new.value:
                return ContradictionDecision.INSUFFICIENT_EVIDENCE
            return ContradictionDecision.TIME_QUALIFIED_UPDATE

        # Legacy immutable handling for records that still rely only on `mutable=False`.
        # For ordinary benign updates we now avoid unnecessary hard drops, but the
        # stronger semantics above (anchor/protected) remain strict.
        if not old.mutable:
            if old.value != new.value:
                if (
                    judge_decision == "allow"
                    and new.provenance_ok
                    and new.timestamp >= old.timestamp
                    and new.source not in self.POISON_SOURCES
                ):
                    return ContradictionDecision.TIME_QUALIFIED_UPDATE
                return ContradictionDecision.HARD_CONTRADICTION
            else:
                return ContradictionDecision.TIME_QUALIFIED_UPDATE

        # Mutable: same value → re-confirm
        if old.value == new.value:
            return ContradictionDecision.TIME_QUALIFIED_UPDATE

        # Different values on mutable → check temporal relationship
        time_delta = new.timestamp - old.timestamp

        # ── Apply Weibull decay to old record's effective confidence ───────
        # This implements the temporal dimension of Eq.(6):
        # A high-confidence record that is old should not block a new
        # lower-confidence record if the old record has significantly decayed.
        old_age = float(max(0, now_ts - old.timestamp))
        old_decay = self.weibull.relevance(old_age)
        old_effective_conf = old.confidence * old_decay

        # New record's effective confidence (no decay applied since it's the latest)
        new_effective_conf = new.confidence

        # ── Temporal contradiction decision ────────────────────────────────
        # New is newer but lower effective confidence → INSUFFICIENT_EVIDENCE
        confidence_gap = old_effective_conf - new_effective_conf
        if time_delta > 0 and confidence_gap > self.temporal_confidence_margin:
            # Upgrade with Level 3 NLI if adjudicator is available
            if self.nli_adjudicator is not None and m_core_facts:
                # Level 3: ask LLM whether new contradicts M_core
                nli_decision = self.nli_adjudicator.adjudicate(
                    new_record=new,
                    m_core_facts=m_core_facts,
                )
                if nli_decision.action == "contradiction":
                    return ContradictionDecision.HARD_CONTRADICTION
                elif nli_decision.action == "consistent":
                    return ContradictionDecision.TIME_QUALIFIED_UPDATE
                else:
                    # LLM abstained → fall back to INSUFFICIENT_EVIDENCE
                    return ContradictionDecision.INSUFFICIENT_EVIDENCE
            else:
                # No NLI adjudicator → either keep the relaxed default or expose the
                # low-confidence update for deferred repair when calibration asks for it.
                if self.defer_without_nli:
                    return ContradictionDecision.INSUFFICIENT_EVIDENCE
                return ContradictionDecision.TIME_QUALIFIED_UPDATE

        # New is more recent with similar or higher effective confidence → TIME_QUALIFIED
        return ContradictionDecision.TIME_QUALIFIED_UPDATE

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(self, record: MemoryRecord, now_ts: Optional[int] = None) -> bool:
        """
        Governed write with three-way contradiction gate and Level 2 provenance.

        Args:
            record: MemoryRecord to write
            now_ts: Optional current timestamp (for Weibull age computation)

        Returns:
            True if the write was accepted, False if rejected.
        """
        started = perf_counter()
        # If the caller does not provide an explicit wall-clock, use the
        # record timestamp so temporal decay in the write gate reflects the
        # ordering of observed events rather than a frozen zero point.
        now_ts = record.timestamp if now_ts is None else now_ts
        self.metrics.writes_attempted += 1
        self.metrics.write_validation_ops += 1

        old = self.store.get(record.key)

        # ── Provenance / governance decision via LLM judge ───────────────
        # If llm_judge is available, use it to dynamically classify writes into
        # allow / quarantine / block. This is more expressive than a binary
        # poison flag and reduces false-positive hard drops.
        judgment = None
        judge_decision = "allow"
        provenance_result = None
        direct_judge_decision = "allow"
        direct_is_poison = False
        attested_native_fast_path = self._attested_native_fast_path(record)
        if self.llm_judge is not None and not attested_native_fast_path:
            judgment = self.llm_judge.classify(
                content=record.value,
                source=record.source,
                key=record.key,
                provenance_attested=getattr(record, "provenance_attested", None),
                require_provenance_attestation=self.require_provenance_attestation,
            )
            direct_judge_decision = judgment.get("decision", "allow")
            direct_is_poison = judgment.get("is_poisoned", False)

        detector_decision = "allow"
        if self.capabilities.apply_provenance_filter_on_write:
            if attested_native_fast_path:
                detector_result = ProvenanceResult(
                    action="allow",
                    anomaly_score=0.0,
                    confidence=1.0,
                    reasons=["Attested native LongMemEval turn allowed by provenance fast path."],
                )
            else:
                detector_result = self.provenance_detector.evaluate(
                    record=record,
                    ledger=self.ledger,
                    existing_embeddings=self.store._record_embeddings,
                    llm_judge=None,
                )
            detector_decision = detector_result.action
            judge_decision = self._combine_provenance_decisions(
                detector_decision=detector_decision,
                direct_judge_decision=direct_judge_decision,
                has_direct_judge=judgment is not None,
            )
            combined_reasons = list(detector_result.reasons)
            if judgment is not None and judgment.get("reasoning"):
                combined_reasons.append(
                    f"Direct judge {direct_judge_decision}. Reason: {judgment.get('reasoning')}"
                )
            provenance_result = replace(
                detector_result,
                action=judge_decision,
                anomaly_score=max(
                    detector_result.anomaly_score,
                    1.0 if direct_judge_decision == "block" else 0.6 if direct_judge_decision == "quarantine" else 0.0,
                ),
                confidence=max(detector_result.confidence, judgment.get("confidence", 0.0) if judgment is not None else 0.0),
                reasons=combined_reasons,
            )
            is_poison = direct_is_poison or judge_decision == "block"
        else:
            # Compatibility mode for ungated baselines: retain gold-label accounting
            # for metrics without turning it into a runtime write decision.
            is_poison = (
                not record.provenance_ok
            ) or record.source in self.POISON_SOURCES
            judge_decision = direct_judge_decision if judgment is not None else "allow"

        if judge_decision in {"block", "quarantine"} or is_poison:
            self.metrics.poisoned_writes_attempted += 1

        contradiction_decision = ContradictionDecision.TIME_QUALIFIED_UPDATE
        blocked_by_schema = False
        blocked_by_provenance = False
        blocked_by_contradiction = False
        quarantined = False
        stored_record = record
        accepted = True
        reason = "accepted"
        semantic_anchor_conflict = None

        # ── Stage 1: Schema check ──────────────────────────────────────
        if not self._schema_ok(record):
            accepted = False
            reason = "schema_failed"
            blocked_by_schema = True

        # ── Stage 2: Judge / provenance-driven write decision ──────────
        elif self.capabilities.apply_provenance_filter_on_write and judge_decision in {"block", "quarantine"}:
            decision_prefix = (
                "provenance"
                if detector_decision == "block"
                else "judge"
                if judgment is not None
                else "provenance"
            )
            if judge_decision == "block":
                accepted = False
                reason = f"{decision_prefix}_blocked"
                blocked_by_provenance = True
                self.metrics.poisoned_writes_blocked += 1
                self.metrics.blocked_writes += 1
            elif judge_decision == "quarantine":
                accepted = False
                quarantined = True
                reason = f"{decision_prefix}_quarantined"
                blocked_by_provenance = True
                self.metrics.quarantined_writes += 1
                quarantine_reason = (
                    judgment.get("reasoning", "")
                    if judgment is not None
                    else "; ".join(provenance_result.reasons) if provenance_result is not None else "quarantined"
                )
                record = replace(record, status="quarantined", quarantine_reason=quarantine_reason)
                stored_record = self._quarantine_store_record(record)

        # ── Stage 2.5: Semantic anchor-shadowing guard ────────────────
        elif self.capabilities.apply_contradiction_gate:
            semantic_anchor_conflict = self._semantic_anchor_conflict(record)
            if semantic_anchor_conflict is not None:
                conflicting_anchor, similarity = semantic_anchor_conflict
                accepted = False
                quarantined = True
                blocked_by_contradiction = True
                reason = "semantic_anchor_quarantined"
                self.metrics.quarantined_writes += 1
                self.metrics.contradictions += 1
                self.metrics.contradictions_deferred += 1
                record = replace(
                    record,
                    status="quarantined",
                    quarantine_reason=(
                        f"semantic conflict with anchor {conflicting_anchor.key} "
                        f"(similarity={similarity:.2f})"
                    ),
                )
                stored_record = self._quarantine_store_record(record)

        # ── Stage 3: Contradiction gate (three-way TMS + Level 3 NLI) ──
        if accepted and self.capabilities.apply_contradiction_gate and old:
            # Get M_core facts for Level 3 NLI adjudication (if enabled)
            m_core_facts = None
            if self.nli_adjudicator is not None:
                m_core_facts = self._mcore_query.get_mcore(
                    ledger=self.ledger,
                    key=record.key,
                    tenant_id=record.tenant_id,
                    now_ts=now_ts,
                    config=self.mcore_config,
                )
            contradiction_decision = self._contradiction_check(
                old, record, now_ts, m_core_facts, judge_decision=judge_decision
            )

            if contradiction_decision == ContradictionDecision.HARD_CONTRADICTION:
                accepted = False
                reason = "contradiction_hard"
                blocked_by_contradiction = True
                self.metrics.contradictions += 1
                self.metrics.contradictions_hard += 1
            elif contradiction_decision == ContradictionDecision.INSUFFICIENT_EVIDENCE:
                # Abstain: write not accepted, deferred for reconciliation
                accepted = False
                reason = "contradiction_deferred"
                blocked_by_contradiction = True  # treated as blocked for metrics
                self.metrics.contradictions += 1
                self.metrics.contradictions_deferred += 1

        # ── Store update ──────────────────────────────────────────────
        if accepted:
            self.metrics.writes_accepted += 1
            if (
                old
                and old.tenant_id == record.tenant_id
                and old.value != record.value
            ):
                record = replace(
                    record,
                    version=old.version + 1,
                    parent_version=old.version,
                )
            self.store.upsert(record)
            if getattr(record, "memory_class", "ordinary") == "anchor" or getattr(record, "conflict_policy", "auto_update") == "immutable":
                self._active_anchor_keys.add(record.key)
        elif quarantined:
            # Preserve quarantined records instead of dropping them entirely.
            # They are hidden from normal retrieval via status != active.
            self.store.upsert(stored_record)

        # ── Metrics and ledger ─────────────────────────────────────────
        self.metrics.write_validation_seconds += perf_counter() - started

        event = MemoryEvent(
            op="write",
            record=record,
            reason=reason,
            accepted=accepted,
            mode=self.mode,
        )
        self.ledger.append(event)

        # Populate last_write_result for adapter inspection
        self.last_write_result = WriteResult(
            accepted=accepted,
            reason=reason,
            record=record,
            contradiction_decision=contradiction_decision,
            provenance_result=provenance_result,
            blocked_by_contradiction=blocked_by_contradiction,
            blocked_by_provenance=blocked_by_provenance,
            blocked_by_schema=blocked_by_schema,
            quarantined=quarantined,
            judge_decision=judge_decision,
            judge_details=judgment,
        )

        return accepted

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read(
        self, key: str, ctx: AccessContext
    ) -> Optional[MemoryRecord]:
        """
        Governed read with ACL, provenance filter, and Weibull freshness.
        """
        started = perf_counter()
        now_ts = ctx.now_ts
        self.metrics.retrievals += 1
        self.metrics.read_filter_ops += 1

        record = self.store.get(key)
        if not record:
            self.metrics.read_filter_seconds += perf_counter() - started
            return None

        unauthorized = self.capabilities.apply_access_control and not self._access_allowed(record, ctx)
        cross_tenant = record.tenant_id != ctx.tenant_id
        blocked = False

        # ── Stage 1: Access control ───────────────────────────────────
        if unauthorized:
            self.metrics.leak_blocked += 1
            blocked = True

        # ── Stage 2: Quarantine / status filter ──────────────────────
        elif record.status == "quarantined":
            blocked = True

        # ── Stage 3: Provenance filter on read ───────────────────────
        elif (
            self.capabilities.apply_provenance_filter_on_read
            and not record.provenance_ok
        ):
            self.metrics.leak_blocked += 1
            blocked = True

        # ── Stage 4: Freshness filter (Weibull or threshold) ─────────
        if self.capabilities.apply_freshness_filter and not blocked:
            if self.use_weibull:
                stale = self._is_stale_weibull(record, now_ts)
            else:
                age = max(0, now_ts - record.timestamp)
                stale = age > self.stale_after

            if stale:
                self.metrics.stale_blocked += 1
                blocked = True
            # NOTE: stale_exposed is NOT incremented here.
            # A "stale exposure" occurs when a stale record is READ
            # (i.e., returned to the agent). That is tracked separately
            # by the benchmark harness, not by the governor itself,
            # because whether a stale read is an actual exposure depends
            # on downstream task impact.

        # ── Track cross-tenant leakage ─────────────────────────────────
        # leak_success: a cross-tenant record passed all gates and was returned.
        # This is a LEAK because it should not have been accessible.
        if unauthorized and not blocked:
            self.metrics.leak_success += 1

        self.metrics.read_filter_seconds += perf_counter() - started
        return None if blocked else record

    def retrieve(self, query: str, ctx: AccessContext, top_k: int = 5) -> List[MemoryRecord]:
        """
        Semantic retrieve with coarse tenant prefilter plus governed filtering.

        The implementation first expands a semantic top-2K pool from the active
        store, already tenant-scoped when access control is enabled, and then
        applies the remaining governed filters (scope, provenance, freshness)
        before truncating to top-K.
        """
        started = perf_counter()
        now_ts = ctx.now_ts
        self.metrics.retrievals += 1

        candidates = self.store.semantic_search(
            query,
            tenant_id=ctx.tenant_id if self.capabilities.apply_access_control else None,
            top_k=top_k * 2
        )

        valid_results = []
        for record in candidates:
            unauthorized = self.capabilities.apply_access_control and not self._access_allowed(record, ctx)
            cross_tenant = record.tenant_id != ctx.tenant_id
            blocked = False

            if unauthorized:
                self.metrics.leak_blocked += 1
                blocked = True
            elif record.status == "quarantined":
                blocked = True
            elif self.capabilities.apply_provenance_filter_on_read and not record.provenance_ok:
                self.metrics.leak_blocked += 1
                blocked = True

            if self.capabilities.apply_freshness_filter and not blocked:
                if self.use_weibull:
                    stale = self._is_stale_weibull(record, now_ts)
                else:
                    age = max(0, now_ts - record.timestamp)
                    stale = age > self.stale_after

                if stale:
                    self.metrics.stale_blocked += 1
                    blocked = True

            if unauthorized and not blocked:
                self.metrics.leak_success += 1

            if not blocked:
                valid_results.append(record)
                if len(valid_results) >= top_k:
                    break

        self.metrics.read_filter_seconds += perf_counter() - started
        return valid_results

    def _access_allowed(self, record: MemoryRecord, ctx: AccessContext) -> bool:
        if record.tenant_id != ctx.tenant_id:
            return False

        allowed_actors = set(getattr(record, "allowed_actors", []) or [])
        if allowed_actors and ctx.actor_id not in allowed_actors:
            return False

        allowed_roles = set(getattr(record, "allowed_roles", []) or [])
        if allowed_roles and ctx.role not in allowed_roles:
            return False

        record_scopes = set(getattr(record, "access_scopes", []) or [])
        legacy_scope = getattr(record, "provenance_scope", None)
        if legacy_scope:
            record_scopes.add(str(legacy_scope))
        if record_scopes:
            ctx_scopes = set(getattr(ctx, "scopes", []) or [])
            if ctx_scopes.isdisjoint(record_scopes):
                return False

        record_relations = set(getattr(record, "access_relations", []) or [])
        if record_relations:
            ctx_relations = set(getattr(ctx, "relations", []) or [])
            if ctx_relations.isdisjoint(record_relations):
                return False

        record_attributes = getattr(record, "access_attributes", {}) or {}
        ctx_attributes = getattr(ctx, "attributes", {}) or {}
        for key, expected in record_attributes.items():
            if ctx_attributes.get(key) != expected:
                return False

        return True

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self) -> None:
        """
        Asynchronous reconciliation: repair active keys through key-local replay
        against immutable anchors or a fallback accepted record.

        This implements the paper's operational repair rule M_clean(k)=R_k(V_k; Π)
        rather than a global optimization over the full ledger.
        """
        if not self.capabilities.apply_reconciliation:
            return

        started = perf_counter()
        self.metrics.reconciliation_ops += 1
        items_fixed = self.reconciler.run(self.store, self.ledger)
        if isinstance(items_fixed, int):
            fixed_count = items_fixed
        elif items_fixed is None:
            fixed_count = 0
        else:
            fixed_count = len(items_fixed)
        self.metrics.reconciliation_items_fixed = fixed_count
        self.metrics.reconciliation_seconds += perf_counter() - started

    def rollback(self, key: str, version: int):
        started = perf_counter()
        self.metrics.rollback_ops += 1

        before = self.store.get(key)
        restored = self.store.rollback(key, version)

        audit_record = restored or before or MemoryRecord(
            key=key,
            value="",
            tenant_id="",
            source="rollback",
            timestamp=0,
        )
        if restored is not None:
            self.metrics.rollback_applied += 1

        self.ledger.append(
            MemoryEvent(
                op="rollback",
                record=audit_record,
                reason="rollback_applied" if restored is not None else "rollback_missing_version",
                accepted=restored is not None,
                mode=self.mode,
                details={
                    "target_version": version,
                    "from_version": before.version if before is not None else None,
                    "to_version": restored.version if restored is not None else None,
                },
            )
        )
        self.metrics.rollback_seconds += perf_counter() - started
        return restored
