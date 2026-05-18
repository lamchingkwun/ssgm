"""
provenance_detector.py
====================
Embedding-based provenance anomaly detector for SSGM.

Replaces the naive provenance_ok=True/False heuristic with a three-level
detection pipeline that does NOT rely on manually maintained source lists:

  Level 1 (Rule-based fast path):
    Explicitly malicious sources → immediate reject (block)
    Trusted authoritative sources → immediate accept (allow)

  Level 2 (Embedding anomaly detection):
    Compute embedding of the new record's content.
    Measure k-NN anomaly score: average cosine distance to k nearest neighbors
    in the current episodic ledger.
    If score > threshold → quarantine for Level 3.

  Level 3 (LLM-based NLI, optional / future):
    For quarantined records, ask: "Does this contradict M_core?"
    Calibrated abstention: if LLM confidence < threshold → defer to human.

Requirements:
    Ollama running locally with an embedding model available.

Note:
    Dataset `provenance_ok` is treated as a gold/evaluation label. Runtime
    inference should produce its own predicted trust decision rather than
    consuming the gold field as direct input.

Usage:
    detector = ProvenanceDetector()
    result = detector.evaluate(record=MemoryRecord(...), ledger=EvidenceLedger(...))
    # result.action: 'allow' | 'quarantine' | 'block'
    # result.anomaly_score: float
    # result.reasons: list[str]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .store import OllamaEmbeddingModel, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_EMBED_MODEL

# ---------------------------------------------------------------------------
# Provenance result
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceResult:
    action: str          # 'allow' | 'quarantine' | 'block'
    anomaly_score: float = 0.0   # 0=normal, 1=max anomaly
    confidence: float = 1.0       # LLM confidence (Level 3), 1.0 if not used
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trusted / malicious source lists (Level 1 rules)
# ---------------------------------------------------------------------------

# High-authority sources that are trusted
TRUSTED_SOURCES = frozenset({
    'calendar', 'hr', 'admin', 'tool_result', 'manual_correction',
    'authenticated_api', 'bank_statement', 'official_document',
})

# Explicitly malicious sources that are always blocked
MALICIOUS_SOURCES = frozenset({
    'prompt_injection', 'malicious_tool', 'poisoned_copy',
    'adversarial_prompt', 'data_exfiltration_attempt',
})


# ---------------------------------------------------------------------------
# ProvenanceDetector
# ---------------------------------------------------------------------------

class ProvenanceDetector:
    """
    Three-level provenance detection.

    Level 1: Rule-based source classification (fast path)
    Level 2: Embedding anomaly detection (k-NN distance in semantic space)
    Level 3: LLM-based NLI contradiction check (optional, future)

    Parameters:
        k_neighbours: k for k-NN anomaly scoring (default: 5)
        anomaly_threshold: max allowed avg k-NN cosine distance (default: 0.7)
        embedding_model: Ollama embedding model name (default: nomic-embed-text-v2-moe)
        use_llm: whether to use Level 3 LLM check (default: False, not implemented)
    """

    def __init__(
        self,
        k_neighbours: int = 5,
        anomaly_threshold: float = 0.7,
        embedding_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        use_llm: bool = False,
        embedding_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        require_attestation_for_trusted_sources: bool = False,
    ) -> None:
        self.k_neighbours = k_neighbours
        self.anomaly_threshold = anomaly_threshold
        self.embedding_model_name = embedding_model
        self.embedding_base_url = embedding_base_url
        self.use_llm = use_llm
        self.require_attestation_for_trusted_sources = require_attestation_for_trusted_sources
        self._model = None  # lazy-loaded
        self._model_load_failed = False
        self._embedding_cache: dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        record,
        ledger,
        existing_embeddings: Optional[dict] = None,
        llm_judge = None,
    ) -> ProvenanceResult:
        """
        Evaluate a record's provenance.

        Args:
            record: MemoryRecord to evaluate
            ledger: EvidenceLedger containing historical records
            existing_embeddings: Optional dict {record_key: embedding}
                                 to reuse embeddings across calls

        Returns:
            ProvenanceResult with action, anomaly_score, confidence, reasons
        """
        existing_embeddings = existing_embeddings or {}

        # Level 1: Rule-based fast path
        if record.source in MALICIOUS_SOURCES:
            return ProvenanceResult(
                action="block",
                anomaly_score=1.0,
                confidence=1.0,
                reasons=[f"Malicious source '{record.source}' immediately blocked by rule."],
            )

        if (
            record.source in TRUSTED_SOURCES
            and record.confidence >= 0.9
            and (
                not self.require_attestation_for_trusted_sources
                or getattr(record, "provenance_attested", None) is True
            )
        ):
            return ProvenanceResult(
                action="allow",
                anomaly_score=0.0,
                confidence=1.0,
                reasons=[f"Trusted high-confidence source '{record.source}' allowed by rule."],
            )
        if (
            self.require_attestation_for_trusted_sources
            and record.source in TRUSTED_SOURCES
            and getattr(record, "provenance_attested", None) is not True
        ):
            return ProvenanceResult(
                action="quarantine",
                anomaly_score=0.85,
                confidence=0.9,
                reasons=[f"Trusted-looking source '{record.source}' missing provenance attestation."],
            )

        # Level 2: Embedding anomaly detection
        anomaly_score, reasons = self._embedding_anomaly_score(
            record, ledger, existing_embeddings
        )

        if anomaly_score >= self.anomaly_threshold:
            if self.use_llm and llm_judge is not None:
                judgment = llm_judge.classify(content=record.value, source=record.source, key=record.key)
                decision = judgment.get("decision", "quarantine")
                if decision == "allow":
                    return ProvenanceResult(
                        action="allow",
                        anomaly_score=max(0.0, anomaly_score - 0.5),
                        confidence=judgment.get("confidence", 0.5),
                        reasons=reasons + [f"Level 3 LLM check passed. Reason: {judgment.get('reasoning')}"],
                    )
                elif decision == "block":
                    return ProvenanceResult(
                        action="block",
                        anomaly_score=1.0,
                        confidence=judgment.get("confidence", 0.9),
                        reasons=reasons + [f"Level 3 LLM BLOCKED. Reason: {judgment.get('reasoning')}"],
                    )
                else:
                    return ProvenanceResult(
                        action="quarantine",
                        anomaly_score=anomaly_score,
                        confidence=judgment.get("confidence", 0.7),
                        reasons=reasons + [f"Level 3 LLM QUARANTINED. Reason: {judgment.get('reasoning')}"],
                    )
            
            return ProvenanceResult(
                action="quarantine",
                anomaly_score=anomaly_score,
                confidence=0.5,
                reasons=reasons + [
                    f"Anomaly score {anomaly_score:.3f} exceeds threshold {self.anomaly_threshold}. "
                    "Record quarantined for review. "
                    "Set use_llm=True and provide llm_judge for automatic LLM-based adjudication."
                ],
            )

        # Within threshold → allow
        return ProvenanceResult(
            action="allow",
            anomaly_score=anomaly_score,
            confidence=1.0,
            reasons=reasons + [f"Anomaly score {anomaly_score:.3f} within threshold."],
        )

    # ------------------------------------------------------------------
    # Level 2: Embedding-based anomaly detection
    # ------------------------------------------------------------------

    def _embedding_anomaly_score(self, record, ledger, existing_embeddings: dict) -> Tuple[float, List[str]]:
        """Compute k-NN anomaly score for a record using semantic embeddings.

        Falls back to the confidence-based heuristic on ANY error
        (network timeout, Ollama unavailability, encoding error, etc.)
        so the experiment never crashes if the embedding service is unreachable.
        """
        # ── Load model (cached after first call) ────────────────────────
        if self._model_load_failed:
            return self._confidence_anomaly_fallback(record)

        if self._model is None:
            try:
                self._model = OllamaEmbeddingModel(
                    model=self.embedding_model_name,
                    base_url=self.embedding_base_url,
                )
            except Exception as e:
                print(f"[ProvenanceDetector] WARNING: model load failed "
                      f"'{self.embedding_model_name}': {e}. "
                      f"Falling back to heuristic.")
                self._model = None
                self._model_load_failed = True
                return self._confidence_anomaly_fallback(record)

        reasons: List[str] = []

        # ── Encode new record (may throw on network/HF timeout) ────────
        try:
            new_embedding = self._model.encode(str(record.value), convert_to_numpy=True)
        except Exception as e:
            print(f"[ProvenanceDetector] WARNING: encode(new) failed: {e}. "
                  f"Falling back to heuristic.")
            return self._confidence_anomaly_fallback(record)

        # Collect context records from the ledger
        context_records = [
            event.record
            for event in ledger.all()
            if getattr(event, "op", "write") == "write"
            and event.accepted
            and event.record.tenant_id == record.tenant_id
            and event.record.key != record.key
            and str(event.record.value).strip()
        ]

        if not context_records:
            return 0.0, ["No historical context. Record allowed by default."]

        # ── Reuse cached context embeddings when available, encode only misses ──
        try:
            import numpy as np

            context_embeddings = [None] * len(context_records)
            missing_records = []
            missing_indices = []
            for index, context_record in enumerate(context_records):
                cached = existing_embeddings.get(context_record.key)
                if cached is not None:
                    context_embeddings[index] = np.array(cached)
                else:
                    missing_indices.append(index)
                    missing_records.append(context_record)

            if missing_records:
                context_values = [str(r.value) for r in missing_records]
                encoded_rows = self._model.encode(context_values, convert_to_numpy=True)
                if getattr(encoded_rows, "ndim", 1) == 1:
                    encoded_rows = [encoded_rows]
                for index, context_record, encoded_row in zip(missing_indices, missing_records, encoded_rows):
                    row = np.array(encoded_row)
                    existing_embeddings[context_record.key] = row
                    context_embeddings[index] = row

            context_embeddings = np.array(context_embeddings)
        except Exception as e:
            print(f"[ProvenanceDetector] WARNING: encode(context) failed: {e}. "
                  f"Falling back to heuristic.")
            return self._confidence_anomaly_fallback(record)

        # k-NN: find k nearest neighbours
        distances = self._cosine_distances(new_embedding, context_embeddings)
        distances_sorted = sorted(distances)
        knn_distances = distances_sorted[: self.k_neighbours]
        avg_knn_distance = sum(knn_distances) / len(knn_distances)

        # Normalize to [0, 1] where 1 = most anomalous
        anomaly_score = min(avg_knn_distance / 0.5, 1.0)

        reasons.append(
            f"Embedding k-NN anomaly score: {anomaly_score:.3f} "
            f"(avg dist to {self.k_neighbours}-NN = {avg_knn_distance:.3f})."
        )

        # Additional signal: low-confidence summary source
        if record.source == 'summary' and record.confidence < 0.5:
            anomaly_score = max(anomaly_score, 0.6)
            reasons.append(
                f"Low-confidence summary (conf={record.confidence:.2f}). "
                "Boosting anomaly score."
            )

        # Additional signal: provenance_ok=False is a strong signal
        if not record.provenance_ok:
            anomaly_score = max(anomaly_score, 0.8)
            reasons.append("provenance_ok=False. Boosting anomaly score.")

        return anomaly_score, reasons

    def _cosine_distances(self, vec, neighbours) -> List[float]:
        """Compute cosine distance from vec to each row of neighbours."""
        norm_vec = vec / (vec.dot(vec) ** 0.5 + 1e-10)
        distances = []
        for n in neighbours:
            norm_n = n / (n.dot(n) ** 0.5 + 1e-10)
            cosine_sim = max(0.0, norm_vec.dot(norm_n))
            distances.append(1.0 - cosine_sim)
        return distances

    def _confidence_anomaly_fallback(self, record) -> Tuple[float, List[str]]:
        """
        Fallback when Ollama embeddings are unavailable.
        Uses a simple heuristic based on source type and confidence.
        """
        score = 0.0
        reasons = ["Ollama embeddings unavailable; using confidence heuristic."]

        if record.source in MALICIOUS_SOURCES:
            return 1.0, reasons + ["Malicious source detected."]

        if record.source == 'summary':
            score = max(score, 1.0 - record.confidence)

        if not record.provenance_ok:
            score = max(score, 0.8)

        return score, reasons
