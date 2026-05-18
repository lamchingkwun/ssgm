from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunMetrics:
    writes_attempted: int = 0
    writes_accepted: int = 0
    contradictions: int = 0          # total contradiction checks (hard + deferred)
    contradictions_hard: int = 0    # hard contradictions (write rejected)
    contradictions_deferred: int = 0  # deferred (write blocked, for reconciliation)
    quarantined_writes: int = 0     # writes quarantined by provenance detector / LLM judge
    blocked_writes: int = 0         # writes explicitly blocked by governance
    stale_blocked: int = 0
    stale_exposed: int = 0
    leak_blocked: int = 0
    leak_success: int = 0
    retrievals: int = 0
    poisoned_writes_attempted: int = 0
    poisoned_writes_blocked: int = 0
    write_validation_seconds: float = 0.0
    write_validation_ops: int = 0
    read_filter_seconds: float = 0.0
    read_filter_ops: int = 0
    reconciliation_seconds: float = 0.0
    reconciliation_ops: int = 0
    reconciliation_items_fixed: int = 0  # items actually changed by reconciliation
    rollback_seconds: float = 0.0
    rollback_ops: int = 0
    rollback_applied: int = 0
    total_runtime_seconds: float = 0.0
    ledger_event_count: int = 0
    active_store_items: int = 0
    version_chain_count: int = 0
    total_versions: int = 0

    @staticmethod
    def _avg_ms(total_seconds: float, ops: int) -> float:
        return (total_seconds / ops * 1000.0) if ops else 0.0

    def snapshot_overhead(self, *, ledger_event_count: int, active_store_items: int, version_chain_count: int, total_versions: int) -> None:
        self.ledger_event_count = ledger_event_count
        self.active_store_items = active_store_items
        self.version_chain_count = version_chain_count
        self.total_versions = total_versions

    def to_dict(self):
        return {
            # Core admission/exposure metrics
            'write_acceptance_rate': self.writes_accepted / self.writes_attempted if self.writes_attempted else 0.0,
            'contradiction_block_rate': self.contradictions_hard / self.writes_attempted if self.writes_attempted else 0.0,
            'contradiction_deferred_rate': self.contradictions_deferred / self.writes_attempted if self.writes_attempted else 0.0,
            'stale_exposure_rate': self.stale_exposed / self.retrievals if self.retrievals else 0.0,
            'leakage_success_rate': self.leak_success / self.retrievals if self.retrievals else 0.0,
            'poisoned_write_acceptance_rate': (
                (self.poisoned_writes_attempted - self.poisoned_writes_blocked) / self.poisoned_writes_attempted
                if self.poisoned_writes_attempted else 0.0
            ),
            'quarantined_writes': self.quarantined_writes,
            'blocked_writes': self.blocked_writes,
            # Reconciliation
            'reconciliation_items_fixed': self.reconciliation_items_fixed,
            'rollback_ops': self.rollback_ops,
            'rollback_applied': self.rollback_applied,
            'avg_rollback_latency_ms': self._avg_ms(self.rollback_seconds, self.rollback_ops),
            # Latency
            'avg_write_validation_latency_ms': self._avg_ms(self.write_validation_seconds, self.write_validation_ops),
            'avg_read_filter_latency_ms': self._avg_ms(self.read_filter_seconds, self.read_filter_ops),
            'avg_reconciliation_latency_ms': self._avg_ms(self.reconciliation_seconds, self.reconciliation_ops),
            'total_runtime_ms': self.total_runtime_seconds * 1000.0,
            # Bookkeeping
            'ledger_event_count': self.ledger_event_count,
            'active_store_items': self.active_store_items,
            'version_chain_count': self.version_chain_count,
            'total_versions': self.total_versions,
            'total_retrievals': self.retrievals,
            'writes_attempted': self.writes_attempted,
            'writes_accepted': self.writes_accepted,
            'contradictions': self.contradictions,
            'contradictions_hard': self.contradictions_hard,
            'contradictions_deferred': self.contradictions_deferred,
            'poisoned_writes_attempted': self.poisoned_writes_attempted,
            'poisoned_writes_blocked': self.poisoned_writes_blocked,
        }
