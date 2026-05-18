from __future__ import annotations

from dataclasses import dataclass

from ssgm.compact_nli_backend import DEFAULT_COMPACT_NLI_MODEL
from ssgm.llm_judge import get_judge


@dataclass
class RuntimeProvenanceInferer:
    judge: object | None = None

    def __post_init__(self) -> None:
        if self.judge is None:
            self.judge = get_judge("compact_nli", model=DEFAULT_COMPACT_NLI_MODEL)

    def infer(self, content: str, source: str, key: str, gold_provenance_ok=None) -> dict:
        result = self.judge.classify(content=content, source=source, key=key)
        return {
            **result,
            "predicted_provenance_ok": result["decision"] == "allow",
            "used_gold_label": False,
        }
