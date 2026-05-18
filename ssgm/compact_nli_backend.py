from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence


DEFAULT_COMPACT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def _prepend_bundled_nvidia_libs() -> None:
    """
    Prefer the CUDA/NVIDIA libraries bundled with the PyTorch wheel.

    Some local environments export system CUDA paths first, which can break
    torch import with symbol mismatches even though the wheel ships a working
    runtime. Prepending the wheel-local directories keeps the compact baseline
    reproducible without requiring shell-specific environment surgery.
    """
    site_packages = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        site_packages / "nvidia" / "nvjitlink" / "lib",
        site_packages / "nvidia" / "cusparse" / "lib",
        site_packages / "nvidia" / "cusolver" / "lib",
        site_packages / "nvidia" / "cublas" / "lib",
    ]
    existing = [str(path) for path in candidate_dirs if path.exists()]
    if not existing:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    merged: list[str] = []
    for part in existing + current_parts:
        if part not in merged:
            merged.append(part)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _import_torch():
    _prepend_bundled_nvidia_libs()
    import torch

    return torch


class CompactZeroShotBackend:
    """Shared zero-shot NLI backend for compact classifier baselines."""

    def __init__(self, model: str = DEFAULT_COMPACT_NLI_MODEL, device: int | None = None):
        self.model = model
        self.device = device if device is not None else int(os.environ.get("SSGM_COMPACT_NLI_DEVICE", "-1"))
        self.batch_size = max(1, int(os.environ.get("SSGM_COMPACT_NLI_BATCH_SIZE", "32")))

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_pipeline(model: str, device: int):
        try:
            torch = _import_torch()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the compact NLI baseline. "
                "Install them with: pip install transformers sentencepiece torch"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model)
        if device >= 0 and torch.cuda.is_available():
            nli_model.to(f"cuda:{device}")
        else:
            nli_model.to("cpu")
        nli_model.eval()
        return tokenizer, nli_model

    def _entailment_scores_many(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[list[float]]:
        if not premises:
            return []

        torch = _import_torch()

        tokenizer, nli_model = self._load_pipeline(self.model, self.device)
        label_map = {int(idx): str(label).lower() for idx, label in nli_model.config.id2label.items()}
        entailment_index = next(
            (idx for idx, label in label_map.items() if "entail" in label),
            max(label_map.keys()),
        )

        rows: list[list[float]] = []
        hypothesis_count = len(hypotheses)
        for start in range(0, len(premises), self.batch_size):
            chunk = premises[start : start + self.batch_size]
            left: list[str] = []
            right: list[str] = []
            for premise in chunk:
                for hypothesis in hypotheses:
                    left.append(premise)
                    right.append(hypothesis)
            encoded = tokenizer(
                left,
                right,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            target_device = next(nli_model.parameters()).device
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = nli_model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
            flat_scores = [float(score) for score in probs[:, entailment_index].tolist()]
            for offset in range(0, len(flat_scores), hypothesis_count):
                rows.append(flat_scores[offset : offset + hypothesis_count])
        return rows

    def _entailment_scores(self, premise: str, hypotheses: Sequence[str]) -> list[float]:
        rows = self._entailment_scores_many([premise], hypotheses)
        return rows[0] if rows else []

    def classify_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        torch = _import_torch()

        tokenizer, nli_model = self._load_pipeline(self.model, self.device)
        encoded = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        target_device = next(nli_model.parameters()).device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = nli_model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        label_map = {str(label).lower(): float(probs[idx].item()) for idx, label in nli_model.config.id2label.items()}
        return label_map

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        hypothesis_template: str,
        *,
        multi_label: bool = False,
    ) -> dict[str, float]:
        hypotheses = [hypothesis_template.format(label) for label in labels]
        entailment_scores = self._entailment_scores(text, hypotheses)
        if multi_label:
            return {
                str(label): float(score)
                for label, score in zip(labels, entailment_scores)
            }

        score_sum = sum(entailment_scores) or 1.0
        return {
            str(label): float(score / score_sum)
            for label, score in zip(labels, entailment_scores)
        }

    def classify_many(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        hypothesis_template: str,
        *,
        multi_label: bool = False,
    ) -> list[dict[str, float]]:
        if not texts:
            return []

        hypotheses = [hypothesis_template.format(label) for label in labels]
        score_rows = self._entailment_scores_many(texts, hypotheses)
        outputs: list[dict[str, float]] = []
        for entailment_scores in score_rows:
            if multi_label:
                outputs.append(
                    {
                        str(label): float(score)
                        for label, score in zip(labels, entailment_scores)
                    }
                )
                continue

            score_sum = sum(entailment_scores) or 1.0
            outputs.append(
                {
                    str(label): float(score / score_sum)
                    for label, score in zip(labels, entailment_scores)
                }
            )
        return outputs
