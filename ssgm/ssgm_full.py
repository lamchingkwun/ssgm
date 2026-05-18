"""
ssgm_full.py
============
Convenience factory for creating a fully-equipped SSGM engine with all layers:

  Layer 1 — Provenance (embedding k-NN + trusted/malicious source rules)
  Layer 2 — Contradiction Gate (three-way TMS decision)
  Layer 3 — Optional NLI adjudicator
  Layer 4 — Weibull Decay
  Layer 5 — Anchor-grounded Reconciliation

Usage:
    from ssgm.ssgm_full import create_full_engine

    engine = create_full_engine(
        mode='full_ssgm',
        api_key='',                 # optional for API-backed providers
        model='qwen3.5:9b',         # local Ollama or API-backed model
        weibull_eta=2.0,            # Weibull scale parameter
        weibull_kappa=1.0,          # Weibull shape parameter
        stale_threshold=0.3,         # Weibull freshness threshold
        nli_abstention=0.7,         # NLI confidence threshold for abstention
        mcore_min_conf=0.75,         # M_core confidence threshold
    )

When no API key is provided, the factory defaults to local Ollama-style
operation and otherwise uses the configured OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from typing import Optional

from .governor import SSGMEngine, WeibullDecayConfig
from .memory_query import MCoreQueryConfig
from .nli_adjudicator import NLIBasedAdjudicator
from .provenance_detector import ProvenanceDetector
from .llm_judge import get_judge


def create_full_engine(
    mode: str = "full_ssgm",
    api_key: Optional[str] = None,
    model: str = "qwen3.5:9b",
    base_url: Optional[str] = None,
    weibull_eta: float = 2.0,
    weibull_kappa: float = 1.0,
    stale_threshold: float = 0.3,
    nli_abstention: float = 0.4,  # 0.4 recommended for qwen3.5 (lenient model)
    mcore_min_conf: float = 0.75,
    use_weibull: bool = True,
    provenance_k_neighbours: int = 5,
    provenance_anomaly_threshold: float = 0.7,
    judge_backend: str = "compact_nli",
    strict_api_failures: bool = True,
) -> SSGMEngine:
    """
    Create a fully-equipped SSGM engine with all theoretical layers implemented.

    Args:
        mode: SSGM governance mode (default: 'full_ssgm')
        api_key: Optional API key for OpenAI-compatible endpoints. Leave empty
                 for local Ollama-style deployments.
        model: LLM model name for the write judge and optional NLI
               adjudicator (default: 'qwen3.5:9b').
        base_url: API base URL. For Ollama use 'http://localhost:11434/v1'.
                  If None, the factory chooses a local Ollama endpoint when no
                  key is supplied and a hosted OpenAI-compatible endpoint
                  otherwise.
        weibull_eta: Weibull scale parameter η (default: 2.0)
        weibull_kappa: Weibull shape parameter κ (default: 1.0)
        stale_threshold: Weibull relevance threshold below which memory is stale
                        (default: 0.3)
        nli_abstention: Minimum LLM confidence to accept NLI verdict;
                        below this the adjudicator abstains (default: 0.7)
        mcore_min_conf: Minimum confidence for mutable records to qualify
                       as M_core established facts (default: 0.75)
        use_weibull: Use Weibull decay instead of simple threshold for freshness
                     (default: True)
        provenance_k_neighbours: k for embedding k-NN anomaly detection (default: 5)
        provenance_anomaly_threshold: Max avg k-NN cosine distance to qualify
                                     as non-anomalous (default: 0.7)
        judge_backend: Write-judge backend for allow/quarantine/block.
                      Supported by `ssgm.llm_judge.get_judge`; defaults to
                      `compact_nli` so the default path stays offline,
                      reproducible, and independent of heuristic-only judging.
        strict_api_failures: If True, provider initialization or call failures
                     raise instead of silently disabling judge/NLI components.

    Returns:
        SSGMEngine with all layers configured and wired together.
    """
    # Auto-detect provider from base_url and api_key
    if base_url:
        if "localhost" in base_url or "ollama" in base_url.lower():
            provider = "ollama"
        else:
            provider = "openai"
    elif api_key in (None, ""):
        # No API key → use Ollama on localhost
        provider = "ollama"
        base_url = base_url or "http://localhost:11434/v1"
    else:
        provider = "minimax"
        base_url = base_url or "https://api.minimax.chat/v1"

    # Weibull decay config
    weibull_config = WeibullDecayConfig(
        eta=weibull_eta,
        kappa=weibull_kappa,
        threshold=stale_threshold,
    )

    # M_core query config (shared between governor and NLI)
    mcore_config = MCoreQueryConfig(
        min_confidence=mcore_min_conf,
        include_mutable=True,
    )

    # Layer 2: Provenance detector
    provenance_detector = ProvenanceDetector(
        k_neighbours=provenance_k_neighbours,
        anomaly_threshold=provenance_anomaly_threshold,
    )

    # Layer 3: NLI adjudicator (optional)
    nli_adjudicator = None
    resolved_key = api_key or os.getenv("MINIMAX_API_KEY", "")

    if provider == "ollama" or resolved_key:
        try:
            nli_adjudicator = NLIBasedAdjudicator(
                api_key=resolved_key,
                model=model,
                base_url=base_url,
                abstention_threshold=nli_abstention,  # 0.4 recommended for qwen3.5
                strict_api_failures=strict_api_failures,
            )
            if provider == "ollama":
                print(f"[ssgm_full] INFO: Level 3 NLI enabled via Ollama ({model} at {base_url}).")
            else:
                print(f"[ssgm_full] INFO: Level 3 NLI enabled via {provider} ({model}).")
        except Exception as e:
            if strict_api_failures:
                raise RuntimeError("Failed to initialize Level 3 NLI in strict API mode") from e
            print(f"[ssgm_full] WARNING: Failed to initialize Level 3 NLI: {e}")
            print(f"[ssgm_full] Level 3 NLI disabled. INSUFFICIENT_EVIDENCE will defer to reconciliation.")
            nli_adjudicator = None

    llm_judge = get_judge(
        judge_backend,
        model=model if judge_backend in {"ollama", "minimax", "compact_nli", "openai_responses"} else None,
        base_url=base_url if judge_backend in {"ollama", "openai_responses"} else None,
        api_key=resolved_key if judge_backend in {"minimax", "openai_responses"} else None,
        strict_api_failures=strict_api_failures,
    )

    return SSGMEngine(
        mode=mode,
        weibull_config=weibull_config,
        provenance_detector=provenance_detector,
        nli_adjudicator=nli_adjudicator,
        mcore_config=mcore_config,
        use_weibull=use_weibull,
        llm_judge=llm_judge,
    )
