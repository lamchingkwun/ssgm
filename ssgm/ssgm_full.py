"""Explicit configurations for Ollama, hosted judges, and compact NLI.

The default uses local Ollama for both write judging and NLI. Model services
and weights must be available before writes are evaluated. For a no-model
smoke test, instantiate SSGMEngine(use_embeddings=False) directly.
"""
from __future__ import annotations

import os
from typing import Optional
from .governor import SSGMEngine, WeibullDecayConfig
from .memory_query import MCoreQueryConfig
from .nli_adjudicator import NLIBasedAdjudicator, CompactNLIBasedAdjudicator
from .llm_judge import get_judge
from .compact_nli_backend import DEFAULT_COMPACT_NLI_MODEL


def create_full_engine(
    mode: str = "full_ssgm",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    weibull_eta: float = 2.0,
    weibull_kappa: float = 1.0,
    stale_threshold: float = 0.3,
    nli_abstention: float = 0.4,
    mcore_min_conf: float = 0.75,
    use_weibull: bool = True,
    provenance_k_neighbours: int = 5,
    provenance_anomaly_threshold: float = 0.7,
    judge_backend: str = "ollama",
    strict_api_failures: bool = True,
    enable_nli: bool = True,
    use_embeddings: bool = True,
    allow_embedding_fallback: bool = False,
    embedding_model: str = "nomic-embed-text-v2-moe",
    embedding_base_url: Optional[str] = None,
) -> SSGMEngine:
    """Build a governed engine with explicitly selected model services.

    Supported backends: ollama, compact_nli, openai_responses, minimax.
    nli_abstention defaults to 0.4. Compact models require the optional
    torch, transformers and sentencepiece packages and may download weights initially.
    Embedding failures raise unless allow_embedding_fallback=True.
    This is a runtime configuration, not the paper's experiment harness.
    """
    defaults = {
        "ollama": ("qwen3.5:9b", "http://localhost:11434/v1", None),
        "compact_nli": (DEFAULT_COMPACT_NLI_MODEL, None, None),
        "openai_responses": ("gpt-5.4", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        "minimax": ("MiniMax-M2.1", "https://api.minimax.chat/v1", "MINIMAX_API_KEY"),
    }
    if judge_backend not in defaults:
        raise ValueError("Select ollama, compact_nli, openai_responses, or minimax explicitly")
    default_model, default_url, key_env = defaults[judge_backend]
    model = model or default_model
    if judge_backend == "compact_nli" and ":" in model:
        raise ValueError("compact_nli requires a Hugging Face sequence-classification model, not an Ollama tag")
    if judge_backend == "openai_responses":
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or default_url
    else:
        base_url = base_url or default_url
    resolved_key = api_key or (os.getenv(key_env) if key_env else None)
    if key_env and not resolved_key:
        raise ValueError(f"{key_env} or an explicit api_key is required")

    judge = get_judge(judge_backend, model=model, base_url=base_url,
                      api_key=resolved_key, strict_api_failures=strict_api_failures)
    nli = None
    if enable_nli:
        if judge_backend == "compact_nli":
            nli = CompactNLIBasedAdjudicator(model=model, abstention_threshold=nli_abstention)
        else:
            nli = NLIBasedAdjudicator(
                api_key=resolved_key, model=model, base_url=base_url,
                abstention_threshold=nli_abstention,
                wire_api="responses" if judge_backend == "openai_responses" else "chat_completions",
                strict_api_failures=strict_api_failures,
            )
    engine = SSGMEngine(
        mode=mode,
        weibull_config=WeibullDecayConfig(eta=weibull_eta, kappa=weibull_kappa,
                                         threshold=stale_threshold),
        nli_adjudicator=nli, llm_judge=judge,
        mcore_config=MCoreQueryConfig(min_confidence=mcore_min_conf, include_mutable=True),
        use_weibull=use_weibull, use_embeddings=use_embeddings,
        allow_embedding_fallback=allow_embedding_fallback,
        embedding_model=embedding_model, embedding_base_url=embedding_base_url,
    )
    engine.provenance_detector.k_neighbours = provenance_k_neighbours
    engine.provenance_detector.anomaly_threshold = provenance_anomaly_threshold
    return engine
