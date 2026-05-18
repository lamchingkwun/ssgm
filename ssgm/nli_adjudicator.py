"""
nli_adjudicator.py
==================
Level 3 LLM-based NLI Contradiction Adjudicator for SSGM.

Implements the calibrated abstention NLI check described in the SSGM paper
(Principle 1, cascade stage iii):

    "calibrated NLI with abstention: the adjudicator abstains
     when its confidence falls below a threshold, deferring to human review."

Given a new record and the established M_core facts from the episodic ledger,
this module asks an LLM:
    "Does 'new_record' contradict any of the established facts?"

Response interpretation:
    "contradiction" → hard block
    "consistent"   → allow
    "uncertain"     → abstain (defer to human / reconciliation)

Requirements:
    pip install requests

Usage:
    adjudicator = NLIBasedAdjudicator(
        api_key="your-minimax-api-key",
        model="MiniMax-Text-01",
        abstention_threshold=0.7,  # abstain if confidence < 0.7
    )
    decision = adjudicator.adjudicate(
        new_record=MemoryRecord(...),
        m_core_facts=[MemoryRecord(...), ...],  # established facts
    )
    # decision.action: 'contradiction' | 'consistent' | 'abstain'
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

from .api_provider import (
    extract_responses_text,
    openai_responses_payload,
    retry_api_request,
    resolve_openai_api_key,
    resolve_openai_base_url,
    strict_api_failures_enabled,
)
from .compact_nli_backend import CompactZeroShotBackend, DEFAULT_COMPACT_NLI_MODEL


# ---------------------------------------------------------------------------
# NLI Decision
# ---------------------------------------------------------------------------

@dataclass
class NLIDecision:
    action: str          # 'contradiction' | 'consistent' | 'abstain'
    confidence: float    # LLM's confidence in its answer [0, 1]
    reasoning: str      # brief LLM explanation
    latency_seconds: float


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_nli_prompt(new_record, m_core_facts: List) -> str:
    """Build the NLI prompt given the new record and M_core established facts."""
    fact_lines = []
    for rec in m_core_facts:
        fact_lines.append(f'  - "{rec.value}" (source: {rec.source}, conf: {rec.confidence:.2f})')

    if not fact_lines:
        established_str = "  (no established facts found)"
    else:
        established_str = "\n".join(fact_lines)

    return NLI_USER_PROMPT_TEMPLATE.format(
        established_facts=established_str,
        new_key=new_record.key,
        new_value=new_record.value,
        new_source=new_record.source,
        new_confidence=f"{new_record.confidence:.2f}",
    )


# ---------------------------------------------------------------------------
# NLIBasedAdjudicator
# ---------------------------------------------------------------------------

class NLIBasedAdjudicator:
    """
    Level 3 LLM-based NLI contradiction adjudicator.

    Uses MiniMax API (or any OpenAI-compatible API) to perform
    calibrated NLI checking with abstention.

    Parameters:
        api_key: MiniMax API key. Can also be set via MINIMAX_API_KEY env var.
        model: Model name (default: MiniMax-Text-01)
        base_url: API base URL (default: https://api.minimax.chat/v1)
        abstention_threshold: minimum LLM confidence to accept verdict;
                              below this → abstain (default: 0.7)
        timeout: request timeout in seconds (default: 15)
        use_cache: cache NLI results for identical record+facts pairs (default: True)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "MiniMax-Text-01",
        base_url: str = "https://api.minimax.chat/v1",
        abstention_threshold: float = 0.7,
        timeout: int = 60,
        use_cache: bool = True,
        # Ollama-specific options
        disable_thinking: bool = True,  # try to disable thinking for cleaner output
        wire_api: str | None = None,
        strict_api_failures: bool | None = None,
    ) -> None:
        # Determine provider type from base_url or model name
        if "localhost" in base_url or "ollama" in base_url.lower():
            # Ollama — no API key needed
            self.api_key = ""
            self.provider = "ollama"
        else:
            self.provider = "openai_responses" if (wire_api == "responses" or "minimax" not in base_url) else "minimax"
            if self.provider == "minimax":
                self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "MiniMax API key must be provided or set via MINIMAX_API_KEY env var. "
                        "Get your key at: https://platform.minimax.chat"
                    )
            else:
                self.api_key = resolve_openai_api_key(api_key)
                base_url = resolve_openai_base_url(base_url)

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.abstention_threshold = abstention_threshold
        self.timeout = timeout
        self.use_cache = use_cache
        self.disable_thinking = disable_thinking
        self.wire_api = wire_api or ("responses" if self.provider == "openai_responses" else "chat_completions")
        self.strict_api_failures = strict_api_failures_enabled(strict_api_failures)
        self._cache: dict[str, NLIDecision] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adjudicate(
        self,
        new_record,
        m_core_facts: List,
    ) -> NLIDecision:
        """
        Perform Level 3 NLI adjudication.

        Args:
            new_record: MemoryRecord being evaluated for write
            m_core_facts: List of established MemoryRecords from the episodic ledger

        Returns:
            NLIDecision with action, confidence, reasoning, and latency
        """
        cache_key = self._cache_key(new_record, m_core_facts)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = _build_nli_prompt(new_record, m_core_facts)

        start = time.perf_counter()
        try:
            raw = self._call_llm(prompt)
            decision = self._parse_response(raw, start)
        except Exception as e:
            mode = " in strict mode" if self.strict_api_failures else ""
            raise RuntimeError(f"NLI adjudication failed{mode}") from e

        if self.use_cache:
            self._cache[cache_key] = decision

        return decision

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Make a single chat completion API call.
        
        Supports: MiniMax, Ollama (localhost), and any OpenAI-compatible endpoint.
        Automatically handles Ollama's thinking/reasoning field for qwen3.5 and similar models.
        """
        headers = {"Content-Type": "application/json"}

        # Build messages
        system_msg = (
            "You are a precise factual consistency checker. "
            "Follow the output format exactly. Be concise."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        # Ollama with thinking models (qwen3.5 etc.): use large max_tokens
        # and tell model to put structured answer AFTER thinking
        if self.provider == "ollama":
            if self.disable_thinking:
                # Instruct model to NOT use thinking mode
                messages[0]["content"] += (
                    " IMPORTANT: Do NOT use thinking/reasoning mode. "
                    "Provide the answer directly in your first response. No preamble."
                )
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.0,
            }
            def request_once():
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp

            resp = retry_api_request(request_once)
        elif self.provider == "openai_responses":
            if self.disable_thinking:
                system_msg += (
                    " Do not provide chain-of-thought. "
                    "Return only the requested final structured fields."
                )
            payload = openai_responses_payload(
                model=self.model,
                instructions=system_msg,
                input_text=prompt,
                max_output_tokens=400,
            )
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Accept"] = "application/json"
            headers["User-Agent"] = "SSGM-eval-harness"
            def request_once():
                resp = requests.post(
                    f"{self.base_url}/responses",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp

            resp = retry_api_request(request_once)
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.0,
            }
            headers["Authorization"] = f"Bearer {self.api_key}"
            def request_once():
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp

            resp = retry_api_request(request_once)

        if resp.status_code == 429:
            raise RuntimeError(
                "MiniMax API rate limit or insufficient balance. "
                "Check your account credits at https://platform.minimax.chat"
            )

        if resp.status_code != 200:
            raise RuntimeError(f"MiniMax API error {resp.status_code}: {resp.text}")

        data = resp.json()
        if self.provider == "openai_responses":
            return extract_responses_text(data)
        message = data["choices"][0]["message"]

        # Handle Ollama's thinking/reasoning field (qwen3.5 etc.)
        # Priority: content > reasoning > empty
        raw_text = message.get("content", "")
        reasoning = message.get("reasoning", "")

        if not raw_text.strip() and reasoning.strip():
            # Ollama with thinking enabled: content empty, reasoning has the text
            # Use reasoning as the response text
            raw_text = reasoning

        return raw_text

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, start: float) -> NLIDecision:
        """
        Parse LLM response into NLIDecision.
        
        Smart extraction: looks for structured format (CONFIDENCE:/VERDICT:/REASONING:)
        in the response. For thinking models (qwen3.5), the structured format appears
        at the END of the reasoning text. For non-thinking models, it appears at the start.
        
        Strategy:
          1. Scan ALL lines for structured format fields
          2. Prefer lines near the END of the response (qwen3.5 pattern)
          3. Only use lines that clearly belong to the structured format, not narrative text
        """
        content_text = raw.strip()
        lines = content_text.split('\n')

        # Collect ALL structured lines from the entire response
        # Key → (value, line_number_from_end)
        all_structured = {}  # key → (value, distance_from_end)

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                continue
            lower_line = line.lower()
            # Only consider lines that are clearly structured format markers
            if not any(k in lower_line for k in ['confidence', 'verdict', 'reasoning']):
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip().lower()
            val = parts[1].strip()
            # Use distance from end (smaller = closer to end = more likely to be the actual structured answer)
            distance_from_end = len(lines) - 1 - idx
            if key not in all_structured or distance_from_end < all_structured[key][1]:
                all_structured[key] = (val, distance_from_end)

        # If no structured format found, abstain
        if not all_structured or 'verdict' not in all_structured:
            return NLIDecision(
                action="abstain",
                confidence=0.0,
                reasoning=content_text[:500],
                latency_seconds=time.perf_counter() - start,
            )

        # Parse confidence — prefer lines near the end
        conf_str = all_structured.get('confidence', ('0.5', 999))[0]
        try:
            confidence = float(conf_str.split()[0])
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, IndexError):
            confidence = 0.5

        # Parse verdict — prefer lines near the end
        verdict_raw = all_structured.get('verdict', ('abstain', 999))[0].lower()
        if 'contradiction' in verdict_raw:
            action = 'contradiction'
        elif 'consistent' in verdict_raw:
            action = 'consistent'
        else:
            action = 'abstain'

        # Apply abstention threshold
        # Low confidence + not contradiction → treat as INSUFFICIENT_EVIDENCE
        # (defer to reconciliation rather than blocking or allowing)
        if action != 'contradiction' and confidence < self.abstention_threshold:
            action = 'abstain'

        # Reasoning — prefer near end
        reasoning_raw = all_structured.get('reasoning', (content_text[:200], 999))[0]

        return NLIDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning_raw,
            latency_seconds=time.perf_counter() - start,
        )

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(new_record, m_core_facts: List) -> str:
        """Build a cache key for (new_record, m_core_facts)."""
        facts_sig = "|".join(
            f"{r.key}={r.value}@{r.version}" for r in sorted(m_core_facts, key=lambda x: x.key)
        )
        return f"{new_record.key}::{new_record.value}::{facts_sig}"


class CompactNLIBasedAdjudicator:
    """Compact supervised NLI adjudicator using a local zero-shot classifier."""

    LABELS = [
        "contradicts the established facts",
        "is consistent with the established facts",
        "is uncertain relative to the established facts",
    ]
    HYPOTHESIS_TEMPLATE = "The new record {}."

    def __init__(
        self,
        model: str = DEFAULT_COMPACT_NLI_MODEL,
        *,
        backend: CompactZeroShotBackend | None = None,
        abstention_threshold: float = 0.7,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.backend = backend or CompactZeroShotBackend(model=model)
        self.abstention_threshold = abstention_threshold
        self.use_cache = use_cache
        self._cache: dict[str, NLIDecision] = {}

    @staticmethod
    def _premise(new_record, m_core_facts: List) -> str:
        fact_lines = [
            f"- {rec.key}: {rec.value} (source={rec.source}, conf={rec.confidence:.2f})"
            for rec in m_core_facts
        ]
        established = "\n".join(fact_lines) if fact_lines else "- no established facts"
        return (
            f"Established facts:\n{established}\n"
            f"New record key: {new_record.key}\n"
            f"New record value: {new_record.value}\n"
            f"New record source: {new_record.source}\n"
            f"New record confidence: {new_record.confidence:.2f}"
        )

    @staticmethod
    def _cache_key(new_record, m_core_facts: List) -> str:
        facts_sig = "|".join(
            f"{r.key}={r.value}@{r.version}" for r in sorted(m_core_facts, key=lambda x: x.key)
        )
        return f"{new_record.key}::{new_record.value}::{facts_sig}"

    def adjudicate(self, new_record, m_core_facts: List) -> NLIDecision:
        cache_key = self._cache_key(new_record, m_core_facts)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        start = time.perf_counter()
        premise = "\n".join(
            f"{rec.key}: {rec.value}"
            for rec in m_core_facts
        ) or "no established facts"
        scores = self.backend.classify_pair(premise, new_record.value)
        contradiction_score = float(scores.get("contradiction", 0.0))
        entailment_score = float(scores.get("entailment", 0.0))
        neutral_score = float(scores.get("neutral", 0.0))
        best_label = max(
            {
                "contradiction": contradiction_score,
                "entailment": entailment_score,
                "neutral": neutral_score,
            },
            key=lambda label: {
                "contradiction": contradiction_score,
                "entailment": entailment_score,
                "neutral": neutral_score,
            }[label],
        )
        confidence = {
            "contradiction": contradiction_score,
            "entailment": entailment_score,
            "neutral": neutral_score,
        }[best_label]

        if best_label == "contradiction":
            action = "contradiction"
        elif best_label == "entailment" and confidence >= self.abstention_threshold:
            action = "consistent"
        else:
            action = "abstain"

        decision = NLIDecision(
            action=action,
            confidence=confidence,
            reasoning=f"compact_nli_pair:{best_label}",
            latency_seconds=time.perf_counter() - start,
        )
        if self.use_cache:
            self._cache[cache_key] = decision
        return decision
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NLI_USER_PROMPT_TEMPLATE = (PROMPTS_DIR / "nli_user_prompt_template.md").read_text(encoding="utf-8").strip()
