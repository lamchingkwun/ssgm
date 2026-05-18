from __future__ import annotations

import os
import time
from typing import Any, Callable, TypeVar


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
API_REQUEST_MAX_RETRIES = 10
API_REQUEST_RETRY_BASE_DELAY_SECONDS = 1.0
API_REQUEST_RETRY_MAX_DELAY_SECONDS = 10.0
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429}

T = TypeVar("T")


def resolve_openai_base_url(configured: str | None = None) -> str:
    if configured:
        return configured.rstrip("/")
    env_value = os.getenv("OPENAI_BASE_URL")
    if env_value:
        return env_value.rstrip("/")
    return DEFAULT_OPENAI_BASE_URL


def resolve_openai_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env_value = os.getenv("OPENAI_API_KEY")
    if env_value:
        return env_value
    raise RuntimeError("OPENAI_API_KEY not set")


def resolve_openai_model(configured: str | None = None, default: str = "gpt-5.4") -> str:
    if configured:
        return configured
    env_value = os.getenv("OPENAI_MODEL")
    if env_value:
        return env_value
    return default


def strict_api_failures_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    value = os.getenv("SSGM_STRICT_API_FAILURES", "")
    return value.lower() in {"1", "true", "yes", "on"}


def is_retryable_api_request_error(exc: Exception) -> bool:
    """Return whether an exception represents a transient API request failure."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code in RETRYABLE_HTTP_STATUS_CODES or status_code >= 500

    try:
        import httpx
    except Exception:  # pragma: no cover - optional dependency guard
        httpx = None
    if httpx is not None and isinstance(exc, httpx.RequestError):
        return True

    try:
        import requests
    except Exception:  # pragma: no cover - optional dependency guard
        requests = None
    if requests is not None and isinstance(exc, requests.RequestException):
        return True

    return False


def retry_api_request(
    operation: Callable[[], T],
    *,
    max_retries: int = API_REQUEST_MAX_RETRIES,
    base_delay_seconds: float = API_REQUEST_RETRY_BASE_DELAY_SECONDS,
    max_delay_seconds: float = API_REQUEST_RETRY_MAX_DELAY_SECONDS,
    retry_on_exception: Callable[[Exception], bool] | None = None,
) -> T:
    """Run an API request with bounded retries for transient request failures.

    ``max_retries`` counts retries after the initial attempt. Non-request
    failures such as malformed JSON or invalid model output are not retried.
    """
    for failures in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            should_retry = (
                retry_on_exception(exc)
                if retry_on_exception is not None
                else is_retryable_api_request_error(exc)
            )
            if failures >= max_retries or not should_retry:
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** failures))
            time.sleep(delay)

    raise RuntimeError("api_request_failed")


def responses_wire_metadata(base_url: str | None = None) -> dict[str, Any]:
    return {
        "provider_config": "environment",
        "wire_api": "responses",
        "base_url": resolve_openai_base_url(base_url),
    }


def extract_responses_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("content")
                if text:
                    parts.append(str(text))
            elif isinstance(content, str):
                parts.append(content)
    return "".join(parts).strip()


def openai_responses_payload(
    *,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        "store": False,
    }
