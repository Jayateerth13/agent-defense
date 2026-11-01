"""Thin, retrying wrapper around the OpenAI-compatible chat API.

Used by (a) the agent LLM element and (b) the Progent policy LLM calls.  Both use
the same model (``config.MODEL``).
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

import config

_client: openai.OpenAI | None = None
_lock = threading.Lock()

#: number of LLM calls by tag ("agent" / "policy"), for cost reporting.
CALL_COUNTS: Counter = Counter()
#: token usage of this process (for cost reporting).
TOKENS: Counter = Counter()


class MissingAPIKeyError(RuntimeError):
    pass


class QuotaExhaustedError(RuntimeError):
    """Daily (per-day) API quota used up - retrying within this session is pointless.

    The eval scripts stop cleanly on this error and do NOT log the interrupted run, so simply
    re-running them the next day resumes where they stopped (Gemini free tier: ~20 requests/day/model).
    """


def _is_daily_quota(e: Exception) -> bool:
    s = str(e)
    # Gemini: per-day quota.  OpenAI: ``insufficient_quota`` (billing cap / no credit) - retrying is pointless.
    return ("PerDay" in s or "per day" in s.lower() or "insufficient_quota" in s
            or "exceeded your current quota" in s)


def get_client() -> openai.OpenAI:
    global _client
    with _lock:
        if _client is None:
            key, base_url = config.api_settings()
            if not key:
                raise MissingAPIKeyError(
                    f"No API key found for model '{config.MODEL}'. Put OPENAI_API_KEY or GOOGLE_API_KEY in .env "
                    "(see .env.example)."
                )
            _client = openai.OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=120)
        return _client


_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
def chat_completion(messages: list[dict], *, tools: list | None = None, json_mode: bool = False,
                    temperature: float = 0.0, tag: str = "agent", model: str | None = None,
                    seed: int | None = None) -> Any:
    kwargs: dict[str, Any] = dict(model=model or config.MODEL, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if seed is not None and config.provider_for(kwargs["model"]) == "openai":
        kwargs["seed"] = seed  # the official policy calls use seed=0
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    CALL_COUNTS[tag] += 1
    try:
        resp = get_client().chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u:
            with _lock:
                TOKENS["input"] += u.prompt_tokens or 0
                TOKENS["output"] += u.completion_tokens or 0
                TOKENS[f"{tag}_input"] += u.prompt_tokens or 0  # split by caller ("agent" / "policy") for costing
                TOKENS[f"{tag}_output"] += u.completion_tokens or 0
        return resp
    except openai.RateLimitError as e:
        if _is_daily_quota(e):
            raise QuotaExhaustedError(str(e)[:300]) from e
        raise


def complete(system: str, user: str, *, json_mode: bool = True, tag: str = "policy", temperature: float = 0.0,
             seed: int | None = None, model: str | None = None) -> str:
    """Single-turn text completion used by the policy generator."""
    resp = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=json_mode,
        tag=tag,
        temperature=temperature,
        seed=seed,
        model=model or config.POLICY_MODEL,
    )
    return resp.choices[0].message.content or ""


