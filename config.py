"""Central configuration for the Progent reproduction.

The *single* place where the model is chosen (Section 2 of the task spec):
set ``PROGENT_MODEL`` in ``.env`` (or the shell) to change the model used for
BOTH the agent LLM and the policy LLM.

Both supported providers are reached through the OpenAI-compatible chat
completions API, so one code path serves them:

* ``gpt-*``/anything else   -> OpenAI                (OPENAI_API_KEY)
* ``gemini-*``              -> Google AI Studio      (GOOGLE_API_KEY or GEMINI_API_KEY)
                               via its OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _default_model() -> str:
    """Pick a model: explicit env var wins; otherwise whichever API key is present."""
    if os.getenv("PROGENT_MODEL"):
        return os.environ["PROGENT_MODEL"]
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        # gemini-2.5-flash returns 404 ("no longer available to new users") for new AI-Studio keys
        return "gemini-3.8-flash"
    return "gpt-4o-mini"


# ---------------------------------------------------------------------------
# THE model constant (agent LLM == policy LLM).  Change it only here / in .env.
# ---------------------------------------------------------------------------
MODEL: str = _default_model()

# Optional: reasoning effort passed to the API ("none"/"low"/"medium"/"high").
# Useful to keep gemini-2.5-flash cheap/fast.  Unset by default.
REASONING_EFFORT: str | None = os.getenv("PROGENT_REASONING_EFFORT") or None

# Policy LLM.  The official code has a separate SECAGENT_POLICY_MODEL (gpt-4o in its run.sh); by default we
# use the same model as the agent (task spec Section 2) but it can be overridden with PROGENT_POLICY_MODEL.
POLICY_MODEL: str = os.getenv("PROGENT_POLICY_MODEL") or MODEL

# --- AgentDojo ---------------------------------------------------------------
BENCHMARK_VERSION = "v1.2.2"  # latest AgentDojo task definitions (Banking: 16 user / 9 injection tasks)
SUITE_NAME = "banking"
ATTACK_NAME = "important_instructions"  # the only attack technique used

# --- Progent ----------------------------------------------------------------
# Architecture follows the official implementation (sunblaze-ucb/progent, secagent/tool.py):
#  * two-step policy updates (decide without seeing the result, then generate) - the only mode;
#  * update errors are ignored (current policy kept), as in the official run.sh;
#  * human-written always-allowed tools per suite (update_always_allowed_tools in the official
#    banking task_suite.py): get_most_recent_transactions + every tool without arguments.
# The official SECAGENT_ONLY_ALLOW_NARROW option (discard expansions) is off by default.
ONLY_ALLOW_NARROW: bool = os.getenv("PROGENT_ONLY_ALLOW_NARROW", "False").lower() == "true"
ALWAYS_ALLOWED: dict[str, tuple[list[str], bool]] = {
    "banking": (["get_most_recent_transactions"], True),  # (tool names, also allow all no-argument tools)
}
# --- Paths -------------------------------------------------------------------
RESULTS_DIR = ROOT / "results"
RAW_LOGS_PATH = RESULTS_DIR / "raw_logs.jsonl"
RAW_LOGS_ASB_PATH = RESULTS_DIR / "raw_logs_asb.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.md"


def provider_for(model: str) -> str:
    return "gemini" if model.lower().startswith("gemini") else "openai"


def api_settings(model: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(api_key, base_url)`` for the configured model."""
    model = model or MODEL
    base_url = os.getenv("PROGENT_BASE_URL")
    if provider_for(model) == "gemini":
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return key, base_url or GEMINI_OPENAI_BASE_URL
    return os.getenv("OPENAI_API_KEY"), base_url


