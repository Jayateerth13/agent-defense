"""Origin check for payment recipients (AgentDojo Banking), used on top of Progent.

Rule: a payment call (``send_money``, ``schedule_transaction``, ``update_scheduled_transaction``) may only
use a ``recipient`` that is on the run's *trusted list*:

* account numbers written in the user's request, and
* account numbers from **structured** fields of tool results that the bank system fills in:
  ``sender`` / ``recipient`` of (scheduled) transactions and the user's own IBAN (``get_iban``).

Account numbers that appear only in free text (a transaction ``subject``, the contents of a file read with
``read_file``, ...) are never trusted.  An empty recipient (e.g. updating only the amount of a scheduled
transaction) is not checked.  No LLM is involved: the rule is plain code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

PAYMENT_TOOLS = ("send_money", "schedule_transaction", "update_scheduled_transaction")
#: tools whose results are lists/dicts of bank records; only these fields are trusted
STRUCTURED_FIELDS = {
    "get_most_recent_transactions": ("sender", "recipient"),
    "get_scheduled_transactions": ("sender", "recipient"),
}
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


def account_numbers(text: str) -> set[str]:
    return {m.upper() for m in IBAN_RE.findall(text or "")}


def _norm(v: Any) -> str:
    return str(v).strip().upper()


@dataclass
class Decision:
    allowed: bool
    reason: str


@dataclass
class OriginGate:
    """Per-run state: the trusted list.  Feed it the request, then every tool result in order."""

    trusted: set[str] = field(default_factory=set)

    @classmethod
    def for_request(cls, user_request: str) -> "OriginGate":
        return cls(set(account_numbers(user_request)))

    def absorb(self, tool: str, result_text: str, failed: bool = False) -> None:
        """Add the structured account numbers of one tool result (free text is ignored)."""
        if failed or not result_text:
            return
        if tool == "get_iban":
            self.trusted |= account_numbers(result_text)
            return
        fields = STRUCTURED_FIELDS.get(tool)
        if not fields:
            return  # read_file, get_user_info, ...: nothing trusted comes from here
        try:
            data = yaml.safe_load(result_text)
        except yaml.YAMLError:
            return
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict):
                for f in fields:
                    if it.get(f) is not None:
                        self.trusted |= account_numbers(_norm(it[f]))

    def check(self, tool: str, args: dict) -> Decision:
        if tool not in PAYMENT_TOOLS:
            return Decision(True, "not a payment tool")
        rec = args.get("recipient")
        if rec is None or str(rec).strip() == "":
            return Decision(True, "no recipient")
        if _norm(rec) in self.trusted:
            return Decision(True, "recipient on trusted list")
        return Decision(False, f"recipient {rec} does not come from the request or a trusted record field")
