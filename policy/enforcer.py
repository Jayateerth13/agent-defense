"""Runtime policy enforcement (Progent paper Algorithm 2, task spec Section 5).

1. filter the policy to the rules of the called tool            -> ``P_t``
2. ``forbid`` rules are checked first; the first one whose conditions the call satisfies blocks it
3. ``allow`` rules are then checked in ascending priority order (official ``sort_policy``).  The first
   rule that the call satisfies allows it.  A *generated* rule (priority 100, written by the policy LLM)
   that the call violates blocks it immediately - later rules are not consulted - exactly as the official
   ``_check_tool_call`` does.  Human-written rules (lower priority, e.g. always-allowed tools) that do
   not match simply fall through.
4. no rule matches -> block by default

Argument semantics (official ``check_arg`` loop): only arguments that are *present* in the call are
checked.  The runner passes the bound arguments with their defaults applied (as the official wrapper
does), so an omitted optional argument arrives as its default (e.g. ``None``) and is validated against
the restriction.  A key that is truly missing from ``args`` is unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import best_match

from policy.schema import DEFAULT_BLOCK_MSG, DEFAULT_FALLBACK, Policy, Rule

GENERATED_PRIORITY = 100  # priority of rules written by the policy LLM (policy/convert.py)


@dataclass
class Decision:
    allowed: bool
    rule: Rule | None = None  # the rule that decided (None => default block)
    fallback: str | None = None  # terminate_execution | request_user_inspection | return_msg
    message: str | None = None  # text returned to the agent in place of the tool result
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "fallback": self.fallback,
            "rule": self.rule.to_json() if self.rule else None,
        }


def rule_matches(rule: Rule, args: dict[str, Any]) -> bool:
    """True iff the call's arguments satisfy every condition of ``rule`` (conjunction)."""
    for arg, _cond in rule.conditions.items():
        v = rule.validator_for(arg)
        if v is None:  # wildcard
            continue
        if arg not in args:  # key missing => unconstrained (official: ``if arg_name in kwargs``)
            continue
        if not v.is_valid(args[arg]):
            return False
    return True


def ordered_rules(rules: list[Rule]) -> list[Rule]:
    """forbid before allow; within an effect, higher priority first (stable)."""
    return sorted(rules, key=lambda r: (0 if r.effect == "forbid" else 1, -r.priority))


def is_allowed(rules: list[Rule], args: dict[str, Any]) -> bool:
    """Boolean form of the decision for one tool's rules (used by the SMT probe fallback)."""
    for r in ordered_rules(rules):
        if rule_matches(r, args):
            return r.effect == "allow"
    return False


def check_call(policy: Policy, tool: str, args: dict[str, Any]) -> Decision:
    rules = policy.rules_for(tool)
    for rule in rules:  # forbid rules first (Algorithm 2)
        if rule.effect == "forbid" and rule_matches(rule, args):
            msg = rule.fallback_msg or f"Tool call blocked by policy: {tool} is forbidden with these arguments."
            return Decision(False, rule, rule.fallback, msg, "forbidden by rule")
    # allow rules: ascending priority (stable), as the official ``sort_policy``
    allows = sorted((r for r in rules if r.effect == "allow"), key=lambda r: r.priority)
    for rule in allows:
        if rule_matches(rule, args):
            return Decision(True, rule, reason="allowed by rule")
        if rule.priority == GENERATED_PRIORITY:
            # official: a violated generated rule raises immediately (later rules are not tried)
            msg = rule.fallback_msg or _failure_text(rule, args) or DEFAULT_BLOCK_MSG
            return Decision(False, None, rule.fallback, msg, "no matching rule (default block)")
    # No rule matched => block by default (official wording when the tool has no allow rule at all).
    if allows:
        r = allows[-1]  # highest priority
        msg = r.fallback_msg or _failure_text(r, args) or DEFAULT_BLOCK_MSG
        return Decision(False, None, r.fallback, msg, "no matching rule (default block)")
    return Decision(False, None, DEFAULT_FALLBACK, f"The tool '{tool}' is not allowed.", "no matching rule (default block)")


def _failure_text(rule: Rule, args: dict[str, Any]) -> str | None:
    """``str(jsonschema.ValidationError)`` of the first restriction of ``rule`` that ``args`` violates."""
    for arg in rule.conditions:
        v = rule.validator_for(arg)
        if v is None or arg not in args:
            continue
        err = best_match(v.iter_errors(args[arg]))  # what ``jsonschema.validate`` (official) raises
        if err is not None:
            return str(err)
    return None


