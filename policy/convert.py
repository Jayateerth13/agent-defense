"""Bridge between the *official Progent policy wire format* and our internal :class:`Policy`.

The official implementation (sunblaze-ucb/progent, ``secagent/tool.py``) exchanges policies with the
policy LLM as a flat JSON array::

    [{"name": "send_money", "args": {"recipient": {"type": "string", "enum": ["UK123..."]}}}, ...]

Every entry is an ``allow`` rule of priority 100 with fallback ``return_msg``; arguments that are not
mentioned are unconstrained.  In addition the *benchmark integration* installs human-written
always-allowed tools at priority 1 (``update_always_allowed_tools``) which survive every policy
update (an update only replaces the priority-100 rules).

Internally we keep our richer :class:`Policy` (needed by the Z3 narrowing/expansion check and the
audit log); this module converts both ways so that the LLM-facing behaviour matches the original.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import jsonschema

from policy.schema import DEFAULT_FALLBACK, Policy, PolicyError, Rule

GENERATED_PRIORITY = 100  # rules written by the policy LLM
ALWAYS_ALLOWED_PRIORITY = 1  # human-written "always allowed" tools


def _condition_from_restriction(tool: str, arg: str, restriction: Any) -> dict:
    """Official ``check_arg`` accepts a JSON-Schema dict, or a string treated as a regex (``re.match``)."""
    where = f"{tool}.{arg}"
    if isinstance(restriction, dict):
        try:
            jsonschema.Draft202012Validator.check_schema(restriction)
        except jsonschema.SchemaError as e:
            raise PolicyError(f"{where}: invalid JSON Schema: {e.message}") from e
        if "pattern" in restriction:
            try:
                re.compile(restriction["pattern"])
            except (re.error, TypeError) as e:
                raise PolicyError(f"{where}: invalid regex {restriction['pattern']!r}: {e}") from e
        return restriction
    if isinstance(restriction, str):
        try:
            re.compile(restriction)
        except re.error as e:
            raise PolicyError(f"{where}: invalid regex {restriction!r}: {e}") from e
        # re.match semantics (anchored at the start) expressed with jsonschema's re.search
        return {"type": "string", "pattern": restriction if restriction.startswith("^") else f"^(?:{restriction})"}
    raise PolicyError(f"{where}: restriction must be a JSON-Schema object or a regex string")


def array_to_policy(arr: Any, tool_args: dict[str, set[str]]) -> Policy:
    """Official wire format -> :class:`Policy` (generated rules only, priority 100).

    * entries for tools that do not exist are ignored (they can never be called);
    * restrictions on arguments the tool does not have are ignored (the official checker only looks
      at arguments that are present in the call, so they never apply).
    """
    if not isinstance(arr, list):
        raise PolicyError("policy must be a JSON array of {\"name\": tool, \"args\": restrictions}")
    rules: dict[str, list[Rule]] = {}
    for item in arr:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("args"), dict):
            raise PolicyError(f"bad policy entry (need name + args object): {str(item)[:120]}")
        tool = item["name"]
        if tool not in tool_args:
            continue
        conditions = {
            arg: _condition_from_restriction(tool, arg, r) for arg, r in item["args"].items() if arg in tool_args[tool]
        }
        rules.setdefault(tool, []).append(
            Rule(tool=tool, effect="allow", conditions=conditions, priority=GENERATED_PRIORITY, fallback=DEFAULT_FALLBACK)
        )
    return Policy(rules)


def policy_to_array(policy: Policy) -> list[dict]:
    """Generated (priority-100) part of ``policy`` in the official wire format (``CURRENT_RESTRICTIONS``)."""
    return [
        {"name": tool, "args": dict(r.conditions)}
        for tool, rules in policy.rules.items()
        for r in rules
        if r.priority == GENERATED_PRIORITY
    ]


def always_allowed_tools(
    tools: list[dict], names: Iterable[str] = (), allow_all_no_arg_tools: bool = False
) -> set[str]:
    """Mirror of ``update_always_allowed_tools``; ``tools`` is the LLM-facing tool list (``args`` = properties)."""
    out = set(names)
    if allow_all_no_arg_tools:
        out.update(t["name"] for t in tools if len(t["args"]) == 0)
    return out


def with_always_allowed(generated: Policy, always: Iterable[str]) -> Policy:
    """Generated rules + a priority-1 unconditional ``allow`` rule for every always-allowed tool."""
    rules = {t: list(rs) for t, rs in generated.rules.items()}
    for tool in sorted(always):
        rules.setdefault(tool, []).append(
            Rule(tool=tool, effect="allow", conditions={}, priority=ALWAYS_ALLOWED_PRIORITY, fallback=DEFAULT_FALLBACK)
        )
    return Policy(rules)
