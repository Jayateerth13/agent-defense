"""Progent policy data structures + JSON-Schema validation (task spec Section 5).

A policy is a JSON object mapping a tool name to a list of rules::

    {"send_money": [{"priority": 100, "effect": "allow",
                     "conditions": {"recipient": {"type": "string", "enum": ["DE89..."]}},
                     "fallback": "return_msg", "fallback_msg": "..."}]}

``conditions`` maps argument names to JSON-Schema fragments (or the wildcard ``"*"``).
The rule's conditions are satisfied when *every* listed argument validates against its
fragment (a conjunction).  This is Progent's own encoding (paper Section 7 / Fig. 9) and it
covers the grammar of the task spec:

=====================  ==========================================================
grammar element        JSON-Schema construct
=====================  ==========================================================
literal equality / in  ``const`` / ``enum``
wildcard "*"           ``"*"`` (or ``{}``)
regex match            ``pattern`` (Python ``re.search`` semantics)
comparisons            ``minimum`` / ``maximum`` / ``exclusiveMinimum`` / ``exclusiveMaximum``
array length           ``minItems`` / ``maxItems`` (strings: ``minLength`` / ``maxLength``)
array / field access   ``items`` / ``properties`` (+ ``required``)
=====================  ==========================================================

Argument semantics (as in the official checker): a condition only applies to arguments that are
*present* in the call.  The Progent runner passes the bound arguments with defaults applied, so an
omitted optional argument is validated as its default (e.g. ``None``); a key that is truly missing is
unconstrained.  See ``policy/enforcer.py``.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

FALLBACKS = ("terminate_execution", "request_user_inspection", "return_msg")
DEFAULT_FALLBACK = "return_msg"
DEFAULT_BLOCK_MSG = "No policy rule permits this tool call."
WILDCARD = "*"

#: JSON-Schema keywords a condition may use.  Anything else is rejected so that every
#: accepted policy is both enforceable and (mostly) SMT-checkable.
SUPPORTED_KEYWORDS = frozenset(
    {
        "type", "enum", "const", "pattern", "minLength", "maxLength",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minItems", "maxItems", "items", "properties", "required", "additionalProperties",
        "anyOf", "allOf", "oneOf", "not",
    }
)
#: Purely informational keywords: allowed but ignored.
METADATA_KEYWORDS = frozenset({"description", "title", "$comment", "default", "examples", "format"})


class PolicyError(ValueError):
    """Raised when a candidate policy is malformed."""


# Schema that validates the *shape* of a whole policy document.
RULE_SCHEMA: dict = {
    "type": "object",
    "required": ["effect", "conditions"],
    "additionalProperties": False,
    "properties": {
        "priority": {"type": "integer"},
        "effect": {"enum": ["allow", "forbid"]},
        "conditions": {
            "type": "object",
            "additionalProperties": {"anyOf": [{"const": WILDCARD}, {"type": "object"}]},
        },
        "fallback": {"enum": list(FALLBACKS)},
        "fallback_msg": {"type": "string"},
    },
}
POLICY_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": {"type": "array", "items": RULE_SCHEMA},
}
_POLICY_VALIDATOR = jsonschema.Draft202012Validator(POLICY_SCHEMA)


@dataclass(frozen=True)
class Rule:
    tool: str
    effect: str
    conditions: dict[str, Any]
    priority: int = 0
    fallback: str = DEFAULT_FALLBACK
    fallback_msg: str | None = None
    # one compiled validator per constrained argument (built lazily, not part of equality)
    _validators: dict = field(default_factory=dict, compare=False, repr=False, hash=False)

    def validator_for(self, arg: str) -> jsonschema.Draft202012Validator | None:
        cond = self.conditions[arg]
        if cond == WILDCARD or cond == {}:
            return None
        if arg not in self._validators:
            self._validators[arg] = jsonschema.Draft202012Validator(cond)
        return self._validators[arg]

    def to_json(self) -> dict:
        out: dict[str, Any] = {
            "priority": self.priority,
            "effect": self.effect,
            "conditions": copy.deepcopy(self.conditions),
            "fallback": self.fallback,
        }
        if self.fallback_msg is not None:
            out["fallback_msg"] = self.fallback_msg
        return out


@dataclass
class Policy:
    rules: dict[str, list[Rule]] = field(default_factory=dict)

    def rules_for(self, tool: str) -> list[Rule]:
        return self.rules.get(tool, [])

    def tools(self) -> list[str]:
        return list(self.rules)

    def to_json(self) -> dict:
        return {t: [r.to_json() for r in rs] for t, rs in self.rules.items()}

    def dumps(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_json(), indent=indent, sort_keys=True)

    def __eq__(self, other: object) -> bool:  # semantic equality on the canonical JSON
        return isinstance(other, Policy) and self.to_json() == other.to_json()

    @staticmethod
    def empty() -> "Policy":
        return Policy({})


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _check_condition(tool: str, arg: str, cond: Any, path: str = "") -> None:
    """Recursively check that a condition fragment only uses supported keywords/regexes."""
    if cond == WILDCARD:
        return
    where = f"{tool}.{arg}{path}"
    if not isinstance(cond, dict):
        raise PolicyError(f"{where}: condition must be an object or '*'")
    for kw in cond:
        if kw not in SUPPORTED_KEYWORDS and kw not in METADATA_KEYWORDS:
            raise PolicyError(f"{where}: unsupported JSON-Schema keyword '{kw}'")
    try:
        jsonschema.Draft202012Validator.check_schema(cond)
    except jsonschema.SchemaError as e:
        raise PolicyError(f"{where}: invalid JSON Schema: {e.message}") from e
    if "pattern" in cond:
        try:
            re.compile(cond["pattern"])
        except re.error as e:
            raise PolicyError(f"{where}: invalid regex {cond['pattern']!r}: {e}") from e
    if isinstance(cond.get("items"), dict):
        _check_condition(tool, arg, cond["items"], path + ".items")
    for sub in ("anyOf", "allOf", "oneOf"):
        for i, s in enumerate(cond.get(sub, [])):
            _check_condition(tool, arg, s, f"{path}.{sub}[{i}]")
    if isinstance(cond.get("not"), dict):
        _check_condition(tool, arg, cond["not"], path + ".not")
    for k, s in (cond.get("properties") or {}).items():
        _check_condition(tool, arg, s, f"{path}.properties.{k}")


def parse_policy(
    obj: Any,
    tool_args: dict[str, set[str]] | None = None,
    *,
    drop_unknown_tools: bool = True,
) -> Policy:
    """Validate ``obj`` (a decoded JSON policy) and convert it to a :class:`Policy`.

    ``tool_args`` maps each *available* tool name to its argument names; when given, rules
    for unknown tools are dropped (``drop_unknown_tools``) and conditions on unknown
    arguments are rejected (a typo'd argument name would silently become a wildcard).
    """
    if not isinstance(obj, dict):
        raise PolicyError("policy must be a JSON object mapping tool names to rule lists")
    errors = sorted(_POLICY_VALIDATOR.iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.path) or "<root>"
        raise PolicyError(f"policy schema violation at {loc}: {e.message}")

    rules: dict[str, list[Rule]] = {}
    for tool, rule_list in obj.items():
        if tool_args is not None and tool not in tool_args:
            if drop_unknown_tools:
                continue
            raise PolicyError(f"unknown tool '{tool}'")
        parsed = []
        for r in rule_list:
            for arg, cond in r["conditions"].items():
                if tool_args is not None and arg not in tool_args[tool]:
                    raise PolicyError(f"{tool}: condition on unknown argument '{arg}'")
                _check_condition(tool, arg, cond)
            parsed.append(
                Rule(
                    tool=tool,
                    effect=r["effect"],
                    conditions=copy.deepcopy(r["conditions"]),
                    priority=r.get("priority", 0),
                    fallback=r.get("fallback", DEFAULT_FALLBACK),
                    fallback_msg=r.get("fallback_msg"),
                )
            )
        if parsed:
            rules[tool] = parsed
    return Policy(rules)


# ---------------------------------------------------------------------------
# helpers for turning AgentDojo tools into (a) prompt text and (b) arg-type maps
# ---------------------------------------------------------------------------
def _simplify_param_schema(s: dict) -> dict:
    s = {k: v for k, v in s.items() if k not in ("title",)}
    if "anyOf" in s:  # Optional[...] => [{"type": X}, {"type": "null"}]
        non_null = [x for x in s["anyOf"] if x.get("type") != "null"]
        if len(non_null) == 1:
            merged = {**{k: v for k, v in s.items() if k != "anyOf"}, **_simplify_param_schema(non_null[0])}
            merged["optional"] = True
            return merged
    return s


def tool_specs(runtime_functions: dict) -> list[dict]:
    """Compact, LLM-friendly description of the tools in an AgentDojo ``FunctionsRuntime``."""
    specs = []
    for name, fn in runtime_functions.items():
        js = fn.parameters.model_json_schema()
        props = {a: _simplify_param_schema(p) for a, p in js.get("properties", {}).items()}
        specs.append(
            {"name": name, "description": fn.description, "arguments": props, "required": js.get("required", [])}
        )
    return specs


def tool_arg_names(specs: list[dict]) -> dict[str, set[str]]:
    return {s["name"]: set(s["arguments"]) for s in specs}


def tool_arg_types(specs: list[dict]) -> dict[str, dict[str, dict]]:
    """``{tool: {arg: {"type": ..., "items_type": ...}}}`` used by the SMT encoder."""
    out: dict[str, dict[str, dict]] = {}
    for s in specs:
        out[s["name"]] = {}
        for a, p in s["arguments"].items():
            info = {"type": p.get("type")}
            if isinstance(p.get("items"), dict):
                info["items_type"] = p["items"].get("type")
            out[s["name"]][a] = info
    return out
