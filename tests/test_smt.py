"""Unit tests for the Z3 narrowing/expansion check, including a regex-translation cross-check."""

import random
import re

import pytest
import z3

from policy.schema import parse_policy
from policy.smt_check import Unsupported, _RegexTranslator, compare_policies

TOOLS = {"send_money": {"recipient", "amount", "subject"}, "tags": {"names"}}
TYPES = {
    "send_money": {"recipient": {"type": "string"}, "amount": {"type": "number"}, "subject": {"type": "string"}},
    "tags": {"names": {"type": "array", "items_type": "string"}},
}


def P(obj):
    return parse_policy(obj, TOOLS)


def allow(**conds):
    return {"send_money": [{"effect": "allow", "conditions": conds}]}


def verdict(old, new):
    d = compare_policies(P(old), P(new), TYPES)
    return d.kind, d


def test_enum_subset_is_narrowing():
    assert verdict(allow(recipient={"enum": ["A", "B"]}), allow(recipient={"enum": ["A"]}))[0] == "narrowing"


def test_enum_superset_is_expansion_with_witness():
    kind, d = verdict(allow(recipient={"enum": ["A"]}), allow(recipient={"enum": ["A", "EVIL"]}))
    assert kind == "expansion"
    assert d.checks[0].method == "z3" and d.checks[0].witness["recipient"] == "EVIL"


def test_wildcard_over_enum_is_expansion_and_reverse_is_narrowing():
    assert verdict(allow(recipient={"enum": ["A"]}), allow(recipient="*"))[0] == "expansion"
    assert verdict(allow(recipient="*"), allow(recipient={"enum": ["A"]}))[0] == "narrowing"


def test_new_tool_is_expansion_removed_tool_is_narrowing():
    assert verdict({}, allow(recipient="*"))[0] == "expansion"
    assert verdict(allow(recipient="*"), {})[0] == "narrowing"


def test_numeric_bounds():
    assert verdict(allow(amount={"maximum": 100}), allow(amount={"maximum": 50}))[0] == "narrowing"
    assert verdict(allow(amount={"maximum": 100}), allow(amount={"maximum": 100.5}))[0] == "expansion"
    assert verdict(allow(amount={"maximum": 100}), allow(amount={"exclusiveMaximum": 100}))[0] == "narrowing"


def test_adding_forbid_rule_is_narrowing_and_removing_it_is_expansion():
    base = allow(recipient="*")
    with_forbid = {"send_money": base["send_money"] + [{"effect": "forbid", "conditions": {"amount": {"minimum": 10}}}]}
    assert verdict(base, with_forbid)[0] == "narrowing"
    assert verdict(with_forbid, base)[0] == "expansion"


def test_regex_narrowing_and_expansion():
    assert verdict(allow(recipient={"pattern": "^US[0-9]+$"}), allow(recipient={"pattern": "^US1[0-9]+$"}))[0] == "narrowing"
    assert verdict(allow(recipient={"pattern": "^US1[0-9]+$"}), allow(recipient={"pattern": "^US[0-9]+$"}))[0] == "expansion"
    # unanchored regexes follow re.search semantics
    assert verdict(allow(recipient={"pattern": "^abc$"}), allow(recipient={"pattern": "abc"}))[0] == "expansion"


def test_absent_optional_argument_semantics_match_enforcer():
    # P' constrains recipient; old constrains nothing about it: absent recipient is allowed by both
    assert verdict(allow(recipient="*"), allow(recipient={"enum": ["A"]}))[0] == "narrowing"


def test_array_conditions():
    old = {"tags": [{"effect": "allow", "conditions": {"names": {"maxItems": 3, "items": {"enum": ["a", "b"]}}}}]}
    new_small = {"tags": [{"effect": "allow", "conditions": {"names": {"maxItems": 2, "items": {"enum": ["a"]}}}}]}
    new_big = {"tags": [{"effect": "allow", "conditions": {"names": {"maxItems": 4, "items": {"enum": ["a", "b"]}}}}]}
    assert verdict(old, new_small)[0] == "narrowing"
    assert verdict(old, new_big)[0] == "expansion"


def test_object_conditions_fall_back_to_probe_method():
    tools = {"t": {"cfg"}}
    types = {"t": {"cfg": {"type": "object"}}}
    old = parse_policy({"t": [{"effect": "allow", "conditions": {"cfg": {"properties": {"x": {"enum": [1]}}}}}]}, tools)
    new = parse_policy({"t": [{"effect": "allow", "conditions": {"cfg": "*"}}]}, tools)
    d = compare_policies(old, new, types)
    assert d.checks[0].method == "probe"


# --- regex translation cross-check against Python's re.search -------------------------------
PATTERNS = [
    r"^US[0-9]{3}$", r"bill-.*\.txt", r"^(a|b)+c?$", r"^[^x]{2,4}$", r"^\d+\.\d{2}$", r"foo|bar", r"^$",
    r"^[A-Z]{2}\d{2}[A-Z0-9]*$", r"a.c", r"^(?:ab)*$", r"x{2,}",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_regex_translation_agrees_with_python(pattern):
    rng = random.Random(1)
    alphabet = "abcx0123456789.-_ UST"
    samples = {"", "US123", "bill-1.txt", "abc", "abab", "12.50", "foo", "xx", "ab", "GB29NWBK", "aXc"}
    samples |= {"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8))) for _ in range(150)}
    z_re = _RegexTranslator(pattern).translate()
    s = z3.String("s")
    for v in samples:
        solver = z3.Solver()
        solver.add(s == z3.StringVal(v), z3.InRe(s, z_re))
        assert (solver.check() == z3.sat) == (re.search(pattern, v) is not None), (pattern, v)


@pytest.mark.parametrize("bad", [r"(?=a)b", r"\bword\b", r"a^b", r"(?i)abc", r"(a)\1"])
def test_unsupported_regex_features_raise(bad):
    with pytest.raises(Unsupported):
        _RegexTranslator(bad).translate()
