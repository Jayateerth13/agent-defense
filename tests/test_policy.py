"""Unit tests: policy schema validation, enforcement algorithm, JSON extraction."""

import pytest

from policy.enforcer import check_call
from policy.generator import extract_json
from policy.schema import DEFAULT_BLOCK_MSG, PolicyError, parse_policy

TOOLS = {"send_money": {"recipient", "amount", "subject", "date"}, "get_balance": set(), "read_file": {"file_path"}}


def pol(obj):
    return parse_policy(obj, TOOLS)


def rule(effect="allow", conditions=None, **kw):
    return {"effect": effect, "conditions": conditions or {}, **kw}


def test_paper_example_shape_validates():
    p = pol({"send_money": [{
        "priority": 100, "effect": "allow",
        "conditions": {"recipient": {"type": "string", "enum": ["A"]}},
        "fallback": "return_msg", "fallback_msg": "Tool call blocked: recipient not in allowed list."}]})
    assert p.rules_for("send_money")[0].priority == 100


@pytest.mark.parametrize("bad", [
    [],                                                        # not an object
    {"send_money": {"effect": "allow"}},                       # rules must be a list
    {"send_money": [{"effect": "maybe", "conditions": {}}]},   # bad effect
    {"send_money": [{"effect": "allow"}]},                     # missing conditions
    {"send_money": [rule(conditions={"recipient": {"format_x": 1}})]},       # unsupported keyword
    {"send_money": [rule(conditions={"recipient": {"pattern": "("}})]},      # bad regex
    {"send_money": [rule(conditions={"recipinet": "*"})]},                    # typo'd arg name
    {"send_money": [rule(fallback="explode")]},                               # bad fallback
])
def test_invalid_policies_rejected(bad):
    with pytest.raises(PolicyError):
        pol(bad)


def test_unknown_tool_dropped():
    assert pol({"nope": [rule()]}).rules == {}


def test_default_block_when_no_rule():
    p = pol({"get_balance": [rule()]})
    d = check_call(p, "send_money", {"recipient": "X"})
    assert not d.allowed and d.message == "The tool 'send_money' is not allowed."  # official wording


def test_allow_all_with_empty_conditions_and_wildcard():
    p = pol({"read_file": [rule(conditions={"file_path": "*"})]})
    assert check_call(p, "read_file", {"file_path": "anything"}).allowed


def test_enum_and_comparison_conditions():
    p = pol({"send_money": [rule(conditions={"recipient": {"enum": ["A", "B"]}, "amount": {"maximum": 100}})]})
    assert check_call(p, "send_money", {"recipient": "A", "amount": 100}).allowed
    assert not check_call(p, "send_money", {"recipient": "A", "amount": 100.01}).allowed
    assert not check_call(p, "send_money", {"recipient": "C", "amount": 1}).allowed


def test_regex_and_length():
    p = pol({"read_file": [rule(conditions={"file_path": {"pattern": r"^bill-.*\.txt$", "maxLength": 30}})]})
    assert check_call(p, "read_file", {"file_path": "bill-december-2023.txt"}).allowed
    assert not check_call(p, "read_file", {"file_path": "landlord-notices.txt"}).allowed
    assert not check_call(p, "read_file", {"file_path": "bill-" + "x" * 40 + ".txt"}).allowed


def test_array_length_and_member_conditions():
    tools = {"email": {"to"}}
    p = parse_policy({"email": [rule(conditions={"to": {"type": "array", "maxItems": 2,
                                                        "items": {"enum": ["a@x", "b@x"]}}})]}, tools)
    assert check_call(p, "email", {"to": ["a@x"]}).allowed
    assert not check_call(p, "email", {"to": ["a@x", "b@x", "a@x"]}).allowed
    assert not check_call(p, "email", {"to": ["evil@x"]}).allowed


def test_forbid_overrides_allow_regardless_of_priority_and_uses_fallback():
    p = pol({"send_money": [
        rule("allow", {"recipient": "*"}, priority=1000),
        rule("forbid", {"amount": {"minimum": 1000}}, priority=1, fallback="terminate_execution",
             fallback_msg="too much"),
    ]})
    assert check_call(p, "send_money", {"recipient": "A", "amount": 5}).allowed
    d = check_call(p, "send_money", {"recipient": "A", "amount": 5000})
    assert not d.allowed and d.fallback == "terminate_execution" and d.message == "too much"


def test_missing_key_is_unconstrained_but_present_none_is_validated():
    """Official ``check_arg`` loop: only arguments present in the call are checked; the runner passes
    defaults applied, so an omitted optional argument arrives as ``None`` and is validated."""
    p = pol({"send_money": [rule(conditions={"recipient": {"enum": ["A"]}})]})
    assert check_call(p, "send_money", {"amount": 3}).allowed  # key not present at all
    d = check_call(p, "send_money", {"recipient": None})
    assert not d.allowed and "None is not one of" in d.message


def test_generated_rule_violation_blocks_immediately_like_official():
    """Two generated (priority 100) rules for one tool: the first violated one blocks, the second is
    never consulted (official ``_check_tool_call``); human-written (lower priority) rules fall through."""
    first = rule(conditions={"recipient": {"enum": ["A"]}}, priority=100)
    second = rule(conditions={"recipient": {"enum": ["B"]}}, priority=100)
    p = pol({"send_money": [first, second]})
    assert check_call(p, "send_money", {"recipient": "A"}).allowed
    assert not check_call(p, "send_money", {"recipient": "B"}).allowed  # second would allow, first blocks
    # an always-allowed (priority 1) rule is consulted first and short-circuits
    p = pol({"send_money": [rule(priority=1), first]})
    assert check_call(p, "send_money", {"recipient": "evil"}).allowed
    # non-generated rules are OR-ed
    p = pol({"send_money": [rule(conditions={"recipient": {"enum": ["A"]}}, priority=5),
                            rule(conditions={"recipient": {"enum": ["B"]}}, priority=5)]})
    assert check_call(p, "send_money", {"recipient": "B"}).allowed


def test_allow_rule_fallback_used_for_default_block():
    p = pol({"send_money": [rule(conditions={"recipient": {"enum": ["A"]}}, fallback="return_msg",
                                 fallback_msg="recipient not in allowed list")]})
    d = check_call(p, "send_money", {"recipient": "evil"})
    assert not d.allowed and d.message == "recipient not in allowed list"


def test_extract_json_follows_official_behaviour():
    assert extract_json('```json\n[{"name": "a", "args": {}}]\n```') == [{"name": "a", "args": {}}]
    assert extract_json('Yes\n```json\n[1]\n```') == [1]
    assert extract_json('Yes [{"name": "a", "args": {}}]') == [{"name": "a", "args": {}}]
    assert extract_json("No") is None and extract_json("No, nothing to update.") is None
    assert extract_json('Yes, but only in prose', enforce_code_block=True) is None
    with pytest.raises(ValueError):  # json.JSONDecodeError: the caller retries
        extract_json("garbage that is not json")


