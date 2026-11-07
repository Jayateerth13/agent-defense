"""Offline tests for task rules: the policy-model answer parser and the named-files baseline."""

from oversight import policy_writer as pw
from oversight.policy import is_generic

PROTECTED = ["Makefile", "pyproject.toml", "test", "tests/conftest.py", "tests/test_tax.py", ".github/workflows/ci.yml"]
EXISTING = {"Makefile", "pyproject.toml", "tests/test_tax.py", ".github/workflows/ci.yml"}


def named(task):
    return [p for p, _ in pw.named_task_rules(task, PROTECTED, EXISTING)]


def test_named_rule_matches_whole_paths_only():
    assert named("The assertion in tests/test_tax.py::test_rate is wrong.") == ["tests/test_tax.py"]
    assert named("Add Python 3.13 to the matrix in .github/workflows/ci.yml.") == [".github/workflows/ci.yml"]
    assert named("Add a lint target to the Makefile.") == ["Makefile"]
    assert named("The test suite fails. Make it pass.") == []  # 'test' is a word here, not the make-target file


def test_parser_keeps_only_protected_paths():
    opens, err = pw.parse('{"open": [{"path": "tests/test_tax.py", "ops": ["write"]}, '
                          '{"path": "src/tax.py", "ops": ["write"]}]}', PROTECTED)
    assert err is None and opens == [("tests/test_tax.py", ("write",))]


def test_malformed_answer_opens_nothing():
    for raw in ("not json", '{"open": [{"path": "x"}]}', '{"open": [{"path": "Makefile", "ops": ["chmod"]}]}'):
        opens, err = pw.parse(raw, PROTECTED)
        assert opens == [] and err


def test_llm_rules_record_has_raw_answer_and_hash():
    rec = pw.llm_task_rules("Fix it", ["src/tax.py"], PROTECTED, lambda s, u: '{"open": []}')
    assert rec["opens"] == [] and rec["raw"] == '{"open": []}' and len(rec["prompt_hash"]) == 16


def test_operator_limits_are_never_task_openable():
    assert is_generic("/workspace/.git/config") and is_generic("/home/agent/.bashrc")
    assert not is_generic("/workspace/tests/test_tax.py")
