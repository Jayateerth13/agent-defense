"""Offline tests for the protection script (no Docker needed)."""

from oversight.protect import ProtectionPlan, build_script
from oversight.sandbox import SESSION_FILE


def test_no_plan_leaves_everything_agent_owned():
    script = "\n".join(build_script(None))
    assert "chown -R agent:agent /workspace /home/agent" in script
    assert "root" not in script


def test_workspace_is_root_owned_with_sticky_bit():
    script = build_script(ProtectionPlan())
    assert "chown root:agent /workspace && chmod 1775 /workspace" in script


def test_open_files_are_handed_back_after_readonly():
    cmds = build_script(ProtectionPlan(readonly=["/workspace/tests"], open_files=["/workspace/tests/test_a.py"]))
    ro = next(i for i, c in enumerate(cmds) if c.startswith("chown -R root:root /workspace/tests"))
    op = next(i for i, c in enumerate(cmds) if c.endswith("chown agent:agent /workspace/tests/test_a.py"))
    assert ro < op


def test_tombstone_refuses_an_existing_entry():
    cmds = build_script(ProtectionPlan(tombstones=["/workspace/conftest.py"]))
    guard = next(c for c in cmds if c.startswith("if [ -e /workspace/conftest.py ]"))
    assert "exit 1" in guard


def test_protected_paths_include_home_unless_disabled():
    plan = ProtectionPlan(readonly=["/workspace/tests"], open_files=["/workspace/tests"])
    assert SESSION_FILE in plan.protected_paths()
    assert "/workspace/tests" not in plan.protected_paths()
    assert SESSION_FILE not in ProtectionPlan(protect_home=False).protected_paths()
