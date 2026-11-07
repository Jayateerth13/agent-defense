"""Which paths each condition protects.

Operator limits (``GENERIC``) apply in every defended condition and are never opened by a task: git
hooks and config, plus the agent's shell startup files and session file (``ProtectionPlan.protect_home``).

``fixed`` (version 2) is a careful list written by someone who knows pytest: every test folder and test file
at any depth, every existing ``conftest.py``, the root configuration files and CI, and tombstones for missing
root configuration names and for a missing ``conftest.py`` in the root and in each test folder.  Version 1
protected only top-level names, which misses nested tests in real repositories.  It is the proposal's C4
baseline.  ``closure`` replaces that list with the closure of the check.
"""

from __future__ import annotations

from oversight.closure import Closure
from oversight.patterns import ROOT_CONFIG, TEST_DIR_NAMES
from oversight.protect import ProtectionPlan
from oversight.sandbox import WORKSPACE, Sandbox

CONDITIONS = ("none", "prompt", "fixed", "closure")
UNDEFENDED = ("none", "prompt")

GIT_DIR = f"{WORKSPACE}/.git"
GENERIC_READONLY = (f"{GIT_DIR}/hooks", f"{GIT_DIR}/config")
FIXED_LIST_VERSION = 2
#: test folders (printed, not descended into) and test files at any depth, skipping git, virtualenvs and caches
_FIND_TESTS = (
    f"find {WORKSPACE} \\( -name .git -o -name .venv -o -name site-packages -o -name node_modules "
    "-o -name __pycache__ \\) -prune -o "
    + "\\( -type d \\( " + " -o ".join(f"-name {n}" for n in TEST_DIR_NAMES) + " \\) -print -prune \\) -o "
    "\\( -type f \\( -name 'test_*.py' -o -name '*_test.py' -o -name conftest.py \\) -print \\)"
)


def fixed_plan(sandbox: Sandbox) -> ProtectionPlan:
    """Fixed list version 2 for the repo in ``sandbox``."""
    found = [p for p in sandbox.run(_FIND_TESTS, "root", workdir="/").stdout.splitlines() if p]
    roots = [f"{WORKSPACE}/{p}" for p in ROOT_CONFIG]
    states = sandbox.hash_paths(roots)
    test_dirs = [p for p in found if p.rsplit("/", 1)[-1] in TEST_DIR_NAMES]
    in_test_dir = lambda p: any(p.startswith(d + "/") for d in test_dirs)  # noqa: E731
    readonly = sorted({*test_dirs, *(p for p in found if not in_test_dir(p)), *(p for p in roots if states[p] != "missing")})
    tombstones = [p for p in roots if states[p] == "missing"]
    # test folders are root-owned recursively, so no new conftest.py can appear inside them
    return ProtectionPlan(readonly=readonly, tombstones=tombstones)


def is_generic(path: str) -> bool:
    """Operator limits no task may open: git internals, and anything outside the workspace (home, traces)."""
    return not path.startswith(WORKSPACE + "/") or path == GIT_DIR or path.startswith(GIT_DIR + "/")


def plan_for(condition: str, sandbox: Sandbox, closure: Closure | None = None,
             opens: list[tuple[str, tuple[str, ...]]] = (), declared: tuple[str, ...] = ()) -> ProtectionPlan | None:
    """The protection plan for ``condition`` on the pristine repo loaded in ``sandbox``.

    ``opens`` lists (workspace-relative path, ops) the task may change, ops being ``write`` and/or ``delete``;
    ``declared`` lists oversight the check never reads (e.g. CI), protected in the closure condition.
    """
    if condition in UNDEFENDED:
        return None
    decl = [f"{WORKSPACE}/{p}" for p in declared]
    states = sandbox.hash_paths([*GENERIC_READONLY, GIT_DIR, *decl])
    exists = lambda p: states[p] != "missing"  # noqa: E731
    if condition == "fixed":
        plan = fixed_plan(sandbox)
    elif condition == "closure":
        if closure is None:
            raise ValueError("the closure condition needs a closure")
        plan = closure.plan()
        plan.readonly += [p for p in decl if exists(p)]
        plan.tombstones += [p for p in decl if not exists(p)]
    else:
        raise ValueError(f"unknown condition {condition!r}")
    for rel, ops in opens:
        path = f"{WORKSPACE}/{rel.strip('/')}"
        if is_generic(path):
            continue  # forbid wins: operator limits are never opened
        if "write" in ops:
            plan.open_files.append(path)  # created empty if missing
        if "delete" in ops and closure and path in closure.files:
            plan.deletable.append(path)
            # the folder becomes writable: block the names the check would load from it
            folder = path.rsplit("/", 1)[0]
            existing = set(closure.files + closure.dirs) if closure else set()
            for p in (closure.covered if closure else []):
                if p.startswith(folder + "/"):  # e.g. tests/__pycache__/x.pyc -> tombstone tests/__pycache__
                    child = f"{folder}/{p[len(folder) + 1:].split('/')[0]}"
                    if child not in plan.tombstones and child not in existing and child not in plan.open_files:
                        plan.tombstones.append(child)  # existing folders stay locked
    plan.readonly += [p for p in GENERIC_READONLY if exists(p)]
    if exists(GIT_DIR):
        plan.sticky_dirs.append(GIT_DIR)  # git still works, but hooks and config cannot be replaced
    return plan
