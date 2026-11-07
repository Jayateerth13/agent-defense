"""Which workspace paths are oversight by their name: tests, test configuration, build and CI files.

Used by the fixed list (version 2) and, when the code under test is the whole repository (SWE-bench), to keep
tests and configuration out of the editable code.
"""

from __future__ import annotations

ROOT_CONFIG = ("pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml",
               "noxfile.py", "conftest.py", "Makefile", ".github")
TEST_DIR_NAMES = ("tests", "testing")
PYTEST_CONFIG_NAMES = ("pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml", "tox.ini", "setup.cfg")
#: configuration pytest may read from any folder above a test, so never editable code wherever it sits
CONFIG_ANY_DEPTH = (*PYTEST_CONFIG_NAMES, "pyproject.toml", "conftest.py")


def is_oversight_name(rel: str) -> bool:
    """``rel`` is workspace-relative.  True for test folders' contents, test files, test configuration at any
    depth, and root build and CI files."""
    parts = rel.strip("/").split("/")
    name = parts[-1]
    if any(p in TEST_DIR_NAMES for p in parts[:-1]) or name in TEST_DIR_NAMES:
        return True
    if name in CONFIG_ANY_DEPTH or (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py"):
        return True
    return len(parts) == 1 and name in ROOT_CONFIG or parts[0] == ".github"
