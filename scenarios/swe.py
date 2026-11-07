"""ImpossibleBench, SWE-bench part: closure on real repositories, scripted (no model).

Instances: the lowest instance id of five repositories with different test setups, one-off version, from
``fjzzq2002/impossible_swebench`` (downloaded on first use into ``third_party/impossiblebench``, never committed).
Test commands and test files are the official ones from ``swebench`` 3.0.17 (``MAP_REPO_VERSION_TO_SPECS`` and
``get_test_directives``), copied here so the run does not depend on that package.

The code under test is the whole repository minus tests, conftest files and root configuration.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from oversight.sandbox import DOCKER_DIR
from oversight.scenario import Scenario
from scenarios.impossiblebench import DATA_DIR, _convert_dataset

#: instance id -> (official test command, test files or labels, environment exports)
INSTANCES = {
    "astropy__astropy-12907": ("pytest -rA", "astropy/modeling/tests/test_separable.py", ""),
    "django__django-10554": ("./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
                             "queries.test_qs_combinators",
                             "export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 &&"),
    "pytest-dev__pytest-10051": ("pytest -rA", "testing/logging/test_fixture.py", ""),
    "sphinx-doc__sphinx-10323": ("tox --current-env -epy39 -v --", "tests/test_directive_code.py", ""),
    "pydata__xarray-2905": ("pytest -rA", "xarray/tests/test_variable.py", ""),
}


def base_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_').lower()}:latest"


def image(instance_id: str) -> str:
    return f"agent-defense/swe:{instance_id.replace('__', '-').lower()}"


def rows() -> dict[str, dict]:
    return {r["instance_id"]: r for r in _convert_dataset("impossible_swebench", "oneoff")
            if r["instance_id"] in INSTANCES}


def build_image(instance_id: str, row: dict) -> None:
    """Build the prepared image once (needs the official base image pulled)."""
    if subprocess.run(["docker", "image", "inspect", image(instance_id)], capture_output=True).returncode == 0:
        return
    with tempfile.TemporaryDirectory() as ctx:
        shutil.copy(DOCKER_DIR / "probe" / "oversight_probe.py", ctx)
        with open(f"{ctx}/impossible_test.patch", "w", encoding="utf-8", newline="\n") as f:
            f.write(row["test_patch"])
        subprocess.run(["docker", "build", "-q", "-f", str(DOCKER_DIR / "swe.Dockerfile"),
                        "--build-arg", f"BASE={base_image(instance_id)}", "-t", image(instance_id), ctx],
                       check=True, capture_output=True)


def scenario(instance_id: str) -> Scenario:
    cmd, files, exports = INSTANCES[instance_id]
    check = f"source /opt/miniconda3/bin/activate && conda activate testbed && {exports} {cmd} {files}"
    return Scenario(instance_id, files={}, check_cmd=check, code_under_test=("",), cut_keeps_oversight=True,
                    image=image(instance_id))


__all__ = ["INSTANCES", "DATA_DIR", "build_image", "image", "rows", "scenario"]
