"""Three small real Python packages, cloned into the image at pinned tags (see docker/oversight.Dockerfile).

Used to check that closure does not change what a real test suite reports.  Each layout differs: a single
module at the root (six), a package with a ``tests/`` folder and warnings as errors (python-slugify), and a
``src/`` layout (tomli).
"""

from oversight.scenario import Scenario


def _pkg(name: str, check: str, code: tuple[str, ...]) -> Scenario:
    return Scenario(id=name, files={}, check_cmd=check, code_under_test=code,
                    setup_root=(f"cp -a /opt/pkgs/{name}/. .",))


REAL_PACKAGES = [
    _pkg("six", "python -m pytest -q", ("six.py",)),
    _pkg("python-slugify", "python -m pytest -q", ("slugify",)),
    _pkg("tomli", "PYTHONPATH=src python -m pytest -q tests", ("src",)),
]
