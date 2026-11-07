"""Oversight closure: the files a check depends on, found by watching the check run once.

Before the agent starts, the check runs as ``checker`` under ``strace`` on the pristine repo.  Every
workspace path it opens, stats, executes or lists belongs to the closure, including paths it looked
for and did not find: a missing ``conftest.py``, or a file named like a ``make`` target, is a place
an agent could add one.

Protection derived from the closure:

* files the check used become read-only;
* folders it listed, workspace folders on its ``sys.path`` and the ancestors of used files are
  locked, so no entry can be added, removed or renamed.  This also covers module shadowing: Python
  finds modules by listing folders, so a shadowing ``pytest.py`` never shows up as a missing path;
* missing paths whose folder is not locked become tombstones;
* anything inside the code under test stays writable and is logged as a known hole.

``strace`` runs as ``checker`` itself: a root tracer cannot attach to a process that switched user
without ``CAP_SYS_PTRACE``, and the sandbox does not get that capability.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field

from oversight.patterns import is_oversight_name
from oversight.protect import ProtectionPlan
from oversight.sandbox import WORKSPACE, Sandbox
from oversight.scenario import Scenario

TRACE_FILE = "/tmp/closure.trace"
PROBE_DIR = "/tmp/closure-probe"
STRACE = "strace -f -qq -y -e trace=%file,%process,chdir,fchdir,getdents64 -o " + TRACE_FILE
#: world-writable places outside the workspace; a check that reads from them is reported, not protected
SHARED_DIRS = ("/tmp/", "/var/tmp/", "/dev/shm/")
#: Python's ``tempfile`` probes the temp folder with random 8-character names; not a dependency
_TEMPFILE_PROBE = re.compile(r"^/(?:var/)?tmp/(?:tmp)?[a-z0-9_]{8}$")

_PID = re.compile(r"^(\d+)\s+(.*)$")
_RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>(.*)$")
_CALL = re.compile(r"^(\w+)\((.*)\)\s+=\s+(-?\d+|\?)(?:<[^>]*>)?(?:\s+(E[A-Z]+))?")
_AT_PATH = re.compile(r'^(?:AT_FDCWD|\d+)<([^>]*)>, "((?:[^"\\]|\\.)*)"')
_FD = re.compile(r"^\d+<([^>]*)>")
_STR = re.compile(r'^"((?:[^"\\]|\\.)*)"')


@dataclass
class Access:
    path: str
    kind: str  # "used" | "missing" | "listed"


def parse_strace(text: str, cwd: str = WORKSPACE) -> list[Access]:
    """Workspace-relevant accesses from ``strace -f -y`` output (all paths absolute and normalised)."""
    cwds: dict[str, str] = {}
    pending: dict[str, str] = {}
    out: list[Access] = []
    for line in text.splitlines():
        m = _PID.match(line)
        if not m:
            continue
        pid, rest = m.groups()
        if rest.endswith("<unfinished ...>"):
            pending[pid] = rest[: -len("<unfinished ...>")].rstrip()
            continue
        r = _RESUMED.match(rest)
        if r:
            rest = pending.pop(pid, r.group(1) + "(") + r.group(2)
        c = _CALL.match(rest)
        if not c:
            continue
        name, args, ret, err = c.groups()
        here = cwds.setdefault(pid, cwd)
        if name in ("clone", "clone3", "fork", "vfork") and ret.isdigit():
            cwds[ret] = here  # the child starts in its parent's folder
            continue
        if name == "getdents64":
            f = _FD.match(args)
            if f:
                out.append(Access(posixpath.normpath(f.group(1)), "listed"))
            continue
        if name == "fchdir":
            f = _FD.match(args)
            if f and ret == "0":
                cwds[pid] = f.group(1)
            continue
        at = _AT_PATH.match(args)
        if at:
            base, raw = at.groups()
            if args.startswith("AT_FDCWD"):  # -y prints the caller's folder: keep it for non-*at calls
                cwds[pid] = here = base
            path = posixpath.join(base, raw)
        else:
            s = _STR.match(args)
            if not s:
                continue
            path = posixpath.join(here, s.group(1))
        path = posixpath.normpath(path)
        if name == "chdir" and ret == "0":
            cwds[pid] = path
        if err == "ENOENT":
            out.append(Access(path, "missing"))
        elif err is None and ret != "-1":
            out.append(Access(path, "used"))
        # ENOTDIR (a path inside a regular file) cannot be created; other errors say nothing about the path
    return out


def _within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _ancestors(path: str) -> list[str]:
    out = []
    while path != WORKSPACE and path != "/":
        path = posixpath.dirname(path)
        out.append(path)
    return out


@dataclass
class Closure:
    files: list[str] = field(default_factory=list)  # existing files the check used -> read-only
    dirs: list[str] = field(default_factory=list)  # folders to lock against new, removed or renamed entries
    tombstones: list[str] = field(default_factory=list)  # missing paths whose folder is not locked
    covered: list[str] = field(default_factory=list)  # missing paths already covered by a locked folder
    holes: list[str] = field(default_factory=list)  # paths in the code under test, left writable
    shared: list[str] = field(default_factory=list)  # paths in world-writable places outside the workspace
    modules: list[str] = field(default_factory=list)  # top-level modules the check loaded (from the probe)

    def plan(self, open_files: tuple[str, ...] = ()) -> ProtectionPlan:
        return ProtectionPlan(readonly=list(self.files), locked_dirs=list(self.dirs),
                              tombstones=list(self.tombstones), open_files=list(open_files))

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def derive(accesses: list[Access], kinds: dict[str, str], code_under_test: list[str],
           sys_path_dirs: list[str] = (), cut_keeps_oversight: bool = False) -> Closure:
    """Apply the closure rules.  ``kinds`` maps each path (and each missing path's ancestors) to
    ``file`` / ``dir`` / ``missing`` as it is in the pristine repo.

    With ``cut_keeps_oversight`` (code under test = the whole repository), tests, conftest files and root
    configuration inside the code under test are still protected (``patterns.is_oversight_name``)."""
    cut = [posixpath.join(WORKSPACE, d) for d in code_under_test]

    def in_cut(p: str) -> bool:
        if not any(_within(p, d) for d in cut):
            return False
        return not (cut_keeps_oversight and p != WORKSPACE and is_oversight_name(p.removeprefix(WORKSPACE + "/")))

    cl = Closure()
    used, missing, locked = set(), set(), set()
    for a in accesses:
        if a.path.startswith(SHARED_DIRS):
            if not (_within(a.path, PROBE_DIR) or _TEMPFILE_PROBE.match(a.path)):
                cl.shared.append(a.path)
            continue
        if not _within(a.path, WORKSPACE):
            continue
        if in_cut(a.path) and not (a.path in cut and kinds.get(a.path) == "dir"):
            cl.holes.append(a.path)  # a code-under-test folder itself is handled below; a file is a hole
            continue
        if a.kind == "listed":
            locked.add(a.path)
        elif a.kind == "missing" or kinds.get(a.path) == "missing":
            missing.add(a.path)
        elif kinds.get(a.path) == "dir":
            pass  # a folder that was only stat'ed or opened is protected through its ancestors below
        else:
            used.add(a.path)
    locked |= {d for d in sys_path_dirs if _within(d, WORKSPACE)}
    # a protected entry is only as safe as its folders: an unlocked parent could be renamed and rebuilt
    for p in used | set(locked):
        locked.update(_ancestors(p))
    locked = {d for d in locked if not in_cut(d)} | {WORKSPACE}

    def covered(p: str) -> bool:
        parent = posixpath.dirname(p)
        if parent in locked:
            return True
        return kinds.get(parent) == "missing" and parent != "/" and covered(parent)

    for p in sorted(missing):
        if in_cut(p):
            cl.holes.append(p)
        elif covered(p):
            cl.covered.append(p)
        else:
            cl.tombstones.append(p)
    cl.files = sorted(used)
    cl.dirs = sorted(locked)
    cl.holes = sorted(set(cl.holes))
    cl.shared = sorted(set(cl.shared))
    return cl


def build(sandbox: Sandbox, scenario: Scenario) -> Closure:
    """Trace the check on the pristine repo in ``sandbox`` and derive its closure.

    The check may leave files behind, so use a throwaway sandbox and run the agent in a fresh one.
    """
    sandbox.root(f"rm -rf {PROBE_DIR} {TRACE_FILE} && mkdir -p {PROBE_DIR} && chmod 777 {PROBE_DIR}")
    sandbox.run(f"{STRACE} bash -c {shlex.quote(scenario.check_cmd)}", "checker", timeout=600,
                env={"OVERSIGHT_PROBE_DIR": PROBE_DIR})
    text = (sandbox.read_file(TRACE_FILE) or b"").decode("utf-8", "replace")
    if not text:
        raise RuntimeError("strace produced no output; is the check command valid?")
    accesses = parse_strace(text)

    probes = sandbox.run("python3 -I -", "root", workdir="/", stdin=(
        f"import glob, json\nprint(json.dumps([json.load(open(f)) for f in glob.glob('{PROBE_DIR}/*.json')]))\n"))
    records = json.loads(probes.stdout or "[]")
    sys_path_dirs = sorted({p for r in records for p in r["path"] if _within(p, WORKSPACE)})
    modules = sorted({m for r in records for m in r["modules"]})

    wanted = {a.path for a in accesses if _within(a.path, WORKSPACE)}
    wanted |= {anc for p in list(wanted) for anc in _ancestors(p)}
    states = sandbox.hash_paths(sorted(wanted))
    kinds = {p: "missing" if s == "missing" else s.split(":", 1)[0].replace("link", "file") for p, s in states.items()}
    cl = derive(accesses, kinds, list(scenario.code_under_test), sys_path_dirs, scenario.cut_keeps_oversight)
    cl.modules = modules
    return cl
