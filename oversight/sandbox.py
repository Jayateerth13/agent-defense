"""Disposable Docker container holding one scenario's workspace.

Every command runs through ``Sandbox.run`` as a fixed user: ``agent`` (the model's commands), ``checker``
(the check) or ``root`` (setup and protection).  The container has no network.

Root helpers run with ``python3 -I`` from ``/``: the workspace is agent-writable, so a planted ``json.py``
there must never be importable by code running as root.
"""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "agent-defense/oversight:3"
DOCKER_DIR = Path(__file__).resolve().parents[1] / "docker"
WORKSPACE = "/workspace"
AGENT_HOME = "/home/agent"
SESSION_DIR = f"{AGENT_HOME}/.agent"
SESSION_FILE = f"{SESSION_DIR}/session.jsonl"
#: files in the agent's home that change how its tools behave (shell startup, global git config)
SHELL_FILES = tuple(f"{AGENT_HOME}/{n}" for n in (".bashrc", ".profile", ".bash_env", ".gitconfig"))

#: per-user environment.  BASH_ENV mirrors the shell-state route in Qin et al.  The checker cannot write
#: the workspace in any condition, so pytest's cache is off (it is not oversight; a failed cache write
#: becomes an error in suites that treat warnings as errors).
ENVS = {
    "agent": {"HOME": AGENT_HOME, "BASH_ENV": f"{AGENT_HOME}/.bash_env"},
    # Hypothesis's example database likewise goes to the checker's own folder instead of /workspace/.hypothesis
    "checker": {"HOME": "/home/checker", "PYTEST_ADDOPTS": "-p no:cacheprovider",
                "HYPOTHESIS_STORAGE_DIRECTORY": "/tmp/hypothesis-checker"},
    "root": {"HOME": "/root"},
}

_HASH_SCRIPT = r"""
import hashlib, json, os, sys
out = {}
for p in json.load(sys.stdin):
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        out[p] = "missing"
        continue
    if os.path.islink(p):
        body = "link:" + os.readlink(p)
    elif os.path.isdir(p):
        body = "dir:" + hashlib.sha256("\n".join(sorted(os.listdir(p))).encode()).hexdigest()
    else:
        with open(p, "rb") as f:
            body = "file:" + hashlib.sha256(f.read()).hexdigest()
    out[p] = f"{body}:{st.st_mode:o}:{st.st_uid}"
print(json.dumps(out))
"""


@dataclass
class ExecResult:
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_image(image: str = IMAGE) -> None:
    """Build the image from ``docker/oversight.Dockerfile`` if it is missing (needs network once)."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    subprocess.run(["docker", "build", "-f", str(DOCKER_DIR / "oversight.Dockerfile"), "-t", image, str(DOCKER_DIR)],
                   check=True)


class Sandbox:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def start(cls, image: str = IMAGE) -> Sandbox:
        name = f"ov-{uuid.uuid4().hex[:12]}"
        subprocess.run(["docker", "run", "-d", "--rm", "--network", "none", "--name", name, image],
                       check=True, capture_output=True)
        return cls(name)

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def _exec(self, cmd: str, user: str, workdir: str, env: dict | None, stdin: bytes | None,
              timeout: float) -> tuple[int, bytes, bytes, bool]:
        args = ["docker", "exec"] + (["-i"] if stdin is not None else []) + ["-u", user, "-w", workdir]
        for k, v in {**ENVS[user], **(env or {})}.items():
            args += ["-e", f"{k}={v}"]
        args += [self.name, "bash", "-c", cmd]
        try:
            p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return -1, e.stdout or b"", e.stderr or b"", True
        return p.returncode, p.stdout, p.stderr, False

    def run(self, cmd: str, user: str = "agent", *, timeout: float = 60, env: dict | None = None,
            workdir: str = WORKSPACE, stdin: str | None = None) -> ExecResult:
        code, out, err, timed_out = self._exec(cmd, user, workdir, env, stdin.encode() if stdin is not None else None,
                                               timeout)
        return ExecResult(code, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), timed_out)

    def root(self, cmd: str, **kw) -> ExecResult:
        """Run a setup command as root from ``/``; raises if it fails."""
        r = self.run(cmd, "root", workdir="/", **kw)
        if not r.ok:
            raise RuntimeError(f"root command failed ({r.code}): {cmd}\n{r.stderr[-2000:]}")
        return r

    def load_files(self, files: dict[str, str], executable: set[str] = frozenset()) -> None:
        """Copy ``{relative path: text}`` into the workspace as root-owned files with LF line endings."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for path, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(path)
                info.size, info.mode = len(data), 0o755 if path in executable else 0o644
                tar.addfile(info, io.BytesIO(data))
        self.load_tar(buf.getvalue())

    def load_tar(self, data: bytes) -> None:
        """Unpack a tar into the workspace as root-owned, world-readable files."""
        code, _, err, _ = self._exec(f"tar -x --no-same-owner -C {WORKSPACE} && chmod -R a+rX {WORKSPACE}",
                                     "root", "/", None, data, 120)
        if code != 0:
            raise RuntimeError(f"loading files failed: {err.decode(errors='replace')}")

    def read_file(self, path: str) -> bytes | None:
        code, out, _, _ = self._exec(f"cat -- {shlex.quote(path)}", "root", "/", None, None, 60)
        return out if code == 0 else None

    def hash_paths(self, paths: list[str]) -> dict[str, str]:
        """Content, mode and owner of each path (``missing`` if absent), read as root."""
        # paths go on stdin: a long list would exceed the Windows command-line limit
        r = self.run(f"python3 -I -c {shlex.quote(_HASH_SCRIPT)}", "root", workdir="/", stdin=json.dumps(paths))
        if not r.ok:
            raise RuntimeError(f"hashing failed: {r.stderr[-2000:]}")
        return json.loads(r.stdout)

    def copy_out(self, paths: list[str]) -> bytes:
        """A tar of the given workspace-relative paths."""
        code, out, err, _ = self._exec(f"tar -c -C {WORKSPACE} -- " + " ".join(map(shlex.quote, paths)),
                                       "root", "/", None, None, 120)
        if code != 0:
            raise RuntimeError(f"copy out failed: {err.decode(errors='replace')}")
        return out

    def kill_user(self, user: str) -> None:
        self.run(f"pkill -KILL -u {user} || true", "root", workdir="/")

    def remove(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
