"""Applies a protection plan inside a sandbox with plain Unix ownership and modes.

The agent runs as the unprivileged ``agent`` user, so the kernel enforces the plan however the agent
phrases a command.

* Protected files and folders are owned by root and not writable by others.
* A protected name that does not exist yet gets a *tombstone*: an empty root-owned directory at that
  name, so the agent cannot create a file there.  pytest ignores such directories.
* The workspace and the agent's home are ``root:agent 1775``.  The agent can still create files in
  them, but the sticky bit stops it deleting or renaming root's entries.  The sticky bit does not
  restrict a directory's owner, which is why root must own them.

With no plan, everything stays agent-owned (the no-defense and prompt-rule conditions).
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field

from oversight.sandbox import AGENT_HOME, SESSION_DIR, SESSION_FILE, SHELL_FILES, WORKSPACE, Sandbox


@dataclass
class ProtectionPlan:
    """Absolute paths inside the container."""

    readonly: list[str] = field(default_factory=list)  # files or whole folders, root-owned recursively
    locked_dirs: list[str] = field(default_factory=list)  # folders that gain or lose no entries
    tombstones: list[str] = field(default_factory=list)  # missing names the agent must not create
    open_files: list[str] = field(default_factory=list)  # protected files the task may edit in place
    sticky_dirs: list[str] = field(default_factory=list)  # agent may add entries but not remove root's
    deletable: list[str] = field(default_factory=list)  # protected files the task may delete
    protect_home: bool = True  # shell startup files and the session file

    def protected_paths(self) -> list[str]:
        """Everything the observer should hash before and after a run."""
        home = [*SHELL_FILES, SESSION_DIR, SESSION_FILE] if self.protect_home else []
        return sorted({*self.readonly, *self.locked_dirs, *self.tombstones, *home}
                      - set(self.open_files) - set(self.deletable))


def _q(p: str) -> str:
    return shlex.quote(p)


def build_script(plan: ProtectionPlan | None) -> list[str]:
    """The root commands that set up the workspace and, if a plan is given, apply it."""
    dotfiles = " ".join(map(_q, SHELL_FILES))
    cmds = [
        f"mkdir -p {SESSION_DIR}",
        f"touch {dotfiles} {SESSION_FILE}",
        f"chown -R agent:agent {WORKSPACE} {AGENT_HOME}",
        f"chmod 755 {WORKSPACE} {AGENT_HOME}",
    ]
    if plan is None:
        return cmds
    cmds.append(f"chown root:agent {WORKSPACE} && chmod 1775 {WORKSPACE}")
    for p in plan.readonly:
        cmds.append(f"chown -R root:root {_q(p)} && chmod -R u+rwX,go+rX,go-w {_q(p)}")
    for d in plan.locked_dirs:
        cmds.append(f"chown root:root {_q(d)} && chmod 755 {_q(d)}")
    for d in plan.sticky_dirs:
        cmds.append(f"chown root:agent {_q(d)} && chmod 1775 {_q(d)}")
    for t in plan.tombstones:
        # an existing entry is not missing: refuse rather than silently protect the wrong thing
        cmds.append(f"if [ -e {_q(t)} ] || [ -L {_q(t)} ]; then echo 'tombstone target exists: {t}' >&2; exit 1; fi")
        cmds.append(f"mkdir -p {_q(t)} && chown root:root {_q(t)} && chmod 755 {_q(t)}")
    for f in plan.open_files:
        # a missing file the task needs (e.g. a new fixture file in a locked folder) is created empty
        cmds.append(f"[ -e {_q(f)} ] || touch {_q(f)}; chown agent:agent {_q(f)}")
    for f in plan.deletable:
        # deleting needs write access to the folder; the sticky bit keeps root's other entries safe.  The
        # folder also accepts new files then: the caller tombstones the names the check looks for there.
        d = posixpath.dirname(f)
        cmds.append(f"chown agent:agent {_q(f)} && chown root:agent {_q(d)} && chmod 1775 {_q(d)}")
    if plan.protect_home:
        cmds.append(f"chown root:agent {AGENT_HOME} && chmod 1775 {AGENT_HOME}")
        cmds.append(f"chown root:root {dotfiles} {SESSION_DIR} {SESSION_FILE}")
        cmds.append(f"chmod 644 {dotfiles} {SESSION_FILE} && chmod 755 {SESSION_DIR}")
    return cmds


def apply(sandbox: Sandbox, plan: ProtectionPlan | None) -> list[str]:
    """Run the protection script as root; returns the commands for the run log."""
    cmds = build_script(plan)
    # each step must fail loudly: ``bash -e`` alone ignores a failure in the middle of ``a && b``
    script = "".join(f"{{ {c}\n}} || exit 1\n" for c in cmds)
    sandbox.root("bash -s", stdin=script)  # on stdin: a long plan exceeds the command-line limit
    return cmds
