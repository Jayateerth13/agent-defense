"""Offline tests for the closure parser and rules, on lines taken from a real trace of the tax repo."""

from oversight.closure import Access, derive, parse_strace

TRACE = """\
37    execve("/usr/bin/make", ["make", "-s", "test"], 0x7ffc /* 13 vars */) = 0
37    newfstatat(AT_FDCWD</workspace>, "Makefile", {st_mode=S_IFREG|0644, st_size=47, ...}, 0) = 0
37    newfstatat(AT_FDCWD</workspace>, "test", 0x7fffa9a0a520, 0) = -1 ENOENT (No such file or directory)
37    newfstatat(AT_FDCWD</workspace>, "RCS", 0x7fffa9a0c750, 0) = -1 ENOENT (No such file or directory)
37    clone(child_stack=0x72475bbc5ff0, flags=CLONE_VM|CLONE_VFORK|SIGCHLD <unfinished ...>
38    execve("/workspace/bin/pytest", ["pytest"], 0x61b2 /* 16 vars */) = -1 ENOENT (No such file or directory)
37    <... clone resumed>)              = 38
38    chdir("sub")                      = 0
38    access("conf.ini", R_OK)          = -1 ENOENT (No such file or directory)
38    chdir("/workspace")               = 0
38    openat(AT_FDCWD</workspace>, "/workspace/pyproject.toml", O_RDONLY|O_CLOEXEC) = 3</workspace/pyproject.toml>
38    openat(AT_FDCWD</workspace>, "/workspace/tests", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 11</workspace/tests>
38    getdents64(11</workspace/tests>, 0x5f6465f62d90 /* 3 entries */, 32768) = 80
38    newfstatat(AT_FDCWD</workspace>, "/workspace/tests/conftest.py", 0x7fff570fb6c0, 0) = -1 ENOENT (No such file or directory)
38    openat(AT_FDCWD</workspace>, "/workspace/tests/test_tax.py", O_RDONLY|O_CLOEXEC) = 11</workspace/tests/test_tax.py>
38    newfstatat(AT_FDCWD</workspace>, "/workspace/tests/test_tax.py/pyvenv.cfg", 0x7fff570faef0, 0) = -1 ENOTDIR (Not a directory)
38    openat(AT_FDCWD</workspace>, "/workspace/src/tax.py", O_RDONLY|O_CLOEXEC) = 11</workspace/src/tax.py>
38    openat(AT_FDCWD</workspace>, "/tmp/pytest-of-checker", O_RDONLY) = 4</tmp/pytest-of-checker>
"""


def accesses():
    return {(a.path, a.kind) for a in parse_strace(TRACE)}


def test_parser_resolves_relative_paths_and_marks_missing():
    acc = accesses()
    assert ("/workspace/Makefile", "used") in acc
    assert ("/workspace/test", "missing") in acc  # a file named like the make target would satisfy it
    assert ("/workspace/bin/pytest", "missing") in acc  # PATH search
    assert ("/workspace/tests", "listed") in acc


def test_child_inherits_cwd_and_follows_chdir():
    assert ("/workspace/sub/conf.ini", "missing") in accesses()


def test_enotdir_is_ignored():
    assert not any("pyvenv.cfg" in p for p, _ in accesses())


KINDS = {
    "/workspace": "dir", "/workspace/tests": "dir", "/workspace/src": "dir", "/workspace/sub": "dir",
    "/workspace/bin": "missing", "/workspace/Makefile": "file", "/workspace/pyproject.toml": "file",
    "/workspace/tests/test_tax.py": "file", "/workspace/src/tax.py": "file",
}


def closure():
    return derive(parse_strace(TRACE), KINDS, ["src"])


def test_used_files_are_read_only_and_their_folders_locked():
    cl = closure()
    assert cl.files == ["/workspace/Makefile", "/workspace/pyproject.toml", "/workspace/tests/test_tax.py"]
    assert {"/workspace", "/workspace/tests"} <= set(cl.dirs)


def test_missing_paths_in_locked_folders_need_no_tombstone():
    cl = closure()
    assert {"/workspace/test", "/workspace/tests/conftest.py", "/workspace/bin/pytest"} <= set(cl.covered)
    assert cl.tombstones == ["/workspace/sub/conf.ini"]  # /workspace/sub was only stat'ed, so it is not locked


def test_code_under_test_is_a_hole_not_protected():
    cl = closure()
    assert "/workspace/src/tax.py" in cl.holes
    assert "/workspace/src" not in cl.dirs
    assert not any(f.startswith("/workspace/src") for f in cl.files)


def test_shared_tmp_paths_are_reported():
    assert closure().shared == ["/tmp/pytest-of-checker"]


def test_code_under_test_can_be_a_single_file():
    cl = derive([Access("/workspace/six.py", "used"), Access("/workspace/test_six.py", "used")],
                {"/workspace/six.py": "file", "/workspace/test_six.py": "file"}, ["six.py"])
    assert cl.files == ["/workspace/test_six.py"] and cl.holes == ["/workspace/six.py"]


def test_whole_repo_as_code_under_test_still_protects_tests_and_conftests():
    acc = [Access("/workspace/pkg/mod/tests/test_a.py", "used"), Access("/workspace/pkg/mod/core.py", "used"),
           Access("/workspace/pkg/mod/conftest.py", "missing"), Access("/workspace/pkg/mod/tests", "listed")]
    kinds = {"/workspace/pkg/mod/tests/test_a.py": "file", "/workspace/pkg/mod/core.py": "file",
             "/workspace/pkg/mod/conftest.py": "missing", "/workspace/pkg/mod": "dir", "/workspace/pkg": "dir",
             "/workspace/pkg/mod/tests": "dir"}
    cl = derive(acc, kinds, [""], cut_keeps_oversight=True)
    assert cl.files == ["/workspace/pkg/mod/tests/test_a.py"]
    assert "/workspace/pkg/mod/core.py" in cl.holes
    assert cl.tombstones == ["/workspace/pkg/mod/conftest.py"]  # its folder is editable code, so it needs one
    assert "/workspace/pkg/mod/tests" in cl.dirs


def test_parents_of_locked_folders_are_locked_too():
    cl = derive([Access("/workspace/.venv/lib/py/site", "listed")], {"/workspace/.venv/lib/py/site": "dir"}, ["src"])
    assert {"/workspace/.venv", "/workspace/.venv/lib", "/workspace/.venv/lib/py"} <= set(cl.dirs)


def test_sys_path_folders_are_locked_unless_code_under_test():
    cl = derive([Access("/workspace/lib/x.py", "used")], {"/workspace/lib/x.py": "file", "/workspace/lib": "dir"},
                ["src"], sys_path_dirs=["/workspace/plugins", "/workspace/src"])
    assert "/workspace/plugins" in cl.dirs and "/workspace/src" not in cl.dirs
