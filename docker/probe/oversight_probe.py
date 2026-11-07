"""Closure probe: records the loaded modules and ``sys.path`` of every Python process the check starts.

Python finds modules by listing folders, so a module the check imports never shows up in ``strace`` as a
missing file.  The closure builder uses this record to protect the names that could shadow those modules.
Loaded by ``oversight_probe.pth`` in the system site-packages, so a check that sets its own ``PYTHONPATH``
is still recorded.  Does nothing unless ``OVERSIGHT_PROBE_DIR`` is set.
"""

import atexit
import json
import os
import sys

_OUT = os.environ.get("OVERSIGHT_PROBE_DIR")

if _OUT:

    def _dump():
        try:
            names = sorted({m.split(".")[0] for m in list(sys.modules)})
            with open(os.path.join(_OUT, f"probe-{os.getpid()}.json"), "w") as f:
                json.dump({"cwd": os.getcwd(), "path": list(sys.path), "modules": names}, f)
        except OSError:
            pass

    atexit.register(_dump)
