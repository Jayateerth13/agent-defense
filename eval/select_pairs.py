"""Choose which (user task, injection task) pairs to run on a suite.

    python eval/select_pairs.py --suite travel --n-in-scope 20 --n-control 10 --out results/pairs_travel.json

A pair is *in scope for tool-level blocking* when the injected action uses only tools that the user's task
already uses (or needs no tool at all, e.g. a recommendation): Progent cannot refuse such an action by
removing a tool.  The selection takes ``--n-in-scope`` of those pairs and ``--n-control`` random pairs from the
others, with a fixed seed, so the result is reproducible.  Classification uses AgentDojo's own ground-truth tool
calls (no API calls).  Read-only tools (``get_*``, ``read_*``, ``search_*``, ``list_*``, ``check_*``) do not
count as actions, except visiting a web page.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import config  # noqa: E402

READ_PREFIX = ("get_", "read_", "search_", "list_", "check_")
ACTION_READS = {"get_webpage"}


def _actions(calls) -> set[str]:
    return {f for f in calls if f in ACTION_READS or not f.startswith(READ_PREFIX)}


def classify_pairs(suite_name: str) -> dict[tuple[str, str], str]:
    """{(user, injection): 'in_scope' | 'new_tool'} from ground-truth tool calls."""
    s = get_suite(config.BENCHMARK_VERSION, suite_name)
    env0 = s.load_and_inject_default_environment({})
    user = {u: _actions(c.function for c in t.ground_truth(t.init_environment(env0.model_copy(deep=True))))
            for u, t in s.user_tasks.items()}
    out = {}
    for i, it in s.injection_tasks.items():
        acts = _actions(c.function for c in it.ground_truth(env0.model_copy(deep=True)))
        for u, ua in user.items():
            out[(u, i)] = "in_scope" if (not acts or acts <= ua) else "new_tool"
    return out


def select(suite_name: str, n_in: int, n_ctrl: int, seed: int = 0) -> dict:
    cls = classify_pairs(suite_name)
    rng = random.Random(seed)
    key = lambda p: (int(p[0].split("_")[-1]), int(p[1].split("_")[-1]))  # noqa: E731
    inside = sorted((p for p, c in cls.items() if c == "in_scope"), key=key)
    rest = sorted((p for p, c in cls.items() if c != "in_scope"), key=key)
    chosen_in = sorted(rng.sample(inside, min(n_in, len(inside))), key=key)
    chosen_ctrl = sorted(rng.sample(rest, min(n_ctrl, len(rest))), key=key)
    return {"suite": suite_name, "seed": seed, "n_in_scope_available": len(inside),
            "in_scope": [list(p) for p in chosen_in], "control": [list(p) for p in chosen_ctrl]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=["banking", "slack", "travel"])
    ap.add_argument("--n-in-scope", type=int, default=20)
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sel = select(a.suite, a.n_in_scope, a.n_control, a.seed)
    out = Path(a.out) if a.out else ROOT / "results" / f"pairs_{a.suite}.json"
    out.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(f"{a.suite}: {len(sel['in_scope'])} in-scope (of {sel['n_in_scope_available']}) + "
          f"{len(sel['control'])} control -> {out}")


if __name__ == "__main__":
    main()
