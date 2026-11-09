"""Offline replay of the origin check (policy/origin_gate.py) over existing AgentDojo Banking logs.

    python eval/replay_origin_gate.py                  # results/raw_logs.jsonl
    python eval/replay_origin_gate.py --log other.jsonl

No API calls.  Each logged transcript is walked in order: the trusted list starts from the user request, each
payment call is judged before that step's results are absorbed (as it would be live), and every successful
tool result is absorbed afterwards.  Calls that the logged defense already blocked are skipped (they never
ran, so they changed nothing).

This shows which payment calls the check WOULD block.  It cannot show how the agent reacts to a block, so the
"estimated" numbers are estimates, not measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.origin_gate import PAYMENT_TOOLS, OriginGate  # noqa: E402


def replay_run(rec: dict) -> dict:
    """Walk one run.  Returns the payment calls that actually ran and which of them the check would block."""
    msgs = rec["messages"]
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    gate = OriginGate.for_request(user)
    executed, blocked = [], []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
            results = []
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "tool" and len(results) < len(calls):
                results.append(msgs[j])
                j += 1
            for k, c in enumerate(calls):
                res = results[k] if k < len(results) else None
                failed = res is None or res.get("error") is not None
                if c["function"] in PAYMENT_TOOLS and not failed:  # the call really executed
                    d = gate.check(c["function"], c["args"])
                    executed.append((c["function"], c["args"]))
                    if not d.allowed:
                        blocked.append((c["function"], c["args"]))
            for k, c in enumerate(calls):  # results become visible only after the whole step
                res = results[k] if k < len(results) else None
                if res is not None:
                    gate.absorb(c["function"], res["content"], failed=res.get("error") is not None)
            i = j
        else:
            i += 1
    return {"executed": executed, "blocked": blocked}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(ROOT / "results" / "raw_logs.jsonl"))
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.log, encoding="utf-8") if line.strip()]
    recs = [r for r in recs if not r.get("error")]
    stats: dict[str, Counter] = defaultdict(Counter)
    detail = []
    for r in recs:
        rp = replay_run(r)
        key = f"{r['condition']}|{'attack' if r.get('injection_task') else 'clean'}"
        s = stats[key]
        s["runs"] += 1
        s["payment_calls_executed"] += len(rp["executed"])
        s["payment_calls_blocked"] += len(rp["blocked"])
        if rp["blocked"]:
            s["runs_with_block"] += 1
        if r.get("injection_task"):
            s["attack_success_logged"] += bool(r["attack_success"])
            s["attack_success_and_blocked"] += bool(r["attack_success"] and rp["blocked"])
        s["utility_logged"] += bool(r["utility"])
        s["utility_pass_and_blocked"] += bool(r["utility"] and rp["blocked"])
        if rp["blocked"]:
            detail.append((key, r["user_task"], r.get("injection_task"), r["attack_success"], r["utility"],
                           [(f, a.get("recipient")) for f, a in rp["blocked"]]))

    for key in sorted(stats):
        print(f"\n== {key}")
        for k, v in stats[key].items():
            print(f"  {k}: {v}")
    print("\nRuns where the check would block at least one executed payment call "
          "(condition|kind, user task, injection task, logged attack_success, logged utility, blocked calls):")
    for d in sorted(detail, key=lambda x: (x[0], x[1], str(x[2]))):
        print(" ", d)


if __name__ == "__main__":
    main()
