"""Utility % and ASR % from the raw JSONL logs (spec Section 8).

* utility (no attack)    = fraction of no-injection runs where AgentDojo's ``utility()`` is True
* utility (under attack) = fraction of injection runs where the *user* task was still completed
* ASR                    = fraction of injection runs where AgentDojo's ``security()`` is True
                           (the attacker's goal was achieved)

Runs that ended in an infrastructure error (API failure, ...) are excluded and counted separately.
By default the two conditions are compared on the *paired* subset of runs that finished without error
in both conditions, so the table is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

CONDITIONS = ("no_defense", "progent", "progent_recheck")
LABELS = {"no_defense": "No defense", "progent": "Progent (ours)", "progent_recheck": "Progent + choice check"}


def run_key(r: dict) -> tuple:
    return (r.get("model"), r["condition"], r["attack"], r["user_task"], r.get("injection_task"))


def load_records(path: Path, model: str | None = None) -> list[dict]:
    """Read the JSONL log; if a run appears several times (retries) the latest entry wins."""
    latest: dict[tuple, dict] = {}
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                latest[run_key(r)] = r
    recs = list(latest.values())
    if model is None and recs:  # default: the model with the most records
        counts = defaultdict(int)
        for r in recs:
            counts[r.get("model")] += 1
        model = max(counts, key=counts.get)
    return [r for r in recs if r.get("model") == model]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def _rate(flags: list[bool]) -> dict:
    n, k = len(flags), sum(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": (100 * k / n) if n else float("nan"), "ci": (100 * lo, 100 * hi)}


def compute(records: list[dict], paired: bool = True) -> dict:
    ok = [r for r in records if not r.get("error")]
    errors = defaultdict(int)
    for r in records:
        if r.get("error"):
            errors[r["condition"]] += 1

    # a run is identified (apart from the condition) by (attack, user_task, injection_task)
    by_id: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in ok:
        by_id[(r["attack"], r["user_task"], r.get("injection_task"))][r["condition"]] = r
    present = [c for c in CONDITIONS if any(c in v for v in by_id.values())]  # conditions that were actually run
    if paired:
        by_id = {k: v for k, v in by_id.items() if all(c in v for c in present)}

    out: dict = {"errors": dict(errors), "paired": paired, "conditions": {}}
    for cond in present:
        runs = [v[cond] for v in by_id.values() if cond in v]
        noatk = [r for r in runs if r["attack"] == "none"]
        atk = [r for r in runs if r["attack"] != "none"]
        c = {
            "utility_no_attack": _rate([r["utility"] for r in noatk]),
            "utility_under_attack": _rate([r["utility"] for r in atk]),
            "asr": _rate([bool(r["attack_success"]) for r in atk]),
        }
        # breakdowns
        per_inj = defaultdict(list)
        for r in atk:
            per_inj[r["injection_task"]].append(bool(r["attack_success"]))
        c["asr_by_injection_task"] = {k: _rate(v) for k, v in sorted(per_inj.items())}
        c["blocked_calls_mean"] = _mean([r["progent"]["n_blocked"] for r in runs if r.get("progent")])
        c["policy_llm_calls_mean"] = _mean([r["progent"]["n_policy_llm_calls"] for r in runs if r.get("progent")])
        c["expansions_mean"] = _mean([r["progent"]["n_expansions"] for r in runs if r.get("progent")])
        c["tool_calls_mean"] = _mean([r["n_tool_calls"] for r in runs])
        out["conditions"][cond] = c
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(rate: dict) -> str:
    if rate["n"] == 0:
        return "n/a"
    return f"{rate['pct']:.1f}% ({rate['k']}/{rate['n']})"


def render_table(m: dict) -> str:
    rows = [
        ("Utility, no attack", "utility_no_attack"),
        ("Utility, under attack", "utility_under_attack"),
        ("ASR (attack success rate, lower is better)", "asr"),
    ]
    conds = [c for c in CONDITIONS if c in m["conditions"]]
    lines = ["| Metric | " + " | ".join(LABELS[c] for c in conds) + " |", "|---|" + "---|" * len(conds)]
    for label, key in rows:
        cells = []
        for cond in conds:
            r = m["conditions"][cond][key]
            ci = f" [95% CI {r['ci'][0]:.0f}-{r['ci'][1]:.0f}%]" if r["n"] else ""
            cells.append(_fmt(r) + ci)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_injection_breakdown(m: dict) -> str:
    conds = [c for c in CONDITIONS if c in m["conditions"]]
    ids = sorted(set().union(*[set(m["conditions"][c]["asr_by_injection_task"]) for c in conds]))
    if not ids:
        return ""
    lines = ["| Injection task | " + " | ".join(f"ASR {LABELS[c]}" for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for i in ids:
        lines.append(f"| {i} | " + " | ".join(
            _fmt(m["conditions"][c]["asr_by_injection_task"].get(i, {"n": 0})) for c in conds) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(config.RAW_LOGS_PATH))
    ap.add_argument("--model", default=None)
    ap.add_argument("--unpaired", action="store_true")
    a = ap.parse_args()
    recs = load_records(Path(a.log), a.model)
    if not recs:
        print(f"no records in {a.log}")
        return
    m = compute(recs, paired=not a.unpaired)
    print(render_table(m))
    print()
    print(render_injection_breakdown(m))
    print("\nerrored runs:", m["errors"])


if __name__ == "__main__":
    main()
