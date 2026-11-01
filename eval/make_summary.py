"""Assemble ``results/summary.md`` from the raw logs.

Tables / statistics are regenerated from ``results/raw_logs*.jsonl`` on every call.  The interpretive
text (why the numbers look the way they do) is hand-written after inspecting the runs and lives in
``results/narrative_agentdojo.md`` / ``results/narrative_asb.md``; it is spliced in verbatim.

    python eval/make_summary.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from eval import metrics  # noqa: E402

PAPER_AGENTDOJO = "ASR 39.9% -> 1.0%, utility ~79% -> ~76% (overall AgentDojo, 4 suites, 629 security cases)"
PAPER_ASB = "ASR 70.3% -> 3.9% (overall ASB)"


def _read(name: str) -> str | None:
    p = config.RESULTS_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _progent_stats(records: list[dict]) -> str:
    runs = [r for r in records if r["condition"] == "progent" and r.get("progent") and not r.get("error")]
    if not runs:
        return ""
    n = len(runs)
    blocked = sum(r["progent"]["n_blocked"] for r in runs)
    runs_with_block = sum(1 for r in runs if r["progent"]["n_blocked"] > 0)
    upd = sum(r["progent"]["n_updates"] for r in runs)
    exp = sum(r["progent"]["n_expansions"] for r in runs)
    llm = sum(r["progent"]["n_policy_llm_calls"] for r in runs)
    methods = Counter()
    for r in runs:
        for e in r["progent"]["events"]:
            if e.get("type") == "update" and "diff" in e:
                for c in e["diff"]["checks"]:
                    methods[c["method"]] += 1
    term = sum(1 for r in runs for e in r["progent"]["events"] if e.get("type") == "terminated")
    return (
        f"- Progent runs analysed: {n}; blocked tool calls: {blocked} (in {runs_with_block} runs); "
        f"runs terminated by `terminate_execution`: {term}\n"
        f"- Policy updates applied: {upd} ({exp} classified as *expansion* and auto-approved, {upd - exp} narrowing)\n"
        f"- Policy-LLM calls: {llm} total, {llm / n:.1f} per run\n"
        f"- Z3 check path per tool comparison: {dict(methods) or 'n/a'} (`z3` = exact symbolic check, "
        f"`probe` = bounded fallback)\n"
    )


def _section(title: str, log: Path, paper: str, narrative_file: str, status_file: str | None, level: str = "##") -> str:
    recs = metrics.load_records(log)
    out = [f"{level} {title}\n"]
    if not recs:
        blocker = _read(status_file) if status_file else None
        out.append(blocker or f"_No results logged yet (`{log.name}` is empty or missing)._\n")
        return "\n".join(out)
    m = metrics.compute(recs)
    model = recs[0]["model"]
    policy_models = sorted({r["policy_model"] for r in recs if r.get("policy_model")})
    n_atk = m["conditions"]["no_defense"]["asr"]["n"]
    n_clean = m["conditions"]["no_defense"]["utility_no_attack"]["n"]
    out.append(
        f"Agent LLM: `{model}`; policy LLM: `{', '.join(policy_models) or model}`. Paired runs analysed: {n_clean} no-attack user tasks and "
        f"**{n_atk} attack cases** (user task x injection task) per condition. Errored runs excluded: "
        f"{m['errors'] or 'none'}.\n"
    )
    out.append(metrics.render_table(m) + "\n")
    out.append("Progent behaviour:\n\n" + _progent_stats(recs))
    out.append("<details><summary>ASR by injection task</summary>\n\n" + metrics.render_injection_breakdown(m)
               + "\n\n</details>\n")
    out.append(f"**Original paper ({paper}).** These numbers are *not directly comparable 1:1*: see "
               f"\"Differences from the original paper\" below.\n")
    narrative = _read(narrative_file)
    out.append(narrative + "\n" if narrative else "_Discussion pending: write `results/" + narrative_file + "`._\n")
    return "\n".join(out)


def write_summary() -> Path:
    parts = [
        "# Progent baseline reproduction - results\n",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `eval/make_summary.py` from `results/raw_logs.jsonl` / "
        "`results/raw_logs_asb.jsonl`._\n",
        _section("1. AgentDojo - Banking suite (primary)", config.RAW_LOGS_PATH, PAPER_AGENTDOJO,
                 "narrative_agentdojo.md", None),
        _section("2. ASB - Agent Security Bench (secondary)", config.RAW_LOGS_ASB_PATH, PAPER_ASB,
                 "narrative_asb.md", "asb_status.md"),
        _read("differences.md") or "",
        _read("choices.md") or "",
    ]
    config.SUMMARY_PATH.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return config.SUMMARY_PATH


if __name__ == "__main__":
    print(write_summary())
