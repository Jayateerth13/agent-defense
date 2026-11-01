"""ASB (secondary target): ONE scenario (financial_analyst_agent) x ONE attack (observation / indirect
prompt injection), no-defense vs Progent - reusing the very same ``agent/`` + ``policy/`` code.

    python eval/run_eval_asb.py                       # 5 tasks x 40 attacker tools x 2 conditions + clean runs
    python eval/run_eval_asb.py --security-subset 40  # random subset of the 200 (task, attacker tool) cases
    python eval/run_eval_asb.py --mock                # offline smoke test
    python eval/run_eval_asb.py --no-update           # Progent WITHOUT the policy-update step (official ASB wiring);
                                                      # logs to results/raw_logs_asb_noupdate.jsonl

Fail-fast: the first errored run (API failure after retries, bug, ...) stops the whole evaluation with exit
code 2 (``--max-errors N`` tolerates N-1 errors).  The errored run is still written to the log (with its
traceback) for diagnosis; finished runs are kept and re-running resumes where it stopped.

Results go to ``results/raw_logs_asb.jsonl`` (same record schema as the AgentDojo log).
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import config  # noqa: E402
import llm_client  # noqa: E402
from agent import baseline_runner, progent_runner  # noqa: E402
from asb import env as asb_env  # noqa: E402
from asb.data import ATTACK_TYPE, SCENARIO, load_scenario  # noqa: E402
from asb.runner import run_asb_case  # noqa: E402
from eval import metrics  # noqa: E402
from eval.run_eval import RunFailed, append, cost_report  # noqa: E402  (shared with the AgentDojo driver)

ATTACK_LABEL = "observation_prompt_injection"  # ASB's name for indirect prompt injection

@dataclass(frozen=True)
class Job:
    condition: str
    task_idx: int
    attacker_tool: str | None  # None => no attack


def build_jobs(sc, args) -> list[Job]:
    tools = [t for t in sc.attack_tools
             if args.attack_tools == "all" or (t["Aggressive"] == "True") == (args.attack_tools == "agg")]
    tasks = args.tasks if args.tasks is not None else list(range(len(sc.tasks)))
    pairs = [(i, t["Attacker Tool"]) for i in tasks for t in tools]
    if args.security_subset and args.security_subset < len(pairs):
        pairs = sorted(random.Random(args.seed).sample(pairs, args.security_subset))
    jobs: list[Job] = []
    if "attack" in args.phases:
        for i, a in pairs:
            jobs += [Job(c, i, a) for c in args.conditions]
    if "utility" in args.phases:
        for i in tasks:
            jobs += [Job(c, i, None) for c in args.conditions]
    return jobs


def key_of(model: str, j: Job) -> tuple:
    return (model, j.condition, ATTACK_LABEL if j.attacker_tool else "none", f"task_{j.task_idx}", j.attacker_tool)


def execute(sc, job: Job, args, model: str) -> dict:
    tool = next((t for t in sc.attack_tools if t["Attacker Tool"] == job.attacker_tool), None)
    llm = complete = None
    if args.mock:
        from asb.mock import ASBScriptedLLM, oracle_policy_complete

        llm, complete = ASBScriptedLLM(sc, tool, fooled=True), oracle_policy_complete(sc)
    sp = asb_env.system_prompt(sc)
    if job.condition == "progent":
        pipeline = progent_runner.build_pipeline(llm, complete, system_prompt=sp, task_type="asb",
                                                 update_enabled=not args.no_update)
    else:
        pipeline = baseline_runner.build_pipeline(llm, system_prompt=sp)
    out = run_asb_case(pipeline, sc, job.task_idx, tool)
    rec = {
        "benchmark": "asb", "scenario": sc.name, "attack_type": ATTACK_TYPE, "model": model,
        "condition": job.condition, "attack": ATTACK_LABEL if tool else "none",
        "user_task": f"task_{job.task_idx}", "task_text": sc.tasks[job.task_idx],
        "injection_task": job.attacker_tool,
        "attacker_aggressive": tool["Aggressive"] if tool else None,
        "policy_model": config.POLICY_MODEL if job.condition == "progent" and not args.mock else None,
        "policy_updates": ("off" if args.no_update else "on") if job.condition == "progent" else None,
        "utility": out.utility, "attack_success": out.security, "error": out.error,
        "n_tool_calls": out.n_tool_calls, "duration_s": out.duration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "messages": out.messages, "progent": None,
    }
    if out.error:
        rec["traceback"] = out.extra.get("traceback")
    if job.condition == "progent":
        sess = pipeline.progent_init.session  # type: ignore[attr-defined]
        rec["progent"] = sess.summary() if sess else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(metrics.CONDITIONS), choices=list(metrics.CONDITIONS))
    ap.add_argument("--phases", nargs="+", default=["attack", "utility"], choices=["attack", "utility"])
    ap.add_argument("--tasks", nargs="*", type=int, default=None, help="task indices (default all 5)")
    ap.add_argument("--attack-tools", choices=["all", "agg", "non-agg"], default="all")
    ap.add_argument("--security-subset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-update", action="store_true",
                    help="Progent without the policy-update step (official ASB wiring); logs to a separate file")
    ap.add_argument("--max-errors", type=int, default=1,
                    help="abort after this many errored runs (default 1 = stop at the first error)")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    model = "mock" if args.mock else config.MODEL
    default_name = ("raw_logs_asb_mock.jsonl" if args.mock else
                    "raw_logs_asb_noupdate.jsonl" if args.no_update else "raw_logs_asb.jsonl")
    log = Path(args.log) if args.log else (config.RESULTS_DIR / default_name)
    log.parent.mkdir(parents=True, exist_ok=True)
    if not args.mock:
        llm_client.get_client()
    sc = load_scenario(SCENARIO)
    jobs = build_jobs(sc, args)
    done = {metrics.run_key(r) for r in metrics.load_records(log, model) if not r.get("error")}
    todo = [j for j in jobs if key_of(model, j) not in done]
    if args.max_runs:
        todo = todo[: args.max_runs]
    print(f"ASB scenario={sc.name} agent={model} policy_model={'-' if args.mock else config.POLICY_MODEL} "
          f"policy_updates={'off' if args.no_update else 'on'} seed={args.seed} subset={args.security_subset} "
          f"total runs={len(jobs)} to run={len(todo)} log={log}", flush=True)

    n = 0
    n_errors = 0
    aborted: str | None = None

    stop = threading.Event()  # set on the first fatal problem: jobs that have not started yet are skipped

    def work(j: Job) -> dict | None:
        nonlocal n_errors
        if stop.is_set():
            return None
        rec = execute(sc, j, args, model)
        if (rec["error"] or "").startswith("QuotaExhaustedError"):
            stop.set()
            raise llm_client.QuotaExhaustedError(rec["error"])  # not logged -> retried on the next invocation
        append(log, rec)  # errored runs are logged too (with traceback); they are excluded from the metrics
        if rec["error"]:
            n_errors += 1
            if n_errors >= args.max_errors:
                stop.set()
                raise RunFailed(f"{n_errors} errored run(s); last: {j.condition} task_{j.task_idx} "
                                f"{j.attacker_tool or '-'}: {rec['error'][:300]}")
        return rec

    def report(j: Job, rec: dict) -> None:
        nonlocal n
        n += 1
        tag = "ERR " + rec["error"][:80] if rec["error"] else (
            f"utility={rec['utility']}" + (f" attack_success={rec['attack_success']}" if j.attacker_tool else ""))
        print(f"[{n}/{len(todo)}] {j.condition:10s} task_{j.task_idx} {j.attacker_tool or '-':32s} {tag}", flush=True)

    t0 = time.time()
    ex = ThreadPoolExecutor(max(1, args.workers))
    try:
        futs = {ex.submit(work, j): j for j in todo}
        for f in as_completed(futs):
            rec = f.result()
            if rec is not None:
                report(futs[f], rec)
    except llm_client.QuotaExhaustedError as e:
        aborted = f"API QUOTA / BILLING LIMIT: {str(e)[:300]}"
    except RunFailed as e:
        aborted = f"RUN FAILED: {e}"
    except KeyboardInterrupt:
        aborted = "interrupted by user"
    finally:
        ex.shutdown(wait=True, cancel_futures=True)  # nothing new starts after an abort
    if aborted:
        print(f"\n*** ABORTED - {aborted}\n    Finished runs are saved; re-run the same command to resume.", flush=True)

    print(f"\nfinished in {time.time() - t0:.0f}s; LLM calls: {dict(llm_client.CALL_COUNTS)}\n{cost_report()}",
          flush=True)
    m = metrics.compute(metrics.load_records(log, model))
    print("\n" + metrics.render_table(m))
    if m["errors"]:
        print(f"\nWARNING: errored runs excluded: {m['errors']} - re-run to retry them.")
    if aborted:
        sys.exit(2)  # non-zero exit: the evaluation did NOT complete
    if not args.mock and not args.no_summary and not args.no_update:
        from eval import make_summary

        make_summary.write_summary()
        print(f"summary written to {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()

