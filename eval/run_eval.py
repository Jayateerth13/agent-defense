"""Main entry point: run both conditions on AgentDojo's Banking suite and log every run.

    python eval/run_eval.py                       # everything (resumable; needs an API key in .env)
    python eval/run_eval.py --security-subset 20  # random subset of the 144 (user, injection) pairs
    python eval/run_eval.py --mock                # offline smoke test with scripted LLMs (no key)

Runs (spec Section 7), in priority order: (1) injection runs for both conditions, (2) no-attack
utility runs for both conditions.  Results are appended to ``results/raw_logs.jsonl``; re-running the
script skips every run that already finished without error, so it can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python eval/run_eval.py` from anywhere
    sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import config  # noqa: E402
import llm_client  # noqa: E402
from agent import baseline_runner, progent_runner  # noqa: E402
from agent.common import run_one  # noqa: E402
from eval import metrics  # noqa: E402

_write_lock = threading.Lock()

# USD per 1M tokens (input, output): OpenAI list prices, used only for the end-of-run cost estimate.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


class RunFailed(RuntimeError):
    """Too many errored runs: abort the evaluation (fail-fast)."""


def cost_report() -> str:
    """Token totals split by caller (agent / policy) and an estimated cost; shared with run_eval_asb.py."""
    t = llm_client.TOKENS
    parts, total = [], 0.0
    for tag, model in (("agent", config.MODEL), ("policy", config.POLICY_MODEL)):
        i, o = t.get(f"{tag}_input", 0), t.get(f"{tag}_output", 0)
        key = max((k for k in PRICES if model.startswith(k)), key=len, default=None)
        c = (i * PRICES[key][0] + o * PRICES[key][1]) / 1e6 if key else None
        total += c or 0.0
        parts.append(f"{tag} ({model}): {i:,} in / {o:,} out tokens" + (f" ~ ${c:.3f}" if c is not None else ""))
    return "; ".join(parts) + f"; estimated total ~ ${total:.3f}"


class _NameShim(BasePipelineElement):
    """AgentDojo's attack looks the victim model's prose name up from ``pipeline.name`` and only knows
    its own model list.  We feed it a known name and then override ``model_name`` explicitly."""

    name = "gpt-4o-mini-2024-07-18"

    def query(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


def make_attack(suite):
    attack = load_attack(config.ATTACK_NAME, suite, _NameShim())
    # prose name used in the "important message from me, <user>, to you, <model>" template
    attack.model_name = "AI model developed by Google" if config.provider_for(config.MODEL) == "gemini" else "GPT-4"
    return attack


@dataclass(frozen=True)
class Job:
    condition: str  # no_defense | progent
    user_task: str
    injection_task: str | None  # None => no attack


def build_jobs(suite, attack, args) -> tuple[list[Job], dict, list]:
    user_ids = args.user_tasks or list(suite.user_tasks)
    pairs = [(u, i) for u in user_ids for i in suite.injection_tasks]
    injections: dict[tuple[str, str], dict] = {}
    skipped = []
    for u, i in pairs:
        try:
            injections[(u, i)] = attack.attack(suite.get_user_task_by_id(u), suite.get_injection_task_by_id(i))
        except ValueError as e:  # user task exposes no injection vector
            skipped.append((u, i, str(e)))
    pairs = [p for p in pairs if p in injections]
    if args.security_subset and args.security_subset < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.security_subset)
        pairs.sort(key=lambda p: (int(p[0].split("_")[-1]), int(p[1].split("_")[-1])))
    jobs: list[Job] = []
    if "attack" in args.phases:
        for u, i in pairs:
            jobs += [Job(c, u, i) for c in args.conditions]
    if "utility" in args.phases:
        for u in user_ids:
            jobs += [Job(c, u, None) for c in args.conditions]
    return jobs, injections, skipped


def job_key(model: str, j: Job) -> tuple:
    return (model, j.condition, config.ATTACK_NAME if j.injection_task else "none", j.user_task, j.injection_task)


def execute(suite, job: Job, injections: dict, args, model: str) -> dict:
    user_task = suite.get_user_task_by_id(job.user_task)
    inj_task = suite.get_injection_task_by_id(job.injection_task) if job.injection_task else None
    inj = injections.get((job.user_task, job.injection_task), {}) if inj_task else {}

    llm = complete = None
    if args.mock:
        from agent.mock_llm import ScriptedLLM, oracle_policy_complete

        env = user_task.init_environment(suite.load_and_inject_default_environment({}))
        llm = ScriptedLLM(user_task, inj_task, fooled=True)
        complete = oracle_policy_complete(user_task, env)

    if job.condition == "progent":
        pipeline = progent_runner.build_pipeline(llm, complete, always_allowed=config.ALWAYS_ALLOWED.get(config.SUITE_NAME))
    else:
        pipeline = baseline_runner.build_pipeline(llm)

    out = run_one(suite, pipeline, user_task, inj_task, inj)
    rec = {
        "benchmark": "agentdojo",
        "suite": config.SUITE_NAME,
        "benchmark_version": config.BENCHMARK_VERSION,
        "model": model,
        "policy_model": config.POLICY_MODEL if job.condition == "progent" and not args.mock else None,
        "condition": job.condition,
        "attack": config.ATTACK_NAME if inj_task else "none",
        "user_task": job.user_task,
        "injection_task": job.injection_task,
        "utility": out.utility,
        "attack_success": out.security,
        "error": out.error,
        "n_tool_calls": out.n_tool_calls,
        "duration_s": out.duration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "messages": out.messages,
        "progent": None,
    }
    if out.error:
        rec["traceback"] = out.extra.get("traceback")
    if job.condition == "progent":
        sess = pipeline.progent_init.session  # type: ignore[attr-defined]
        rec["progent"] = sess.summary() if sess else None
    return rec


def append(path: Path, rec: dict) -> None:
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="JSONL log path (default results/raw_logs.jsonl)")
    ap.add_argument("--conditions", nargs="+", default=list(metrics.CONDITIONS), choices=list(metrics.CONDITIONS))
    ap.add_argument("--phases", nargs="+", default=["attack", "utility"], choices=["attack", "utility"])
    ap.add_argument("--user-tasks", nargs="*", default=None, help="subset of user task ids, e.g. user_task_0")
    ap.add_argument("--security-subset", type=int, default=None,
                    help="random subset of N (user, injection) pairs instead of all of Banking's pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (mind API rate limits)")
    ap.add_argument("--max-runs", type=int, default=None, help="debug: stop after N new runs")
    ap.add_argument("--max-errors", type=int, default=1,
                    help="abort after this many errored runs (default 1 = stop at the first error)")
    ap.add_argument("--mock", action="store_true", help="offline smoke test with scripted LLMs; logs to a separate file")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    model = "mock" if args.mock else config.MODEL
    log = Path(args.log) if args.log else (ROOT / "results" / ("raw_logs_mock.jsonl" if args.mock else "raw_logs.jsonl"))
    log.parent.mkdir(parents=True, exist_ok=True)
    if not args.mock:
        llm_client.get_client()  # fail fast if there is no API key

    suite = get_suite(config.BENCHMARK_VERSION, config.SUITE_NAME)
    attack = make_attack(suite)
    jobs, injections, skipped = build_jobs(suite, attack, args)
    for u, i, why in skipped:
        print(f"[skip] {u} x {i}: {why}")

    done = {metrics.run_key(r) for r in metrics.load_records(log, model) if not r.get("error")}
    todo = [j for j in jobs if job_key(model, j) not in done]
    if args.max_runs:
        todo = todo[: args.max_runs]
    print(f"model={model}  total runs={len(jobs)}  already done={len(jobs) - len(todo) if not args.max_runs else '-'}  "
          f"to run={len(todo)}  log={log}")

    t0 = time.time()
    n_done = 0

    stop = threading.Event()  # set on the first fatal problem: jobs that have not started yet are skipped
    n_errors = 0
    aborted: str | None = None

    def work(j: Job) -> dict | None:
        nonlocal n_errors
        if stop.is_set():
            return None
        rec = execute(suite, j, injections, args, model)
        if (rec["error"] or "").startswith("QuotaExhaustedError"):
            stop.set()
            raise llm_client.QuotaExhaustedError(rec["error"])  # not logged -> retried on the next invocation
        append(log, rec)  # errored runs are logged too (with traceback); they are excluded from the metrics
        if rec["error"]:
            n_errors += 1
            if n_errors >= args.max_errors:
                stop.set()
                raise RunFailed(f"{n_errors} errored run(s); last: {j.condition} {j.user_task} "
                                f"{j.injection_task or '-'}: {rec['error'][:300]}")
        return rec

    def report(j: Job, rec: dict) -> None:
        nonlocal n_done
        n_done += 1
        tag = "ERR " + rec["error"][:80] if rec["error"] else (
            f"utility={rec['utility']}" + (f" attack_success={rec['attack_success']}" if j.injection_task else ""))
        print(f"[{n_done}/{len(todo)}] {j.condition:10s} {j.user_task:13s} {j.injection_task or '-':16s} {tag}  "
              f"({rec['duration_s']}s)", flush=True)

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

    recs = metrics.load_records(log, model)
    m = metrics.compute(recs)
    print("\n" + metrics.render_table(m))
    if m["errors"]:
        print(f"\nWARNING: errored runs excluded from metrics: {m['errors']} - re-run the script to retry them.")
    if aborted:
        sys.exit(2)  # non-zero exit: the evaluation did NOT complete
    if not args.mock and not args.no_summary:
        from eval import make_summary

        make_summary.write_summary()
        print(f"summary written to {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()


