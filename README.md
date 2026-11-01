# agent-defense

A small, own-code implementation of **Progent** (Shi et al., 2025, programmable privilege control for LLM agents)
on top of the `agentdojo` package, evaluated against an undefended agent on AgentDojo Banking and ASB. It is the
baseline for research on defending AI agents.

## Setup

Python 3.13 was used for the reported runs.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Put `OPENAI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`. The tests and `--mock` runs need no key.

## Running

```bash
python -m pytest -q                            # 69 offline tests, no API key needed
python eval/run_eval.py --mock --security-subset 6 --no-summary      # offline smoke test

python eval/run_eval.py                        # AgentDojo Banking: 144 attack cases + 16 clean tasks, x2 conditions
python eval/run_eval_asb.py                    # ASB financial_analyst_agent: 200 attack cases + 5 clean tasks, x2 conditions
python eval/make_summary.py                    # rebuilds results/summary.md from the raw logs
```

**Real runs cost money.** A Progent run on AgentDojo cost about $0.06 with `gpt-4o` as the policy LLM, so use
`--security-subset N` to run a random sample. Runs resume from the log: anything that already finished without error
is skipped, so `results/raw_logs.jsonl` must be moved aside to re-run those cases.

The first ASB run clones [agiresearch/ASB](https://github.com/agiresearch/ASB) into `third_party/` (needs `git` and
network access). On Windows, keep this repo in a folder under about 140 characters or run
`git config --global core.longpaths true` first, because ASB has long file paths. If that clone fails, delete
`third_party/ASB` before retrying.

`results/summary.md` is generated locally and is not in the repo; its hand-written discussion sections are also local,
so a fresh clone gets the tables only.

Useful flags:

| Flag | Effect |
|---|---|
| `--security-subset N` | Random subset of the attack cases (`--seed` sets the sample) |
| `--workers K` | Parallel runs; mind rate limits |
| `--max-errors N` | Stop after N errored runs (default 1) |
| `--mock` | Offline run with scripted LLMs; writes to `results/raw_logs*_mock.jsonl`, never to the real logs |
| `--no-summary` | Do not rebuild `results/summary.md` at the end |
| `--no-update` | ASB only: no policy updates; logs to `results/raw_logs_asb_noupdate.jsonl` |

## Models

| Variable | Role | Default |
|---|---|---|
| `PROGENT_MODEL` | Agent LLM | `gpt-4o-mini` when `OPENAI_API_KEY` is set |
| `PROGENT_POLICY_MODEL` | Policy LLM | Same as the agent model |

Set `PROGENT_POLICY_MODEL=gpt-4o`, as `.env.example` does: it matches the official code, and in a probe the mini models
returned empty policies. Only one client is created, so both models must come from the same provider. See `config.py`
for the remaining options.

## Layout

```
config.py            models, benchmark settings and paths
llm_client.py        retrying OpenAI-compatible client (OpenAI / Gemini), token counts
policy/
  schema.py          policy data structures, JSON-Schema validation, tool-spec helpers
  convert.py         official policy wire format <-> internal policy
  generator.py       LLM calls: initial policy + policy updates (official prompts)
  smt_check.py       Z3 narrowing-vs-expansion check (+ bounded probe fallback)
  enforcer.py        runtime check of a tool call against the policy (paper Algorithm 2)
agent/
  baseline_runner.py AgentDojo's standard agent loop, no defense
  progent_runner.py  same loop with ProgentInit + ProgentToolsExecutor around tool calls
  llm.py, common.py  agent LLM element, shared run/log helpers
  mock_llm.py        scripted agent + oracle policy LLM for offline tests
asb/                 ASB adapter: data loader, environment, runner, mocks
eval/
  run_eval.py        AgentDojo experiment driver      run_eval_asb.py   ASB experiment driver
  metrics.py         utility % / ASR % (+ Wilson CIs)  make_summary.py   builds results/summary.md
tests/               offline tests; tests/_*.py are manual scripts that pytest does not collect
results/             raw run logs (.jsonl) and console output
```

## How a Progent run works

1. `ProgentInit` asks the policy LLM for a least-privilege policy from the tool schemas and the trusted user task.
2. Each tool call goes through `ProgentToolsExecutor`. Forbid rules are checked before allow rules, and a call that
   matches no rule is blocked. A blocked call returns an error to the agent, which can try something else
   (`terminate_execution` rules abort the run instead).
3. After each executor step the policy can be updated in two LLM calls: the first decides whether an update is needed
   without seeing the tool result, and the second writes the new policy using it. Z3 classifies each update as
   narrowing or expansion.
4. Expansions are auto-approved (the paper's worst-case setting) unless `PROGENT_ONLY_ALLOW_NARROW=True`. The Z3 label
   is logged and does not otherwise change a decision.

Everything the policy layer did (initial and final policy, every decision, every update with its Z3 verdict) is stored
per run in the `progent` field of the JSONL logs.
