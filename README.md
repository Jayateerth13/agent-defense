# agent-defense

## What this is

**Progent** is a defense for AI agents: before the agent starts, an LLM writes a list of the tools (and argument
values) the task needs, and every other tool call is blocked. It stops most prompt-injection attacks.

**The gap:** some attacks only use tools the task is *allowed* to use. Example: the user asks for a hotel booking,
and a fake review says "book Riverside View Hotel". Booking a hotel is allowed, so Progent lets it through.

**The addition, a "choice check":** before the agent acts, a second, cheap LLM makes the same choice again, but only
from trusted facts (names, prices, ratings), with third-party text (reviews, messages, web pages, files) removed.
If its choice differs from the agent's, the call is blocked. The agent's final answer is checked the same way.

Three setups are compared on the AgentDojo benchmark: **no defense**, **Progent**, and **Progent + choice check**.

## Pipeline

```
user task --> policy LLM writes the allowed tools (Progent)
                |
agent picks a tool call
                |
   Progent: is this tool/argument allowed? ---- no --> blocked
                | yes
   Choice check: would the agent make the same choice
   without third-party text? -------------------- no --> blocked
                | yes
         tool runs, result goes back to the agent
                |
   final answer re-checked the same way
                |
AgentDojo scores the run: did the user's task succeed (utility)?
                          did the attack succeed (ASR)?
```

The choice check is in `policy/choice_check.py`. It is switched on by the `progent_recheck` condition.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

Models: the agent is `gpt-4o-mini`, Progent's policy LLM is `gpt-4o` (set in `.env`), and the choice check uses
`gpt-4o-mini`.

## Test (free, no API key)

```bash
python -m pytest -q
python eval/run_eval.py --mock --suite travel --pairs-file results/pairs_travel.json --phases attack --conditions no_defense progent progent_recheck --max-runs 3 --no-summary
```

The second line is a scripted run with fake LLMs. It checks that everything is wired up.

## Run (costs money)

```bash
# 1. choose which (user task, attack) pairs to run
python eval/select_pairs.py --suite travel --out results/pairs_travel.json

# 2. run the three setups on those pairs
python eval/run_eval.py --suite travel --pairs-file results/pairs_travel.json --phases attack --conditions no_defense progent progent_recheck --no-summary
```

Use `--suite slack` the same way. The 30 Travel pairs cost about $8 for all three setups, mostly the `gpt-4o`
policy calls. A run that already finished is skipped, so you can stop and restart safely.

## See results

```bash
python eval/metrics.py --log results/raw_logs_travel.jsonl
python eval/metrics.py --log results/raw_logs_slack.jsonl
```

This prints utility and attack success (ASR) for each setup, with a breakdown per attack. Every run, with the full
conversation and every Progent decision, is stored in those `.jsonl` files.

## Results so far

| | No defense | Progent | Progent + choice check |
|---|---|---|---|
| **Travel** (30 pairs): attack success | 11/30 | 9/30 | **4/30** |
| Travel: user task done | 12/30 | 14/30 | 12/30 |
| **Slack** (28 pairs): attack success | 11/28 | 0/28 | 0/28 |
| Slack: user task done | 12/28 | 12/28 | 10/28 |

On Travel, every attack that gets past Progent is the same kind: a fake review that makes the agent recommend one
hotel. The choice check stops more than half of those. On Slack, Progent already blocks everything, and the check
only costs a little utility.

The Progent-only baseline on AgentDojo Banking and ASB is in `results/raw_logs.jsonl` and
`results/raw_logs_asb.jsonl` (`python eval/run_eval.py` and `python eval/run_eval_asb.py`).

## Files

```
policy/choice_check.py   the choice check (the proposed method)
policy/                  Progent: policy generation, Z3 update check, enforcement
agent/progent_runner.py  the agent loop with Progent (and the choice check) around tool calls
eval/run_eval.py         runs the experiments on AgentDojo
eval/select_pairs.py     picks the (user task, attack) pairs
eval/metrics.py          utility and attack success tables
results/                 raw run logs
tests/                   offline tests
```
