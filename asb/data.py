"""Loader for ASB (Agent Security Bench, Zhang et al., ICLR 2025) data.

ASB is a separate codebase (github.com/agiresearch/ASB, built on the AIOS agent OS), not a pip package.
We clone it into ``third_party/ASB`` and read its *data files* only:

    data/agent_task.jsonl         per-agent user tasks            (5 tasks / agent)
    data/all_normal_tools.jsonl   per-agent normal tools          (2 tools / agent)
    data/all_attack_tools.jsonl   per-agent attacker tools        (40 tools / agent)
    pyopenagi/agents/example/<agent>/config.json   agent description (system prompt)

See ``asb/env.py`` for how they are turned into an executable environment, and
``results/summary.md`` for the deviation (ASB's AIOS/torch/chroma agent runtime is not used).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import config

ASB_REPO_URL = "https://github.com/agiresearch/ASB.git"
ASB_DIR = config.ROOT / "third_party" / "ASB"

#: The ONE scenario used (spec Section 11): the finance scenario - closest in spirit to Banking.
#: Like every ASB agent it has just two normal tools, so tool-count is minimal.
SCENARIO = "financial_analyst_agent"
#: ASB's observation-prompt-injection variant used (ASB's ``config/OPI.yml`` default attack type).
ATTACK_TYPE = "context_ignoring"


def ensure_asb() -> Path:
    if not (ASB_DIR / "data" / "agent_task.jsonl").exists():
        ASB_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", ASB_REPO_URL, str(ASB_DIR)], check=True)
    return ASB_DIR


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Scenario:
    name: str
    description: str  # agent persona; used as the system prompt
    tasks: list[str]
    normal_tools: list[dict]  # {Tool Name, Description, Expected Achievements, ...}
    attack_tools: list[dict]  # {Attacker Tool, Attacker Instruction, Description, Attack goal, Aggressive, ...}


def load_scenario(name: str = SCENARIO) -> Scenario:
    root = ensure_asb()
    tasks = next(r for r in _jsonl(root / "data" / "agent_task.jsonl") if r["agent_name"] == name)["tasks"]
    normal = [r for r in _jsonl(root / "data" / "all_normal_tools.jsonl") if r["Corresponding Agent"] == name]
    attack = [r for r in _jsonl(root / "data" / "all_attack_tools.jsonl") if r["Corresponding Agent"] == name]
    cfg = json.loads((root / "pyopenagi" / "agents" / "example" / name / "config.json").read_text(encoding="utf-8"))
    return Scenario(name, "".join(cfg["description"]).strip(), tasks, normal, attack)
