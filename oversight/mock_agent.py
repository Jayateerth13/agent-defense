"""A scripted stand-in for the agent model: replays fixed bash commands, then finishes.

It returns objects shaped like the OpenAI chat completion response, so ``agent_loop`` runs unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


def _response(content: str | None, commands: list[str], step: int):
    calls = [SimpleNamespace(id=f"call_{step}_{i}", type="function",
                             function=SimpleNamespace(name="bash", arguments=json.dumps({"command": c})))
             for i, c in enumerate(commands)]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None))])


class ScriptedAgent:
    """``turns`` is a list of command lists, one per model turn; after the last turn it says it is done.
    With several rounds, the script restarts each round (after the feedback message)."""

    def __init__(self, turns: list[list[str]], final: str = "Done."):
        self.turns, self.final = turns, final
        self.calls = 0
        self._i = 0

    def __call__(self, messages: list[dict], tools: list[dict]):
        self.calls += 1
        if messages[-1]["role"] == "user" and len(messages) > 2:  # feedback arrived: start the script again
            self._i = 0
        if self._i < len(self.turns):
            self._i += 1
            return _response(None, self.turns[self._i - 1], self.calls)
        return _response(self.final, [], self.calls)
