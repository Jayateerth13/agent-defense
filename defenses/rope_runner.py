"""ROPE (Ma et al., arXiv 2608.27496) on our AgentDojo pipeline, plus the record-laundering defenses.

The defense core (origin tracker, policy compiler, guard) is vendored unchanged in ``defenses/rope``
(MIT, upstream commit in ``defenses/rope/UPSTREAM_COMMIT``).  This module only re-assembles ROPE's
pipeline (``rope/pipeline.py``: bootstrap -> LLM -> loop(ToolsExecutor, OriginTracker, LLM)) around
our agent LLM, with the paper's default configuration: cached Opus router scopes, no clamp, ``routed``
policy mode.

Three modes:

* ``rope``         ROPE as published.  One deliberate, documented change shared by all modes:
                   ``SENDER_ALIASES`` maps the user's own IBAN to ``"me"``.  AgentDojo's seed data labels
                   the user's past payments ``sender: me`` but payments the agent makes get
                   ``sender: <user IBAN>``; both are the user's own records, and ROPE's routed scopes
                   name ``"me"``.  Without the alias ROPE would not recognise payments the user's
                   agent made as the user's own (a utility bug in multi-session use).
* ``rope_nofree``  quick patch: identifiers inside free-text record fields (``subject``) never license
                   a value; only structured fields (sender, recipient, ids, ...) do.
* ``rope_prov``    persistent provenance (our defense): when the agent writes a record, the identifiers
                   in its free-text fields are labelled with whether they were trusted at write time
                   (admitted by the session's own origin rule).  The labels are kept in a
                   :class:`ProvenanceStore` that outlives the session; when a later session reads the
                   record, only identifiers that were trusted at write time are licensed.  Records with
                   no label (seed data) behave exactly as in ``rope``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import MethodType

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str

from agent.common import base_elements
from agent.llm import AgentLLM
from defenses.rope import markers as _m
from defenses.rope import origin as _origin
from defenses.rope import policy_synthesis as _ps
from defenses.rope.enforce import Enforcer

SCOPES_DIR = Path(__file__).resolve().parent / "rope" / "scopes"
MODES = ("rope", "rope_nofree", "rope_prov")

USER_IBAN = "DE89370400440532013000"  # AgentDojo banking account of the user
SENDER_ALIASES = {USER_IBAN: "me"}
FREE_TEXT_FIELDS = ("subject",)  # record fields whose content the writer chooses freely (unguarded in the floor)
WRITE_TOOLS = ("send_money", "schedule_transaction", "update_scheduled_transaction")


def load_banking_scopes() -> tuple[dict, dict]:
    """(floor, {prompt: TaskScope}) exactly as rope.scopes_io.load_router_scopes('opus', 'banking')."""
    floor_raw = json.loads((SCOPES_DIR / "floor_banking.json").read_text())
    floor = {t: {a: _m.str_to_rule(s) for a, s in am.items()} for t, am in floor_raw.items()}
    raw = json.loads((SCOPES_DIR / "opus_banking.json").read_text())
    return floor, {prompt: _m.scope_from_dict(d) for prompt, d in raw.items()}


# ── persistent provenance (rope_prov) ────────────────────────────────────────────────
class ProvenanceStore:
    """Write-time trust labels for free-text record fields, keyed by record id.  Lives across sessions."""

    def __init__(self) -> None:
        self.labels: dict[int, dict[str, set[str]]] = {}  # id -> {field: identifiers trusted at write time}
        self.log: list[dict] = []

    def licensed(self, rec_id, field: str, ids: set[str]) -> set[str]:
        lab = self.labels.get(rec_id)
        if lab is None or field not in lab:
            return ids  # unlabelled record (seed data): unchanged ROPE behaviour
        return ids & lab[field]


class ProvenanceRecorder(BasePipelineElement):
    """After each executor step: for every successful write, label the free-text fields it wrote."""

    def __init__(self, store: ProvenanceStore, bootstrap: "RopeBootstrap") -> None:
        self.store = store
        self.bootstrap = bootstrap
        self._seen = 0

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        new = list(messages)[self._seen:]
        self._seen = len(messages)
        for msg in new:
            if msg.get("role") != "tool" or msg.get("error"):
                continue
            tc = msg.get("tool_call")
            fn = getattr(tc, "function", None)
            if fn not in WRITE_TOOLS:
                continue
            args = dict(getattr(tc, "args", {}) or {})
            if fn == "update_scheduled_transaction":
                rec_id = args.get("id")
            else:
                lst = env.bank_account.transactions if fn == "send_money" else env.bank_account.scheduled_transactions
                rec_id = lst[-1].id if lst else None
            if rec_id is None:
                continue
            pred = self.bootstrap.trust_predicate()
            lab = self.store.labels.setdefault(rec_id, {})
            for field in FREE_TEXT_FIELDS:
                if not args.get(field):
                    continue
                ids = _origin.tokenize_identifiers(str(args[field]))
                trusted = {t for t in ids if pred(t)}
                lab[field] = trusted
                self.store.log.append({"id": rec_id, "tool": fn, "field": field,
                                       "trusted": sorted(trusted), "untrusted": sorted(ids - trusted)})
        return query, runtime, env, messages, extra_args


# ── origin tracker with the banking-specific sender alias and the two patches ─────────────
class BankingOriginTracker(_origin.OriginTracker):
    def __init__(self, origin_map: dict, mode: str, store: ProvenanceStore | None = None) -> None:
        super().__init__(origin_map)
        self.mode = mode
        self.store = store

    def _item_ids(self, item: dict) -> set[str]:
        if self.mode == "rope":
            return _origin._item_identifiers(item)  # upstream behaviour: every string field
        out: set[str] = set()
        for k, v in item.items():
            if not isinstance(v, str):
                continue
            ids = _origin.tokenize_identifiers(v)
            if k in FREE_TEXT_FIELDS:
                if self.mode == "rope_nofree":
                    continue
                ids = self.store.licensed(item.get("id"), k, ids) if self.store else ids
            out |= ids
        return out

    def _absorb(self, tool_msg) -> None:
        text = _origin._message_text(tool_msg)
        items = _origin._structured_items(text)
        fn = _origin._tool_function(tool_msg)
        if items is None or fn in _origin.AUTHORITATIVE_RECORD_TOOLS or fn in _origin.VERIFICATION_EXECUTOR_TOOLS:
            return super()._absorb(tool_msg)
        # same structured-item branch as upstream OriginTracker._absorb, with the alias + per-mode ids
        for item in items:
            origin = _origin._item_origin(item)
            if origin is None:
                self._add(_origin._source_descriptor(tool_msg), _origin._item_identifiers(item), injectable=True)
            else:
                self._add(SENDER_ALIASES.get(origin, origin), self._item_ids(item), injectable=False)


class TrackerElement(BasePipelineElement):
    def __init__(self, tracker: BankingOriginTracker) -> None:
        self._tracker = tracker

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        return self._tracker.query(query, runtime, env, messages, extra_args or {})


class RopeBootstrap(BasePipelineElement):
    """rope.pipeline.RoutedPolicyBootstrap, cached-router path, clamp off, policy_mode 'routed'."""

    def __init__(self, floor: dict, scopes: dict, origin_map: dict) -> None:
        self.floor = floor
        self.scopes = scopes
        self.origin_map = origin_map
        self.enforcer = Enforcer()
        self.scope = None
        self.prompt_ids: set[str] = set()
        self.blocked: list[dict] = []

    def trust_predicate(self) -> _origin.OriginPredicate:
        """The task's SOURCED rule: request ids, named non-injectable sources (used for write-time labels)."""
        named = set(self.scope.named_source) if self.scope and self.scope.named_source else None
        return _origin.OriginPredicate("subject", named, self.origin_map, self.prompt_ids)

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        self.enforcer.update_available_tools([
            {"name": t.name, "description": t.description,
             "args": t.parameters.model_json_schema().get("properties", {})}
            for t in runtime.functions.values()])
        self.enforcer.set_user_query(query)
        self.origin_map.clear()
        self.prompt_ids = _origin.tokenize_identifiers(query)
        self.scope = self.scopes.get(query)
        if self.scope is None:
            warnings.warn(f"no cached ROPE scope for prompt {query[:60]!r}; state-changing tools blocked")
            policy = {}
        else:
            policy = _ps.compile_policy(self.floor, self.scope, origin_map=self.origin_map,
                                        prompt_ids=self.prompt_ids, policy_mode="routed")
        self.enforcer.update_security_policy(policy)
        self.enforcer.update_always_allowed_tools([t.name for t in runtime.functions.values() if t.name not in self.floor])

        if getattr(runtime.run_function, "__name__", "") != "guarded_run_function":
            original = runtime.run_function
            enforcer, blocked = self.enforcer, self.blocked

            def guarded_run_function(self_runtime, call_env, function_name, args):
                try:
                    enforcer.check_tool_call(function_name, args)
                except Exception as e:  # noqa: BLE001 - same as upstream: the block surfaces as a tool error
                    blocked.append({"tool": function_name, "args": dict(args), "reason": str(e)[:300]})
                    return "", str(e)
                return original(call_env, function_name, args)

            runtime.run_function = MethodType(guarded_run_function, runtime)
        return query, runtime, env, messages, extra_args


def build_pipeline(mode: str, llm: BasePipelineElement | None = None,
                   store: ProvenanceStore | None = None) -> AgentPipeline:
    """A fresh ROPE pipeline for one session.  ``store`` (rope_prov) must be shared across sessions."""
    if mode not in MODES:
        raise ValueError(f"unknown ROPE mode {mode!r}")
    if mode == "rope_prov" and store is None:
        raise ValueError("rope_prov needs a ProvenanceStore shared across sessions")
    llm = llm or AgentLLM()
    floor, scopes = load_banking_scopes()
    origin_map: dict = {}
    boot = RopeBootstrap(floor, scopes, origin_map)
    tracker = TrackerElement(BankingOriginTracker(origin_map, mode, store if mode == "rope_prov" else None))
    loop_elems: list[BasePipelineElement] = [ToolsExecutor(tool_result_to_str), tracker]
    if mode == "rope_prov":
        loop_elems.append(ProvenanceRecorder(store, boot))
    loop = ToolsExecutionLoop([*loop_elems, llm])
    elements = base_elements(llm)
    pipe = AgentPipeline([*elements[:2], boot, elements[2], loop])
    pipe.name = getattr(llm, "name", None) or mode
    pipe.rope_bootstrap = boot  # type: ignore[attr-defined]
    return pipe
