"""Z3-based narrowing-vs-expansion check for policy updates (task spec Section 6.3).

For a proposed update ``P -> P'`` and every tool ``t`` we build

    Phi_P(args)  : "the call is permitted by P"  (= no forbid rule matches AND some allow rule matches)
    Phi_P'(args) : same for P'

and ask Z3 whether ``exists args. Phi_P'(args) /\\ not Phi_P(args)``.  (This is the negation of
``forall args. Phi_P'(args) => Phi_P(args)``, but is phrased existentially so the solver never
needs quantifier elimination.)

* UNSAT for every tool  -> P' is a *narrowing* (or equal) of P  -> applied automatically.
* SAT for some tool     -> *expansion*; the model is a concrete witness call that P' permits and P
                           does not.  Per task scope the approver auto-approves expansions, but the
                           event is logged as "expansion (approved)".

Encoding (symbolic, exact):  string / integer / number / boolean arguments; ``enum``/``const``;
``pattern`` (Python regex subset translated to Z3 regular expressions, see ``_RegexTranslator``);
``minLength/maxLength``; numeric comparisons; ``anyOf/allOf/oneOf/not``; arrays via
(length, one witness element) with ``minItems/maxItems/items``.  Absent optional arguments are
modelled with a per-argument ``present`` flag.

NOTE: the encoding models an argument as present/absent ("key missing" is unconstrained).  The enforcer
additionally validates defaulted ``None`` values of omitted optional arguments (official behaviour); that
case is not representable here.  Since this check only labels updates in the log and never affects an
enforcement decision, the difference is cosmetic.

SIMPLIFICATION (explicitly allowed by the spec): if a tool's rules use something we cannot encode
(object ``properties`` conditions, regex features outside the supported subset, Z3 timeout, ...)
we fall back, *for that tool only*, to a **bounded check**: the old and new rules are evaluated
concretely on a finite set of representative argument values (every literal mentioned in either
policy, boundary values, the current call's arguments, and adversarial probes).  That fallback can
miss an expansion that needs a value outside the probe set.  ``ToolCheck.method`` records which path
was used ("z3" or "probe").
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import z3

from policy.enforcer import is_allowed
from policy.schema import METADATA_KEYWORDS, WILDCARD, Policy, Rule


class Unsupported(Exception):
    """The construct cannot be encoded symbolically; caller falls back to the bounded check."""


# ---------------------------------------------------------------------------
# regex -> Z3 regular expression
# ---------------------------------------------------------------------------
_STR = z3.StringSort()
_RE = z3.ReSort(_STR)


def _cat(items):
    """z3.Concat needs >= 2 arguments."""
    items = list(items)
    if not items:
        return z3.Re(z3.StringVal(""))
    return items[0] if len(items) == 1 else z3.Concat(*items)


def _all_char():
    return z3.AllChar(_RE)


class _RegexTranslator:
    """Translate a (practical) subset of Python regex to a Z3 ``ReRef`` with ``re.search`` semantics.

    Supported: literals, ``.``, character classes (ranges, negation, ``\\d \\w \\s``), groups
    (capturing / ``(?:...)``), alternation, ``* + ? {m} {m,} {m,n}``, and ``^`` / ``$`` at the very
    start / end of a top-level alternative.  Everything else (lookaround, backrefs, ``\\b``, inline
    flags, anchors elsewhere, lazy quantifiers are accepted but treated as greedy) -> ``Unsupported``.
    ``\\w`` / ``\\d`` / ``\\s`` are ASCII-only approximations of Python's Unicode classes.
    """

    def __init__(self, pattern: str):
        self.p = pattern
        self.i = 0

    # -- public ---------------------------------------------------------------
    def translate(self):
        branches = []
        while True:
            branches.append(self._top_branch())
            if self._peek() == "|":
                self.i += 1
                continue
            break
        if self.i != len(self.p):
            raise Unsupported(f"unparsed regex tail at {self.i}: {self.p!r}")
        return branches[0] if len(branches) == 1 else z3.Union(*branches)

    # -- helpers --------------------------------------------------------------
    def _peek(self) -> str | None:
        return self.p[self.i] if self.i < len(self.p) else None

    def _top_branch(self):
        start = False
        if self._peek() == "^":
            start = True
            self.i += 1
        items = self._seq(top=True)
        end = False
        if self._peek() == "$":
            end = True
            self.i += 1
            if self._peek() not in (None, "|"):
                raise Unsupported("'$' not at end of alternative")
        body = _cat(items)
        parts = []
        if not start:
            parts.append(z3.Star(_all_char()))
        parts.append(body)
        if not end:
            parts.append(z3.Star(_all_char()))
        return _cat(parts)

    def _seq(self, top: bool = False) -> list:
        items = []
        while True:
            c = self._peek()
            if c is None or c == "|" or c == ")":
                break
            if c == "$" and top:
                break
            if c in "^$":
                raise Unsupported("anchor in unsupported position")
            atom = self._atom()
            items.append(self._quant(atom))
        return items

    def _alt(self):
        branches = []
        while True:
            items = self._seq()
            branches.append(_cat(items))
            if self._peek() == "|":
                self.i += 1
                continue
            break
        return branches[0] if len(branches) == 1 else z3.Union(*branches)

    def _quant(self, atom):
        c = self._peek()
        if c == "*":
            self.i += 1
            r = z3.Star(atom)
        elif c == "+":
            self.i += 1
            r = z3.Plus(atom)
        elif c == "?":
            self.i += 1
            r = z3.Option(atom)
        elif c == "{":
            j = self.p.find("}", self.i)
            if j < 0:
                raise Unsupported("unterminated {m,n}")
            body = self.p[self.i + 1 : j]
            self.i = j + 1
            try:
                if "," in body:
                    lo_s, hi_s = body.split(",", 1)
                    lo = int(lo_s or 0)
                    r = _cat([z3.Loop(atom, lo, lo), z3.Star(atom)]) if hi_s == "" else z3.Loop(atom, lo, int(hi_s))
                else:
                    r = z3.Loop(atom, int(body), int(body))
            except ValueError as e:
                raise Unsupported(f"bad quantifier {{{body}}}") from e
        else:
            return atom
        if self._peek() == "?":  # lazy quantifier: same language for a pure membership test
            self.i += 1
        return r

    def _atom(self):
        c = self.p[self.i]
        if c == "(":
            self.i += 1
            if self.p.startswith("?:", self.i):
                self.i += 2
            elif self._peek() == "?":
                raise Unsupported("lookaround / inline flags / named groups")
            r = self._alt()
            if self._peek() != ")":
                raise Unsupported("unbalanced parenthesis")
            self.i += 1
            return r
        if c == "[":
            return self._char_class()
        if c == ".":
            self.i += 1
            return z3.Intersect(_all_char(), z3.Complement(z3.Re(z3.StringVal("\n"))))
        if c == "\\":
            return self._escape_atom()
        if c in "*+?{":
            raise Unsupported(f"dangling quantifier {c!r}")
        self.i += 1
        return z3.Re(z3.StringVal(c))

    _CLASS_ESC = {
        "d": [("0", "9")],
        "w": [("a", "z"), ("A", "Z"), ("0", "9"), ("_", "_")],
        "s": [(" ", " "), ("\t", "\t"), ("\n", "\n"), ("\r", "\r"), ("\f", "\f"), ("\v", "\v")],
    }
    _LIT_ESC = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}

    def _class_re(self, ranges):
        return z3.Union(*[z3.Range(a, b) if a != b else z3.Re(z3.StringVal(a)) for a, b in ranges]) if len(ranges) > 1 else (
            z3.Range(*ranges[0]) if ranges[0][0] != ranges[0][1] else z3.Re(z3.StringVal(ranges[0][0]))
        )

    def _escape_atom(self):
        self.i += 1
        if self.i >= len(self.p):
            raise Unsupported("trailing backslash")
        c = self.p[self.i]
        self.i += 1
        if c in self._CLASS_ESC:
            return self._class_re(self._CLASS_ESC[c])
        if c.lower() in self._CLASS_ESC and c.isupper():
            return z3.Intersect(_all_char(), z3.Complement(self._class_re(self._CLASS_ESC[c.lower()])))
        if c in self._LIT_ESC:
            return z3.Re(z3.StringVal(self._LIT_ESC[c]))
        if c.isalnum():
            raise Unsupported(f"escape \\{c}")
        return z3.Re(z3.StringVal(c))  # escaped punctuation

    def _char_class(self):
        self.i += 1  # '['
        negate = False
        if self._peek() == "^":
            negate = True
            self.i += 1
        ranges: list[tuple[str, str]] = []
        first = True
        while True:
            c = self._peek()
            if c is None:
                raise Unsupported("unterminated character class")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "\\":
                self.i += 1
                e = self.p[self.i]
                self.i += 1
                if e in self._CLASS_ESC:
                    ranges.extend(self._CLASS_ESC[e])
                    continue
                if e.isalnum() and e not in self._LIT_ESC:
                    raise Unsupported(f"escape \\{e} in class")
                lo = self._LIT_ESC.get(e, e)
            else:
                self.i += 1
                lo = c
            if self._peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("", "]"):
                self.i += 1
                hi = self.p[self.i]
                self.i += 1
                if hi == "\\":
                    hi = self.p[self.i]
                    self.i += 1
                if ord(hi) < ord(lo):
                    raise Unsupported("bad range")
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        if not ranges:
            raise Unsupported("empty class")
        cls = self._class_re(ranges)
        return z3.Intersect(_all_char(), z3.Complement(cls)) if negate else cls


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------
@dataclass
class ToolCheck:
    tool: str
    verdict: str  # "expansion" | "narrowing"  (narrowing includes "equal")
    method: str  # "z3" | "probe"
    witness: dict | None = None  # concrete call permitted by P' but not by P (expansion only)
    note: str = ""

    def to_json(self) -> dict:
        return {"tool": self.tool, "verdict": self.verdict, "method": self.method,
                "witness": self.witness, "note": self.note}


@dataclass
class PolicyDiff:
    checks: list[ToolCheck] = field(default_factory=list)

    @property
    def is_expansion(self) -> bool:
        return any(c.verdict == "expansion" for c in self.checks)

    @property
    def kind(self) -> str:
        return "expansion" if self.is_expansion else "narrowing"

    def to_json(self) -> dict:
        return {"kind": self.kind, "checks": [c.to_json() for c in self.checks]}


# ---------------------------------------------------------------------------
# symbolic encoder
# ---------------------------------------------------------------------------
def _and(xs):
    xs = list(xs)
    return z3.BoolVal(True) if not xs else (xs[0] if len(xs) == 1 else z3.And(*xs))


def _or(xs):
    xs = list(xs)
    return z3.BoolVal(False) if not xs else (xs[0] if len(xs) == 1 else z3.Or(*xs))


def _rv(x: float | int):
    return z3.RealVal(str(Fraction(str(x))))


def _types_of(schema: dict) -> set[str] | None:
    t = schema.get("type")
    if t is None:
        return None
    return {t} if isinstance(t, str) else set(t)


_KIND_BY_TYPE = {"string": "string", "integer": "int", "number": "real", "boolean": "bool", "array": "array"}


class _Arg:
    """Symbolic variables for one tool argument."""

    def __init__(self, tool: str, name: str, kind: str, item_kind: str | None):
        self.name, self.kind, self.item_kind = name, kind, item_kind
        self.present = z3.Bool(f"{tool}.{name}.present")
        if kind == "array":
            self.len = z3.Int(f"{tool}.{name}.len")
            self.elem = self._mk(f"{tool}.{name}.elem", item_kind)
            self.var = None
        else:
            self.var = self._mk(f"{tool}.{name}", kind)

    @staticmethod
    def _mk(name: str, kind: str | None):
        if kind == "string":
            return z3.String(name)
        if kind == "int":
            return z3.Int(name)
        if kind == "real":
            return z3.Real(name)
        if kind == "bool":
            return z3.Bool(name)
        raise Unsupported(f"unsupported argument kind {kind!r}")


class _Encoder:
    def __init__(self, tool: str, arg_types: dict[str, dict], rules: list[Rule]):
        self.tool = tool
        self.args: dict[str, _Arg] = {}
        self._re_cache: dict[str, Any] = {}
        names = {a for r in rules for a in r.conditions}
        for a in sorted(names):
            info = arg_types.get(a, {})
            kind = _KIND_BY_TYPE.get(info.get("type") or "")
            item_kind = _KIND_BY_TYPE.get(info.get("items_type") or "") if kind == "array" else None
            if kind is None:  # infer from the conditions
                kind = self._infer(a, rules)
            if kind == "array" and item_kind is None:
                item_kind = self._infer_item(a, rules)
            self.args[a] = _Arg(tool, a, kind, item_kind)

    # -- type inference when the tool schema does not say -------------------------
    @staticmethod
    def _infer(arg: str, rules: list[Rule]) -> str:
        for r in rules:
            c = r.conditions.get(arg)
            if not isinstance(c, dict):
                continue
            ts = _types_of(c) or set()
            for t, k in _KIND_BY_TYPE.items():
                if t in ts:
                    return k
            if "pattern" in c or "minLength" in c or "maxLength" in c:
                return "string"
            if any(k in c for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")):
                return "real"
            if "items" in c or "minItems" in c or "maxItems" in c:
                return "array"
            vals = c.get("enum") or ([c["const"]] if "const" in c else [])
            if vals:
                v = vals[0]
                return "bool" if isinstance(v, bool) else "string" if isinstance(v, str) else "real"
        raise Unsupported(f"cannot infer type of argument '{arg}'")

    @staticmethod
    def _infer_item(arg: str, rules: list[Rule]) -> str:
        for r in rules:
            c = r.conditions.get(arg)
            if isinstance(c, dict) and isinstance(c.get("items"), dict):
                ts = _types_of(c["items"]) or set()
                for t, k in _KIND_BY_TYPE.items():
                    if t in ts and t != "array":
                        return k
        return "string"

    # -- regex ------------------------------------------------------------------
    def _regex(self, pattern: str):
        if pattern not in self._re_cache:
            self._re_cache[pattern] = _RegexTranslator(pattern).translate()
        return self._re_cache[pattern]

    # -- scalar conditions --------------------------------------------------------
    def _scalar(self, schema: dict, term, kind: str):
        if kind not in ("string", "int", "real", "bool"):
            raise Unsupported(f"nested {kind} conditions")
        cs: list = []
        for kw in schema:
            if kw in METADATA_KEYWORDS:
                continue
            if kw == "type":
                ts = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
                ts = set(ts)
                if kind == "string":
                    cs.append(z3.BoolVal("string" in ts))
                elif kind == "bool":
                    cs.append(z3.BoolVal("boolean" in ts))
                elif kind == "int":
                    cs.append(z3.BoolVal(bool(ts & {"integer", "number"})))
                elif "number" in ts:  # real
                    cs.append(z3.BoolVal(True))
                else:
                    cs.append(z3.IsInt(term) if "integer" in ts else z3.BoolVal(False))
            elif kw in ("enum", "const"):
                vals = schema["enum"] if kw == "enum" else [schema["const"]]
                lits = []
                for v in vals:
                    if kind == "string" and isinstance(v, str):
                        lits.append(term == z3.StringVal(v))
                    elif kind == "bool" and isinstance(v, bool):
                        lits.append(term == z3.BoolVal(v))
                    elif kind in ("int", "real") and isinstance(v, (int, float)) and not isinstance(v, bool):
                        if kind == "int":
                            if float(v) == int(v):
                                lits.append(term == z3.IntVal(int(v)))
                        else:
                            lits.append(term == _rv(v))
                cs.append(_or(lits))
            elif kw == "pattern":
                if kind == "string":
                    cs.append(z3.InRe(term, self._regex(schema["pattern"])))
            elif kw in ("minLength", "maxLength"):
                if kind == "string":
                    cs.append(z3.Length(term) >= schema[kw] if kw == "minLength" else z3.Length(term) <= schema[kw])
            elif kw in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                if kind in ("int", "real"):
                    t = z3.ToReal(term) if kind == "int" else term
                    b = _rv(schema[kw])
                    cs.append({"minimum": t >= b, "maximum": t <= b, "exclusiveMinimum": t > b,
                               "exclusiveMaximum": t < b}[kw])
            elif kw in ("minItems", "maxItems", "items"):
                pass  # array-only keywords: vacuously true for scalars
            elif kw == "allOf":
                cs.append(_and(self._scalar(s, term, kind) for s in schema["allOf"]))
            elif kw == "anyOf":
                cs.append(_or(self._scalar(s, term, kind) for s in schema["anyOf"]))
            elif kw == "oneOf":
                subs = [self._scalar(s, term, kind) for s in schema["oneOf"]]
                cs.append(z3.PbEq([(s, 1) for s in subs], 1) if subs else z3.BoolVal(False))
            elif kw == "not":
                cs.append(z3.Not(self._scalar(schema["not"], term, kind)))
            else:  # properties / required / additionalProperties (object member access) ...
                raise Unsupported(f"keyword '{kw}' on scalar argument")
        return _and(cs)

    # -- array conditions -----------------------------------------------------------
    def _array(self, schema: dict, a: _Arg):
        cs: list = []
        for kw in schema:
            if kw in METADATA_KEYWORDS:
                continue
            if kw == "type":
                ts = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
                cs.append(z3.BoolVal("array" in ts))
            elif kw == "minItems":
                cs.append(a.len >= schema[kw])
            elif kw == "maxItems":
                cs.append(a.len <= schema[kw])
            elif kw == "items":
                if not isinstance(schema["items"], dict):
                    raise Unsupported("tuple-style items")
                cs.append(z3.Implies(a.len >= 1, self._scalar(schema["items"], a.elem, a.item_kind)))
            elif kw in ("pattern", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
                        "exclusiveMaximum"):
                pass
            elif kw == "allOf":
                cs.append(_and(self._array(s, a) for s in schema["allOf"]))
            elif kw == "anyOf":
                cs.append(_or(self._array(s, a) for s in schema["anyOf"]))
            elif kw == "not":
                cs.append(z3.Not(self._array(schema["not"], a)))
            else:
                raise Unsupported(f"keyword '{kw}' on array argument")
        return _and(cs)

    def _cond(self, a: _Arg, cond: Any):
        if cond == WILDCARD or cond == {}:
            return z3.BoolVal(True)
        body = self._array(cond, a) if a.kind == "array" else self._scalar(cond, a.var, a.kind)
        return z3.Or(z3.Not(a.present), body)  # absent argument => unconstrained

    def rule_formula(self, r: Rule):
        return _and(self._cond(self.args[a], c) for a, c in r.conditions.items())

    def allowed(self, rules: list[Rule]):
        forbids = [self.rule_formula(r) for r in rules if r.effect == "forbid"]
        allows = [self.rule_formula(r) for r in rules if r.effect == "allow"]
        return z3.And(z3.Not(_or(forbids)), _or(allows))

    def witness(self, model) -> dict:
        out: dict[str, Any] = {}
        for n, a in self.args.items():
            if not z3.is_true(model.eval(a.present, model_completion=True)):
                continue
            if a.kind == "array":
                ln = model.eval(a.len, model_completion=True).as_long()
                ev = self._py(model.eval(a.elem, model_completion=True), a.item_kind)
                out[n] = [ev] * max(0, min(ln, 3))
            else:
                out[n] = self._py(model.eval(a.var, model_completion=True), a.kind)
        return out

    @staticmethod
    def _py(v, kind):
        if kind == "string":
            return v.as_string()
        if kind == "int":
            return v.as_long()
        if kind == "bool":
            return z3.is_true(v)
        f = v.as_fraction()
        return float(f)


# ---------------------------------------------------------------------------
# bounded fallback
# ---------------------------------------------------------------------------
_MISSING = object()
ADVERSARIAL_PROBES: dict[str, list] = {
    "string": ["", "x", "attacker@evil.example", "US133000000121212121212", "GB00ATTACKER0000000000", "*",
               "new_password", "../../etc/passwd"],
    "int": [0, 1, -1, 5, 10_000, 1_000_000],
    "real": [0.0, 0.01, 1.0, 100.0, 10_000.0, 30_000.0, 1e9, -1.0],
    "bool": [True, False],
    "array": [[], ["x"], ["attacker@evil.example"], ["a", "b", "c", "d"]],
}


def _collect_literals(cond: Any, out: set) -> None:
    if not isinstance(cond, dict):
        return
    for k in ("enum", "const"):
        if k in cond:
            for v in (cond["enum"] if k == "enum" else [cond["const"]]):
                out.add(v if isinstance(v, (str, int, float, bool)) else repr(v))
    for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if k in cond and isinstance(cond[k], (int, float)):
            b = cond[k]
            out.update({b, b - 1, b + 1, b + 0.01, b - 0.01})
    for k in ("minLength", "maxLength"):
        if k in cond and isinstance(cond[k], int):
            out.update({"a" * cond[k], "a" * (cond[k] + 1), "a" * max(cond[k] - 1, 0)})
    for k in ("minItems", "maxItems"):
        if k in cond and isinstance(cond[k], int):
            out.update({tuple(["a"] * cond[k]), tuple(["a"] * (cond[k] + 1))})
    if isinstance(cond.get("items"), dict):
        inner: set = set()
        _collect_literals(cond["items"], inner)
        for v in inner:
            out.add((v,))
    for sub in ("anyOf", "allOf", "oneOf"):
        for s in cond.get(sub, []):
            _collect_literals(s, out)
    if isinstance(cond.get("not"), dict):
        _collect_literals(cond["not"], out)
    for s in (cond.get("properties") or {}).values():
        _collect_literals(s, out)


def _probe_check(tool: str, old: list[Rule], new: list[Rule], arg_types: dict[str, dict],
                 extra_args: dict | None, max_combos: int = 20000, seed: int = 0) -> ToolCheck:
    names = sorted({a for r in old + new for a in r.conditions})
    cands: dict[str, list] = {}
    for a in names:
        lits: set = set()
        for r in old + new:
            _collect_literals(r.conditions.get(a), lits)
        vals: list = [_MISSING]
        for v in lits:
            vals.append(list(v) if isinstance(v, tuple) else v)
        t = (arg_types.get(a) or {}).get("type") or "string"
        kind = _KIND_BY_TYPE.get(t, "string")
        vals.extend(ADVERSARIAL_PROBES[kind])
        if extra_args and a in extra_args:
            vals.append(extra_args[a])
        # de-duplicate (values may be unhashable lists)
        seen, uniq = [], []
        for v in vals:
            key = repr(v)
            if key not in seen:
                seen.append(key)
                uniq.append(v)
        cands[a] = uniq
    space = 1
    for a in names:
        space *= len(cands[a])
    rng = random.Random(seed)
    combos = (itertools.product(*[cands[a] for a in names]) if space <= max_combos else
              (tuple(rng.choice(cands[a]) for a in names) for _ in range(max_combos)))
    for combo in combos:
        args = {a: v for a, v in zip(names, combo) if v is not _MISSING}
        if is_allowed(new, args) and not is_allowed(old, args):
            return ToolCheck(tool, "expansion", "probe", args, "found by bounded probing")
    return ToolCheck(tool, "narrowing", "probe", None, f"no counterexample among {min(space, max_combos)} probes")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _check_tool_z3(tool: str, old: list[Rule], new: list[Rule], arg_types: dict[str, dict],
                   timeout_ms: int) -> ToolCheck:
    enc = _Encoder(tool, arg_types, old + new)
    phi_old, phi_new = enc.allowed(old), enc.allowed(new)
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(phi_new, z3.Not(phi_old))
    r = s.check()
    if r == z3.sat:
        return ToolCheck(tool, "expansion", "z3", enc.witness(s.model()), "Z3 counterexample")
    if r == z3.unsat:
        return ToolCheck(tool, "narrowing", "z3", None, "Z3: forall args. Phi_P' => Phi_P")
    raise Unsupported(f"z3 returned {r} (reason: {s.reason_unknown()})")


def compare_policies(
    old: Policy,
    new: Policy,
    arg_types: dict[str, dict[str, dict]],
    current_call: tuple[str, dict] | None = None,
    timeout_ms: int = 8000,
) -> PolicyDiff:
    """Classify ``old -> new`` per tool as narrowing or expansion.

    ``arg_types`` comes from :func:`policy.schema.tool_arg_types`.  ``current_call`` (tool, args) is
    only used to enrich the probe set of the bounded fallback.
    """
    diff = PolicyDiff()
    for tool in sorted(set(old.rules) | set(new.rules)):
        o, n = old.rules_for(tool), new.rules_for(tool)
        if [r.to_json() for r in o] == [r.to_json() for r in n]:
            continue  # unchanged tool
        types = arg_types.get(tool, {})
        try:
            chk = _check_tool_z3(tool, o, n, types, timeout_ms)
        except Unsupported as e:
            extra = current_call[1] if current_call and current_call[0] == tool else None
            chk = _probe_check(tool, o, n, types, extra)
            chk.note += f" (z3 fallback: {e})"
        except z3.Z3Exception as e:  # pragma: no cover - defensive
            chk = _probe_check(tool, o, n, types, None)
            chk.note += f" (z3 error: {e})"
        diff.checks.append(chk)
    return diff
