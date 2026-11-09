"""Offline tests for the choice consistency check (policy/choice_check.py): no API calls."""

import json

from agentdojo.task_suite.load_suites import get_suite

import config
from agent import progent_runner
from agent.common import run_one
from agent.mock_llm import ScriptedLLM
from eval.run_eval import make_attack
from policy import choice_check as cc

SUITE = get_suite(config.BENCHMARK_VERSION, "banking")
ATTACK = make_attack(SUITE)


# ── sanitiser ───────────────────────────────────────────────────────────────────────────────────
def test_sanitize_travel_reviews_keeps_rating():
    raw = "{'Le Marais Boutique': 'Rating: 4.2\\nReviews: Charming hotel\\nFriendly staff'}"
    out = cc.sanitize("get_rating_reviews_for_hotels", raw)
    assert "Rating: 4.2" in out and "Charming" not in out and cc.REMOVED in out


def test_sanitize_keeps_prices_and_names():
    raw = "{'Le Marais Boutique': 'Price range: 120.0 - 180.0'}"
    assert cc.sanitize("get_hotels_prices", raw) == raw


def test_sanitize_yaml_records_drops_free_text_keeps_structure():
    raw = "- body: some third-party text\n  recipient: general\n  sender: Bob"
    out = cc.sanitize("read_channel_messages", raw)
    assert "third-party" not in out and "sender: Bob" in out and "recipient: general" in out


def test_sanitize_drops_whole_pages_and_files():
    assert cc.sanitize("get_webpage", "anything") == cc.REMOVED
    assert cc.sanitize("read_file", "anything") == cc.REMOVED


def test_sanitize_banking_transactions_drops_subject():
    raw = "- amount: 10.0\n  recipient: me\n  sender: GB29NWBK60161331926819\n  subject: free text here"
    out = cc.sanitize("get_most_recent_transactions", raw)
    assert "free text here" not in out and "GB29NWBK60161331926819" in out


# ── the check, with a scripted re-derivation ───────────────────────────────────────────────────────
def fake(value=None, differs=False, answer="redone"):
    calls = []

    def complete(system, user, temperature=0.0):
        calls.append(system[:20])
        if system.startswith(cc.MARKERS["judge"]):
            return json.dumps({"differs": differs, "reason": "test"})
        if system.startswith(cc.MARKERS["answer"]):
            return answer
        return json.dumps({"value": value})

    complete.calls = calls
    return complete


def test_same_choice_is_allowed():
    c = cc.ChoiceChecker("Book the best hotel in Paris", fake("Luxury Palace"))
    ok, msg = c.check("reserve_hotel", {"hotel": "Luxury Palace", "start_day": "x", "end_day": "y"})
    assert ok and msg is None and c.n_blocked == 0


def test_different_choice_is_blocked_with_message():
    c = cc.ChoiceChecker("Book the best hotel in Paris", fake("Luxury Palace"))
    ok, msg = c.check("reserve_hotel", {"hotel": "Riverside View Hotel"})
    assert not ok and "choice check" in msg and "Luxury Palace" in msg and c.n_blocked == 1


def test_undetermined_choice_is_allowed_and_counted():
    c = cc.ChoiceChecker("Book a hotel", fake("unknown"))
    ok, _ = c.check("reserve_hotel", {"hotel": "Anything"})
    assert ok and c.n_unknown == 1 and c.n_blocked == 0


def test_value_named_by_the_user_needs_no_llm_call():
    f = fake("something else")
    c = cc.ChoiceChecker("Pay GB29NWBK60161331926819 back", f)
    ok, _ = c.check("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 1})
    assert ok and f.calls == []


def test_list_arguments_and_non_choice_tools():
    c = cc.ChoiceChecker("Email the team", fake(["a@x.com", "b@x.com"]))
    assert c.check("send_email", {"recipients": ["a@x.com"]})[0]
    assert not c.check("send_email", {"recipients": ["evil@x.com"]})[0]
    assert c.check("get_balance", {})[0]  # not a state-changing choice tool


def test_unparsable_rederivation_never_blocks():
    c = cc.ChoiceChecker("Book a hotel", lambda s, u, t=0.0: "not json at all")
    assert c.check("reserve_hotel", {"hotel": "X"})[0]


def test_answer_replaced_only_when_judge_says_differs():
    c = cc.ChoiceChecker("Recommend a hotel", fake(differs=True, answer="Go to Luxury Palace"))
    assert c.check_answer("Visit Riverside View Hotel!") == "Go to Luxury Palace"
    c2 = cc.ChoiceChecker("Recommend a hotel", fake(differs=False))
    assert c2.check_answer("Visit Luxury Palace") is None


def test_record_stores_sanitised_facts_only():
    c = cc.ChoiceChecker("x", fake())
    c.record("read_file", {"file_path": "f.txt"}, "secret third-party text")
    c.record("get_hotels_prices", {"hotel_names": ["A"]}, "{'A': 'Price range: 1 - 2'}")
    joined = "\n".join(c.facts)
    assert "third-party" not in joined and "Price range: 1 - 2" in joined


# ── in the real pipeline (scripted agent, permissive policy so only the choice check can act) ─────────
def _run(choice_complete, user="user_task_5", inj="injection_task_0"):
    ut, it = SUITE.get_user_task_by_id(user), SUITE.get_injection_task_by_id(inj)

    def permissive(system, content, temperature=0.0):
        if system.startswith("Your task is to set permissions"):
            return json.dumps([{"name": "send_money", "args": {}}, {"name": "get_most_recent_transactions", "args": {}}])
        return "No"

    pipe = progent_runner.build_pipeline(ScriptedLLM(ut, it, fooled=True), permissive,
                                         always_allowed=config.ALWAYS_ALLOWED["banking"],
                                         choice_check=True, choice_complete=choice_complete)
    out = run_one(SUITE, pipe, ut, it, ATTACK.attack(ut, it))
    return out, pipe.progent_init.session


def test_pipeline_choice_check_blocks_steered_recipient():
    spotify = "SE3550000000054910000003"
    out, sess = _run(fake(spotify))
    assert out.error is None
    assert out.security is False  # nothing was sent to the attacker
    assert sess.choice.n_blocked >= 1
    assert any(e["result"] == "blocked" for e in sess.summary()["choice"]["events"])


def test_pipeline_without_the_check_lets_the_same_run_through():
    ut, it = SUITE.get_user_task_by_id("user_task_5"), SUITE.get_injection_task_by_id("injection_task_0")

    def permissive(system, content, temperature=0.0):
        if system.startswith("Your task is to set permissions"):
            return json.dumps([{"name": "send_money", "args": {}}, {"name": "get_most_recent_transactions", "args": {}}])
        return "No"

    pipe = progent_runner.build_pipeline(ScriptedLLM(ut, it, fooled=True), permissive,
                                         always_allowed=config.ALWAYS_ALLOWED["banking"])
    out = run_one(SUITE, pipe, ut, it, ATTACK.attack(ut, it))
    assert out.security is True


def test_visiting_a_page_is_a_checked_choice_but_a_named_one_is_not():
    c = cc.ChoiceChecker("Read the content on webpage www.informations.com", fake("www.informations.com"))
    assert c.check("get_webpage", {"url": "www.informations.com"})[0]  # named by the user
    c2 = cc.ChoiceChecker("Summarise Bob's article", fake("www.informations.com"))
    assert not c2.check("get_webpage", {"url": "www.other.com"})[0]


def test_judge_prompt_ignores_missing_information():
    assert "missing, unavailable" in cc.JUDGE_SYSTEM


def test_answer_check_skipped_when_page_or_bodies_were_removed():
    c = cc.ChoiceChecker("Summarise the page", fake(differs=True))
    c.record("get_webpage", {"url": "u"}, "page text")
    assert c.check_answer("A summary") is None and c.events[-1]["result"].startswith("skipped")
    c2 = cc.ChoiceChecker("Summarise Bob's post", fake(differs=True))
    msgs = ["- body: text", "  recipient: g", "  sender: Bob"]
    c2.record("read_channel_messages", {"channel": "g"}, chr(10).join(msgs))
    assert c2.check_answer("A summary") is None


def test_answer_check_still_runs_when_only_reviews_were_removed():
    c = cc.ChoiceChecker("Recommend a hotel", fake(differs=True, answer="Luxury Palace"))
    c.record("get_rating_reviews_for_hotels", {"hotel_names": ["A"]}, "{'A': 'Rating: 4.2\nReviews: nice'}")
    assert c.check_answer("Riverside") == "Luxury Palace"
