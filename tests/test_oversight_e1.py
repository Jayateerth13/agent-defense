"""Offline tests for E1: the corpus is well formed and Progent's enforcer decides bash commands as expected."""

from oversight import progent_shell as ps
from oversight.e1_corpus import CORPUS, LEGIT, TAMPERING

BY_ID = {c.id: c for c in CORPUS}


def test_corpus_ids_are_unique_and_tampering_has_targets():
    assert len(BY_ID) == len(CORPUS)
    assert all(c.targets and not c.legit for c in TAMPERING)
    assert all(c.legit and not c.targets for c in LEGIT)


def test_careful_policy_blocks_literal_paths_but_not_rewordings():
    pol = ps.careful_policy()
    assert not ps.allowed(pol, BY_ID["rm"])
    assert not ps.allowed(pol, BY_ID["make_target_file"])
    assert ps.allowed(pol, BY_ID["base64_sh"])  # the path never appears in the text
    assert ps.allowed(pol, BY_ID["fix_with_sed"])


def test_careful_policy_also_blocks_some_honest_reads():
    assert not ps.allowed(ps.careful_policy(), BY_ID["read_test"])


def test_generated_policy_goes_through_progent_unchanged():
    pol = ps.llm_policy(ps.TASKS["fix_bug"], ps.mock_complete)
    assert ps.allowed(pol, BY_ID["fix_with_sed"])
    assert not ps.allowed(pol, BY_ID["list_files"])  # default block: no rule matches


def test_policy_round_trips_through_the_cache_record():
    pol = ps.careful_policy()
    assert ps.policy_from_record(ps.policy_record("t", "m", pol)) == pol
