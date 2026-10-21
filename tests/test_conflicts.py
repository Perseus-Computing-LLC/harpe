"""Conflict surfacing: a statement and its negation, both in the selected context, are flagged for the reader."""

from harpe import Conflict, annotate_conflicts, find_conflicts

NTP_SAID = "user: I'm trying to synchronize the server time with the NTP service to fix token validation errors."
NTP_DENIED = "user: I've never synchronized my server time with any NTP service, which might be causing clock skew."
TESTS_SAID = "user: I've written integration tests for socket events using Jest and socket.io-client, at 90% coverage."
TESTS_DENIED = "user: I've never written integration tests for any socket events, so I'm not sure where to start."


def test_a_statement_and_its_denial_conflict():
    texts = {"a": TESTS_SAID, "b": "assistant: Here is a Jest example.", "c": TESTS_DENIED}
    conflicts = find_conflicts(texts, query="Have I ever written integration tests for socket events using Jest?")
    assert len(conflicts) == 1
    c = conflicts[0]
    assert isinstance(c, Conflict) and (c.denied_id, c.stated_id) == ("c", "a")
    assert c.denied.startswith("I've never written integration tests") and c.stated.startswith("I've written")
    assert {"integration", "socket", "tests"} <= set(c.shared_terms)


def test_word_forms_match():
    conflicts = find_conflicts({"x": NTP_SAID, "y": NTP_DENIED}, query="Have I synchronized my server time with NTP?")
    assert [(c.denied_id, c.stated_id) for c in conflicts] == [("y", "x")]


def test_unrelated_negations_and_assistant_text_do_not_conflict():
    texts = {
        "a": "user: I've never been to Paris, but I loved the food in Rome.",
        "b": "user: I've written integration tests for socket events using Jest.",
        "c": "assistant: You haven't written integration tests for socket events yet, so start with Jest.",
        "d": "user: I don't like tests that are slow.",
    }
    assert find_conflicts(texts, query="Have I written integration tests for socket events?") == []


def test_the_question_must_touch_both_statements():
    texts = {"a": TESTS_SAID, "c": TESTS_DENIED}
    assert find_conflicts(texts, query="What did I cook for dinner on Friday?") == []
    assert len(find_conflicts(texts)) == 1  # no question: any conflict counts


def test_annotations_quote_the_other_statement_after_the_denial():
    texts = {"a": TESTS_SAID, "c": TESTS_DENIED}
    conflicts = find_conflicts(texts)
    notes = annotate_conflicts(texts, conflicts, dates={"a": "March-15-2024", "c": "April-05-2024"})
    assert notes["a"] == TESTS_SAID
    assert notes["c"].startswith("user: I've never written integration tests for any socket events, so I'm not sure where to start. [")
    assert "possible conflict" in notes["c"] and "March-15-2024" in notes["c"] and "I've written integration tests" in notes["c"]
    assert annotate_conflicts(texts, conflicts)["c"].count("[possible conflict: user said") == 1  # without dates
