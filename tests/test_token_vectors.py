"""Token counting: the vectors shared with the Rust tests in src/lib.rs, which both languages must count the same way,
and which items synthesize() counts and copies."""

from pathlib import Path

import pytest

from harpe import ContextBudgetEngine, ContextItem, HeuristicTokenCounter, TokenBudget

VECTORS = [
    line.split("\t")
    for line in Path(__file__).with_name("token_vectors.tsv").read_text(encoding="utf-8").splitlines()
    if not line.startswith("#")
]


def test_the_file_holds_every_vector():
    assert len(VECTORS) == 199
    assert all(len(fields) == 3 for fields in VECTORS)


@pytest.mark.parametrize("expected, hex_text, description", VECTORS, ids=[fields[-1] for fields in VECTORS])
def test_the_counter_matches_the_shared_vector(expected, hex_text, description):
    assert HeuristicTokenCounter().count_tokens(bytes.fromhex(hex_text).decode("utf-8")) == int(expected)


def test_synthesize_copies_only_the_items_it_counts():
    counted = ContextItem(id="counted", content="already counted", token_count=5, priority=10)
    uncounted = ContextItem(id="uncounted", content="needs a count", priority=20)
    report = ContextBudgetEngine().synthesize([counted, uncounted], TokenBudget(total_budget=100))
    included = {it.id: it for it in report.included_items}
    assert included["counted"] is counted
    assert included["uncounted"] is not uncounted
    assert included["uncounted"].token_count == HeuristicTokenCounter().count_tokens("needs a count")
    assert uncounted.token_count == 0


def test_synthesize_copies_every_item_for_hooks():
    class RaisePriority:
        def pre_prune(self, items):
            for it in items:
                it.priority = 90

        def post_prune(self, report):
            pass

    counted = ContextItem(id="counted", content="already counted", token_count=5, priority=10)
    report = ContextBudgetEngine(hooks=[RaisePriority()]).synthesize([counted], TokenBudget(total_budget=100))
    assert report.included_items[0] is not counted
    assert report.included_items[0].priority == 90
    assert counted.priority == 10
