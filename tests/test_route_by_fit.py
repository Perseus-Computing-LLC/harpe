"""Fit-aware routing: hand a whole history over when it fits under a token threshold, else retrieve."""

import pytest
from harpe import (
    ContextBudgetEngine,
    ContextItem,
    ContextRoute,
    HeuristicTokenCounter,
    RouteDecision,
    route_by_fit,
)


class Words:
    """Counts words, and records what it counted."""

    def __init__(self):
        self.seen = []

    def count_tokens(self, text):
        self.seen.append(text)
        return len(text.split())


def test_a_history_that_fits_goes_whole():
    decision = route_by_fit(["a b c", "d e", "f"], threshold_tokens=6, token_counter=Words())
    assert decision == RouteDecision(route=ContextRoute.WHOLE, threshold_tokens=6, whole_tokens=6)


def test_one_token_over_retrieves_and_stops_counting():
    counter = Words()
    decision = route_by_fit(["a b c", "d e f", "g h", "never counted"], threshold_tokens=5, token_counter=counter)
    assert decision.route is ContextRoute.RETRIEVE
    assert decision.whole_tokens is None
    assert counter.seen == ["a b c", "d e f"]  # stops at the first item that goes over


@pytest.mark.parametrize("threshold", [0, None, -5])
def test_no_threshold_means_routing_is_off(threshold):
    counter = Words()
    decision = route_by_fit(["a"], threshold_tokens=threshold, token_counter=counter)
    assert decision.route is ContextRoute.RETRIEVE
    assert decision.threshold_tokens == 0 and decision.whole_tokens is None
    assert counter.seen == []


def test_an_empty_history_goes_whole():
    assert route_by_fit([], threshold_tokens=10, token_counter=Words()).whole_tokens == 0


def test_context_items_count_with_their_token_count_or_the_counter():
    items = [ContextItem(id="x", content="x y z", token_count=10), ContextItem(id="a", content="a b")]
    assert route_by_fit(items, threshold_tokens=12, token_counter=Words()).whole_tokens == 12
    assert route_by_fit(items, threshold_tokens=11, token_counter=Words()).route is ContextRoute.RETRIEVE


def test_the_default_counter_is_harpes_heuristic():
    text = "Alice moved to Paris in May 2023 after the conference."
    decision = route_by_fit([text], threshold_tokens=1000)
    assert decision.whole_tokens == HeuristicTokenCounter().count_tokens(text)


def test_the_engine_routes_with_its_own_counter():
    engine = ContextBudgetEngine(token_counter=Words())
    assert engine.route(["a b", "c"], threshold_tokens=3).route is ContextRoute.WHOLE
    assert engine.route(["a b", "c"], threshold_tokens=2).route is ContextRoute.RETRIEVE
