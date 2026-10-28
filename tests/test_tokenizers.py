"""The optional tiktoken adapter: importable without tiktoken, exact when a backend is supplied."""

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget, TiktokenCounter, tiktoken_available
from harpe.protocols import TokenCounter


class FakeEncoder:
    """Stands in for tiktoken.Encoding so the adapter is tested without the extra installed."""

    def __init__(self):
        self.calls = []

    def encode(self, text, disallowed_special=()):
        self.calls.append((text, disallowed_special))
        return text.split()


def test_counter_delegates_to_the_encoder_and_counts_special_tokens_as_text():
    enc = FakeEncoder()
    counter = TiktokenCounter("o200k_base", encoder=enc)
    assert isinstance(counter, TokenCounter)
    assert counter.count_tokens("") == 0
    assert counter.count_tokens("one two <|endoftext|>") == 3
    assert enc.calls == [("one two <|endoftext|>", ())]


def test_engine_uses_the_adapter_for_uncounted_items():
    engine = ContextBudgetEngine(token_counter=TiktokenCounter(encoder=FakeEncoder()))
    report = engine.synthesize([ContextItem(id="a", content="five words in this item", priority=10)], TokenBudget(total_budget=50))
    assert report.consumed_tokens == 5


def test_missing_backend_raises_at_construction_only():
    if tiktoken_available():
        pytest.skip("tiktoken installed in this interpreter; the ImportError path is not reachable")
    with pytest.raises(ImportError, match="harpe\\[tiktoken\\]"):
        TiktokenCounter()


def test_real_backend_matches_the_reference_counts_in_the_shared_vectors():
    if not tiktoken_available():
        pytest.skip("tiktoken not installed")
    counter = TiktokenCounter("o200k_base")
    # The o200k_base counts recorded in tests/token_vectors.tsv descriptions when the vectors were added.
    assert counter.count_tokens("fc4e964c925fec6cef00d34b6670de63f6578557a48a77999e93fe73dc8f68bb") == 35
    assert counter.count_tokens("今天的天气很好，我们去公园散步吧。") == 12
    assert counter.count_tokens("https://github.com/Perseus-Computing-LLC/harpe/blob/main/README.md") == 22
