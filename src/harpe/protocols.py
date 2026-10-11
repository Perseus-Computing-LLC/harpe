"""Interface protocols for Harpe components."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from .models import ContextItem, PrunePolicy, SynthesisReport, TokenBudget


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in raw strings."""

    def count_tokens(self, text: str) -> int:
        """Return estimated or exact token count for the given text."""
        ...


@runtime_checkable
class PruningHook(Protocol):
    """Lifecycle hook invoked during context synthesis passes."""

    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        """Invoked prior to any item filtering or pruning."""
        ...

    def post_prune(self, report: SynthesisReport) -> None:
        """Invoked after synthesis completes with the finalized report."""
        ...


@runtime_checkable
class ContextSynthesizer(Protocol):
    """Protocol for deterministic context synthesis against a token budget."""

    def synthesize(
        self,
        items: Sequence[ContextItem],
        budget: TokenBudget,
        policy: PrunePolicy = PrunePolicy.LOWEST_PRIORITY,
        destination: str = "remote",
    ) -> SynthesisReport:
        """Synthesize and prune context items to fit within the designated budget."""
        ...
