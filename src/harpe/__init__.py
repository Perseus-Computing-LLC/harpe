"""harpe: Deterministic context synthesis and token budgeting for cognitive AI agents."""

from .models import ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from .protocols import ContextSynthesizer, PruningHook, TokenCounter
from .engine import ContextBudgetEngine, HeuristicTokenCounter
from .dtoc import CompressedToolOutput, DynamicToolCompressor

__version__ = "0.1.0a3"

__all__ = [
    "ContextItem",
    "PrunePolicy",
    "SynthesisReport",
    "TokenBudget",
    "ContextSynthesizer",
    "PruningHook",
    "TokenCounter",
    "ContextBudgetEngine",
    "HeuristicTokenCounter",
    "CompressedToolOutput",
    "DynamicToolCompressor",
]
