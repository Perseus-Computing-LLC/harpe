"""harpe: Deterministic context synthesis and token budgeting for cognitive AI agents."""

from .models import CognitiveDimension, ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from .protocols import ContextSynthesizer, PruningHook, TokenCounter
from .engine import ContextBudgetEngine, HeuristicTokenCounter
from .dtoc import CompressedToolOutput, DynamicToolCompressor
from .prompts import format_official_cot_prompt, format_qa_prompt
from .routing import (
    ContextRoute,
    QueryIntent,
    RouteDecision,
    classify_query_intent,
    create_persona_context_item,
    has_temporal_anchor,
    is_aggregation_query,
    is_preference_query,
    is_temporal_query,
    normalize_temporal_chunk,
    resolve_dynamic_budget,
    route_by_fit,
)

# NVIDIA DLI Cognitive & Acceleration Extensions
from .compression_vllm import (
    BlackwellNVFP4Quantizer,
    DuoAttentionKVCacheManager,
    FlashAttention3Model,
    ModelArchitectureSpec,
    PagedKVCacheManager,
    PrecisionFormat,
    QuantizationAwareDistillationLoss,
    VRAMFootprintEstimator,
)
from .state_graph import (
    END,
    START,
    CompiledGraph,
    ReActAgent,
    ReActStep,
    StateGraph,
    ToolDefinition,
    ToolExecutionResult,
)
from .rag_triad import (
    HyDEQueryTransformer,
    RAGTriadEvaluator,
    RAGTriadReport,
    RetrievalCandidate,
    TwoStageReranker,
)
from .cuda_memory import (
    ManagedPage,
    MemoryLocation,
    UnifiedMemoryPrefetcher,
    WarpCoalescingAnalyzer,
)
from .knowledge_engine import (
    CatastrophicForgettingEstimator,
    KnowledgeIncorporationSelector,
    KnowledgeRecommendation,
    KnowledgeRequirements,
    KnowledgeStrategy,
)

from .open_loops import OpenLoop, OpenLoopStatus, OpenLoopTracker
from .suppression import SuppressionController, SuppressionRule, SuppressionScope
from .governor import BehavioralGovernor, GovernorDecision, GovernorVerdict
from .self_history import ActionOutcome, AgentSelfHistoryTracker, SelfHistoryRecord

__version__ = "0.1.0a4"

__all__ = [
    "CognitiveDimension",
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
    "format_official_cot_prompt",
    "format_qa_prompt",
    "ContextRoute",
    "QueryIntent",
    "RouteDecision",
    "classify_query_intent",
    "create_persona_context_item",
    "has_temporal_anchor",
    "is_aggregation_query",
    "is_preference_query",
    "is_temporal_query",
    "normalize_temporal_chunk",
    "resolve_dynamic_budget",
    "route_by_fit",
    # NVIDIA DLI Extensions
    "PrecisionFormat",
    "ModelArchitectureSpec",
    "VRAMFootprintEstimator",
    "PagedKVCacheManager",
    "DuoAttentionKVCacheManager",
    "BlackwellNVFP4Quantizer",
    "FlashAttention3Model",
    "QuantizationAwareDistillationLoss",
    "START",
    "END",
    "StateGraph",
    "CompiledGraph",
    "ReActAgent",
    "ReActStep",
    "ToolDefinition",
    "ToolExecutionResult",
    "RetrievalCandidate",
    "TwoStageReranker",
    "HyDEQueryTransformer",
    "RAGTriadEvaluator",
    "RAGTriadReport",
    "MemoryLocation",
    "ManagedPage",
    "UnifiedMemoryPrefetcher",
    "WarpCoalescingAnalyzer",
    "KnowledgeStrategy",
    "KnowledgeRequirements",
    "KnowledgeRecommendation",
    "KnowledgeIncorporationSelector",
    "CatastrophicForgettingEstimator",
    # Governed Cognitive Extensions (CAAI / Evolution 2 Integration)
    "OpenLoop",
    "OpenLoopStatus",
    "OpenLoopTracker",
    "SuppressionScope",
    "SuppressionRule",
    "SuppressionController",
    "GovernorVerdict",
    "GovernorDecision",
    "BehavioralGovernor",
    "ActionOutcome",
    "SelfHistoryRecord",
    "AgentSelfHistoryTracker",
]
