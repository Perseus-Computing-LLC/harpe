"""Knowledge Incorporation Selector & Catastrophic Forgetting Mitigation Engine.

Derived from NVIDIA DLI: Adding New Knowledge to LLMs (course-v1:DLI+S-FX-44+V1).
Provides:
- Knowledge Integration Spectrum decision engine (Prompting vs RAG vs PEFT/LoRA vs SFT vs Continual Pre-Training).
- Quantitative recommendation scoring based on volatility, latency, dataset size, and task complexity.
- Replay buffer mixture calculation (10-20% general data blending) to mitigate catastrophic forgetting.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class KnowledgeStrategy(str, Enum):
    """The spectrum of knowledge integration strategies."""
    PROMPT_ENGINEERING = "prompt_engineering"
    RETRIEVAL_AUGMENTED_GENERATION = "rag"
    PARAMETER_EFFICIENT_FINE_TUNING = "peft_lora"
    SUPERVISED_FINE_TUNING = "sft"
    CONTINUAL_PRETRAINING = "continual_pretraining"


class KnowledgeRequirements(BaseModel):
    """Operational requirements for a target capability."""
    task_name: str
    data_volume_tokens: int = Field(gt=0, description="Approximate size of knowledge corpus in tokens.")
    update_frequency_hours: float = Field(gt=0, description="How often knowledge changes (lower = more volatile).")
    latency_sla_ms: float = Field(gt=0, description="Maximum allowable end-to-end latency in milliseconds.")
    requires_citation: bool = Field(default=True, description="Whether strict auditable source citation is mandatory.")
    requires_novel_reasoning_style: bool = Field(default=False, description="Whether specialized output syntax/reasoning is needed.")
    gpu_budget_available: bool = Field(default=True, description="Whether training GPU budget is accessible.")


class KnowledgeRecommendation(BaseModel):
    """Strategic recommendation for knowledge integration."""
    recommended_strategy: KnowledgeStrategy
    confidence_score: float = Field(ge=0.0, le=1.0)
    primary_justification: str
    secondary_options: List[KnowledgeStrategy] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)


class KnowledgeIncorporationSelector:
    """Evaluates task requirements and selects the optimal integration method across the spectrum."""

    @classmethod
    def evaluate(cls, req: KnowledgeRequirements) -> KnowledgeRecommendation:
        """Select optimal strategy based on the NVIDIA Knowledge Integration Continuum."""
        risks: List[str] = []

        # Rule 1: High volatility (< 24 hours) or strict citation requirement strongly indicates RAG
        if req.requires_citation or req.update_frequency_hours < 48:
            if req.latency_sla_ms < 50:
                risks.append("Low latency SLA (<50ms) makes multi-step RAG retrieval challenging; consider caching or HyDE.")

            if req.requires_novel_reasoning_style and req.gpu_budget_available:
                return KnowledgeRecommendation(
                    recommended_strategy=KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION,
                    confidence_score=0.92,
                    primary_justification=(
                        "High data volatility and mandatory citation require RAG. "
                        "Pair with a lightweight LoRA adapter to satisfy the specialized reasoning format."
                    ),
                    secondary_options=[KnowledgeStrategy.PARAMETER_EFFICIENT_FINE_TUNING],
                    risk_factors=risks,
                )
            else:
                return KnowledgeRecommendation(
                    recommended_strategy=KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION,
                    confidence_score=0.95,
                    primary_justification=(
                        f"Corpus updates every {req.update_frequency_hours:.1f} hours and/or requires explicit citation. "
                        "RAG provides instant updating without costly model retraining or catastrophic forgetting."
                    ),
                    secondary_options=[KnowledgeStrategy.PROMPT_ENGINEERING],
                    risk_factors=risks,
                )

        # Rule 2: Large static corpora (>10M tokens) with foundational domain vocabulary
        if req.data_volume_tokens > 50_000_000 and req.gpu_budget_available:
            return KnowledgeRecommendation(
                recommended_strategy=KnowledgeStrategy.CONTINUAL_PRETRAINING,
                confidence_score=0.88,
                primary_justification=(
                    f"Massive domain corpus ({req.data_volume_tokens:,} tokens) with low volatility. "
                    "Continual pre-training is necessary to align model representations and internalize domain concepts."
                ),
                secondary_options=[KnowledgeStrategy.SUPERVISED_FINE_TUNING, KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION],
                risk_factors=["Requires high GPU compute", "High risk of catastrophic forgetting without replay buffers"],
            )

        # Rule 3: Specialized reasoning / output format on moderate static dataset
        if req.requires_novel_reasoning_style and req.gpu_budget_available:
            return KnowledgeRecommendation(
                recommended_strategy=KnowledgeStrategy.PARAMETER_EFFICIENT_FINE_TUNING,
                confidence_score=0.90,
                primary_justification=(
                    "Task requires specialized reasoning or structured output syntax on static data. "
                    "LoRA/QLoRA achieves domain behavior adaptation while freezing 99%+ of base model weights."
                ),
                secondary_options=[KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION, KnowledgeStrategy.SUPERVISED_FINE_TUNING],
                risk_factors=risks,
            )

        # Rule 4: Small dataset (< 10,000 tokens)
        if req.data_volume_tokens < 10_000:
            return KnowledgeRecommendation(
                recommended_strategy=KnowledgeStrategy.PROMPT_ENGINEERING,
                confidence_score=0.94,
                primary_justification=(
                    f"Small corpus ({req.data_volume_tokens:,} tokens) can fit directly into prompt context window. "
                    "Zero infrastructure overhead, zero training costs."
                ),
                secondary_options=[KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION],
                risk_factors=risks,
            )

        # Default fallback
        return KnowledgeRecommendation(
            recommended_strategy=KnowledgeStrategy.RETRIEVAL_AUGMENTED_GENERATION,
            confidence_score=0.80,
            primary_justification="General domain knowledge retrieval is best served by RAG.",
            secondary_options=[KnowledgeStrategy.PARAMETER_EFFICIENT_FINE_TUNING],
            risk_factors=risks,
        )


class CatastrophicForgettingEstimator:
    """Calculates dataset mixture ratios and replay buffer sizes to preserve base model capabilities."""

    @staticmethod
    def calculate_replay_buffer(
        domain_tokens: int,
        target_general_retention_pct: float = 85.0,
        recommended_replay_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Compute the recommended number of general pretraining/instruction tokens to blend into fine-tuning.

        Based on NVIDIA NeMo best practices:
        Blending 10-20% general tokens preserves MMLU, GSM8K, and coding benchmarks while tuning domain skills.
        """
        if domain_tokens <= 0:
            raise ValueError("domain_tokens must be > 0")
        if not (0.05 <= recommended_replay_ratio <= 0.40):
            raise ValueError("recommended_replay_ratio should be between 0.05 and 0.40")

        # total_tokens = domain_tokens + replay_tokens
        # replay_tokens / total_tokens = recommended_replay_ratio
        # replay_tokens = domain_tokens * ratio / (1 - ratio)
        replay_tokens = int(math.ceil(domain_tokens * recommended_replay_ratio / (1.0 - recommended_replay_ratio)))
        total_tokens = domain_tokens + replay_tokens

        return {
            "domain_tokens": domain_tokens,
            "replay_tokens": replay_tokens,
            "total_mixture_tokens": total_tokens,
            "replay_percentage": round(recommended_replay_ratio * 100.0, 2),
            "estimated_retention_pct": target_general_retention_pct,
            "recommended_learning_rate_scale": 0.5 if recommended_replay_ratio > 0.20 else 1.0,
        }
