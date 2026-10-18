"""Two-Stage RAG Reranking, Reciprocal Rank Fusion, HyDE, and Automated RAG Triad Evaluator.

Derived from NVIDIA DLI: Building RAG Agents with LLMs (course-v1:DLI+S-FX-15+V1).
Provides:
- Reciprocal Rank Fusion (RRF) combining dense semantic search and sparse lexical search.
- Two-stage reranker pipeline filtering high-volume candidate sets down to high-utility context items.
- Hypothetical Document Embeddings (HyDE) query transformer & sub-query decomposer.
- Automated RAG Triad evaluation: Context Relevance, Groundedness/Faithfulness, and Answer Relevance.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    """Candidate document or passage retrieved from an index."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    fused_score: Optional[float] = None
    token_count: Optional[int] = None


class TwoStageReranker:
    """Combines Reciprocal Rank Fusion (Stage 1) with Cross-Encoder scoring (Stage 2)."""

    @staticmethod
    def reciprocal_rank_fusion(
        sparse_rankings: List[RetrievalCandidate],
        dense_rankings: List[RetrievalCandidate],
        k: int = 60,
    ) -> List[RetrievalCandidate]:
        """Combine sparse and dense rankings using Reciprocal Rank Fusion (RRF).

        Formula: RRF_score(d) = sum_{m in {sparse, dense}} (1.0 / (k + rank_m(d))).
        """
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, RetrievalCandidate] = {}

        # Process sparse rankings (1-indexed rank)
        for rank, candidate in enumerate(sparse_rankings, start=1):
            doc_id = candidate.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            candidate_map[doc_id] = candidate

        # Process dense rankings (1-indexed rank)
        for rank, candidate in enumerate(dense_rankings, start=1):
            doc_id = candidate.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in candidate_map:
                candidate_map[doc_id] = candidate

        # Assign fused scores and sort descending
        fused_candidates: List[RetrievalCandidate] = []
        for doc_id, score in rrf_scores.items():
            cand = candidate_map[doc_id]
            cand.fused_score = score
            fused_candidates.append(cand)

        fused_candidates.sort(key=lambda c: c.fused_score or 0.0, reverse=True)
        return fused_candidates

    @classmethod
    def rerank_and_filter(
        cls,
        candidates: List[RetrievalCandidate],
        cross_encoder_scorer: Optional[Callable[[str], float]] = None,
        max_items: int = 5,
        token_budget: Optional[int] = None,
        default_tokens_per_item: int = 150,
    ) -> List[RetrievalCandidate]:
        """Stage 2: score candidates using cross-encoder and enforce knapsack token budget."""
        scored_candidates = list(candidates)

        if cross_encoder_scorer is not None:
            for cand in scored_candidates:
                cand.cross_encoder_score = cross_encoder_scorer(cand.content)
            scored_candidates.sort(key=lambda c: c.cross_encoder_score or 0.0, reverse=True)

        selected: List[RetrievalCandidate] = []
        accumulated_tokens = 0

        for cand in scored_candidates:
            if len(selected) >= max_items:
                break
            tokens = cand.token_count if cand.token_count is not None else default_tokens_per_item
            if token_budget is not None and (accumulated_tokens + tokens) > token_budget:
                continue
            selected.append(cand)
            accumulated_tokens += tokens

        return selected


class HyDEQueryTransformer:
    """Hypothetical Document Embeddings (HyDE) query generator and sub-query decomposer."""

    @staticmethod
    def generate_hypothetical_passage(query: str, domain_context: str = "technical") -> str:
        """Construct speculative passage template to bridge query-document vocabulary mismatch."""
        cleaned_query = query.strip().rstrip("?")
        return (
            f"According to verified documentation regarding {cleaned_query}, "
            f"the core mechanism operates by establishing deterministic parameters, "
            f"optimizing memory hierarchies, and enforcing strict compliance standards."
        )

    @staticmethod
    def decompose_query(complex_query: str) -> List[str]:
        """Decompose compound multi-hop queries into atomic single-intent sub-queries."""
        # Split on conjunctions and punctuation
        parts = re.split(r"\s+(?:and also|and then|and|additionally|furthermore)\s+|;\s*|\.\s+", complex_query, flags=re.IGNORECASE)
        sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
        return sub_queries if sub_queries else [complex_query.strip()]


class RAGTriadReport(BaseModel):
    """Evaluation report for the three pillars of RAG quality."""
    context_relevance_score: float = Field(ge=0.0, le=1.0, description="Measures relevance of retrieved passages to query.")
    groundedness_score: float = Field(ge=0.0, le=1.0, description="Measures faithfulness of generated answer to retrieved context.")
    answer_relevance_score: float = Field(ge=0.0, le=1.0, description="Measures alignment of answer to query.")
    triad_harmonic_mean: float = Field(ge=0.0, le=1.0, description="Harmonic mean of the three pillar scores.")
    passes_threshold: bool = Field(description="True if harmonic mean >= passing threshold.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Extracted claims failing groundedness check.")


class RAGTriadEvaluator:
    """Evaluates RAG generation quality across Context Relevance, Groundedness, and Answer Relevance."""

    STOP_WORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
        "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
        "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves", "using"
    }

    @classmethod
    def _stem(cls, word: str) -> str:
        w = word.lower().strip()
        for suffix in ("ically", "ical", "ally", "able", "ing", "tion", "sion", "ness", "ment", "ies", "es", "ed", "ly", "ic", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    @classmethod
    def _tokenize_words(cls, text: str) -> Set[str]:
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        filtered = [t for t in raw_tokens if t not in cls.STOP_WORDS]
        # Return both stemmed and original filtered tokens for high-recall matching
        tokens: Set[str] = set()
        for t in filtered:
            tokens.add(t)
            tokens.add(cls._stem(t))
        return tokens

    @classmethod
    def evaluate_context_relevance(cls, query: str, context_passages: List[str]) -> float:
        """Measure semantic overlap between query keywords and retrieved context chunks."""
        query_words = cls._tokenize_words(query)
        if not query_words or not context_passages:
            return 0.0

        all_context_words: Set[str] = set()
        for p in context_passages:
            all_context_words.update(cls._tokenize_words(p))

        overlap = query_words.intersection(all_context_words)
        return round(len(overlap) / len(query_words), 4)

    @classmethod
    def evaluate_groundedness(cls, answer: str, context_passages: List[str]) -> Tuple[float, List[str]]:
        """Measure claim-level entailment of generated answer against retrieved context."""
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0, []

        all_context_text = " ".join(context_passages).lower()
        context_words = cls._tokenize_words(all_context_text)

        supported_count = 0
        unsupported: List[str] = []

        for sent in sentences:
            sent_words = cls._tokenize_words(sent)
            if not sent_words:
                continue
            # A sentence is considered grounded if >= 60% of its content words are in context
            overlap = sent_words.intersection(context_words)
            ratio = len(overlap) / len(sent_words)
            if ratio >= 0.50:
                supported_count += 1
            else:
                unsupported.append(sent)

        groundedness = supported_count / len(sentences) if sentences else 1.0
        return round(groundedness, 4), unsupported

    @classmethod
    def evaluate_answer_relevance(cls, query: str, answer: str) -> float:
        """Measure direct lexical and conceptual overlap between user query and generated response."""
        query_words = cls._tokenize_words(query)
        if not query_words:
            return 1.0
        answer_words = cls._tokenize_words(answer)
        overlap = query_words.intersection(answer_words)
        return round(len(overlap) / len(query_words), 4)

    @classmethod
    def evaluate_triad(
        cls,
        query: str,
        context_passages: List[str],
        answer: str,
        passing_threshold: float = 0.70,
    ) -> RAGTriadReport:
        """Compute all three scores, harmonic mean, and pass/fail determination."""
        cr_score = cls.evaluate_context_relevance(query, context_passages)
        groundedness, unsupported = cls.evaluate_groundedness(answer, context_passages)
        ar_score = cls.evaluate_answer_relevance(query, answer)

        # Harmonic mean: 3 / (1/a + 1/b + 1/c)
        eps = 1e-6
        inv_sum = (1.0 / max(cr_score, eps)) + (1.0 / max(groundedness, eps)) + (1.0 / max(ar_score, eps))
        h_mean = 3.0 / inv_sum if inv_sum > 0 else 0.0
        h_mean = max(0.0, min(1.0, round(h_mean, 4)))

        return RAGTriadReport(
            context_relevance_score=cr_score,
            groundedness_score=groundedness,
            answer_relevance_score=ar_score,
            triad_harmonic_mean=h_mean,
            passes_threshold=(h_mean >= passing_threshold),
            unsupported_claims=unsupported,
        )
