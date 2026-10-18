"""Reusable prompt formatting utilities for benchmark evaluation and QA."""

from __future__ import annotations

from typing import Optional


def format_official_cot_prompt(
    retrieved_context: str,
    arg2: str = "",
    arg3: Optional[str] = None,
    *,
    question_content: Optional[str] = None,
    question_date: Optional[str] = None,
) -> str:
    """Format prompt matching the official LongMemEval Chain-of-Thought (CoT) template.

    Supports both positional calling conventions:
    1. format_official_cot_prompt(retrieved_context, question_date, question_content)
    2. format_official_cot_prompt(retrieved_context, question_content, question_date)
    As well as explicit keyword arguments.

    Template schema:
    <context>
    {retrieved_context}
    </context>

    <question_date> {question_date} </question_date>  (if present)

    <question>
      {question_content}
    </question>

    Answer:
    """
    # Resolve keyword overrides first
    final_content = question_content
    final_date = question_date

    if final_content is None:
        if arg3 is not None:
            # 3 positional arguments: (context, question_date, question_content)
            final_date = final_date or arg2
            final_content = arg3
        else:
            # 2 positional arguments: (context, question_content)
            final_content = arg2

    date_part = f"<question_date> {final_date} </question_date>\n\n" if final_date else ""
    return (
        f"<context>\n"
        f"{retrieved_context}\n"
        f"</context>\n\n"
        f"{date_part}"
        f"<question>\n"
        f"  {final_content}\n"
        f"</question>\n\n"
        f"Answer:"
    )


def format_qa_prompt(retrieved_context: str, question: str) -> str:
    """Standard grounded question-answering prompt."""
    return (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{retrieved_context}\n"
        f"---------------------\n"
        f"Given the context information and no prior knowledge, answer the question: {question}\n"
    )
