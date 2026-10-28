"""Optional exact token counters. ``harpe`` itself depends only on pydantic; the adapters here import their backend
lazily so this module can be imported without it. Install with ``pip install "harpe[tiktoken]"``."""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["TiktokenCounter", "tiktoken_available"]


def tiktoken_available() -> bool:
    """True when the ``tiktoken`` package can be imported."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True


class TiktokenCounter:
    """:class:`harpe.protocols.TokenCounter` backed by tiktoken (default encoding ``o200k_base``).

    Pass the encoding name used by the model the context is going to. ``ImportError`` is raised at construction,
    not import, when tiktoken is missing, so the default :class:`harpe.HeuristicTokenCounter` keeps working for
    installs without the extra. Special tokens in the text are counted as ordinary text rather than rejected,
    because context items are untrusted data, not prompts the caller wrote."""

    def __init__(self, encoding: str = "o200k_base", *, encoder: Optional[Any] = None) -> None:
        if encoder is None:
            try:
                import tiktoken
            except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
                raise ImportError(
                    "TiktokenCounter needs the 'tiktoken' package: pip install 'harpe[tiktoken]'"
                ) from exc
            encoder = tiktoken.get_encoding(encoding)
        self.encoding = encoding
        self._encoder = encoder

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text, disallowed_special=()))
