"""Dynamic Tool Output Compression (DTOC) for adaptive context management.

Implements explicit and reversible compression of verbose tool outputs within the agent
reasoning loop, maintaining raw tool outputs in external memory while inserting compact,
structured placeholders into the active context window.

Based on DTOC: Dynamic Tool Output Compression for Adaptive Context Management in AI Agents
(arXiv:2609.26121).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class CompressedToolOutput(BaseModel):
    """Reversible compression artifact for a tool execution output."""
    ref_id: str = Field(description="Unique reference handle for retrieving the raw output.")
    tool_name: str = Field(description="Name of the invoked tool.")
    summary: str = Field(description="High-signal semantic summary or truncated preview.")
    raw_payload: str = Field(description="Full uncompressed output payload stored in external storage.")
    raw_byte_size: int = Field(ge=0, description="Size of uncompressed payload in bytes.")
    raw_lines: int = Field(ge=0, description="Total line count of uncompressed payload.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata.")

    def format_placeholder(self) -> str:
        """Render the compact, structured placeholder inserted into active context."""
        return (
            f"[TOOL_OUTPUT_COMPRESSED: ref={self.ref_id} tool={self.tool_name} "
            f"lines={self.raw_lines} bytes={self.raw_byte_size}]\n"
            f"Summary: {self.summary}\n"
            f"[Use ref='{self.ref_id}' to request full uncompressed expansion if needed]"
        )


class DynamicToolCompressor:
    """Manages reversible compression and expansion of verbose tool outputs."""

    def __init__(
        self,
        min_compress_chars: int = 800,
        preview_lines: int = 4,
        max_preview_chars: int = 600,
    ) -> None:
        if min_compress_chars < 1:
            raise ValueError("min_compress_chars must be >= 1")
        if preview_lines < 1:
            raise ValueError("preview_lines must be >= 1")
        if max_preview_chars < 1:
            raise ValueError("max_preview_chars must be >= 1")
        self.min_compress_chars = min_compress_chars
        self.preview_lines = preview_lines
        self.max_preview_chars = max_preview_chars
        self._store: Dict[str, CompressedToolOutput] = {}

    def maybe_compress(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[CompressedToolOutput]]:
        """Compress tool output if it exceeds threshold, otherwise return raw.

        Returns:
            (context_text, compressed_artifact_if_any)
        """
        if len(output) <= self.min_compress_chars:
            return output, None

        lines = output.splitlines()
        raw_lines = len(lines)
        raw_size = len(output.encode("utf-8"))

        # Compute deterministic reference ID
        h = hashlib.sha256(f"{tool_name}:{output}".encode("utf-8")).hexdigest()[:12]
        ref_id = f"dtoc_{h}"

        # Generate preview summary
        head = "\n".join(lines[:self.preview_lines])
        tail = "\n".join(lines[-2:]) if raw_lines > self.preview_lines + 2 else ""
        summary_preview = f"{head}\n...\n{tail}" if tail else head
        if len(summary_preview) > self.max_preview_chars:
            summary_preview = summary_preview[:self.max_preview_chars].rstrip() + "\n..."

        artifact = CompressedToolOutput(
            ref_id=ref_id,
            tool_name=tool_name,
            summary=summary_preview,
            raw_payload=output,
            raw_byte_size=raw_size,
            raw_lines=raw_lines,
            metadata=metadata or {},
        )
        self._store[ref_id] = artifact
        return artifact.format_placeholder(), artifact

    def restore(self, ref_id: str) -> Optional[str]:
        """Reversibly restore the raw tool output payload from external store."""
        artifact = self._store.get(ref_id)
        if artifact is None:
            return None
        return artifact.raw_payload
