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
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, unquote
from pydantic import BaseModel, Field
from .models import _utf8

DEFAULT_STORE_MAX_BYTES = 64 * 1024 * 1024

# A ref is the handle ``restore`` takes and the one a placeholder tells the model to request, so two different
# outputs must never share one. It is the first 128 bits of a sha256: two outputs with one ref take about 2**64
# hashes to find. Earlier releases kept 48 bits, which a birthday search of about 2**24 hashes defeats.
_REF_HEX_DIGITS = 32

# Written right after the ref when the header's values are percent-encoded (see ``format_placeholder``). A header
# without it is read as written, which is also how every earlier release wrote and read its placeholders.
_HEADER_ENCODED = "enc=pct"

# Header values that are text: read back as written, never as a bool or a number, whatever they spell.
_HEADER_TEXT_KEYS = frozenset({"ref", "tool", "scope"})


def _header_unsafe(ch: str) -> bool:
    """A character a header value cannot carry as is: ``]`` ends the header, whitespace ends the token (and lets
    the rest pass for another ``key=value``), ``%`` is the escape itself, and other non-printing characters
    (controls, format and separator characters) would be invisible."""
    return ch in "%]" or ch.isspace() or not ch.isprintable()


def _header_encode(text: str) -> str:
    return "".join(quote(ch, safe="", errors="surrogatepass") if _header_unsafe(ch) else ch for ch in text)


class CompressedToolOutput(BaseModel):
    """Reversible compression artifact for a tool execution output."""
    ref_id: str = Field(description="Unique reference handle for retrieving the raw output.")
    tool_name: str = Field(description="Name of the invoked tool.")
    summary: str = Field(description="High-signal semantic summary or truncated preview.")
    raw_payload: str = Field(description="Full uncompressed output payload stored in external storage.")
    raw_byte_size: int = Field(ge=0, description="Size of uncompressed payload in bytes.")
    raw_lines: int = Field(ge=0, description="Total line count of uncompressed payload.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    confidential: bool = Field(default=False, description="Whether output is confidential (withheld from remote destinations).")
    scope: Optional[str] = Field(default=None, description="Tenant or project isolation scope.")
    exit_code: Optional[int] = Field(default=None, description="Process exit code if output was from a command.")
    partial: bool = Field(default=False, description="Whether output is truncated or partial (e.g. killed process exit 143).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata.")

    def _sidecar(self) -> Tuple[bool, Optional[str], Optional[int], bool]:
        """The typed sidecar ``(confidential, scope, exit_code, partial)`` as the placeholder and the context item
        both report it: a typed field wins over the same key in the free ``metadata``, which only fills a field that
        is unset; confidential and partial are set when either says so."""
        ec = self.exit_code if self.exit_code is not None else self.metadata.get("exit_code")
        return (
            self.confidential or bool(self.metadata.get("confidential")),
            self.scope or self.metadata.get("scope"),
            ec,
            self.partial or bool(self.metadata.get("partial")) or ec == 143,
        )

    def format_placeholder(self) -> str:
        """Render the compact, structured placeholder inserted into active context.

        The header is space-separated ``key=value`` tokens. When the ref, the tool name, the scope or the exit code
        holds a ``]``, a ``%``, whitespace or another non-printing character, ``enc=pct`` follows the ref and every
        value is percent-encoded (UTF-8): no value can end the header or pass for a token of its own, so
        ``parse_placeholder`` reads back exactly this artifact's ref and sidecar. Any other header is written as
        earlier releases wrote it."""
        confidential, scope_val, ec, partial = self._sidecar()
        text = {"ref": str(self.ref_id), "tool": str(self.tool_name)}
        if scope_val:
            text["scope"] = str(scope_val)
        if ec is not None:
            text["exit_code"] = str(ec)
        if any(_header_unsafe(ch) for value in text.values() for ch in value):
            text = {key: _header_encode(value) for key, value in text.items()}
            tokens = [f"ref={text['ref']}", _HEADER_ENCODED]
        else:
            tokens = [f"ref={text['ref']}"]
        tokens += [
            f"tool={text['tool']}",
            f"lines={self.raw_lines}",
            f"bytes={self.raw_byte_size}",
        ]
        if confidential:
            tokens.append("confidential=true")
        if "scope" in text:
            tokens.append(f"scope={text['scope']}")
        if "exit_code" in text:
            tokens.append(f"exit_code={text['exit_code']}")
        if partial:
            tokens.append("partial=true")

        header = " ".join(tokens)
        return (
            f"[TOOL_OUTPUT_COMPRESSED: {header}]\n"
            f"Summary: {self.summary}\n"
            f"[Use ref='{text['ref']}' to expand if needed]"
        )


class LRUArtifactStore(OrderedDict):
    """Artifact store bounded by bytes of raw payload, least recently used evicted first.

    A compressor that keeps every artifact for the life of the process holds every large tool output it ever
    saw; a gateway running for weeks grows without limit. The cap is on ``raw_byte_size`` because that is what
    the artifact costs; the count of artifacts is not bounded separately. A read (``get``) keeps an artifact
    warm. One artifact larger than the cap is still stored on its own, so a single huge output can always be
    restored once; the next insert evicts it. ``evictions`` counts what was dropped."""

    def __init__(self, max_bytes: int = DEFAULT_STORE_MAX_BYTES) -> None:
        super().__init__()
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.max_bytes = int(max_bytes)
        self.total_bytes = 0
        self.evictions = 0

    @staticmethod
    def _size(artifact: Any) -> int:
        return int(getattr(artifact, "raw_byte_size", 0) or len(str(getattr(artifact, "raw_payload", ""))))

    def __setitem__(self, key: str, artifact: Any) -> None:
        if key in self:
            self.total_bytes -= self._size(super().__getitem__(key))
            super().__delitem__(key)
        super().__setitem__(key, artifact)
        self.total_bytes += self._size(artifact)
        # len > 1: never evict the artifact just inserted, even when it alone exceeds the cap.
        while self.total_bytes > self.max_bytes and len(self) > 1:
            _, oldest = self.popitem(last=False)
            self.total_bytes -= self._size(oldest)
            self.evictions += 1

    def __delitem__(self, key: str) -> None:
        self.total_bytes -= self._size(super().__getitem__(key))
        super().__delitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def pop(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            value = super().__getitem__(key)
            del self[key]
            return value
        return default

    def clear(self) -> None:
        super().clear()
        self.total_bytes = 0


class DynamicToolCompressor:
    """Manages reversible compression and expansion of verbose tool outputs.

    Artifacts live in an :class:`LRUArtifactStore` capped at ``store_max_bytes`` of raw payload (default 64 MiB).
    ``restore`` returns ``None`` for a reference that was evicted, the same as for one that never existed, so
    callers must treat a placeholder as a hint, not a guarantee."""

    def __init__(
        self,
        min_compress_chars: int = 800,
        preview_lines: int = 4,
        tail_lines: int = 4,
        max_preview_chars: int = 600,
        store_max_bytes: int = DEFAULT_STORE_MAX_BYTES,
    ) -> None:
        if min_compress_chars < 1:
            raise ValueError("min_compress_chars must be >= 1")
        if preview_lines < 1:
            raise ValueError("preview_lines must be >= 1")
        if tail_lines < 1:
            raise ValueError("tail_lines must be >= 1")
        if max_preview_chars < 1:
            raise ValueError("max_preview_chars must be >= 1")
        self.min_compress_chars = min_compress_chars
        self.preview_lines = preview_lines
        self.tail_lines = tail_lines
        self.max_preview_chars = max_preview_chars
        self._store: Dict[str, CompressedToolOutput] = LRUArtifactStore(store_max_bytes)

    @property
    def store_bytes(self) -> int:
        """Bytes of raw payload currently held."""
        return getattr(self._store, "total_bytes", sum(a.raw_byte_size for a in self._store.values()))

    @property
    def store_evictions(self) -> int:
        """Artifacts dropped to stay under the cap since construction."""
        return getattr(self._store, "evictions", 0)

    @classmethod
    def parse_placeholder(cls, text: str) -> Optional[Dict[str, Any]]:
        """Parse structured header tokens from a [TOOL_OUTPUT_COMPRESSED: ...] placeholder.

        Reads both header forms ``CompressedToolOutput.format_placeholder`` writes: with ``enc=pct`` right after
        the ref every value is percent-decoded, and without it the values are taken as written, as earlier
        releases wrote them. ``ref``, ``tool`` and ``scope`` are always text; ``lines``, ``bytes`` and
        ``exit_code`` are ints when they read as one; any other ``true``/``false`` is a bool."""
        import re
        m = re.search(r"\[TOOL_OUTPUT_COMPRESSED:\s+([^\]]+)\]", text)
        if not m:
            return None
        tokens = m.group(1).split()
        encoded = len(tokens) > 1 and tokens[1] == _HEADER_ENCODED
        if encoded:
            del tokens[1]
        result: Dict[str, Any] = {}
        for t in tokens:
            if "=" in t:
                k, v = t.split("=", 1)
                if encoded:
                    v = unquote(v, errors="surrogatepass")
                if k in _HEADER_TEXT_KEYS:
                    result[k] = v
                elif v.lower() == "true":
                    result[k] = True
                elif v.lower() == "false":
                    result[k] = False
                elif k in ("lines", "bytes", "exit_code"):
                    try:
                        result[k] = int(v)
                    except ValueError:
                        result[k] = v
                else:
                    result[k] = v
        return result

    def maybe_compress(
        self,
        tool_name: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidential: bool = False,
        scope: Optional[str] = None,
        exit_code: Optional[int] = None,
        partial: bool = False,
    ) -> tuple[str, Optional[CompressedToolOutput]]:
        """Compress tool output if it exceeds threshold, otherwise return raw.

        Returns:
            (context_text, compressed_artifact_if_any)
        """
        if len(output) <= self.min_compress_chars:
            return output, None

        lines = output.splitlines()
        raw_lines = len(lines)
        # Tool output cut in the middle of an escaped emoji holds a lone surrogate: _utf8 sizes and hashes it
        # instead of raising, and gives valid text the size and the ref it always had.
        raw_size = len(_utf8(output))

        # Resolve typed sidecar flags from arguments and metadata
        meta = dict(metadata or {})
        is_confidential = confidential or bool(meta.get("confidential"))
        item_scope = scope or meta.get("scope")
        ec = exit_code if exit_code is not None else meta.get("exit_code")
        is_partial = partial or bool(meta.get("partial")) or (ec == 143) or ("[TRUNCATED" in output)

        # Compute deterministic reference ID: the first 128 bits of a sha256 (``_REF_HEX_DIGITS``). In the tool name a
        # backslash and a colon are escaped, so the colon after it always ends it: tool "a:b" with output "c" and
        # tool "a" with output "b:c" hash different text. A name without either character hashes as before.
        safe_tool = str(tool_name).replace("\\", "\\\\").replace(":", "\\:")
        h = hashlib.sha256(_utf8(f"{safe_tool}:{output}")).hexdigest()[:_REF_HEX_DIGITS]
        ref_id = f"dtoc_{h}"
        # The same tool and output always get the same ref, and storing them again only refreshes the artifact.
        # A ref already holding another tool's or another output is never replaced: restore() of that ref must keep
        # returning the output its placeholder summarized.
        held = self._store.get(ref_id)
        if held is not None and (held.tool_name != tool_name or held.raw_payload != output):
            raise ValueError(f"DTOC ref {ref_id} already holds a different tool output; refusing to replace it")

        # Generate preview summary
        effective_tail = max(self.tail_lines, 5) if (ec not in (None, 0) or is_partial) else self.tail_lines
        head = "\n".join(lines[:self.preview_lines])
        tail = "\n".join(lines[-effective_tail:]) if raw_lines > self.preview_lines + effective_tail else ""
        summary_preview = f"{head}\n...\n{tail}" if tail else head
        if len(summary_preview) > self.max_preview_chars:
            summary_preview = summary_preview[:self.max_preview_chars].rstrip() + "\n..."

        if (ec == 143 or is_partial) and not summary_preview.startswith("[PARTIAL"):
            if ec == 143:
                summary_preview = f"[PARTIAL / KILLED: exit 143]\n{summary_preview}"
            else:
                summary_preview = f"[PARTIAL OUTPUT]\n{summary_preview}"

        artifact = CompressedToolOutput(
            ref_id=ref_id,
            tool_name=tool_name,
            summary=summary_preview,
            raw_payload=output,
            raw_byte_size=raw_size,
            raw_lines=raw_lines,
            confidential=is_confidential,
            scope=item_scope,
            exit_code=ec,
            partial=is_partial,
            metadata=meta,
        )
        self._store[ref_id] = artifact
        return artifact.format_placeholder(), artifact

    def to_context_item(
        self,
        artifact: CompressedToolOutput,
        role: str = "tool",
        priority: int = 50,
    ) -> Any:
        """Convert a CompressedToolOutput into a ContextItem, preserving sidecar flags.

        The item's metadata is the artifact's free metadata with the typed sidecar written over it: ``ref_id``,
        ``tool_name``, ``scope``, ``exit_code`` and ``partial`` (and ``confidential`` when the free metadata has that
        key) are always this artifact's own, as its placeholder reports them, so metadata carried over from an
        earlier item cannot point this one at the earlier artifact or replace its scope or exit code."""
        from .models import ContextItem

        confidential, scope, exit_code, partial = artifact._sidecar()
        metadata = dict(artifact.metadata)
        if "confidential" in metadata:
            metadata["confidential"] = confidential
        metadata.update(
            ref_id=artifact.ref_id,
            tool_name=artifact.tool_name,
            scope=scope,
            exit_code=exit_code,
            partial=partial,
        )
        return ContextItem(
            id=f"item_{artifact.ref_id}",
            role=role,
            content=artifact.format_placeholder(),
            priority=priority,
            sensitivity="confidential" if confidential else "normal",
            metadata=metadata,
        )

    def restore(self, ref_id: str) -> Optional[str]:
        """Reversibly restore the raw tool output payload from external store."""
        artifact = self._store.get(ref_id)
        if artifact is None:
            return None
        return artifact.raw_payload
