"""severity: medium
finding: a DTOC placeholder header carries tool_name and scope unescaped, so parse_placeholder reads back a different sidecar: a space or ']' drops confidential, truncates scope or tool, or replaces ref

``CompressedToolOutput.format_placeholder`` (src/harpe/dtoc.py:49-72) writes the header as space-separated
``key=value`` tokens, putting ``tool={self.tool_name}`` (dtoc.py:53) and ``scope={scope_val}`` (dtoc.py:60) in as
given. ``DynamicToolCompressor.parse_placeholder`` (dtoc.py:169-192) captures the header up to the first ``]``
(``\\[TOOL_OUTPUT_COMPRESSED:\\s+([^\\]]+)\\]``, dtoc.py:173), splits it on whitespace (dtoc.py:176), and lets a later
``key=`` token overwrite an earlier one (dtoc.py:178-191). The two do not agree as soon as a tool name or a scope
holds a space or a ``]``:

* scope ``"Acme Corp"`` parses as ``"Acme"`` and tool ``"Google Search"`` as ``"Google"``: the scope is a
  tenant/project isolation scope (dtoc.py:32), and it comes back as a different scope;
* tool ``"search]"`` ends the header early, so ``confidential`` (and lines, bytes, scope) is gone from the parse of
  a confidential artifact's placeholder;
* a scope that contains ``confidential=false`` after a space overrides the artifact's own ``confidential=true``,
  and a tool name that contains ``ref=...`` replaces the artifact's ``ref``, which is the handle ``restore`` takes.

H-R3's acceptance is that the sidecar flag survives the compaction round trip (tests/test_dtoc_sidecar_hr34.py:1-6,
39-45 parse the placeholder back and compare it with the artifact), and the round-1 fix made ``to_context_item``
write the typed sidecar over free metadata "matching the placeholder" (CHANGELOG.md, Unreleased/Fixed). Round 1's
F5 closed the free-metadata route by which a typed sidecar value is replaced; the placeholder header is a sibling
route, still open: free text in two typed fields rewrites the others. Severity medium, as for F5: nothing inside
harpe makes an access decision from a parsed placeholder (the engine reads ``ContextItem.sensitivity``), but a
consumer that reads the placeholder - the documented way to read one - gets the wrong scope, flag or ref.

The tests accept any fix: ``maybe_compress`` may refuse such a tool name or scope (``ValueError``/``TypeError``), or
the placeholder may encode them so that the parse gives back exactly the artifact's own sidecar.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from harpe import DynamicToolCompressor


def _raw() -> str:
    return "\n".join(f"row {i}: account balance {1000 + i}" for i in range(40))


def _round_trip(tool: str, scope: Optional[str]) -> Optional[Dict[str, Any]]:
    """Compress a confidential output; return the artifact and its parsed placeholder, or None if refused."""
    comp = DynamicToolCompressor(min_compress_chars=50)
    try:
        placeholder, artifact = comp.maybe_compress(tool, _raw(), confidential=True, scope=scope)
    except (ValueError, TypeError):
        return None  # refusing a tool name or scope the header cannot carry is a correct fix
    assert artifact is not None
    parsed = DynamicToolCompressor.parse_placeholder(placeholder)
    assert parsed is not None
    return {"artifact": artifact, "parsed": parsed}


def _assert_sidecar_survives(tool: str, scope: Optional[str]) -> None:
    got = _round_trip(tool, scope)
    if got is None:
        return
    artifact, parsed = got["artifact"], got["parsed"]
    assert parsed.get("confidential") is True, f"confidential read back as {parsed.get('confidential')!r}: {parsed}"
    assert parsed.get("scope") == scope, f"scope read back as {parsed.get('scope')!r}: {parsed}"
    assert parsed.get("ref") == artifact.ref_id, f"ref read back as {parsed.get('ref')!r}: {parsed}"
    assert parsed.get("tool") == tool, f"tool read back as {parsed.get('tool')!r}: {parsed}"


def test_control_an_ordinary_tool_and_scope_survive_the_round_trip():
    got = _round_trip("query_vault", "tenant-perseus")
    assert got is not None
    artifact, parsed = got["artifact"], got["parsed"]
    assert parsed["ref"] == artifact.ref_id and parsed["tool"] == "query_vault"
    assert parsed["confidential"] is True and parsed["scope"] == "tenant-perseus"
    assert parsed["lines"] == 40 and parsed["bytes"] == artifact.raw_byte_size


def test_a_scope_with_a_space_is_read_back_as_that_scope():
    _assert_sidecar_survives("Google Search", "Acme Corp")


def test_a_tool_name_with_a_bracket_keeps_the_confidential_flag():
    _assert_sidecar_survives("search]", "tenant-perseus")


def test_a_scope_cannot_override_the_artifacts_own_confidential_flag():
    _assert_sidecar_survives("query_vault", "acme confidential=false")


def test_a_tool_name_cannot_replace_the_artifacts_ref():
    _assert_sidecar_survives("fetch ref=dtoc_000000000000", "tenant-perseus")
