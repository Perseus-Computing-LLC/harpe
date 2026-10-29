"""Tests for H-R3 (confidential/scope sidecar flags) and H-R4 (exit 143 partial output).

H-R3: Typed sidecar for confidential/scope flags on compressed blocks.
      Acceptance: flag survives compaction round-trip.
H-R4: Killed-process (exit 143) partial output stays marked partial;
      exit code + truncation marker preserved through compaction.
"""

import pytest

from harpe import (
    ContextBudgetEngine,
    DynamicToolCompressor,
    TokenBudget,
)
from harpe.dtoc import CompressedToolOutput


def _long_output(prefix="line", count=50):
    return "\n".join(f"{prefix} {i} " + "data" * 20 for i in range(count))


def test_confidential_and_scope_sidecar_flags_in_placeholder():
    """H-R3: confidential and scope flags are preserved in artifact and placeholder header."""
    comp = DynamicToolCompressor(min_compress_chars=100)
    raw = _long_output("secret_dump")
    placeholder, artifact = comp.maybe_compress(
        "query_vault",
        raw,
        confidential=True,
        scope="tenant-perseus",
    )
    assert artifact is not None
    assert artifact.confidential is True
    assert artifact.scope == "tenant-perseus"
    assert "confidential=true" in placeholder
    assert "scope=tenant-perseus" in placeholder

    # Parse back from placeholder
    parsed = DynamicToolCompressor.parse_placeholder(placeholder)
    assert parsed is not None
    assert parsed.get("confidential") is True
    assert parsed.get("scope") == "tenant-perseus"
    assert parsed.get("ref") == artifact.ref_id
    assert parsed.get("tool") == "query_vault"


def test_confidential_flag_survives_context_synthesis_roundtrip():
    """H-R3: confidential item created from compressed artifact is withheld from remote destination."""
    comp = DynamicToolCompressor(min_compress_chars=100)
    raw = _long_output("confidential_record")
    placeholder, artifact = comp.maybe_compress("fetch_record", raw, confidential=True, scope="proj_defense")
    assert artifact is not None

    item = comp.to_context_item(artifact)
    assert item.sensitivity == "confidential"
    assert item.metadata.get("scope") == "proj_defense"

    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=4000)

    # Remote destination must withhold confidential item
    report_remote = engine.synthesize([item], budget, destination="remote")
    assert len(report_remote.included_items) == 0
    assert len(report_remote.withheld_items) == 1
    assert report_remote.withheld_items[0].id == item.id

    # Local destination includes it
    report_local = engine.synthesize([item], budget, destination="local")
    assert len(report_local.included_items) == 1
    assert "confidential=true" in report_local.included_items[0].content
    assert "scope=proj_defense" in report_local.included_items[0].content


def test_killed_process_exit_143_marked_partial():
    """H-R4: Killed process with exit code 143 (SIGTERM) is automatically marked partial."""
    comp = DynamicToolCompressor(min_compress_chars=100)
    raw = _long_output("tail_build_log") + "\n[SIGTERM received, process killed]"
    placeholder, artifact = comp.maybe_compress("terminal", raw, exit_code=143)

    assert artifact is not None
    assert artifact.exit_code == 143
    assert artifact.partial is True
    assert "exit_code=143" in placeholder
    assert "partial=true" in placeholder
    assert "[PARTIAL" in artifact.summary or "[KILLED: exit 143]" in artifact.summary

    parsed = DynamicToolCompressor.parse_placeholder(placeholder)
    assert parsed is not None
    assert parsed.get("exit_code") == 143
    assert parsed.get("partial") is True


def test_metadata_dict_passthrough_for_flags():
    """H-R3/H-R4: metadata dict with confidential, scope, exit_code, partial is respected."""
    comp = DynamicToolCompressor(min_compress_chars=100)
    raw = _long_output("runner_output")
    meta = {
        "confidential": True,
        "scope": "cross-check",
        "exit_code": 137,
        "partial": True,
    }
    placeholder, artifact = comp.maybe_compress("test_runner", raw, metadata=meta)

    assert artifact is not None
    assert artifact.confidential is True
    assert artifact.scope == "cross-check"
    assert artifact.exit_code == 137
    assert artifact.partial is True
    assert "confidential=true" in placeholder
    assert "scope=cross-check" in placeholder
    assert "exit_code=137" in placeholder
    assert "partial=true" in placeholder
