"""severity: none (compatibility control, expected to PASS: exit 0 on 1451dc4)
finding: none. Task C answer: a pre-upgrade DTOC sidecar (`dtoc_` + 12 hex) loads, restores and parses with the new code.

The fixture below is the literal output of the 723ff5b code (`git show 723ff5b:src/harpe/dtoc.py`, executed in
memory on 2026-11-10): `DynamicToolCompressor(min_compress_chars=100).maybe_compress("terminal", OUTPUT,
scope="tenant-a", exit_code=1)`, the artifact as `model_dump()` JSON (the only sidecar form the library has; it never
wrote files itself) and the placeholder text that went into the conversation.

Not covered here because it is a caveat, not a defect under the documented contract ("a placeholder is a hint, not a
guarantee"): a fresh process that re-compresses the same tool output now stores it under the 32-hex ref, so the old
12-hex ref in a live conversation can no longer be revived by re-running the tool
(`test_caveat_old_ref_is_prefix_of_new_ref` pins the relation a migration shim could use).
"""
import hashlib
import json

from harpe.dtoc import CompressedToolOutput, DynamicToolCompressor

OUTPUT = "".join(f"row {i}: value={i * 7}\n" for i in range(40))
OLD_REF = "dtoc_54398e1c589a"
OLD_SUMMARY = (
    "row 0: value=0\nrow 1: value=7\nrow 2: value=14\nrow 3: value=21\n...\n"
    "row 35: value=245\nrow 36: value=252\nrow 37: value=259\nrow 38: value=266\nrow 39: value=273"
)
# The sidecar exactly as 723ff5b serialized it (created_at fixed).
OLD_SIDECAR_JSON = json.dumps({
    "ref_id": OLD_REF, "tool_name": "terminal", "summary": OLD_SUMMARY, "raw_payload": OUTPUT,
    "raw_byte_size": 693, "raw_lines": 40, "created_at": 1762732800.0, "confidential": False,
    "scope": "tenant-a", "exit_code": 1, "partial": False, "metadata": {},
})
# The placeholder exactly as 723ff5b wrote it into the conversation.
OLD_PLACEHOLDER = (
    f"[TOOL_OUTPUT_COMPRESSED: ref={OLD_REF} tool=terminal lines=40 bytes=693 scope=tenant-a exit_code=1]\n"
    f"Summary: {OLD_SUMMARY}\n"
    f"[Use ref='{OLD_REF}' to request full uncompressed expansion if needed]"
)
OLD_CONFIDENTIAL_HEADER = "[TOOL_OUTPUT_COMPRESSED: ref=dtoc_f9bd27bd0351 tool=mcp:fs:read lines=40 bytes=693 confidential=true]"


def test_fixture_is_what_the_old_version_computed():
    assert OLD_REF == "dtoc_" + hashlib.sha256(f"terminal:{OUTPUT}".encode("utf-8")).hexdigest()[:12]
    assert len(OUTPUT.encode("utf-8")) == 693


def test_old_sidecar_loads_and_old_ref_expands():
    artifact = CompressedToolOutput.model_validate_json(OLD_SIDECAR_JSON)
    assert artifact.ref_id == OLD_REF
    comp = DynamicToolCompressor(min_compress_chars=100)
    comp._store[artifact.ref_id] = artifact  # the only way either version re-hydrates a store (the plugin does this)
    assert comp.restore(OLD_REF) == OUTPUT
    assert comp.store_bytes == 693


def test_old_placeholder_still_parses():
    parsed = DynamicToolCompressor.parse_placeholder(OLD_PLACEHOLDER)
    assert parsed == {"ref": OLD_REF, "tool": "terminal", "lines": 40, "bytes": 693, "scope": "tenant-a", "exit_code": 1}
    parsed = DynamicToolCompressor.parse_placeholder(OLD_CONFIDENTIAL_HEADER + "\nSummary: x\n[Use ref='dtoc_f9bd27bd0351' to request full uncompressed expansion if needed]")
    assert parsed == {"ref": "dtoc_f9bd27bd0351", "tool": "mcp:fs:read", "lines": 40, "bytes": 693, "confidential": True}


def test_old_artifact_survives_a_new_compression_of_the_same_output():
    artifact = CompressedToolOutput.model_validate_json(OLD_SIDECAR_JSON)
    comp = DynamicToolCompressor(min_compress_chars=100)
    comp._store[artifact.ref_id] = artifact
    placeholder, new_artifact = comp.maybe_compress("terminal", OUTPUT, scope="tenant-a", exit_code=1)  # no ValueError
    assert new_artifact is not None and new_artifact.ref_id != OLD_REF
    assert comp.restore(OLD_REF) == OUTPUT
    assert comp.restore(new_artifact.ref_id) == OUTPUT


def test_old_artifact_still_becomes_a_context_item_with_its_own_ref():
    artifact = CompressedToolOutput.model_validate_json(OLD_SIDECAR_JSON)
    item = DynamicToolCompressor(min_compress_chars=100).to_context_item(artifact)
    assert item.metadata["ref_id"] == OLD_REF and item.id == f"item_{OLD_REF}"
    assert DynamicToolCompressor.parse_placeholder(item.content)["ref"] == OLD_REF


def test_caveat_old_ref_is_prefix_of_new_ref():
    """For a tool name without ':' or a backslash the hashed text is unchanged, so the old ref is the first 17
    characters of the new one. A tool name with ':' (e.g. 'mcp:fs:read') hashes different text: no relation."""
    _, new_artifact = DynamicToolCompressor(min_compress_chars=100).maybe_compress("terminal", OUTPUT)
    assert new_artifact.ref_id.startswith(OLD_REF) and len(new_artifact.ref_id) == len("dtoc_") + 32
    _, colon = DynamicToolCompressor(min_compress_chars=100).maybe_compress("mcp:fs:read", OUTPUT)
    assert not colon.ref_id.startswith("dtoc_f9bd27bd0351")
