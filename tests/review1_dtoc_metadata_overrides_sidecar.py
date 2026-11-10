"""severity: medium
finding: DynamicToolCompressor.to_context_item lets the artifact's free metadata override its typed sidecar, so a replayed metadata dict points the new item at the old ref_id and launders scope and exit_code

to_context_item (src/harpe/dtoc.py:246-269) builds the item's metadata as {"ref_id": artifact.ref_id,
"tool_name": ..., "scope": artifact.scope, "exit_code": artifact.exit_code, "partial": artifact.partial,
**artifact.metadata} (dtoc.py:262-267): the caller's untyped metadata is spread last and wins. maybe_compress
resolves the same keys the other way round, typed argument first (dtoc.py:208-210), and so does the placeholder
the item carries as its content (dtoc.py:47). CHANGELOG 0.1.0-alpha.11 (H-R3/H-R4) promises "matching ... scope
metadata" and exit codes preserved through DTOC.

The ordinary way to hit it is a refresh: re-running a tool and passing the previous item's metadata along (it is
exactly what to_context_item wrote: ref_id, tool_name, scope, exit_code, partial). The new item then names the
previous artifact's ref_id, so restoring through ``item.metadata["ref_id"]`` returns the old output, and a stale
exit code or scope replaces the typed one while the item's own placeholder says otherwise.
"""
from __future__ import annotations

from harpe.dtoc import DynamicToolCompressor


def _output(tag: str) -> str:
    return "\n".join(f"{tag} line {i}: value={i * 7}" for i in range(80))


def _item_view(comp: DynamicToolCompressor, artifact):
    item = comp.to_context_item(artifact)
    return item, {k: item.metadata.get(k) for k in ("ref_id", "tool_name", "scope", "exit_code", "partial")}


def test_control_item_metadata_matches_the_artifact_without_caller_metadata():
    comp = DynamicToolCompressor(min_compress_chars=100)
    placeholder, artifact = comp.maybe_compress("fetch", _output("first"), scope="tenant-a", exit_code=143)
    assert artifact is not None
    item, view = _item_view(comp, artifact)
    assert view == {"ref_id": artifact.ref_id, "tool_name": "fetch", "scope": "tenant-a", "exit_code": 143,
                    "partial": True}
    assert item.content == placeholder
    assert comp.restore(view["ref_id"]) == _output("first")


def test_a_replayed_metadata_dict_does_not_point_the_new_item_at_the_old_artifact():
    comp = DynamicToolCompressor(min_compress_chars=100)
    _, first = comp.maybe_compress("fetch", _output("first"), scope="tenant-a")
    assert first is not None
    first_item = comp.to_context_item(first)
    # Refresh: run the tool again and carry the previous item's metadata forward.
    _, second = comp.maybe_compress("fetch", _output("second"), metadata=dict(first_item.metadata), scope="tenant-a")
    assert second is not None and second.ref_id != first.ref_id
    second_item, view = _item_view(comp, second)
    assert view["ref_id"] == second.ref_id, "the refreshed item names the previous artifact"
    assert comp.restore(second_item.metadata["ref_id"]) == _output("second")


def test_typed_scope_and_exit_code_are_what_the_item_reports():
    comp = DynamicToolCompressor(min_compress_chars=100)
    placeholder, artifact = comp.maybe_compress(
        "run_job", _output("job"), metadata={"scope": "public", "exit_code": 0}, scope="tenant-a", exit_code=143
    )
    assert artifact is not None
    assert (artifact.scope, artifact.exit_code, artifact.partial) == ("tenant-a", 143, True)
    assert "scope=tenant-a" in placeholder and "exit_code=143" in placeholder
    _, view = _item_view(comp, artifact)
    assert (view["scope"], view["exit_code"], view["partial"]) == ("tenant-a", 143, True)
