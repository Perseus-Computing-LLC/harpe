"""Write paths are all or nothing: a step that raises part-way leaves no half-applied state behind.

Covers the two-pass receipt seal (sealed before the ledger write starts), ProfileLedger.add and
StateLedger.add/add_units (read in full, then committed), and the KV-cache simulators' allocations.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

import harpe.profile as profile_mod
from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.compression_vllm import DuoAttentionKVCacheManager, PagedKVCacheManager
from harpe.profile import DEFAULT_ATTRIBUTES, AttributeSpec, ProfileLedger
from harpe.state_ledger import StateLedger
from harpe.two_pass import TwoPassBuild


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _build(total: int = 600) -> TwoPassBuild:
    policy = ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)
    a = ContextItem(id="a", content="a: a one-line description", token_count=20)
    b = ContextItem(id="b", content="b: a one-line description", token_count=15)
    report = ContextBudgetEngine().synthesize([policy, a, b], _budget(total))
    bodies = {
        "a": ContextItem(id="a", content="a: the full body", token_count=200),
        "b": ContextItem(id="b", content="b: the full body", token_count=150),
    }
    return TwoPassBuild(report, bodies=bodies, budget_tokens=total)


# ---- two-pass receipt -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, ""])
def test_an_emit_refused_before_writing_leaves_the_build_open(tmp_path: Path, bad):
    build = _build()
    with pytest.raises((TypeError, ValueError)):
        build.emit(ledger_path=bad)  # type: ignore[arg-type]
    assert not build.sealed and build.receipt_sequence is None
    assert build.request("b") is True  # Pass 2 may continue: nothing was written
    ledger = tmp_path / "argus.jsonl"
    assert build.emit(ledger_path=ledger) == 1
    assert build.sealed and build.receipt_sequence == 1


def test_an_emit_refused_for_its_actor_writes_nothing_and_stays_open(tmp_path: Path):
    build = _build()
    ledger = tmp_path / "argus.jsonl"
    with pytest.raises(ValueError):
        build.emit(ledger_path=ledger, actor="")
    assert not ledger.exists() and not build.sealed
    assert build.emit(ledger_path=ledger) == 1


def test_an_emit_whose_write_raises_keeps_the_build_sealed(tmp_path: Path, monkeypatch):
    import argus_ledger

    def failing_append(self, *args, **kwargs):
        raise OSError(5, "simulated: the ledger write failed part-way")

    build = _build()
    before = [(it.id, it.content, it.token_count) for it in build.items()]
    monkeypatch.setattr(argus_ledger.AppendOnlyLedger, "append", failing_append)
    with pytest.raises(OSError):
        build.emit(ledger_path=tmp_path / "argus.jsonl")
    monkeypatch.undo()

    assert build.sealed and build.receipt_sequence is None
    with pytest.raises(RuntimeError, match="may be in the ledger"):
        build.request("a")
    with pytest.raises(RuntimeError, match="may be in the ledger"):
        build.emit(ledger_path=tmp_path / "argus.jsonl")
    assert [(it.id, it.content, it.token_count) for it in build.items()] == before
    assert not (tmp_path / "argus.jsonl").exists()  # the refused retry wrote nothing


# ---- ProfileLedger.add ------------------------------------------------------------------------------------------

def _raising_value(_sentence: str, _start: int):
    raise RuntimeError("simulated: a custom value extractor failed")


def test_profile_add_that_raises_takes_no_arrival_and_moves_nothing():
    broken = AttributeSpec("pet", "The user's pet", True, statements=(r"\bmy\s+dog\b",), question=(r"\bdog\b",),
                           value=_raising_value)
    ledger = ProfileLedger(list(DEFAULT_ATTRIBUTES) + [broken])
    ledger.add("u1", "I live in Denver now.")  # arrival (0,)
    state = (ledger.latest_order, ledger.latest_date, ledger.statements("residence"))

    with pytest.raises(RuntimeError):
        ledger.add("u2", "I live in Austin now and my dog loves it.")
    with pytest.raises(RuntimeError):
        ledger.add("u3", "I live in Boston now and my dog loves it.", date="2026-03-01", order=(9, 0))
    assert (ledger.latest_order, ledger.latest_date, ledger.statements("residence")) == state

    found = ledger.add("u4", "I live in Seattle now.")
    assert [s.order for s in found] == [(1, 0)], "a failed add consumed an arrival position"


def test_profile_add_with_text_of_the_wrong_type_records_nothing():
    ledger = ProfileLedger()
    with pytest.raises(Exception):
        ledger.add("u1", b"I live in Denver now.", date="2026-01-01", order=(1, 0))  # type: ignore[arg-type]
    assert ledger.latest_order is None and ledger.latest_date is None
    assert ledger.statements("residence") == []


def test_profile_add_from_another_speaker_still_moves_the_latest_session():
    ledger = ProfileLedger()
    assert ledger.add("a1", "You live in Denver.", date="2026-02-01", order=(2, 0), speaker="assistant") == []
    assert ledger.latest_order == (2, 0) and ledger.latest_date == "2026-02-01"


def test_profile_add_crash_in_the_parse_leaves_the_latest_session(monkeypatch):
    ledger = ProfileLedger()
    ledger.add("u1", "I live in Denver now.", date="2026-01-01", order=(1, 0))

    def crash(*_a, **_k):
        raise RuntimeError("simulated kill during ingest")

    monkeypatch.setattr(profile_mod, "split_sentences", crash)
    with pytest.raises(RuntimeError):
        ledger.add("u5", "I moved to Seattle.", date="2026-05-01", order=(5, 0))
    monkeypatch.undo()
    assert ledger.latest_order == (1, 0) and ledger.latest_date == "2026-01-01"


# ---- StateLedger ------------------------------------------------------------------------------------------------

AFFIRM = "I have completed the tax filing paperwork."
DENY = "I never completed the tax filing paperwork."


def test_state_ledger_failed_batch_records_nothing_and_retry_pairs_once():
    ledger = StateLedger()
    ledger.add("u0", "I have fixed the garden fence.", "2025-12-01")
    with pytest.raises(Exception):
        ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, datetime.date(2026, 2, 1))})
    assert [c.unit_id for c in ledger.claims] == ["u0"]
    assert ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, "2026-02-01")}) == 2
    assert [c.unit_id for c in ledger.claims] == ["u0", "u1", "u2"]
    assert [c.seq for c in ledger.claims] == [0, 1, 2]
    assert len(ledger.conflicts_for_entity(["tax filing paperwork"])) == 1


def test_state_ledger_add_that_raises_records_nothing():
    ledger = StateLedger()
    with pytest.raises(Exception):
        ledger.add("u1", AFFIRM + " " + DENY, datetime.date(2026, 1, 1))  # type: ignore[arg-type]
    assert ledger.claims == []
    assert ledger.add("u1", AFFIRM + " " + DENY, "2026-01-01") == 2
    assert [c.seq for c in ledger.claims] == [0, 1]


# ---- KV-cache simulators ----------------------------------------------------------------------------------------

def test_paged_allocation_takes_blocks_in_order_and_out_of_memory_takes_none():
    mgr = PagedKVCacheManager(total_vram_bytes_for_kv=16 * 64 * 10, bytes_per_token=64, block_size_tokens=16)
    free = list(mgr.free_blocks)
    ids = mgr.allocate_sequence("s1", 40)
    assert ids == free[:3] and mgr.free_blocks == free[3:]
    assert [mgr.physical_blocks[i].num_tokens_stored for i in ids] == [16, 16, 8]
    with pytest.raises(MemoryError):
        mgr.allocate_sequence("s2", 16 * (len(mgr.free_blocks) + 1))
    assert mgr.free_blocks == free[3:] and "s2" not in mgr.block_tables


def test_duo_allocation_that_fails_to_size_registers_nothing(monkeypatch):
    mgr = DuoAttentionKVCacheManager(num_layers=2, num_heads=8, head_dim=64)

    def failing(_seq_len):
        raise ValueError("simulated: sizing failed")

    monkeypatch.setattr(mgr, "compute_sequence_vram_bytes", failing)
    with pytest.raises(ValueError):
        mgr.allocate_sequence("s1", 128)
    assert "s1" not in mgr.sequences
