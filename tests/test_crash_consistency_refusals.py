"""A refusal that provably writes nothing leaves the state open; a failure once a write may have begun does not.

Two-pass receipts: every gate Argus applies before its first byte (actor grammar, payload int range, a ledger path
that cannot be opened or created as a regular file) is applied before the build is sealed, so such an ``emit`` can
be repeated. A failure inside the ledger append itself keeps the build sealed. KV-cache simulators: a token count
that is not an int >= 0 is refused before any block or counter moves, and a failed append counts no token.
"""
from __future__ import annotations

import os
import random
import string
from pathlib import Path

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.compression_vllm import DuoAttentionKVCacheManager, PagedKVCacheManager
from harpe.two_pass import TwoPassBuild, append_context_built


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _build(budget_tokens: int = 600) -> TwoPassBuild:
    policy = ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)
    a = ContextItem(id="a", content="a: a one-line description", token_count=20)
    b = ContextItem(id="b", content="b: a one-line description", token_count=15)
    report = ContextBudgetEngine().synthesize([policy, a, b], _budget(600))
    bodies = {
        "a": ContextItem(id="a", content="a: the full body", token_count=200),
        "b": ContextItem(id="b", content="b: the full body", token_count=150),
    }
    return TwoPassBuild(report, bodies=bodies, budget_tokens=budget_tokens)


def _assert_open_then_emits(build: TwoPassBuild, ledger: Path) -> None:
    assert not build.sealed and build.receipt_sequence is None
    assert build.request("b") is True
    assert build.emit(ledger_path=ledger) == 1
    assert build.sealed and build.receipt_sequence == 1


# ---- two-pass: refusals that write nothing ----------------------------------------------------------------------

@pytest.mark.parametrize("actor", ["harpe agent", "harpe\tagent", "harpe\n", 'ha"rpe', "ha\\rpe", "härpe",
                                   "x" * 257, "\x7f"])
def test_an_actor_argus_refuses_leaves_the_build_open(tmp_path: Path, actor: str):
    build = _build()
    ledger = tmp_path / "argus.jsonl"
    with pytest.raises(ValueError):
        build.emit(ledger_path=ledger, actor=actor)
    assert not ledger.exists()
    _assert_open_then_emits(build, ledger)


@pytest.mark.parametrize("budget", [2 ** 53, 2 ** 64])
def test_a_payload_int_argus_refuses_leaves_the_build_open(tmp_path: Path, budget: int):
    build = _build(budget_tokens=budget)
    ledger = tmp_path / "argus.jsonl"
    with pytest.raises(ValueError):
        build.emit(ledger_path=ledger)
    assert not ledger.exists() and not build.sealed and build.receipt_sequence is None
    assert build.request("a") is True


def test_the_largest_int_argus_takes_is_written(tmp_path: Path):
    build = _build(budget_tokens=2 ** 53 - 1)
    assert build.emit(ledger_path=tmp_path / "argus.jsonl") == 1


def test_append_context_built_refuses_an_out_of_range_int_and_writes_nothing(tmp_path: Path):
    payload = {**_build().receipt_payload(), "used_tokens": -(2 ** 53)}
    ledger = tmp_path / "argus.jsonl"
    with pytest.raises(ValueError):
        append_context_built(payload, ledger_path=ledger)
    assert not ledger.exists()


def test_a_ledger_path_that_is_a_directory_leaves_the_build_open(tmp_path: Path):
    directory = tmp_path / "ledger-dir"
    directory.mkdir()
    build = _build()
    with pytest.raises(OSError):
        build.emit(ledger_path=directory)
    assert list(directory.iterdir()) == []
    _assert_open_then_emits(build, tmp_path / "argus.jsonl")


def test_a_ledger_path_under_a_regular_file_leaves_the_build_open(tmp_path: Path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_bytes(b"x")
    build = _build()
    with pytest.raises(OSError):
        build.emit(ledger_path=blocker / "sub" / "argus.jsonl")
    assert blocker.read_bytes() == b"x"
    _assert_open_then_emits(build, tmp_path / "argus.jsonl")


def test_a_ledger_path_that_is_a_fifo_leaves_the_build_open(tmp_path: Path):
    fifo = tmp_path / "ledger.fifo"
    os.mkfifo(fifo)
    build = _build()
    with pytest.raises(OSError):
        build.emit(ledger_path=fifo)
    _assert_open_then_emits(build, tmp_path / "argus.jsonl")


def test_a_ledger_in_missing_directories_is_created_and_written(tmp_path: Path):
    ledger = tmp_path / "a" / "b" / "argus.jsonl"
    build = _build()
    assert build.emit(ledger_path=ledger) == 1
    assert ledger.is_file()


def test_harpe_refuses_exactly_the_actors_argus_refuses():
    from argus_ledger import validate_identifiers

    from harpe.two_pass import _check_actor

    rng = random.Random(20261120)
    alphabet = string.printable + "\x00\x7f\u00e4\u2028 "
    samples = ["harpe", "a", "x" * 256, "x" * 257] + [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))) for _ in range(2000)
    ]
    for actor in samples:
        try:
            validate_identifiers("harpe.context_built", actor)
            argus_ok = True
        except ValueError:
            argus_ok = False
        try:
            _check_actor(actor)
            harpe_ok = True
        except ValueError:
            harpe_ok = False
        assert harpe_ok <= argus_ok, actor  # whatever Argus refuses, harpe refuses before sealing
        # Argus's Python grammar uses ``$``, which also matches before a trailing newline; harpe refuses that too.
        assert harpe_ok == argus_ok or actor.endswith("\n"), actor


# ---- two-pass: a failure once the write may have begun keeps the seal -------------------------------------------

@pytest.mark.parametrize("error", [IsADirectoryError(21, "replaced by a directory"), OSError(24, "EMFILE")])
def test_a_failure_inside_the_ledger_append_keeps_the_build_sealed(tmp_path: Path, monkeypatch, error):
    """The path passed every pre-seal check and then failed inside Argus (it changed in between, a descriptor
    limit): harpe cannot prove nothing was written, so the build stays sealed."""
    import argus_ledger

    def failing_open(self):
        raise error

    build = _build()
    ledger = tmp_path / "argus.jsonl"
    monkeypatch.setattr(argus_ledger.AppendOnlyLedger, "_open_locked_for_append", failing_open)
    with pytest.raises(OSError):
        build.emit(ledger_path=ledger)
    monkeypatch.undo()
    assert build.sealed and build.receipt_sequence is None
    with pytest.raises(RuntimeError, match="may be in the ledger"):
        build.emit(ledger_path=ledger)
    with pytest.raises(RuntimeError, match="may be in the ledger"):
        build.request("a")


# ---- KV-cache simulators -----------------------------------------------------------------------------------------

def _paged() -> PagedKVCacheManager:
    return PagedKVCacheManager(total_vram_bytes_for_kv=16 * 64 * 10, bytes_per_token=64, block_size_tokens=16)


@pytest.mark.parametrize("bad, error", [(-1, ValueError), (-20, ValueError), (-(10 ** 6), ValueError),
                                        (True, TypeError), (16.0, TypeError), ("16", TypeError)])
def test_paged_refuses_a_bad_prompt_length_and_takes_nothing(bad, error):
    mgr = _paged()
    free = list(mgr.free_blocks)
    with pytest.raises(error):
        mgr.allocate_sequence("bad", bad)
    with pytest.raises(error):
        mgr.can_admit_sequence(bad)
    assert mgr.free_blocks == free and mgr.block_tables == {}
    assert all(b.num_tokens_stored == 0 and b.ref_count == 0 for b in mgr.physical_blocks.values())
    assert mgr.allocate_sequence("next", 160) == list(range(10))


def test_paged_zero_prompt_takes_no_block():
    mgr = _paged()
    assert mgr.allocate_sequence("empty", 0) == []
    assert len(mgr.free_blocks) == 10


@pytest.mark.parametrize("bad, error", [(-5, ValueError), (False, TypeError), (1.5, TypeError)])
def test_duo_refuses_a_bad_prompt_length_and_registers_nothing(bad, error):
    mgr = DuoAttentionKVCacheManager(num_layers=2, num_heads=8, head_dim=64)
    with pytest.raises(error):
        mgr.allocate_sequence("bad", bad)
    assert mgr.sequences == {}


def test_duo_failed_appends_count_nothing_and_a_retry_counts_once(monkeypatch):
    mgr = DuoAttentionKVCacheManager(num_layers=2, num_heads=8, head_dim=64)
    mgr.allocate_sequence("s1", 10)
    calls = {"n": 0}
    real = mgr.compute_sequence_vram_bytes

    def flaky(seq_len):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("simulated: sizing failed")
        return real(seq_len)

    monkeypatch.setattr(mgr, "compute_sequence_vram_bytes", flaky)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            mgr.append_token("s1")
    assert mgr.sequences["s1"] == {"prompt_tokens": 10, "generated_tokens": 0}
    assert mgr.append_token("s1")["seq_len"] == 11
    assert mgr.sequences["s1"]["generated_tokens"] == 1
