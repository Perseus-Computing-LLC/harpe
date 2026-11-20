"""severity: high
finding: TwoPassBuild.emit() seals the build only after append returns, so a crash after the receipt is durable leaves the build unsealed: a retry writes a second receipt and request() changes the context under the first

src/harpe/two_pass.py:9-11 promises at most one ``harpe.context_built`` receipt per build, and says writing it seals
the build. ``emit`` (two_pass.py:305-310) sets ``_receipt_sequence``, which is the only seal ``request`` (:229) and
``emit`` (:306) check, from the return value of ``append_context_built`` (:94-107). If the append raises after the
record is on disk, the build stays unsealed. Argus raises ``LedgerCorruptionError`` in exactly that case: fsync of
the record fails and the rollback ftruncate also fails (argus ledger.py:308-320, "the caller must not take the
original error to mean that nothing was written"). The same happens if close() reports EIO after a successful fsync.
After that, two things go wrong:
  * a retried ``emit`` appends a second ``harpe.context_built`` record for the same build;
  * ``request`` still expands a row, so ``items()`` no longer matches the receipt that is already on disk.

The crash is simulated in-process in two ways:
  (a) ``os.fsync`` of the ledger file raises EIO once, after the write() of the record, and ``os.ftruncate``
      raises EIO once;
  (b) ``os.close`` reports EIO once, after a successful fsync.
"""
from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.two_pass import CONTEXT_BUILT_EVENT, TwoPassBuild


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _build(total: int = 600) -> TwoPassBuild:
    policy = ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)
    a = ContextItem(id="a", content="a: a one-line description", token_count=20)
    b = ContextItem(id="b", content="b: a one-line description", token_count=15)
    report = ContextBudgetEngine().synthesize([policy, a, b], _budget(total))
    assert [it.id for it in report.included_items] == ["policy", "a", "b"]
    bodies = {
        "a": ContextItem(id="a", content="a: the full body", token_count=200),
        "b": ContextItem(id="b", content="b: the full body", token_count=150),
    }
    return TwoPassBuild(report, bodies=bodies, budget_tokens=total)


def _receipts(ledger: Path) -> list:
    """The harpe.context_built records physically present in the file (complete JSON lines)."""
    import json

    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_bytes().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event_type") == CONTEXT_BUILT_EVENT:
            out.append(rec)
    return out


def _crash_after_record_write(monkeypatch):
    """The fsync of the record fails (EIO) after write(), and so does the rollback ftruncate."""
    real_fsync, real_ftruncate = os.fsync, os.ftruncate
    state = {"fsync": 1, "ftruncate": 1}

    def fsync(fd):
        if state["fsync"] and stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_size > 0:
            state["fsync"] -= 1
            raise OSError(errno.EIO, "simulated: fsync of the ledger failed")
        return real_fsync(fd)

    def ftruncate(fd, length):
        if state["ftruncate"]:
            state["ftruncate"] -= 1
            raise OSError(errno.EIO, "simulated: rollback truncate failed")
        return real_ftruncate(fd, length)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "ftruncate", ftruncate)
    return state


def test_control_a_clean_emit_writes_one_receipt_and_seals(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    assert build.request("b") is True
    seq = build.emit(ledger_path=ledger)
    assert seq == 1 and len(_receipts(ledger)) == 1
    with pytest.raises(RuntimeError):
        build.emit(ledger_path=ledger)
    with pytest.raises(RuntimeError):
        build.request("a")
    assert len(_receipts(ledger)) == 1


def test_a_retry_after_a_crashed_emit_never_writes_a_second_receipt(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    state = _crash_after_record_write(monkeypatch)
    with pytest.raises(Exception):
        build.emit(ledger_path=ledger)
    assert state == {"fsync": 0, "ftruncate": 0}, "the simulated crash window was not reached"
    monkeypatch.undo()
    durable = _receipts(ledger)
    assert len(durable) == 1, "precondition: the crashed append left its record in the ledger"

    try:
        build.emit(ledger_path=ledger)  # the caller retries the failed emit
    except Exception:
        pass  # refusing the retry is the correct behaviour
    assert len(_receipts(ledger)) == 1, (
        "one build now has two harpe.context_built receipts in the ledger"
    )


def test_close_eio_after_a_fsynced_receipt_still_seals_the_build(tmp_path: Path, monkeypatch):
    """The record is written and fsynced; only the final close() reports EIO (the fd is closed regardless)."""
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    real_fsync, real_close = os.fsync, os.close
    state = {"armed": False, "fired": False}

    def fsync(fd):
        out = real_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_size > 0:
            state["armed"] = True
        return out

    def close(fd):
        real_close(fd)
        if state["armed"] and not state["fired"]:
            state["fired"] = True
            raise OSError(errno.EIO, "simulated: close() after fsync reported EIO")

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError):
        build.emit(ledger_path=ledger)
    monkeypatch.undo()
    assert state["fired"] and len(_receipts(ledger)) == 1, "precondition: the receipt is durable"

    try:
        build.emit(ledger_path=ledger)
    except Exception:
        pass
    assert len(_receipts(ledger)) == 1, "a fsynced receipt was followed by a second one for the same build"


def test_a_crashed_emit_seals_the_context_its_durable_receipt_describes(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    _crash_after_record_write(monkeypatch)
    with pytest.raises(Exception):
        build.emit(ledger_path=ledger)
    monkeypatch.undo()
    durable = _receipts(ledger)
    assert len(durable) == 1
    receipt = durable[0]["payload"]
    before = [(it.id, it.content, it.token_count) for it in build.items()]

    try:
        build.request("a")  # Pass 2 continues after the failed emit
    except Exception:
        pass
    after = [(it.id, it.content, it.token_count) for it in build.items()]
    assert after == before, "the context changed under a receipt that is already in the ledger"
    assert build.used_tokens == receipt["used_tokens"] == sum(t for _, _, t in after)
