"""severity: medium
finding: TwoPassBuild.emit() seals the build for good when Argus refuses the receipt before writing a byte (an actor or payload int Argus rejects, a ledger path that is a directory), so nothing is in the ledger, no receipt can ever be written, and items() keeps handing out an unreceipted context

The round-1 fix (src/harpe/two_pass.py:332-347) sets ``_sealed = True`` (:345) after
``_prepared_context_built`` returns and before ``write()`` runs. ``_prepared_context_built`` (:98-117) promises
that "Every refusal (payload, path, actor, a missing ``argus_ledger``) raises here, before any byte is written, so
a caller can tell a receipt that was never attempted from one whose write began", and ``emit``'s docstring
(:336-337) says "A refusal that writes nothing (a bad ``ledger_path`` or ``actor`` ...) leaves the build open, so
the call can be repeated". But harpe's own checks are weaker than Argus's gates, which run inside ``write()``
before the ledger file is even opened (argus canonical.py:59-79 ``validate_identifiers``, :129 ``validate_payload``;
ledger.py:448-451) or fail at ``os.open`` (ledger.py:406):

  * actor ``"harpe agent"``: harpe only checks "non-empty str" (two_pass.py:106-107); Argus raises
    ``IdentifierError`` (no whitespace allowed);
  * ``budget_tokens = 2**53``: harpe only checks ``type(value) is int`` (:84); Argus raises
    ``IntegerRangeError`` (|int| <= 2**53-1);
  * a ``ledger_path`` that names an existing directory: ``_check_ledger_path`` (:89-95) accepts it; Argus's
    ``os.open`` raises ``IsADirectoryError``.

In each case the ledger holds nothing, yet the build is sealed with ``receipt_sequence is None`` for ever:
``request`` and every later ``emit`` raise "may be in the ledger", and ``items()`` still hands out the context. The
observable post-failure state is a context in use that has no receipt and can never get one, which is the "sealed
with no receipt" state the fix was meant to make impossible for refusals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.two_pass import CONTEXT_BUILT_EVENT, TwoPassBuild


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _build(budget_tokens: int = 600) -> TwoPassBuild:
    policy = ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)
    a = ContextItem(id="a", content="a: a one-line description", token_count=20)
    b = ContextItem(id="b", content="b: a one-line description", token_count=15)
    report = ContextBudgetEngine().synthesize([policy, a, b], _budget(600))
    assert [it.id for it in report.included_items] == ["policy", "a", "b"]
    bodies = {
        "a": ContextItem(id="a", content="a: the full body", token_count=200),
        "b": ContextItem(id="b", content="b: the full body", token_count=150),
    }
    return TwoPassBuild(report, bodies=bodies, budget_tokens=budget_tokens)


def _receipts(ledger: Path) -> list:
    if not ledger.is_file():
        return []
    out = []
    for line in ledger.read_bytes().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("event_type") == CONTEXT_BUILT_EVENT:
                out.append(rec)
    return out


def test_control_a_clean_emit_writes_one_receipt_and_seals(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    assert build.emit(ledger_path=ledger) == 1
    assert build.sealed and len(_receipts(ledger)) == 1


def test_an_actor_argus_refuses_writes_nothing_and_must_leave_the_build_open(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    with pytest.raises(ValueError):
        build.emit(ledger_path=ledger, actor="harpe agent")  # harpe's check passes; Argus refuses the space
    assert _receipts(ledger) == [], "precondition: the refused emit wrote nothing"
    assert not build.sealed, "a refusal that wrote nothing sealed the build: no receipt can ever be written for it"
    assert build.emit(ledger_path=ledger, actor="harpe") == 1
    assert len(_receipts(ledger)) == 1


def test_a_ledger_path_that_is_a_directory_writes_nothing_and_must_leave_the_build_open(tmp_path: Path):
    directory = tmp_path / "ledger-dir"
    directory.mkdir()
    build = _build()
    with pytest.raises(OSError):
        build.emit(ledger_path=directory)
    assert list(directory.iterdir()) == [], "precondition: nothing was written"
    assert not build.sealed, "a bad ledger_path that wrote nothing sealed the build for good"
    ledger = tmp_path / "argus.jsonl"
    assert build.emit(ledger_path=ledger) == 1
    assert len(_receipts(ledger)) == 1


def test_a_payload_int_argus_refuses_is_refused_before_the_build_is_sealed(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build(budget_tokens=2 ** 53)  # harpe accepts any int >= 0; Argus refuses |int| > 2**53-1
    with pytest.raises(ValueError):
        build.emit(ledger_path=ledger)
    assert _receipts(ledger) == [], "precondition: the refused emit wrote nothing"
    assert not build.sealed and build.receipt_sequence is None, (
        "a payload refusal that wrote nothing sealed the build, while items() still hands out its context"
    )
    assert build.request("a") is True
