"""severity: low
finding: StateLedger.add_units() is not all-or-nothing: a unit that fails mid-batch leaves the earlier units' claims recorded, and the natural retry records them twice

src/harpe/state_ledger.py:139-141 sums ``self.add`` over the mapping, and each ``add`` appends its claims to
``self.claims`` as soon as they are built (:135-136). When a later unit raises, the earlier units are already
indexed, but the caller gets an exception instead of the count. In the test, the second unit's date is a
``datetime.date``, and pydantic rejects it in ``Claim``. The ledger cannot tell the caller how far it got. Re-running
the batch, the obvious recovery, indexes the first unit a second time. The affirmation then shows up twice in
``timeline``/``latest_state``, and ``conflicts_for_entity`` returns the same pair twice, because ``add`` has no
idempotence check on ``unit_id``.
"""
from __future__ import annotations

import datetime

import pytest

from harpe.state_ledger import StateLedger

AFFIRM = "I have completed the tax filing paperwork."
DENY = "I never completed the tax filing paperwork."


def test_control_a_clean_batch_pairs_once():
    ledger = StateLedger()
    assert ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, "2026-02-01")}) == 2
    assert len(ledger.conflicts_for_entity(["tax filing paperwork"])) == 1


def test_a_failed_batch_records_nothing():
    ledger = StateLedger()
    with pytest.raises(Exception):
        ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, datetime.date(2026, 2, 1))})
    assert ledger.claims == [], (
        f"the failed batch left {len(ledger.claims)} claim(s) from {sorted({c.unit_id for c in ledger.claims})}"
    )


def test_retrying_a_failed_batch_does_not_duplicate_claims():
    ledger = StateLedger()
    with pytest.raises(Exception):
        ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, datetime.date(2026, 2, 1))})
    ledger.add_units({"u1": (AFFIRM, "2026-01-01"), "u2": (DENY, "2026-02-01")})  # the caller's retry
    ids = [c.unit_id for c in ledger.claims]
    assert ids.count("u1") == 1, f"u1 indexed {ids.count('u1')} times after the retry"
    assert len(ledger.conflicts_for_entity(["tax filing paperwork"])) == 1
