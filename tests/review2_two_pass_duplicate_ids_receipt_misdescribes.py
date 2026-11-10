"""severity: high
finding: TwoPassBuild keys a report's rows by id without refusing a repeated id, so items() hands out a context its receipt does not describe (rows_pass1 and used_tokens count the repeated row once)

two_pass.py:11-12: "Writing the receipt seals the build ... so the context the build hands out is always the one
its receipt describes." TwoPassBuild.__init__ (src/harpe/two_pass.py:153-162) walks ``report.included_items`` and
stores each row in a dict keyed by id (``self._desc[item.id]`` / ``self._tier0[item.id]``) while appending every
occurrence to ``self._order``. A repeated id therefore overwrites the first row in the dict but stays twice in the
order: ``items()`` (two_pass.py:264-274) hands out the row twice, while ``rows_pass1`` (``len(self._desc)``, two_pass.py:245) and
``used_tokens`` (the sum over the dict's values, two_pass.py:191-193) count it once. The receipt that ``emit()``
writes from ``receipt_payload()`` (two_pass.py:277-288) then names a smaller context than the one the build hands
to the model. The first occurrence's own content is also lost: both copies handed out are the last one.

Nothing refuses the repeat: the id checks (two_pass.py:164-184) compare id sets only. The report used here is the
one ContextBudgetEngine returns on this head when a pre_prune hook re-adds an item that a replayed turn already
carries (review2_hook_duplicate_ids_not_refused.py); it is built directly so this test depends only on
TwoPassBuild. A report is a public pydantic model, and a post_prune hook or a stored report can carry the same
shape.

The test accepts either fix: the build refuses a report whose ids repeat (``ValueError``/``TypeError``), or its
receipt describes exactly the items it hands out.
"""
from __future__ import annotations

from typing import Dict, List

from harpe import ContextItem, SynthesisReport, TwoPassBuild


def _row(item_id: str, content: str, tokens: int) -> ContextItem:
    return ContextItem(id=item_id, content=content, token_count=tokens)


def _report(rows: List[ContextItem]) -> SynthesisReport:
    tokens = sum(it.token_count for it in rows)
    return SynthesisReport(included_items=rows, consumed_tokens=tokens, budget_remaining=1000 - tokens,
                           effective_budget=1000)


def _bodies(ids) -> Dict[str, ContextItem]:
    return {i: _row(i, f"the full text of {i}", 30) for i in ids}


def _receipt_describes_items(build: TwoPassBuild) -> None:
    payload = build.receipt_payload()
    handed_out = build.items()
    assert payload["used_tokens"] == sum(it.token_count for it in handed_out), (
        f"receipt used_tokens={payload['used_tokens']} but items() hands out "
        f"{sum(it.token_count for it in handed_out)} tokens: {[it.id for it in handed_out]}"
    )
    assert payload["rows_pass1"] == len(handed_out), (
        f"receipt rows_pass1={payload['rows_pass1']} but items() hands out {len(handed_out)} rows: "
        f"{[it.id for it in handed_out]}"
    )


def test_control_a_report_with_unique_ids_has_a_receipt_that_describes_its_items():
    rows = [_row("q1", "What failed in last night's deploy?", 9), _row("reminder", "Cite your sources.", 5),
            _row("q2", "And who rolled it back?", 7)]
    build = TwoPassBuild(_report(rows), bodies=_bodies(["q1", "reminder", "q2"]), budget_tokens=1000)
    _receipt_describes_items(build)
    assert build.request("q2") is True
    _receipt_describes_items(build)


def test_a_repeated_row_id_is_refused_or_described_by_the_receipt():
    rows = [_row("q1", "What failed in last night's deploy?", 9), _row("reminder", "Cite your sources.", 5),
            _row("q2", "And who rolled it back?", 7), _row("reminder", "Cite your sources (turn 2).", 6)]
    try:
        build = TwoPassBuild(_report(rows), bodies=_bodies(["q1", "reminder", "q2"]), budget_tokens=1000)
    except (ValueError, TypeError):
        return  # refusing the ambiguous report is a correct fix
    _receipt_describes_items(build)
