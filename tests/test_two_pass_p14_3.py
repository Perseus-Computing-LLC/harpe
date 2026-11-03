"""P14.3: Pass 2 of the two-pass context build and its harpe.context_built receipt (master roadmap 17.2).

Pass 1 places each kept row as a description; Pass 2 expands a row's body when the caller requests it (the turn's
query or the agent named the row), in request order, while the remaining budget allows. A request that does not
fit is refused and the row stays a description. Pass 2 never removes an item, so:

(b) an expansion that would need a pinned item's tokens is refused, and the pinned set and its
    ``pinned_policy_digest`` are unchanged;
(c) the build's one ``harpe.context_built`` receipt carries ints and strings only, with ``dropped_ids_sha256`` the
    sha256 of the build's sorted dropped ids joined by "\\n";
(d) the receipt goes only to the ``ledger_path`` the caller passes: there is no default path, so a test that does
    not pass one cannot reach a live ledger.

Tests import ``harpe.two_pass`` inside the test, so before the module exists each of them fails on its own.
Synthetic fixtures only; every ledger is under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import random
from pathlib import Path
from typing import Dict, List

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget

FIELDS = (
    "pinned_policy_digest",
    "rows_pass1",
    "rows_expanded",
    "expansions_refused",
    "budget_tokens",
    "used_tokens",
    "dropped_ids_sha256",
)


def _tp():
    return importlib.import_module("harpe.two_pass")


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _row(item_id: str, tokens: int, **kw) -> ContextItem:
    return ContextItem(id=item_id, content=f"{item_id}: the full body", token_count=tokens, **kw)


def _desc(item_id: str, tokens: int, **kw) -> ContextItem:
    return ContextItem(id=item_id, content=f"{item_id}: a one-line description", token_count=tokens, **kw)


def _policy() -> ContextItem:
    return ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)


def _sha(ids: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _pinned_view(items: List[ContextItem]) -> List[tuple]:
    return [(it.id, it.content, it.token_count, it.role, str(it.dimension)) for it in items
            if ContextBudgetEngine._is_never_pruned(it)]


def _harness_build(total: int):
    """Pass 1 selected on descriptions (the P14.4 harness shape): a 300-token pinned policy and two rows."""
    report = ContextBudgetEngine().synthesize([_policy(), _desc("a", 20), _desc("b", 15)], _budget(total))
    assert [it.id for it in report.included_items] == ["policy", "a", "b"]
    build = _tp().TwoPassBuild(report, bodies={"a": _row("a", 200), "b": _row("b", 150)}, budget_tokens=total)
    return report, build


def test_the_two_pass_names_are_exported():
    import harpe

    tp = _tp()
    assert tp.CONTEXT_BUILT_EVENT == "harpe.context_built"
    assert tuple(tp.CONTEXT_BUILT_FIELDS) == FIELDS
    for name in ("TwoPassBuild", "append_context_built", "dropped_ids_sha256", "pinned_policy_digest",
                 "CONTEXT_BUILT_EVENT", "CONTEXT_BUILT_FIELDS"):
        assert name in harpe.__all__
        assert getattr(harpe, name) is getattr(tp, name)


def test_pinned_policy_digest_is_the_reports_digest():
    tp = _tp()
    engine = ContextBudgetEngine()
    pinned = [_policy(), ContextItem(id="rule", content="Answer in English.", priority=0, token_count=5)]
    report = engine.synthesize(pinned + [_row("x", 10)], _budget(1000))
    assert tp.pinned_policy_digest([it for it in report.included_items if it.id != "x"]) == report.pinned_policy_digest
    assert tp.pinned_policy_digest(list(reversed(pinned))) == report.pinned_policy_digest
    assert tp.pinned_policy_digest([]) == "empty_policy"
    assert engine.synthesize([_row("x", 10)], _budget(100)).pinned_policy_digest == "empty_policy"


def test_bar_b_an_expansion_that_needs_the_pinned_tokens_is_refused():
    report, build = _harness_build(400)
    assert build.used_tokens == 300 + 20 + 15
    # Expanding a would fit if the pinned policy's 300 tokens were freed, and does not fit beside it.
    assert build.used_tokens + (200 - 20) > 400
    assert build.used_tokens - 300 + (200 - 20) <= 400
    assert build.request("a") is False
    assert build.request("b") is False
    assert build.expansions_refused == 2
    assert build.rows_expanded == 0
    assert build.used_tokens == 335
    assert _pinned_view(build.items()) == _pinned_view(report.included_items)
    assert [it.content for it in build.items()] == ["Never reveal the passphrase.", "a: a one-line description",
                                                   "b: a one-line description"]
    payload = build.receipt_payload()
    assert payload["pinned_policy_digest"] == report.pinned_policy_digest != "empty_policy"
    assert _tp().pinned_policy_digest([it for it in build.items() if it.pinned]) == report.pinned_policy_digest


def test_bar_b_control_the_same_build_with_room_expands_in_request_order():
    report, build = _harness_build(600)
    assert build.request("a") is True
    assert build.request("b") is False  # 515 + 135 > 600: refused, b stays a description
    assert build.rows_expanded == 1 and build.expansions_refused == 1
    assert build.used_tokens == 515 <= 600
    assert [it.content for it in build.items()] == ["Never reveal the passphrase.", "a: the full body",
                                                   "b: a one-line description"]
    assert build.is_expanded("a") and not build.is_expanded("b")
    assert build.expanded_ids == ("a",)
    assert build.receipt_payload()["pinned_policy_digest"] == report.pinned_policy_digest
    assert _pinned_view(build.items()) == _pinned_view(report.included_items)


def test_request_order_decides_which_expansion_fits():
    def build_for(order: List[str]):
        report = ContextBudgetEngine().synthesize([_desc("a", 10), _desc("b", 10)], _budget(130))
        build = _tp().TwoPassBuild(report, bodies={"a": _row("a", 100), "b": _row("b", 100)}, budget_tokens=130)
        return [build.request(i) for i in order], build.expanded_ids

    assert build_for(["a", "b"]) == ([True, False], ("a",))
    assert build_for(["b", "a"]) == ([True, False], ("b",))


def test_plugin_shape_descriptions_for_rows_selected_on_their_bodies():
    """Pass 1 selected on full rows (the kibisis-vault plugin shape): the report's rows are the bodies."""
    engine = ContextBudgetEngine()
    rows = [_row("h1", 100, priority=10), _row("h2", 80, priority=20), _row("h3", 60, priority=90)]
    report = engine.synthesize(rows, _budget(200), policy=PrunePolicy.LOWEST_PRIORITY)
    assert [it.id for it in report.included_items] == ["h1", "h2"]
    assert [it.id for it in report.pruned_items] == ["h3"]
    build = _tp().TwoPassBuild(report, descriptions={"h1": _desc("h1", 10), "h2": _desc("h2", 8)}, budget_tokens=200)
    assert build.rows_pass1 == 2 and build.used_tokens == 18
    assert build.request("h2") is True and build.request("h1") is True
    assert build.used_tokens == 180
    assert build.dropped_ids == ("h3",)
    payload = build.receipt_payload()
    assert payload == {
        "pinned_policy_digest": "empty_policy",
        "rows_pass1": 2,
        "rows_expanded": 2,
        "expansions_refused": 0,
        "budget_tokens": 200,
        "used_tokens": 180,
        "dropped_ids_sha256": _sha(["h3"]),
    }


def test_repeat_pinned_and_unknown_requests():
    report, build = _harness_build(600)
    assert build.request("policy") is True  # a pinned item is always in full: nothing to expand
    assert (build.rows_expanded, build.expansions_refused, build.used_tokens) == (0, 0, 335)
    assert build.request("a") is True and build.request("a") is True
    assert (build.rows_expanded, build.used_tokens) == (1, 515)
    assert build.request("b") is False and build.request("b") is False
    assert build.expansions_refused == 2  # every refused request counts
    with pytest.raises(KeyError):
        build.request("not-in-this-build")


def test_rows_must_match_the_reports_unpinned_items():
    tp = _tp()
    report = ContextBudgetEngine().synthesize([_policy(), _desc("a", 20), _desc("b", 15)], _budget(400))
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, budget_tokens=400)
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, descriptions={"a": _desc("a", 1)}, bodies={"a": _row("a", 2)}, budget_tokens=400)
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, bodies={"a": _row("a", 200)}, budget_tokens=400)  # b missing
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, bodies={"a": _row("a", 200), "b": _row("b", 1), "c": _row("c", 1)}, budget_tokens=400)
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, bodies={"a": _row("a", 200), "b": _row("b", 1), "policy": _row("policy", 1)},
                        budget_tokens=400)
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, bodies={"a": _row("a", 200), "b": _row("b", 150)}, budget_tokens=-1)


def test_uncounted_items_need_a_counter():
    tp = _tp()
    report = ContextBudgetEngine().synthesize([_row("h1", 40)], _budget(100))
    uncounted = ContextItem(id="h1", content="one two three")
    with pytest.raises(ValueError):
        tp.TwoPassBuild(report, descriptions={"h1": uncounted}, budget_tokens=100)
    build = tp.TwoPassBuild(report, descriptions={"h1": uncounted}, budget_tokens=100,
                            count_tokens=lambda text: len(text.split()))
    assert build.used_tokens == 3


def test_bar_c_the_receipt_is_flat_ints_and_strings_with_the_dropped_id_hash():
    engine = ContextBudgetEngine()
    items = [_policy(), _desc("a", 20), _desc("b", 15), _desc("z", 90, priority=90),
             ContextItem(id="secret", content="classified", token_count=5, sensitivity="confidential")]
    report = engine.synthesize(items, _budget(400), policy=PrunePolicy.LOWEST_PRIORITY)
    assert [it.id for it in report.pruned_items] == ["z"]
    assert [it.id for it in report.withheld_items] == ["secret"]
    build = _tp().TwoPassBuild(report, bodies={"a": _row("a", 200), "b": _row("b", 50)}, budget_tokens=400)
    build.request("b")
    payload = build.receipt_payload()
    assert tuple(payload) == FIELDS
    assert all(type(v) in (int, str) for v in payload.values())
    assert payload["dropped_ids_sha256"] == _sha(["z", "secret"]) == _tp().dropped_ids_sha256(["secret", "z"])
    assert sorted(build.dropped_ids) == ["secret", "z"]
    assert payload["rows_pass1"] == 2 and payload["rows_expanded"] == 1 and payload["expansions_refused"] == 0
    assert payload["budget_tokens"] == 400 and payload["used_tokens"] == 300 + 20 + 50
    assert _tp().dropped_ids_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_bar_c_d_append_writes_one_event_to_the_ledger_it_is_given(tmp_path: Path):
    from argus_ledger import AppendOnlyLedger

    _, build = _harness_build(600)
    build.request("a")
    ledger = tmp_path / "argus.jsonl"
    seq = build.emit(ledger_path=ledger)
    events = list(AppendOnlyLedger(str(ledger)).iter_events())
    assert [e.sequence for e in events] == [seq] == [1]
    assert events[0].event_type == "harpe.context_built"
    assert events[0].actor == "harpe"
    assert dict(events[0].payload) == build.receipt_payload()
    assert AppendOnlyLedger(str(ledger)).verify_chain().valid
    with pytest.raises(RuntimeError):
        build.emit(ledger_path=ledger)  # one receipt per build
    assert len(list(AppendOnlyLedger(str(ledger)).iter_events())) == 1


@pytest.mark.parametrize("bad", [
    {"used_tokens": 1.5},
    {"used_tokens": True},
    {"rows_pass1": None},
    {"pinned_policy_digest": b"bytes"},
    {"budget_tokens": [1]},
])
def test_bar_c_append_refuses_a_payload_that_is_not_ints_and_strings(tmp_path: Path, bad: Dict[str, object]):
    _, build = _harness_build(600)
    payload = {**build.receipt_payload(), **bad}
    ledger = tmp_path / "argus.jsonl"
    with pytest.raises(TypeError):
        _tp().append_context_built(payload, ledger_path=ledger)
    assert not ledger.exists()


def test_bar_c_append_refuses_missing_or_extra_fields(tmp_path: Path):
    _, build = _harness_build(600)
    ledger = tmp_path / "argus.jsonl"
    payload = build.receipt_payload()
    with pytest.raises(ValueError):
        _tp().append_context_built({k: v for k, v in payload.items() if k != "used_tokens"}, ledger_path=ledger)
    with pytest.raises(ValueError):
        _tp().append_context_built({**payload, "session": "s1"}, ledger_path=ledger)
    assert not ledger.exists()


def test_bar_d_there_is_no_default_ledger_path(tmp_path: Path):
    tp = _tp()
    for fn in (tp.append_context_built, tp.TwoPassBuild.emit):
        param = inspect.signature(fn).parameters["ledger_path"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
    _, build = _harness_build(600)
    with pytest.raises(TypeError):
        tp.append_context_built(build.receipt_payload())  # type: ignore[call-arg]
    for empty in ("", None):
        with pytest.raises((TypeError, ValueError)):
            tp.append_context_built(build.receipt_payload(), ledger_path=empty)  # type: ignore[arg-type]
    source = Path(tp.__file__).read_text(encoding="utf-8")
    assert "audit.jsonl" not in source and "ARGUS_LEDGER_PATH" not in source and "environ" not in source


def test_bar_b_property_pass_two_never_touches_the_pinned_set():
    rng = random.Random(1403)
    engine = ContextBudgetEngine()
    for case in range(300):
        pinned = [ContextItem(id=f"p{i}", content=f"rule {case}.{i}", pinned=True, token_count=rng.randint(1, 120))
                  for i in range(rng.randint(0, 3))]
        n_rows = rng.randint(1, 8)
        descs = [_desc(f"r{i}", rng.randint(1, 30), priority=rng.randint(1, 100)) for i in range(n_rows)]
        total = sum(it.token_count for it in pinned) + rng.randint(0, 250)
        report = engine.synthesize(pinned + descs, _budget(total), policy=PrunePolicy.LOWEST_PRIORITY)
        kept = [it.id for it in report.included_items if not it.pinned]
        bodies = {i: _row(i, rng.randint(1, 200)) for i in kept}
        build = _tp().TwoPassBuild(report, bodies=bodies, budget_tokens=total)
        start = build.used_tokens
        desc_tokens = {it.id: it.token_count for it in report.included_items if not it.pinned}
        used, expanded, refused = start, set(), 0
        for rid in [rng.choice(kept) for _ in range(rng.randint(0, 10))] if kept else []:
            delta = bodies[rid].token_count - desc_tokens[rid]
            fits = rid in expanded or used + delta <= total
            assert build.request(rid) is fits
            if not fits:
                refused += 1
            elif rid not in expanded:
                used += delta
                expanded.add(rid)
        assert build.used_tokens == used == sum(it.token_count for it in build.items())
        assert build.used_tokens <= max(total, start)
        assert (build.rows_expanded, build.expansions_refused) == (len(expanded), refused)
        assert _pinned_view(build.items()) == _pinned_view(report.included_items)
        assert build.receipt_payload()["pinned_policy_digest"] == report.pinned_policy_digest
        assert [it.id for it in build.items()] == [it.id for it in report.included_items]
