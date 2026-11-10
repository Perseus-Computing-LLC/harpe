"""Regression tests for the third review of PR #16 (2026-11-10, review_notes/R3_PR16.md).

* The copies ``synthesize`` gives its hooks and the copies ``TwoPassBuild`` keeps and hands out own the dicts, lists,
  tuples and sets inside ``metadata`` and share every other object there: a lock, a client or an open handle no
  longer raises, and nothing written into a copy's ``metadata``, at any depth, reaches the item it was made from.
* An item whose content a ``pre_prune`` hook rewrote is costed as rewritten, so the budget holds for what the report
  includes and ``consumed_tokens`` is the count of that text; a count a hook or the caller set on purpose is kept.
* Text holding a lone surrogate hashes instead of raising (DTOC ref and size, ``pinned_policy_digest``,
  ``dropped_ids_sha256``), and every digest and ref of valid text is the strict UTF-8 one it always was.
"""
from __future__ import annotations

import collections
import hashlib
import json
import threading
from typing import Any, Dict, List, Sequence

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, SynthesisReport, TokenBudget, TwoPassBuild
from harpe.dtoc import DynamicToolCompressor
from harpe.engine import pinned_policy_digest
from harpe.models import _detached_copy
from harpe.two_pass import dropped_ids_sha256


class _Handle:
    """Stands for a client or an open handle: it cannot be copied, deeply or otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_Handle":
        raise TypeError("a live handle cannot be copied")

    def __copy__(self) -> "_Handle":
        raise TypeError("a live handle cannot be copied")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("a live handle cannot be pickled")


class _Noop:
    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        pass

    def post_prune(self, report: SynthesisReport) -> None:
        pass


# ---- the copy itself ---------------------------------------------------------------------------------------------

def test_a_detached_copy_owns_every_nested_container_and_shares_the_other_objects():
    lock, handle, blob = threading.Lock(), _Handle(), bytearray(b"raw")
    point = collections.namedtuple("point", "where tags")
    item = ContextItem(id="a", content="alpha", token_count=3, metadata={
        "lock": lock, "handle": handle, "blob": blob, "frozen": frozenset({1, 2}),
        "tags": ["x"], "nested": {"deep": {"deeper": [{"k": [1]}]}}, "seen": {1, 2},
        "pair": ("p", ["q"]), "flat": (1, "two", 3.0), "point": point("here", ["t"]),
        "embedding": [0.25] * 16,
    })
    snapshot = repr(item.metadata)
    clone = _detached_copy(item)
    old, new = item.metadata, clone.metadata
    assert clone == item and clone is not item and new is not old
    # Objects that are not containers are the same objects: never cloned, so never a TypeError.
    assert new["lock"] is lock and new["handle"] is handle and new["blob"] is blob and new["frozen"] is old["frozen"]
    assert new["flat"] is old["flat"]  # nothing inside it can change
    # Every dict, list, tuple and set that can be written into (or holds something that can) is the copy's own.
    assert new["tags"] is not old["tags"] and new["seen"] is not old["seen"] and new["embedding"] is not old["embedding"]
    assert new["nested"]["deep"]["deeper"][0]["k"] is not old["nested"]["deep"]["deeper"][0]["k"]
    assert new["pair"] is not old["pair"] and new["pair"][1] is not old["pair"][1]
    assert type(new["point"]) is point and new["point"].tags is not old["point"].tags
    # Writing into the copy at any depth leaves the original as it was.
    new["added"] = True
    new["tags"].append("y")
    new["seen"].add(3)
    new["nested"]["deep"]["deeper"][0]["k"].append(2)
    new["pair"][1].append("r")
    new["point"].tags.append("u")
    new["embedding"][0] = 9.0
    assert repr(item.metadata) == snapshot


def test_a_detached_copy_keeps_cycles_shared_references_and_container_types():
    ring: Dict[str, Any] = {"members": [1]}
    ring["self"] = ring
    ring["members"].append(ring["members"])
    held: List[Any] = []
    knot = (held, "end")
    held.append(knot)  # a tuple that holds itself through a list
    twice = ["shared", ["inner"]]
    ordered = collections.OrderedDict(first=["a"], second={"b": ["c"]})
    grouped: Any = collections.defaultdict(list)
    grouped["g"].append(["h"])
    item = ContextItem(id="a", content="alpha", metadata={
        "ring": ring, "knot": knot, "one": twice, "two": twice, "ordered": ordered, "grouped": grouped,
    })
    new = _detached_copy(item).metadata
    assert new["ring"] is not ring and new["ring"]["self"] is new["ring"]
    assert new["ring"]["members"] is not ring["members"] and new["ring"]["members"][1] is new["ring"]["members"]
    assert new["knot"] is not knot and new["knot"][0] is not held and new["knot"][0][0] is new["knot"]
    assert new["one"] is new["two"] and new["one"] is not twice and new["one"][1] is not twice[1]
    assert type(new["ordered"]) is collections.OrderedDict and list(new["ordered"]) == ["first", "second"]
    assert new["ordered"]["first"] is not ordered["first"] and new["ordered"]["second"]["b"] is not ordered["second"]["b"]
    assert type(new["grouped"]) is collections.defaultdict and new["grouped"].default_factory is list
    assert new["grouped"]["g"][0] is not grouped["g"][0]
    new["grouped"]["other"].append(1)  # the copy still makes its own defaults, and only in the copy
    assert "other" not in grouped


def test_a_container_that_cannot_be_copied_or_written_is_shared_not_an_error():
    class _Sealed(dict):
        def __setitem__(self, key: Any, value: Any) -> None:
            raise TypeError("sealed")

    class _Stubborn(list):
        def __copy__(self) -> "_Stubborn":
            raise RuntimeError("no copies")

    sealed = _Sealed()
    dict.__setitem__(sealed, "k", ["v"])
    stubborn = _Stubborn([["w"]])
    item = ContextItem(id="a", content="alpha", metadata={"sealed": sealed, "stubborn": stubborn, "tags": ["x"]})
    new = _detached_copy(item).metadata
    assert new["sealed"] is sealed and new["stubborn"] is stubborn
    assert new["tags"] is not item.metadata["tags"]


def test_a_detached_copy_applies_the_update_and_keeps_which_fields_were_set():
    bare = ContextItem(id="a", content="alpha")
    clone = _detached_copy(bare)
    assert clone.model_fields_set == bare.model_fields_set
    assert clone.model_dump(exclude_unset=True) == bare.model_dump(exclude_unset=True)
    assert clone.metadata == {} and clone.metadata is not bare.metadata
    counted = _detached_copy(bare, {"token_count": 7})
    assert counted.token_count == 7 and bare.token_count == 0
    replaced = _detached_copy(bare, {"metadata": {"tags": ["x"]}})
    assert replaced.metadata == {"tags": ["x"]} and bare.metadata == {}


def test_a_detached_copy_of_a_subclass_owns_the_containers_in_its_own_fields_and_extras():
    class _Scored(ContextItem):
        model_config = {"extra": "allow"}
        scores: List[Any] = []

    item = _Scored(id="a", content="alpha", scores=[[0.5]], notes={"k": ["v"]}, metadata={"tags": ["x"]})
    clone = _detached_copy(item)
    assert type(clone) is _Scored and clone == item
    clone.scores[0].append(0.9)
    clone.notes["k"].append("w")  # type: ignore[attr-defined]
    clone.metadata["tags"].append("y")
    assert item.scores == [[0.5]] and item.notes == {"k": ["v"]} and item.metadata == {"tags": ["x"]}  # type: ignore[attr-defined]


# ---- hooks: copies that own their metadata, without cloning what metadata holds -------------------------------------

def _live_items() -> List[ContextItem]:
    return [
        ContextItem(id="doc", content="alpha beta gamma", token_count=3,
                    metadata={"client": _Handle(), "lock": threading.Lock(), "tags": ["a"], "nested": {"by": "caller"}}),
        ContextItem(id="uncounted", content="delta epsilon", metadata={"client": _Handle(), "tags": []}),
        ContextItem(id="secret", content="zeta", token_count=1, sensitivity="confidential",
                    metadata={"client": _Handle(), "tags": ["hr"]}),
    ]


@pytest.mark.parametrize("destination", ["remote", "local"])
@pytest.mark.parametrize("policy", list(PrunePolicy))
def test_live_objects_in_metadata_are_accepted_with_a_hook_as_without_one(destination, policy):
    items = _live_items()
    budget = TokenBudget(total_budget=1000)
    plain = ContextBudgetEngine().synthesize(items, budget, policy=policy, destination=destination)
    hooked = ContextBudgetEngine(hooks=[_Noop()]).synthesize(items, budget, policy=policy, destination=destination)
    assert [it.id for it in hooked.included_items] == [it.id for it in plain.included_items]
    assert [it.id for it in hooked.withheld_items] == [it.id for it in plain.withheld_items]
    assert hooked.consumed_tokens == plain.consumed_tokens
    # The live objects are handed on as they are; the containers around them are copies.
    by_id = {it.id: it for it in items}
    for it in hooked.included_items + hooked.withheld_items:
        assert it is not by_id[it.id] and it.metadata is not by_id[it.id].metadata
        assert it.metadata["client"] is by_id[it.id].metadata["client"]
        assert it.metadata["tags"] is not by_id[it.id].metadata["tags"]


@pytest.mark.parametrize("destination", ["remote", "local"])
def test_a_hook_writing_next_to_a_live_object_does_not_reach_the_callers_items(destination):
    class _Writes:
        def pre_prune(self, items: Sequence[ContextItem]) -> None:
            for it in items:
                it.metadata["relevance"] = 0.0
                it.metadata["tags"].append("pre")
                it.metadata.setdefault("nested", {})["by"] = "pre"

        def post_prune(self, report: SynthesisReport) -> None:
            for it in report.included_items + report.pruned_items + report.withheld_items:
                it.metadata["tags"].append("post")
                it.metadata.setdefault("nested", {})["by"] = "post"

    items = _live_items()
    before = [repr(sorted((k, v) for k, v in it.metadata.items() if k != "client" and k != "lock")) for it in items]
    report = ContextBudgetEngine(hooks=[_Writes()]).synthesize(
        items, TokenBudget(total_budget=1000), destination=destination)
    assert [repr(sorted((k, v) for k, v in it.metadata.items() if k != "client" and k != "lock")) for it in items] == before
    seen = report.included_items + report.pruned_items + report.withheld_items
    assert {it.id for it in seen} == {"doc", "uncounted", "secret"}
    assert all(it.metadata["tags"][-1] == "post" and it.metadata["nested"] == {"by": "post"} for it in seen)


def test_what_a_hook_keeps_from_pre_prune_is_not_what_the_engine_selects_and_reports():
    """After the hooks the engine takes its own copies again: an item a hook held on to and edits later is no
    longer the item in the report."""

    class _Keeps:
        def __init__(self) -> None:
            self.kept: List[ContextItem] = []

        def pre_prune(self, items: Sequence[ContextItem]) -> None:
            self.kept = list(items)

        def post_prune(self, report: SynthesisReport) -> None:
            for it in self.kept:
                it.content = "edited behind the report"
                it.metadata["tags"].append("late")

    hook = _Keeps()
    items = [ContextItem(id="doc", content="alpha beta gamma", token_count=3, metadata={"tags": ["a"]})]
    report = ContextBudgetEngine(hooks=[hook]).synthesize(items, TokenBudget(total_budget=1000), destination="local")
    assert [it.content for it in report.included_items] == ["alpha beta gamma"]
    assert report.included_items[0].metadata == {"tags": ["a"]}
    assert items[0].content == "alpha beta gamma" and items[0].metadata == {"tags": ["a"]}


def test_an_item_a_hook_adds_enters_the_report_as_a_copy_of_the_hooks_object():
    reminder = ContextItem(id="reminder", content="answer in English", token_count=4, metadata={"tags": ["hook"]})

    class _Adds(_Noop):
        def pre_prune(self, items: Sequence[ContextItem]) -> None:
            items.append(reminder)  # type: ignore[attr-defined]

    report = ContextBudgetEngine(hooks=[_Adds()]).synthesize(
        [ContextItem(id="doc", content="alpha", token_count=1)], TokenBudget(total_budget=1000), destination="local")
    added = [it for it in report.included_items if it.id == "reminder"]
    assert len(added) == 1 and added[0] is not reminder
    added[0].metadata["tags"].append("edited by the caller")
    assert reminder.metadata == {"tags": ["hook"]}


# ---- two-pass builds: the same copies ------------------------------------------------------------------------------

def _live_report() -> SynthesisReport:
    items = [
        ContextItem(id="rule", content="always answer in English", token_count=6, pinned=True,
                    metadata={"client": _Handle(), "tags": ["policy"]}),
        ContextItem(id="row", content="short description", token_count=5,
                    metadata={"client": _Handle(), "lock": threading.Lock(), "tags": ["a"]}),
        ContextItem(id="other", content="another description", token_count=5, metadata={"tags": ["b"]}),
    ]
    return ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=1000))


@pytest.mark.parametrize("rows", ["bodies", "descriptions"])
@pytest.mark.parametrize("counted", [True, False])
def test_a_build_accepts_live_objects_in_its_items_rows_and_tier0(rows, counted):
    report = _live_report()
    given = {
        "row": ContextItem(id="row", content="the full body " * 10, token_count=40 if counted else 0,
                           metadata={"client": _Handle(), "tags": ["body"]}),
        "other": ContextItem(id="other", content="more text " * 10, token_count=30 if counted else 0),
    }
    build = TwoPassBuild(report, budget_tokens=500, count_tokens=lambda text: len(text.split()), **{rows: given})
    assert build.request("row") is True
    out = build.items()
    assert [it.id for it in out] == ["rule", "row", "other"]
    # The live objects are the caller's; the items and the containers around them are the build's.
    assert out[0].metadata["client"] is report.included_items[0].metadata["client"]
    assert out[0] is not report.included_items[0] and out[0].metadata["tags"] is not report.included_items[0].metadata["tags"]


def test_a_build_and_the_items_it_hands_out_share_no_container_with_the_report_or_each_other():
    report = _live_report()
    bodies = {
        "row": ContextItem(id="row", content="the full body " * 10, token_count=40, metadata={"tags": ["body"]}),
        "other": ContextItem(id="other", content="more text " * 10, token_count=30, metadata={"nested": {"k": ["v"]}}),
    }
    build = TwoPassBuild(report, bodies=bodies, budget_tokens=500)
    assert build.request("row") is True and build.request("other") is True
    # Edits to what the caller gave, after the build was made, do not reach it ...
    report.included_items[0].metadata["tags"].append("edited")
    bodies["row"].metadata["tags"].append("edited")
    bodies["other"].metadata["nested"]["k"].append("edited")
    bodies["row"].content = "edited after the build"
    first = build.items()
    assert [it.metadata.get("tags") for it in first] == [["policy"], ["body"], None]
    assert first[2].metadata == {"nested": {"k": ["v"]}}
    assert first[1].content == "the full body " * 10
    # ... and neither do edits to the items it handed out.
    first[0].metadata["tags"].append("edited")
    first[2].metadata["nested"]["k"].append("edited")
    first[1].content = "edited"
    second = build.items()
    assert [it.metadata.get("tags") for it in second] == [["policy"], ["body"], None]
    assert second[2].metadata == {"nested": {"k": ["v"]}} and second[1].content == "the full body " * 10
    assert build.used_tokens == 6 + 40 + 30


# ---- what an item costs after a hook rewrote it ---------------------------------------------------------------------

class _Rewrite(_Noop):
    """Applies ``edit(item)`` to every item in place; ``edit`` may return a replacement object."""

    def __init__(self, edit: Any) -> None:
        self.edit = edit

    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        for index, it in enumerate(items):
            replacement = self.edit(it)
            if replacement is not None:
                items[index] = replacement  # type: ignore[index]


def _short_items(count: int = 5) -> List[ContextItem]:
    return [ContextItem(id=f"i{n}", content=f"see document {n}", priority=50, created_at=1000.0 + n) for n in range(count)]


def _actual(engine: ContextBudgetEngine, items: Sequence[ContextItem]) -> int:
    return sum(engine.token_counter.count_tokens(it.content) for it in items)


def _expand(it: ContextItem) -> None:
    it.content = it.content + " resolved detail" * 150


def _expand_by_replacing(it: ContextItem) -> ContextItem:
    return it.model_copy(update={"content": it.content + " resolved detail" * 150})


def _expand_and_rename(it: ContextItem) -> None:
    it.id = "resolved-" + it.id
    it.content = it.content + " resolved detail" * 150


@pytest.mark.parametrize("policy", list(PrunePolicy))
@pytest.mark.parametrize("edit", [_expand, _expand_by_replacing, _expand_and_rename])
@pytest.mark.parametrize("counted", [False, True])
def test_items_a_hook_expanded_are_selected_and_reported_at_their_new_cost(policy, edit, counted):
    engine = ContextBudgetEngine(hooks=[_Rewrite(edit)])
    items = _short_items()
    if counted:
        items = [it.model_copy(update={"token_count": engine.token_counter.count_tokens(it.content)}) for it in items]
    budget = TokenBudget(total_budget=1000)
    report = engine.synthesize(items, budget, policy=policy, destination="local")
    assert len(report.included_items) + len(report.pruned_items) == 5
    assert report.consumed_tokens == _actual(engine, report.included_items) <= budget.available_context
    assert report.pruned_tokens == _actual(engine, report.pruned_items)
    each = engine.token_counter.count_tokens(report.pruned_items[0].content)
    assert budget.available_context // each == 1  # the fixture: one rewritten item fits, not five
    assert len(report.included_items) == 1 and not report.over_budget
    assert all(it.token_count < 10 for it in items) and all(len(it.content) < 20 for it in items)


def test_items_a_hook_shortened_cost_less_so_more_of_them_fit():
    def summarize(it: ContextItem) -> None:
        it.content = it.content.split(".")[0]

    long_items = [ContextItem(id=f"i{n}", content=f"Finding {n}." + " supporting detail" * 200) for n in range(5)]
    budget = TokenBudget(total_budget=1000)
    plain = ContextBudgetEngine().synthesize(long_items, budget, destination="local")
    engine = ContextBudgetEngine(hooks=[_Rewrite(summarize)])
    report = engine.synthesize(long_items, budget, destination="local")
    assert len(plain.included_items) < 5 and len(report.included_items) == 5
    assert report.consumed_tokens == _actual(engine, report.included_items) < 50


def test_a_rewritten_pinned_item_is_reported_over_budget_at_its_new_cost():
    engine = ContextBudgetEngine(hooks=[_Rewrite(_expand)])
    items = [ContextItem(id="rule", content="see the policy", pinned=True), ContextItem(id="doc", content="alpha")]
    report = engine.synthesize(items, TokenBudget(total_budget=500), destination="local")
    assert [it.id for it in report.included_items] == ["rule"]
    assert report.consumed_tokens == _actual(engine, report.included_items) > 500
    assert report.over_budget and report.overflow_tokens == report.consumed_tokens - 500


def test_a_count_a_hook_sets_with_its_rewrite_is_kept():
    def rewrite_and_count(it: ContextItem) -> None:
        it.content = it.content + " resolved detail" * 150
        it.token_count = 123  # the hook's own tokenizer

    engine = ContextBudgetEngine(hooks=[_Rewrite(rewrite_and_count)])
    report = engine.synthesize(_short_items(), TokenBudget(total_budget=1000), destination="local")
    assert [it.token_count for it in report.included_items] == [123] * 5
    assert report.consumed_tokens == 615


def test_a_hook_that_clears_the_count_of_what_it_rewrote_gets_it_counted():
    def rewrite_and_clear(it: ContextItem) -> None:
        it.content = "one two three"
        it.token_count = 0

    engine = ContextBudgetEngine(hooks=[_Rewrite(rewrite_and_clear)])
    report = engine.synthesize(_short_items(), TokenBudget(total_budget=1000), destination="local")
    each = engine.token_counter.count_tokens("one two three")
    assert [it.token_count for it in report.included_items] == [each] * 5 and report.consumed_tokens == 5 * each


def test_counts_are_left_alone_where_no_hook_changed_the_content():
    """A count the caller gives is the caller's (another tokenizer, a reserve): a hook that leaves the text alone,
    or only re-costs it, must not have it replaced by the engine's own count."""
    def recost(it: ContextItem) -> None:
        if it.id == "i1":
            it.token_count = 77

    items = [it.model_copy(update={"token_count": 40}) for it in _short_items(3)]
    for hook in (_Noop(), _Rewrite(recost), _Rewrite(lambda it: it.model_copy())):
        report = ContextBudgetEngine(hooks=[hook]).synthesize(items, TokenBudget(total_budget=1000), destination="local")
        expected = [40, 77, 40] if isinstance(hook, _Rewrite) and hook.edit is recost else [40, 40, 40]
        assert [it.token_count for it in report.included_items] == expected
        assert report.consumed_tokens == sum(expected)


def test_an_item_a_hook_adds_keeps_the_count_it_came_with_or_is_counted():
    class _Adds(_Noop):
        def pre_prune(self, items: Sequence[ContextItem]) -> None:
            items.append(ContextItem(id="costed", content="one two three", token_count=50))  # type: ignore[attr-defined]
            items.append(ContextItem(id="bare", content="one two three"))  # type: ignore[attr-defined]

    engine = ContextBudgetEngine(hooks=[_Adds()])
    report = engine.synthesize(_short_items(1), TokenBudget(total_budget=1000), destination="local")
    count = engine.token_counter.count_tokens
    assert {it.id: it.token_count for it in report.included_items} == {
        "i0": count("see document 0"), "costed": 50, "bare": count("one two three")}


def test_the_recount_uses_the_engines_token_counter():
    class _Chars:
        def count_tokens(self, text: str) -> int:
            return len(text)

    engine = ContextBudgetEngine(token_counter=_Chars(), hooks=[_Rewrite(_expand)])
    report = engine.synthesize(_short_items(), TokenBudget(total_budget=8000), destination="local")
    assert report.consumed_tokens == sum(len(it.content) for it in report.included_items) <= 8000
    assert len(report.included_items) == 8000 // len(report.included_items[0].content) == 3


def test_an_item_a_hook_rewrote_and_marked_confidential_is_withheld_at_its_new_cost():
    def redact(it: ContextItem) -> None:
        if it.id == "i2":
            it.content = "payroll row " * 50
            it.sensitivity = "confidential"

    engine = ContextBudgetEngine(hooks=[_Rewrite(redact)])
    report = engine.synthesize(_short_items(), TokenBudget(total_budget=1000))
    assert [it.id for it in report.withheld_items] == ["i2"]
    assert report.withheld_items[0].token_count == _actual(engine, report.withheld_items) > 100
    assert len(report.included_items) == 4 and report.consumed_tokens == _actual(engine, report.included_items)


# ---- lone surrogates in hashed text --------------------------------------------------------------------------------

HIGH_HALF = json.loads('"status: done \\ud83d"')  # "\ud83d\ude00" cut after its first escape
LOW_HALF = json.loads('"status: done \\ude00"')
VALID_TEXTS = ["plain ascii", "caf\u00e9 \u6f22\u5b57", "whole emoji \U0001F600", "a:b\\c\nd", "\uffff\x00 edge"]


def _body(tail: str) -> str:
    return ("row of output\n" * 200) + tail


@pytest.mark.parametrize("tail", [HIGH_HALF, LOW_HALF, HIGH_HALF + " and more", "\ud83d" * 3])
def test_tool_output_with_a_lone_surrogate_compresses_restores_and_parses(tail):
    output = _body(tail)
    comp = DynamicToolCompressor(min_compress_chars=100)
    placeholder, artifact = comp.maybe_compress("terminal", output)
    assert artifact is not None and comp.restore(artifact.ref_id) == output
    assert DynamicToolCompressor.parse_placeholder(placeholder)["ref"] == artifact.ref_id
    # The same output gets the same ref again, and the size counts three bytes for the half pair.
    assert comp.maybe_compress("terminal", output)[1].ref_id == artifact.ref_id
    assert artifact.raw_byte_size == len(output.encode("utf-8", "surrogatepass"))


def test_outputs_that_differ_only_in_a_lone_surrogate_get_different_refs_and_digests():
    comp = DynamicToolCompressor(min_compress_chars=100)
    high, low = comp.maybe_compress("t", _body(HIGH_HALF))[1], comp.maybe_compress("t", _body(LOW_HALF))[1]
    marked = comp.maybe_compress("t", _body("status: done \ufffd"))[1]  # what errors="replace" would hash both as
    assert len({high.ref_id, low.ref_id, marked.ref_id}) == 3
    assert comp.restore(high.ref_id) == _body(HIGH_HALF) and comp.restore(low.ref_id) == _body(LOW_HALF)

    def pinned(content: str) -> str:
        return pinned_policy_digest([ContextItem(id="a", content=content, pinned=True)])

    assert len({pinned(HIGH_HALF), pinned(LOW_HALF), pinned("status: done \ufffd")}) == 3
    assert len({dropped_ids_sha256(["\ud83d"]), dropped_ids_sha256(["\ude00"]), dropped_ids_sha256(["\ufffd"])}) == 3


@pytest.mark.parametrize("pinned", [True, False])
@pytest.mark.parametrize("field", ["content", "id", "dimension"])  # the fields of the digest that accept one
def test_synthesize_accepts_a_lone_surrogate_pinned_as_unpinned(pinned, field):
    values = {"id": "a", "content": "alpha", "dimension": "invariant"}
    values[field] = values[field] + "\ud83d"
    report = ContextBudgetEngine().synthesize([ContextItem(pinned=pinned, **values)], TokenBudget(total_budget=1000))
    assert [it.id for it in report.included_items] == [values["id"]]
    if pinned:
        assert len(report.pinned_policy_digest) == 64
        assert report.pinned_policy_digest == pinned_policy_digest(report.included_items)


def test_dropped_ids_with_a_lone_surrogate_hash_and_a_build_reports_them():
    assert len(dropped_ids_sha256(["ok", "half \ud83d"])) == 64
    assert dropped_ids_sha256(["half \ud83d", "ok"]) == dropped_ids_sha256(["ok", "half \ud83d"])
    items = [ContextItem(id="half \ud83d", content="too large to fit", token_count=5000),
             ContextItem(id="rule", content="always answer in English", token_count=6, pinned=True)]
    report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=1000))
    assert [it.id for it in report.pruned_items] == ["half \ud83d"]
    build = TwoPassBuild(report, budget_tokens=500, bodies={})
    assert build.dropped_ids == ("half \ud83d",)
    assert build.receipt_payload()["dropped_ids_sha256"] == dropped_ids_sha256(["half \ud83d"])


@pytest.mark.parametrize("text", VALID_TEXTS)
def test_valid_text_keeps_the_strict_utf8_digests_and_refs(text):
    """Old sidecars, placeholders and receipts name these values: for text strict UTF-8 can encode, nothing moves."""
    output = _body(text)
    artifact = DynamicToolCompressor(min_compress_chars=100).maybe_compress("terminal", output)[1]
    strict = hashlib.sha256(f"terminal:{output}".encode("utf-8")).hexdigest()
    assert artifact.ref_id.startswith("dtoc_") and strict.startswith(artifact.ref_id[len("dtoc_"):])
    assert len(artifact.ref_id) > len("dtoc_") + 16 and artifact.raw_byte_size == len(output.encode("utf-8"))
    assert dropped_ids_sha256(["b", text + "x"]) == hashlib.sha256(
        "\n".join(sorted(["b", (text + "x").replace("\\", "\\\\").replace("\n", "\\n")])).encode("utf-8")).hexdigest()


def test_known_digests_of_valid_text_are_unchanged():
    """Values taken from the pull request head (1451dc4) before any lone-surrogate change."""
    assert dropped_ids_sha256([]) == hashlib.sha256(b"").hexdigest()
    assert dropped_ids_sha256(["b", "a"]) == hashlib.sha256(b"a\nb").hexdigest()
    assert pinned_policy_digest([]) == "empty_policy"
    plain = [ContextItem(id="rule", content="always answer in English \U0001F600", role="system", pinned=True)]
    assert pinned_policy_digest(plain) == hashlib.sha256(
        "system:None:rule:always answer in English \U0001F600\n".encode("utf-8")).hexdigest()
