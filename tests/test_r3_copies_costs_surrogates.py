"""Regression tests for the third review of PR #16 (2026-11-10, review_notes/R3_PR16.md).

* The copies ``synthesize`` gives its hooks and the copies ``TwoPassBuild`` keeps and hands out own the dicts, lists,
  tuples and sets inside ``metadata`` and share every other object there: a lock, a client or an open handle no
  longer raises, and nothing written into a copy's ``metadata``, at any depth, reaches the item it was made from.
"""
from __future__ import annotations

import collections
import threading
from typing import Any, Dict, List, Sequence

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from harpe.models import _detached_copy


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
