"""Regression tests for the laundering and fencing fixes: every path with the same flaw as a reviewed finding.

Each test pins one rule the fixes established on a path the finding tests do not reach: fencing refuses a run
that drops its token or is superseded while a node runs, compiled graphs and two-pass builds keep their own copies,
the receipt and DTOC hashes are injective while keeping their values for ordinary input, the remote check fails
closed and is made after the hooks, and a typed sidecar or loop field always wins over free metadata.
"""
from __future__ import annotations

import hashlib

import pytest

from harpe import (
    ContextBudgetEngine,
    ContextItem,
    DynamicToolCompressor,
    ProfileLedger,
    StaleFencingTokenError,
    StateGraph,
    TokenBudget,
    TwoPassBuild,
    dropped_ids_sha256,
    pinned_policy_digest,
)
from harpe.dtoc import CompressedToolOutput
from harpe.governor import BehavioralGovernor, GovernorVerdict
from harpe.state_graph import END
from harpe.suppression import SuppressionController, SuppressionScope


# ---- state graph fencing -------------------------------------------------------------------------------------
def _one_node_graph(calls):
    def write(state):
        calls.append(state.get("holder"))
        return {"written": True}

    return StateGraph().add_node("write", write).set_entry_point("write").add_edge("write", END).compile()


def test_a_run_without_a_token_is_refused_once_the_graph_has_run_under_one():
    calls = []
    graph = _one_node_graph(calls)
    assert graph.invoke({"holder": "unfenced"})["written"] is True  # unfenced while no token was used
    graph.invoke({"holder": "t5"}, lease_fencing_token=5)
    with pytest.raises(StaleFencingTokenError):
        graph.invoke({"holder": "dropped-token"})
    assert calls == ["unfenced", "t5"]
    assert graph.fencing_high_water == 5


def test_equal_and_newer_tokens_run_and_raise_the_high_water():
    calls = []
    graph = _one_node_graph(calls)
    for token in (1, 1, 4, 9):
        graph.invoke({"holder": token}, lease_fencing_token=token)
    assert calls == [1, 1, 4, 9] and graph.fencing_high_water == 9


def test_a_node_superseded_while_it_runs_has_its_output_discarded_and_no_next_node_runs():
    ran = []
    graph_box = {}

    def first(state):
        ran.append("first")
        graph_box["graph"].invoke({"newer": True}, lease_fencing_token=8)  # a newer holder runs meanwhile
        return {"first_done": True}

    def second(state):
        ran.append("second")
        return {}

    def entry(state):
        return {}

    graph = (
        StateGraph()
        .add_node("entry", entry)
        .add_node("first", first)
        .add_node("second", second)
        .set_entry_point("entry")
        .add_conditional_edges("entry", lambda s: "newer" if s.get("newer") else "older",
                               {"newer": END, "older": "first"})
        .add_edge("first", "second")
        .add_edge("second", END)
        .compile()
    )
    graph_box["graph"] = graph
    with pytest.raises(StaleFencingTokenError):
        graph.invoke({}, lease_fencing_token=3)
    assert ran == ["first"]


def test_a_fencing_token_must_be_an_int():
    graph = _one_node_graph([])
    for bad in (True, "7", 7.0):
        with pytest.raises(TypeError):
            graph.invoke({}, lease_fencing_token=bad)  # type: ignore[arg-type]
    assert graph.fencing_high_water is None


def test_editing_a_branch_map_after_add_conditional_edges_does_not_reroute_the_compiled_graph():
    ran = []
    path_map = {"go": END}
    builder = (
        StateGraph()
        .add_node("plan", lambda s: ran.append("plan") or {})
        .add_node("write", lambda s: ran.append("write") or {})
        .set_entry_point("plan")
        .add_conditional_edges("plan", lambda s: "go", path_map)
    )
    graph = builder.compile()
    path_map["go"] = "write"  # the caller's dict, edited after registration
    graph.invoke({})
    builder.add_edge("write", END)
    assert ran == ["plan"]
    assert builder.compile().invoke({}) is not None and ran == ["plan", "plan"]


# ---- hooks and the remote check --------------------------------------------------------------------------------
class _Hook:
    def __init__(self, fn):
        self.fn = fn

    def pre_prune(self, items):
        self.fn(items)

    def post_prune(self, report):
        pass


def test_an_item_a_hook_gives_an_unvalidated_sensitivity_is_withheld_from_remote():
    def relabel(items):
        for it in items:
            if it.id == "x":
                it.sensitivity = "restricted"  # assignment is not validated: fail closed

    engine = ContextBudgetEngine(hooks=[_Hook(relabel)])
    items = [ContextItem(id="x", content="internal figures", token_count=5),
             ContextItem(id="y", content="public notes", token_count=5)]
    report = engine.synthesize(items, TokenBudget(total_budget=1000))
    assert [it.id for it in report.included_items] == ["y"]
    assert [it.id for it in report.withheld_items] == ["x"]
    local = engine.synthesize(items, TokenBudget(total_budget=1000), destination="local")
    assert {it.id for it in local.included_items} == {"x", "y"}


def test_an_item_a_hook_leaves_without_a_token_count_is_counted_before_selection():
    def blank(items):
        for it in items:
            it.token_count = 0
        items.append(ContextItem(id="added", content="added by the hook " * 50))

    engine = ContextBudgetEngine(hooks=[_Hook(blank)])
    report = engine.synthesize([ContextItem(id="a", content="alpha " * 40, token_count=3)],
                               TokenBudget(total_budget=10_000))
    assert all(it.token_count > 0 for it in report.included_items)
    assert report.consumed_tokens == sum(it.token_count for it in report.included_items)
    assert {it.id for it in report.included_items} == {"a", "added"}


# ---- two-pass build: copies and receipt hashing ----------------------------------------------------------------
def _build():
    engine = ContextBudgetEngine()
    pinned = ContextItem(id="rule", content="Always cite sources.", pinned=True, token_count=5)
    row = ContextItem(id="doc", content="doc description", token_count=10)
    report = engine.synthesize([pinned, row], TokenBudget(total_budget=1000))
    body = ContextItem(id="doc", content="the full document body", token_count=40)
    return report, TwoPassBuild(report, bodies={"doc": body}, budget_tokens=1000)


def test_a_two_pass_build_does_not_follow_later_edits_to_the_callers_items():
    report, build = _build()
    digest = build.receipt_payload()["pinned_policy_digest"]
    for it in report.included_items:
        it.content = "rewritten after the build"
    assert build.receipt_payload()["pinned_policy_digest"] == digest
    assert [it.content for it in build.items()] == ["Always cite sources.", "doc description"]


def test_items_handed_out_by_a_two_pass_build_are_copies():
    _, build = _build()
    build.items()[0].content = "edited by a consumer"
    assert build.items()[0].content == "Always cite sources."


def test_dropped_ids_sha256_keeps_its_value_for_ordinary_ids():
    assert dropped_ids_sha256([]) == hashlib.sha256(b"").hexdigest()
    assert dropped_ids_sha256(["z", "secret", "a-1"]) == hashlib.sha256(b"a-1\nsecret\nz").hexdigest()


def test_dropped_ids_sha256_names_exactly_one_set_of_ids():
    distinct = [[], [""], ["\\"], ["a\nb"], ["a", "b"], ["a\\nb"], ["a\\", "b"]]
    digests = [dropped_ids_sha256(ids) for ids in distinct]
    assert len(set(digests)) == len(distinct)


def test_pinned_policy_digest_separates_absent_none_and_empty_dimensions():
    def pin(dimension):
        return ContextItem(id="p", content="c", pinned=True, dimension=dimension)

    digests = {pinned_policy_digest([pin(None)]), pinned_policy_digest([pin("None")]), pinned_policy_digest([pin("")])}
    assert len(digests) == 3


# ---- typed fields win over free metadata ----------------------------------------------------------------------
def test_token_budget_aliases_are_resolved_without_writing_into_the_callers_dict():
    given = {"total_budget": 1000, "system_reserve": 50}
    budget = TokenBudget.model_validate(given)
    assert budget.safety_margin == 50
    assert given == {"total_budget": 1000, "system_reserve": 50}


def test_dtoc_refs_do_not_collide_across_the_tool_name_separator():
    comp = DynamicToolCompressor(min_compress_chars=10)
    tail = "x" * 40
    _, first = comp.maybe_compress("mcp:fetch", tail)
    _, second = comp.maybe_compress("mcp", "fetch:" + tail)
    assert first is not None and second is not None
    assert first.ref_id != second.ref_id
    assert comp.restore(first.ref_id) == tail and comp.restore(second.ref_id) == "fetch:" + tail


def test_dtoc_ref_keeps_its_value_for_an_ordinary_tool_name():
    comp = DynamicToolCompressor(min_compress_chars=10)
    output = "line\n" * 20
    _, artifact = comp.maybe_compress("terminal", output)
    assert artifact is not None
    assert artifact.ref_id == "dtoc_" + hashlib.sha256(f"terminal:{output}".encode()).hexdigest()[:32]


def test_a_dtoc_item_is_confidential_whenever_its_placeholder_says_so():
    artifact = CompressedToolOutput(ref_id="dtoc_manual", tool_name="vault", summary="s", raw_payload="p",
                                    raw_byte_size=1, raw_lines=1, metadata={"confidential": True, "scope": "old"})
    item = DynamicToolCompressor().to_context_item(artifact)
    assert "confidential=true" in item.content
    assert item.sensitivity == "confidential" and item.metadata["confidential"] is True
    assert item.metadata["scope"] == "old"  # free metadata still fills a typed field that is unset


def test_a_dtoc_item_reports_the_typed_confidential_flag_over_caller_metadata():
    comp = DynamicToolCompressor(min_compress_chars=10)
    _, artifact = comp.maybe_compress("vault", "secret row\n" * 10, metadata={"confidential": False},
                                      confidential=True)
    assert artifact is not None
    item = comp.to_context_item(artifact)
    assert item.sensitivity == "confidential" and item.metadata["confidential"] is True


# ---- order-independent and case-independent rules -------------------------------------------------------------
def test_the_strongest_soft_suppression_applies_whatever_order_the_rules_were_added_in():
    item = ContextItem(id="row-1", content="stale figures", metadata={"category": "finance"})
    for order in ((0.8, 0.3), (0.3, 0.8)):
        ctl = SuppressionController()
        for factor in order:
            ctl.add_rule(SuppressionScope.EXACT_ID, "row-1", reason=f"damp {factor}", discount_factor=factor)
        assert ctl.evaluate_item(item) == (True, 0.3, "damp 0.3")


def test_an_uppercase_revoked_history_directive_needs_review():
    hist = ContextItem(id="h", content="Use the old token.", metadata={"epistemic_state": "REVOKED"})
    decision = BehavioralGovernor().evaluate("deploy build 42", [], candidate_history=[hist])
    assert decision.verdict == GovernorVerdict.NEEDS_REVIEW


def test_default_order_one_sentence_turns_make_only_the_latest_residence_current():
    ledger = ProfileLedger()
    ledger.add("t0", "I live in Denver with my two cats.")
    ledger.add("t1", "I live in Seattle now, near the office.")
    status = {r.statement.unit_id: r.status for r in ledger.resolve("residence")}
    assert status == {"t0": "superseded", "t1": "current"}
    assert ledger.latest_order is None  # only an order the caller gave moves latest_order
