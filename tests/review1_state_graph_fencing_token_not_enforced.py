"""severity: high
finding: CompiledGraph.invoke records lease_fencing_token but never checks it, so a stale lease holder's run (lower token) executes every node's write

src/harpe/state_graph.py:8 promises "Concurrency isolation integration: binds node execution to atomic leases
with monotonic fencing tokens", and CompiledGraph.invoke (state_graph.py:146-152) takes ``lease_fencing_token``
for "optional concurrency fencing". The body (state_graph.py:153-200) only copies the token into each checkpoint
dict (line 164); nothing compares it with the highest token the graph has already run under, before the run or
before each node. A holder whose lease was superseded (token 3 after token 7 ran) therefore runs every node, and
a holder superseded between two of its own nodes keeps writing after the newer holder took over. Fencing exists
to stop exactly these writes; here it is a label on the history.

The tests count calls into a node that stands for a protected write (an external store append). They accept any
way of refusing the stale run (an exception, or returning without running the node).
"""
from __future__ import annotations

from typing import Any, Dict, List

from harpe import END, StateGraph


def _write_graph(store: List[Any]):
    """One node that performs the protected write: it appends the run's tag to an external store."""

    def write(state: Dict[str, Any]) -> Dict[str, Any]:
        store.append(state["tag"])
        return {"written": True}

    g = StateGraph()
    g.add_node("write", write)
    g.set_entry_point("write")
    g.add_edge("write", END)
    return g.compile()


def test_control_increasing_tokens_both_write_and_the_token_is_checkpointed():
    store: List[Any] = []
    graph = _write_graph(store)
    first = graph.invoke({"tag": "holder-3"}, lease_fencing_token=3)
    second = graph.invoke({"tag": "holder-7"}, lease_fencing_token=7)
    assert store == ["holder-3", "holder-7"]
    assert first["written"] is True and second["written"] is True
    assert [c["fencing_token"] for c in second["_graph_history"]] == [7]


def test_a_stale_lease_holder_cannot_write_after_a_newer_token_ran():
    store: List[Any] = []
    graph = _write_graph(store)
    graph.invoke({"tag": "holder-7"}, lease_fencing_token=7)
    assert store == ["holder-7"]
    try:
        graph.invoke({"tag": "stale-holder-3"}, lease_fencing_token=3)
    except Exception:
        pass  # refusing the stale run by raising is fine
    assert store == ["holder-7"], f"a run fenced by token 3 wrote after token 7: {store}"


def test_a_holder_superseded_mid_run_cannot_write_its_next_node():
    """Deterministic interleaving: node one of the token-1 run is where a newer holder (token 2) takes the lease
    and runs; the token-1 run must not go on to perform its second write."""
    store: List[Any] = []
    graph_box: Dict[str, Any] = {}

    def first(state: Dict[str, Any]) -> Dict[str, Any]:
        if state["tag"] == "holder-1":
            graph_box["g"].invoke({"tag": "holder-2"}, lease_fencing_token=2)  # the lease moves on
        return {}

    def second(state: Dict[str, Any]) -> Dict[str, Any]:
        store.append(state["tag"])
        return {}

    g = StateGraph()
    g.add_node("first", first)
    g.add_node("second", second)
    g.set_entry_point("first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    graph_box["g"] = g.compile()
    try:
        graph_box["g"].invoke({"tag": "holder-1"}, lease_fencing_token=1)
    except Exception:
        pass
    assert store == ["holder-2"], f"the superseded token-1 run wrote after token 2 took over: {store}"
