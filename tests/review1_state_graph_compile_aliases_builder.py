"""severity: medium
finding: StateGraph.compile() hands the builder's own node and edge dicts to CompiledGraph, so editing the builder after compile rewires the already-compiled graph and its runs perform writes it was compiled without

StateGraph.compile (src/harpe/state_graph.py:116-127) passes ``self.nodes``, ``self.edges`` and
``self.conditional_edges`` by reference, and CompiledGraph keeps them as they are (state_graph.py:132-144). The
compiled object is described as a "Compiled, verified executable state machine" (state_graph.py:130), but any
later add_node/add_edge/add_conditional_edges on the builder (state_graph.py:72-114) changes it: a dry-run graph
compiled first and then extended into the real pipeline from the same builder starts running the real pipeline's
write node. The compiled route should be the one that existed at compile time.
"""
from __future__ import annotations

from typing import Any, Dict, List

from harpe import END, StateGraph


def _builder(store: List[str]) -> StateGraph:
    def plan(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"planned": True}

    def write(state: Dict[str, Any]) -> Dict[str, Any]:
        store.append("write")
        return {"written": True}

    g = StateGraph()
    g.add_node("plan", plan)
    g.add_node("write", write)
    g.set_entry_point("plan")
    g.add_edge("plan", END)
    return g


def test_control_each_compiled_route_runs_as_built():
    store: List[str] = []
    g = _builder(store)
    g.add_edge("plan", "write")
    g.add_edge("write", END)
    out = g.compile().invoke({})
    assert store == ["write"] and out["planned"] and out["written"]
    assert [c["node"] for c in out["_graph_history"]] == ["plan", "write"]


def test_extending_the_builder_does_not_rewire_a_graph_compiled_earlier():
    store: List[str] = []
    g = _builder(store)
    dry_run = g.compile()  # plan -> END
    g.add_edge("plan", "write")
    g.add_edge("write", END)
    real = g.compile()  # plan -> write -> END
    out = dry_run.invoke({})
    assert store == [], "the dry-run graph ran the write node added after it was compiled"
    assert [c["node"] for c in out["_graph_history"]] == ["plan"]
    real.invoke({})
    assert store == ["write"]
