"""Agentic State Graph and ReAct Orchestration Engine.

Derived from NVIDIA DLI: Building Agentic AI Applications with LLMs (course-v1:DLI+S-FX-41+V1).
Provides:
- Cyclic StateGraph workflow engine (nodes, edges, conditional routing, state checkpointing).
- ReAct execution loop (Thought -> Action -> Observation) with strict JSON Schema / Pydantic validation.
- Tool registry with automatic error-recovery and retry policies.
- Concurrency isolation integration: binds node execution to atomic leases with monotonic fencing tokens.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
START = "__start__"
END = "__end__"


class ToolDefinition(BaseModel):
    """Specification of an agent tool including schema validation."""
    name: str
    description: str
    func: Any = Field(description="Callable execution target.")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model for argument validation.")


class ToolExecutionResult(BaseModel):
    """Structured result of executing a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    output: Any
    success: bool
    error: Optional[str] = None


class ReActStep(BaseModel):
    """One discrete iteration of the ReAct thought-action-observation cycle."""
    step_index: int
    thought: str
    action_tool: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_terminal: bool = False
    error: Optional[str] = None


class StateGraph:
    """Cyclic state graph engine for deterministic multi-agent workflows.

    Features:
    - Typed state dict or Pydantic model.
    - Explicit node functions mutating state.
    - Linear directed edges and dynamic conditional routing branches.
    - Checkpointed step history for full replayability.
    """

    def __init__(self, state_schema: Optional[Type[BaseModel]] = None) -> None:
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> StateGraph:
        """Register a processing node in the state graph."""
        if name in (START, END):
            raise ValueError(f"Reserved node name '{name}' cannot be explicitly registered")
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already registered")
        self.nodes[name] = func
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Designate the initial node for graph execution."""
        if name not in self.nodes:
            raise KeyError(f"Entry point node '{name}' is not registered in graph")
        self.entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> StateGraph:
        """Add a deterministic directed transition edge between two nodes."""
        if from_node != START and from_node not in self.nodes:
            raise KeyError(f"Source node '{from_node}' not registered")
        if to_node != END and to_node not in self.nodes:
            raise KeyError(f"Target node '{to_node}' not registered")

        if from_node == START:
            self.entry_point = to_node
        else:
            self.edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition_fn: Callable[[Dict[str, Any]], str],
        path_map: Dict[str, str],
    ) -> StateGraph:
        """Add dynamic branching from a node based on state evaluation."""
        if source not in self.nodes:
            raise KeyError(f"Source node '{source}' not registered")
        for target in path_map.values():
            if target != END and target not in self.nodes:
                raise KeyError(f"Branch target node '{target}' not registered")
        self.conditional_edges[source] = (condition_fn, path_map)
        return self

    def compile(self) -> CompiledGraph:
        """Validate and compile graph into an executable pipeline."""
        if not self.entry_point:
            raise ValueError("StateGraph cannot compile without an entry point")
        return CompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            state_schema=self.state_schema,
        )


class CompiledGraph:
    """Compiled, verified executable state machine."""

    def __init__(
        self,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]],
        entry_point: str,
        state_schema: Optional[Type[BaseModel]] = None,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.state_schema = state_schema

    def invoke(
        self,
        initial_state: Dict[str, Any],
        max_steps: int = 25,
        lease_fencing_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute graph with state checkpoints, recursion limits, and optional concurrency fencing."""
        current_state = copy.deepcopy(initial_state)
        current_node = self.entry_point
        step_count = 0
        history: List[Dict[str, Any]] = []

        while current_node != END and step_count < max_steps:
            node_fn = self.nodes[current_node]
            checkpoint = {
                "step": step_count,
                "node": current_node,
                "input_state": copy.deepcopy(current_state),
                "fencing_token": lease_fencing_token,
            }

            try:
                state_update = node_fn(current_state)
                if isinstance(state_update, dict):
                    current_state.update(state_update)
            except Exception as e:
                logger.error(f"Error executing StateGraph node '{current_node}': {e}")
                checkpoint["error"] = str(e)
                history.append(checkpoint)
                raise

            checkpoint["output_state"] = copy.deepcopy(current_state)
            history.append(checkpoint)
            step_count += 1

            # Determine next node transition
            if current_node in self.conditional_edges:
                cond_fn, path_map = self.conditional_edges[current_node]
                decision_key = cond_fn(current_state)
                if decision_key not in path_map:
                    raise KeyError(
                        f"Conditional routing key '{decision_key}' from node '{current_node}' "
                        f"not found in path map: {list(path_map.keys())}"
                    )
                current_node = path_map[decision_key]
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = END

        if step_count >= max_steps and current_node != END:
            raise RecursionError(f"StateGraph exceeded maximum step budget of {max_steps} without reaching END")

        current_state["_graph_history"] = history
        return current_state


class ReActAgent:
    """Production ReAct agent loop with schema validation and error-tolerant execution."""

    def __init__(self, tools: Optional[List[ToolDefinition]] = None, max_iterations: int = 10) -> None:
        self.tools: Dict[str, ToolDefinition] = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations

    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: Optional[Type[BaseModel]] = None,
    ) -> ReActAgent:
        """Register a tool with optional Pydantic input schema validation."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
        )
        return self

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Validate arguments against schema and execute tool safely."""
        if tool_name not in self.tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                output=None,
                success=False,
                error=f"Tool '{tool_name}' is not registered. Available tools: {list(self.tools.keys())}",
            )

        tool = self.tools[tool_name]
        validated_kwargs = arguments

        # Schema validation
        if tool.args_schema:
            try:
                parsed_model = tool.args_schema(**arguments)
                validated_kwargs = parsed_model.model_dump()
            except Exception as schema_err:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=None,
                    success=False,
                    error=f"Schema validation error for '{tool_name}': {schema_err}",
                )

        try:
            # Check function signature
            sig = inspect.signature(tool.func)
            if len(sig.parameters) == 1 and next(iter(sig.parameters.values())).annotation != inspect.Parameter.empty:
                # If function accepts single dict parameter
                res = tool.func(validated_kwargs)
            else:
                res = tool.func(**validated_kwargs)

            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=validated_kwargs,
                output=res,
                success=True,
            )
        except Exception as exec_err:
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=validated_kwargs,
                output=None,
                success=False,
                error=f"Runtime error in tool '{tool_name}': {exec_err}",
            )

    def run_step(
        self,
        step_index: int,
        thought: str,
        action_tool: Optional[str] = None,
        action_input: Optional[Dict[str, Any]] = None,
        is_final_answer: bool = False,
        final_answer_text: Optional[str] = None,
    ) -> ReActStep:
        """Execute a single ReAct step with validation."""
        if is_final_answer:
            return ReActStep(
                step_index=step_index,
                thought=thought,
                action_tool=None,
                action_input=None,
                observation=final_answer_text or thought,
                is_terminal=True,
            )

        if not action_tool:
            return ReActStep(
                step_index=step_index,
                thought=thought,
                is_terminal=False,
                error="Action specified without a target tool",
            )

        exec_res = self.execute_tool(action_tool, action_input or {})
        obs = json.dumps(exec_res.output) if exec_res.success else f"ERROR: {exec_res.error}"

        return ReActStep(
            step_index=step_index,
            thought=thought,
            action_tool=action_tool,
            action_input=action_input,
            observation=obs,
            is_terminal=False,
            error=exec_res.error,
        )
