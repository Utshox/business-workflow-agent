"""LangGraph workflow definition — the core agent graph."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import (
    approval_gate,
    classify_workflow,
    execute_workflow,
    finalize,
    process_approval,
    retrieve_memory,
    route_after_approval,
    route_after_execute,
)
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS


def build_graph() -> StateGraph:
    """Build and compile the workflow agent graph."""
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("classify", classify_workflow)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("execute", execute_workflow)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("process_approval", process_approval)
    graph.add_node("finalize", finalize)

    # --- Edges ---
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve_memory")
    graph.add_edge("retrieve_memory", "execute")

    # After execution: call tools, go to approval, or finalize
    graph.add_conditional_edges("execute", route_after_execute, {
        "tools": "tools",
        "approval_gate": "approval_gate",
        "finalize": "finalize",
    })

    # Tool results loop back to execute for the LLM to process
    graph.add_edge("tools", "execute")

    # Approval gate uses LangGraph's interrupt mechanism
    graph.add_conditional_edges("approval_gate", route_after_approval, {
        "finalize": "finalize",
        "done": END,
    })

    graph.add_edge("process_approval", "finalize")
    graph.add_edge("finalize", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with optional checkpointer for persistence."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    graph = build_graph()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval_gate"],  # Human-in-the-loop pause point
    )
