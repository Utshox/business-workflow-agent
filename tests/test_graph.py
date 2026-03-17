"""Tests for the LangGraph workflow construction."""

from src.agent.graph import build_graph, compile_graph


def test_graph_builds():
    graph = build_graph()
    assert graph is not None


def test_graph_compiles():
    compiled = compile_graph()
    assert compiled is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {"classify", "retrieve_memory", "execute", "tools", "approval_gate", "finalize"}
    assert expected.issubset(node_names)
