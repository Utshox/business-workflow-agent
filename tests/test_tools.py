"""Tests for the workflow agent tools."""

import json

from src.agent.tools import (
    create_ticket,
    lookup_customer,
    query_metrics_db,
    search_knowledge_base,
)


def test_lookup_customer_found():
    result = json.loads(lookup_customer.invoke({"customer_id": "C001"}))
    assert result["name"] == "Acme Corp"
    assert result["tier"] == "enterprise"


def test_lookup_customer_not_found():
    result = json.loads(lookup_customer.invoke({"customer_id": "C999"}))
    assert "error" in result


def test_search_knowledge_base():
    result = json.loads(search_knowledge_base.invoke({"query": "password reset auth"}))
    assert len(result) > 0
    assert any("Password" in a["title"] for a in result)


def test_query_metrics_db():
    result = json.loads(query_metrics_db.invoke({
        "metric_name": "api_calls",
        "customer_id": "C001",
        "days": 7,
    }))
    assert result["metric"] == "api_calls"
    assert result["customer_id"] == "C001"
    assert len(result["data"]) > 0


def test_create_ticket():
    result = json.loads(create_ticket.invoke({
        "title": "Test ticket",
        "description": "Testing",
        "priority": "medium",
    }))
    assert result["ticket_id"].startswith("TKT-")
    assert result["status"] == "open"
