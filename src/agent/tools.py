"""Tool definitions for the business workflow agent."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from langchain_core.tools import tool


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up customer information by ID. Returns account details and history."""
    customers = {
        "C001": {
            "name": "Acme Corp",
            "tier": "enterprise",
            "arr": 120000,
            "health_score": 85,
            "csm": "Jane Smith",
            "open_tickets": 2,
        },
        "C002": {
            "name": "StartupXYZ",
            "tier": "growth",
            "arr": 24000,
            "health_score": 62,
            "csm": "Bob Lee",
            "open_tickets": 5,
        },
        "C003": {
            "name": "MegaBank Inc",
            "tier": "enterprise",
            "arr": 500000,
            "health_score": 91,
            "csm": "Alice Chen",
            "open_tickets": 1,
        },
    }
    customer = customers.get(customer_id)
    if not customer:
        return json.dumps({"error": f"Customer {customer_id} not found"})
    return json.dumps(customer)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant articles and documentation."""
    articles = [
        {
            "id": "KB001",
            "title": "Password Reset Procedure",
            "summary": "Steps to reset user passwords via admin portal or self-service.",
            "tags": ["auth", "password", "access"],
        },
        {
            "id": "KB002",
            "title": "API Rate Limiting Policy",
            "summary": "Rate limits are 1000 req/min for enterprise, 100 req/min for growth tier.",
            "tags": ["api", "rate-limit", "performance"],
        },
        {
            "id": "KB003",
            "title": "Data Export Guide",
            "summary": "How to export account data in CSV/JSON. Available for enterprise tier only.",
            "tags": ["data", "export", "enterprise"],
        },
        {
            "id": "KB004",
            "title": "Billing & Invoice FAQ",
            "summary": "Common billing questions: proration, refunds, plan changes.",
            "tags": ["billing", "invoice", "payment"],
        },
    ]
    query_lower = query.lower()
    matches = [a for a in articles if any(tag in query_lower for tag in a["tags"])
               or any(word in a["title"].lower() for word in query_lower.split())]
    return json.dumps(matches[:3] if matches else [articles[0]])


@tool
def query_metrics_db(metric_name: str, customer_id: str, days: int = 30) -> str:
    """Query the metrics database for customer usage data."""
    base = random.randint(100, 10000)
    data_points = []
    for i in range(min(days, 30)):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        value = base + random.randint(-50, 50)
        data_points.append({"date": date, "value": value})

    return json.dumps({
        "metric": metric_name,
        "customer_id": customer_id,
        "period_days": days,
        "data": data_points[:7],
        "avg": base,
        "trend": random.choice(["up", "down", "stable"]),
    })


@tool
def create_ticket(
    title: str,
    description: str,
    priority: str,
    assignee: str | None = None,
) -> str:
    """Create a new support ticket in the ticketing system."""
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    return json.dumps({
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "assignee": assignee or "unassigned",
        "status": "open",
        "created_at": datetime.now().isoformat(),
    })


@tool
def send_notification(channel: str, recipient: str, message: str) -> str:
    """Send a notification via email or Slack. Requires human approval for external sends."""
    return json.dumps({
        "status": "sent",
        "channel": channel,
        "recipient": recipient,
        "message_preview": message[:100],
        "timestamp": datetime.now().isoformat(),
    })


@tool
def generate_report_section(
    section_title: str,
    data_summary: str,
    tone: str = "professional",
) -> str:
    """Generate a formatted report section from summarized data."""
    return json.dumps({
        "section_title": section_title,
        "content": f"## {section_title}\n\n{data_summary}\n\n"
                   f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                   f"in {tone} tone.*",
        "word_count": len(data_summary.split()),
    })


ALL_TOOLS = [
    lookup_customer,
    search_knowledge_base,
    query_metrics_db,
    create_ticket,
    send_notification,
    generate_report_section,
]
