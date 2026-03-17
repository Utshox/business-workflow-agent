"""Shared state definitions for the workflow agent."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

from langgraph.graph import MessagesState


class WorkflowType(str, Enum):
    TICKET_TRIAGE = "ticket_triage"
    REPORT_DRAFT = "report_draft"
    DATA_LOOKUP = "data_lookup"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    tool_name: str
    input_data: dict[str, Any]
    output_data: Any
    success: bool
    error: str | None = None


class AgentState(MessagesState):
    """Full state flowing through the LangGraph workflow."""

    workflow_type: WorkflowType | None
    current_step: str
    input_data: dict[str, Any]

    # Tool execution tracking
    tool_results: Annotated[list[ToolResult], operator.add]

    # Human-in-the-loop
    approval_status: ApprovalStatus
    approval_reason: str
    requires_approval: bool

    # Workflow outputs
    draft_output: str
    final_output: str
    confidence: float

    # Memory context retrieved from vector store
    memory_context: list[str]
