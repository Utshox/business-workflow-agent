"""LangGraph node functions for the workflow agent."""

from __future__ import annotations

from typing import Any

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState, ApprovalStatus, WorkflowType
from src.agent.tools import ALL_TOOLS
from src.memory.vector_store import MemoryStore


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@lru_cache(maxsize=1)
def get_llm_with_tools():
    return get_llm().bind_tools(ALL_TOOLS)


@lru_cache(maxsize=1)
def get_memory_store():
    return MemoryStore()

CLASSIFIER_PROMPT = """\
You are a workflow classifier. Given a user request, classify it into exactly one of:
- ticket_triage: Support tickets, bug reports, customer issues, escalations
- report_draft: Report generation, data summaries, status updates, analytics
- data_lookup: Customer lookups, metric queries, knowledge base searches

Respond with ONLY the workflow type, nothing else."""

TRIAGE_PROMPT = """\
You are a support ticket triage agent. Analyze the ticket and:
1. Look up the customer if a customer ID is mentioned
2. Search the knowledge base for relevant articles
3. Assign a priority (critical/high/medium/low) based on:
   - Customer tier (enterprise = higher priority)
   - Issue severity and business impact
   - Customer health score
4. Create a ticket with your recommended priority and provide a triage summary.

Use the available tools to gather information before making decisions."""

REPORT_PROMPT = """\
You are a report drafting agent. Given the request:
1. Query relevant metrics and customer data
2. Generate formatted report sections
3. Compile a professional draft report

Use tools to gather real data before writing. Structure the report clearly."""

LOOKUP_PROMPT = """\
You are a data lookup agent. Given the query:
1. Identify what data is needed (customer info, metrics, KB articles)
2. Use the appropriate tools to fetch the data
3. Synthesize findings into a clear, concise answer

Always cite which tools/sources you used."""


def classify_workflow(state: AgentState) -> dict[str, Any]:
    """Classify the incoming request into a workflow type."""
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=str(state["messages"][-1].content)),
    ]
    response = get_llm().invoke(messages)
    text = response.content.strip().lower()

    wf_map = {
        "ticket_triage": WorkflowType.TICKET_TRIAGE,
        "report_draft": WorkflowType.REPORT_DRAFT,
        "data_lookup": WorkflowType.DATA_LOOKUP,
    }
    workflow_type = wf_map.get(text, WorkflowType.DATA_LOOKUP)

    return {
        "workflow_type": workflow_type,
        "current_step": "retrieve_memory",
    }


def retrieve_memory(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant context from vector memory."""
    query = str(state["messages"][-1].content)
    docs = get_memory_store().search(query, k=3)
    return {
        "memory_context": [doc.page_content for doc in docs],
        "current_step": "execute",
    }


def execute_workflow(state: AgentState) -> dict[str, Any]:
    """Execute the appropriate workflow with tool calling."""
    workflow_type = state.get("workflow_type", WorkflowType.DATA_LOOKUP)
    memory_ctx = state.get("memory_context", [])

    prompt_map = {
        WorkflowType.TICKET_TRIAGE: TRIAGE_PROMPT,
        WorkflowType.REPORT_DRAFT: REPORT_PROMPT,
        WorkflowType.DATA_LOOKUP: LOOKUP_PROMPT,
    }
    system_prompt = prompt_map[workflow_type]

    if memory_ctx:
        system_prompt += "\n\nRelevant context from memory:\n" + "\n".join(
            f"- {ctx}" for ctx in memory_ctx
        )

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = get_llm_with_tools().invoke(messages)

    # Determine if this workflow needs human approval
    needs_approval = workflow_type == WorkflowType.TICKET_TRIAGE and _is_high_priority(response)

    return {
        "messages": [response],
        "draft_output": response.content or "",
        "requires_approval": needs_approval,
        "current_step": "approval_gate" if needs_approval else "finalize",
    }


def handle_tool_calls(state: AgentState) -> dict[str, Any]:
    """Process tool calls from the LLM response."""
    from langgraph.prebuilt import ToolNode

    tool_node = ToolNode(ALL_TOOLS)
    return tool_node.invoke(state)


def approval_gate(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval checkpoint. LangGraph interrupt handles the pause."""
    return {
        "approval_status": ApprovalStatus.PENDING,
        "current_step": "approval_gate",
    }


def process_approval(state: AgentState) -> dict[str, Any]:
    """Process the human approval decision."""
    status = state.get("approval_status", ApprovalStatus.PENDING)
    if status == ApprovalStatus.APPROVED:
        return {"current_step": "finalize"}
    else:
        reason = state.get("approval_reason", "Rejected by reviewer")
        return {
            "messages": [AIMessage(content=f"Workflow paused: {reason}. Please revise and resubmit.")],
            "current_step": "done",
        }


def finalize(state: AgentState) -> dict[str, Any]:
    """Finalize the workflow output and store in memory."""
    draft = state.get("draft_output", "")
    workflow_type = state.get("workflow_type", WorkflowType.DATA_LOOKUP)

    # Store the completed workflow in memory for future reference
    if draft:
        user_msg = str(state["messages"][0].content) if state["messages"] else ""
        get_memory_store().add(
            text=f"[{workflow_type.value}] Q: {user_msg[:200]} A: {draft[:500]}",
            metadata={"workflow_type": workflow_type.value},
        )

    return {
        "final_output": draft,
        "current_step": "done",
    }


def _is_high_priority(response) -> bool:
    """Check if the LLM response indicates a high-priority ticket."""
    content = (response.content or "").lower()
    return any(kw in content for kw in ["critical", "high priority", "urgent", "p0", "p1"])


def route_after_classify(state: AgentState) -> str:
    return "retrieve_memory"


def route_after_execute(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if state.get("requires_approval"):
        return "approval_gate"
    return "finalize"


def route_after_approval(state: AgentState) -> str:
    status = state.get("approval_status", ApprovalStatus.PENDING)
    if status == ApprovalStatus.APPROVED:
        return "finalize"
    return "done"
