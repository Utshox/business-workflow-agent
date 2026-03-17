"""FastAPI server exposing the workflow agent and approval endpoints."""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.graph import compile_graph
from src.agent.state import ApprovalStatus
from src.approval.human_loop import approval_manager

load_dotenv()

app = FastAPI(
    title="Business Workflow Agent",
    description="Multi-step workflow agent with tool integration and human-in-the-loop approval",
    version="0.1.0",
)

graph = compile_graph()


# --- Request/Response models ---

class WorkflowRequest(BaseModel):
    message: str
    thread_id: str | None = None


class WorkflowResponse(BaseModel):
    thread_id: str
    status: str
    output: str
    workflow_type: str | None = None
    requires_approval: bool = False


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


class ApprovalResponse(BaseModel):
    workflow_id: str
    summary: str
    draft_output: str
    workflow_type: str


# --- Endpoints ---

@app.post("/workflow/run", response_model=WorkflowResponse)
async def run_workflow(request: WorkflowRequest):
    """Submit a request to the workflow agent."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config=config,
    )

    # Check if the graph was interrupted for approval
    state = graph.get_state(config)
    is_interrupted = bool(state.next)

    output = result.get("final_output", "")
    if is_interrupted:
        output = result.get("draft_output", "Awaiting approval...")

    return WorkflowResponse(
        thread_id=thread_id,
        status="awaiting_approval" if is_interrupted else "completed",
        output=output,
        workflow_type=result.get("workflow_type", {}).value
        if result.get("workflow_type") else None,
        requires_approval=is_interrupted,
    )


@app.post("/workflow/{thread_id}/approve", response_model=WorkflowResponse)
async def approve_workflow(thread_id: str, decision: ApprovalDecision):
    """Approve or reject a paused workflow."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if not state.next:
        raise HTTPException(404, "No pending approval for this thread")

    new_status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED

    result = await graph.ainvoke(
        {
            "approval_status": new_status,
            "approval_reason": decision.reason,
        },
        config=config,
    )

    return WorkflowResponse(
        thread_id=thread_id,
        status="completed" if decision.approved else "rejected",
        output=result.get("final_output", ""),
        workflow_type=result.get("workflow_type", {}).value
        if result.get("workflow_type") else None,
        requires_approval=False,
    )


@app.get("/approvals/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals():
    """List all workflows awaiting human approval."""
    pending = approval_manager.list_pending()
    return [
        ApprovalResponse(
            workflow_id=p.workflow_id,
            summary=p.summary,
            draft_output=p.draft_output,
            workflow_type=p.workflow_type,
        )
        for p in pending
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}


def start():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()
