"""Human-in-the-loop approval mechanism for the workflow agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.agent.state import ApprovalStatus


@dataclass
class ApprovalRequest:
    workflow_id: str
    thread_id: str
    summary: str
    draft_output: str
    workflow_type: str
    metadata: dict[str, Any]


class ApprovalManager:
    """Manages human approval requests for workflow actions.

    In production, this would integrate with Slack/email/web UI.
    Here it exposes an async API that the FastAPI server bridges to users.
    """

    def __init__(self):
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, asyncio.Future] = {}

    def request_approval(self, request: ApprovalRequest) -> asyncio.Future:
        """Submit an approval request and return a future for the decision."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[request.workflow_id] = request
        self._decisions[request.workflow_id] = future
        return future

    def approve(self, workflow_id: str, reason: str = "") -> bool:
        """Approve a pending request."""
        return self._resolve(workflow_id, ApprovalStatus.APPROVED, reason)

    def reject(self, workflow_id: str, reason: str = "") -> bool:
        """Reject a pending request."""
        return self._resolve(workflow_id, ApprovalStatus.REJECTED, reason)

    def list_pending(self) -> list[ApprovalRequest]:
        """List all pending approval requests."""
        return list(self._pending.values())

    def _resolve(self, workflow_id: str, status: ApprovalStatus, reason: str) -> bool:
        future = self._decisions.get(workflow_id)
        if not future or future.done():
            return False
        future.set_result({"status": status, "reason": reason})
        del self._pending[workflow_id]
        return True


# Singleton instance
approval_manager = ApprovalManager()
