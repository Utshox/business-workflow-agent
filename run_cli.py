"""Interactive CLI for testing the workflow agent locally."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent.graph import compile_graph

load_dotenv()
console = Console()


async def main():
    graph = compile_graph()
    thread_id = "cli-session"
    config = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(
        "[bold]Business Workflow Agent[/bold]\n"
        "Supports: ticket triage, report drafting, data lookup\n"
        "Type 'quit' to exit, 'approve' or 'reject' for pending approvals.",
        title="Welcome",
    ))

    while True:
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Handle approval commands
        if user_input.lower() in ("approve", "reject"):
            state = graph.get_state(config)
            if not state.next:
                console.print("[yellow]No pending approval.[/yellow]")
                continue

            from src.agent.state import ApprovalStatus
            status = ApprovalStatus.APPROVED if user_input.lower() == "approve" else ApprovalStatus.REJECTED
            result = await graph.ainvoke(
                {"approval_status": status, "approval_reason": user_input},
                config=config,
            )
            output = result.get("final_output", "Done.")
            console.print(Panel(output, title="Agent", border_style="green"))
            continue

        # Run the workflow
        console.print("[dim]Processing...[/dim]")
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Check for approval interrupt
        state = graph.get_state(config)
        if state.next:
            draft = result.get("draft_output", "")
            console.print(Panel(
                f"{draft}\n\n[bold yellow]This action requires approval. "
                f"Type 'approve' or 'reject'.[/bold yellow]",
                title="Approval Required",
                border_style="yellow",
            ))
        else:
            output = result.get("final_output", "")
            if not output:
                # Fallback to last AI message
                for msg in reversed(result.get("messages", [])):
                    if hasattr(msg, "content") and msg.content:
                        output = msg.content
                        break
            console.print(Panel(output or "Done.", title="Agent", border_style="green"))


if __name__ == "__main__":
    asyncio.run(main())
