"""End-to-end benchmark for the workflow agent.

Tests the full LangGraph agent pipeline with mock scenarios and measures:
- Workflow classification accuracy
- Tool usage correctness
- Output quality
- Latency
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from src.agent.graph import compile_graph

load_dotenv()


@dataclass
class BenchmarkCase:
    name: str
    input_message: str
    expected_workflow: str
    expected_tools: list[str]
    quality_check: str  # substring expected in output


BENCHMARK_CASES = [
    BenchmarkCase(
        name="ticket_triage_enterprise",
        input_message="Customer C001 is reporting that they cannot access the dashboard since 9am. This is blocking their entire team.",
        expected_workflow="ticket_triage",
        expected_tools=["lookup_customer", "search_knowledge_base"],
        quality_check="priority",
    ),
    BenchmarkCase(
        name="ticket_triage_growth",
        input_message="Customer C002 has a question about upgrading their API rate limits.",
        expected_workflow="ticket_triage",
        expected_tools=["search_knowledge_base"],
        quality_check="rate",
    ),
    BenchmarkCase(
        name="report_generation",
        input_message="Generate a usage report for customer C003 covering the last 30 days of API activity.",
        expected_workflow="report_draft",
        expected_tools=["query_metrics_db"],
        quality_check="report",
    ),
    BenchmarkCase(
        name="data_lookup_simple",
        input_message="What tier is customer C001 on and who is their CSM?",
        expected_workflow="data_lookup",
        expected_tools=["lookup_customer"],
        quality_check="enterprise",
    ),
    BenchmarkCase(
        name="data_lookup_metrics",
        input_message="Show me the error rate trend for customer C002 over the past week.",
        expected_workflow="data_lookup",
        expected_tools=["query_metrics_db"],
        quality_check="trend",
    ),
]


async def run_benchmark():
    graph = compile_graph()
    results = []

    print("Running agent benchmark...\n")

    for case in BENCHMARK_CASES:
        print(f"  [{case.name}] ", end="", flush=True)
        start = time.time()

        try:
            config = {"configurable": {"thread_id": f"bench_{case.name}"}}
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": case.input_message}]},
                config=config,
            )
            elapsed = time.time() - start

            # Check workflow classification
            wf_type = result.get("workflow_type")
            wf_correct = wf_type and wf_type.value == case.expected_workflow

            # Check tool usage (from messages)
            tools_used = set()
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls"):
                    for tc in msg.tool_calls:
                        tools_used.add(tc["name"])

            tools_correct = all(t in tools_used for t in case.expected_tools)

            # Check output quality
            output = result.get("final_output", "") or ""
            quality_ok = case.quality_check.lower() in output.lower()

            status = "PASS" if (wf_correct and tools_correct) else "FAIL"
            print(f"{status} ({elapsed:.1f}s)")

            results.append({
                "name": case.name,
                "passed": status == "PASS",
                "workflow_correct": wf_correct,
                "tools_correct": tools_correct,
                "quality_check_passed": quality_ok,
                "tools_used": list(tools_used),
                "latency_seconds": round(elapsed, 2),
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append({
                "name": case.name,
                "passed": False,
                "error": str(e),
                "latency_seconds": round(elapsed, 2),
            })

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / total

    print(f"\nResults: {passed}/{total} passed | Avg latency: {avg_latency:.1f}s")

    report = {"cases": results, "passed": passed, "total": total, "avg_latency": avg_latency}
    with open("eval/benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to eval/benchmark_report.json")

    return report


if __name__ == "__main__":
    asyncio.run(run_benchmark())
