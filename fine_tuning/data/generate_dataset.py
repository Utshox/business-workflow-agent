"""Generate synthetic training data for the fine-tuning track.

Creates (input, output) pairs for workflow classification and triage reasoning,
saved in formats suitable for both prompt-only evaluation and LoRA training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TICKET_TEMPLATES = [
    {
        "input": "Customer {cid} reports login failures since this morning. They are on the {tier} tier.",
        "output": {
            "workflow_type": "ticket_triage",
            "priority": "{priority}",
            "reasoning": "Login failures indicate access issues. {tier_reason}",
            "action": "Search KB for auth articles, check customer health score, create {priority} priority ticket.",
        },
    },
    {
        "input": "Need a weekly usage report for customer {cid} covering API calls and error rates.",
        "output": {
            "workflow_type": "report_draft",
            "sections": ["API Usage Summary", "Error Rate Analysis", "Recommendations"],
            "reasoning": "Report request requires metrics lookup and structured formatting.",
            "action": "Query metrics DB for api_calls and error_rate, generate report sections.",
        },
    },
    {
        "input": "What is the current health score and ARR for customer {cid}?",
        "output": {
            "workflow_type": "data_lookup",
            "reasoning": "Direct data retrieval request for customer metrics.",
            "action": "Look up customer {cid} and return health_score and ARR.",
        },
    },
    {
        "input": "Customer {cid} on {tier} plan is experiencing API rate limiting errors. Their traffic has doubled this week.",
        "output": {
            "workflow_type": "ticket_triage",
            "priority": "{priority}",
            "reasoning": "Rate limiting with traffic spike suggests capacity issue. {tier_reason}",
            "action": "Search KB for rate-limit policy, query metrics for api_calls, create ticket.",
        },
    },
    {
        "input": "Generate a monthly churn risk analysis report for all enterprise customers.",
        "output": {
            "workflow_type": "report_draft",
            "sections": ["At-Risk Customers", "Health Score Trends", "Recommended Actions"],
            "reasoning": "Analytical report requiring aggregation of customer health data.",
            "action": "Query metrics for health_score trends, lookup enterprise customers, generate report.",
        },
    },
    {
        "input": "Customer {cid} wants to export their data but says the option is grayed out.",
        "output": {
            "workflow_type": "ticket_triage",
            "priority": "{priority}",
            "reasoning": "Feature access issue, likely tier-related. {tier_reason}",
            "action": "Search KB for data export guide, check customer tier, create ticket.",
        },
    },
]

CUSTOMER_IDS = ["C001", "C002", "C003", "C004", "C005"]
TIERS = [
    ("enterprise", "Enterprise tier warrants elevated priority.", "high"),
    ("growth", "Growth tier uses standard priority handling.", "medium"),
    ("starter", "Starter tier uses standard priority handling.", "low"),
]


def generate_examples(n: int = 200) -> list[dict]:
    examples = []
    for _ in range(n):
        template = random.choice(TICKET_TEMPLATES)
        cid = random.choice(CUSTOMER_IDS)
        tier_name, tier_reason, base_priority = random.choice(TIERS)

        # Occasionally escalate priority
        priority = base_priority
        if random.random() < 0.2:
            priority = "critical"

        input_text = template["input"].format(cid=cid, tier=tier_name)
        output_data = json.loads(
            json.dumps(template["output"])
            .replace("{cid}", cid)
            .replace("{tier}", tier_name)
            .replace("{priority}", priority)
            .replace("{tier_reason}", tier_reason)
        )

        examples.append({
            "input": input_text,
            "expected_output": json.dumps(output_data),
            "metadata": {"customer_id": cid, "tier": tier_name},
        })

    return examples


def save_dataset(examples: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full dataset
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(examples, f, indent=2)

    # Train/test split (80/20)
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train, test = examples[:split], examples[split:]

    with open(output_dir / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(output_dir / "test.json", "w") as f:
        json.dump(test, f, indent=2)

    # Alpaca-style format for LoRA training
    alpaca = []
    for ex in train:
        alpaca.append({
            "instruction": "Classify this business request and determine the appropriate workflow, priority, and actions.",
            "input": ex["input"],
            "output": ex["expected_output"],
        })
    with open(output_dir / "train_alpaca.json", "w") as f:
        json.dump(alpaca, f, indent=2)

    print(f"Generated {len(examples)} examples: {len(train)} train, {len(test)} test")


if __name__ == "__main__":
    random.seed(42)
    data = generate_examples(200)
    save_dataset(data, Path(__file__).parent)
