"""Evaluate prompt-only baseline vs LoRA fine-tuned model.

Produces a comparison report with metrics: accuracy, F1, JSON validity, action match.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(path: str = "fine_tuning/data/test.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def format_prompt(input_text: str) -> str:
    return (
        "### Instruction:\n"
        "Classify this business request and determine the appropriate workflow, "
        "priority, and actions.\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json_output(text: str) -> dict | None:
    """Try to extract JSON from model output."""
    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx != -1:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                continue
    return None


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """Compute evaluation metrics."""
    # JSON validity
    json_valid = sum(1 for p in predictions if p is not None) / len(predictions)

    # Workflow type accuracy
    pred_types = []
    ref_types = []
    for p, r in zip(predictions, references):
        ref_types.append(r.get("workflow_type", "unknown"))
        if p:
            pred_types.append(p.get("workflow_type", "unknown"))
        else:
            pred_types.append("invalid")

    type_accuracy = accuracy_score(ref_types, pred_types)
    type_f1 = f1_score(ref_types, pred_types, average="weighted", zero_division=0)

    # Action match (fuzzy: check if key action verbs appear)
    action_matches = 0
    for p, r in zip(predictions, references):
        if p and "action" in r and "action" in p:
            ref_words = set(r["action"].lower().split())
            pred_words = set(p["action"].lower().split())
            overlap = len(ref_words & pred_words) / max(len(ref_words), 1)
            if overlap > 0.3:
                action_matches += 1
    action_match_rate = action_matches / len(predictions)

    return {
        "json_validity": round(json_valid, 3),
        "workflow_type_accuracy": round(type_accuracy, 3),
        "workflow_type_f1": round(type_f1, 3),
        "action_match_rate": round(action_match_rate, 3),
    }


def evaluate_model(model, tokenizer, test_data: list[dict], label: str) -> dict:
    """Run evaluation on a model and return metrics."""
    print(f"\nEvaluating: {label}")
    predictions = []
    references = []

    for i, example in enumerate(test_data):
        prompt = format_prompt(example["input"])
        output = generate(model, tokenizer, prompt)
        parsed = parse_json_output(output)
        predictions.append(parsed)
        references.append(json.loads(example["expected_output"]))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_data)}")

    metrics = compute_metrics(predictions, references)
    print(f"  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def run_evaluation(
    config_path: str = "fine_tuning/configs/lora_config.yaml",
    adapter_path: str = "fine_tuning/output/lora_adapter",
):
    cfg = yaml.safe_load(open(config_path))
    base_model_name = cfg["model"]["base_model"]
    test_data = load_test_data(cfg["evaluation"]["test_file"])

    print(f"Loaded {len(test_data)} test examples")
    print(f"Base model: {base_model_name}")

    # Load base model (prompt-only baseline)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    baseline_metrics = evaluate_model(base_model, tokenizer, test_data, "Prompt-Only Baseline")

    # Load LoRA fine-tuned model
    if Path(adapter_path).exists():
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        ft_metrics = evaluate_model(ft_model, tokenizer, test_data, "LoRA Fine-Tuned")
    else:
        print(f"\nAdapter not found at {adapter_path}. Run train_lora.py first.")
        ft_metrics = None

    # Generate comparison report
    report = {
        "base_model": base_model_name,
        "test_examples": len(test_data),
        "baseline": baseline_metrics,
        "fine_tuned": ft_metrics,
    }

    if ft_metrics:
        report["improvement"] = {
            k: round(ft_metrics[k] - baseline_metrics[k], 3)
            for k in baseline_metrics
        }

    output_path = Path("fine_tuning/output/eval_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'Fine-Tuned':>12} {'Delta':>10}")
    print("-" * 60)
    for metric in baseline_metrics:
        base_val = baseline_metrics[metric]
        ft_val = ft_metrics[metric] if ft_metrics else "N/A"
        delta = f"{ft_metrics[metric] - base_val:+.3f}" if ft_metrics else "N/A"
        ft_display = f"{ft_val:.3f}" if isinstance(ft_val, float) else ft_val
        print(f"{metric:<25} {base_val:>12.3f} {ft_display:>12} {delta:>10}")
    print("=" * 60)
    print(f"\nReport saved to {output_path}")

    return report


if __name__ == "__main__":
    run_evaluation()
