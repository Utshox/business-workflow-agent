"""Shared evaluation metrics for both the agent benchmark and fine-tuning eval."""

from __future__ import annotations

import json


def json_validity_rate(outputs: list[str]) -> float:
    """Fraction of outputs that are valid JSON."""
    valid = 0
    for out in outputs:
        try:
            json.loads(out)
            valid += 1
        except (json.JSONDecodeError, TypeError):
            pass
    return valid / max(len(outputs), 1)


def classification_accuracy(predictions: list[str], references: list[str]) -> float:
    """Exact match accuracy for workflow type classification."""
    correct = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return correct / max(len(predictions), 1)


def action_overlap(pred_action: str, ref_action: str) -> float:
    """Word-level overlap between predicted and reference action strings."""
    pred_words = set(pred_action.lower().split())
    ref_words = set(ref_action.lower().split())
    if not ref_words:
        return 0.0
    return len(pred_words & ref_words) / len(ref_words)
