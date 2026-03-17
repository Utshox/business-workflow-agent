"""LoRA fine-tuning script using PEFT + TRL on an open-source model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str = "fine_tuning/configs/lora_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_data(path: str = "fine_tuning/data/train_alpaca.json") -> Dataset:
    with open(path) as f:
        data = json.load(f)

    def format_example(ex: dict) -> str:
        return (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )

    formatted = [{"text": format_example(ex)} for ex in data]
    return Dataset.from_list(formatted)


def train(config_path: str = "fine_tuning/configs/lora_config.yaml"):
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    print(f"Loading base model: {model_cfg['base_model']}")

    # Quantization config for 4-bit training
    bnb_config = None
    if model_cfg.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_training_data()
    print(f"Training on {len(dataset)} examples")

    # Training arguments
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=model_cfg["max_length"],
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save the LoRA adapter
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")

    return str(adapter_path)


if __name__ == "__main__":
    train()
