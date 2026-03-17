# Business Workflow Agent

Multi-step business workflow agent with tool integration, human-in-the-loop approval, and fine-tuning evaluation track.

## Architecture

```
User Request
    │
    ▼
┌──────────┐    ┌─────────────┐    ┌──────────┐
│ Classify  │───▶│ Vector Memory│───▶│ Execute  │
│ Workflow  │    │  Retrieval   │    │ Workflow │
└──────────┘    └─────────────┘    └────┬─────┘
                                        │
                               ┌────────┼────────┐
                               ▼        ▼        ▼
                          ┌────────┐ ┌──────┐ ┌──────────┐
                          │ Tools  │ │Approve│ │ Finalize │
                          │ (6+)   │ │ Gate  │ │ & Store  │
                          └────────┘ └──────┘ └──────────┘
```

**Workflows:** Ticket Triage | Report Drafting | Data Lookup

**Stack:** Python 3.12 · LangGraph · LangChain · ChromaDB · FastAPI · Docker

## Quick Start

```bash
# 1. Clone and setup
cp .env.example .env        # Add your OPENAI_API_KEY
pip install -e .

# 2. Run the interactive CLI
python run_cli.py

# 3. Or start the API server
uvicorn src.api.server:app --reload

# 4. Run tests
pip install -e ".[dev]"
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/workflow/run` | Submit a workflow request |
| POST | `/workflow/{thread_id}/approve` | Approve/reject a paused workflow |
| GET | `/approvals/pending` | List pending approvals |
| GET | `/health` | Health check |

**Example:**

```bash
curl -X POST http://localhost:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Customer C001 cannot log in and their entire team is blocked"}'
```

## Workflows

### Ticket Triage
Classifies support tickets, looks up customer data, searches knowledge base, assigns priority, and creates tickets. High-priority tickets require human approval.

### Report Drafting
Queries metrics databases, generates formatted report sections, and compiles professional draft reports.

### Data Lookup
Retrieves customer information, usage metrics, and knowledge base articles based on natural language queries.

## Tools

| Tool | Purpose |
|------|---------|
| `lookup_customer` | Customer data (tier, ARR, health score) |
| `search_knowledge_base` | Internal KB article search |
| `query_metrics_db` | Usage metrics and trends |
| `create_ticket` | Create support tickets |
| `send_notification` | Send email/Slack notifications |
| `generate_report_section` | Format report sections |

## Fine-Tuning Track

Compare a prompt-only baseline against a LoRA fine-tuned open-source model:

```bash
# Install fine-tuning deps
pip install -e ".[fine-tuning]"

# 1. Generate synthetic training data
python fine_tuning/data/generate_dataset.py

# 2. Train LoRA adapter (requires GPU)
python fine_tuning/train_lora.py

# 3. Evaluate baseline vs fine-tuned
python fine_tuning/evaluate.py
```

**Metrics:** JSON validity, workflow classification accuracy/F1, action match rate.

**Config:** `fine_tuning/configs/lora_config.yaml` — adjust base model, LoRA rank, training params.

## Agent Benchmark

```bash
python eval/benchmark.py
```

Runs 5 end-to-end scenarios testing classification accuracy, tool usage, output quality, and latency.

## Docker

```bash
# API server
docker compose up agent-api

# Fine-tuning (requires nvidia-docker)
docker compose --profile training up fine-tune
```

## Kubernetes

```bash
# Create secret first
kubectl create secret generic workflow-agent-secrets \
  --from-literal=OPENAI_API_KEY=sk-...

kubectl apply -f k8s/
```

## Project Structure

```
├── src/
│   ├── agent/          # LangGraph workflow (graph, nodes, state, tools)
│   ├── memory/         # ChromaDB vector memory store
│   ├── approval/       # Human-in-the-loop approval manager
│   └── api/            # FastAPI server
├── fine_tuning/
│   ├── data/           # Synthetic dataset generator
│   ├── configs/        # LoRA training config
│   ├── train_lora.py   # PEFT/LoRA training script
│   └── evaluate.py     # Baseline vs fine-tuned evaluation
├── eval/               # Agent benchmarks and metrics
├── tests/              # Unit tests
├── k8s/                # Kubernetes manifests
├── Dockerfile          # Multi-stage (api + fine-tune)
└── docker-compose.yml
```

## Author

- [Utshox](https://github.com/Utshox)
