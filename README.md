# Return Risk Agent

Razorpay AI Buildathon, Track 02: AI Risk Manager

**Defense-only:** recommends `ALLOW` / `MONITOR` / `REVIEW` / `HOLD`. Never auto-blocks, charges, bans, or takes an irreversible action against a customer. Dashboard metrics are **DEMO DATA** (synthetic).

## Overview

Agentic return-abuse risk scorer for e-commerce refunds. An agent gathers evidence with tools (order context, customer history, ensemble score, SHAP drivers, behaviour patterns, linked accounts). A deterministic policy engine maps evidence to an action - the LLM never chooses the final action. Every step is written to a SHA-256 hash-chained SQLite audit log. A Dash console and FastAPI expose scoring, analytics, audit verify, and optional Razorpay test-mode orders.

## Quick start

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set PYTHONPATH=.
python -m src.models.train_ensemble --sample-n 20000
python dashboard/app.py
```

- Dashboard: http://127.0.0.1:8050
- Optional API: `uvicorn src.api.main:app --reload --port 8000` then http://127.0.0.1:8000/docs

Copy `.env.example` to `.env` for optional `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `RAZORPAY_KEY_ID`, and `RAZORPAY_KEY_SECRET`. Without an LLM key, Thorough review uses the rules-based planner.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/architecture.png`](docs/architecture.png).

```
Case -> agent tools -> policy engine (ALLOW | MONITOR | REVIEW | HOLD)
    -> hash-chained audit -> Dash / FastAPI
```

## Data and model

Training CSVs live in `data/raw/`. Split is chronological, not random. Metrics are synthetic held-out results, not production claims. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). Train locally; model artifacts are not committed.

## Tests

```bash
set PYTHONPATH=.
pytest -q
```

## License

MIT. Demo and research prototype on synthetic datasets. Not a production fraud decisioning system.
