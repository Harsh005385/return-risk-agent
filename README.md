# Return Risk Agent

Razorpay AI Buildathon · Track 02: AI Risk Manager

**Live demo:** [https://return-risk-agent.onrender.com](https://return-risk-agent.onrender.com)

> Free Render instances sleep when idle. First open after sleep can take about one minute, then the Risk console loads.

**Defense-only:** recommends `ALLOW` / `MONITOR` / `REVIEW` / `HOLD`. Never auto-blocks, charges, bans, or takes an irreversible action against a customer. Dashboard metrics are **DEMO DATA** (synthetic).

---

## Problem

Return abuse (wardrobing, serial returners, linked accounts) costs merchants money. A single opaque risk score is hard to trust. Analysts need clear reasons and an audit trail before they hold or approve a refund.

## What this project does

1. Takes a return / refund case (from the demo dataset or manual entry).
2. Investigates with multiple checks: order context, customer history, ML risk score, top drivers, behaviour patterns, linked accounts.
3. Recommends one action: **ALLOW**, **MONITOR**, **REVIEW**, or **HOLD**.
4. Writes every step to a tamper-evident audit log so a reviewer can verify integrity.

A human still makes the final call. The system only advises.

## How to try the live demo

1. Open [https://return-risk-agent.onrender.com](https://return-risk-agent.onrender.com) (wait if the service is waking up).
2. Use Quick presets: **Safe**, **Risky**, **Ambiguous**, **Syndicate**.
3. Or use **Dataset Lookup** / **Manual Entry**, then score a case.
4. Open **Audit trail** and click **Verify Integrity**.

## Screenshots

Live GUI captures in [`docs/screenshots/`](docs/screenshots/).

---

## Stack

| Layer | Choice |
|-------|--------|
| Desk UI | Dash + Plotly + dash-bootstrap-components |
| API | FastAPI (`src/api/main.py`) |
| Agent | Multi-tool orchestrator (Quick parallel / Thorough sequential) |
| Scoring | Soft-voting ensemble (RandomForest + XGBoost + GradientBoosting) |
| Explainability | SHAP (TreeExplainer on RF member) |
| Policy | Deterministic rules in `src/policy/policy_engine.py` (no LLM) |
| Audit | SQLite + SHA-256 `prev_hash` / `record_hash` chain |
| Optional LLM | OpenAI-compatible chat API via env vars (planner only) |
| Optional payments touchpoint | Razorpay test-mode order create |
| Hosting | Render Web Service + gunicorn (`dashboard.app:server`) |

## Architecture

```
Case
  -> Agent tools (history, score, SHAP, pattern, order, linked accounts)
  -> Policy engine -> ALLOW | MONITOR | REVIEW | HOLD
  -> Hash-chained audit DB
  -> Dash console / FastAPI
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · diagram: [`docs/architecture.png`](docs/architecture.png)

## Design choices

| Topic | Choice | Why |
|-------|--------|-----|
| Final action | Policy engine only | LLM may plan tools; never emits ALLOW/HOLD |
| Train/test split | Chronological | Avoid future leakage on time-ordered returns |
| Quick vs Thorough | Parallel vs sequential (max 4 steps) | Desk speed vs analyst depth |
| Linked accounts | Full fingerprint + behaviour | Avoid false rings from coarse attributes |
| Metrics | Disclosed DEMO DATA | Synthetic separability is not a production claim |

## Data and model

- Training CSVs: `data/raw/`
- Chronological split (not random)
- Model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- Artifacts (`.joblib`, processed tables, audit DB) are **not** committed; train at build/run time

## Project layout

```
src/           agent, policy, models, audit, api, features
dashboard/     Dash Risk console
tests/         pytest suite
data/raw/      synthetic training CSVs
docs/          architecture, model card, screenshots
scripts/       train smoke, ablation, architecture diagram
```

## Quick start (local)

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set PYTHONPATH=.
python -m src.models.train_ensemble --sample-n 20000
python dashboard/app.py
```

- Local dashboard: http://127.0.0.1:8050
- Optional API: `uvicorn src.api.main:app --reload --port 8000` then http://127.0.0.1:8000/docs
- Live demo: [https://return-risk-agent.onrender.com](https://return-risk-agent.onrender.com)

Copy `.env.example` to `.env` for optional:

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`

Without an LLM key, Thorough review uses the rules-based planner (same tools + same policy).

## Tests

```bash
set PYTHONPATH=.
pytest -q
```

## Deploy notes (Render)

- Start: `PYTHONPATH=. gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`
- Build should train the model (artifacts are gitignored)
- See `render.yaml` and `runtime.txt` (Python 3.12)

---

## License

MIT. Demo and research prototype on synthetic datasets. Not a production fraud decisioning system.
