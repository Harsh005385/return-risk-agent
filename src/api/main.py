"""FastAPI surface: /score, /agent-trace, /audit, /metrics, /scenarios."""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.orchestrator import run_agent
from src.audit.db import init_db, list_events, log_event, verify_chain_integrity
from src.env import load_env
from src.paths import METRICS_PATH

load_env()

app = FastAPI(
    title="Return Risk Agent",
    description=(
        "Defense-only agentic return-abuse risk scorer. "
        "Recommends ALLOW/MONITOR/REVIEW/HOLD for human review - never auto-punishes."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rolling latency samples for /score (seconds)
_LATENCY_SAMPLES: deque[float] = deque(maxlen=500)
_LAST_VERDICTS: deque[dict] = deque(maxlen=100)


class ScoreRequest(BaseModel):
    order_id: str | None = None
    customer_id: str
    case_id: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    force_syndicate_check: bool = False
    # Allow passing raw case fields used by feature_matrix
    extra: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def _startup():
    init_db()
    try:
        from src.bootstrap import warm_runtime

        warm_runtime()
    except Exception:
        pass  # Train model first if artifacts missing


@app.get("/")
def root():
    return {
        "service": "Return Risk Agent",
        "defense_only": True,
        "demo_data": True,
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /score",
            "GET /agent-trace?case_id=",
            "GET /audit",
            "GET /metrics",
            "GET /scenarios",
            "POST /scenarios/{name}",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "defense_only": True, "demo_data": True}


@app.post("/score")
def score(req: ScoreRequest):
    case = {
        "order_id": req.order_id or req.case_id or f"CASE-{req.customer_id}",
        "customer_id": req.customer_id,
        "force_syndicate_check": req.force_syndicate_check,
        **req.extra,
        "features": req.features or req.extra,
    }
    t0 = time.perf_counter()
    try:
        verdict = run_agent(case)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LATENCY_SAMPLES.append(elapsed_ms)
    verdict["latency_ms"] = elapsed_ms
    _LAST_VERDICTS.appendleft(verdict)
    log_event(case["order_id"], req.customer_id, "score", {"latency_ms": elapsed_ms})
    return verdict


@app.get("/agent-trace")
def agent_trace(case_id: str = Query(...)):
    events = list_events(case_id=case_id, limit=100)
    return {"case_id": case_id, "events": events}


@app.get("/audit")
def audit(limit: int = Query(50, ge=1, le=500)):
    return {"events": list_events(limit=limit)}


@app.get("/audit/verify")
def audit_verify():
    return verify_chain_integrity()


@app.get("/metrics")
def metrics():
    model_metrics = {}
    if METRICS_PATH.exists():
        model_metrics = json.loads(METRICS_PATH.read_text())
    samples = sorted(_LATENCY_SAMPLES)
    latency = {}
    if samples:
        def pct(p: float) -> float:
            idx = min(len(samples) - 1, max(0, int(round(p * (len(samples) - 1)))))
            return float(samples[idx])

        latency = {
            "n": len(samples),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "mean_ms": float(sum(samples) / len(samples)),
        }
    return {
        "model": model_metrics,
        "latency": latency,
        "recent_verdicts": list(_LAST_VERDICTS)[:20],
        "demo_data": True,
    }


@app.get("/scenarios")
def scenarios():
    from src.agent.scenarios import list_scenarios

    return {"scenarios": list_scenarios()}


@app.post("/scenarios/{name}")
def run_scenario(name: str):
    from src.agent.scenarios import get_scenario

    case = get_scenario(name)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {name}")
    t0 = time.perf_counter()
    verdict = run_agent(case)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LATENCY_SAMPLES.append(elapsed_ms)
    verdict["latency_ms"] = elapsed_ms
    verdict["scenario"] = name
    _LAST_VERDICTS.appendleft(verdict)
    return verdict
