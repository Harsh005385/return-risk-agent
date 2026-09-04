"""Smoke: train small sample, score scenarios."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.agent.orchestrator import run_agent
from src.agent.scenarios import get_scenario
from src.bootstrap import warm_runtime
from src.models.train_ensemble import train


def main():
    metrics = train(sample_n=5000)
    (ROOT / "data" / "processed" / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    warm_runtime()
    for name in ("safe", "risky", "syndicate"):
        verdict = run_agent(get_scenario(name))
        print(name, verdict["action"], round(verdict["risk_score"], 3))


if __name__ == "__main__":
    main()
