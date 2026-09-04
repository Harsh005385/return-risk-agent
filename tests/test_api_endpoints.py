"""FastAPI endpoint smoke tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.db import init_db, verify_chain_integrity


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api_audit.db"
    monkeypatch.setattr("src.paths.AUDIT_DB", db_path)
    import src.audit.db as audit_mod

    audit_mod._engine = None
    audit_mod._SessionLocal = None
    init_db(db_path)
    try:
        from src.bootstrap import warm_runtime

        warm_runtime()
    except Exception:
        pass
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_scenarios_list(client):
    r = client.get("/scenarios")
    assert r.status_code == 200
    assert len(r.json().get("scenarios") or []) >= 4


def test_scenario_safe(client):
    r = client.post("/scenarios/safe")
    assert r.status_code == 200
    body = r.json()
    assert body.get("action") in {"ALLOW", "MONITOR"}


def test_scenario_syndicate(client):
    r = client.post("/scenarios/syndicate")
    assert r.status_code == 200
    body = r.json()
    assert body.get("action") in {"HOLD", "REVIEW"}


def test_audit_verify_fresh_db(client):
    r = client.get("/audit/verify")
    assert r.status_code == 200
    body = r.json()
    assert body.get("valid") is True
