"""Tamper-evident audit hash chain tests."""
from __future__ import annotations

from sqlalchemy import text

from src.audit.db import init_db, log_event, verify_chain_integrity


def test_audit_chain_valid_and_tamper_detected(tmp_path):
    db_path = tmp_path / "audit_test.db"
    init_db(db_path)

    log_event("CASE-1", "CUST-1", "reasoning", {"step": 1})
    log_event("CASE-1", "CUST-1", "tool_call", {"tool": "get_customer_history"})
    log_event("CASE-1", "CUST-1", "verdict", {"action": "ALLOW"})

    result = verify_chain_integrity()
    assert result["valid"] is True
    assert result["checked"] == 3
    assert result["first_break_id"] is None

    from src.audit.db import get_session

    session = get_session()
    try:
        session.execute(
            text("UPDATE audit_events SET payload_json = :p WHERE id = 2"),
            {"p": '{"tool": "tampered"}'},
        )
        session.commit()
    finally:
        session.close()

    broken = verify_chain_integrity()
    assert broken["valid"] is False
    assert broken["first_break_id"] == 2
