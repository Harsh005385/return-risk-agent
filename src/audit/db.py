"""SQLite audit trail for agent steps and verdicts (hash-chained)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.paths import AUDIT_DB

GENESIS = "GENESIS"


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case_id = Column(String(128), index=True)
    customer_id = Column(String(128), index=True)
    event_type = Column(String(64))  # tool_call | reasoning | verdict | score
    payload_json = Column(Text)
    prev_hash = Column(String(64), nullable=True)
    record_hash = Column(String(64), nullable=True)


_engine = None
_SessionLocal = None


def compute_record_hash(
    prev_hash: str,
    created_at: datetime,
    case_id: str,
    event_type: str,
    payload_json: str,
) -> str:
    ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    blob = f"{prev_hash}{ts.isoformat()}{case_id}{event_type}{payload_json}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _ensure_hash_columns(engine) -> None:
    insp = inspect(engine)
    if not insp.has_table("audit_events"):
        return
    cols = {c["name"] for c in insp.get_columns("audit_events")}
    with engine.begin() as conn:
        if "prev_hash" not in cols:
            conn.execute(text("ALTER TABLE audit_events ADD COLUMN prev_hash VARCHAR(64)"))
        if "record_hash" not in cols:
            conn.execute(text("ALTER TABLE audit_events ADD COLUMN record_hash VARCHAR(64)"))


def _backfill_hashes(session: Session) -> None:
    rows = session.query(AuditEvent).order_by(AuditEvent.id.asc()).all()
    prev = GENESIS
    for row in rows:
        if row.record_hash:
            prev = row.record_hash
            continue
        payload = row.payload_json or "{}"
        ts = row.created_at or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        row.prev_hash = prev
        row.record_hash = compute_record_hash(
            prev, ts, row.case_id or "", row.event_type or "", payload
        )
        prev = row.record_hash
    session.commit()


def init_db(db_path: Path | None = None):
    global _engine, _SessionLocal
    path = db_path or AUDIT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(_engine)
    _ensure_hash_columns(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    session = _SessionLocal()
    try:
        _backfill_hashes(session)
    finally:
        session.close()
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def log_event(case_id: str, customer_id: str, event_type: str, payload: dict) -> int:
    session = get_session()
    try:
        last = session.query(AuditEvent).order_by(AuditEvent.id.desc()).first()
        prev_hash = last.record_hash if last and last.record_hash else GENESIS
        created_at = datetime.now(timezone.utc)
        payload_json = json.dumps(payload, default=str)
        record_hash = compute_record_hash(
            prev_hash, created_at, case_id, event_type, payload_json
        )
        row = AuditEvent(
            case_id=case_id,
            customer_id=customer_id,
            event_type=event_type,
            payload_json=payload_json,
            created_at=created_at,
            prev_hash=prev_hash,
            record_hash=record_hash,
        )
        session.add(row)
        session.commit()
        return int(row.id)
    finally:
        session.close()


def list_events(case_id: str | None = None, limit: int = 50) -> list[dict]:
    session = get_session()
    try:
        q = session.query(AuditEvent).order_by(AuditEvent.id.desc())
        if case_id:
            q = q.filter(AuditEvent.case_id == case_id)
        rows = q.limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "case_id": r.case_id,
                "customer_id": r.customer_id,
                "event_type": r.event_type,
                "payload": json.loads(r.payload_json or "{}"),
                "prev_hash": r.prev_hash,
                "record_hash": r.record_hash,
            }
            for r in rows
        ]
    finally:
        session.close()


def verify_chain_integrity() -> dict:
    session = get_session()
    try:
        rows = session.query(AuditEvent).order_by(AuditEvent.id.asc()).all()
        prev_hash = GENESIS
        checked = 0
        for row in rows:
            if not row.record_hash:
                return {
                    "valid": False,
                    "checked": checked,
                    "first_break_id": row.id,
                }
            if row.prev_hash != prev_hash:
                return {
                    "valid": False,
                    "checked": checked,
                    "first_break_id": row.id,
                }
            payload = row.payload_json or "{}"
            ts = row.created_at or datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            expected = compute_record_hash(
                prev_hash,
                ts,
                row.case_id or "",
                row.event_type or "",
                payload,
            )
            if expected != row.record_hash:
                return {
                    "valid": False,
                    "checked": checked,
                    "first_break_id": row.id,
                }
            prev_hash = row.record_hash
            checked += 1
        return {"valid": True, "checked": checked, "first_break_id": None}
    finally:
        session.close()
