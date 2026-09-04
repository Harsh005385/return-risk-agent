from __future__ import annotations

from src.agent.scenarios import get_scenario, list_scenarios


def test_scenarios_present():
    ids = {s["id"] for s in list_scenarios()}
    assert ids == {"safe", "risky", "ambiguous", "syndicate"}
    for sid in ids:
        case = get_scenario(sid)
        assert case is not None
        assert "customer_id" in case
        assert "features" in case
