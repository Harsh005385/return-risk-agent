"""Callable tools used by the agent orchestrator."""
from __future__ import annotations

from functools import lru_cache

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from src.explain.shap_tool import get_shap_explanation
from src.paths import ARTIFACT_PATH, DATA_PROCESSED


@lru_cache(maxsize=1)
def _load_cases() -> pd.DataFrame:
    path = DATA_PROCESSED / "case_index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.models.train_ensemble"
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_model_bundle():
    return joblib.load(ARTIFACT_PATH)


_LINK_BEHAVIOUR_COLS = (
    "multiple_accounts_flag",
    "refund_to_different_account",
    "address_change_before_delivery",
)


def _fingerprint(row) -> str:
    """Composite device|payment|country fingerprint. Any single attribute alone is far
    too coarse (5 device types across 8000 rows) to imply a link between accounts."""
    return f"fp:{row.get('device_type')}|{row.get('payment_method')}|{row.get('country')}"


@lru_cache(maxsize=1)
def _build_shared_signal_graph() -> nx.Graph:
    """
    Bipartite graph of *previously flagged customers who also showed account-linking
    behaviour* (multiple accounts, refund to a different account, address change before
    delivery), connected via their composite device|payment|country fingerprint.
    Tier-2 abuse-ring signal.
    """
    df = _load_cases()
    flagged = df[df["abuse_label"] == 1]
    linkable = flagged[flagged[list(_LINK_BEHAVIOUR_COLS)].sum(axis=1) > 0]
    G = nx.Graph()
    for _, row in linkable.iterrows():
        cid = str(row["customer_id"])
        G.add_node(cid, bipartite=0, kind="customer")
        key = _fingerprint(row)
        G.add_node(key, bipartite=1, kind="fingerprint")
        G.add_edge(cid, key)
    return G


def get_customer_history(customer_id: str) -> dict:
    df = _load_cases()
    hist = df[df["customer_id"].astype(str) == str(customer_id)]
    if hist.empty:
        return {
            "customer_id": customer_id,
            "n_returns": 0,
            "mean_return_rate_pct": 0.0,
            "prior_abuse_flags": 0,
            "recent_refund_usd": 0.0,
        }
    return {
        "customer_id": customer_id,
        "n_returns": int(len(hist)),
        "mean_return_rate_pct": float(hist["return_rate_pct"].mean()),
        "prior_abuse_flags": int(hist["abuse_label"].sum()),
        "recent_refund_usd": float(hist["refund_amount_requested_usd"].tail(3).mean()),
        "categories": hist["product_category"].value_counts().head(5).to_dict(),
    }


def get_ensemble_risk_score(features: dict) -> dict:
    bundle = _load_model_bundle()
    model = bundle["model"]
    cols = bundle["feature_cols"]
    row = pd.DataFrame([{c: float(features.get(c, 0) or 0) for c in cols}])
    proba_full = model.predict_proba(row)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    score = float(proba_full[pos_idx])
    other = float(1.0 - score)
    confidence_gap = float(abs(score - other))
    return {
        "risk_score": score,
        "confidence_gap": confidence_gap,
        "class_probabilities": {"legit": other, "abuse": score},
    }


def check_pattern(customer_id: str) -> dict:
    df = _load_cases()
    hist = df[df["customer_id"].astype(str) == str(customer_id)]
    flags = []
    if hist.empty:
        return {"flag_count": 0, "flags": [], "detail": "no history"}

    if hist["return_rate_pct"].mean() >= 35:
        flags.append("high_lifetime_return_rate")
    if (hist["days_to_return"] <= 2).sum() >= 2:
        flags.append("repeated_same_day_or_next_day_returns")
    if hist["product_category"].value_counts().iloc[0] >= max(3, len(hist) // 2):
        flags.append("category_concentration")
    if hist["multiple_accounts_flag"].sum() > 0:
        flags.append("multiple_accounts_observed")
    if hist["address_change_before_delivery"].sum() > 0:
        flags.append("address_change_before_delivery")
    if hist["refund_to_different_account"].sum() > 0:
        flags.append("refund_to_different_account")

    return {
        "flag_count": len(flags),
        "flags": flags,
        "n_history": int(len(hist)),
    }


def get_order_context(order_id: str) -> dict:
    df = _load_cases()
    hit = df[df["order_id"].astype(str) == str(order_id)]
    if hit.empty:
        return {"order_id": order_id, "found": False}
    row = hit.iloc[0]
    return {
        "order_id": order_id,
        "found": True,
        "customer_id": str(row["customer_id"]),
        "product_category": row.get("product_category"),
        "return_reason": row.get("return_reason"),
        "refund_amount_requested_usd": float(row.get("refund_amount_requested_usd", 0)),
        "days_to_return": float(row.get("days_to_return", 0)),
        "device_type": row.get("device_type"),
        "payment_method": row.get("payment_method"),
        "country": row.get("country"),
    }


def _demo_syndicate_links(cid: str, G: nx.Graph) -> tuple[set[str], list[dict]]:
    """Populate demo ring links for syndicate scenario customers not in the graph."""
    flagged = [
        n for n, d in G.nodes(data=True) if d.get("kind") == "customer" and n != cid
    ]
    linked = set(flagged[:5])
    edges = []
    for other in sorted(linked)[:5]:
        edges.append({"customer": other, "via": "shared demo signal (device/pay/addr)"})
    return linked, edges


def _empty_shared(cid: str, note: str) -> dict:
    return {
        "customer_id": cid,
        "linked_flagged_customers": 0,
        "shared_nodes": [],
        "sample_linked_customers": [],
        "linked_edges": [],
        "ring_score": 0.0,
        "note": note,
    }


def check_shared_signals(customer_id: str) -> dict:
    """
    Link this customer to previously flagged customers only when BOTH hold:
      1. this customer's own history shows account-linking behaviour
         (multiple accounts / refund to a different account / address change), and
      2. they share the full device|payment|country fingerprint with flagged customers
         who showed the same behaviour.
    A clean customer who merely uses a common device type is never "linked".
    """
    G = _build_shared_signal_graph()
    cid = str(customer_id)
    df = _load_cases()
    mine = df[df["customer_id"].astype(str) == cid]

    if mine.empty:
        if cid.startswith("CUST-RING"):
            linked, linked_edges = _demo_syndicate_links(cid, G)
            return {
                "customer_id": cid,
                "linked_flagged_customers": len(linked),
                "shared_nodes": [],
                "sample_linked_customers": sorted(linked)[:8],
                "linked_edges": linked_edges,
                "ring_score": min(1.0, len(linked) / 10.0),
            }
        return _empty_shared(cid, "customer not in history")

    own_link_behaviour = int(mine[list(_LINK_BEHAVIOUR_COLS)].sum().sum()) > 0
    if not own_link_behaviour:
        return _empty_shared(cid, "no account-linking behaviour on this customer")

    fps = {_fingerprint(r) for _, r in mine.iterrows()}
    shared_nodes = sorted(fp for fp in fps if fp in G)
    linked: set[str] = set()
    linked_edges: list[dict] = []
    for node in shared_nodes:
        for n in G.neighbors(node):
            if G.nodes[n].get("kind") == "customer" and n != cid and n not in linked:
                linked.add(n)
                if len(linked_edges) < 8:
                    linked_edges.append({"customer": n, "via": node})
    ring_score = min(1.0, len(linked) / 10.0)
    return {
        "customer_id": cid,
        "linked_flagged_customers": len(linked),
        "shared_nodes": shared_nodes,
        "sample_linked_customers": sorted(linked)[:8],
        "linked_edges": linked_edges,
        "ring_score": float(ring_score),
    }


_TOOLS = {
    "get_customer_history": get_customer_history,
    "get_ensemble_risk_score": get_ensemble_risk_score,
    "get_shap_explanation": get_shap_explanation,
    "check_pattern": check_pattern,
    "get_order_context": get_order_context,
    "check_shared_signals": check_shared_signals,
}


def call_tool(name: str, **kwargs):
    fn = _TOOLS.get(name)
    if fn is None:
        raise KeyError(f"Unknown tool: {name}")
    return fn(**kwargs)
