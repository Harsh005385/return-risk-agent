"""Return Risk Agent - single-page Dash + Plotly stakeholder desk.

Run: python dashboard/app.py  (from repo root; sets PYTHONPATH automatically)
Open: http://127.0.0.1:8050
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
import sys

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import load_env

load_env()

from src.agent.orchestrator import disable_llm, llm_health, run_agent
from src.agent.scenarios import get_scenario, list_scenarios
from src.audit.db import init_db, list_events, log_event, verify_chain_integrity
from src.bootstrap import warm_runtime
from src.demo_catalog import (
    build_case_from_selection,
    category_values,
    customer_detail,
    customer_options,
    encode_label,
    manual_feature_template,
    order_detail,
    order_options,
)
from src.rzp.client import create_test_order, is_configured as rzp_configured

_VERDICTS: deque[dict] = deque(maxlen=80)
_LATENCY: deque[float] = deque(maxlen=200)
_LLM_STATUS: dict = {"status": "unknown", "detail": ""}  # filled once at startup


def _llm_status_text() -> tuple[bool, str]:
    st = _LLM_STATUS.get("status")
    if st == "ok":
        return True, f"AI reasoning active ({_LLM_STATUS.get('detail')})"
    if st == "no_credits":
        return False, "AI key has no credits - rules-based reasoning"
    if st == "not_configured":
        return False, "Rules-based reasoning (no AI key)"
    if st == "error":
        return False, f"AI unavailable - rules-based ({_LLM_STATUS.get('detail')})"
    return False, "Rules-based reasoning"

PLOTLY_TEMPLATE = "plotly_white"
FIG_BG = "#ffffff"
BRAND_BLUE = "#3395FF"
BRAND_NAVY = "#072654"
TEXT_MUTED = "#6B7A90"

ACTION_COLORS = {
    "ALLOW": "#12B76A",
    "MONITOR": "#F79009",
    "REVIEW": "#EF6820",
    "HOLD": "#D92D20",
}

ACTION_BANNER_CLASS = {
    "ALLOW": "allow",
    "MONITOR": "monitor",
    "REVIEW": "review",
    "HOLD": "hold",
}

ACTION_HEADLINE = {
    "ALLOW": "ALLOW - Low Risk",
    "MONITOR": "MONITOR - Moderate Risk",
    "REVIEW": "REVIEW - Analyst Required",
    "HOLD": "HOLD - High Risk",
}


def _score_case(case: dict, fast: bool = True) -> dict:
    t0 = time.perf_counter()
    verdict = run_agent(case, fast=fast)
    ms = (time.perf_counter() - t0) * 1000.0
    verdict["latency_ms"] = ms
    verdict["investigation_mode"] = "fast" if fast else "deep"
    if not fast:
        if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
            verdict["deep_investigation_note"] = (
                "Deep Investigation - sequential LLM-driven reasoning, one tool at a time."
            )
        else:
            verdict["deep_investigation_note"] = (
                "No LLM key configured - running heuristic sequential planner instead."
            )
    else:
        verdict["deep_investigation_note"] = (
            "Fast Path - parallel tool calls, low latency."
        )
    _LATENCY.append(ms)
    _VERDICTS.appendleft(verdict)
    print(
        f"[score] case={verdict.get('case_id')} customer={verdict.get('customer_id')} "
        f"action={verdict.get('action')} risk={float(verdict.get('risk_score') or 0):.1%} "
        f"mode={'deep' if not fast else 'fast'} latency={ms:.0f}ms",
        flush=True,
    )
    return verdict


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor=FIG_BG,
        plot_bgcolor=FIG_BG,
        annotations=[
            dict(
                text="No data yet",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=TEXT_MUTED),
            )
        ],
    )
    return fig


def _syndicate_network_fig(verdict: dict | None) -> go.Figure:
    """Node-edge map for the active case only (check_shared_signals links)."""
    title = "Linked Accounts Map"
    if not verdict:
        return _empty_fig(title)

    shared = (verdict.get("evidence") or {}).get("shared_signals") or {}
    n_linked = int(shared.get("linked_flagged_customers", 0) or 0)
    linked_edges = list(shared.get("linked_edges") or [])
    if not linked_edges:
        for c in shared.get("sample_linked_customers") or []:
            linked_edges.append({"customer": c, "via": "shared fingerprint"})

    if n_linked <= 0 and not linked_edges:
        fig = _empty_fig(title)
        fig.layout.annotations[0].update(text="No linked accounts", font=dict(size=13, color=TEXT_MUTED))
        return fig

    center = str(verdict.get("customer_id") or verdict.get("case_id") or "customer")
    G = nx.Graph()
    G.add_node(center, role="center")
    for edge in linked_edges:
        other = str(edge.get("customer") or "")
        if not other or other == center:
            continue
        G.add_node(other, role="linked")
        G.add_edge(center, other, label=str(edge.get("via", "shared fingerprint")))

    pos = nx.spring_layout(G, seed=42, k=0.9)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color=BRAND_BLUE),
        hoverinfo="none",
    )
    node_x, node_y, colors, labels, sizes, hover = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        role = G.nodes[node].get("role", "linked")
        colors.append(ACTION_COLORS["HOLD"] if role == "linked" else ACTION_COLORS["ALLOW"])
        labels.append(node)
        sizes.append(28 if role == "center" else 18)
        hover.append("This case" if role == "center" else "Linked account")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=labels, textposition="top center",
        customdata=hover,
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#1A2233")),
        hovertemplate="%{text}<br>%{customdata}<extra></extra>",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=320,
        showlegend=False,
        paper_bgcolor=FIG_BG,
        plot_bgcolor=FIG_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _shap_fig(verdict: dict | None) -> go.Figure:
    if not verdict:
        return _empty_fig("What drove this decision")
    drivers = ((verdict.get("evidence") or {}).get("shap") or {}).get("top_drivers") or []
    if not drivers:
        return _empty_fig("What drove this decision")
    df = pd.DataFrame(drivers)
    df["label"] = df["feature"].str.replace("_", " ")
    fig = px.bar(
        df, x="shap_value", y="label", orientation="h",
        color="shap_value", color_continuous_scale="RdYlGn_r",
        title="What drove this decision",
        labels={"shap_value": "Effect on risk", "label": ""},
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def _action_pie(verdicts: list[dict]) -> go.Figure:
    if not verdicts:
        return _empty_fig("Action mix (session)")
    df = pd.DataFrame([{"action": v.get("action")} for v in verdicts])
    counts = df["action"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    fig = px.pie(
        counts, names="action", values="count", color="action",
        color_discrete_map=ACTION_COLORS, title="Actions (session)", hole=0.45,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300,
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _recent_bar(verdicts: list[dict]) -> go.Figure:
    if not verdicts:
        return _empty_fig("Recent cases")
    plotted = list(verdicts)[:20][::-1]
    df = pd.DataFrame([
        {"case": v.get("case_id"), "risk": float(v.get("risk_score") or 0),
         "action": v.get("action"), "pos": i}
        for i, v in enumerate(plotted)
    ])
    fig = px.bar(
        df, x="case", y="risk", color="action",
        color_discrete_map=ACTION_COLORS, title="Recent cases (DEMO DATA)",
        labels={"risk": "Risk score", "case": "Case"}, custom_data=["pos"],
    )
    fig.update_traces(
        hovertemplate="%{x}<br>risk=%{y:.1%}<br>action=%{fullData.name}<extra></extra>"
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        margin=dict(l=40, r=20, t=50, b=80), xaxis_tickangle=-35,
        clickmode="event+select",
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
    )
    return fig


def _latency_fig() -> go.Figure:
    if not _LATENCY:
        return _empty_fig("Response time (ms)")
    s = list(_LATENCY)
    p50 = sorted(s)[len(s) // 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=s, mode="lines+markers", name="latency_ms",
        line=dict(color=BRAND_BLUE),
    ))
    fig.update_layout(
        title=f"Response time - last {len(s)} cases",
        template=PLOTLY_TEMPLATE, height=280, yaxis_title="ms",
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _latency_stats_ui() -> html.Div:
    if not _LATENCY:
        return html.Div("No latency samples yet.", className="text-muted small")
    s = np.asarray(list(_LATENCY), dtype=float)
    p50 = float(np.percentile(s, 50))
    p99 = float(np.percentile(s, 99))
    return html.Div([
        html.Div([
            html.Div(f"{p50:.0f}", className="stat-value mono-num"),
            html.Div("typical ms", className="stat-label"),
        ], className="latency-stat-tile mb-2"),
        html.Div([
            html.Div(f"{p99:.0f}", className="stat-value mono-num"),
            html.Div("slowest ms", className="stat-label"),
        ], className="latency-stat-tile"),
    ])


# Human-readable (title, description) for each agent step. Internal tool names never shown.
_TOOL_LABELS = {
    "get_order_context":       ("Reviewed the order",        "What was bought, why it is coming back and how much refund is requested"),
    "get_customer_history":    ("Reviewed customer history", "Past orders, past returns and any earlier flags on this customer"),
    "get_ensemble_risk_score": ("Assessed the risk",         "Estimated how likely this return is to be abusive"),
    "get_shap_explanation":    ("Found the reasons",         "Identified which factors raised or lowered the risk"),
    "check_pattern":           ("Checked behaviour",         "Looked for suspicious return patterns"),
    "check_shared_signals":    ("Checked linked accounts",   "Looked for connections to other flagged customers"),
    "conclude":                ("Reached a decision",        "Applied refund policy to choose the final action"),
}


def _trace_timeline(verdict: dict) -> html.Div | None:
    trace = verdict.get("trace") or []
    if not trace:
        return None
    mode = verdict.get("investigation_mode", "fast")
    mode_label = "Thorough review" if mode == "deep" else "Quick review"
    steps = []
    for i, step in enumerate(trace, 1):
        plan = step.get("plan") or {}
        tool = plan.get("tool") or plan.get("action", "step")
        title, desc = _TOOL_LABELS.get(tool, (tool.replace("_", " ").title(), plan.get("reason", "")))
        is_final = tool == "conclude"
        steps.append(html.Div([
            html.Div(str(i), className="trace-num final" if is_final else "trace-num"),
            html.Div([
                html.Div(title, className="trace-title"),
                html.Div(desc, className="trace-desc"),
            ]),
        ], className="trace-step"))
    return html.Div([
        html.Div([
            html.Span("Investigation trace", className="section-label"),
            html.Span(mode_label, className="pill pill-muted ms-2"),
        ], className="mb-2 d-flex align-items-center"),
        html.Div(steps, className="trace-timeline"),
    ], className="mb-3")


def _risk_gauge(verdict: dict | None) -> go.Figure:
    risk = float((verdict or {}).get("risk_score") or 0) * 100
    action = str((verdict or {}).get("action") or "")
    color = ACTION_COLORS.get(action, BRAND_BLUE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_MUTED},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": FIG_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(18,183,106,0.12)"},
                {"range": [30, 55], "color": "rgba(247,144,9,0.12)"},
                {"range": [55, 75], "color": "rgba(239,104,32,0.12)"},
                {"range": [75, 100],"color": "rgba(217,45,32,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=210,
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        margin=dict(l=20, r=20, t=30, b=10),
        title={"text": "Abuse risk", "font": {"size": 13, "color": TEXT_MUTED}},
    )
    return fig


def _driver_frequency_fig(verdicts: list[dict]) -> go.Figure:
    """How often each feature appears as a top risk driver across the session."""
    rows = []
    for v in verdicts:
        for d in ((v.get("evidence") or {}).get("shap") or {}).get("top_drivers") or []:
            rows.append({"feature": d["feature"].replace("_", " "),
                         "direction": "raises risk" if d["shap_value"] > 0 else "lowers risk"})
    if not rows:
        return _empty_fig("Most common risk drivers (session)")
    df = pd.DataFrame(rows).value_counts().reset_index(name="count")
    fig = px.bar(
        df, x="count", y="feature", color="direction", orientation="h",
        color_discrete_map={"raises risk": ACTION_COLORS["HOLD"], "lowers risk": ACTION_COLORS["ALLOW"]},
        title="Most common risk drivers (session)",
        labels={"count": "Times in top drivers", "feature": "", "direction": ""},
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320, barmode="stack",
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def _kpi_tiles(verdicts: list[dict]) -> html.Div:
    n = len(verdicts)
    hold = sum(1 for v in verdicts if v.get("action") == "HOLD")
    review = sum(1 for v in verdicts if v.get("action") == "REVIEW")
    allow = sum(1 for v in verdicts if v.get("action") == "ALLOW")
    avg_risk = float(np.mean([float(v.get("risk_score") or 0) for v in verdicts])) if n else 0.0
    lats = [float(v.get("latency_ms") or 0) for v in verdicts if v.get("latency_ms") is not None]
    p50 = float(np.percentile(lats, 50)) if lats else 0.0

    def tile(label, value, sub, tone=""):
        return dbc.Col(html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className=f"kpi-value {tone}".strip()),
            html.Div(sub, className="kpi-sub"),
        ], className="kpi-tile"), md=3, sm=6)

    return dbc.Row([
        tile("Cases scored", f"{n}", "this session"),
        tile("Blocked / flagged", f"{hold + review}", f"{hold} HOLD · {review} REVIEW", "bad" if hold else ""),
        tile("Approved", f"{allow}", f"{(allow / n * 100) if n else 0:.0f}% of cases", "good"),
        tile("Average risk", f"{avg_risk:.0%}", f"{p50:.0f} ms typical response" if lats else "-", "warn" if avg_risk >= 0.5 else ""),
    ], className="g-3 mb-3")


def _session_latency_fig(verdicts: list[dict]) -> go.Figure:
    s = [float(v.get("latency_ms") or 0) for v in reversed(verdicts) if v.get("latency_ms") is not None]
    if not s:
        return _empty_fig("Response time (ms)")
    fig = go.Figure(go.Scatter(
        y=s, mode="lines+markers",
        line=dict(color=BRAND_BLUE, width=2),
        marker=dict(size=7, color=BRAND_BLUE),
    ))
    fig.update_layout(
        title=f"Response time - last {len(s)} cases",
        template=PLOTLY_TEMPLATE, height=280, showlegend=False,
        yaxis_title="ms",
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _session_latency_stats(verdicts: list[dict]) -> html.Div:
    s = [float(v.get("latency_ms") or 0) for v in verdicts if v.get("latency_ms") is not None]
    if not s:
        return html.Div([
            html.Div([html.Div("-", className="stat-value"), html.Div("typical ms", className="stat-label")],
                     className="latency-stat-tile"),
            html.Div([html.Div("-", className="stat-value"), html.Div("slowest ms", className="stat-label")],
                     className="latency-stat-tile"),
        ])
    p50 = float(np.percentile(s, 50))
    p99 = float(np.percentile(s, 99)) if len(s) >= 2 else float(max(s))
    return html.Div([
        html.Div([html.Div(f"{p50:.0f}", className="stat-value"), html.Div("typical ms", className="stat-label")],
                 className="latency-stat-tile"),
        html.Div([html.Div(f"{p99:.0f}", className="stat-value"), html.Div("slowest ms", className="stat-label")],
                 className="latency-stat-tile"),
    ])


def _action_banner(verdict: dict) -> html.Div:
    action = str(verdict.get("action") or "UNKNOWN")
    cls = ACTION_BANNER_CLASS.get(action, "monitor")
    headline = ACTION_HEADLINE.get(action, action)
    risk = float(verdict.get("risk_score") or 0)
    ms = float(verdict.get("latency_ms") or 0)
    return html.Div(
        [
            html.Div(headline, style={"fontSize": "1.4rem"}),
            html.Div(
                [
                    html.Span(f"Risk {risk:.1%}", className="mono-num me-3"),
                    html.Span(f"{ms:.0f} ms", className="mono-num"),
                ],
                style={"fontSize": "0.95rem", "marginTop": "0.25rem"},
            ),
        ],
        className=f"action-banner {cls}",
    )


def _review_panel(verdict: dict | None) -> html.Div:
    if not verdict:
        return html.Div([
            html.Div("No case scored yet", className="empty-title"),
            html.Div("Select a customer/order or use Manual Entry, then run score.",
                     className="empty-sub"),
        ], className="empty-state")

    evidence = verdict.get("evidence") or {}
    hist = evidence.get("history") or {}
    pattern = evidence.get("pattern") or {}
    shared = evidence.get("shared_signals") or {}
    score = evidence.get("score") or {}
    drivers = (evidence.get("shap") or {}).get("top_drivers") or []
    trace_block = _trace_timeline(verdict)

    flags = pattern.get("flags") or []
    linked = int(shared.get("linked_flagged_customers", 0) or 0)
    action = str(verdict.get("action") or "")

    return html.Div([
        dbc.Row([
            dbc.Col([
                _action_banner(verdict),
                html.Div([
                    html.Span(f"Case {verdict.get('case_id')}", className="case-id mono-num"),
                    html.Span(
                        "Thorough review" if verdict.get("investigation_mode") == "deep" else "Quick review",
                        className="pill pill-muted ms-2",
                    ),
                    html.Span(f"confidence {verdict.get('confidence')}", className="pill pill-muted ms-1"),
                ], className="mb-2 d-flex align-items-center flex-wrap"),
                html.P(verdict.get("rationale") or "", className="rationale"),
            ], md=8),
            dbc.Col(dcc.Graph(id="fig-gauge", figure=_risk_gauge(verdict),
                              config={"displayModeBar": False}), md=4),
        ], className="g-2 align-items-center"),
        trace_block,
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("Customer", className="profile-title"),
                html.Div(verdict.get("customer_id"), className="mono-num stat-big"),
                _kv("Prior returns", hist.get("n_returns", 0)),
                _kv("Prior abuse flags", hist.get("prior_abuse_flags", 0),
                    "bad" if int(hist.get("prior_abuse_flags", 0) or 0) else "good"),
            ], className="profile-card h-100"), md=4),
            dbc.Col(html.Div([
                html.Div("Risk score", className="profile-title"),
                html.Div(f"{float(score.get('risk_score') or 0):.1%}",
                         className="mono-num stat-big",
                         style={"color": ACTION_COLORS.get(action, BRAND_BLUE)}),
                _kv("Decision", action, {"ALLOW": "good", "MONITOR": "warn"}.get(action, "bad")),
                _kv("Confidence", verdict.get("confidence")),
            ], className="profile-card h-100"), md=4),
            dbc.Col(html.Div([
                html.Div("Signals", className="profile-title"),
                html.Div(str(len(flags) + linked), className="mono-num stat-big",
                         style={"color": ACTION_COLORS["HOLD"] if (flags or linked) else ACTION_COLORS["ALLOW"]}),
                _kv("Linked customers", linked, "bad" if linked else "good"),
                _kv("Pattern flags", ", ".join(f.replace("_", " ") for f in flags) or "none",
                    "bad" if flags else "good"),
            ], className="profile-card h-100"), md=4),
        ], className="mb-3 g-2"),
        html.Div("Top drivers", className="section-label mb-1"),
        html.Div([
            html.Div([
                html.Span(d["feature"].replace("_", " "), className="driver-name"),
                html.Span(f"{d['feature_value']:.2f}", className="driver-val mono-num"),
                html.Span(f"{d['shap_value']:+.3f}",
                          className="driver-shap mono-num " + ("up" if d["shap_value"] > 0 else "down")),
            ], className="driver-row")
            for d in drivers[:5]
        ], className="driver-list"),
        html.Div(
            f"Razorpay test order: {verdict.get('razorpay_order', {}).get('order_id', '-')}",
            className="small text-muted mt-2 mono-num",
        ) if verdict.get("razorpay_order") else None,
    ])


def _kv(label: str, value, tone: str = "") -> html.Div:
    """One key/value row for profile cards. tone: '', 'good', 'warn', 'bad'."""
    return html.Div([
        html.Span(label, className="kv-key"),
        html.Span(str(value), className=f"kv-val {tone}".strip()),
    ], className="kv-row")


def _rate_tone(rate_pct: float) -> str:
    if rate_pct >= 50:
        return "bad"
    if rate_pct >= 25:
        return "warn"
    return "good"


def _customer_detail_ui(detail: dict | None) -> html.Div:
    if not detail:
        return html.Div(
            "Select a customer to see their history.",
            className="profile-card profile-empty",
        )
    cats = ", ".join(f"{k} ({v})" for k, v in (detail.get("categories") or {}).items()) or "-"
    rate = float(detail["mean_return_rate_pct"])
    flags = int(detail["prior_abuse_flags"])
    return html.Div([
        html.Div("Customer profile", className="profile-title"),
        _kv("Orders", detail["n_orders"]),
        _kv("Avg return rate", f"{rate:.1f}%", _rate_tone(rate)),
        _kv("Prior abuse flags", flags, "bad" if flags else "good"),
        _kv("Top categories", cats),
    ], className="profile-card")


def _order_detail_ui(detail: dict | None) -> html.Div:
    if not detail:
        return html.Div(
            "Select an order to see its details.",
            className="profile-card profile-empty",
        )
    rate = float(detail["return_rate_pct"])
    refund = float(detail["refund_amount_requested_usd"])
    return html.Div([
        html.Div("Order profile", className="profile-title"),
        _kv("Customer", detail["customer_id"]),
        _kv("Category", detail.get("product_category") or "-"),
        _kv("Reason", detail.get("return_reason") or "-"),
        _kv("Refund", f"${refund:,.2f}", "warn" if refund >= 300 else ""),
        _kv("Return rate", f"{rate:.1f}%", _rate_tone(rate)),
        _kv("Device / pay", f"{detail.get('device_type')} / {detail.get('payment_method')}"),
    ], className="profile-card")


def _build_layout() -> html.Div:
    scenarios = list_scenarios()
    scenario_buttons = []
    for sc in scenarios:
        is_syndicate = sc["id"] == "syndicate"
        scenario_buttons.append(
            dbc.Button(
                "Syndicate - linked accounts demo" if is_syndicate else sc["id"].title(),
                id={"type": "scenario-btn", "index": sc["id"]},
                color="danger" if is_syndicate else "primary",
                outline=not is_syndicate,
                className="me-2 mb-2",
                n_clicks=0,
            )
        )
    rzp_ok = rzp_configured()
    llm_ok, llm_text = _llm_status_text()
    reason_opts = [{"label": v, "value": v} for v in category_values("return_reason")]
    category_opts = [{"label": v, "value": v} for v in category_values("product_category")]
    default_reason = "Changed Mind" if any(o["value"] == "Changed Mind" for o in reason_opts) else (reason_opts[0]["value"] if reason_opts else None)
    default_category = "Electronics" if any(o["value"] == "Electronics" for o in category_opts) else (category_opts[0]["value"] if category_opts else None)

    def status_row(label: str, ok: bool, on_text: str, off_text: str) -> html.Div:
        return html.Div([
            html.Span(className="status-dot on" if ok else "status-dot off"),
            html.Div([
                html.Div(label, className="status-label"),
                html.Div(on_text if ok else off_text, className="status-value"),
            ]),
        ], className="status-row")

    sidebar = html.Aside([
        html.Div([
            html.Div("RR", className="brand-mark"),
            html.Div([
                html.Div("Return Risk Agent", className="brand-title"),
                html.Div("Razorpay AI Buildathon", className="brand-sub"),
            ]),
        ], className="brand"),
        html.Nav([
            html.A([html.Span("01", className="nav-num"), "Overview"], href="#overview", className="nav-link-rz"),
            html.A([html.Span("02", className="nav-num"), "Score a return"], href="#score", className="nav-link-rz"),
            html.A([html.Span("03", className="nav-num"), "Analytics"], href="#analytics", className="nav-link-rz"),
            html.A([html.Span("04", className="nav-num"), "Audit trail"], href="#audit", className="nav-link-rz"),
        ], className="side-nav"),
        html.Div("System status", className="side-section-title"),
        status_row("Reasoning", llm_ok, llm_text, llm_text),
        status_row("Razorpay test mode", rzp_ok, "Connected", "Keys not set"),
        status_row("Audit trail", True, "Tamper-evident, verified on demand", ""),
        html.Div([
            html.Div("Defense-only", className="side-note-title"),
            html.Div("Scores refund-abuse risk to protect merchants. Never used to target customers.",
                     className="side-note"),
        ], className="side-footer"),
    ], className="rz-sidebar")

    topbar = html.Div([
        html.Div([
            html.H2("Risk console", className="page-title"),
            html.Div("Return-abuse risk checks with clear reasons and a full audit trail",
                     className="page-sub"),
        ]),
        html.Div([
            dbc.Badge("DEMO DATA", className="demo-badge-light me-2"),
            html.Span(id="topbar-clock", className="pill pill-muted"),
        ], className="d-flex align-items-center"),
    ], className="topbar", id="overview")

    main = dbc.Container([
        dcc.Store(id="selected-verdict"),
        dcc.Store(id="verdict-tick", data=0),
        dcc.Store(id="session-verdicts", data=[]),
        dcc.Interval(id="refresh-clock", interval=15_000, n_intervals=0),
        topbar,
        html.Div(id="kpi-row"),
        html.Div(id="score"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Tabs(id="input-tabs", active_tab="tab-dataset", children=[

                    # ── Tab 1: Dataset lookup ────────────────────────────────
                    dbc.Tab(label="Dataset Lookup", tab_id="tab-dataset", children=[
                        html.P(
                            "Pick customer and order. Agent reads history and order data automatically.",
                            className="small text-muted mb-2 mt-2",
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Customer", className="fw-semibold small"),
                                dcc.Dropdown(
                                    id="live-customer", options=customer_options(),
                                    searchable=True, clearable=True,
                                    placeholder="Search customer from dataset…", className="mb-1",
                                ),
                                html.Div(id="customer-detail"),
                            ], md=6),
                            dbc.Col([
                                html.Label("Order", className="fw-semibold small"),
                                dcc.Dropdown(
                                    id="live-order", options=order_options(limit=2000),
                                    searchable=True, clearable=True,
                                    placeholder="Search order from dataset…", className="mb-1",
                                ),
                                html.Div(id="order-detail"),
                            ], md=6),
                        ], className="mb-2"),
                        html.Div([
                            dbc.Checklist(
                                id="live-syndicate",
                                options=[{"label": " Check for linked accounts", "value": 1}],
                                value=[], switch=True,
                            ),
                        ], className="toggle-mini-card"),
                        html.Div([
                            dbc.Checklist(
                                id="deep-investigation",
                                options=[{"label": " Thorough review", "value": 1}],
                                value=[], switch=True,
                            ),
                        ], className="toggle-mini-card"),
                        html.Div([
                            dbc.Checklist(
                                id="live-razorpay",
                                options=[{
                                    "label": " Create linked Razorpay test-mode order",
                                    "value": 1,
                                    "disabled": not rzp_ok,
                                }],
                                value=[], switch=True,
                            ),
                        ], className="toggle-mini-card"),
                        dbc.Button(
                            "Run agent score", id="live-run", color="success",
                            className="w-100", disabled=True,
                        ),
                        dbc.Tooltip(
                            "Select an order first. The agent reviews the customer, scores the "
                            "return, explains the reasons and applies refund policy.",
                            target="live-run", placement="top",
                        ),
                        dcc.Loading(
                            html.Div(id="live-status", className="mt-2"),
                            type="dot", color=BRAND_BLUE, delay_show=150,
                        ),
                        html.Hr(className="my-3"),
                        html.Div("Demo presets", className="section-label mb-2"),
                        html.Div(scenario_buttons),
                    ]),

                    # ── Tab 2: Manual Entry ──────────────────────────────────
                    dbc.Tab(label="Manual Entry", tab_id="tab-manual", children=[
                        html.P(
                            "Enter the return details and the agent will assess it. "
                            "Fields you leave alone use typical values.",
                            className="small text-muted mb-2 mt-2",
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Refund requested (USD)", className="fw-semibold small"),
                                dbc.Input(id="m-refund", type="number", min=0, max=5000, value=100, className="mb-2"),
                            ], md=6),
                            dbc.Col([
                                html.Label("Return rate %", className="fw-semibold small"),
                                dbc.Input(id="m-return-rate", type="number", min=0, max=100, value=10, className="mb-2"),
                                html.Div("In this dataset, rates above ~20% are strongly abusive.",
                                         className="field-hint"),
                            ], md=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Total orders (lifetime)", className="fw-semibold small"),
                                dbc.Input(id="m-total-orders", type="number", min=0, max=500, value=10, className="mb-2"),
                            ], md=6),
                            dbc.Col([
                                html.Label("Prior dispute count", className="fw-semibold small"),
                                dbc.Input(id="m-disputes", type="number", min=0, max=20, value=0, className="mb-2"),
                            ], md=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Return reason", className="fw-semibold small"),
                                dcc.Dropdown(
                                    id="m-return-reason", options=reason_opts,
                                    value=default_reason, clearable=False, className="mb-2",
                                ),
                            ], md=6),
                            dbc.Col([
                                html.Label("Product category", className="fw-semibold small"),
                                dcc.Dropdown(
                                    id="m-category", options=category_opts,
                                    value=default_category, clearable=False, className="mb-2",
                                ),
                            ], md=6),
                        ]),
                        dbc.Checklist(
                            id="m-flags",
                            options=[
                                {"label": " High-value item", "value": "high_value"},
                                {"label": " Discount used", "value": "discount"},
                                {"label": " Photo evidence provided", "value": "photo"},
                                {"label": " Valid tracking number", "value": "tracking"},
                                {"label": " Packaging intact", "value": "packaging"},
                            ],
                            value=["tracking"],
                            switch=True,
                            inline=True,
                            className="mb-3 small",
                        ),
                        dbc.Button(
                            "Score this case", id="manual-run", color="warning",
                            className="w-100",
                        ),
                        dcc.Loading(
                            html.Div(id="manual-status", className="mt-2"),
                            type="dot", color=ACTION_COLORS["MONITOR"], delay_show=150,
                        ),
                    ]),
                ]),
            ]), className="rz-card h-100"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.H5("Case details", className="card-title-rz me-2"),
                    dbc.Badge("DEMO DATA", className="demo-badge-light align-middle"),
                ], className="d-flex align-items-center mb-2"),
                dcc.Loading(
                    html.Div(id="review-panel"),
                    type="dot", color=BRAND_BLUE, delay_show=150,
                ),
            ]), className="rz-card h-100"), md=8),
        ], className="mb-3 g-3"),

        html.Div(id="analytics"),
        html.Div([
            html.H5("Analytics", className="card-title-rz mb-0"),
            html.Span("Session view - refreshes after every scored case",
                      className="small text-muted ms-2"),
        ], className="section-header"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-recent", config={"displayModeBar": False})),
                             className="rz-card"), md=7),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-actions", config={"displayModeBar": False})),
                             className="rz-card"), md=5),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id="fig-shap", config={"displayModeBar": False}),
                html.Div(
                    "Bars to the right raised the risk; bars to the left lowered it.",
                    className="shap-caption",
                ),
            ]), className="rz-card"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-drivers", config={"displayModeBar": False})),
                             className="rz-card"), md=6),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dbc.Row([
                dbc.Col(dcc.Graph(id="fig-latency", config={"displayModeBar": False}), md=8),
                dbc.Col(html.Div(id="latency-stats"), md=4, className="d-flex flex-column justify-content-center"),
            ])), className="rz-card"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id="fig-syndicate", config={"displayModeBar": False}),
            ]), className="rz-card"), md=6),
        ], className="mb-3 g-3"),

        html.Div(id="audit"),
        dbc.Card(dbc.CardBody([
            html.Div([
                html.H5("Audit trail", className="card-title-rz me-2 mb-0"),
                dbc.Badge("DEMO DATA", className="demo-badge-light me-3"),
                dbc.Button("Verify integrity", id="verify-integrity-btn", color="primary",
                           outline=True, size="sm"),
                html.Div(id="integrity-status", className="ms-3"),
            ], className="d-flex align-items-center mb-2 flex-wrap"),
            html.Div(
                "Every step of every decision is recorded in a tamper-evident log. "
                "Verify checks that nothing has been altered since it was written.",
                className="small text-muted mb-2",
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-audit", config={"displayModeBar": False}), md=6),
                dbc.Col([
                    html.Div("Latest events", className="section-label mb-1"),
                    html.Div(id="audit-lines"),
                ], md=6),
            ]),
        ]), className="rz-card mb-4"),
    ], fluid=True, className="rz-main")

    return html.Div([sidebar, main], className="rz-shell")


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Return Risk Agent",
)
server = app.server
app.layout = _build_layout

# Gunicorn / production entry: warm once on import (main() also warms for local runs).
init_db()
try:
    warm_runtime()
except Exception as exc:  # noqa: BLE001
    print(f"Warmup deferred: {exc}", flush=True)


@callback(
    Output("live-order", "options"),
    Output("live-order", "value"),
    Output("customer-detail", "children"),
    Input("live-customer", "value"),
    State("live-order", "value"),
)
def on_customer_selected(customer_id, current_order):
    if customer_id:
        orders = order_options(customer_id)
        detail = customer_detail(customer_id)
        order_ids = {o["value"] for o in orders}
        order_val = current_order if current_order in order_ids else None
        return orders, order_val, _customer_detail_ui(detail)
    return order_options(limit=2000), no_update, _customer_detail_ui(None)


@callback(
    Output("order-detail", "children"),
    Output("live-customer", "value", allow_duplicate=True),
    Input("live-order", "value"),
    prevent_initial_call=True,
)
def on_order_selected(order_id):
    detail = order_detail(order_id)
    if not detail:
        return _order_detail_ui(None), no_update
    return _order_detail_ui(detail), detail["customer_id"]


@callback(Output("live-run", "disabled"), Input("live-order", "value"))
def toggle_run_button(order_id):
    return not bool(order_id)


def _maybe_rzp_order(verdict: dict, order_id: str | None, enabled: bool) -> dict:
    if not enabled or not rzp_configured():
        return verdict
    detail = order_detail(order_id) if order_id else None
    amount_inr = float((detail or {}).get("refund_amount_requested_usd", 75) or 75) * 83.0
    try:
        rzp = create_test_order(amount_inr, {"case_id": verdict.get("case_id")})
        verdict["razorpay_order"] = rzp
        log_event(
            str(verdict.get("case_id")),
            str(verdict.get("customer_id")),
            "razorpay_order",
            rzp,
        )
    except Exception as exc:  # noqa: BLE001
        verdict["razorpay_error"] = str(exc)
    return verdict


@callback(
    Output("selected-verdict", "data"),
    Output("verdict-tick", "data"),
    Output("live-status", "children"),
    Output("session-verdicts", "data"),
    Input({"type": "scenario-btn", "index": dash.ALL}, "n_clicks"),
    Input("live-run", "n_clicks"),
    State("live-customer", "value"),
    State("live-order", "value"),
    State("live-syndicate", "value"),
    State("deep-investigation", "value"),
    State("live-razorpay", "value"),
    State("verdict-tick", "data"),
    State("session-verdicts", "data"),
    prevent_initial_call=True,
)
def on_score(scenario_clicks, live_clicks, customer, order, syndicate, deep, rzp_flag, tick, session):
    triggered = dash.ctx.triggered_id
    if triggered is None:
        return no_update, no_update, no_update, no_update

    def _push(verdict):
        hist = list(session or [])
        hist.insert(0, verdict)
        return hist[:80]

    fast = not bool(deep)
    try:
        if isinstance(triggered, dict) and triggered.get("type") == "scenario-btn":
            case = get_scenario(triggered["index"])
            if not case:
                return no_update, no_update, dbc.Alert("Unknown scenario", color="danger"), no_update
            verdict = _score_case(case, fast=fast)
            verdict["scenario"] = triggered["index"]
            verdict = _maybe_rzp_order(verdict, case.get("order_id"), bool(rzp_flag))
            linked = int(
                ((verdict.get("evidence") or {}).get("shared_signals") or {})
                .get("linked_flagged_customers", 0)
                or 0
            )
            msg = dbc.Alert(
                f"Scenario {triggered['index']} → {verdict['action']} "
                f"({verdict['risk_score']:.1%}, {verdict['latency_ms']:.0f} ms"
                + (f", {linked} linked accounts)" if linked else ")"),
                color="success",
            )
            print(
                f"[ui] scenario={triggered['index']} action={verdict['action']} "
                f"linked={linked}",
                flush=True,
            )
            return verdict, (tick or 0) + 1, msg, _push(verdict)

        if triggered == "live-run":
            if not order:
                return no_update, no_update, dbc.Alert(
                    "Select an order from dataset first.", color="warning",
                ), no_update
            case = build_case_from_selection(
                customer_id=customer, order_id=order,
                refund_usd=None, return_rate_pct=None,
                force_syndicate_check=bool(syndicate),
            )
            verdict = _score_case(case, fast=fast)
            verdict = _maybe_rzp_order(verdict, order, bool(rzp_flag))
            msg = dbc.Alert(
                f"{order} → {verdict['action']} "
                f"({verdict['risk_score']:.1%}, {verdict['latency_ms']:.0f} ms)",
                color="success",
            )
            return verdict, (tick or 0) + 1, msg, _push(verdict)
    except Exception as exc:  # noqa: BLE001
        return no_update, no_update, dbc.Alert(str(exc), color="danger"), no_update

    return no_update, no_update, no_update, no_update


@callback(
    Output("selected-verdict", "data", allow_duplicate=True),
    Input("fig-recent", "clickData"),
    State("session-verdicts", "data"),
    prevent_initial_call=True,
)
def on_bar_click(click_data, session):
    if not click_data:
        return no_update
    try:
        pos = int(click_data["points"][0]["customdata"][0])
        plotted = list(session or [])[:20][::-1]
        if 0 <= pos < len(plotted):
            return plotted[pos]
    except Exception:
        return no_update
    return no_update


@callback(
    Output("selected-verdict", "data", allow_duplicate=True),
    Output("verdict-tick", "data", allow_duplicate=True),
    Output("manual-status", "children"),
    Output("session-verdicts", "data", allow_duplicate=True),
    Input("manual-run", "n_clicks"),
    State("m-refund", "value"),
    State("m-return-rate", "value"),
    State("m-total-orders", "value"),
    State("m-disputes", "value"),
    State("m-return-reason", "value"),
    State("m-category", "value"),
    State("m-flags", "value"),
    State("verdict-tick", "data"),
    State("session-verdicts", "data"),
    prevent_initial_call=True,
)
def on_manual_score(n_clicks, refund, return_rate, total_orders, disputes,
                    return_reason, category, flags, tick, session):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    flags = flags or []
    refund_val = float(refund or 100)
    rr = float(return_rate or 10)
    orders = float(total_orders or 10)
    # Start from dataset-typical values so untouched fields are realistic, then override.
    features = dict(manual_feature_template())
    features.update({
        "refund_amount_requested_usd": refund_val,
        "avg_order_value_usd": max(features.get("avg_order_value_usd", 0.0), refund_val),
        "total_orders_lifetime": orders,
        "total_returns_lifetime": round(orders * rr / 100.0),
        "return_rate_pct": rr,
        "previous_dispute_count": float(disputes or 0),
        "is_high_value_item": int("high_value" in flags or refund_val > 300),
        "discount_used": int("discount" in flags),
        "return_packaging_intact": int("packaging" in flags),
        "photo_evidence_provided": int("photo" in flags),
        "tracking_number_valid": int("tracking" in flags),
        "return_reason_enc": encode_label("return_reason", return_reason),
        "product_category_enc": encode_label("product_category", category),
        "return_cost_proxy": refund_val * 0.15,
    })
    case = {
        "order_id": f"MANUAL-{int(time.time())}",
        "customer_id": "MANUAL-ENTRY",
        "return_reason": return_reason,
        "product_category": category,
        "features": features,
        "force_syndicate_check": False,
    }
    try:
        verdict = _score_case(case, fast=True)
        color = "success" if verdict["action"] == "ALLOW" else "warning" if verdict["action"] == "MONITOR" else "danger"
        msg = dbc.Alert(
            f"Manual case scored: {verdict['action']} ({verdict['risk_score']:.1%}, {verdict['latency_ms']:.0f} ms)",
            color=color,
        )
        hist = list(session or [])
        hist.insert(0, verdict)
        return verdict, (tick or 0) + 1, msg, hist[:80]
    except Exception as exc:  # noqa: BLE001
        return no_update, no_update, dbc.Alert(str(exc), color="danger"), no_update


@callback(Output("integrity-status", "children"), Input("verify-integrity-btn", "n_clicks"))
def on_verify_integrity(n_clicks):
    if not n_clicks:
        return ""
    result = verify_chain_integrity()
    if result["valid"]:
        return html.Span(
            f"Chain verified · {result['checked']} records intact",
            className="pill pill-ok",
        )
    return html.Span(
        f"Tampering detected at record #{result['first_break_id']}",
        className="pill pill-bad",
    )


@callback(
    Output("review-panel", "children"),
    Output("fig-shap", "figure"),
    Output("fig-recent", "figure"),
    Output("fig-actions", "figure"),
    Output("fig-latency", "figure"),
    Output("fig-audit", "figure"),
    Output("fig-syndicate", "figure"),
    Output("latency-stats", "children"),
    Output("audit-lines", "children"),
    Output("fig-drivers", "figure"),
    Output("kpi-row", "children"),
    Output("topbar-clock", "children"),
    Input("selected-verdict", "data"),
    Input("verdict-tick", "data"),
    Input("session-verdicts", "data"),
    Input("refresh-clock", "n_intervals"),
)
def refresh_views(selected, _tick, session, _n):
    # Browser-session only - never reuse prior server memory when page is fresh.
    verdicts = list(session or [])
    active = selected if selected else (verdicts[0] if verdicts else None)
    clock = time.strftime("Updated %H:%M:%S")

    events = list_events(limit=40)
    if events:
        edf = pd.DataFrame([{"event": e.get("event_type")} for e in events])
        counts = edf["event"].value_counts().reset_index()
        counts.columns = ["event", "count"]
        audit_fig = go.Figure(go.Bar(
            x=counts["event"], y=counts["count"],
            text=counts["count"], textposition="outside",
            marker_color=BRAND_BLUE,
        ))
        audit_fig.update_layout(
            template=PLOTLY_TEMPLATE, height=280, showlegend=False,
            title="Audit event mix (DEMO DATA)",
            paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        lines = [
            html.Div(
                f"{(e.get('created_at') or '')[:19].replace('T', ' ')} · "
                f"{e.get('event_type')} · {e.get('case_id')}",
                className="audit-line",
            )
            for e in events[:8]
        ]
    else:
        audit_fig = _empty_fig("Audit event mix")
        lines = [html.Div("No audit events yet.")]

    return (
        _review_panel(active),
        _shap_fig(active),
        _recent_bar(verdicts),
        _action_pie(verdicts),
        _session_latency_fig(verdicts),
        audit_fig,
        _syndicate_network_fig(active),
        _session_latency_stats(verdicts),
        lines,
        _driver_frequency_fig(verdicts),
        _kpi_tiles(verdicts),
        clock,
    )


def main():
    init_db()
    try:
        warm_runtime()
        print("Warmup complete.")
    except Exception as exc:  # noqa: BLE001
        print(f"Warmup skipped: {exc}")
    _LLM_STATUS.update(llm_health())
    print(f"LLM status: {_LLM_STATUS['status']} - {_LLM_STATUS['detail']}", flush=True)
    if _LLM_STATUS.get("status") in {"no_credits", "error"}:
        disable_llm(24 * 3600)
        print("LLM planner skipped for this process (startup probe failed).", flush=True)
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
