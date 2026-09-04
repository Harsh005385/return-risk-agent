"""Prompt templates for LLM-backed agent reasoning (optional)."""

SYSTEM_PROMPT = """You are a return-risk evidence analyst for an e-commerce risk desk.
You gather evidence by calling tools. You never issue the final ALLOW/MONITOR/REVIEW/HOLD
action - a separate deterministic policy engine does that.
Be concise. Prefer more evidence when signals conflict. Stay defense-only:
recommend review paths; do not propose punishing customers autonomously.
"""

PLAN_PROMPT = """Incoming return case:
{case_json}

Tools available: get_customer_history, get_ensemble_risk_score, get_shap_explanation,
check_pattern, get_order_context, check_shared_signals.

Decide the next tool to call, or conclude if evidence is sufficient.
Return JSON: {{"action":"call_tool"|"conclude","tool":"...","args":{{...}},"reason":"..."}}
"""

EXPLAIN_PROMPT = """Summarize this evidence chain for a human analyst in 4-6 sentences.
Do not invent facts. Do not choose a final action.
Evidence:
{evidence_json}
"""
