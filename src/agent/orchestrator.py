"""
Agent orchestrator: iterative tool-use loop with graceful LLM degradation.

Default path uses a fast parallel evidence gather (still multi-tool + audited).
Optional LLM planner when LLM_API_KEY is set (sequential).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent import prompts
from src.agent.tools import call_tool
from src.audit.db import log_event
from src.features import row_to_features
from src.policy.policy_engine import decide

MAX_ITERS = 4
LLM_COOLDOWN_SEC = 120.0
_llm_disabled_until = 0.0  # after an LLM failure, skip LLM calls until this time


def disable_llm(seconds: float | None = None) -> None:
    """Skip LLM planner until `seconds` elapse (default: LLM_COOLDOWN_SEC)."""
    global _llm_disabled_until
    import time as _time

    _llm_disabled_until = _time.time() + (seconds if seconds is not None else LLM_COOLDOWN_SEC)


def llm_health() -> dict:
    """One cheap probe of the configured LLM endpoint. Returns {'status': ..., 'detail': ...}.
    status: 'ok' | 'not_configured' | 'no_credits' | 'error'."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "not_configured", "detail": "No LLM_API_KEY set"}
    import urllib.error
    import urllib.request

    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            return {"status": "ok", "detail": model}
    except urllib.error.HTTPError as exc:
        text = ""
        try:
            text = exc.read().decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            pass
        if exc.code == 429 and ("quota" in text or "credit" in text):
            return {"status": "no_credits", "detail": "API key valid but account has no credits"}
        return {"status": "error", "detail": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


def _features_from_case(case: dict) -> dict:
    if "features" in case and isinstance(case["features"], dict):
        return row_to_features(case["features"])
    return row_to_features(case)


def _gather_evidence_fast(case: dict) -> tuple[dict[str, Any], list[dict]]:
    """Run core tools in parallel, then optional syndicate check."""
    customer_id = str(case.get("customer_id", ""))
    order_id = str(case.get("order_id") or case.get("case_id") or "")
    features = _features_from_case(case)
    evidence: dict[str, Any] = {}
    trace: list[dict] = []

    jobs = {
        "order_context": ("get_order_context", {"order_id": order_id}),
        "history": ("get_customer_history", {"customer_id": customer_id}),
        "score": ("get_ensemble_risk_score", {"features": features}),
        "shap": ("get_shap_explanation", {"features": features, "top_k": 5}),
        "pattern": ("check_pattern", {"customer_id": customer_id}),
    }

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            key: pool.submit(call_tool, name, **kwargs)
            for key, (name, kwargs) in jobs.items()
        }
        for key, fut in futures.items():
            result = fut.result()
            evidence[key] = result
            plan = {
                "action": "call_tool",
                "tool": jobs[key][0],
                "args": jobs[key][1],
                "reason": f"Parallel evidence gather: {key}",
            }
            trace.append({"iteration": 0, "plan": plan, "tool_result": {key: result}})
            log_event(order_id or "unknown", customer_id, "tool_call", {
                "tool": jobs[key][0],
                "args": jobs[key][1],
                "result": result,
                "mode": "parallel",
            })

    score = float((evidence.get("score") or {}).get("risk_score", 0))
    flags = int((evidence.get("pattern") or {}).get("flag_count", 0))
    if score >= 0.55 or flags >= 1 or case.get("force_syndicate_check"):
        shared = call_tool("check_shared_signals", customer_id=customer_id)
        evidence["shared_signals"] = shared
        plan = {
            "action": "call_tool",
            "tool": "check_shared_signals",
            "args": {"customer_id": customer_id},
            "reason": "Elevated risk - shared-signal / abuse-ring check.",
        }
        trace.append({"iteration": 1, "plan": plan, "tool_result": {"shared_signals": shared}})
        log_event(order_id or "unknown", customer_id, "tool_call", {
            "tool": "check_shared_signals",
            "args": {"customer_id": customer_id},
            "result": shared,
            "mode": "tier2",
        })

    trace.append({
        "iteration": 2,
        "plan": {"action": "conclude", "reason": "Parallel evidence complete."},
    })
    return evidence, trace


def _heuristic_plan(case: dict, evidence: dict, iteration: int) -> dict:
    customer_id = str(case.get("customer_id", ""))
    order_id = str(case.get("order_id", ""))
    features = _features_from_case(case)

    if iteration == 0 and "order_context" not in evidence and order_id:
        return {
            "action": "call_tool",
            "tool": "get_order_context",
            "args": {"order_id": order_id},
            "reason": "Need order/return context first.",
        }
    if "history" not in evidence:
        return {
            "action": "call_tool",
            "tool": "get_customer_history",
            "args": {"customer_id": customer_id},
            "reason": "Need customer return history.",
        }
    if "score" not in evidence:
        return {
            "action": "call_tool",
            "tool": "get_ensemble_risk_score",
            "args": {"features": features},
            "reason": "Need ensemble risk score.",
        }
    if "shap" not in evidence:
        return {
            "action": "call_tool",
            "tool": "get_shap_explanation",
            "args": {"features": features, "top_k": 3},
            "reason": "Need SHAP drivers for explainability.",
        }
    if "pattern" not in evidence:
        return {
            "action": "call_tool",
            "tool": "check_pattern",
            "args": {"customer_id": customer_id},
            "reason": "Need behavioral pattern flags.",
        }
    score = float((evidence.get("score") or {}).get("risk_score", 0))
    flags = int((evidence.get("pattern") or {}).get("flag_count", 0))
    if "shared_signals" not in evidence and (
        score >= 0.55 or flags >= 1 or case.get("force_syndicate_check")
    ):
        return {
            "action": "call_tool",
            "tool": "check_shared_signals",
            "args": {"customer_id": customer_id},
            "reason": "Elevated risk - check shared device/payment/address syndicate signals.",
        }
    return {
        "action": "conclude",
        "tool": None,
        "args": {},
        "reason": "Evidence sufficient for policy engine.",
    }


def _try_llm_plan(case: dict, evidence: dict) -> dict | None:
    global _llm_disabled_until
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    import time as _time

    if _time.time() < _llm_disabled_until:
        return None  # recent failure; do not pay the round-trip again yet
    try:
        import urllib.request

        base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        body = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompts.PLAN_PROMPT.format(
                        case_json=json.dumps(
                            {"case": case, "evidence_so_far": evidence}, default=str
                        )
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        # Do not conclude on LLM failure; let the heuristic planner take over and
        # back off from the LLM for a while so every run does not pay the failed call.
        _llm_disabled_until = _time.time() + LLM_COOLDOWN_SEC
        print(
            f"[agent] LLM planner unavailable ({exc}); using rules-based planner "
            f"for the next {LLM_COOLDOWN_SEC:.0f}s",
            flush=True,
        )
        return None


def _template_explanation(evidence: dict, decision) -> str:
    score = (evidence.get("score") or {}).get("risk_score")
    drivers = (evidence.get("shap") or {}).get("top_drivers") or []
    driver_txt = ", ".join(
        f"{d['feature'].replace('_', ' ')} ({d['shap_value']:+.2f})"
        for d in drivers[:3]
    ) or "n/a"
    pattern = evidence.get("pattern") or {}
    shared = evidence.get("shared_signals") or {}
    return (
        f"Risk {float(score or 0):.1%}. "
        f"Drivers: {driver_txt}. "
        f"Flags: {len(pattern.get('flags') or [])}. "
        f"Linked: {shared.get('linked_flagged_customers', 0)}."
    )


def _finalize(case: dict, evidence: dict, trace: list[dict]) -> dict[str, Any]:
    case_id = str(case.get("order_id") or case.get("case_id") or "unknown")
    customer_id = str(case.get("customer_id") or "")
    score_info = evidence.get("score") or call_tool(
        "get_ensemble_risk_score", features=_features_from_case(case)
    )
    evidence.setdefault("score", score_info)
    decision = decide(
        risk_score=float(score_info["risk_score"]),
        evidence=evidence,
        confidence_gap=float(score_info.get("confidence_gap") or 0),
    )
    explanation = _template_explanation(evidence, decision)
    verdict = {
        "case_id": case_id,
        "customer_id": customer_id,
        "risk_score": decision.risk_score,
        "action": decision.action,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "explanation": explanation,
        "evidence": evidence,
        "trace": trace,
        "defense_only": True,
        "mode": "parallel" if any(
            (s.get("plan") or {}).get("reason", "").startswith("Parallel") for s in trace
        ) else "sequential",
    }
    log_event(case_id, customer_id, "verdict", {
        "action": verdict["action"],
        "risk_score": verdict["risk_score"],
        "confidence": verdict["confidence"],
    })
    return verdict


def run_agent(
    case: dict,
    max_iters: int = MAX_ITERS,
    fast: bool = True,
) -> dict[str, Any]:
    """
    Run agentic evidence gather + policy decision.
    fast=True (default): parallel tool gather for low latency.
    fast=False or LLM key set: sequential planner loop.
    """
    case_id = str(case.get("order_id") or case.get("case_id") or "unknown")
    customer_id = str(case.get("customer_id") or "")
    use_llm = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))

    if fast:
        log_event(case_id, customer_id, "reasoning", {
            "mode": "parallel_fast",
            "reason": "Dashboard/API fast path",
        })
        evidence, trace = _gather_evidence_fast(case)
        return _finalize(case, evidence, trace)

    evidence: dict[str, Any] = {}
    trace: list[dict] = []
    for i in range(max_iters):
        plan = _try_llm_plan(case, evidence) if use_llm else None
        if plan is None and use_llm:
            use_llm = False  # LLM failed once; skip it for the rest of this run
        if plan is None or plan.get("action") not in {"call_tool", "conclude"}:
            plan = _heuristic_plan(case, evidence, i)

        step = {"iteration": i, "plan": plan}
        log_event(case_id, customer_id, "reasoning", plan)

        if plan.get("action") == "conclude":
            trace.append(step)
            break

        tool = plan.get("tool")
        args = plan.get("args") or {}
        if tool in {"get_customer_history", "check_pattern", "check_shared_signals"}:
            args.setdefault("customer_id", customer_id)
        if tool == "get_order_context":
            args.setdefault("order_id", case_id)
        if tool in {"get_ensemble_risk_score", "get_shap_explanation"}:
            args.setdefault("features", _features_from_case(case))

        result = call_tool(tool, **args)
        evidence_key = {
            "get_customer_history": "history",
            "get_ensemble_risk_score": "score",
            "get_shap_explanation": "shap",
            "check_pattern": "pattern",
            "get_order_context": "order_context",
            "check_shared_signals": "shared_signals",
        }.get(tool, tool)
        evidence[evidence_key] = result
        step["tool_result"] = {evidence_key: result}
        trace.append(step)
        log_event(case_id, customer_id, "tool_call", {
            "tool": tool, "args": args, "result": result,
        })

    return _finalize(case, evidence, trace)
