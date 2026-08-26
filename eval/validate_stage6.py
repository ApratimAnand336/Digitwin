"""
eval/validate_stage6.py - Stage 6 validation: Prescriptive Action Engine.

Tests the rule evaluation engine with mocked states representing the output 
of Stages 4 (forecasts) and 5 (anomalies).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from action import PrescriptiveEngine

def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 6 VALIDATION - Prescriptive Action Engine")
    print("=" * 65)

    engine = PrescriptiveEngine()
    print(f"\n[1/3] Loaded {len(engine.rules)} rules from config.")

    # ---------------------------------------------------------
    # Scenario A: Downstream queue buildup due to bottleneck
    # ---------------------------------------------------------
    print("\n[2/3] Evaluating Scenario A: Forecasted Queue Buildup")
    state_a = {
        "station": "Engine Bay Prep (S4)",
        "feeder": "Underbody Framing (S3)",
        "queue_depth_forecast_ratio": 1.7,  # Above warning (1.5), below critical (2.0)
        "anomaly_flagged": False
    }
    
    actions_a = engine.evaluate(state_a)
    
    pass_a = False
    for a in actions_a:
        print(f"  -> [{a.severity}] {a.rule_id} -> Target: {a.target}")
        print(f"     Recommendation: {a.recommendation}")
        print(f"     Message: {a.message}")
        if a.rule_id == "QUEUE_BUILDUP_WARNING":
            pass_a = True

    if not pass_a:
        print("  [FAIL] Did not trigger QUEUE_BUILDUP_WARNING correctly.")

    # ---------------------------------------------------------
    # Scenario B: Critical defect anomaly identified
    # ---------------------------------------------------------
    print("\n[3/3] Evaluating Scenario B: Defect Anomaly Detected")
    state_b = {
        "station": "Brake Line Routing (S8)",
        "anomaly_flagged": True,
        "anomaly_type": "defect",
        "attribution_confidence": 0.85,
        "attribution_origin": "Brake Line Routing (S8)",
    }
    
    actions_b = engine.evaluate(state_b)
    
    pass_b_attr = False
    pass_b_def = False
    for a in actions_b:
        print(f"  -> [{a.severity}] {a.rule_id} -> Target: {a.target}")
        print(f"     Recommendation: {a.recommendation}")
        print(f"     Message: {a.message}")
        if a.rule_id == "ANOMALY_WITH_ATTRIBUTION": pass_b_attr = True
        if a.rule_id == "DEFECT_DETECTED": pass_b_def = True

    if not (pass_b_attr and pass_b_def):
        print("  [FAIL] Did not trigger defect and attribution rules correctly.")

    all_ok = pass_a and pass_b_attr and pass_b_def

    print("\n" + "=" * 65)
    if all_ok:
        print("VERDICT: PASS -- Action engine successfully mapped states to prescriptive actions.")
    else:
        print("VERDICT: STOP -- Action engine rules failed to trigger correctly.")
    print("=" * 65 + "\n")

    return all_ok

if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
