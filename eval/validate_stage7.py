"""
eval/validate_stage7.py - Stage 7 validation: Dashboard API integration.

Instantiates the unified DigitalTwinAPI, steps to a normal state,
steps to an anomalous state, and prints the structured JSON response
to prove all layers (Simulator -> Soft Sensor -> GCN -> Anomaly -> Actions)
are properly connected and formatted for the UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import DigitalTwinAPI

def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 7 VALIDATION - Dashboard API Data Layer")
    print("=" * 65)

    print("\n[1/3] Initializing full DigitalTwinAPI stack (may take ~10 seconds)...")
    try:
        api = DigitalTwinAPI(run_steps=200, seed=42)
        print("  [OK] Stack initialized successfully.")
    except Exception as e:
        print(f"  [FAIL] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # -------------------------------------------------------------------
    # Step 1: Normal Operation (t=30)
    # -------------------------------------------------------------------
    print("\n[2/3] Polling API at t=30 (Normal operation)")
    state_normal = api.get_state(t=30)
    
    print(f"  Timestep: {state_normal['timestep']}")
    print(f"  Stations returned: {len(state_normal['stations'])}")
    print(f"  Actions generated: {len(state_normal['actions'])}")
    
    if len(state_normal['actions']) > 0:
        print("  [WARNING] Expected no actions during normal operation, but got some:")
        for act in state_normal['actions']:
            print(f"    - {act['message']}")

    # -------------------------------------------------------------------
    # Step 2: Fault Operation (t=90)
    # (Bottleneck was scheduled to start at t=60 at Station 3)
    # -------------------------------------------------------------------
    print("\n[3/3] Polling API at t=90 (During S3 bottleneck fault)")
    
    # We must step sequentially to build the anomaly detector history properly
    for t in range(31, 91):
        state_fault = api.get_state(t)
        
    s3_state = next(s for s in state_fault['stations'] if s['station_id'] == 3)
    s4_state = next(s for s in state_fault['stations'] if s['station_id'] == 4)
    
    print(f"  S3 Anomaly Flagged: {s3_state['anomaly_flagged']} (Residual: {s3_state['residual_error']:.2f})")
    print(f"  S4 Queue Ratio: {s4_state['queue_depth_forecast_ratio']:.2f}")
    
    print(f"  Actions generated: {len(state_fault['actions'])}")
    for act in state_fault['actions']:
        print(f"  -> [{act['severity']}] {act['target']}: {act['recommendation']}")
        print(f"     {act['message']}")
        
    # Validation checks
    passed = True
    if not s3_state['anomaly_flagged']:
        print("  [FAIL] Expected S3 to be flagged anomalous at t=90.")
        passed = False
        
    if len(state_fault['actions']) == 0:
        print("  [FAIL] Expected actions to be generated due to S3 bottleneck and S4 queue.")
        passed = False
        
    print("\n" + "=" * 65)
    if passed:
        print("VERDICT: PASS -- Dashboard API integrates all layers correctly.")
    else:
        print("VERDICT: STOP -- API data structure is incorrect.")
    print("=" * 65 + "\n")
    
    return passed


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
