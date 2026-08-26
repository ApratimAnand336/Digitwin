"""
eval/validate_stage8.py - Stage 8 validation

Validates the Streamlit app parses correctly.
"""
import sys
from pathlib import Path

def run_validation():
    print("\n" + "=" * 65)
    print("STAGE 8 VALIDATION - Streamlit App")
    print("=" * 65)
    
    try:
        import streamlit
        print("  [OK] streamlit is installed.")
    except ImportError:
        print("  [FAIL] streamlit is not installed.")
        return False
        
    try:
        import plotly
        print("  [OK] plotly is installed.")
    except ImportError:
        print("  [FAIL] plotly is not installed.")
        return False
        
    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    if app_path.exists():
        print(f"  [OK] Found {app_path.name}.")
    else:
        print("  [FAIL] App file missing.")
        return False
        
    # Syntax check
    try:
        compile(app_path.read_text("utf-8"), str(app_path), 'exec')
        print("  [OK] App syntax is valid.")
    except Exception as e:
        print(f"  [FAIL] App syntax error: {e}")
        return False
        
    print("\n" + "=" * 65)
    print("VERDICT: PASS -- Dashboard is ready to launch.")
    print("=" * 65 + "\n")
    return True

if __name__ == "__main__":
    run_validation()
