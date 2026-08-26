"""
eval/validate_stage0.py - Stage 0 validation: repo scaffold.

Checks:
  1. All packages are importable.
  2. All config YAML files are loadable and contain expected keys.
  3. The data directory exists.
  4. Prints a PASS / STOP verdict.
"""

import sys
import importlib
from pathlib import Path


def check_imports() -> list[str]:
    """Return list of packages that failed to import."""
    packages = [
        "simulator",
        "soft_sensor",
        "model",
        "anomaly",
        "action",
        "dashboard",
        "eval",
    ]
    failures = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  [OK]   {pkg}")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {e}")
            failures.append(pkg)
    return failures


def check_configs(root: Path) -> list[str]:
    """Return list of config files that failed to load."""
    import yaml  # noqa: PLC0415

    configs = {
        "line_config.yaml": ["line", "stations", "sensors", "noise"],
        "graph.yaml": ["nodes", "edges"],
        "rules.yaml": ["thresholds", "rules"],
    }
    failures = []
    for fname, required_keys in configs.items():
        fpath = root / "configs" / fname
        try:
            with open(fpath) as f:
                data = yaml.safe_load(f)
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"  [FAIL] {fname}: missing keys {missing}")
                failures.append(fname)
            else:
                print(f"  [OK]   {fname} (keys: {list(data.keys())})")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {fname}: {e}")
            failures.append(fname)
    return failures


def check_directories(root: Path) -> list[str]:
    """Return list of expected directories that are missing."""
    expected = [
        "simulator", "soft_sensor", "model", "anomaly",
        "action", "dashboard", "eval", "data", "configs",
    ]
    failures = []
    for d in expected:
        p = root / d
        if p.is_dir():
            print(f"  [OK]   {d}/")
        else:
            print(f"  [FAIL] {d}/ -- directory missing")
            failures.append(d)
    return failures


def run_validation() -> bool:
    root = Path(__file__).parent.parent
    all_ok = True

    print("\n" + "=" * 60)
    print("STAGE 0 VALIDATION - Repo Scaffold")
    print("=" * 60)

    print("\n[1/3] Checking directory structure ...")
    dir_failures = check_directories(root)
    if dir_failures:
        all_ok = False

    print("\n[2/3] Checking package imports ...")
    import_failures = check_imports()
    if import_failures:
        all_ok = False

    print("\n[3/3] Checking config files ...")
    config_failures = check_configs(root)
    if config_failures:
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("VERDICT: PASS -- Stage 0 scaffold is clean.")
        print("         All packages import, all configs load, all dirs exist.")
    else:
        problems = dir_failures + import_failures + config_failures
        print(f"VERDICT: STOP -- {len(problems)} problem(s) found:")
        for p in problems:
            print(f"           * {p}")
        print("         Fix these before proceeding to Stage 1.")
    print("=" * 60 + "\n")

    return all_ok


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
