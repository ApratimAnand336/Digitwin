"""
eval — Validation scripts and metrics, one per pipeline stage.

Each sub-module provides a `run_validation()` function that:
  1. Loads the relevant stage's outputs.
  2. Computes metrics or runs visual checks.
  3. Prints a plain-language PASS / STOP verdict.

Populated stage-by-stage.
"""
