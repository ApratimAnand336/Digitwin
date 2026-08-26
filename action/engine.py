"""
action/engine.py - Prescriptive Action Engine.

Evaluates real-time anomaly states and long-horizon forecasts against
the business rules configured in configs/rules.yaml.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Action:
    rule_id: str
    severity: str
    target: str
    recommendation: str
    message: str


class PrescriptiveEngine:
    """
    Evaluates system state against defined rules and returns actionable recommendations.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "rules.yaml"
            
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.rules = self.config.get("rules", [])
        self.thresholds = self.config.get("thresholds", {})

    def evaluate(self, state: Dict[str, Any]) -> List[Action]:
        """
        Evaluate a given station's state dict against all rules.
        
        Args:
            state: Dictionary containing metrics like `queue_depth_forecast_ratio`,
                   `anomaly_flagged`, `anomaly_type`, `station`, etc.
                   
        Returns:
            List of matching Action objects.
        """
        actions = []
        
        # Merge thresholds into the evaluation namespace
        eval_env = state.copy()
        eval_env.update(self.thresholds)
        
        for rule in self.rules:
            # Prepare condition string for Python eval
            cond_str = rule["condition"]
            cond_str = cond_str.replace("AND", "and").replace("OR", "or")
            # Special case for string literal in config
            cond_str = cond_str.replace("anomaly_type == defect", "anomaly_type == 'defect'")
            
            try:
                # Safe-ish evaluation since we control the config and environment
                # If a variable is missing (e.g., station has no anomaly_flagged), it raises NameError and skips.
                if eval(cond_str, {"__builtins__": {}}, eval_env):
                    
                    # Format message dynamically
                    # Provide defaults for formatting if not in state
                    fmt_args = {}
                    fmt_args["ratio"] = eval_env.get("queue_depth_forecast_ratio", 0.0)
                    fmt_args["confidence"] = eval_env.get("attribution_confidence", 0.0)
                    fmt_args["origin"] = eval_env.get("attribution_origin", "Unknown")
                    fmt_args["feeder"] = eval_env.get("feeder", "Unknown Feeder")
                    fmt_args["station"] = eval_env.get("station", "Unknown Station")
                    
                    msg = rule["message"].format(**fmt_args)
                    
                    # Resolve target alias if it points to a variable in state
                    target = rule["target"]
                    if target in state:
                        target = state[target]
                        
                    actions.append(Action(
                        rule_id=rule["id"],
                        severity=rule["severity"],
                        target=target,
                        recommendation=rule["recommendation"],
                        message=msg
                    ))
            except NameError:
                # Rule doesn't apply (missing metric in state)
                continue
            except Exception as e:
                # Catch syntax errors in config rules
                print(f"Warning: Failed to evaluate rule {rule['id']}: {e}")
                
        return actions
