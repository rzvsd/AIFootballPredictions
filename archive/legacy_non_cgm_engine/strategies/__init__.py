"""Strategy registry and utilities."""

import importlib
from typing import Dict, Any, List, Callable

StrategyFunc = Callable[[Dict[str, Any], Any], List[Dict[str, Any]]]


def load_strategy(path: str) -> StrategyFunc:
    """
    Load a strategy function from a dotted path, e.g., strategies.elo_favourite_strategy:generate_candidates
    """
    if ":" in path:
        mod_name, fn_name = path.split(":", 1)
    else:
        mod_name, fn_name = path, "generate_candidates"
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn
