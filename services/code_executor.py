"""
Pattern B — Sandboxed Pandas Code Execution (Fallback Only)
============================================================
Used ONLY when no pre-defined Pattern A function matches the user's question.

Security model:
  - df is passed as a READ-ONLY copy (mutations don't escape the sandbox)
  - __builtins__ is restricted to a whitelist of safe built-ins
  - No import statements allowed
  - Execution timeout: 8 seconds
  - Generated code MUST assign its answer to a variable named `result`

Logging:
  - Every fallback question + generated code is logged to logs/fallback_queries.jsonl
  - Review this log regularly to decide which new functions to add to the library
  - Goal: fallback usage should shrink over time as the library grows
"""

import os
import json
import threading
import datetime
import traceback
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Path for the fallback question log (in project root/logs/)
_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "fallback_queries.jsonl",
)

import math

# Whitelisted built-ins — everything in this list is available in the sandbox.
# Nothing else is importable or callable.
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max,
    "min": min, "print": print, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_fallback_query(
    question: str,
    generated_code: str,
    success: bool,
    result_preview: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    Append a structured log entry for a fallback question.
    Failures are also logged so you can see what code Gemini generated
    and why it broke — critical for iterating on the library.
    """
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "code": generated_code,
        "success": success,
        "result_preview": (result_preview or "")[:300],
        "error": error,
    }
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Never let logging crash the user request


# ---------------------------------------------------------------------------
# Sandbox Executor
# ---------------------------------------------------------------------------

def execute_sandboxed(df: pd.DataFrame, code: str, timeout: int = 8) -> Dict:
    """
    Execute pandas code in an isolated namespace with timeout and builtins whitelist.

    The code has access to:
      df    — read-only copy of the DataFrame
      pd    — pandas
      np    — numpy
      math  — math module
      (whitelisted built-ins only — no open(), __import__(), etc.)

    The code MUST set a variable named `result`.

    Returns:
      { "result": str | None, "result_df": pd.DataFrame | None, "error": str | None }
    """
    container: Dict = {"result": None, "result_df": None, "error": None}

    def _run():
        try:
            namespace = {
                "__builtins__": _SAFE_BUILTINS,
                "df": df.copy(),   # copy → mutations don't propagate
                "pd": pd,
                "np": np,
                "math": math,
            }
            exec(compile(code, "<pattern_b_sandbox>", "exec"), namespace)  # noqa: S102
            raw = namespace.get("result")
            if raw is None:
                container["error"] = (
                    "Code executed successfully but did not set a `result` variable. "
                    "Add `result = ...` at the end of your code."
                )
            else:
                if isinstance(raw, pd.DataFrame):
                    container["result_df"] = raw.copy()
                    container["result"] = raw.to_string(index=False if not isinstance(raw.index, pd.MultiIndex) else True)
                elif isinstance(raw, pd.Series):
                    container["result_df"] = raw.reset_index()
                    container["result"] = raw.to_string()
                elif isinstance(raw, dict):
                    container["result"] = json.dumps(raw, indent=2, default=str)
                elif isinstance(raw, (int, float, np.integer, np.floating)):
                    container["result"] = f"{raw:,.2f}" if isinstance(raw, float) or isinstance(raw, np.floating) else f"{raw:,}"
                else:
                    container["result"] = str(raw)
        except SyntaxError as e:
            container["error"] = f"Syntax error in generated code: {e}"
        except Exception:
            container["error"] = traceback.format_exc(limit=3)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        container["error"] = (
            f"Code execution timed out after {timeout}s. "
            "The question may require a simpler approach."
        )

    return container


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_fallback_log_path() -> str:
    """Returns the absolute path to the fallback query log file."""
    return _LOG_PATH


def get_fallback_log_entries(last_n: int = 50):
    """
    Read the last N entries from the fallback log.
    Useful for the admin endpoint that shows which questions need new functions.
    """
    if not os.path.exists(_LOG_PATH):
        return []
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        return entries[-last_n:]
    except Exception:
        return []
