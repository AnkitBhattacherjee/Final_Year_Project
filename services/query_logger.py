"""
Query Logger — Real-Time Terminal & Console Execution Inspector
================================================================
Provides clear, formatted, colored logs in the terminal for every
query generated and executed (Local Router, Gemini Pass 1, Gemini Pattern B Code,
Sandbox Execution, and Fallbacks).
Helps developers and analysts immediately inspect generated code and diagnose errors.
"""

import sys
import json
from typing import Any, Dict, Optional

# ANSI Color Codes for terminal readability
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

def _safe_print(*args, **kwargs):
    """Safe UTF-8/fallback printer that immediately flushes to terminal output."""
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        ascii_text = text.encode("ascii", errors="replace").decode("ascii")
        print(ascii_text, **kwargs)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def log_query_start(question: str) -> None:
    """Print the incoming query banner."""
    divider = "=" * 80
    _safe_print(f"\n{_CYAN}{_BOLD}{divider}{_RESET}")
    _safe_print(f"{_CYAN}{_BOLD}📥 [QUERY RECEIVED]:{_RESET} \"{_BOLD}{question}{_RESET}\"")
    _safe_print(f"{_CYAN}{divider[:40]}{_RESET}")


def log_local_route(fn_name: str, fn_args: Dict[str, Any], answer: Optional[str] = None, chart: bool = False) -> None:
    """Log when a query is successfully routed by the Local Intent Router (Pattern A)."""
    _safe_print(f"{_GREEN}{_BOLD}📍 [ROUTE]:{_RESET} {_GREEN}Local Intent Router (Pattern A — Fast Local Execution){_RESET}")
    _safe_print(f"{_GREEN}{_BOLD}⚙️  [MATCHED FUNCTION]:{_RESET} {_BOLD}{fn_name}{_RESET}")
    _safe_print(f"{_DIM}   Arguments:{_RESET} {json.dumps(fn_args, default=str)}")
    if chart:
        _safe_print(f"{_BLUE}📊 [CHART]:{_RESET} Chart generated successfully")
    if answer:
        preview = answer.strip().replace("\n", " ")[:200]
        _safe_print(f"{_DIM}💬 [OUTPUT PREVIEW]:{_RESET} {preview}...")
    _safe_print(f"{_CYAN}{'=' * 80}{_RESET}\n")


def log_gemini_pass1(model_name: str, fn_name: str, fn_args: Dict[str, Any], answer: Optional[str] = None, chart: bool = False) -> None:
    """Log when Gemini Pass 1 selects a declared function."""
    _safe_print(f"{_MAGENTA}{_BOLD}📍 [ROUTE]:{_RESET} {_MAGENTA}Gemini Pass 1 (Function Calling via AI){_RESET}")
    _safe_print(f"{_DIM}🤖 [MODEL]:{_RESET} {model_name}")
    _safe_print(f"{_MAGENTA}{_BOLD}⚙️  [GEMINI SELECTED FUNCTION]:{_RESET} {_BOLD}{fn_name}{_RESET}")
    _safe_print(f"{_DIM}   Arguments:{_RESET} {json.dumps(fn_args, default=str)}")
    if chart:
        _safe_print(f"{_BLUE}📊 [CHART]:{_RESET} Chart generated successfully")
    if answer:
        preview = answer.strip().replace("\n", " ")[:200]
        _safe_print(f"{_DIM}💬 [OUTPUT PREVIEW]:{_RESET} {preview}...")
    _safe_print(f"{_CYAN}{'=' * 80}{_RESET}\n")


def log_pattern_b_code(model_name: str, code: str, success: bool, result: Optional[str] = None, error: Optional[str] = None) -> None:
    """Log generated Python / Pandas code from Gemini Pattern B and sandbox execution result."""
    _safe_print(f"{_YELLOW}{_BOLD}📍 [ROUTE]:{_RESET} {_YELLOW}Gemini Pattern B (Dynamic Pandas Code Generation Fallback){_RESET}")
    _safe_print(f"{_DIM}🤖 [MODEL]:{_RESET} {model_name}")
    _safe_print(f"{_YELLOW}{_BOLD}💻 [GENERATED PYTHON / PANDAS CODE]:{_RESET}")
    _safe_print(f"{_DIM}{'-' * 80}{_RESET}")
    for line in code.split("\n"):
        _safe_print(f"   {line}")
    _safe_print(f"{_DIM}{'-' * 80}{_RESET}")

    if success:
        _safe_print(f"{_GREEN}{_BOLD}⚡ [SANDBOX EXECUTION STATUS]:{_RESET} {_GREEN}SUCCESS{_RESET}")
        if result:
            preview = str(result).strip().replace("\n", " ")[:250]
            _safe_print(f"{_DIM}   Raw Result:{_RESET} {preview}...")
    else:
        _safe_print(f"{_RED}{_BOLD}❌ [SANDBOX EXECUTION ERROR]:{_RESET}")
        _safe_print(f"{_RED}   {error}{_RESET}")

    _safe_print(f"{_CYAN}{'=' * 80}{_RESET}\n")


def log_api_failure(model_name: str, error_msg: str, is_quota: bool = False) -> None:
    """Log a Gemini API failure (rate limit, quota, network, or auth)."""
    _safe_print(f"{_RED}{_BOLD}⚠️  [GEMINI API ERROR]:{_RESET}")
    _safe_print(f"{_DIM}🤖 [MODEL]:{_RESET} {model_name}")
    if is_quota:
        _safe_print(f"{_YELLOW}{_BOLD}🚨 [REASON]:{_RESET} {_YELLOW}API Quota / Rate Limit Exceeded (HTTP 429). Falling back to local heuristics.{_RESET}")
    else:
        _safe_print(f"{_RED}{_BOLD}🚨 [ERROR DETAILS]:{_RESET} {error_msg[:300]}")


def log_heuristic_fallback(reason: str, answer: Optional[str] = None) -> None:
    """Log when local heuristic fallback is used."""
    _safe_print(f"{_BLUE}{_BOLD}📍 [ROUTE]:{_RESET} {_BLUE}Local Heuristic Fallback{_RESET}")
    _safe_print(f"{_DIM}   Reason:{_RESET} {reason}")
    if answer:
        preview = answer.strip().replace("\n", " ")[:200]
        _safe_print(f"{_DIM}💬 [OUTPUT PREVIEW]:{_RESET} {preview}...")
    _safe_print(f"{_CYAN}{'=' * 80}{_RESET}\n")
