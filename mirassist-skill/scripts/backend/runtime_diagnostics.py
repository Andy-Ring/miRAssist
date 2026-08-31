"""Process-start diagnostics used for hosted native-crash isolation."""
from __future__ import annotations
import faulthandler
import os
from contextlib import contextmanager
from typing import Any, Iterator

_THREAD_LIMITS = {"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1", "BLIS_NUM_THREADS":"1"}
_STAGES = ("planner", "retrieval", "ranking", "prompt", "synthesis")

class DebugStageStop(RuntimeError):
    def __init__(self, stage: str):
        super().__init__(f"debug stop after {stage}")
        self.stage = stage

def initialize_process() -> None:
    for name, value in _THREAD_LIMITS.items():
        os.environ.setdefault(name, value)
    if not faulthandler.is_enabled():
        faulthandler.enable()
    print(f"[miRAssist] process diagnostics initialized: faulthandler={faulthandler.is_enabled()} native_threads=1", flush=True)

def debug_stage() -> str | None:
    value = (os.getenv("MIRASSIST_DEBUG_STAGE") or "").strip().lower()
    return value if value in _STAGES else None

def checkpoint(stage: str) -> None:
    if debug_stage() == stage:
        print(f"[miRAssist] debug stage stop requested after {stage}", flush=True)
        raise DebugStageStop(stage)

def trace(message: str) -> None:
    print(f"[miRAssist][trace] {message}", flush=True)

def frame_info(frame: Any) -> str:
    try:
        return f"rows={len(frame)} cols={len(frame.columns)} columns={list(frame.columns)[:12]}"
    except Exception:
        return type(frame).__name__

@contextmanager
def traced(operation: str, frame: Any = None) -> Iterator[None]:
    trace(f"before {operation}" + (f" ({frame_info(frame)})" if frame is not None else ""))
    try:
        yield
    except BaseException as exc:
        trace(f"failed {operation}: {type(exc).__name__}: {exc}")
        raise
    else:
        trace(f"after {operation}" + (f" ({frame_info(frame)})" if frame is not None else ""))
