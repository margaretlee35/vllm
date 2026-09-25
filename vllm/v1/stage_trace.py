# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, cross-process stage tracing for disaggregated (E/P/D) serving.

Set VLLM_STAGE_TRACE_DIR to enable. Every process appends JSON lines to
``<dir>/<role>-<pid>.jsonl``; VLLM_STAGE_TRACE_ROLE labels the server
(e.g. encoder / prefill / decode). Timestamps are ``time.time()``, so events
written by different processes on the same host share a clock and can be
stitched together per request afterwards.

GPU spans (``sync=True``) synchronize the device on entry and exit so the
measured time belongs to that span. That removes CPU/GPU overlap: use traced
runs to attribute latency to stages, not to measure peak throughput.
"""

import contextvars
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

TRACE_DIR = os.environ.get("VLLM_STAGE_TRACE_DIR") or None
ENABLED = TRACE_DIR is not None
ROLE = os.environ.get("VLLM_STAGE_TRACE_ROLE", "vllm")

# External request id (X-Request-Id) of the request the current asyncio task is
# serving. Spans emitted from that task without an explicit `req` get it.
current_request: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "stage_trace_request", default=None
)

_lock = threading.Lock()
_fh = None
_fh_pid = -1


def _write(line: str) -> None:
    global _fh, _fh_pid
    with _lock:
        # Reopen after fork so each process writes its own file.
        if _fh is None or _fh_pid != os.getpid():
            assert TRACE_DIR is not None
            os.makedirs(TRACE_DIR, exist_ok=True)
            path = os.path.join(TRACE_DIR, f"{ROLE}-{os.getpid()}.jsonl")
            _fh = open(path, "a", buffering=1)  # noqa: SIM115
            _fh_pid = os.getpid()
        _fh.write(line + "\n")


def emit(event: str, ts: float | None = None, **fields: Any) -> None:
    if not ENABLED:
        return
    if "req" not in fields and (req := current_request.get()) is not None:
        fields["req"] = req
    record = {
        "ts": time.time() if ts is None else ts,
        "role": ROLE,
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    _write(json.dumps(record, separators=(",", ":")))


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def span(event: str, sync: bool = False, **fields: Any) -> Iterator[dict[str, Any]]:
    """Emit `event` with its start time and `dur_ms` when the block exits.

    Yields the fields dict, so the block can attach values it computes.
    """
    if not ENABLED:
        yield fields
        return
    if sync:
        _sync()
    start = time.time()
    t0 = time.perf_counter()
    try:
        yield fields
    finally:
        if sync:
            _sync()
        emit(event, ts=start, dur_ms=(time.perf_counter() - t0) * 1e3, **fields)


def tensor_nbytes(obj: Any) -> int:
    """Total bytes of the tensors nested in dicts / lists / tuples."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(tensor_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_nbytes(v) for v in obj)
    return 0
