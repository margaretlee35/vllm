# SPDX-License-Identifier: Apache-2.0
"""Per-request latency breakdown across E / P / D from a stage trace.

Reads the <role>-<pid>.jsonl files that `vllm/v1/stage_trace.py` writes when
VLLM_STAGE_TRACE_DIR is set (run.sh --stage-trace), stitches the events of all
three servers together by the proxy's request id, and prints each stage's
latency in pipeline order.

Batched GPU work (enc_h2d, enc_forward) is charged in full to every request in
the batch: that is the time each of them waited, not a per-request share.

Usage: python stage_breakdown.py <trace_dir> [--csv out.csv] [--json out.json]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# (name, description) in pipeline order.
STAGES = [
    ("e_decode", "E  video decode (media load, incl. thread-pool wait)"),
    ("e_hf", "E  HF processor (resize / normalize / patchify)"),
    ("e_render_other", "E  rest of render (chat template, tokenize)"),
    ("e_queue", "E  API -> engine core -> scheduled"),
    ("e_h2d", "E  collate + pin + CPU->GPU copy"),
    ("e_encode", "E  vision encoder forward"),
    ("e_ec_save", "E  embedding GPU->CPU + file write"),
    ("e_to_p", "E done -> P request arrives (response + proxy hop)"),
    ("p_render", "P  render (video decode + HF processor, again)"),
    ("p_queue", "P  API -> engine core -> scheduled"),
    ("p_ec_load", "P  embedding file read + CPU->GPU"),
    ("p_prefill", "P  prefill forward (sched -> first token, minus load)"),
    ("p_to_d", "P done -> D request arrives (response + proxy hop)"),
    ("d_render", "D  render (video decode + HF processor, again)"),
    ("d_queue", "D  API -> engine core -> KV load starts"),
    ("d_kv", "D  NIXL KV transfer P->D"),
    ("d_first_token", "D  KV received -> first token"),
    ("d_decode_rest", "D  first token -> last token"),
]
TOTALS = [
    ("ttft", "E arrival -> D first token (server-side TTFT)"),
    ("e2e", "E arrival -> D finished"),
]

_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def parent_id(req: str, internal: bool) -> str:
    """Proxy request id from an API-level (x-request-id) or engine-level id.

    Engine ids are 'chatcmpl-<x-request-id>-<8 hex>'; the proxy's encoder
    requests use '<parent>:<item>:<6 hex>' as x-request-id.
    """
    if internal:
        req = _SUFFIX.sub("", req.removeprefix("chatcmpl-"))
    return req.split(":")[0]


def load(trace_dir: Path) -> dict[str, dict]:
    events = [
        json.loads(line)
        for f in sorted(trace_dir.glob("*.jsonl"))
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    # role -> mm hash -> parent, from the scheduler's view of each request
    mm_owner: dict[str, dict[str, str]] = defaultdict(dict)
    for e in events:
        if e["event"] == "sched":
            for h in e.get("mm") or ():
                mm_owner[e["role"]][h] = parent_id(e["req"], internal=True)

    reqs: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for e in events:
        role, ev = e["role"][0], e["event"]  # e / p / d
        if ev in (
            "api_arrival",
            "api_render",
            "api_submit",
            "media_load",
            "hf_processor",
        ):
            if "req" not in e:
                continue
            owners = [parent_id(e["req"], internal=False)]
        elif ev in ("enc_h2d", "enc_forward"):
            owners = sorted({parent_id(r, internal=True) for r in e["reqs"]})
        elif ev in ("ec_save", "ec_load"):
            owner = mm_owner[e["role"]].get(e["mm"])
            owners = [owner] if owner else []
        else:
            owners = [parent_id(e["req"], internal=True)]
        for o in owners:
            reqs[o][f"{role}.{ev}"].append(e)
    return reqs


def breakdown(r: dict) -> dict[str, float]:
    def first(key):
        return min(e["ts"] for e in r[key]) if r.get(key) else np.nan

    def total_ms(key):
        return sum(e["dur_ms"] for e in r[key]) if r.get(key) else np.nan

    ms = 1e3
    out = {}
    out["e_decode"] = total_ms("e.media_load")
    out["e_hf"] = total_ms("e.hf_processor")
    out["e_render_other"] = total_ms("e.api_render") - out["e_decode"] - out["e_hf"]
    out["e_queue"] = (first("e.sched") - first("e.api_submit")) * ms
    out["e_h2d"] = total_ms("e.enc_h2d")
    out["e_encode"] = total_ms("e.enc_forward")
    out["e_ec_save"] = total_ms("e.ec_save")
    out["e_to_p"] = (first("p.api_arrival") - first("e.finished")) * ms
    out["p_render"] = total_ms("p.api_render")
    out["p_queue"] = (first("p.sched") - first("p.api_submit")) * ms
    out["p_ec_load"] = total_ms("p.ec_load")
    out["p_prefill"] = (first("p.first_token") - first("p.sched")) * ms - np.nan_to_num(
        out["p_ec_load"]
    )
    out["p_to_d"] = (first("d.api_arrival") - first("p.finished")) * ms
    out["d_render"] = total_ms("d.api_render")
    out["d_queue"] = (first("d.kv_wait_start") - first("d.api_submit")) * ms
    out["d_kv"] = (first("d.kv_recv_done") - first("d.kv_wait_start")) * ms
    out["d_first_token"] = (first("d.first_token") - first("d.kv_recv_done")) * ms
    out["d_decode_rest"] = (first("d.finished") - first("d.first_token")) * ms
    start = first("e.api_arrival")
    out["ttft"] = (first("d.first_token") - start) * ms
    out["e2e"] = (first("d.finished") - start) * ms
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="drop the N earliest requests (cold start)",
    )
    args = p.parse_args()

    reqs = load(args.trace_dir)
    rows = {rid: breakdown(r) for rid, r in reqs.items()}
    # Only requests that went all the way through E, P and D.
    rows = {rid: b for rid, b in rows.items() if not np.isnan(b["e2e"])}
    order = sorted(
        rows, key=lambda rid: min(e["ts"] for e in reqs[rid]["e.api_arrival"])
    )
    order = order[args.skip_first :]
    if not order:
        raise SystemExit(f"no complete E->P->D requests in {args.trace_dir}")

    names = [n for n, _ in STAGES + TOTALS]
    desc = dict(STAGES + TOTALS)
    mean_ttft = np.nanmean([rows[r]["ttft"] for r in order])
    summary = {}
    print(f"\n{len(order)} requests (of {len(reqs)} traced), ms\n")
    print(
        f"{'stage':<15}{'mean':>9}{'p50':>9}{'p90':>9}{'p99':>9}{'%ttft':>7}"
        "  description"
    )
    for n in names:
        v = np.array([rows[r][n] for r in order], dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            print(
                f"{n:<15}{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'':>7}  {desc[n]} (no events)"
            )
            continue
        s = {
            "mean": v.mean(),
            "p50": np.percentile(v, 50),
            "p90": np.percentile(v, 90),
            "p99": np.percentile(v, 99),
        }
        summary[n] = s
        share = (
            ""
            if n in ("ttft", "e2e", "d_decode_rest")
            else f"{100 * s['mean'] / mean_ttft:.0f}%"
        )
        if n == "ttft":
            print("-" * 60)
        print(
            f"{n:<15}{s['mean']:>9.1f}{s['p50']:>9.1f}{s['p90']:>9.1f}"
            f"{s['p99']:>9.1f}{share:>7}  {desc[n]}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["request_id", *names])
            for r in order:
                w.writerow([r, *(round(rows[r][n], 3) for n in names)])
    if args.json:
        args.json.write_text(
            json.dumps(
                {"num_requests": len(order), "stages": summary, "descriptions": desc},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
