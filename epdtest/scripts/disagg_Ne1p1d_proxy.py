#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
disagg_encoder_proxy.py

Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)

For MM input we:
    1. Extract *every* image/audio item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("proxy")

app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
prefill_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio"}


def unique_preserving_order(urls: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result


def begin_endpoint_request(url: str, *, pd_stage: bool) -> None:
    app.state.endpoint_inflight_total[url] += 1
    if pd_stage:
        app.state.endpoint_inflight_pd[url] += 1


def end_endpoint_request(url: str, *, pd_stage: bool) -> None:
    app.state.endpoint_inflight_total[url] = max(
        0, app.state.endpoint_inflight_total[url] - 1
    )
    if pd_stage:
        app.state.endpoint_inflight_pd[url] = max(
            0, app.state.endpoint_inflight_pd[url] - 1
        )


def round_robin_pick(urls: list[str], cursor_key: str) -> str:
    if not urls:
        raise ValueError("round_robin_pick called with empty urls")
    idx = app.state.route_rr_cursor[cursor_key] % len(urls)
    app.state.route_rr_cursor[cursor_key] += 1
    return urls[idx]


def choose_encode_target_url(e_urls: list[str]) -> str:
    # Prefer prefill/decode endpoints when they are idle from PD work.
    if app.state.prefer_idle_prefill_decode_for_encode:
        e_url_set = set(e_urls)
        idle_candidates = [
            url
            for url in app.state.encode_idle_preferred_urls
            if url in e_url_set
            and app.state.endpoint_inflight_pd[url] == 0
            and app.state.endpoint_inflight_total[url]
            < app.state.idle_encode_max_inflight_per_server
        ]
        if idle_candidates:
            return round_robin_pick(idle_candidates, "idle_prefill_decode")

    return round_robin_pick(e_urls, "weighted_all")


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
) -> None:
    """
    1. Build one request *per MM item* with all text removed.
    2. Send them concurrently to the encode cluster.
    3. Raise if any of them fails.
    """
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return  # nothing to do

    logger.info("[%s] got %d multimodal items...", req_id, len(mm_items))

    async def send_encoder_item(item: dict, idx: int, target_url: str) -> dict:
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}
        encoder_req = {
            "model": orig_request.get("model"),
            "messages": [
                {"role": "user", "content": [item]},
            ],
            # Only need 1 token so the server actually runs the encoder path.
            "max_tokens": 1,
            "stream": False,
        }
        # Keep encode lower priority than prefill/decode requests.
        if app.state.encode_request_priority != 0:
            encoder_req["priority"] = app.state.encode_request_priority

        try:
            async with encode_session.post(
                f"{target_url}/v1/chat/completions",
                json=encoder_req,
                headers=headers,
            ) as response:
                response_text = await response.text()
                return {
                    "target_url": target_url,
                    "status": response.status,
                    "body": response_text,
                }
        except Exception as exc:
            return {
                "target_url": target_url,
                "exception": exc,
            }
        finally:
            end_endpoint_request(target_url, pd_stage=False)

    tasks = []
    chosen_targets: list[str] = []

    for idx, item in enumerate(mm_items):
        target_url = choose_encode_target_url(e_urls)
        chosen_targets.append(target_url)
        # Reserve slot immediately so the next target choice sees current load.
        begin_endpoint_request(target_url, pd_stage=False)
        tasks.append(send_encoder_item(item, idx, target_url))

    if app.state.prefer_idle_prefill_decode_for_encode:
        idle_hits = sum(
            1 for target in chosen_targets if target in app.state.encode_idle_preferred_set
        )
        logger.debug(
            "[%s] Encode idle-routing hits=%d/%d",
            req_id,
            idle_hits,
            len(chosen_targets),
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast if any sub-request failed
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "[%s] Encoder request #%d raised exception: %s",
                req_id,
                idx,
                result,
                exc_info=result,
            )
            raise HTTPException(
                status_code=502, detail=f"Encoder request failed: {str(result)}"
            )
        if "exception" in result:
            exc = result["exception"]
            logger.error(
                "[%s] Encoder request #%d to %s raised exception: %s",
                req_id,
                idx,
                result.get("target_url"),
                exc,
                exc_info=exc,
            )
            raise HTTPException(status_code=502, detail=f"Encoder request failed: {str(exc)}")
        if result["status"] != 200:
            detail = result["body"]
            logger.error(
                "[%s] Encoder request #%d to %s returned status %s: %s",
                req_id,
                idx,
                result["target_url"],
                result["status"],
                detail,
            )
            raise HTTPException(
                status_code=result["status"],
                detail=f"Encoder request failed: {detail}",
            )

    logger.info(
        "[%s] All %d encoder requests completed successfully", req_id, len(mm_items)
    )


async def maybe_prefill(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """
    - Do prefill-only task if p_url exist;
    - Return modified request data with kv transfer params (for nixl connector)
    - Else, skip and return the original request data for decode
    """
    if p_url:
        logger.info("[%s] Processing through prefill: %s", req_id, p_url)

        prefill_response_json = await process_prefill_stage(req_data, p_url, req_id)
        # for nixl connector to facilitate kv transfer...
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        return req_data
    else:
        return req_data


async def process_prefill_stage(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """Process request through Prefill stage and return kv_transfer_params"""
    logger.info("[%s] Sending prefill request to: %s", req_id, p_url)

    prefill_request = req_data.copy()
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]

    headers = {"x-request-id": req_id}
    try:
        begin_endpoint_request(p_url, pd_stage=True)
        try:
            async with prefill_session.post(
                f"{p_url}/v1/chat/completions",
                json=prefill_request,
                headers=headers,
            ) as prefill_response:
                prefill_response.raise_for_status()

                if prefill_response.status != 200:
                    error_text = await prefill_response.text()
                    logger.error(
                        "[%s] Prefill request failed with status %d: %s",
                        req_id,
                        prefill_response.status,
                        error_text,
                    )
                    raise HTTPException(
                        status_code=prefill_response.status,
                        detail={"error": "Prefill request failed", "message": error_text},
                    )
                logger.info("[%s] Prefill request completed successfully", req_id)

                return await prefill_response.json()
        finally:
            end_endpoint_request(p_url, pd_stage=True)

    except Exception as e:
        logger.error("Prefill processing failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Prefill processing error", "message": str(e)},
        ) from e


###############################################################################
# Middleware for request/response logging
###############################################################################


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    if app.state.p_urls:
        # only setup if prefill instance(s) exist
        prefill_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def forward_non_stream(
    req_data: dict, req_id: str, e_urls: list[str], p_url: str, d_url: str
) -> dict:
    try:
        # Step 1: Process through Encoder instance (if has MM input)
        await fanout_encoder_primer(req_data, e_urls, req_id)

        # Step 2: Process through Prefill instance
        req_data = await maybe_prefill(req_data, p_url, req_id)

        # Step 3: Process through Decode instance
        logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
        headers = {"x-request-id": req_id}

        # Non-streaming response
        begin_endpoint_request(d_url, pd_stage=True)
        try:
            async with decode_session.post(
                f"{d_url}/v1/chat/completions", json=req_data, headers=headers
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        finally:
            end_endpoint_request(d_url, pd_stage=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def forward_stream(
    req_data: dict, req_id: str, e_urls: list[str], p_url: str, d_url: str
) -> AsyncIterator[str]:
    try:
        # Step 1: Process through Encoder instance (if has MM input)
        await fanout_encoder_primer(req_data, e_urls, req_id)

        # Step 2: Process through Prefill instance
        req_data = await maybe_prefill(req_data, p_url, req_id)

        # Step 3: Process through Decode instance
        logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
        headers = {"x-request-id": req_id}

        # Streaming response
        begin_endpoint_request(d_url, pd_stage=True)
        try:
            async with decode_session.post(
                f"{d_url}/v1/chat/completions",
                json=req_data,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(1024):
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")
        finally:
            end_endpoint_request(d_url, pd_stage=True)

        logger.info("[%s] Streaming completed", req_id)

    except HTTPException:
        logger.exception("[%s] HTTPException in forward_stream", req_id)
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        raise HTTPException(
            status_code=500, detail=f"Proxy streaming error: {str(e)}"
        ) from e


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        e_urls = app.state.e_urls  # we want the full list for fan-out
        p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
        d_url = random.choice(app.state.d_urls)

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(req_data, req_id, e_urls, p_url, d_url),
                media_type="text/event-stream",
            )
        result = await forward_non_stream(req_data, req_id, e_urls, p_url, d_url)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls):
        if not urls:
            return "empty"
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, p_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls), healthy(app.state.p_urls), healthy(app.state.d_urls)
    )

    overall_healthy = all(
        status != "unhealthy" for status in (e_status, p_status, d_status)
    )

    status_code = 200 if overall_healthy else 503

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "prefill_cluster": p_status,
            "decode_cluster": d_status,
        },
        status_code=status_code,
    )


###############################################################################
# Simple profiler fan-out (unchanged except for sessions)
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    """
    POST `payload` to `url`.

    Returns
    -------
    • The decoded JSON body on success (2xx)
    • None if the endpoint does not exist (404)
    • Raises for anything else.
    """
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:  # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    prefill_task = (
        _post_if_available(prefill_session, f"{p_url}/{cmd}_profile", payload, headers)
        if p_url is not None
        else asyncio.sleep(0)
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, prefill_res, decode_res = await asyncio.gather(
        encode_task, prefill_task, decode_task
    )

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,  # may be None
        "prefill": prefill_res,  # may be None
        "decode": decode_res,  # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        required=True,
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") ',
            'to enable E->P->D, set "disable" or "none" to enable E->PD',
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--prefer-idle-prefill-decode-for-encode",
        action="store_true",
        help=(
            "Prefer prefill/decode endpoints for encode fanout when they are "
            "idle from prefill/decode workload perspective."
        ),
    )
    parser.add_argument(
        "--idle-encode-max-inflight-per-server",
        type=int,
        default=8,
        help=(
            "Max total in-flight requests per preferred endpoint when "
            "--prefer-idle-prefill-decode-for-encode is enabled."
        ),
    )
    parser.add_argument(
        "--encode-request-priority",
        type=int,
        default=100,
        help="Priority value used for proxy-generated encode primer requests.",
    )

    args = parser.parse_args()
    app.state.e_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_urls = [
        u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
    ]
    # handle prefill instances
    if args.prefill_servers_urls.lower() in ("disable", "none", ""):
        app.state.p_urls = []
        logger.info(
            "Disaggregated prefill phase explicitly disabled by user. Running E + PD..."
        )
    else:
        app.state.p_urls = [
            u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()
        ]
        logger.info("Disaggregated prefill phase is enabled. Running E + P + D...")

    app.state.prefer_idle_prefill_decode_for_encode = (
        args.prefer_idle_prefill_decode_for_encode
    )
    app.state.idle_encode_max_inflight_per_server = max(
        1, args.idle_encode_max_inflight_per_server
    )
    app.state.encode_request_priority = args.encode_request_priority
    app.state.endpoint_inflight_total = defaultdict(int)
    app.state.endpoint_inflight_pd = defaultdict(int)
    app.state.route_rr_cursor = defaultdict(int)
    app.state.encode_idle_preferred_urls = unique_preserving_order(
        app.state.p_urls + app.state.d_urls
    )
    app.state.encode_idle_preferred_set = set(app.state.encode_idle_preferred_urls)

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Prefill instances %s", app.state.p_urls)
    logger.info("Decode servers: %s", app.state.d_urls)
    logger.info(
        "Idle-aware encode routing: %s (max_inflight_per_server=%d, encode_priority=%d)",
        app.state.prefer_idle_prefill_decode_for_encode,
        app.state.idle_encode_max_inflight_per_server,
        app.state.encode_request_priority,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
