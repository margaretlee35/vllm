# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand a rendered multimodal prompt from the encoder to prefill / decode.

In an E/P/D deployment every server would otherwise render the same request:
fetch and decode the media, then run the HF processor (resize, normalise,
patchify). Only the encoder needs the pixels. Prefill loads the embeddings from
the EC cache and decode receives the KV cache, so both need only

- the prompt token ids, with the placeholders already expanded,
- each item's mm hash (the EC cache key) and placeholder range,
- the small per-item kwargs the language model reads, such as
  `video_grid_thw` for M-RoPE positions.

The encoder returns these as `rendered_prompt` (see `encode_rendered_prompt`),
and the proxy forwards them to P and D, which rebuild the engine prompt with
`decode_rendered_prompt` instead of rendering.

Kwargs above `MAX_FORWARDED_FIELD_BYTES` (the pixel tensors) are dropped. A
server that receives such an item must therefore find its embedding in the EC
cache, or already hold the prompt's KV. If it does not, the vision tower runs
with its pixel inputs missing and fails.
"""

import base64
from collections import OrderedDict
from typing import Any

import torch

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.inputs.data import ProcessorInputs
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
    PlaceholderRange,
    mm_inputs,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

MAX_FORWARDED_FIELD_BYTES = 4096

# Keys of the stand-in item that the shared-memory processor cache hands out
# in place of real kwargs.
_SHM_ADDRESS_KEYS = frozenset({"address", "monotonic_id"})

# Every forwarded tensor is small, so inline them all into one msgpack buffer.
_encoder = MsgpackEncoder(size_threshold=1 << 62)
_decoder = MsgpackDecoder(MultiModalKwargsItem)


def _nbytes(data: NestedTensors) -> int:
    if isinstance(data, torch.Tensor):
        return data.nbytes
    if isinstance(data, (list, tuple)):
        return sum(_nbytes(x) for x in data)
    return 0


def _slim_item(item: MultiModalKwargsItem) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            key: MultiModalFieldElem(data=elem.data, field=elem.field)
            for key, elem in item.items()
            if _nbytes(elem.data) <= MAX_FORWARDED_FIELD_BYTES
        }
    )


def _encode_item(item: MultiModalKwargsItem) -> str:
    (buf,) = _encoder.encode(_slim_item(item))
    return base64.b64encode(buf).decode("ascii")


class SlimItemCache:
    """Encoded slim kwargs by mm hash, kept on the rendering (encoder) side.

    With the IPC processor cache, the renderer replaces an item that the engine
    core already holds with `None`, so the kwargs are gone by the time the
    response is built. Remembering the slim encoding the first time the item
    passes through covers those repeats.
    """

    def __init__(self, max_entries: int = 1 << 16) -> None:
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._max_entries = max_entries

    def put(self, mm_hash: str, item: MultiModalKwargsItem) -> str:
        encoded = self._entries.get(mm_hash)
        if encoded is None:
            encoded = _encode_item(item)
            self._entries[mm_hash] = encoded
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(mm_hash)
        return encoded

    def get(self, mm_hash: str) -> str | None:
        encoded = self._entries.get(mm_hash)
        if encoded is not None:
            self._entries.move_to_end(mm_hash)
        return encoded


def encode_rendered_prompt(
    engine_prompt: ProcessorInputs,
    cache: SlimItemCache,
) -> tuple[dict[str, Any] | None, str | None]:
    """Serialise a rendered prompt for `decode_rendered_prompt`.

    Returns `(payload, None)`, or `(None, reason)` when the prompt cannot be
    forwarded and the receiver has to render the request itself.
    """
    prompt_type = engine_prompt.get("type")
    if prompt_type == "token":
        return {"prompt_token_ids": list(engine_prompt["prompt_token_ids"])}, None
    if prompt_type != "multimodal":
        return None, f"unsupported prompt type {prompt_type!r}"

    mm: dict[str, list[dict[str, Any]]] = {}
    for modality, hashes in engine_prompt["mm_hashes"].items():
        items = engine_prompt["mm_kwargs"].get(modality, [])
        placeholders = engine_prompt["mm_placeholders"][modality]
        entries = mm[modality] = []
        for idx, mm_hash in enumerate(hashes):
            item = items[idx] if idx < len(items) else None
            if item is None or _SHM_ADDRESS_KEYS.issuperset(item.keys()):
                kwargs = cache.get(mm_hash)
            else:
                kwargs = cache.put(mm_hash, item)
            if kwargs is None:
                return None, f"kwargs of {modality} item {mm_hash} not available"

            placeholder = placeholders[idx]
            entries.append(
                {
                    "hash": mm_hash,
                    "offset": placeholder.offset,
                    "length": placeholder.length,
                    "is_embed": (
                        None
                        if placeholder.is_embed is None
                        else placeholder.is_embed.tolist()
                    ),
                    "kwargs": kwargs,
                }
            )

    return {
        "prompt_token_ids": list(engine_prompt["prompt_token_ids"]),
        "mm": mm,
    }, None


def decode_rendered_prompt(
    payload: dict[str, Any],
    cache_salt: str | None = None,
) -> ProcessorInputs:
    """Rebuild the engine prompt from an `encode_rendered_prompt` payload."""
    prompt_token_ids = payload["prompt_token_ids"]
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(t, int) for t in prompt_token_ids
    ):
        raise ValueError("rendered_prompt.prompt_token_ids must be a list of ints")

    mm = payload.get("mm")
    if not mm:
        prompt: ProcessorInputs = {
            "type": "token",
            "prompt_token_ids": prompt_token_ids,
        }
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        return prompt

    mm_kwargs: dict[str, list[MultiModalKwargsItem]] = {}
    mm_hashes: dict[str, list[str]] = {}
    mm_placeholders: dict[str, list[PlaceholderRange]] = {}
    for modality, entries in mm.items():
        mm_kwargs[modality] = [
            _decoder.decode(base64.b64decode(e["kwargs"])) for e in entries
        ]
        mm_hashes[modality] = [str(e["hash"]) for e in entries]
        mm_placeholders[modality] = [
            PlaceholderRange(
                offset=int(e["offset"]),
                length=int(e["length"]),
                is_embed=(
                    None
                    if e.get("is_embed") is None
                    else torch.tensor(e["is_embed"], dtype=torch.bool)
                ),
            )
            for e in entries
        ]

    return mm_inputs(
        prompt_token_ids,
        MultiModalKwargsItems(mm_kwargs),
        mm_hashes,
        mm_placeholders,
        cache_salt=cache_salt,
    )


def text_only_conversation(messages: list[Any]) -> list[ConversationMessage]:
    """The request's messages without media, in place of the rendered conversation.

    The chat generators read the conversation only for the last message's text
    (`echo`) and for the number of earlier tool calls, so the media parts are
    not needed.
    """
    conversation: list[ConversationMessage] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        entry = ConversationMessage(role=msg["role"], content=content)
        if msg.get("tool_calls") is not None:
            entry["tool_calls"] = msg["tool_calls"]
        conversation.append(entry)
    return conversation
