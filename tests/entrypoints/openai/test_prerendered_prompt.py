# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round trip of the encoder's rendered prompt to prefill / decode."""

import json

import torch

from vllm.entrypoints.openai.chat_completion.prerendered import (
    SlimItemCache,
    decode_rendered_prompt,
    encode_rendered_prompt,
    text_only_conversation,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
    mm_inputs,
)


def _video_item(t: int, h: int, w: int) -> MultiModalKwargsItem:
    num_patches = t * h * w
    return MultiModalKwargsItem(
        {
            "pixel_values_videos": MultiModalFieldElem(
                data=torch.randn(num_patches, 1176),
                field=MultiModalFlatField(slices=[slice(0, num_patches)]),
            ),
            "video_grid_thw": MultiModalFieldElem(
                data=torch.tensor([t, h, w]),
                field=MultiModalBatchedField(),
            ),
            "second_per_grid_ts": MultiModalFieldElem(
                data=torch.tensor(0.25),
                field=MultiModalBatchedField(),
            ),
        }
    )


def _prompt(items: list[MultiModalKwargsItem | None]):
    # 5 text tokens, 64 placeholder tokens per item, 3 text tokens
    token_ids = [1] * 5 + [151656] * (64 * len(items)) + [2] * 3
    return mm_inputs(
        token_ids,
        MultiModalKwargsItems({"video": items}),
        {"video": [f"hash{i}" for i in range(len(items))]},
        {
            "video": [
                PlaceholderRange(offset=5 + 64 * i, length=64)
                for i in range(len(items))
            ]
        },
    )


def test_round_trip_drops_pixels_keeps_metadata():
    engine_prompt = _prompt([_video_item(2, 16, 8), _video_item(4, 8, 8)])

    payload, reason = encode_rendered_prompt(engine_prompt, SlimItemCache())
    assert reason is None
    # What travels through the proxy is JSON, and small.
    wire = json.dumps(payload)
    assert len(wire) < 4096

    rebuilt = decode_rendered_prompt(json.loads(wire), cache_salt="salt")
    assert rebuilt["type"] == "multimodal"
    assert rebuilt["prompt_token_ids"] == engine_prompt["prompt_token_ids"]
    assert rebuilt["mm_hashes"] == {"video": ["hash0", "hash1"]}
    assert rebuilt["cache_salt"] == "salt"

    placeholders = rebuilt["mm_placeholders"]["video"]
    assert [(p.offset, p.length) for p in placeholders] == [(5, 64), (69, 64)]

    items = rebuilt["mm_kwargs"]["video"]
    assert "pixel_values_videos" not in items[0]
    assert items[0]["video_grid_thw"].data.tolist() == [2, 16, 8]
    assert items[1]["video_grid_thw"].data.tolist() == [4, 8, 8]
    assert items[1]["second_per_grid_ts"].data.item() == 0.25
    assert isinstance(items[0]["video_grid_thw"].field, MultiModalBatchedField)


def test_processor_cache_hit_uses_slim_cache():
    cache = SlimItemCache()
    first, _ = encode_rendered_prompt(_prompt([_video_item(2, 16, 8)]), cache)

    # The IPC processor cache hands out None for an item the engine holds.
    repeat, reason = encode_rendered_prompt(_prompt([None]), cache)
    assert reason is None
    assert repeat == first

    # Never seen with data: cannot be forwarded.
    missing, reason = encode_rendered_prompt(_prompt([None]), SlimItemCache())
    assert missing is None
    assert "hash0" in reason


def test_is_embed_round_trip():
    engine_prompt = _prompt([_video_item(2, 16, 8)])
    is_embed = torch.zeros(64, dtype=torch.bool)
    is_embed[1:63] = True
    engine_prompt["mm_placeholders"]["video"][0] = PlaceholderRange(
        offset=5, length=64, is_embed=is_embed
    )

    payload, _ = encode_rendered_prompt(engine_prompt, SlimItemCache())
    rebuilt = decode_rendered_prompt(json.loads(json.dumps(payload)))
    placeholder = rebuilt["mm_placeholders"]["video"][0]
    assert torch.equal(placeholder.is_embed, is_embed)
    assert placeholder.get_num_embeds() == 62


def test_text_only_prompt():
    payload, _ = encode_rendered_prompt(
        {"type": "token", "prompt_token_ids": [1, 2, 3]}, SlimItemCache()
    )
    assert decode_rendered_prompt(payload) == {
        "type": "token",
        "prompt_token_ids": [1, 2, 3],
    }


def test_text_only_conversation():
    conversation = text_only_conversation(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            },
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        ]
    )
    assert conversation == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
    ]
