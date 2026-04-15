# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.qwen2_5_vl_prune import (
    apply_qwen2_5_cdpruner,
    apply_qwen2_5_vision_zip,
    get_qwen2_5_cdpruner_num_tokens,
    get_qwen2_5_vision_zip_num_tokens,
)


def test_get_qwen2_5_vision_zip_num_tokens():
    mm_config = MultiModalConfig(vt_prune_rate=0.75)
    assert get_qwen2_5_vision_zip_num_tokens(576, mm_config) == 144


def test_apply_qwen2_5_vision_zip_keeps_positions_in_token_order():
    image_features = torch.arange(1, 1 + 6 * 4, dtype=torch.float32).view(6, 4)
    importance_scores = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.3, 0.4])
    key_states = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 4.0],
            [4.0, 0.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ]
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 2.0, 1.0, 2.0],
        ]
    )
    mm_config = MultiModalConfig(
        vt_prune_rate=0.5,
        vision_zip_dominant_ratio=2.0 / 3.0,
    )

    compressed, compressed_positions = apply_qwen2_5_vision_zip(
        image_features,
        importance_scores,
        key_states,
        positions,
        mm_config,
    )

    assert compressed.shape == (3, 4)
    assert torch.equal(compressed[0], image_features[0])
    assert torch.equal(compressed[1], image_features[2])
    expected_context = image_features[[1, 3, 4, 5]].mean(dim=0)
    assert torch.allclose(compressed[2], expected_context)

    assert torch.equal(compressed_positions[0], positions[0])
    assert torch.equal(compressed_positions[1], positions[2])
    assert torch.equal(compressed_positions[2], positions[1])


def test_get_qwen2_5_cdpruner_num_tokens():
    mm_config = MultiModalConfig(
        visual_token_pruning_method="cdpruner",
        vt_prune_rate=0.75,
    )
    assert get_qwen2_5_cdpruner_num_tokens(576, mm_config) == 144


def test_apply_qwen2_5_cdpruner_keeps_positions_in_token_order():
    image_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    mm_config = MultiModalConfig(
        visual_token_pruning_method="cdpruner",
        vt_prune_rate=0.5,
    )

    compressed, compressed_positions = apply_qwen2_5_cdpruner(
        image_features,
        positions,
        mm_config,
    )

    assert compressed.shape == (2, 3)
    assert torch.equal(compressed, image_features[[0, 2]])
    assert torch.equal(compressed_positions, positions[[0, 2]])


def test_apply_qwen2_5_cdpruner_text_conditioning_filters_irrelevant_token():
    image_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    positions = torch.arange(16, dtype=torch.float32).view(4, 4)
    mm_config = MultiModalConfig(
        visual_token_pruning_method="cdpruner",
        vt_prune_rate=0.5,
    )
    text_embedding = torch.tensor([1.0, 0.0], dtype=torch.float32)

    compressed, _ = apply_qwen2_5_cdpruner(
        image_features,
        positions,
        mm_config,
        text_embedding=text_embedding,
    )

    irrelevant = image_features[3]
    contains_irrelevant = torch.any(
        torch.all(torch.isclose(compressed, irrelevant.unsqueeze(0)), dim=-1)
    )
    assert not contains_irrelevant
