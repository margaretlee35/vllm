# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.qwen2_5_vl_visionzip import (
    apply_qwen2_5_vision_zip,
    get_qwen2_5_vision_zip_num_tokens,
)


def test_get_qwen2_5_vision_zip_num_tokens():
    mm_config = MultiModalConfig(vision_zip_rate=0.75)
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
        vision_zip_rate=0.5,
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
