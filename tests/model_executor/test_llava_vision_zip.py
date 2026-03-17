# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.llava import (
    apply_llava_vision_zip,
    get_llava_vision_zip_num_tokens,
)


def test_get_llava_vision_zip_num_tokens():
    mm_config = MultiModalConfig(vision_zip_rate=0.75)
    assert get_llava_vision_zip_num_tokens(576, mm_config) == 144


def test_apply_llava_vision_zip_keeps_expected_shape():
    image_features = torch.arange(1, 1 + 6 * 4, dtype=torch.float32).view(1, 6, 4)
    cls_attention = torch.tensor([[0.9, 0.8, 0.4, 0.3, 0.2, 0.1]])
    key_states = torch.tensor(
        [
            [
                [4.0, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
                [0.0, 3.0],
                [0.0, 2.0],
                [0.0, 1.0],
            ]
        ]
    )
    mm_config = MultiModalConfig(
        vision_zip_rate=0.5,
        vision_zip_dominant_ratio=2.0 / 3.0,
    )

    compressed = apply_llava_vision_zip(
        image_features,
        cls_attention,
        key_states,
        mm_config,
    )

    assert compressed.shape == (1, 3, 4)
    assert torch.equal(compressed[0, 0], image_features[0, 0])
    assert torch.equal(compressed[0, 1], image_features[0, 1])
    expected_context = image_features[0, 2:].mean(dim=0)
    assert torch.allclose(compressed[0, 2], expected_context)
