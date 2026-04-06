# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.model import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-VL-3B-Instruct"])
def test_qwen2_5_vl_visionzip_processor_reduces_image_placeholders(
    image_assets: ImageTestAssets,
    model_id: str,
):
    base_ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    base_processor = MULTIMODAL_REGISTRY.create_processor(base_ctx.model_config)
    base_tokenizer = base_processor.info.get_tokenizer()

    model_info = HF_EXAMPLE_MODELS.get_hf_info(
        "Qwen2_5_VLPruneForConditionalGeneration"
    )
    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        seed=0,
        limit_mm_per_prompt={"image": 1},
        hf_overrides=model_info.hf_overrides,
        max_model_len=model_info.max_model_len,
        vt_prune_rate=0.5,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    mm_data = {"image": [image_assets[0].pil_image]}

    base_processed_inputs = base_processor(
        prompt,
        mm_items=base_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )
    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    hf_processor = base_processor.info.get_hf_processor()
    image_token_id = base_tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    base_img_tok_count = base_processed_inputs["prompt_token_ids"].count(image_token_id)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)

    assert img_tok_count == (base_img_tok_count + 1) // 2
