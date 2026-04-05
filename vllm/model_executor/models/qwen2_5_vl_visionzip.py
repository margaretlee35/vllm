# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VisionZip wrapper for Qwen2.5-VL.

This keeps the base Qwen2.5-VL path unchanged and exposes VisionZip as a
separate architecture that can be selected via HF architecture override.

The implementation follows the VisionZip Qwen2.5-VL integration pattern from
JIA-Lab's public repository by collecting attention-derived token importance
from a configurable vision layer and pruning image tokens before they are fed
into the language model.
"""

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Protocol

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate

from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionBlock,
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor,
)
from .utils import maybe_prefix
from .utils import cast_overflow_tensors

logger = init_logger(__name__)


class _VisionZipMultiModalConfig(Protocol):
    vision_zip_rate: float | None
    vision_zip_dominant_ratio: float
    vision_zip_attention_layer: int
    vision_zip_debug: bool

    def is_vision_zip_enabled(self) -> bool: ...


def _get_qwen2_5_vision_zip_config(
    mm_config: _VisionZipMultiModalConfig | None,
) -> _VisionZipMultiModalConfig | None:
    if mm_config is None or not mm_config.is_vision_zip_enabled():
        return None
    return mm_config


def get_qwen2_5_vision_zip_num_tokens(
    num_tokens: int,
    mm_config: _VisionZipMultiModalConfig | None,
) -> int:
    vision_zip_config = _get_qwen2_5_vision_zip_config(mm_config)
    if vision_zip_config is None:
        return num_tokens

    keep_tokens = math.ceil(num_tokens * (1.0 - vision_zip_config.vision_zip_rate))
    return max(1, min(num_tokens, keep_tokens))


def _get_qwen2_5_vision_zip_split(
    keep_tokens: int,
    dominant_ratio: float,
) -> tuple[int, int]:
    dominant_tokens = math.ceil(keep_tokens * dominant_ratio)
    dominant_tokens = max(1, min(keep_tokens, dominant_tokens))
    contextual_tokens = keep_tokens - dominant_tokens
    return dominant_tokens, contextual_tokens


def apply_qwen2_5_vision_zip(
    image_features: torch.Tensor,
    importance_scores: torch.Tensor,
    key_states: torch.Tensor,
    positions: torch.Tensor,
    mm_config: _VisionZipMultiModalConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vision_zip_config = _get_qwen2_5_vision_zip_config(mm_config)
    if vision_zip_config is None:
        return image_features, positions

    num_tokens, hidden_size = image_features.shape
    if num_tokens == 0:
        return image_features, positions

    keep_tokens = get_qwen2_5_vision_zip_num_tokens(num_tokens, vision_zip_config)
    if keep_tokens >= num_tokens:
        return image_features, positions

    if vision_zip_config.vision_zip_debug:
        logger.info(
            "VisionZip pruned %d/%d visual tokens for Qwen2.5-VL (kept %d).",
            num_tokens - keep_tokens,
            num_tokens,
            keep_tokens,
        )

    dominant_tokens, contextual_tokens = _get_qwen2_5_vision_zip_split(
        keep_tokens,
        vision_zip_config.vision_zip_dominant_ratio,
    )
    dominant_tokens = min(dominant_tokens, num_tokens)

    dominant_indices = importance_scores.topk(
        k=dominant_tokens,
        largest=True,
        sorted=False,
    ).indices
    dominant_indices = dominant_indices.sort().values

    dominant_mask = torch.zeros(
        num_tokens,
        dtype=torch.bool,
        device=image_features.device,
    )
    dominant_mask[dominant_indices] = True

    dominant_features = image_features[dominant_indices]
    dominant_positions = positions[dominant_indices]

    if contextual_tokens == 0 or dominant_tokens == num_tokens:
        return dominant_features, dominant_positions

    remaining_indices = torch.arange(num_tokens, device=image_features.device)[
        ~dominant_mask
    ]
    if remaining_indices.numel() == 0:
        return dominant_features, dominant_positions

    num_contextual = min(contextual_tokens, remaining_indices.numel())
    remaining_keys = key_states[remaining_indices]

    target_token_indices = torch.stack([
        remaining_indices[chunk[0]]
        for chunk in torch.tensor_split(
            torch.arange(
                remaining_indices.numel(),
                device=image_features.device,
            ),
            num_contextual,
        )
    ])
    target_keys = key_states[target_token_indices]

    similarity = torch.matmul(remaining_keys, target_keys.transpose(0, 1))
    assignments = similarity.argmax(dim=-1)

    merged_features = image_features.new_zeros((num_contextual, hidden_size))
    counts = image_features.new_zeros((num_contextual, 1))
    merged_features.index_add_(0, assignments, image_features[remaining_indices])
    counts.index_add_(0, assignments, counts.new_ones((assignments.shape[0], 1)))
    merged_features = merged_features / counts.clamp_min_(1.0)

    merged_positions = positions[target_token_indices]

    kept_anchor_indices = torch.cat([dominant_indices, target_token_indices], dim=0)
    kept_features = torch.cat([dominant_features, merged_features], dim=0)
    kept_positions = torch.cat([dominant_positions, merged_positions], dim=0)

    order = kept_anchor_indices.argsort()
    return kept_features[order], kept_positions[order]


class Qwen2_5_VisionZipAttention(Qwen2_5_VisionAttention):
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
        return_visionzip_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not return_visionzip_metadata:
            return (
                super().forward(
                    x,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb_cos=rotary_pos_emb_cos,
                    rotary_pos_emb_sin=rotary_pos_emb_sin,
                    max_seqlen=max_seqlen,
                    sequence_lengths=sequence_lengths,
                ),
                None,
                None,
            )

        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape
        assert batch_size == 1, (
            "Qwen2.5-VL VisionZip expects the vision encoder batch dimension "
            "to be folded into the sequence dimension."
        )

        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]
            qk_reshaped = einops.rearrange(
                qk,
                "b s two head head_dim -> (two b) s head head_dim",
                two=2,
            ).contiguous()
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        token_importance = q.new_zeros(seq_len)
        context = q.new_empty(
            (
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
        )

        segment_bounds = cu_seqlens.tolist()
        for start, end in zip(segment_bounds[:-1], segment_bounds[1:]):
            q_seg = q[0, start:end].transpose(0, 1)
            k_seg = k[0, start:end].transpose(0, 1)
            v_seg = v[0, start:end].transpose(0, 1)

            attn_scores = torch.matmul(q_seg, k_seg.transpose(-1, -2))
            attn_scores = attn_scores * (self.hidden_size_per_attention_head**-0.5)
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q_seg.dtype)

            token_importance[start:end] = attn_probs.mean(dim=(0, 1)).to(
                token_importance.dtype
            )
            context[start:end] = torch.matmul(attn_probs, v_seg).transpose(0, 1)

        context_layer = context.reshape(seq_len, batch_size, -1).contiguous()
        output, _ = self.proj(context_layer)
        flat_keys = k.squeeze(0).reshape(seq_len, -1).contiguous()
        return output, token_importance, flat_keys


class Qwen2_5_VisionZipBlock(Qwen2_5_VisionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn,
        norm_layer,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            act_fn=act_fn,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.attn = Qwen2_5_VisionZipAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        return_visionzip_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x_attn, token_importance, key_states = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=None,
            return_visionzip_metadata=return_visionzip_metadata,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x, token_importance, key_states


class Qwen2_5_VisionTransformerVisionZip(Qwen2_5_VisionTransformer):
    def __init__(
        self,
        vision_config,
        *,
        vision_zip_attention_layer: int,
        norm_eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.vision_zip_attention_layer = vision_zip_attention_layer

        norm_layer = partial(RMSNorm, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionZipBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
        return_visionzip_metadata: bool = False,
    ):
        seq_len, _ = x.size()
        rotary_pos_emb_cos = []
        rotary_pos_emb_sin = []
        window_index = []
        cu_window_seqlens = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                cos_thw,
                sin_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)
            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb_cos = torch.cat(rotary_pos_emb_cos)
        rotary_pos_emb_sin = torch.cat(rotary_pos_emb_sin)
        window_index = torch.cat(window_index)
        reverse_indices = self.invert_permutation(window_index)
        cu_window_seqlens = torch.unique_consecutive(torch.cat(cu_window_seqlens))
        cu_seqlens = F.pad(
            torch.cumsum(torch.cat(cu_seqlens), dim=0, dtype=torch.int32),
            (1, 0),
            "constant",
            0,
        )

        max_seqlen_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window = self.compute_attn_mask_seqlen(cu_window_seqlens)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device, non_blocking=True)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(device=self.device, non_blocking=True)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(device=self.device, non_blocking=True)
        window_index = window_index.to(device=hidden_states.device, non_blocking=True)
        reverse_indices = reverse_indices.to(
            device=hidden_states.device,
            non_blocking=True,
        )

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        captured_importance = None
        captured_keys = None
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window

            hidden_states, token_importance, key_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen_now,
                return_visionzip_metadata=(
                    return_visionzip_metadata
                    and layer_num == self.vision_zip_attention_layer
                ),
            )
            if token_importance is not None:
                captured_importance = token_importance
                captured_keys = key_states

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]

        if not return_visionzip_metadata:
            return hidden_states

        if captured_importance is None or captured_keys is None:
            raise RuntimeError(
                "VisionZip metadata was requested but no attention metadata "
                "was captured from the configured layer."
            )

        merged_importance = captured_importance.view(
            -1, self.spatial_merge_unit
        ).mean(dim=1)
        merged_keys = captured_keys.view(
            -1,
            self.spatial_merge_unit,
            captured_keys.shape[-1],
        ).mean(dim=1)

        merged_importance = merged_importance[reverse_indices]
        merged_keys = merged_keys[reverse_indices]
        return hidden_states, merged_importance, merged_keys


class Qwen2_5_VLVisionZipProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        num_tokens = super().get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
        )
        return get_qwen2_5_vision_zip_num_tokens(
            num_tokens,
            self.ctx.model_config.multimodal_config,
        )


class Qwen2_5_VLVisionZipMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        mm_config = self.info.ctx.model_config.multimodal_config

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }
        merge_length = image_processor.merge_size**2

        def get_replacement(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            if modality == "image":
                num_tokens = get_qwen2_5_vision_zip_num_tokens(
                    num_tokens,
                    mm_config,
                )
            elif (
                modality == "video"
                and mm_config.video_pruning_rate is not None
                and mm_config.video_pruning_rate > 0.0
            ):
                t, h, w = map(int, grid_thw)
                tokens_per_frame = (h // image_processor.merge_size) * (
                    w // image_processor.merge_size
                )
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame,
                    t,
                    mm_config.video_pruning_rate,
                )
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement, modality=modality),
            )
            for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLVisionZipMultiModalProcessor,
    info=Qwen2_5_VLVisionZipProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VLVisionZipForConditionalGeneration(
    Qwen2_5_VLForConditionalGeneration
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        multimodal_config = vllm_config.model_config.multimodal_config
        self.vision_zip_config = _get_qwen2_5_vision_zip_config(multimodal_config)
        self.is_vision_zip_recompute_enabled = self.vision_zip_config is not None

        if self.vision_zip_config is None:
            return

        if self.use_data_parallel:
            raise NotImplementedError(
                "Qwen2.5-VL VisionZip is not currently supported with "
                "`mm_encoder_tp_mode=\"data\"`."
            )

        attention_layer = self.vision_zip_config.vision_zip_attention_layer
        if attention_layer < 0:
            attention_layer += self.config.vision_config.depth
        if not 0 <= attention_layer < self.config.vision_config.depth:
            raise ValueError(
                "VisionZip attention layer must be within the Qwen2.5-VL vision "
                "encoder depth."
            )

        self.visual = Qwen2_5_VisionTransformerVisionZip(
            vision_config=self.config.vision_config,
            vision_zip_attention_layer=attention_layer,
            norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _append_original_mrope_positions_to_videos(
        self,
        video_embeds_split: tuple[torch.Tensor, ...],
        video_input: Qwen2_5_VLVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        second_per_grid_ts = video_input.get("second_per_grid_ts")
        if second_per_grid_ts is None:
            raise ValueError(
                "second_per_grid_ts is required for Qwen2.5-VL video inputs "
                "when VisionZip triggers mRoPE recomputation."
            )

        second_per_grid_ts = second_per_grid_ts.long()
        merge_size = self.visual.spatial_merge_size
        tokens_per_second = self.config.vision_config.tokens_per_second

        video_embeds_out = []
        for emb, size, video_second_per_grid_t in zip(
            video_embeds_split,
            grid_thw_list,
            second_per_grid_ts,
        ):
            positions = compute_mrope_for_media(
                size,
                merge_size,
                tokens_per_second=tokens_per_second,
                video_second_per_grid=video_second_per_grid_t.item(),
            ).to(emb.device)
            video_embeds_out.append(torch.cat([emb, positions], dim=1))
        return tuple(video_embeds_out)

    def _process_image_input_with_visionzip(
        self,
        image_input: Qwen2_5_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            raise ValueError(
                "Qwen2.5-VL VisionZip currently requires `pixel_values`; "
                "`image_embeds` bypass the vision encoder attention metadata "
                "needed for token selection."
            )

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        pixel_values = image_input["pixel_values"]
        with set_forward_context(None, self.vllm_config):
            image_embeds, image_scores, image_keys = self.visual(
                pixel_values,
                grid_thw=grid_thw_list,
                return_visionzip_metadata=True,
            )

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()

        image_embeds_split = image_embeds.split(sizes)
        image_scores_split = image_scores.split(sizes)
        image_keys_split = image_keys.split(sizes)

        image_embeds_out = []
        for emb, scores, keys, size in zip(
            image_embeds_split,
            image_scores_split,
            image_keys_split,
            grid_thw_list,
        ):
            positions = compute_mrope_for_media(size, merge_size).to(emb.device)
            emb, positions = apply_qwen2_5_vision_zip(
                emb,
                scores,
                keys,
                positions,
                self.vision_zip_config,
            )
            image_embeds_out.append(torch.cat([emb, positions], dim=1))
        return tuple(image_embeds_out)

    def embed_multimodal(self, **kwargs: object):
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                if self.vision_zip_config is not None:
                    image_embeddings = self._process_image_input_with_visionzip(
                        multimodal_input
                    )
                else:
                    image_embeddings = self._process_image_input(multimodal_input)
                    if self.is_multimodal_pruning_enabled:
                        image_embeddings = self._postprocess_image_embeds_evs(
                            image_embeddings,
                            multimodal_input,
                        )
                multimodal_embeddings += tuple(image_embeddings)

            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    video_embeddings = self._postprocess_video_embeds_evs(
                        video_embeddings,
                        multimodal_input,
                    )
                elif self.vision_zip_config is not None:
                    video_embeddings = self._append_original_mrope_positions_to_videos(
                        video_embeddings,
                        multimodal_input,
                    )
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings
