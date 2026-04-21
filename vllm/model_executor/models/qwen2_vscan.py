#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VScan hook models for Qwen2.

This file provides:

  Qwen2AttentionVScan
    At the Stage-2 prune layer (k=14 for Qwen2.5-VL), computes
    attention from the *last instruction token* to all remaining visual
    tokens using a manual scaled-dot-product softmax (bypassing
    FlashAttention, which does not expose attention weights).
    Stores the resulting importance scores in ForwardContext.

  Qwen2ModelVScan
    Iterates decoder layers.  Before layer 0 it applies the Stage-1
    token merge (merge unselected visual tokens into their nearest
    selected neighbour, then zero-out unselected positions and
    invalidate their KV-cache slots).
    After the Stage-2 prune layer (k) it reads the scores written by
    the attention layer and drops the lowest-scoring visual tokens.

  Qwen2ForCausalLMVScan
    Top-level causal LM shell (identical to Qwen2ForCausalLMSparseVLM
    in structure).

Stage-1 token merge detail
--------------------------
The Stage-1 merge is driven by the ``vscan`` dict stored in
``ForwardContext.additional_kwargs`` by the outer
``Qwen2_5_VLForConditionalGenerationVScan`` model:

  vscan["stage1_masks"]   — list[Tensor(N,)bool] per image, filled by
                            VScanStrategy.prune (called from
                            _maybe_prune_mm_embeds).
  vscan["stage1_merged"]  — list[Tensor(N,D)] with merged embeddings.

At the start of layer 0 the model applies these masks to hidden_states
(which already contain the merged embeddings from the embed step) and
zeros unselected positions.

Stage-2 detail
--------------
At layer ``prune_layer`` the attention module captures Q for the last
instruction token and K for all remaining visual tokens, computes a
one-row scaled-dot-product attention, and writes the resulting (Lv,)
score tensor to ``vscan["tv_logits_stage2"]``.  After that layer,
``Qwen2ModelVScan`` applies top-R2 retention: keeps the highest-scored
visual tokens, zeros the rest, and invalidates their KV-cache slots.

FlashAttention compatibility
-----------------------------
Stage-2 attention is computed entirely in Python (``torch.einsum`` + F.softmax)
on the extracted Q / K slices — we never ask FlashAttention to expose weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, EncoderOnlyAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved, set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.multimodal.visual_token_pruning.vscan import (
    VScanConfig,
    vscan_stage2_scores,
)


# ---------------------------------------------------------------------------
# MLP (identical to SparseVLM version — no changes needed)
# ---------------------------------------------------------------------------

class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Attention with VScan Stage-2 capture
# ---------------------------------------------------------------------------

class Qwen2AttentionVScan(nn.Module):
    """Qwen2 self-attention with optional VScan Stage-2 Q/K capture.

    At the designated prune layer (``vscan["config"].prune_layer``) this module
    extracts the query of the *last instruction token* and the keys of all
    remaining visual tokens, computes a manual scaled-dot-product attention,
    and writes the (Lv,) importance scores to ``vscan["tv_logits_stage2"]``
    in ForwardContext.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.qk_norm = qk_norm

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = EncoderOnlyAttention if attn_type == AttentionType.ENCODER_ONLY else Attention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": self.layer_idx,
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm:
            T = q.shape[0]
            q = q.view(T, self.num_heads, self.head_dim)
            k = k.view(T, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(T, self.q_size)
            k = k.view(T, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)

        # ---- VScan Stage-2 capture ----
        if is_forward_context_available():
            ctx = get_forward_context()
            vs = ctx.additional_kwargs.get("vscan")
            if isinstance(vs, dict) and vs.get("enabled", False):
                vscan_cfg: VScanConfig = vs.get("config", VScanConfig())
                if self.layer_idx == vscan_cfg.prune_layer:
                    text_mask = vs.get("text_mask")
                    vision_mask = vs.get("vision_mask")
                    T_total = q.shape[0]
                    if (
                        isinstance(text_mask, torch.Tensor)
                        and isinstance(vision_mask, torch.Tensor)
                        and text_mask.numel() == T_total
                        and vision_mask.numel() == T_total
                    ):
                        # Expand KV heads to match Q heads for the manual computation.
                        qh = q.view(T_total, self.num_heads, self.head_dim)
                        kh = k.view(T_total, self.num_kv_heads, self.head_dim)
                        if self.num_kv_heads != self.num_heads:
                            rep = self.num_heads // self.num_kv_heads
                            kh = kh.repeat_interleave(rep, dim=1)

                        # Find the last instruction (text) token.
                        text_positions = torch.nonzero(
                            text_mask, as_tuple=False
                        ).flatten()

                        if text_positions.numel() > 0 and vision_mask.any():
                            last_text_idx = text_positions[-1].item()
                            q_last = qh[last_text_idx]   # (H, D)
                            k_vis = kh[vision_mask]       # (Lv, H, D)

                            if k_vis.numel() > 0:
                                scores = vscan_stage2_scores(
                                    q_last, k_vis, self.scaling
                                )  # (Lv,)
                                vs["tv_logits_stage2"] = scores

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Qwen2DecoderLayerVScan(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        qk_norm = getattr(config, "qk_norm", False)

        self.self_attn = Qwen2AttentionVScan(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _qwen2_model_invariants(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    torch._check(input_ids.size()[0] == positions.size()[-1])
    if intermediate_tensors is not None:
        torch._check(
            input_ids.size()[0] == intermediate_tensors["hidden_states"].size()[0]
        )
    if inputs_embeds is not None:
        torch._check(input_ids.size()[0] == inputs_embeds.size()[0])
    if inputs_embeds is not None and intermediate_tensors is not None:
        torch._check(
            inputs_embeds.size()[1]
            == intermediate_tensors["hidden_states"].size()[1]
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    shape_invariants=_qwen2_model_invariants,
)
class Qwen2ModelVScan(nn.Module):
    """Qwen2 language model with VScan Stage-1 merge + Stage-2 pruning."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayerVScan(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @staticmethod
    def _invalidate_kv_slots(drop_idx: torch.Tensor) -> None:
        """Set slot_mapping to -1 for dropped positions across all layers.

        Handles both dict and list-of-dicts forms of ``slot_mapping``
        (the type depends on the attention backend).
        """
        if not is_forward_context_available():
            return
        ctx = get_forward_context()
        sm = ctx.slot_mapping
        if isinstance(sm, dict):
            dicts = [sm]
        elif isinstance(sm, list):
            dicts = sm
        else:
            return
        for d in dicts:
            for slot_map in d.values():
                if isinstance(slot_map, torch.Tensor):
                    slot_map[drop_idx] = -1

    def _apply_stage1_merge(
        self,
        hidden_states: torch.Tensor,
        vs: dict,
    ) -> None:
        """Apply Stage-1 token merge in-place on hidden_states.

        The VScanStrategy.prune() call (triggered at embed time before the
        LLM forward) already computed ``stage1_masks`` (which visual token
        positions to keep) and ``stage1_merged`` (the merged embeddings).

        Here we:
        1. Zero-out unselected visual token positions.
        2. Invalidate KV-cache slots for unselected positions.
        3. Update vision_mask to reflect only the kept positions.
        """
        vision_mask: torch.Tensor | None = vs.get("vision_mask")
        stage1_masks: list = vs.get("stage1_masks", [])

        if not stage1_masks or vision_mask is None:
            return
        if vision_mask.numel() != hidden_states.shape[0]:
            return

        # Concatenate the per-image selected masks into a single (Lv,) mask
        # that covers all visual token positions in the sequence.
        if not all(isinstance(m, torch.Tensor) for m in stage1_masks):
            return
        combined_keep = torch.cat(stage1_masks)  # (Lv,)

        vis_idx = torch.nonzero(vision_mask, as_tuple=False).flatten()
        if vis_idx.numel() != combined_keep.numel():
            return

        drop_idx = vis_idx[~combined_keep]
        if drop_idx.numel() == 0:
            return

        # Zero-out dropped positions (residual is None before layer 0).
        hidden_states[drop_idx] = 0.0

        # Invalidate KV-cache slots.
        self._invalidate_kv_slots(drop_idx)

        # Update vision_mask so Stage-2 only sees remaining visual tokens.
        vision_mask[drop_idx] = False

    def _apply_stage2_prune(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        vs: dict,
    ) -> None:
        """Apply Stage-2 pruning in-place after layer prune_layer.

        Reads ``vs["tv_logits_stage2"]`` (Lv,) scores written by
        Qwen2AttentionVScan, keeps top ``stage2_retain`` tokens, zeros the
        rest, and invalidates their KV-cache slots.

        Both ``hidden_states`` and ``residual`` must be zeroed for pruned
        positions; otherwise the residual stream leaks unpruned signal
        through the fused add-norm in subsequent layers.
        """
        scores: torch.Tensor | None = vs.get("tv_logits_stage2")
        vision_mask: torch.Tensor | None = vs.get("vision_mask")
        stage2_retain: int = int(vs.get("stage2_retain", 0) or 0)

        if scores is None or vision_mask is None or stage2_retain <= 0:
            return
        if vision_mask.numel() != hidden_states.shape[0]:
            return

        vis_idx = torch.nonzero(vision_mask, as_tuple=False).flatten()  # (Lv,)
        Lv = vis_idx.numel()

        if Lv == 0 or scores.numel() != Lv:
            return

        n_keep = max(1, min(stage2_retain, Lv))
        _, topk_local = torch.topk(scores, n_keep)  # indices into vis_idx

        keep_mask = torch.zeros(Lv, dtype=torch.bool, device=hidden_states.device)
        keep_mask[topk_local] = True
        drop_local = torch.nonzero(~keep_mask, as_tuple=False).flatten()

        if drop_local.numel() == 0:
            return

        drop_idx = vis_idx[drop_local]

        # Zero both hidden_states AND residual to prevent leakage through
        # the fused add-norm (hidden_states + residual) in the next layer.
        hidden_states[drop_idx] = 0.0
        if residual is not None:
            residual[drop_idx] = 0.0

        # Invalidate KV-cache slots.
        self._invalidate_kv_slots(drop_idx)

        vision_mask[drop_idx] = False
        # Clear stage2 scores so they are not reused.
        vs["tv_logits_stage2"] = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)

            # Absolute layer index (accounts for pipeline parallelism).
            abs_layer_idx = self.start_layer + idx

            # ---- VScan Stage-1 merge (before layer 0) ----
            if abs_layer_idx == 0 and residual is None and is_forward_context_available():
                ctx = get_forward_context()
                vs = ctx.additional_kwargs.get("vscan")
                if isinstance(vs, dict) and vs.get("enabled", False):
                    self._apply_stage1_merge(hidden_states, vs)

            hidden_states, residual = layer(positions, hidden_states, residual)

            # ---- VScan Stage-2 prune (after prune_layer) ----
            if is_forward_context_available():
                ctx = get_forward_context()
                vs = ctx.additional_kwargs.get("vscan")
                if isinstance(vs, dict) and vs.get("enabled", False):
                    vscan_cfg: VScanConfig = vs.get("config", VScanConfig())
                    if abs_layer_idx == vscan_cfg.prune_layer:
                        self._apply_stage2_prune(hidden_states, residual, vs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name2 = name.replace(weight_name, param_name)
                if name2.endswith(".bias") and name2 not in params_dict:
                    continue
                if is_pp_missing_parameter(name2, self):
                    continue
                if name2.endswith("scale"):
                    name2 = maybe_remap_kv_scale_name(name2, params_dict)
                    if name2 is None:
                        continue
                param = params_dict[name2]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name2)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name2 = maybe_remap_kv_scale_name(name, params_dict)
                if (
                    name2 is None
                    or is_pp_missing_parameter(name2, self)
                    or name2 not in params_dict
                ):
                    continue
                param = params_dict[name2]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name2)
        return loaded_params


# ---------------------------------------------------------------------------
# Top-level causal LM
# ---------------------------------------------------------------------------

class Qwen2ForCausalLMVScan(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2ModelVScan(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
