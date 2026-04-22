#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VScan hook model for Qwen2.5-VL.

This wrapper keeps ``qwen2_5_vl.py`` intact and provides:
  1. PyTorch forward hooks on two ViT blocks (shallow layer l and final layer)
     to capture raw QKV tensors during the visual-encoder forward pass.
  2. Manual attention-score computation from the captured QKV.
  3. Window-permutation un-permutation so scores align with the spatial order
     of the merged tokens returned by the encoder.
  4. Token merging (Stage 1) applied before the LLM prefix phase.
  5. ForwardContext state injected for Stage 2 (LLM middle-layer pruning).

Qwen2.5-VL window-permutation recap
--------------------------------------
The vision transformer permutes tokens by ``window_index`` (in merged-token
space) before its blocks and restores spatial order via ``reverse_indices``
after the merger.  We reconstruct ``reverse_indices`` from ``grid_thw`` using
the public ``get_rope_by_thw`` helper (``@lru_cache``-d, so no extra compute).

Hook mechanics
--------------
Each hook targets ``self.visual.blocks[l].attn.qkv`` — a ``QKVParallelLinear``
whose output is ``(seq_len, batch=1, 3*local_heads*head_dim)``.  Q,K,V are
packed in that order (equal-heads MHA).  Captured Q and K are at raw-patch
level in window-permuted order; after grouping 4 patches per merged token and
applying ``reverse_indices`` we obtain per-merged-token scores in spatial order.

embed_multimodal flow
---------------------
1. Parse image/video inputs.
2. Register QKV hooks on target ViT layers.
3. For each modality, call ``_process_{modality}_input`` (hooks fire here).
4. Remove hooks; compute Stage-1 importance scores from captures.
5. Reconstruct ``reverse_indices``; aggregate patch → merged-token scores.
6. Run ``VScanStrategy.prune`` on embeddings (global+local scan + token merge).
7. Inject VScan state into ``ForwardContext.additional_kwargs["vscan"]`` for
   Stage 2 in the LLM decoder (``Qwen2ModelVScan``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available

from .qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformer
from .utils import init_vllm_registered_model, maybe_prefix

from vllm.multimodal.visual_token_pruning.vscan import (
    VScanConfig,
    compute_self_attn_importance,
)
from vllm.multimodal.visual_token_pruning.registry import get_visual_pruning_strategy


# ---------------------------------------------------------------------------
# Utility: clear stale static_forward_context entries from the default LM
# ---------------------------------------------------------------------------

def _clear_language_model_static_forward_context(
    vllm_config: VllmConfig,
    prefix: str,
) -> None:
    """Remove static_forward_context entries registered by the default LM.

    When ``super().__init__()`` builds the default ``Qwen2ForCausalLM``, each
    ``Attention`` layer registers itself in
    ``compilation_config.static_forward_context[layer_prefix]``.  Before we
    swap to a VScan-hooked LM (``Qwen2ForCausalLMVScan``), we must remove
    those entries so the new attention layers can register the same prefixes
    without hitting "Duplicate layer name" errors.
    """
    sfc = vllm_config.compilation_config.static_forward_context
    lm_prefix = maybe_prefix(prefix, "language_model")
    stale_keys = [k for k in sfc if k.startswith(lm_prefix)]
    for k in stale_keys:
        del sfc[k]


# ---------------------------------------------------------------------------
# Hook context manager
# ---------------------------------------------------------------------------

@contextmanager
def _capture_vit_qkv(
    visual: Qwen2_5_VisionTransformer,
    target_layers: list[int],
):
    """Install forward hooks on target ViT blocks' qkv linear.

    Yields a dict ``captures`` where ``captures[layer_idx]`` is filled
    with the raw output tensor ``(seq_len, 1, 3*H*D)`` on each forward call.
    Multiple forward calls accumulate (each overwrites the previous).
    """
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(module, inputs, output):
            captures[layer_idx] = output[0].detach()
        return _hook

    for idx in target_layers:
        if idx < len(visual.blocks):
            h = visual.blocks[idx].attn.qkv.register_forward_hook(
                _make_hook(idx)
            )
            handles.append(h)

    try:
        yield captures
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _qkv_raw_to_qk(
    qkv_raw: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (seq_len, 1, 3*H*D) into Q and K tensors."""
    x = qkv_raw.squeeze(1)  # (seq_len, 3*H*D)
    qkv = x.view(x.shape[0], 3, num_heads, head_dim)  # (S, 3, H, D)
    return qkv[:, 0], qkv[:, 1]  # (S, H, D), (S, H, D)


def _merged_token_scores(
    qkv_raw: torch.Tensor,
    attn_module,
    spatial_merge_unit: int,
    reverse_indices: torch.Tensor,
) -> torch.Tensor:
    """Per-merged-token importance scores from raw QKV capture.

    1. Split QKV → Q, K  (raw-patch level, window-permuted order).
    2. Compute column-mean self-attention importance at raw-patch level.
    3. Average 4 adjacent raw-patch scores → 1 merged-token score.
    4. Un-permute merged-token scores using ``reverse_indices``.

    Returns: (N_merged,) float32 in spatial raster order.
    """
    num_heads = attn_module.num_attention_heads_per_partition
    head_dim = attn_module.hidden_size_per_attention_head
    scale = head_dim ** -0.5

    q, k = _qkv_raw_to_qk(qkv_raw, num_heads, head_dim)
    patch_scores = compute_self_attn_importance(q, k, scale)  # (S,)

    S = patch_scores.shape[0]
    n_merged = S // spatial_merge_unit
    if n_merged > 0:
        merged_perm = patch_scores[: n_merged * spatial_merge_unit].view(
            n_merged, spatial_merge_unit
        ).mean(dim=1)
    else:
        merged_perm = patch_scores.new_zeros(0)

    if reverse_indices.shape[0] == n_merged:
        return merged_perm[reverse_indices].float()
    return merged_perm.float()


def _build_reverse_indices(
    visual: Qwen2_5_VisionTransformer,
    grid_thw_list: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct ``reverse_indices`` from ``grid_thw`` without re-running
    the encoder (uses the cached ``get_rope_by_thw``)."""
    window_index_parts: list[torch.Tensor] = []
    offset = 0
    ms = visual.spatial_merge_size

    for row in grid_thw_list:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        _, _, window_index_thw, _, _ = visual.get_rope_by_thw(t, h, w)
        window_index_parts.append(window_index_thw + offset)
        offset += t * (h // ms) * (w // ms)

    if not window_index_parts:
        return torch.zeros(0, dtype=torch.long, device=device)

    # invert_permutation uses pin_memory which requires CPU tensors.
    # Build on CPU, invert, then move to the target device.
    window_index = torch.cat(window_index_parts).cpu()
    reverse_indices = Qwen2_5_VisionTransformer.invert_permutation(window_index)
    return reverse_indices.to(device=device)


# ---------------------------------------------------------------------------
# VScan Qwen2.5-VL wrapper
# ---------------------------------------------------------------------------

class Qwen2_5_VLForConditionalGenerationVScan(Qwen2_5_VLForConditionalGeneration):
    """VScan wrapper for Qwen2.5-VL.

    Adds visual-encoder QKV hooks (Stage-1 attention capture) and ForwardContext
    injection for Stage-2 pruning in the VScan LLM decoder.
    """

    is_vscan_wrapper = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Swap to the VScan-hooked LM backbone.
        _clear_language_model_static_forward_context(vllm_config, prefix)
        self.language_model = init_vllm_registered_model(
            vllm_config=self.vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLMVScan"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ------------------------------------------------------------------
    # VScan config
    # ------------------------------------------------------------------

    def _vscan_cfg(self) -> VScanConfig:
        mm = self.multimodal_config
        rate = mm.vt_prune_rate or 0.0
        r1 = float(os.environ.get("VSCAN_R1", str(1.0 - rate) if rate > 0 else "0.167"))
        r2 = float(os.environ.get("VSCAN_R2", "0.333"))
        local_layer = int(os.environ.get("VSCAN_LOCAL_LAYER", "8"))
        prune_layer = int(os.environ.get("VSCAN_PRUNE_LAYER", "14"))
        return VScanConfig(r1=r1, r2=r2, local_layer=local_layer, prune_layer=prune_layer)

    def _is_vscan_active(self) -> bool:
        mm = self.multimodal_config
        method = mm.get_visual_token_pruning_method()
        return (
            method == "vscan"
            or (
                mm.vt_prune_rate is not None
                and mm.vt_prune_rate > 0
                and method in ("vscan", None)
            )
        )

    # ------------------------------------------------------------------
    # embed_multimodal: fully overridden for Stage-1
    # ------------------------------------------------------------------

    def embed_multimodal(self, **kwargs: object):
        if not self._is_vscan_active():
            return super().embed_multimodal(**kwargs)

        vscan_cfg = self._vscan_cfg()
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # Collect grid_thw for all items (needed to reconstruct reverse_indices).
        grid_thw_all: list[list[int]] = []
        for modality, mm_input in mm_input_by_modality.items():
            key = "image_grid_thw" if modality == "image" else "video_grid_thw"
            gt = mm_input.get(key)
            if isinstance(gt, torch.Tensor) and gt.numel() > 0:
                grid_thw_all.extend(gt.tolist())

        # Target ViT layer indices.
        n_blocks = len(self.visual.blocks)
        local_layer_idx = min(vscan_cfg.local_layer, n_blocks - 1)
        final_layer_idx = n_blocks - 1
        target_layers = list(dict.fromkeys([local_layer_idx, final_layer_idx]))

        # ---- Step 1: Run visual encoder with QKV hooks ----
        raw_embeddings_by_modality: dict[str, tuple[torch.Tensor, ...]] = {}

        with _capture_vit_qkv(self.visual, target_layers) as qkv_captures:
            for modality, mm_input in mm_input_by_modality.items():
                if modality == "image":
                    raw_embeddings_by_modality["image"] = self._process_image_input(
                        mm_input
                    )
                elif modality == "video":
                    raw_embeddings_by_modality["video"] = self._process_video_input(
                        mm_input
                    )
        # Hooks have fired; qkv_captures is now populated.

        # ---- Step 2: Compute Stage-1 scores ----
        device = next(self.visual.parameters()).device
        ms_unit = self.visual.spatial_merge_unit  # 4 for spatial_merge_size=2

        reverse_indices = _build_reverse_indices(self.visual, grid_thw_all, device)

        global_scores: torch.Tensor | None = None
        local_scores: torch.Tensor | None = None

        if final_layer_idx in qkv_captures and qkv_captures[final_layer_idx] is not None:
            try:
                global_scores = _merged_token_scores(
                    qkv_captures[final_layer_idx],
                    self.visual.blocks[final_layer_idx].attn,
                    ms_unit,
                    reverse_indices,
                )
            except Exception:
                global_scores = None

        if local_layer_idx in qkv_captures and qkv_captures[local_layer_idx] is not None:
            try:
                local_scores = _merged_token_scores(
                    qkv_captures[local_layer_idx],
                    self.visual.blocks[local_layer_idx].attn,
                    ms_unit,
                    reverse_indices,
                )
            except Exception:
                local_scores = None

        if global_scores is None and local_scores is None:
            # No hooks fired — fall back to parent (no Stage-1 pruning).
            return super().embed_multimodal(**kwargs)

        if global_scores is None:
            global_scores = local_scores
        if local_scores is None:
            local_scores = global_scores

        # ---- Step 3: Build per-item grid_hw list for local scan ----
        grid_hw_list: list[tuple[int, int]] = []
        for row in grid_thw_all:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            ms = self.visual.spatial_merge_size
            grid_hw_list.append((t * (h // ms), w // ms))

        # ---- Step 4: Build vscan state dict ----
        vscan_state: dict = {
            "enabled": True,
            "config": vscan_cfg,
            "global_scores": global_scores,
            "local_scores": local_scores,
            "grid_hw_list": grid_hw_list,
            "stage1_masks": [],
            "stage1_merged": [],
            "stage2_keep_mask": None,
            "tv_logits_stage2": None,
        }

        # ---- Step 5: Run VScan Stage-1 pruning on embeddings ----
        strategy = get_visual_pruning_strategy("vscan")
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            mm_input = mm_input_by_modality[modality]
            raw_embs = raw_embeddings_by_modality.get(modality, ())
            if not raw_embs:
                continue

            if strategy is not None and strategy.is_applicable(self):
                pruned_embs = strategy.prune(
                    model=self,
                    modality=modality,
                    embeds_split=raw_embs,
                    aux={"vscan": vscan_state},
                    config=self.multimodal_config,
                )
            else:
                pruned_embs = raw_embs

            multimodal_embeddings += tuple(pruned_embs)

        # ---- Step 6: Inject into ForwardContext for Stage 2 ----
        if is_forward_context_available():
            ctx = get_forward_context()
            ctx.additional_kwargs.setdefault("vscan", {})
            ctx.additional_kwargs["vscan"].update(vscan_state)

        return list(multimodal_embeddings)

    # ------------------------------------------------------------------
    # forward: inject vision/text masks for Stage 2
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)

        if self._is_vscan_active() and input_ids is not None and is_forward_context_available():
            ctx = get_forward_context()
            vs = ctx.additional_kwargs.get("vscan")
            if isinstance(vs, dict) and vs.get("enabled", False):
                vision_mask = (input_ids == self.config.image_token_id) | (
                    input_ids == self.config.video_token_id
                )
                text_mask = ~vision_mask
                v_cnt = int(vision_mask.sum().item())
                vscan_cfg: VScanConfig = vs.get("config", VScanConfig())
                stage2_retain = max(
                    1, int(round(v_cnt * vscan_cfg.r1 * vscan_cfg.r2))
                )
                vs.update({
                    "text_mask": text_mask,
                    "vision_mask": vision_mask.clone(),
                    "stage2_retain": stage2_retain,
                })

        return super().forward(*args, **kwargs)
