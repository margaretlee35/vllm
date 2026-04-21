"""
VScan: Two-stage visual token pruning for VLMs.

Based on: "VScan: Rethinking Visual Token Reduction for Efficient Large
Vision-Language Models" (arXiv:2505.XXXXX).

Stage 1 — Visual-encoder-level pruning (runs between encoder and LLM prefill):
  - Global scan: top-R1_global% tokens by mean attention-received score at the
    output ViT layer.
  - Local scan: top-R1_local% tokens per non-overlapping spatial window at a
    shallow ViT layer (l=8 for Qwen2.5-VL).
  - Token merge: unselected tokens are cosine-similarity-averaged into their
    nearest selected neighbour, enriching selected token content.
  - Net retention: R1 = 16.7% of original visual tokens by default.

Stage 2 — LLM-middle-layer pruning (runs during prefill at decoder layer k):
  - Compute attention from the last instruction token to all visual tokens
    using a manual softmax (not FlashAttention), averaged across heads.
  - Keep top R2 = 33.3% of remaining tokens; zero-out and KV-cache-invalidate
    the rest.

Default hyperparameters (Qwen2.5-VL):
  local_layer   = 8      (shallow ViT layer for local scan)
  prune_layer   = 14     (LLM decoder layer for Stage 2)
  r1            = 0.167  (Stage 1 retention: ~16.7% of visual tokens kept)
  r2            = 0.333  (Stage 2 retention: ~33.3% of Stage-1 survivors kept)
  global_frac   = 0.5    (50% of R1 budget from global scan, 50% local)

Note on throughput:
  Stage 1 provides real throughput / KV-cache savings because it reduces the
  token count BEFORE the LLM prefill begins (the tokens are merged into the
  selected positions and the rest are zeroed; slot_mapping invalidation lets
  later layers skip those positions).
  Stage 2 is a during-prefill operation; physical KV-cache blocks are NOT
  reclaimed by the default vLLM scheduler, so its savings are algebraic
  (cheaper attention post-layer-k) rather than memory-bandwidth savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.visual_token_pruning.registry import (
    VisualPruningStrategy,
    register_visual_pruning_strategy,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VScanConfig:
    """Hyperparameters for VScan (Qwen2.5-VL defaults)."""

    r1: float = 0.167
    """Stage-1 visual-token retention rate (fraction of original visual
    tokens kept after the encoder-side global+local scan + merge)."""

    r2: float = 0.333
    """Stage-2 retention rate (fraction of Stage-1 survivors kept after
    the LLM middle-layer instruction-guided pruning)."""

    local_layer: int = 8
    """Shallow ViT layer index used for the local scan (0-indexed).
    Qwen2.5-VL default: 8.  LLaVA-series default: 6."""

    prune_layer: int = 14
    """LLM decoder layer index at which Stage 2 runs (0-indexed).
    Qwen2.5-VL default: 14.  LLaVA-series default: 16."""

    global_frac: float = 0.5
    """Fraction of the Stage-1 budget allocated to the global scan.
    The remainder goes to the local scan.  Paper default: 0.5 (50/50)."""

    window_h: int = 4
    """Number of spatial-window rows for the local scan.  Each window
    spans (grid_h // window_h) merged-token rows."""

    window_w: int = 4
    """Number of spatial-window columns for the local scan."""


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------

def compute_self_attn_importance(
    q: Tensor,
    k: Tensor,
    scale: float,
) -> Tensor:
    """Average attention *received* per token (no-CLS proxy).

    For models without a [CLS] token (e.g. Qwen2.5-VL), each token's
    importance is the mean over heads of the column sum of the row-softmax
    attention matrix — i.e. how much attention it attracts from all other
    tokens.

    Args:
        q: (N, H, D) query tensors (pre- or post-rotary; values affect scores
           but not the relative ranking appreciably).
        k: (N, H, D) key tensors.
        scale: attention scale factor, typically 1/sqrt(head_dim).

    Returns:
        scores: (N,) float32 importance scores (higher = more important).
    """
    N, H, D = q.shape
    if N == 0:
        return q.new_zeros(0)

    # Work in float32 to avoid overflow for large N.
    qf = q.float().transpose(0, 1)  # (H, N, D)
    kf = k.float().transpose(0, 1)  # (H, N, D)

    # (H, N, N)  — cap N to avoid OOM; if N>2048 use a subsample.
    if N <= 2048:
        attn = torch.einsum("hid,hjd->hij", qf, kf) * scale  # (H, N, N)
        attn = F.softmax(attn, dim=-1)
        col_sums = attn.sum(dim=1)  # (H, N): total attention received
    else:
        # Sub-sample query side to keep memory bounded.
        step = max(1, N // 1024)
        qf_sub = qf[:, ::step, :]  # (H, N', D)
        attn = torch.einsum("hid,hjd->hij", qf_sub, kf) * scale  # (H, N', N)
        attn = F.softmax(attn, dim=-1)
        col_sums = attn.sum(dim=1)  # (H, N)

    scores = col_sums.mean(dim=0)  # (N,)
    return scores


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def global_scan(scores: Tensor, n_keep: int) -> Tensor:
    """Select the top-``n_keep`` token indices by importance score.

    Args:
        scores: (N,) importance scores.
        n_keep: number of tokens to select.

    Returns:
        indices: (≤n_keep,) long indices into ``scores``.
    """
    N = scores.shape[0]
    n_keep = max(1, min(n_keep, N))
    _, idx = torch.topk(scores, n_keep)
    return idx


def local_scan(
    scores: Tensor,
    grid_hw: tuple[int, int],
    n_keep_total: int,
    window_h: int = 4,
    window_w: int = 4,
) -> Tensor:
    """Select top tokens per non-overlapping spatial window.

    The image grid is divided into ``window_h × window_w`` windows with a
    budget of ``n_keep_total // n_windows`` tokens per window (uniform
    allocation).

    Args:
        scores: (N,) importance scores in raster (row-major) spatial order.
        grid_hw: ``(H, W)`` spatial dimensions of the merged-token grid.
        n_keep_total: total token budget across all windows.
        window_h: number of windows along the height dimension.
        window_w: number of windows along the width dimension.

    Returns:
        indices: (≤n_keep_total,) long indices into ``scores``.
    """
    H, W = grid_hw
    N = scores.shape[0]
    if N == 0 or H == 0 or W == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)

    # Determine window dimensions (in merged-token cells).
    wh = max(1, (H + window_h - 1) // window_h)
    ww = max(1, (W + window_w - 1) // window_w)

    n_windows_h = (H + wh - 1) // wh
    n_windows_w = (W + ww - 1) // ww
    n_windows = n_windows_h * n_windows_w
    per_window = max(1, n_keep_total // n_windows)

    scores_2d = scores.view(H, W)  # (H, W)

    selected: list[Tensor] = []
    for iy in range(0, H, wh):
        for ix in range(0, W, ww):
            ey = min(iy + wh, H)
            ex = min(ix + ww, W)

            win_scores = scores_2d[iy:ey, ix:ex].flatten()  # (wh*ww,)
            n_sel = min(per_window, win_scores.numel())
            if n_sel == 0:
                continue

            _, local_topk = torch.topk(win_scores, n_sel)

            # Convert flat window-local indices → global raster indices.
            local_row = local_topk // (ex - ix)
            local_col = local_topk % (ex - ix)
            global_idx = (iy + local_row) * W + (ix + local_col)
            selected.append(global_idx)

    if not selected:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    return torch.cat(selected)


def token_merge(
    tokens: Tensor,
    selected_mask: Tensor,
) -> Tensor:
    """Merge unselected tokens into their nearest selected neighbour.

    For each unselected token, the cosine-nearest selected token is found and
    the unselected token's embedding is averaged into it (weighted equally).
    Selected positions receive the merged result; unselected positions are
    zeroed.

    Args:
        tokens: (N, D) token embeddings (fp16/bf16/fp32).
        selected_mask: (N,) bool mask — True for selected tokens.

    Returns:
        merged: (N, D) tensor with merged embeddings at selected positions
        and zeros elsewhere.  Same dtype as ``tokens``.
    """
    N, D = tokens.shape
    orig_dtype = tokens.dtype

    selected_idx = torch.nonzero(selected_mask, as_tuple=False).flatten()  # (S,)
    unselected_idx = torch.nonzero(~selected_mask, as_tuple=False).flatten()  # (U,)

    if unselected_idx.numel() == 0:
        # Nothing to merge; just zero out (shouldn't happen here, but safe).
        out = tokens.clone()
        out[~selected_mask] = 0
        return out

    if selected_idx.numel() == 0:
        return tokens.new_zeros(N, D)

    sel_emb = tokens[selected_idx].float()   # (S, D)
    unsel_emb = tokens[unselected_idx].float()  # (U, D)

    # Cosine similarity: (U, S)
    sel_norm = F.normalize(sel_emb, dim=-1)
    unsel_norm = F.normalize(unsel_emb, dim=-1)
    sim = unsel_norm @ sel_norm.T  # (U, S)

    # For each unselected token, find the nearest selected token.
    nearest_sel_local = sim.argmax(dim=-1)  # (U,) indices into selected_idx
    target_global_idx = selected_idx[nearest_sel_local]  # (U,) global positions

    # Accumulate unselected embeddings into their targets (float for precision).
    merged = tokens.float().clone()
    merged[~selected_mask] = 0.0

    merged.scatter_add_(
        0,
        target_global_idx.unsqueeze(1).expand(-1, D),
        unsel_emb,
    )

    # Normalise by merge count so each selected token holds the *average*.
    counts = torch.ones(N, device=tokens.device, dtype=torch.float32)
    ones = torch.ones(unselected_idx.numel(), device=tokens.device, dtype=torch.float32)
    counts.scatter_add_(0, target_global_idx, ones)

    merged = merged / counts.unsqueeze(1).clamp(min=1.0)
    merged[~selected_mask] = 0.0

    return merged.to(orig_dtype)


def vscan_stage1(
    tokens: Tensor,
    global_scores: Tensor,
    local_scores: Tensor,
    grid_hw: tuple[int, int],
    n_keep: int,
    global_frac: float = 0.5,
    window_h: int = 4,
    window_w: int = 4,
) -> tuple[Tensor, Tensor]:
    """Run VScan Stage 1: global+local scan → token merge.

    Args:
        tokens: (N, D) visual token embeddings.
        global_scores: (N,) importance scores for the global scan (from
            final ViT layer attention; if unavailable pass ``local_scores``).
        local_scores: (N,) importance scores for the local scan (from
            shallow ViT layer l; if unavailable pass ``global_scores``).
        grid_hw: ``(H, W)`` merged-token spatial grid for the local scan.
        n_keep: total number of tokens to retain after merging.
        global_frac: fraction of ``n_keep`` for the global scan.
        window_h: window count along height for the local scan.
        window_w: window count along width for the local scan.

    Returns:
        merged_tokens: (N, D) — merged content at selected positions,
            zeros elsewhere.
        selected_mask: (N,) bool — True at the n_keep selected positions.
    """
    N = tokens.shape[0]
    n_keep = max(1, min(n_keep, N))

    n_global = max(1, int(round(n_keep * global_frac)))
    n_local = max(0, n_keep - n_global)

    # --- Global scan (output-layer scores) ---
    global_idx = global_scan(global_scores, n_global)  # (n_global,)

    # --- Local scan (shallow-layer scores, excluding global selections) ---
    local_scores_filtered = local_scores.clone()
    global_mask_tmp = torch.zeros(N, dtype=torch.bool, device=tokens.device)
    global_mask_tmp[global_idx] = True
    local_scores_filtered[global_mask_tmp] = float("-inf")  # exclude already selected

    if n_local > 0:
        if grid_hw[0] * grid_hw[1] == N:
            local_idx = local_scan(
                local_scores_filtered, grid_hw, n_local, window_h, window_w
            )
        else:
            # Grid shape mismatch (e.g. multi-image batch); fall back to global topk.
            local_idx = global_scan(local_scores_filtered, n_local)
    else:
        local_idx = torch.zeros(0, dtype=torch.long, device=tokens.device)

    # --- Union ---
    all_idx = torch.cat([global_idx, local_idx]).unique()

    selected_mask = torch.zeros(N, dtype=torch.bool, device=tokens.device)
    selected_mask[all_idx] = True

    # --- Token merge ---
    merged = token_merge(tokens, selected_mask)

    return merged, selected_mask


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

def vscan_stage2_scores(
    q_last_instr: Tensor,
    k_visual: Tensor,
    scale: float,
) -> Tensor:
    """Compute Stage-2 importance scores using the last-instruction-token query.

    Manual softmax attention (compatible with FlashAttention — we bypass the
    fused kernel entirely and recompute for just the single query row).

    Args:
        q_last_instr: (H, D) query for the last instruction token.
        k_visual: (Lv, H, D) keys of all current visual tokens.
        scale: attention scaling factor.

    Returns:
        scores: (Lv,) float32 importance scores for each visual token.
    """
    H, D = q_last_instr.shape
    Lv = k_visual.shape[0]

    if Lv == 0:
        return torch.zeros(0, device=q_last_instr.device)

    q = q_last_instr.float()  # (H, D)
    k = k_visual.float().permute(1, 0, 2)  # (H, Lv, D)

    # Scores per head: (H, Lv)
    logits = torch.einsum("hd,hvd->hv", q, k) * scale  # (H, Lv)
    attn = F.softmax(logits, dim=-1)  # (H, Lv)
    scores = attn.mean(dim=0)  # (Lv,)
    return scores


# ---------------------------------------------------------------------------
# VScanStrategy — plugs into the VisualPruningRegistry
# ---------------------------------------------------------------------------

class VScanStrategy(VisualPruningStrategy):
    """VScan embed-time strategy (Stage 1 only).

    This strategy is invoked by ``_maybe_prune_mm_embeds`` at embed time.
    It requires Stage-1 attention scores pre-computed by the VScan wrapper
    model and placed in ``aux`` under the key ``"vscan"``.

    Stage 2 is handled in the decoder forward (``Qwen2ForCausalLMVScan``).
    """

    name = "vscan"

    def is_applicable(self, model: nn.Module) -> bool:
        # Only activate when the outer multimodal model marked itself as a
        # VScan wrapper (to avoid accidental activation).
        return getattr(model, "is_vscan_wrapper", False)

    def prune(
        self,
        *,
        model: nn.Module,
        modality: str,
        embeds_split: tuple[Tensor, ...],
        aux: dict[str, Any],
        config: MultiModalConfig,
    ) -> tuple[Tensor, ...]:
        vscan_state = aux.get("vscan")
        if vscan_state is None:
            # No Stage-1 info available; return embeddings unchanged.
            return embeds_split

        vscan_cfg: VScanConfig = vscan_state.get("config", VScanConfig())
        global_scores_all: Tensor | None = vscan_state.get("global_scores")
        local_scores_all: Tensor | None = vscan_state.get("local_scores")
        grid_hw_list: list[tuple[int, int]] = vscan_state.get("grid_hw_list", [])

        pruned: list[Tensor] = []
        # Track score_offset in vscan_state so it persists across
        # multiple prune() calls (one per modality: images then videos).
        score_offset = vscan_state.get("_score_offset", 0)
        item_offset = vscan_state.get("_item_offset", 0)
        for i, emb in enumerate(embeds_split):
            N = emb.shape[0]
            if N == 0:
                pruned.append(emb)
                continue

            n_keep = max(1, int(round(N * vscan_cfg.r1)))

            # Slice scores for this embed item.
            if global_scores_all is not None and global_scores_all.shape[0] >= score_offset + N:
                g_scores = global_scores_all[score_offset: score_offset + N]
            else:
                # Fallback: use token embedding L2 norm as proxy.
                g_scores = emb.float().norm(dim=-1)

            if local_scores_all is not None and local_scores_all.shape[0] >= score_offset + N:
                l_scores = local_scores_all[score_offset: score_offset + N]
            else:
                l_scores = g_scores

            # Use absolute item index (accounting for items from prior
            # modality calls) to look up the correct grid_hw entry.
            abs_idx = item_offset + i
            grid_hw = grid_hw_list[abs_idx] if abs_idx < len(grid_hw_list) else (1, N)

            merged, selected_mask = vscan_stage1(
                tokens=emb,
                global_scores=g_scores,
                local_scores=l_scores,
                grid_hw=grid_hw,
                n_keep=n_keep,
                global_frac=vscan_cfg.global_frac,
                window_h=vscan_cfg.window_h,
                window_w=vscan_cfg.window_w,
            )

            # Store the selected mask back so the decoder wrapper can pick it up.
            vscan_state.setdefault("stage1_masks", []).append(selected_mask)
            vscan_state.setdefault("stage1_merged", []).append(merged)

            # Return the merged embeddings; only selected positions carry real data.
            # The decoder will zero-out unselected positions in hidden_states.
            pruned.append(merged)
            score_offset += N

        # Persist offsets for the next modality call.
        vscan_state["_score_offset"] = score_offset
        vscan_state["_item_offset"] = item_offset + len(embeds_split)

        return tuple(pruned)


register_visual_pruning_strategy(VScanStrategy())
