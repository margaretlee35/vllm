# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CDPruner helpers for Qwen2.5-VL visual-token pruning."""

import torch
import torch.nn.functional as F

_CDPRUNER_EPS = 1e-8


def _compute_cdpruner_kernel(
    image_features: torch.Tensor,
    text_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized_features = F.normalize(
        image_features.to(torch.float32),
        dim=-1,
        eps=_CDPRUNER_EPS,
    )
    kernel = torch.matmul(normalized_features, normalized_features.T).clamp(-1.0, 1.0)

    if text_embedding is not None:
        normalized_text = F.normalize(
            text_embedding.to(torch.float32),
            dim=-1,
            eps=_CDPRUNER_EPS,
        )
        relevance = torch.matmul(normalized_features, normalized_text)
        rel_min = relevance.min()
        rel_range = relevance.max() - rel_min
        if rel_range > _CDPRUNER_EPS:
            relevance = (relevance - rel_min) / rel_range
        else:
            relevance = torch.ones_like(relevance)
        kernel = relevance.unsqueeze(1) * kernel * relevance.unsqueeze(0)

    kernel = (kernel + kernel.T) * 0.5
    kernel.diagonal().clamp_(min=0.0)
    return kernel


def _fast_greedy_dpp_select(kernel: torch.Tensor, k: int) -> torch.Tensor:
    num_tokens = kernel.size(0)
    if k >= num_tokens:
        return torch.arange(num_tokens, dtype=torch.long, device=kernel.device)

    cis = kernel.new_zeros((k, num_tokens))
    di2s = kernel.diagonal().clone()
    selected = torch.empty(k, dtype=torch.long, device=kernel.device)
    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=kernel.device)

    for step in range(k):
        scores = di2s.masked_fill(selected_mask, float("-inf"))
        index = torch.argmax(scores)
        selected[step] = index
        selected_mask[index] = True

        if step + 1 == k:
            break

        projection = kernel.new_zeros((num_tokens,))
        if step > 0:
            projection = torch.matmul(cis[:step, index], cis[:step, :])

        denom = torch.sqrt(torch.clamp(di2s[index], min=_CDPRUNER_EPS))
        e = (kernel[index, :] - projection) / denom
        cis[step, :] = e
        di2s = torch.clamp(di2s - e.square(), min=0.0)

    return selected


def apply_qwen2_5_cdpruner(
    image_features: torch.Tensor,
    positions: torch.Tensor,
    keep_tokens: int,
    text_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = image_features.shape[0]
    if num_tokens == 0:
        return image_features, positions
    if keep_tokens <= 0:
        raise ValueError(f"keep_tokens must be > 0, got {keep_tokens}.")
    if keep_tokens >= num_tokens:
        return image_features, positions

    kernel = _compute_cdpruner_kernel(image_features, text_embedding=text_embedding)
    selected = _fast_greedy_dpp_select(kernel, keep_tokens).sort().values
    return (
        image_features.index_select(0, selected),
        positions.index_select(0, selected),
    )
