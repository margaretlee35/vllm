import torch
import torch.nn.functional as F

CDPRUNER_TEXT_EMBEDDINGS_KEY = "__cd_instruction_embeddings"


class SimpleCDPruner:
    """Fast greedy MAP DPP token pruning matching CDPruner equations."""

    def __init__(self, keep_ratio: float = 0.5):
        """
        Args:
            keep_ratio: Fraction of tokens to retain in (0, 1].
        """
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")

        self.keep_ratio = keep_ratio
        self._eps = 1e-8

    def _compute_conditional_kernel(
        self,
        vision_embeddings: torch.Tensor,
        text_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_visual = F.normalize(
            vision_embeddings.to(torch.float32), dim=-1, eps=self._eps
        )
        # Eq. (3): pairwise cosine similarity kernel.
        kernel = torch.matmul(normalized_visual, normalized_visual.T).clamp(-1.0, 1.0)

        if text_embedding is not None:
            normalized_text = F.normalize(
                text_embedding.to(torch.float32), dim=-1, eps=self._eps
            )
            # Eq. (5) + Eq. (6): instruction relevance and min-max normalization.
            relevance = torch.matmul(normalized_visual, normalized_text)
            rel_min = relevance.min()
            rel_range = relevance.max() - rel_min
            if rel_range > self._eps:
                relevance = (relevance - rel_min) / rel_range
            else:
                relevance = torch.ones_like(relevance)

            # Eq. (7): conditional kernel L̃ = diag(r̃) L diag(r̃).
            kernel = relevance.unsqueeze(1) * kernel * relevance.unsqueeze(0)

        kernel = (kernel + kernel.T) * 0.5
        # Keep diagonal non-negative to avoid numerical artifacts in greedy updates.
        kernel.diagonal().clamp_(min=0.0)
        return kernel

    def _fast_greedy_dpp_select(self, kernel: torch.Tensor, k: int) -> torch.Tensor:
        n = kernel.size(0)
        if k >= n:
            return torch.arange(n, dtype=torch.long, device=kernel.device)

        cis = kernel.new_zeros((k, n))
        di2s = kernel.diagonal().clone()
        selected = torch.empty(k, dtype=torch.long, device=kernel.device)
        selected_mask = torch.zeros(n, dtype=torch.bool, device=kernel.device)

        for step in range(k):
            scores = di2s.masked_fill(selected_mask, float("-inf"))
            index = torch.argmax(scores)
            selected[step] = index
            selected_mask[index] = True

            if step + 1 == k:
                break

            projection = kernel.new_zeros(n)
            if step > 0:
                projection = torch.matmul(cis[:step, index], cis[:step, :])

            denom = torch.sqrt(torch.clamp(di2s[index], min=self._eps))
            e = (kernel[index, :] - projection) / denom
            cis[step, :] = e
            di2s = torch.clamp(di2s - e.square(), min=0.0)

        return selected

    def _prune_single(
        self,
        vision_embeddings: torch.Tensor,
        k: int,
        text_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        kernel = self._compute_conditional_kernel(vision_embeddings, text_embedding)
        selected = self._fast_greedy_dpp_select(kernel, k)
        selected = torch.sort(selected).values
        return vision_embeddings.index_select(0, selected)

    def __call__(
        self,
        vision_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_embeddings: (B, N, D) or (N, D) tensor of visual token embeddings.
            text_embeddings: Optional (B, D) or (D,) text embedding(s) for
                relevance-conditioned pruning.
        """
        if vision_embeddings.ndim not in (2, 3):
            raise ValueError(
                "vision_embeddings must be 2D or 3D, "
                f"got shape {tuple(vision_embeddings.shape)}."
            )

        squeeze_batch = vision_embeddings.ndim == 2
        if squeeze_batch:
            vision_embeddings = vision_embeddings.unsqueeze(0)

        batch_size, num_tokens, hidden_dim = vision_embeddings.shape
        if hidden_dim <= 0:
            raise ValueError("Embeddings must have nonzero hidden dimension.")

        k = max(1, min(num_tokens, int(num_tokens * self.keep_ratio)))
        if k == num_tokens:
            return vision_embeddings.squeeze(0) if squeeze_batch else vision_embeddings

        if text_embeddings is not None:
            if text_embeddings.ndim == 1:
                text_embeddings = text_embeddings.unsqueeze(0)
            if text_embeddings.ndim != 2:
                raise ValueError(
                    "text_embeddings must be 1D or 2D, "
                    f"got shape {tuple(text_embeddings.shape)}."
                )
            if text_embeddings.size(0) not in (1, batch_size):
                raise ValueError(
                    "text_embeddings batch dimension must be 1 or match "
                    f"vision_embeddings batch size ({batch_size}), got "
                    f"{text_embeddings.size(0)}."
                )
            if text_embeddings.size(0) == 1 and batch_size > 1:
                text_embeddings = text_embeddings.expand(batch_size, -1)

        pruned = [
            self._prune_single(
                vision_embeddings[b],
                k,
                None if text_embeddings is None else text_embeddings[b],
            )
            for b in range(batch_size)
        ]
        pruned_embeddings = torch.stack(pruned, dim=0)

        return pruned_embeddings.squeeze(0) if squeeze_batch else pruned_embeddings
