import torch


class SimpleCDPruner:
    def __init__(self, keep_ratio: float = 0.5, temperature: float = 0.1):
        """
        Args:
            keep_ratio: fraction of tokens to retain
            temperature: softmax scaling for conditional similarity
        """
        self.keep_ratio = keep_ratio
        self.temperature = temperature

    def __call__(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_embeddings: (B, N, D) or (N, D) tensor of visual token embeddings
        """
        # ensure 3D
        if vision_embeddings.ndim == 2:
            vision_embeddings = vision_embeddings.unsqueeze(0)

        B, N, D = vision_embeddings.shape
        k = max(1, int(N * self.keep_ratio))

        # Step 1: compute similarity (RBF / cosine for flexibility)
        # (B, N, N) similarity matrix
        e = vision_embeddings
        if D > 0:
            sim = torch.matmul(e, e.transpose(-1, -2)) / (
                (
                    torch.norm(e, dim=-1, keepdim=True)
                    * torch.norm(e, dim=-1, keepdim=True).transpose(-1, -2)
                )
                + 1e-8
            )
        else:
            raise ValueError("Embeddings must have nonzero dimension")

        # optional softmax scaling
        sim = torch.softmax(sim / self.temperature, dim=-1)

        # Step 2: DPP Selection (Greedy approximation)
        # We choose k tokens greedily maximizing det(K_subset)
        # Simple greedy DPP approximate algorithm
        selected = []
        remaining = list(range(N))
        # Initialize
        L = sim.clone()
        for _ in range(k):
            best = None
            best_gain = -1.0
            for idx in remaining:
                if len(selected) == 0:
                    gain = L[0, idx, idx].item()
                else:
                    # incremental determinant gain = L[idx, idx] - L[idx, selected] L[selected, selected]^-1 L[selected, idx]
                    s = torch.tensor(selected, dtype=torch.long)
                    # K_ss (k x k)
                    K_ss = L[0, s][:, s]
                    # vector K_is
                    K_is = L[0, s, idx]
                    inv = torch.inverse(
                        K_ss + 1e-6 * torch.eye(K_ss.size(0)).to(K_ss.device)
                    )
                    gain = (L[0, idx, idx] - K_is @ inv @ K_is.T).item()
                if gain > best_gain:
                    best_gain = gain
                    best = idx

            if best is None:
                break

            selected.append(best)
            remaining.remove(best)

        # Step 3: pack pruned tokens
        selected_idx = torch.tensor(selected, dtype=torch.long)
        pruned = torch.gather(
            vision_embeddings,
            1,
            selected_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D),
        )

        # remove batch dimension if original was 2D
        if pruned.size(0) == 1:
            pruned = pruned.squeeze(0)

        return pruned
