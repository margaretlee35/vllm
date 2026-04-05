import abc
from typing import Any

import torch


class BasePruningStrategy(abc.ABC):
    """
    Base class for multimodal token pruning strategies.
    Any custom pruning strategy should inherit from this class.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abc.abstractmethod
    def get_retained_token_count(self, tokens_per_frame: int, num_frames: int) -> int:
        """
        Compute the deterministic number of retained tokens.
        This must be computable purely from metadata before the forward pass.
        
        Args:
            tokens_per_frame: Number of tokens per frame (or spatial size).
            num_frames: Number of temporal frames.
            
        Returns:
            The number of tokens that will be retained after pruning.
        """
        pass

    @abc.abstractmethod
    def get_retention_mask(
        self,
        video_embeds: torch.Tensor,
        video_size_thw: torch.LongTensor | tuple[int, int, int],
        spatial_merge_size: int,
    ) -> torch.Tensor:
        """
        Computes the retention mask for the input embeddings.
        
        Args:
            video_embeds: Input embeddings of shape (T * H * W, D) or similar depending on the caller.
            video_size_thw: The temporal, height, and width sizes.
            spatial_merge_size: The spatial downsample ratio.
            
        Returns:
            A boolean tensor of the same length as the number of visual tokens
            indicating which ones to retain (True) and which to prune (False).
        """
        pass


class EVSTemporalPruningStrategy(BasePruningStrategy):
    """
    Temporal pruning strategy based on Efficient Video Sampling (EVS).
    It retains all tokens from the first frame and then drops the most similar tokens
    along the temporal axis based on cosine similarity.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Expected config contains a 'video_pruning_rate' or 'spatial_pruning_rate'
        self.q = self.config.get("video_pruning_rate", 0.0)

    def get_retained_token_count(self, tokens_per_frame: int, num_frames: int) -> int:
        """
        Compute the number of retained tokens for a given video.
        Ensures we retain all tokens from the first frame.
        """
        total_tokens = tokens_per_frame * num_frames
        evs_num_tokens = int(total_tokens * (1 - self.q))
        min_num_tokens = tokens_per_frame
        return max(min_num_tokens, evs_num_tokens)

    def get_retention_mask(
        self,
        video_embeds: torch.Tensor,
        video_size_thw: torch.LongTensor | tuple[int, int, int],
        spatial_merge_size: int,
    ) -> torch.Tensor:
        T, H, W = map(int, video_size_thw)

        # Use reshape instead of einops to avoid graph breaks
        reshaped_embeds = video_embeds.reshape(
            T,
            H // spatial_merge_size,
            W // spatial_merge_size,
            video_embeds.size(-1),
        )
        tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)
        
        # Core EVS
        similarity = torch.nn.functional.cosine_similarity(
            reshaped_embeds[1:, ...], reshaped_embeds[:-1, ...], dim=-1
        )
        dissimilarity = 1 - similarity

        # Always ensure we include all tokens from the first frame
        dissimilarity = torch.cat(
            [255 * torch.ones_like(reshaped_embeds[:1, :, :, 0]), dissimilarity], dim=0
        )

        dissimilarity_flat = dissimilarity.view(-1)
        order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
        retain_num_tokens = self.get_retained_token_count(
            tokens_per_frame=tokens_per_frame, num_frames=T
        )
        topk_indices = order[:retain_num_tokens]

        retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
        retention_mask[topk_indices] = True
        retention_mask = retention_mask.reshape(dissimilarity.size())

        mask = retention_mask.view(-1)  # "T H W -> (T H W)"
        return mask

class SpatialPruningStrategy(BasePruningStrategy):
    """
    Spatial pruning strategy.
    It drops a fixed percentage of spatial tokens per frame.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.spatial_pruning_rate = self.config.get("spatial_pruning_rate", 0.0)

    def get_retained_token_count(self, tokens_per_frame: int, num_frames: int) -> int:
        retained_per_frame = max(1, int(tokens_per_frame * (1 - self.spatial_pruning_rate)))
        return retained_per_frame * num_frames

    def get_retention_mask(
        self,
        video_embeds: torch.Tensor,
        video_size_thw: torch.LongTensor | tuple[int, int, int],
        spatial_merge_size: int,
    ) -> torch.Tensor:
        T, H, W = map(int, video_size_thw)
        tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)
        retained_per_frame = max(1, int(tokens_per_frame * (1 - self.spatial_pruning_rate)))

        # For deterministic spatial pruning, we retain the first N tokens
        # Or a random selection. Here we retain the first N tokens for simplicity.
        mask = torch.zeros(T, tokens_per_frame, dtype=torch.bool, device=video_embeds.device)
        mask[:, :retained_per_frame] = True
        return mask.view(-1)

def get_pruning_strategy(mm_pruning_config: dict[str, Any] | None, video_pruning_rate: float | None) -> BasePruningStrategy | None:
    """
    Factory function to instantiate the correct pruning strategy.
    """
    if mm_pruning_config is not None:
        strategy_name = mm_pruning_config.get("strategy", "evs_temporal")
        if strategy_name == "evs_temporal":
            return EVSTemporalPruningStrategy(mm_pruning_config)
        elif strategy_name == "spatial":
            return SpatialPruningStrategy(mm_pruning_config)
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy_name}")
            
    if video_pruning_rate is not None and video_pruning_rate > 0:
        # Fallback for backward compatibility
        return EVSTemporalPruningStrategy({"video_pruning_rate": video_pruning_rate})
        
    return None
