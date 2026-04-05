import torch
from vllm.multimodal.token_pruning import EVSTemporalPruningStrategy, SpatialPruningStrategy, get_pruning_strategy

def test_evs():
    config = {"strategy": "evs_temporal", "video_pruning_rate": 0.5}
    strategy = get_pruning_strategy(config, None)
    
    # 2 frames, 4 tokens per frame
    video_embeds = torch.randn(2 * 4, 16)
    video_size_thw = torch.LongTensor([2, 2, 2])
    spatial_merge_size = 1
    
    mask = strategy.get_retention_mask(video_embeds, video_size_thw, spatial_merge_size)
    assert mask.sum().item() == 4, f"Expected 4 retained tokens, got {mask.sum().item()}"
    print("EVS test passed!")

def test_spatial():
    config = {"strategy": "spatial", "spatial_pruning_rate": 0.5}
    strategy = get_pruning_strategy(config, None)
    
    # 2 frames, 4 tokens per frame
    video_embeds = torch.randn(2 * 4, 16)
    video_size_thw = torch.LongTensor([2, 2, 2])
    spatial_merge_size = 1
    
    mask = strategy.get_retention_mask(video_embeds, video_size_thw, spatial_merge_size)
    assert mask.sum().item() == 4, f"Expected 4 retained tokens, got {mask.sum().item()}"
    print("Spatial test passed!")

if __name__ == "__main__":
    test_evs()
    test_spatial()
