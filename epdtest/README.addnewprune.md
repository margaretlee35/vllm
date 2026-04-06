# Visual Token Pruning Extension Guide

This guide explains how to add a new pruning method (example: `sparsevlm`) to the current prune framework.

Current design:
- Registered model entry: `Qwen2_5_VLPruneForConditionalGeneration`
- Method-specific subclasses:
  - `Qwen2_5_VLVisionZipForConditionalGeneration`
  - `Qwen2_5_VLCDPruneForConditionalGeneration`
- Method selection: `visual_token_pruning_method` (`VISUAL_TOKEN_PRUNING_METHOD` env fallback)

## 1) Extend Config Types (`vllm/config/multimodal.py`)

1. Extend the literal:
```py
VisualTokenPruningMethod = Literal["vision_zip", "cdpruner", "sparsevlm"]
```

2. Add `sparsevlm`-specific config fields in `MultiModalConfig` (example names):
```py
sparsevlm_keep_strategy: str = "topk"
sparsevlm_debug: bool = False
```

3. Update `_validate_visual_token_pruning_method` aliases:
- accept `sparsevlm` (and optional alias like `sparse_vlm`)

4. Update multimodal validation (`_validate_multimodal_config`) for `sparsevlm` if needed.

5. If useful, add helper:
```py
def is_sparsevlm_enabled(self) -> bool: ...
```

## 2) Pass New Fields Through ModelConfig (`vllm/config/model.py`)

1. Add new `InitVar[...]` fields for any `sparsevlm`-specific parameters.
2. Include them in `__post_init__` args.
3. Forward them in `mm_config_kwargs` when building `MultiModalConfig`.

## 3) CLI Wiring (`vllm/engine/arg_utils.py`)

1. If `sparsevlm` needs extra knobs, add new CLI flags.
2. Ensure values flow into `ModelConfig(...)`.

Note: `--visual-token-pruning-method` choices are derived from `VisualTokenPruningMethod`, so extending the literal updates allowed values.

## 4) Add Method Path in Prune Model (`vllm/model_executor/models/qwen2_5_vl_prune.py`)

### 4.1 Extend method enum and config protocols

1. Extend local `VisualTokenPruningMethod` literal.
2. Add method protocol:
```py
class _SparseVLMConfig(_VisualTokenPruneMultiModalConfig, Protocol):
    sparsevlm_keep_strategy: str
    sparsevlm_debug: bool
```

3. Extend `_get_qwen2_5_prune_config(...)` union and add:
```py
def _get_qwen2_5_sparsevlm_config(...) -> _SparseVLMConfig | None:
```

### 4.2 Extend router class

In `Qwen2_5_VLPruneForConditionalGeneration.__init__`:
```py
elif self.visual_token_pruning_method == "sparsevlm":
    self._prune_impl_cls = Qwen2_5_VLSparseVLMForConditionalGeneration
```

### 4.3 Add method subclass

Add:
```py
class Qwen2_5_VLSparseVLMForConditionalGeneration(
    Qwen2_5_VLPruneForConditionalGeneration
):
    @staticmethod
    def _init_prune_method(...): ...
    @staticmethod
    def _process_image_input_for_pruning(...): ...
    @staticmethod
    def _process_video_input_for_pruning(...): ...
```

If algorithm is not ready yet, keep explicit `NotImplementedError`.

### 4.4 Token count behavior

Update `get_qwen2_5_pruned_num_tokens(...)` if `sparsevlm` needs different counting logic from `vt_prune_rate`.

## 5) Keep Registry Entry Simple

Do not add a separate registry entry for `sparsevlm` class.
Keep only:
```py
"Qwen2_5_VLPruneForConditionalGeneration": (
    "qwen2_5_vl_prune",
    "Qwen2_5_VLPruneForConditionalGeneration",
),
```

The prune class routes internally by `visual_token_pruning_method`.

## 6) EPD Script Updates (`epdtest/*`)

1. Update `epdtest/vtp_prune.sh` argument parsing:
- accept `sparsevlm`
- export `VISUAL_TOKEN_PRUNING_METHOD="sparsevlm"`

2. If needed, add new env vars for sparsevlm knobs.
3. Pass method/knobs in disagg scripts via existing `VISION_ZIP_ARGS`-style argument arrays (you can rename this array to a method-agnostic name like `VT_PRUNE_ARGS`).

## 7) Runtime Checkpoints

1. Verify prune architecture is selected:
- expected log: `Resolved architecture: Qwen2_5_VLPruneForConditionalGeneration`

2. If you still see:
- `Resolved architecture: Qwen2_5_VLForConditionalGeneration`
- add hf override in serve command:
```bash
--hf-overrides '{"architectures":["Qwen2_5_VLPruneForConditionalGeneration"]}'
```

## 8) Minimal Test Checklist

1. Config parsing:
- method accepts `sparsevlm`
- env fallback works (`VISUAL_TOKEN_PRUNING_METHOD=sparsevlm`)

2. Router behavior:
- `Qwen2_5_VLPruneForConditionalGeneration` selects sparsevlm subclass

3. Placeholder behavior:
- if sparsevlm not implemented yet, raises explicit `NotImplementedError`

4. Existing methods unaffected:
- visionzip path still works
- cdpruner path unchanged

