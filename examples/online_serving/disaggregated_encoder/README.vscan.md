# VScan — Two-Stage Visual Token Pruning for Qwen2.5-VL

VScan is a two-stage visual token pruning method that reduces the number of
visual tokens fed to the LLM, improving prefill throughput while maintaining
strong accuracy.

Reference: *"VScan: Rethinking Visual Token Reduction for Efficient Large
Vision-Language Models"*

---

## Quick Start

```bash
# Stage 1 + Stage 2, ~16.7% Stage-1 retention (default)
TIMEOUT_SECONDS=600 NUM_PROMPTS=500 \
  uv run --extra bench ./lovelace/non_disagg_pruning_example.sh vscan

# Custom prune rate (e.g., prune 50% at Stage 1 → r1=0.5 retained)
TIMEOUT_SECONDS=600 NUM_PROMPTS=500 VISUAL_PRUNING_RATE=0.5 \
  uv run --extra bench ./lovelace/non_disagg_pruning_example.sh vscan

# Smoke test (10 prompts)
NUM_PROMPTS=10 uv run --extra bench ./lovelace/non_disagg_pruning_example.sh vscan
```

---

## How It Works

### Stage 1 — Visual Encoder (before LLM prefill)

Runs **after** encoding, **before** the LLM prefix phase.

1. **Global Scan** — At the *final* ViT block, compute self-attention column
   sums (each token's score = average attention it receives from all other
   tokens). Select the top `R1_global = r1 × 50%` tokens.

2. **Local Scan** — At a *shallow* ViT block (`local_layer=8` for Qwen2.5-VL),
   compute the same column-sum scores. Divide the merged-token grid into
   non-overlapping windows; select the top tokens *per window* with uniform
   budget. This retains fine-grained local details. Local tokens that were
   already selected by the global scan are excluded.

3. **Token Merge** — For every unselected token, find the cosine-nearest
   selected token and average-merge it in. Selected positions hold enriched
   content; unselected positions are zeroed.

Net retention: `r1 = 16.7%` by default (≈ 96/576 for a 448×448 image).

### Stage 2 — LLM Middle Layer (during prefill at layer k)

At decoder layer `prune_layer=14` (Qwen2.5-VL default):

1. Extract the query of the **last instruction token** and keys of all
   remaining visual tokens.
2. Compute a manual `softmax(Q_last · K_visual^T / sqrt(d))` averaged across
   heads (bypasses FlashAttention — the score is computed in plain PyTorch).
3. Keep the top `r2 = 33.3%` visual tokens; zero-out and KV-cache-invalidate
   the rest.

Overall retention: `r1 × r2 ≈ 5.6%` of original tokens (≈ 32/576).

---

## Hyperparameter Reference

| Parameter | Qwen2.5-VL | LLaVA-series | Description |
|-----------|-----------|-------------|-------------|
| `local_layer` (l) | 8 | 6 | Shallow ViT layer for local scan |
| `prune_layer` (k) | 14 | 16 | LLM decoder layer for Stage 2 |
| `r1` | 16.7% | 16.7% | Stage-1 token retention rate |
| `r2` | 33.3% | 33.3% | Stage-2 token retention rate |
| Global/Local split | 50/50 | 50/50 | Budget allocation |

### Environment Variable Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `VSCAN_R1` | `1 - VISUAL_PRUNING_RATE` | Stage-1 retention fraction |
| `VSCAN_R2` | `0.333` | Stage-2 retention fraction |
| `VSCAN_LOCAL_LAYER` | `8` | ViT local scan layer index |
| `VSCAN_PRUNE_LAYER` | `14` | LLM Stage-2 prune layer index |

### Mapping `VISUAL_PRUNING_RATE` to r1/r2

`--visual-pruning-rate P` sets the Stage-1 prune rate so `r1 = 1 - P`.
Stage 2 is always `r2 = 0.333` unless overridden via `VSCAN_R2`.

Example: `VISUAL_PRUNING_RATE=0.833` → `r1 = 0.167` (paper default).

---

## Server Launch Arguments

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --mm-pruning-method vscan \
    --visual-pruning-rate 0.833 \
    --hf-overrides '{"architectures":["Qwen2_5_VLForConditionalGenerationVScan"]}'
```

---

## Token Count Verification

For a 448×448 image with Qwen2.5-VL-3B (spatial_merge_size=2, patch_size=14):
- Raw ViT patches: 32×32 = 1024
- After merger: 16×16 = 256 merged tokens
- After Stage 1 (r1=16.7%): ~43 tokens
- After Stage 2 (r2=33.3%): ~14 tokens

For a typical 672×672 image:
- After merger: 576 merged tokens
- After Stage 1 (r1=16.7%): ~96 tokens
- After Stage 2 (r2=33.3%): ~32 tokens

---

## Known Limitations

### Stage 2 KV-cache behaviour
Stage 2 runs **during** prefill at layer k=14.  In the default vLLM
scheduler, physical KV-cache blocks are allocated for the full sequence before
prefill starts and are **not reclaimed** mid-prefill.  Stage 2 therefore
provides algebraic savings (cheaper attention for layers > k) but does **not**
free GPU memory.

### Stage 1 does provide real savings
Stage 1 merges and zeros visual tokens **before** the LLM prefix phase.
Zeroed token slots have their KV-cache slot mappings invalidated
(`slot_mapping = -1`), so subsequent attention layers skip writing/reading
those slots.  This is the primary source of throughput improvement.

### For maximum savings
Use the disaggregated encoder setup (`disagg_1e1pd_pruning_example.sh`) where
the encoder runs on a separate GPU; the pruned (Stage-1 merged) token count
is what crosses the PCIe/NVLink boundary to the prefill GPU.

---

## Expected Results (from paper)

| Model | Method | Retention | Prefill Speedup | Accuracy |
|-------|--------|-----------|----------------|----------|
| LLaVA-NeXT-7B | Baseline | 100% | 1× | — |
| LLaVA-NeXT-7B | VScan | 11.1% | 2.91× | ~97% of baseline |
| Qwen2.5-VL-7B | VScan | 11.1% | ~2.5× | ~97% of baseline |

---

## Implementation Notes

- `vllm/multimodal/visual_token_pruning/vscan.py` — core algorithms
- `vllm/model_executor/models/qwen2_5_vl_vscan.py` — visual encoder hooks + Stage-1
- `vllm/model_executor/models/qwen2_vscan.py` — Stage-1 merge + Stage-2 LLM hooks
