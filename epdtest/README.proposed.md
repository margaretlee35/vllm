# Proposed Design: PD Preempts E

This note explains the core idea behind "prefill/decode (PD) preempting encode (E)" in `disagg_Ne1p1d_pd_preempt.sh`.

## 1) Goal

When encoder and PD share a GPU, prioritize latency-critical PD work over encoder work.

- `P` (prefill) and `D` (decode) are on the serving critical path.
- `E` (multimodal encoder) is important but should yield when contention appears.

## 2) Core Idea

The script applies **priority bias**, not kernel-level hard preemption.

- On shared GPUs, encoder processes are deprioritized.
- Prefill/decode processes get more scheduling share.
- On non-shared GPUs, behavior remains normal.

In other words: "If E and PD contend on the same GPU, PD should win most of the time."

## 3) Step-by-Step Mechanism

### Step 1: Parse topology and detect overlaps

The script parses `GPU_E`, `GPU_P`, `GPU_D`, then computes which encoder workers share a GPU with prefill or decode.

- Shared detection logic: `configure_pd_preempt_layout()`
- Outputs include:
  - `ENCODE_SHARED_WITH_PD[idx]`: per-encoder flag (`1`/`0`). `1` means encoder worker `idx` is on a GPU shared with prefill or decode.
  - `PREFILL_SHARES_ENCODER_GPU`: global flag (`1`/`0`). `1` means prefill GPU is also used by at least one encoder worker.
  - `DECODE_SHARES_ENCODER_GPU`: global flag (`1`/`0`). `1` means decode GPU is also used by at least one encoder worker.
  - `PREEMPT_SHARED_GPU_CSV`: CSV list of shared GPU IDs where PD-over-E bias should be applied.

If no overlap exists, no special preempt bias is applied.

### Step 2: Define preempt bias knobs

The script sets default knobs:

- `ENCODER_NICE=10`
- `ENCODER_MPS_ACTIVE_THREAD_PERCENTAGE=20`
- `PD_MPS_ACTIVE_THREAD_PERCENTAGE=100`
- `ENCODER_CUDA_DEVICE_MAX_CONNECTIONS=1`
- `PD_CUDA_DEVICE_MAX_CONNECTIONS=32`

Interpretation:

- Encoder gets lower CPU scheduling priority (`nice`).
- Encoder gets smaller GPU scheduler share on overlap.
- PD gets full/large share on overlap.

What each knob means:

- `ENCODER_NICE=10`: run encoder with a positive nice value (lower CPU priority).
- `ENCODER_MPS_ACTIVE_THREAD_PERCENTAGE=20`: cap encoder CUDA MPS active thread share to 20% on shared GPUs.
- `PD_MPS_ACTIVE_THREAD_PERCENTAGE=100`: allow prefill/decode to use full CUDA MPS thread share on shared GPUs.
- `ENCODER_CUDA_DEVICE_MAX_CONNECTIONS=1`: limit encoder CUDA work-queue connection fanout (more conservative scheduling behavior).
- `PD_CUDA_DEVICE_MAX_CONNECTIONS=32`: allow broader PD CUDA connection fanout (more aggressive scheduling opportunity).

### Step 3: Launch encoder with reduced priority on shared GPUs

For each encoder worker:

- If encoder GPU overlaps with `P` or `D`:
  - set encoder env:
    - `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$ENCODER_MPS_ACTIVE_THREAD_PERCENTAGE`
    - `CUDA_DEVICE_MAX_CONNECTIONS=$ENCODER_CUDA_DEVICE_MAX_CONNECTIONS`
  - start encoder via:
    - `nice -n $ENCODER_NICE ...`
- If no overlap:
  - launch encoder normally.

This is the main "E yields to PD" action.

### Step 4: Launch prefill/decode with preferred share on shared GPUs

For prefill and decode:

- If they share a GPU with any encoder:
  - set env:
    - `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PD_MPS_ACTIVE_THREAD_PERCENTAGE`
    - `CUDA_DEVICE_MAX_CONNECTIONS=$PD_CUDA_DEVICE_MAX_CONNECTIONS`
- Otherwise launch normally.

This complements Step 3 by explicitly favoring PD workers in contention.

### Step 5: Keep dataflow unchanged

The script keeps the original E -> P -> D data path intact.

- Encoder cluster still runs as producers.
- Prefill/Decode still use configured KV connector.
- Proxy routing remains unchanged.

Only scheduling bias changes; model semantics do not.

## 4) Why This Improves Latency

When shared GPUs are overloaded:

- PD receives more immediate GPU scheduling opportunities.
- Encoder is less likely to delay decode token generation.
- TTFT/TPOT/ITL tails typically improve, especially under high concurrency.

This is why PD-preempt variants often show better serving responsiveness.

## 5) Important Limitation

This is **soft preemption**, not hardware force-preempt.

- CUDA kernels already running are not forcibly interrupted.
- Behavior depends on workload shape, kernel lengths, and driver/runtime scheduling.
- Expect "PD usually wins," not "PD always instantly interrupts E."

## 6) What `nice` Is (Briefly)

`nice` is a Linux CPU scheduling priority setting for processes.

- Higher nice value (for example `+10`) means lower CPU priority.
- Lower/negative nice value means higher CPU priority.

In this design, encoder gets a higher nice value so it yields CPU time more easily when PD is active.

## 7) Practical Tuning Order

If you want stronger PD priority, tune in this order:

1. Lower `ENCODER_MPS_ACTIVE_THREAD_PERCENTAGE` (e.g., `20 -> 10`).
2. Increase `ENCODER_NICE` (e.g., `10 -> 15`).
3. Keep `PD_MPS_ACTIVE_THREAD_PERCENTAGE=100`.
4. Keep encoder `CUDA_DEVICE_MAX_CONNECTIONS` low and PD high.

If encoder starvation appears, relax the first two knobs.

## 8) Key Files

- PD-preempt implementation: `epdtest/scripts/disagg_Ne1p1d_pd_preempt.sh`
- Topology entrypoint routing: `epdtest/run.sh`

## 9) One-Line Summary

"PD preempts E" here means: **on shared GPUs, we intentionally lower encoder scheduling priority and increase prefill/decode scheduling share so critical-path serving work gets GPU time first.**
