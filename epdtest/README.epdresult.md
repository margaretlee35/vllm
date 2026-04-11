# EPD GPU Placement Result Summary

## NoPrune (`none`)

### Setup

Compared three `1e1p1d` GPU placements:

- `A`: `GPU_E=0, GPU_P=0, GPU_D=1`
- `B`: `GPU_E=0, GPU_P=1, GPU_D=1`
- `C`: `GPU_E=0, GPU_P=1, GPU_D=0`

Common run conditions:

- successful requests: `300`
- failed requests: `0`
- request rate: `32`
- max concurrency: `32`
- total input tokens: `307200`
- total output tokens: `38400`

### Raw Comparison

| Placement | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (`E0,P0,D1`) | 48.49 | 6.19 | 791.99 | 7127.89 | 3511.85 | 10.84 | 11.33 |
| B (`E0,P1,D1`) | 61.09 | 4.91 | 628.60 | 5657.41 | 3768.12 | 20.67 | 21.04 |
| C (`E0,P1,D0`) | 38.95 | 7.70 | 985.77 | 8871.97 | 1917.00 | 16.06 | 16.79 |

### Analysis

1. **Best throughput and TTFT**: `C (E+D colocated)`
   - Req/s is highest (`7.70`)
   - Mean TTFT is lowest (`1917 ms`)
   - Duration is shortest (`38.95 s`)

2. **Best per-token latency**: `A (E+P colocated)`
   - Mean TPOT lowest (`10.84 ms`)
   - Mean ITL lowest (`11.33 ms`)
   - Better decode smoothness than `C`

3. **Worst overall**: `B (P+D colocated)`
   - Lowest throughput (`4.91 req/s`)
   - Highest TTFT/TPOT/ITL
   - Strong evidence that prefill+decode colocation creates the most harmful contention

### Relative Deltas

- `C` vs `A`
  - Req/s: `+24.5%`
  - Mean TTFT: `-45.4%` (better)
  - Mean TPOT: `+48.2%` (worse)
  - Mean ITL: `+48.2%` (worse)

- `C` vs `B`
  - Req/s: `+56.8%`
  - Mean TTFT: `-49.1%` (better)
  - Mean TPOT: `-22.3%` (better)
  - Mean ITL: `-20.2%` (better)

- `A` vs `B`
  - Req/s: `+26.1%`
  - Mean TTFT: `-6.8%` (better)
  - Mean TPOT: `-47.6%` (better)
  - Mean ITL: `-46.2%` (better)

## VisionZip (`visionzip`)

### Setup

Compared three `1e1p1d` GPU placements:

- `A`: `GPU_E=0, GPU_P=0, GPU_D=1`
- `B`: `GPU_E=0, GPU_P=1, GPU_D=1`
- `C`: `GPU_E=0, GPU_P=1, GPU_D=0`

Common run conditions:

- request rate: `32`
- max concurrency: `32`
- target prompts: `300`

Observed request totals:

- `A`: success `300`, fail `0`
- `B`: success `299`, fail `1`
- `C`: success `300`, fail `0`

### Raw Comparison

| Placement | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (`E0,P0,D1`) | 43.43 | 6.91 | 884.11 | 7956.97 | 3084.61 | 10.83 | 11.39 |
| B (`E0,P1,D1`) | 55.83 | 5.36 | 683.22 | 6167.30 | 3711.04 | 16.41 | 16.68 |
| C (`E0,P1,D0`) | 36.12 | 8.31 | 1063.23 | 9569.06 | 1562.08 | 16.72 | 17.43 |

### Analysis

1. **Best throughput and TTFT**: `C (E+D colocated)`
   - Highest req/s (`8.31`) and output tok/s (`1063.23`)
   - Lowest mean TTFT (`1562.08 ms`)
   - Shortest duration (`36.12 s`)

2. **Best per-token latency**: `A (E+P colocated)`
   - Lowest mean TPOT (`10.83 ms`)
   - Lowest mean ITL (`11.39 ms`)

3. **Worst overall and least stable**: `B (P+D colocated)`
   - Lowest throughput
   - Highest TTFT/TPOT/ITL among the three
   - `1` failed request

### Relative Deltas

- `C` vs `A`
  - Req/s: `+20.3%`
  - Mean TTFT: `-49.4%` (better)
  - Mean TPOT: `+54.4%` (worse)
  - Mean ITL: `+53.0%` (worse)

- `C` vs `B`
  - Req/s: `+55.0%`
  - Mean TTFT: `-57.9%` (better)
  - Mean TPOT: `+1.9%` (slightly worse)
  - Mean ITL: `+4.5%` (slightly worse)

- `A` vs `B`
  - Req/s: `+28.9%`
  - Mean TTFT: `-16.9%` (better)
  - Mean TPOT: `-34.0%` (better)
  - Mean ITL: `-31.7%` (better)

## Relative Delta Comparison (`none` vs `visionzip`)

`Δ(visionzip-none)` shows how each relative delta changes when switching from NoPrune to VisionZip.

| Pair | Metric | NoPrune | VisionZip | Δ(visionzip-none) | Interpretation |
|---|---|---:|---:|---:|---|
| `C vs A` | Req/s | +24.5% | +20.3% | -4.2%p | C still leads A in throughput, but by less under VisionZip |
| `C vs A` | Mean TTFT | -45.4% | -49.4% | -4.0%p | C's TTFT lead over A is stronger under VisionZip |
| `C vs A` | Mean TPOT | +48.2% | +54.4% | +6.2%p | C's TPOT penalty vs A is larger under VisionZip |
| `C vs A` | Mean ITL | +48.2% | +53.0% | +4.8%p | C's ITL penalty vs A is larger under VisionZip |
| `C vs B` | Req/s | +56.8% | +55.0% | -1.8%p | Throughput gap stays almost the same |
| `C vs B` | Mean TTFT | -49.1% | -57.9% | -8.8%p | C's TTFT lead over B is much stronger under VisionZip |
| `C vs B` | Mean TPOT | -22.3% | +1.9% | +24.2%p | C loses TPOT advantage over B under VisionZip |
| `C vs B` | Mean ITL | -20.2% | +4.5% | +24.7%p | C loses ITL advantage over B under VisionZip |
| `A vs B` | Req/s | +26.1% | +28.9% | +2.8%p | A's throughput lead over B is slightly stronger |
| `A vs B` | Mean TTFT | -6.8% | -16.9% | -10.1%p | A's TTFT lead over B is much stronger under VisionZip |
| `A vs B` | Mean TPOT | -47.6% | -34.0% | +13.6%p | A still better than B, but TPOT gap narrows |
| `A vs B` | Mean ITL | -46.2% | -31.7% | +14.5%p | A still better than B, but ITL gap narrows |

### Cross-Method Summary

1. `C (E+D colocated)` remains the throughput/TTFT winner in both methods.
2. VisionZip strengthens TTFT separation (especially against `B`).
3. VisionZip weakens or reverses some TPOT/ITL relative advantages for `C`.

## SM Timeline Analysis (`visionzip`, same three runs)

The following `sm.log` files match the three VisionZip runs above:

- `A (E0,P0,D1)`: `logs/20260410_222825/sm.log`
- `B (E0,P1,D1)`: `logs/20260410_223035/sm.log`
- `C (E0,P1,D0)`: `logs/20260410_223327/sm.log`

### Timeline/Utilization Summary

| Placement | Warmup (s) | Both GPUs Active | One GPU Active | Idle | Key role utilization pattern |
|---|---:|---:|---:|---:|---|
| A (`E0,P0,D1`) | 7 | 75.5% | 10.2% | 14.3% | `encoder+prefill@g0`: avg SM `74.6%` (active 81.6%), `decode@g1`: avg SM `72.2%` (active 79.6%) |
| B (`E0,P1,D1`) | 7 | 23.0% | 65.6% | 11.5% | `prefill+decode@g1` near saturation: avg SM `87.3%` (active 88.5%); `encoder@g0` mostly idle: avg SM `16.8%` (active 23.0%) |
| C (`E0,P1,D0`) | 8 | 71.4% | 11.9% | 16.7% | `encoder+decode@g0`: avg SM `81.0%` (active 83.3%); `prefill@g1`: avg SM `54.0%` (active 71.4%) |

### What This Explains

1. `B (E0,P1,D1)` is bottlenecked by GPU1 contention.
   - `prefill+decode` stays close to full SM occupancy for most of the run, while encoder GPU0 is underused.
   - This matches the worst throughput (`5.36 req/s`), worst TTFT (`3711 ms`), and one failed request.

2. `A` and `C` both keep two-stage overlap high (`~71-76%` both-active), so they outperform `B`.
   - They avoid the severe single-GPU hotspot seen in `B`.

3. `C (E+D colocated)` wins throughput/TTFT because its critical decode path is co-located.
   - `encoder+decode@g0` is heavily utilized and prefill runs on a separate GPU, giving the highest req/s (`8.31`) and lowest TTFT (`1562 ms`).

4. `A (E+P colocated)` still has the best TPOT/ITL.
   - Dedicated decode GPU (`D@1`) sees strong sustained decode activity, consistent with smoother per-token latency.
