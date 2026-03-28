# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

### Implementation advice: add prompt-level accuracy tracking to your benchmark

If you are building your own pruning benchmark script, include the same core
pattern used in `benchmark_cd_pruner.py` so you can compare quality before/after
pruning on a per-prompt basis.

#### 1) Add these helper functions

```py
import re
from difflib import SequenceMatcher

_PUNCT_RE = re.compile(r"[^\w\s]")

def normalize_text(text: str) -> str:
    normalized = _PUNCT_RE.sub(" ", text.lower().strip())
    return " ".join(normalized.split())

def get_expected_answers(example: dict) -> list[str]:
    raw = example.get("answer", "")
    if isinstance(raw, (list, tuple)):
        answers = [str(x).strip() for x in raw if str(x).strip()]
        return answers or [""]
    return [str(raw).strip()]

def exact_match(prediction: str, expected_answers: list[str]) -> bool:
    pred_norm = normalize_text(prediction)
    expected_norm = [normalize_text(x) for x in expected_answers if normalize_text(x)]
    if not expected_norm:
        return False
    return any(pred_norm == x for x in expected_norm)

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def best_similarity(prediction: str, expected_answers: list[str]) -> float:
    if not expected_answers:
        return 0.0
    return max(similarity(prediction, x) for x in expected_answers)
```

#### 2) Add per-sample scoring inside your inference loop

```py
if args.compute_accuracy and i >= args.warmup:
    gen_text = out.text
    expected_answers = get_expected_answers(example)
    is_exact_match = exact_match(gen_text, expected_answers)
    sim_score = best_similarity(gen_text, expected_answers)

    exact_match_flags.append(1.0 if is_exact_match else 0.0)
    similarity_scores.append(sim_score)

    sample_records.append(
        {
            "run_index": i,
            "prompt": prompt_text,
            "expected_outputs": expected_answers,
            "model_output": gen_text,
            "normalized_expected_outputs": [normalize_text(x) for x in expected_answers],
            "normalized_model_output": normalize_text(gen_text),
            "is_exact_match": is_exact_match,
            "similarity": sim_score,
            "latency_sec": latency,
            "num_tokens": len(out.token_ids),
        }
    )
```

#### 3) Report aggregate metrics at the end

```py
if args.compute_accuracy and exact_match_flags:
    exact_match_acc = float(np.mean(exact_match_flags))
    print(
        f"Accuracy (normalized exact match): {exact_match_acc:.3f} "
        f"({int(np.sum(exact_match_flags))}/{len(exact_match_flags)})"
    )
    print(f"Similarity mean: {np.mean(similarity_scores):.3f}")
```

#### 4) Save JSONL for offline analysis

```py
if args.save_accuracy_jsonl:
    with open(args.save_accuracy_jsonl, "w", encoding="utf-8") as f:
        for row in sample_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

This gives you prompt-by-prompt evidence when pruning harms quality.

#### 5) Run baseline vs pruned with identical conditions

When comparing pruning strategies, keep all non-pruning settings fixed:

- same dataset/split and number of prompts
- same seed and warmup
- same generation parameters

Then compare:

- normalized exact-match accuracy (primary)
- mean similarity (secondary)
- latency/throughput (performance)
