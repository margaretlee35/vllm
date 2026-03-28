import argparse
import json
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-name", default="lmms-lab/MMVet")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cd-prune", action="store_true")  # kept for printing
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--compute-accuracy", action="store_true")
    parser.add_argument(
        "--save-accuracy-jsonl",
        type=str,
        default=None,
        help=(
            "Optional path to write per-sample prompt/expected/prediction "
            "records as JSONL."
        ),
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print prompt, expected output, prediction, and exact-match result.",
    )
    parser.add_argument(
        "--max-print-chars",
        type=int,
        default=180,
        help="Max characters shown for sample prompt/answers when --print-samples is set.",
    )

    return parser.parse_args()


def similarity(a, b):
    """Return a simple similarity score between 0 and 1"""
    return SequenceMatcher(None, a, b).ratio()


_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    normalized = _PUNCT_RE.sub(" ", text.lower().strip())
    return " ".join(normalized.split())


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def get_expected_answers(example: dict[str, Any]) -> list[str]:
    raw = example.get("answer", "")
    if isinstance(raw, (list, tuple)):
        answers = [as_text(item).strip() for item in raw]
        answers = [item for item in answers if item]
        return answers or [""]
    return [as_text(raw).strip()]


def exact_match(prediction: str, expected_answers: list[str]) -> bool:
    pred_norm = normalize_text(prediction)
    expected_norm = [
        normalize_text(answer) for answer in expected_answers if normalize_text(answer)
    ]
    if not expected_norm:
        return False
    return any(pred_norm == answer for answer in expected_norm)


def best_similarity(prediction: str, expected_answers: list[str]) -> float:
    if not expected_answers:
        return 0.0
    return max(similarity(prediction, answer) for answer in expected_answers)


def build_sample_indices(dataset_size: int, num_runs: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if num_runs <= dataset_size:
        return rng.sample(range(dataset_size), k=num_runs)
    return [rng.randrange(dataset_size) for _ in range(num_runs)]


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")
    if args.warmup >= args.num_runs:
        print(
            "[WARN] warmup >= num_runs, no samples will be left for metrics "
            "after warmup."
        )

    # -----------------------------
    # LOAD DATASET
    # -----------------------------
    print(f"[INFO] Loading dataset: {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"[INFO] Loaded {len(ds)} examples")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    print(f"[INFO] Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    latencies = []
    ttfts = []
    token_counts = []
    exact_match_flags = []
    similarity_scores = []
    sample_records: list[dict[str, Any]] = []

    sample_indices = build_sample_indices(len(ds), args.num_runs, args.seed)

    for i, sample_idx in enumerate(sample_indices):
        example = ds[sample_idx]
        image = example["image"]
        prompt_text = as_text(example.get("question", "Describe this image."))

        prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt_text}"
        mm_data = {"image": image}

        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        # -----------------------------
        # RUN INFERENCE
        # -----------------------------
        start = time.time()
        outputs = llm.generate([inputs], sampling_params)
        end = time.time()

        latency = end - start
        out = outputs[0].outputs[0]
        ttft = latency
        num_tokens = len(out.token_ids)

        if i >= args.warmup:
            latencies.append(latency)
            ttfts.append(ttft)
            token_counts.append(num_tokens)

            if args.compute_accuracy:
                gen_text = out.text
                expected_answers = get_expected_answers(example)
                expected_answers_norm = [normalize_text(x) for x in expected_answers]
                gen_text_norm = normalize_text(gen_text)
                is_exact_match = exact_match(gen_text, expected_answers)
                sim_score = best_similarity(gen_text, expected_answers)

                exact_match_flags.append(1.0 if is_exact_match else 0.0)
                similarity_scores.append(sim_score)

                record = {
                    "run_index": i,
                    "dataset_index": sample_idx,
                    "prompt": prompt_text,
                    "expected_outputs": expected_answers,
                    "normalized_expected_outputs": expected_answers_norm,
                    "model_output": gen_text,
                    "normalized_model_output": gen_text_norm,
                    "is_exact_match": is_exact_match,
                    "similarity": sim_score,
                    "latency_sec": latency,
                    "num_tokens": num_tokens,
                    "cd_prune": args.cd_prune,
                    "keep_ratio": args.keep_ratio,
                }
                sample_records.append(record)

        print(f"[{i}] latency={latency:.3f}s tokens={num_tokens}")
        if args.compute_accuracy and i >= args.warmup:
            print(
                "     accuracy "
                f"(normalized_exact_match={bool(exact_match_flags[-1])}, "
                f"similarity={similarity_scores[-1]:.3f})"
            )
            if args.print_samples:
                print(f"     prompt:   {shorten(prompt_text, args.max_print_chars)}")
                print(
                    "     expected: "
                    f"{shorten(' | '.join(record['expected_outputs']), args.max_print_chars)}"
                )
                print(
                    f"     output:   {shorten(record['model_output'], args.max_print_chars)}"
                )

    # -----------------------------
    # RESULTS
    # -----------------------------
    print("\n==== RESULTS ====")
    print(f"CD_PRUNE: {args.cd_prune}")
    print(f"KEEP_RATIO: {args.keep_ratio}")

    if latencies:
        print(f"Latency mean: {np.mean(latencies):.3f}s")
        print(f"Latency p95:  {np.percentile(latencies, 95):.3f}s")
        print(f"TTFT mean: {np.mean(ttfts):.3f}s")
    else:
        print("Latency mean: n/a (no post-warmup samples)")
        print("Latency p95:  n/a (no post-warmup samples)")
        print("TTFT mean: n/a (no post-warmup samples)")

    if latencies and np.sum(latencies) > 0:
        throughput = np.sum(token_counts) / np.sum(latencies)
        print(f"Throughput (tokens/sec): {throughput:.2f}")
    else:
        print("Throughput (tokens/sec): n/a (no post-warmup samples)")

    if args.compute_accuracy:
        if exact_match_flags:
            exact_match_acc = float(np.mean(exact_match_flags))
            print(
                "Accuracy (normalized exact match): "
                f"{exact_match_acc:.3f} "
                f"({int(np.sum(exact_match_flags))}/{len(exact_match_flags)})"
            )
            print(f"Similarity mean: {np.mean(similarity_scores):.3f}")
        else:
            print("Accuracy: n/a (no post-warmup samples)")

        if args.save_accuracy_jsonl:
            output_path = Path(args.save_accuracy_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for record in sample_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved per-sample accuracy records: {output_path}")


if __name__ == "__main__":
    main()
