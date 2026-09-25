# SPDX-License-Identifier: Apache-2.0
"""Generate an aiperf single_turn JSONL of text + video requests.

Adapted from dynamo's benchmarks/multimodal/jsonl (video-single-turn strategy,
Apache-2.0, NVIDIA). Each request samples videos from a fixed pool of
deterministic synthetic MP4s; a pool smaller than the total number of video
slots produces cross-request reuse (hits in the processor / encoder cache).

aiperf reads each local path, base64-encodes it and sends it as a
`video_url` data URL, so the server never touches the filesystem.

Needs imageio + imageio-ffmpeg (bundled ffmpeg binary, works on aarch64).
Run with the aiperf venv's python; see vlmtest/setup_aiperf.sh.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

# Short words that are one BPE token on most tokenizers, so --user-text-tokens
# lands close to the real prompt length.
_WORDS = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "it",
    "for",
    "not",
    "on",
    "with",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "by",
    "from",
    "they",
    "we",
    "say",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
]


def filler_text(rng: random.Random, n_tokens: int) -> str:
    words = [rng.choice(_WORDS) for _ in range(max(0, n_tokens - 8))]
    return "Describe what happens in the video. " + " ".join(words)


def write_synthetic_video(
    path: Path, seed: int, width: int, height: int, fps: int, seconds: int
) -> None:
    """Gradient background plus a moving rectangle, H.264 / yuv420p."""
    import imageio.v2 as imageio

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=3)
    accent = rng.integers(0, 256, size=3, dtype=np.uint8)
    rect_w, rect_h = max(4, width // 4), max(4, height // 4)
    span_x, span_y = width - rect_w + 1, height - rect_h + 1
    off_x, off_y = int(rng.integers(0, span_x)), int(rng.integers(0, span_y))
    vx = int(rng.integers(1, max(2, width // 8)))
    vy = int(rng.integers(1, max(2, height // 8)))
    xx = np.arange(width)[None, :]
    yy = np.arange(height)[:, None]

    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        output_params=["-map_metadata", "-1", "-threads", "1"],
    ) as writer:
        for t in range(fps * seconds):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (xx + base[0] + t * vx) % 256
            frame[:, :, 1] = (yy + base[1] + t * vy) % 256
            frame[:, :, 2] = (xx // 2 + yy // 2 + base[2] + t * 7) % 256
            x, y = (off_x + t * vx) % span_x, (off_y + t * vy) % span_y
            frame[y : y + rect_h, x : x + rect_w] = accent
            writer.append_data(frame)


def build_pool(args: argparse.Namespace) -> list[str]:
    w, h = args.video_size
    if w % 2 or h % 2:
        raise SystemExit(f"--video-size must be even for yuv420p, got {w}x{h}")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    pool = []
    for idx in range(args.videos_pool):
        key = f"seed{args.seed}_idx{idx:04d}_{w}x{h}_{args.fps}fps_{args.seconds}s"
        path = args.video_dir / f"synthetic_{key}.mp4"
        if not path.exists() or path.stat().st_size == 0:
            vseed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
            write_synthetic_video(path, vseed, w, h, args.fps, args.seconds)
        pool.append(str(path.resolve()))
    return pool


def sample_slots(
    rng: random.Random, pool: list[str], num_requests: int, per_req: int
) -> list[list[str]]:
    """Every pool video appears at least once; no duplicates within a request."""
    if len(pool) < per_req:
        raise SystemExit(f"--videos-pool ({len(pool)}) < --videos-per-request")
    if num_requests * per_req < len(pool):
        raise SystemExit(
            f"{num_requests}x{per_req} slots < --videos-pool ({len(pool)})"
        )
    shuffled = list(pool)
    rng.shuffle(shuffled)
    reqs: list[list[str]] = [[] for _ in range(num_requests)]
    for i, v in enumerate(shuffled):
        reqs[i % num_requests].append(v)
    for req in reqs:
        missing = per_req - len(req)
        if missing > 0:
            used = set(req)
            req.extend(rng.sample([v for v in pool if v not in used], missing))
        rng.shuffle(req)
    return reqs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--num-requests", type=int, default=200)
    p.add_argument("--videos-per-request", type=int, default=1)
    p.add_argument(
        "--videos-pool",
        type=int,
        default=None,
        help="unique videos (default: all slots unique, i.e. no reuse)",
    )
    p.add_argument("--user-text-tokens", type=int, default=300)
    p.add_argument(
        "--video-size", type=int, nargs=2, default=[320, 240], metavar=("W", "H")
    )
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seconds", type=int, default=4)
    p.add_argument("--video-dir", type=Path, default=Path("/tmp/vlmtest_videos"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    args.videos_pool = args.videos_pool or args.num_requests * args.videos_per_request
    rng = random.Random(args.seed)

    pool = build_pool(args)
    reqs = sample_slots(rng, pool, args.num_requests, args.videos_per_request)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for videos in reqs:
            row = {"text": filler_text(rng, args.user_text_tokens), "videos": videos}
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    slots = args.num_requests * args.videos_per_request
    print(
        f"{args.num_requests} requests x {args.videos_per_request} video(s), "
        f"pool {len(pool)} ({1 - len(pool) / slots:.0%} reuse), "
        f"{args.video_size[0]}x{args.video_size[1]} {args.fps}fps {args.seconds}s "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
