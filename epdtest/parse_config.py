#!/usr/bin/env python3
"""Parse visual token pruning config and emit launcher-friendly key/value lines."""

import json
import sys


def emit(key: str, value) -> None:
    if value is None:
        value = ""
    print(f"{key}\t{value}")


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: parse_config.py <config_path> [visual_token_pruning_method]",
            file=sys.stderr,
        )
        return 1

    config_path = sys.argv[1]
    raw_method = (sys.argv[2] if len(sys.argv) > 2 else "").strip()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config file {config_path}: {exc}", file=sys.stderr)
        return 1

    methods = config.get("methods", {})
    if not isinstance(methods, dict):
        print(
            f"Invalid config schema in {config_path}: 'methods' must be an object",
            file=sys.stderr,
        )
        return 1

    method = raw_method.lower() if raw_method else ""
    if method and method not in methods:
        available = sorted(methods.keys())
        print(
            f"Unsupported visual token pruning method: {raw_method}. "
            f"Supported: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    method_cfg = methods.get(method, {})
    if not isinstance(method_cfg, dict):
        print(
            f"Invalid config schema in {config_path}: method '{method}' "
            "must map to an object",
            file=sys.stderr,
        )
        return 1

    emit("VISUAL_TOKEN_PRUNING_METHOD", method)
    emit("VISUAL_TOKEN_PRUNING_RATE", method_cfg.get("visual_token_pruning_rate", ""))
    emit("VISION_ZIP_DOMINANT_RATIO", method_cfg.get("vision_zip_dominant_ratio", ""))
    emit("VISION_ZIP_ATTENTION_LAYER", method_cfg.get("vision_zip_attention_layer", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
