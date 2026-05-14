"""amortization.py — Empirically measure the headline storage ratio.

This is the benchmark that turns the paper's claim into a number you can
read off the screen. Run it against a populated cache and it computes
the actual compression ratio for several target-model configurations,
using ONLY the actual bytes on disk and no extrapolation.

What it computes:

    For each (target_model, quant_level) configuration:
        raw_kv_bytes_for_session_at_target = avg_tokens_per_session * bytes_per_token(target)
        stored_bytes_for_session = (
            avg_dynamic_token_bytes  # what populate.py wrote per session
            + amortized_base_bytes   # base / N_sessions
        )
        ratio = raw_kv_bytes_for_session / stored_bytes_for_session

    For AC-compressed-only storage (the paper's headline mode):
        stored_bytes = avg_compressed_bytes_per_session + amortized_base_bytes
        compressed_ratio = raw_kv_bytes / stored_bytes

What it does NOT do:

    Does not project to imaginary larger deployments. Every number is
    measured from real files on real disk in --cache-dir.
    Does not assume any particular target model is running. The target
    parameter is just the per-token byte rate of the model's KV cache,
    which is a well-known number for each architecture.

Usage:

    # After populate.py has generated some sessions:
    python benchmarks/amortization.py --cache-dir ~/towers_cache

    # JSON output for piping into other tools:
    python benchmarks/amortization.py --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.tower_store import TowerStore


# KV cache bytes per token for several target models. These are from
# published model architecture details:
#   bytes_per_token = 2 (FP16) * n_layers * 2 (K and V) * n_kv_heads * head_dim
#
# We list:
#   Qwen2.5-14B (GQA n_kv_heads=8): 48 layers * 2 * 8 * 128 * 2 = 196,608 B  ≈ 192 KB
#   Llama-3-8B (GQA n_kv_heads=8):  32 * 2 * 8 * 128 * 2          = 131,072 B  ≈ 128 KB
#   Llama-3-70B (GQA n_kv_heads=8): 80 * 2 * 8 * 128 * 2          = 327,680 B  ≈ 320 KB
#   Llama-3-405B (n_kv_heads=8):    126 * 2 * 8 * 128 * 2         = 516,096 B  ≈ 504 KB
#   Llama-3-405B (no GQA, n_heads=128): 126*2*128*128*2           = 8,257,536 B ≈ 8 MB  (theoretical max)
#
# For Q4 quantized models the WEIGHTS are 4-bit but the KV cache typically
# remains in FP16 unless explicitly quantized; we list both.
TARGET_KV_BYTES_PER_TOKEN = [
    ("Qwen2.5-14B  (GQA, FP16 KV)",  192_000),
    ("Llama-3-8B   (GQA, FP16 KV)",  131_000),
    ("Llama-3-70B  (GQA, FP16 KV)",  320_000),
    ("Llama-3-405B (GQA, FP16 KV)",  504_000),
    ("Llama-3-70B  (GQA, INT8 KV)",  160_000),
    ("Llama-3-70B  (GQA, INT4 KV)",   80_000),
]


def human_bytes(n: float) -> str:
    for unit, lim in [("B", 1024), ("KB", 1024**2), ("MB", 1024**3), ("GB", 1024**4)]:
        if abs(n) < lim:
            return f"{n / (lim/1024):.2f} {unit}"
    return f"{n / 1024**4:.2f} TB"


def compute_amortization(cache_dir: Path) -> dict:
    """Read the cache and compute the measured ratios."""
    store = TowerStore(cache_dir)
    stats = store.stats
    if stats.n_sessions == 0:
        return {
            "error": f"No sessions in {cache_dir}. Run populate.py first.",
            "n_sessions": 0,
        }

    # Per-session averages
    avg_dyn_token_bytes = stats.total_dynamic_token_bytes / stats.n_sessions
    avg_dyn_compressed = stats.total_dynamic_compressed_bytes / stats.n_sessions
    base_bytes_amortized = stats.total_base_token_bytes / stats.n_sessions

    # Approximate session token count from dynamic + per-bot base average
    avg_dyn_tokens = avg_dyn_token_bytes / 4  # int32 stored tokens
    if stats.n_base_segments > 0:
        avg_base_chain_tokens = stats.total_base_token_bytes / stats.n_base_segments / 4
    else:
        avg_base_chain_tokens = 0
    avg_session_tokens = avg_dyn_tokens + avg_base_chain_tokens

    # Two storage modes to report:
    #   A) Tokens only (no AC): dynamic tokens + amortized base tokens
    #   B) AC bytes + amortized base tokens (the paper's headline mode)
    tokens_mode_bytes = avg_dyn_token_bytes + base_bytes_amortized
    compressed_mode_bytes = avg_dyn_compressed + base_bytes_amortized

    rows = []
    for target_label, bytes_per_token in TARGET_KV_BYTES_PER_TOKEN:
        raw_kv_bytes = avg_session_tokens * bytes_per_token
        ratio_tokens = raw_kv_bytes / tokens_mode_bytes if tokens_mode_bytes > 0 else 0
        ratio_compressed = raw_kv_bytes / compressed_mode_bytes if compressed_mode_bytes > 0 else 0
        rows.append({
            "target": target_label,
            "kv_bytes_per_token": bytes_per_token,
            "raw_kv_bytes_per_session": raw_kv_bytes,
            "ratio_tokens_mode": ratio_tokens,
            "ratio_compressed_mode": ratio_compressed,
        })

    return {
        "cache_dir": str(cache_dir),
        "n_sessions": stats.n_sessions,
        "n_base_segments": stats.n_base_segments,
        "avg_dynamic_tokens": avg_dyn_tokens,
        "avg_session_tokens": avg_session_tokens,
        "avg_base_chain_tokens": avg_base_chain_tokens,
        "total_base_bytes": stats.total_base_token_bytes,
        "total_dynamic_token_bytes": stats.total_dynamic_token_bytes,
        "total_dynamic_compressed_bytes": stats.total_dynamic_compressed_bytes,
        "amortized_base_bytes_per_session": base_bytes_amortized,
        "tokens_mode_bytes_per_session": tokens_mode_bytes,
        "compressed_mode_bytes_per_session": compressed_mode_bytes,
        "rows": rows,
    }


def print_report(result: dict) -> None:
    if "error" in result:
        print(result["error"])
        return

    print("Towers of Segments — empirical amortization measurement")
    print("=" * 72)
    print(f"  Cache:           {result['cache_dir']}")
    print(f"  N sessions:      {result['n_sessions']:>12,}")
    print(f"  N base segments: {result['n_base_segments']:>12}")
    print()

    print("  Measured per-session storage (NOT projected):")
    print(f"    Avg dynamic tokens:           {result['avg_dynamic_tokens']:>10,.0f}")
    print(f"    Avg full-session tokens:      {result['avg_session_tokens']:>10,.0f}")
    print(f"    Tokens-mode bytes/session:    "
          f"{human_bytes(result['tokens_mode_bytes_per_session']):>12}")
    print(f"    Compressed-mode bytes/session: "
          f"{human_bytes(result['compressed_mode_bytes_per_session']):>11}")
    print(f"    Amortized base bytes/session: "
          f"{human_bytes(result['amortized_base_bytes_per_session']):>12} "
          f"(approaches zero as N grows)")
    print()

    print("  Storage compression vs raw FP16 KV cache (for various target models):")
    print()
    print(f"  {'Target model':<32} {'KV bytes/tok':>12}   "
          f"{'tokens-only':>12}  {'AC-compressed':>14}")
    print(f"  {'-'*32}  {'-'*12}   {'-'*12}  {'-'*14}")
    for row in result["rows"]:
        print(f"  {row['target']:<32}  "
              f"{row['kv_bytes_per_token']:>10,}  "
              f"{row['ratio_tokens_mode']:>10,.0f}x  "
              f"{row['ratio_compressed_mode']:>12,.0f}x")
    print()
    print("  Reading the table:")
    print(f"    'tokens-only' = ratio if we ONLY stored raw token ids per session.")
    print(f"                    This is what populate.py actually writes today.")
    print(f"    'AC-compressed' = ratio with AC compression of the dynamic block.")
    print(f"                      Subject to the codec-determinism caveat in fast_codec.py.")
    print()
    print("  These ratios INCLUDE the amortized base-segment cost. As N grows")
    print(f"  ({result['n_sessions']:,} so far), base cost per session approaches zero.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Empirical amortization benchmark")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "towers_cache")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    result = compute_amortization(args.cache_dir)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
