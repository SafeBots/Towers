"""Example 2: Depth-adaptive quantization (Theorem 6, paper §6).

The paper's depth-adaptive quantization theorem says segments at the base
of the hierarchy can be stored at much lower precision than the most recent
tokens, with attention output error bounded by O(alpha^(1/2) * 2^(-b)).

This example doesn't actually quantize KV state (that requires custom
kernels). It demonstrates the *size effect*: if we apply the recommended
bit-width per level, what does deployment storage look like?

For a production implementation, see the llama_patch/ directory (future work):
a per-segment quantization patch to llama.cpp that respects the level
metadata.

Run:
    python examples/02_quantized_bases.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.cache_trie import make_simple_deployment, SegmentLevel


# Bit-width recommendations from paper §6, Table 1
# (matches the depth-adaptive quantization derivation; level 0 = oldest/deepest)
BITS_PER_LEVEL = {
    SegmentLevel.PLATFORM:  4,   # platform base: heavy quantization OK
    SegmentLevel.COMMUNITY: 4,   # community base: still safe
    SegmentLevel.BOT:       6,   # bot base: slightly tighter
    SegmentLevel.DYNAMIC:   8,   # recent conversational: keep at Q8
}

# Baseline FP16
BITS_FP16 = 16


def fake_tokens(n: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed)
    return [rng.randint(0, 32000) for _ in range(n)]


def storage_with_bits(segments, bits_per_level, fp16_bytes_per_token=320_000):
    """Compute storage in bytes assuming each segment is quantized to its level's
    bit width. fp16_bytes_per_token is the per-token KV state size at FP16.
    """
    total_bytes = 0.0
    for seg in segments:
        bits = bits_per_level.get(seg.level, BITS_FP16)
        bytes_per_token = fp16_bytes_per_token * (bits / BITS_FP16)
        total_bytes += seg.length * bytes_per_token
    return total_bytes


def main():
    print("Towers of Segments — Example 02: Depth-adaptive quantization")
    print("=" * 60)

    trie, leaves = make_simple_deployment(
        platform_tokens=fake_tokens(200, seed=0),
        community_prompts={f"comm{c}": fake_tokens(400, seed=c) for c in range(3)},
        bots_per_community={
            f"comm{c}": {f"bot{b}": fake_tokens(150, seed=c * 10 + b) for b in range(3)}
            for c in range(3)
        },
        sessions={
            f"comm{c}/bot{b}/sess{s}": fake_tokens(800, seed=1000 + c * 100 + b * 10 + s)
            for c in range(3)
            for b in range(3)
            for s in range(20)
        },
    )

    all_segs = list(trie.all_segments())

    print(f"\nDeployment: {len(leaves)} sessions, {len(all_segs)} segments")

    # Tokens per level
    print(f"\nTokens per level:")
    print(f"  {'Level':<12} {'#segs':>6} {'tokens':>10} {'bits/tok':>10}")
    for level in SegmentLevel:
        segs = [s for s in all_segs if s.level == level]
        n_segs = len(segs)
        n_tokens = sum(s.length for s in segs)
        bits = BITS_PER_LEVEL[level]
        print(f"  {level.name:<12} {n_segs:>6} {n_tokens:>10,} {bits:>10}")

    # Storage comparison
    fp16_bits = {level: BITS_FP16 for level in SegmentLevel}
    fp16_total = storage_with_bits(all_segs, fp16_bits)
    q_total = storage_with_bits(all_segs, BITS_PER_LEVEL)

    print(f"\nStorage (after Towers structural amortization):")
    print(f"  Baseline FP16 everywhere: {fp16_total / 1e9:>8.2f} GB")
    print(f"  Depth-adaptive quant:     {q_total / 1e9:>8.2f} GB")
    print(f"  Reduction from quant:     {fp16_total / q_total:>8.2f}x")

    # Per-level contribution to savings
    print(f"\nWhere the savings come from:")
    for level in SegmentLevel:
        segs = [s for s in all_segs if s.level == level]
        fp16_bytes = storage_with_bits(segs, fp16_bits)
        q_bytes = storage_with_bits(segs, BITS_PER_LEVEL)
        if fp16_bytes > 0:
            saved = fp16_bytes - q_bytes
            print(f"  {level.name:<12}: saved {saved/1e9:>6.2f} GB "
                  f"({saved/fp16_bytes*100:.0f}% of level's FP16 cost)")

    # The big picture: quant on top of structural amortization is multiplicative
    # with AC compression on top of that.
    print(f"\nCompounding compression layers (paper Theorem 8.1):")
    print(f"  Raw FP16 / naive:                           ~89 GB (from example 01)")
    print(f"  + structural amortization (Theorem 4):      ~47 GB")
    print(f"  + depth-adaptive quantization (Theorem 6): ~{q_total/1e9:.1f} GB")
    print(f"  + AC compression to entropy floor:          ~0.06 MB")
    print(f"  Compose: each layer multiplies the previous one's gains.")
    print(f"\nNote: actual KV quantization not implemented in v0.1; see")
    print(f"llama_patch/README.md for production deployment notes.")


if __name__ == "__main__":
    main()
