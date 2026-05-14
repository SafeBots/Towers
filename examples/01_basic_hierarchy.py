"""Example 1: Cache trie data structure demo.

Doesn't require a model — just demonstrates the structural amortization
of paper §4 by building a multi-tenant deployment and showing how
per-session storage decreases as the number of sessions grows.

Run:
    python examples/01_basic_hierarchy.py
"""

import sys
from pathlib import Path

# Allow running directly from the examples/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.cache_trie import make_simple_deployment, SegmentLevel


def fake_tokens(n: int, seed: int) -> list[int]:
    """Generate deterministic 'fake' token IDs for the demo.

    Real deployments use the model's tokenizer; we just need distinct
    token sequences here to demonstrate the data structure.
    """
    import random
    rng = random.Random(seed)
    return [rng.randint(0, 32000) for _ in range(n)]


def main():
    print("Towers of Segments — Example 01: Cache trie demo")
    print("=" * 60)

    # Build a small deployment: 1 platform, 3 communities, 2-3 bots each,
    # many sessions per bot.
    n_communities = 3
    bots_per_community = 3
    sessions_per_bot = 20

    platform_tokens = fake_tokens(200, seed=0)
    community_prompts = {
        f"comm{c}": fake_tokens(400, seed=10 + c)
        for c in range(n_communities)
    }
    bots_per_community_dict = {
        f"comm{c}": {
            f"bot{b}": fake_tokens(150, seed=100 + 10 * c + b)
            for b in range(bots_per_community)
        }
        for c in range(n_communities)
    }
    sessions = {
        f"comm{c}/bot{b}/sess{s}": fake_tokens(800, seed=1000 + 1000 * c + 100 * b + s)
        for c in range(n_communities)
        for b in range(bots_per_community)
        for s in range(sessions_per_bot)
    }

    trie, leaves = make_simple_deployment(
        platform_tokens=platform_tokens,
        community_prompts=community_prompts,
        bots_per_community=bots_per_community_dict,
        sessions=sessions,
    )

    n_sessions = len(leaves)
    n_segments = len(list(trie.all_segments()))
    naive = trie.total_tokens_naive(leaves)
    shared = trie.total_tokens_shared()

    print(f"\nDeployment shape:")
    print(f"  Communities:       {n_communities}")
    print(f"  Bots per community: {bots_per_community}")
    print(f"  Sessions per bot:  {sessions_per_bot}")
    print(f"  Total sessions:    {n_sessions}")
    print(f"  Total segments:    {n_segments}")

    print(f"\nStorage (in tokens, before compression):")
    print(f"  Naive (each session stores full prefix): {naive:>10,}")
    print(f"  Shared (Towers structural amortization):  {shared:>10,}")
    print(f"  Amortization ratio:                       {trie.amortization_ratio(leaves):>10.2f}x")

    # Per-session breakdown
    avg_naive = naive / n_sessions
    avg_shared = shared / n_sessions
    print(f"\nPer-session storage:")
    print(f"  Naive average:  {avg_naive:>10,.0f} tokens/session")
    print(f"  Shared average: {avg_shared:>10,.0f} tokens/session")

    # Show a tower walk
    print(f"\nSample tower (session 0):")
    for seg in trie.tower_for(leaves[0]):
        print(f"  [{seg.level.name:>9}] {seg.segment_id:<35} {seg.length:>5} tokens")

    # The asymptotic claim from Theorem 4.1: per-session storage approaches
    # |B_dyn| / L as N -> infinity. With sessions_per_bot >> 1, we're
    # already close to that limit.
    dyn_tokens = 800  # the dynamic block size
    print(f"\nAsymptotic limit (Theorem 4.1):")
    print(f"  As N -> infinity, per-session storage -> |B_dyn| = {dyn_tokens} tokens")
    print(f"  Currently at {avg_shared:.0f}/session with {n_sessions} sessions; "
          f"asymptote is {dyn_tokens}.")

    # Storage in bytes for a real 70B model
    BYTES_PER_TOKEN_FP16 = 320_000  # ~320 KB/token for Llama-3-70B at FP16, GQA
    BYTES_PER_TOKEN_AC = 3 / 8  # ~3 bits/token under AC compression

    print(f"\nProjected storage on a real 70B model:")
    print(f"  Raw FP16 (naive):       {naive * BYTES_PER_TOKEN_FP16 / 1e9:>10.2f} GB")
    print(f"  Raw FP16 (shared):      {shared * BYTES_PER_TOKEN_FP16 / 1e9:>10.2f} GB")
    print(f"  AC-compressed (shared): {shared * BYTES_PER_TOKEN_AC / 1e6:>10.2f} MB")
    raw_naive_bytes = naive * BYTES_PER_TOKEN_FP16
    ac_shared_bytes = shared * BYTES_PER_TOKEN_AC
    overall = raw_naive_bytes / ac_shared_bytes
    print(f"  Overall compression vs. raw naive: {overall:>10,.0f}x")


if __name__ == "__main__":
    main()
