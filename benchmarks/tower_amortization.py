"""Tower amortization benchmark.

Reproduces Theorem 4.1 of the paper using only the cache_trie module,
no LLM required. This is the cheapest way to verify the structural
amortization claim works as advertised.

We build a deployment with:
    1 platform base (~250 tokens)
    4 community bases (~350 tokens each)
    12 bot bases (~450 tokens each, distributed across communities)
    N session leaves (each with a ~2800-token dynamic block)

We then plot:
    - naive_per_session vs N (constant: every session stores its own bases)
    - actual_per_session vs N (decreasing: bases amortize as N grows)
    - asymptote (the dynamic-block-only cost)

For large N, actual_per_session -> dynamic-only cost. The convergence
rate is O(1/N), as Theorem 4.1 predicts.

Run:
    python benchmarks/tower_amortization.py

Expected output: a series of (N, naive_kb, actual_kb, savings) rows
showing the savings ratio increasing toward ~1.4x as N grows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random
from tower_runner.cache_trie import (
    CacheTrie,
    Segment,
    SegmentLevel,
    make_simple_deployment,
)


# Realistic deployment shape — sizes match what populate.py's bases produce
# at SmolLM2 tokenization (~4 chars/token average).
PLATFORM_TOKENS = 250
COMMUNITY_TOKENS = 350
BOT_TOKENS = 450
DYN_TOKENS = 2800     # per-session dynamic block

N_COMMUNITIES = 4
BOTS_PER_COMMUNITY = 3   # 4 * 3 = 12 bots total

# How many sessions to simulate at each step
CHECKPOINTS = [10, 50, 100, 500, 1_000, 5_000, 10_000, 25_000]


def synth_tokens(n: int, seed: int) -> list[int]:
    """Make n deterministic 'tokens' from a seed. Doesn't matter what they
    are; the trie only cares about counts and identity."""
    rng = random.Random(seed)
    return [rng.randint(0, 32000) for _ in range(n)]


def build_bases():
    """Build the base segments. Returns (platform_tokens, community_dict, bot_dict)."""
    platform_tokens = synth_tokens(PLATFORM_TOKENS, seed=1)
    community_dict = {}
    bot_dict = {}
    for c_idx in range(N_COMMUNITIES):
        cid = f"community_{c_idx}"
        community_dict[cid] = synth_tokens(COMMUNITY_TOKENS, seed=100 + c_idx)
        bot_dict[cid] = {}
        for b_idx in range(BOTS_PER_COMMUNITY):
            bid = f"bot_{b_idx}"
            bot_dict[cid][bid] = synth_tokens(BOT_TOKENS, seed=1000 + c_idx * 100 + b_idx)
    return platform_tokens, community_dict, bot_dict


def simulate(N: int) -> dict:
    """Build a deployment of N sessions. Return storage statistics."""
    platform_tokens, community_dict, bot_dict = build_bases()

    sessions = {}
    rng = random.Random(42)
    for i in range(N):
        cid = rng.choice(list(community_dict.keys()))
        bid = rng.choice(list(bot_dict[cid].keys()))
        sid = f"{cid}/{bid}/sess_{i:06d}"
        sessions[sid] = synth_tokens(DYN_TOKENS, seed=10_000_000 + i)

    trie, leaves = make_simple_deployment(
        platform_tokens=platform_tokens,
        community_prompts=community_dict,
        bots_per_community=bot_dict,
        sessions=sessions,
    )

    # Naive: every session stores its full tower's worth of tokens
    naive_total_tokens = trie.total_tokens_naive(leaves)
    # Amortized: each unique segment stored once
    actual_total_tokens = trie.total_tokens_shared()

    # Per-session metrics (bytes; int32 tokens = 4 bytes each)
    bytes_per_tok = 4
    naive_total_bytes = naive_total_tokens * bytes_per_tok
    actual_total_bytes = actual_total_tokens * bytes_per_tok
    naive_per_session = naive_total_bytes / N
    actual_per_session = actual_total_bytes / N
    asymptote = DYN_TOKENS * bytes_per_tok   # dynamic-only

    return {
        "N": N,
        "n_unique_segments": len(trie.by_content_hash),
        "naive_per_session_bytes": naive_per_session,
        "actual_per_session_bytes": actual_per_session,
        "asymptote_bytes": asymptote,
        "savings_ratio": naive_per_session / actual_per_session,
        "convergence_pct": 100.0 * asymptote / actual_per_session,
    }


def main():
    print("Tower amortization benchmark (paper Theorem 4.1)")
    print("=" * 72)
    print()
    print(f"Deployment: 1 platform × {N_COMMUNITIES} communities × "
          f"{BOTS_PER_COMMUNITY} bots per community")
    print(f"  Platform base:  {PLATFORM_TOKENS} tokens")
    print(f"  Community base: {COMMUNITY_TOKENS} tokens × {N_COMMUNITIES} = "
          f"{COMMUNITY_TOKENS * N_COMMUNITIES} tokens")
    print(f"  Bot bases:      {BOT_TOKENS} tokens × {N_COMMUNITIES * BOTS_PER_COMMUNITY} = "
          f"{BOT_TOKENS * N_COMMUNITIES * BOTS_PER_COMMUNITY} tokens")
    print(f"  Total unique base tokens: "
          f"{PLATFORM_TOKENS + COMMUNITY_TOKENS * N_COMMUNITIES + BOT_TOKENS * N_COMMUNITIES * BOTS_PER_COMMUNITY}")
    print(f"  Dynamic block per session: {DYN_TOKENS} tokens")
    print()

    print(f"{'N':>8} {'naive (KB)':>13} {'actual (KB)':>13} "
          f"{'asymptote (KB)':>15} {'savings':>10} {'converged':>11}")
    print("-" * 72)

    for N in CHECKPOINTS:
        s = simulate(N)
        print(f"{s['N']:>8,} "
              f"{s['naive_per_session_bytes']/1024:>11.2f}KB "
              f"{s['actual_per_session_bytes']/1024:>11.2f}KB "
              f"{s['asymptote_bytes']/1024:>13.2f}KB "
              f"{s['savings_ratio']:>9.3f}x "
              f"{s['convergence_pct']:>10.2f}%")

    print()
    print("Reading: As N grows, actual per-session bytes converges toward")
    print("the asymptote (the dynamic block alone). The savings ratio")
    print("approaches a constant determined by the base/dynamic mix.")
    print()
    print("For the demo deployment, the limit savings ratio is:")
    print(f"  (platform + community + bot + dynamic) / dynamic")
    print(f"  = ({PLATFORM_TOKENS} + {COMMUNITY_TOKENS} + {BOT_TOKENS} + {DYN_TOKENS}) / {DYN_TOKENS}")
    print(f"  = {(PLATFORM_TOKENS + COMMUNITY_TOKENS + BOT_TOKENS + DYN_TOKENS) / DYN_TOKENS:.3f}x")


if __name__ == "__main__":
    main()
