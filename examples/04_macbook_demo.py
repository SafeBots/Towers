"""Example 4: End-to-end multi-tenant scenario on llama-server.

Demonstrates Tower assembly on a real model running in llama-server.
Multiple sessions share platform/community/bot bases; each session's
tower is composed from the trie.

Prerequisites:
    1. Build llama.cpp with the server enabled:
       https://github.com/ggml-org/llama.cpp
    2. Download a small GGUF model (e.g., from Hugging Face)
    3. Start llama-server in another terminal:
       llama-server -m models/llama-3.2-1b-instruct-q4.gguf -c 8192 \
                    --parallel 4 --slots --slot-save-path /tmp/towers_slots

Then run this example:
    python examples/04_macbook_demo.py

What you should see:
    - First session in a community: full prefill (slow)
    - Subsequent sessions in the same community: cache restore + small prefill
    - Repeated session: immediate restore, no prefill
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.cache_trie import CacheTrie, SegmentLevel
from tower_runner.llama_client import LlamaServerClient
from tower_runner.tower import TowerRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="llama-server URL")
    parser.add_argument("--slot-dir", default="/tmp/towers_slots",
                        help="Directory llama-server was started with via --slot-save-path")
    parser.add_argument("--reset", action="store_true",
                        help="Clear slot directory before starting")
    args = parser.parse_args()

    print("Towers of Segments — Example 04: End-to-end multi-tenant on llama-server")
    print("=" * 70)

    slot_dir = Path(args.slot_dir)
    if args.reset and slot_dir.exists():
        print(f"Resetting {slot_dir}...")
        for f in slot_dir.glob("seg_*.bin"):
            f.unlink()
    slot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConnecting to llama-server at {args.url} ...")
    client = LlamaServerClient(base_url=args.url)
    if not client.wait_until_ready(max_wait_s=10):
        print("ERROR: llama-server not responding. Start it with:")
        print(f"  llama-server -m <model.gguf> -c 8192 --parallel 4 \\")
        print(f"               --slots --slot-save-path {args.slot_dir}")
        sys.exit(1)

    props = client.props()
    print(f"  Server OK. Context: {props.get('default_generation_settings', {}).get('n_ctx', '?')}")
    print(f"  Total slots: {props.get('total_slots', '?')}")

    # Build a small deployment: 1 platform, 2 communities, 1 bot each, 3 sessions each
    runner = TowerRunner(client, slot_dir=slot_dir)
    trie = CacheTrie()

    platform_text = (
        "You are an assistant on the Magarshak platform. "
        "All responses should be concise, factual, and honest. "
        "Decline harmful requests but explain why briefly."
    )
    platform_tokens = client.tokenize(platform_text, add_special=True)
    print(f"\nPlatform base: {len(platform_tokens)} tokens")
    platform = trie.extend(None, platform_tokens, SegmentLevel.PLATFORM, segment_id="platform")
    trie.root = platform
    trie.add(platform)

    communities = {
        "tech": "This community discusses software engineering, debugging, and system design.",
        "cooking": "This community shares recipes, kitchen techniques, and food science.",
    }
    community_segs = {}
    for cid, ctext in communities.items():
        ctokens = client.tokenize(ctext, add_special=False)
        seg = trie.extend(platform, ctokens, SegmentLevel.COMMUNITY,
                          segment_id=f"community/{cid}")
        community_segs[cid] = seg
        print(f"Community '{cid}': {len(ctokens)} tokens")

    bots = {
        "tech": "You are a Python debugging assistant. Be specific, suggest tools.",
        "cooking": "You are a recipe assistant. Ask about dietary restrictions before suggesting.",
    }
    bot_segs = {}
    for cid, btext in bots.items():
        btokens = client.tokenize(btext, add_special=False)
        seg = trie.extend(community_segs[cid], btokens, SegmentLevel.BOT,
                          segment_id=f"bot/{cid}/main")
        bot_segs[cid] = seg
        print(f"Bot '{cid}/main': {len(btokens)} tokens")

    # Three sessions per community
    sessions = [
        ("tech", "user_1", "I'm debugging a memory leak in a FastAPI service. Where do I start?"),
        ("tech", "user_2", "I have a slow query in Postgres. How do I find the bottleneck?"),
        ("tech", "user_3", "My async code occasionally deadlocks. Tips for tracing it?"),
        ("cooking", "user_1", "I have leftover roast chicken. What can I make tomorrow?"),
        ("cooking", "user_2", "How do I sub silken tofu for cream in a sauce?"),
    ]

    print(f"\n{'=' * 70}")
    print("Materializing sessions (first use of each community = full prefill)")
    print(f"{'=' * 70}")

    leaves = []
    for cid, uid, user_msg in sessions:
        full_msg = f"User: {user_msg}\nAssistant:"
        msg_tokens = client.tokenize(full_msg, add_special=False)
        leaf = trie.extend(bot_segs[cid], msg_tokens, SegmentLevel.DYNAMIC,
                           segment_id=f"session/{cid}/{uid}")
        leaves.append((cid, uid, leaf))

        tower = runner.assemble(trie, leaf, slot_id=0)
        print(f"\n  [{cid:>7}/{uid}] {tower.stats.summary()}")

    # Now show what happens on repeated access — should be near-instant
    print(f"\n{'=' * 70}")
    print("Repeated access (each session already on disk → pure restore)")
    print(f"{'=' * 70}")
    for cid, uid, leaf in leaves[:3]:
        tower = runner.assemble(trie, leaf, slot_id=0)
        print(f"  [{cid:>7}/{uid}] {tower.stats.summary()}")

    # Cumulative storage cost on disk
    total_disk = sum(f.stat().st_size for f in slot_dir.glob("seg_*.bin"))
    n_segs = len(list(slot_dir.glob("seg_*.bin")))
    print(f"\n{'=' * 70}")
    print(f"Disk usage: {n_segs} segment files, {total_disk / 1e6:.2f} MB total")
    print(f"  (raw FP16 KV; AC compression in v0.2 would shrink each ~100,000x)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
