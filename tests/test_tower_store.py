"""Unit tests for TowerStore — no llama-server required.

These tests verify the in-process logic of the tower store (trie
persistence, session storage, amortization math) without depending
on a live llama-server. The slot save/restore API calls fail gracefully
when no server is reachable; we test that the storage layer keeps
working in offline mode.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.tower_store import TowerStore
from tower_runner.cache_trie import SegmentLevel


def test_basic_trie_and_sessions():
    """Build a small trie, add sessions, verify amortization math."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # Bogus URL — slot save calls will fail silently
        store = TowerStore(tmp, target_url="http://localhost:65535")

        # Build a 3-level hierarchy
        plat = store.ensure_segment(
            parent=None,
            tokens=list(range(50)),
            level=SegmentLevel.PLATFORM,
            segment_id="platform_test",
        )
        comm = store.ensure_segment(
            parent=plat,
            tokens=list(range(50, 80)),
            level=SegmentLevel.COMMUNITY,
            segment_id="community_test_general",
        )
        bot = store.ensure_segment(
            parent=comm,
            tokens=list(range(80, 130)),
            level=SegmentLevel.BOT,
            segment_id="bot_test_general_assistant",
        )

        # Add 100 sessions sharing this bot
        for i in range(100):
            dyn_tokens = list(range(200 + i * 5, 200 + i * 5 + 3000))
            store.add_session(
                bot_seg=bot,
                dynamic_tokens=dyn_tokens,
                compressed_bytes=b"\x00" * 1500,  # fake AC bytes
                metadata={"topic": f"topic_{i}"},
                bits_per_token=4.0,
            )

        # Verify stats
        assert store.stats.n_sessions == 100, store.stats.n_sessions
        assert store.stats.n_base_segments == 3, store.stats.n_base_segments
        # 3000 tokens * 4 bytes each = 12000 per session
        expected_dyn = 100 * 3000 * 4
        assert store.stats.total_dynamic_token_bytes == expected_dyn, (
            f"got {store.stats.total_dynamic_token_bytes}, expected {expected_dyn}"
        )
        # 50 + 30 + 50 = 130 base tokens, * 4 bytes = 520 base bytes
        expected_base = (50 + 30 + 50) * 4
        assert store.stats.total_base_token_bytes == expected_base, (
            f"got {store.stats.total_base_token_bytes}, expected {expected_base}"
        )

        # Verify amortization summary makes sense
        amort = store.amortization_summary()
        # Each session: 12000 dynamic + (520 / 100 = 5.2) amortized base = ~12005
        assert amort["amortized_bytes_per_session"] - 12005 < 1, amort
        # Naive: each session pays full 520 base + 12000 dynamic = 12520
        # (using per-segment-chain average since we have only one chain)
        # Actually with 3 segments of chain (130 tokens total = 520 bytes):
        # naive = 520 + 12000 = 12520
        # amortized = 5.2 + 12000 = 12005
        # ratio = 12520 / 12005 ≈ 1.04
        # But we'd expect this to APPROACH the dynamic_only number as N grows.
        # With 100 sessions of one bot, amortization saves ~4% vs naive.
        assert amort["amortization_ratio"] > 1.0, amort

        # Verify we can reload from disk
        store2 = TowerStore(tmp, target_url="http://localhost:65535")
        assert store2.stats.n_sessions == 100
        assert store2.stats.n_base_segments == 3
        assert len(list(store2.trie.all_segments())) == 3
        assert store2.trie.root.segment_id == "platform_test"

        # Load a session and verify it links back to the correct bot
        rec = store2.load_session("sess_00000050")
        assert rec["bot_segment"].segment_id == "bot_test_general_assistant"
        assert len(rec["dynamic_tokens"]) == 3000
        assert len(rec["tower"]) == 3
        assert rec["tower"][0].segment_id == "platform_test"
        assert rec["tower"][-1].segment_id == "bot_test_general_assistant"

        print("test_basic_trie_and_sessions: OK")
        print(f"  100 sessions stored: {store.stats.total_dynamic_token_bytes:,} dynamic bytes")
        print(f"  Amortization ratio:  {amort['amortization_ratio']:.4f}x")
        print(f"  Per-session bytes:   {amort['amortized_bytes_per_session']:,.0f}")

    finally:
        shutil.rmtree(tmp)


def test_amortization_grows_with_n():
    """Verify amortization improves as N sessions grows for same M bases."""
    tmp = Path(tempfile.mkdtemp())
    try:
        store = TowerStore(tmp, target_url="http://localhost:65535")
        plat = store.ensure_segment(
            parent=None, tokens=list(range(100)),
            level=SegmentLevel.PLATFORM, segment_id="platform_x"
        )
        bot = store.ensure_segment(
            parent=plat, tokens=list(range(100, 200)),
            level=SegmentLevel.BOT, segment_id="bot_x"
        )
        # Note: skipped community for simplicity (2 levels: platform + bot)

        # Track amortization ratio every 100 sessions
        ratios_at_n = []
        for batch in range(10):
            for _ in range(100):
                store.add_session(
                    bot_seg=bot,
                    dynamic_tokens=list(range(3000)),
                    compressed_bytes=b"\x00" * 1500,
                    metadata={"topic": "x"},
                    bits_per_token=4.0,
                )
            amort = store.amortization_summary()
            ratios_at_n.append((store.stats.n_sessions,
                                amort["amortized_bytes_per_session"],
                                amort["amortization_ratio"]))

        # As N grows, amortized per-session bytes should DECREASE
        # (approach 12000 = dynamic_only) and ratio should INCREASE
        amortized_first = ratios_at_n[0][1]
        amortized_last = ratios_at_n[-1][1]
        assert amortized_last < amortized_first, (amortized_last, amortized_first)
        # Last value should be very close to 12000 (just dynamic, no base)
        assert amortized_last - 12000 < 50, amortized_last

        print("test_amortization_grows_with_n: OK")
        print(f"  N=100:  amortized {ratios_at_n[0][1]:,.1f} B/session")
        print(f"  N=500:  amortized {ratios_at_n[4][1]:,.1f} B/session")
        print(f"  N=1000: amortized {ratios_at_n[-1][1]:,.1f} B/session")

    finally:
        shutil.rmtree(tmp)


def test_dedup_segments():
    """Two segments with the same parent and tokens collapse to one node."""
    tmp = Path(tempfile.mkdtemp())
    try:
        store = TowerStore(tmp, target_url="http://localhost:65535")
        plat = store.ensure_segment(
            parent=None, tokens=list(range(50)),
            level=SegmentLevel.PLATFORM, segment_id="platform_a"
        )
        bot1 = store.ensure_segment(
            parent=plat, tokens=list(range(100, 150)),
            level=SegmentLevel.BOT, segment_id="bot_first"
        )
        # Now try to add a second bot with identical parent and tokens
        bot2 = store.ensure_segment(
            parent=plat, tokens=list(range(100, 150)),
            level=SegmentLevel.BOT, segment_id="bot_second"
        )
        # Should be the same object (content-addressed dedup)
        assert bot1 is bot2, "Dedup failed: bot1 and bot2 should be the same segment"
        # Stats should reflect only TWO segments (plat + bot)
        # Note: stats track unique segments added to disk
        print("test_dedup_segments: OK")
        print(f"  Identical (parent, tokens) collapse to one node: {bot1.segment_id}")
    finally:
        shutil.rmtree(tmp)


def main():
    print("Tower store unit tests (no llama-server required)")
    print("=" * 60)
    test_basic_trie_and_sessions()
    print()
    test_amortization_grows_with_n()
    print()
    test_dedup_segments()
    print()
    print("All tower-store tests passed.")


if __name__ == "__main__":
    main()
