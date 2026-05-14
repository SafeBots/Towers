#!/usr/bin/env python3
"""Smoke test: verifies the v0.1 release works without any model downloads.

Runs:
    - Cache trie data structure tests
    - Arithmetic codec round-trip tests
    - Example 01 (cache trie demo)
    - Example 02 (depth-adaptive quantization sizes)

Does NOT run anything requiring a downloaded model or running llama-server.

Use this to verify the install works before downloading SmolLM2:
    python tests/smoke_test.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def section(name):
    bar = "=" * 70
    print(f"\n{bar}\n  {name}\n{bar}")


def test_cache_trie():
    from tower_runner.cache_trie import make_simple_deployment, SegmentLevel

    trie, leaves = make_simple_deployment(
        platform_tokens=list(range(100)),
        community_prompts={"a": list(range(200))},
        bots_per_community={"a": {"x": list(range(50))}},
        sessions={"a/x/s1": list(range(800)), "a/x/s2": list(range(700))},
    )
    assert len(leaves) == 2
    assert trie.root is not None
    assert trie.root.level == SegmentLevel.PLATFORM
    tower = trie.tower_for(leaves[0])
    assert [s.level.name for s in tower] == ["PLATFORM", "COMMUNITY", "BOT", "DYNAMIC"]

    # Asymptotic check: with shared bases, amortization > 1
    assert trie.amortization_ratio(leaves) > 1.0
    print("  cache trie: OK")


def test_ac_codec():
    import numpy as np
    from tower_runner.ac_codec import PLTEncoder

    # Test 1: uniform
    V = 256
    def uniform_logp(prev):
        return np.full(V, -np.log(V), dtype=np.float64)
    enc = PLTEncoder(uniform_logp, vocab_size=V)
    np.random.seed(42)
    tokens = np.random.randint(0, V, size=200).tolist()
    encoded, stats = enc.encode(tokens)
    assert abs(stats.bits_per_token - 8.0) < 0.2, f"bpt={stats.bits_per_token}"
    decoded = enc.decode(encoded, len(tokens))
    assert decoded == tokens, "Round-trip failed (uniform)"
    print(f"  ac_codec uniform V=256: {stats.bits_per_token:.2f} bpt (expected ~8.0), round-trip OK")

    # Test 2: skewed
    def skewed_logp(prev):
        logp = np.full(V, np.log(0.05 / (V - 1)), dtype=np.float64)
        logp[0] = np.log(0.95)
        return logp
    enc2 = PLTEncoder(skewed_logp, vocab_size=V)
    probs = np.full(V, 0.05 / (V - 1))
    probs[0] = 0.95
    np.random.seed(7)
    tokens2 = np.random.choice(V, size=500, p=probs).tolist()
    encoded2, stats2 = enc2.encode(tokens2)
    assert stats2.bits_per_token < 1.5, f"bpt={stats2.bits_per_token} (should be < 1.5)"
    decoded2 = enc2.decode(encoded2, len(tokens2))
    assert decoded2 == tokens2, "Round-trip failed (skewed)"
    print(f"  ac_codec skewed: {stats2.bits_per_token:.2f} bpt (expected ~0.7), round-trip OK")


def test_fast_codec():
    """Test FastPLTEncoder using a real (random-weight) tiny GPT-2.

    Skipped silently if torch + transformers aren't installed (smoke test
    is meant to work without the v0.2 demo deps).
    """
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        print("  fast_codec: SKIPPED (torch or transformers not installed)")
        return

    import numpy as np
    from tower_runner.fast_codec import FastPLTEncoder

    torch.manual_seed(42)
    np.random.seed(42)
    config = GPT2Config(
        vocab_size=256, n_positions=512, n_embd=64, n_layer=2, n_head=4,
    )
    model = GPT2LMHeadModel(config).eval()

    class TinyTok:
        bos_token_id = 0
        def encode(self, t, add_special_tokens=False):
            return [ord(c) % 256 for c in t]
        def decode(self, ts):
            return "".join(chr(t) for t in ts)

    enc = FastPLTEncoder(model, TinyTok(), device="cpu", max_chunk=512)
    tokens = np.random.randint(0, 256, size=80).tolist()
    encoded, stats = enc.encode_tokens(tokens)
    # Random init: bpt should be very close to log2(256) = 8
    assert 7.5 < stats.bits_per_token < 8.6, f"bpt={stats.bits_per_token}"
    # Overhead vs entropy floor should be small
    assert stats.overhead_factor < 1.05, f"overhead={stats.overhead_factor}"
    print(f"  fast_codec encode: {stats.bits_per_token:.2f} bpt "
          f"({stats.overhead_factor:.3f}x overhead), {len(tokens)} tokens -> "
          f"{stats.compressed_bytes} bytes")



def test_tower_store():
    """Verify the persistent tower store works end-to-end without llama-server."""
    import tempfile
    import shutil
    from tower_runner.tower_store import TowerStore
    from tower_runner.cache_trie import SegmentLevel

    tmp = Path(tempfile.mkdtemp())
    try:
        store = TowerStore(tmp, target_url="http://localhost:65535")
        plat = store.ensure_segment(
            parent=None, tokens=list(range(50)),
            level=SegmentLevel.PLATFORM, segment_id="platform_demo",
        )
        bot = store.ensure_segment(
            parent=plat, tokens=list(range(50, 100)),
            level=SegmentLevel.BOT, segment_id="bot_demo",
        )
        for i in range(50):
            store.add_session(
                bot_seg=bot,
                dynamic_tokens=list(range(200 + i, 200 + i + 1000)),
                compressed_bytes=b"\x00" * 500,
                metadata={"topic": f"t_{i}"},
                bits_per_token=4.0,
            )
        assert store.stats.n_sessions == 50
        # Reload and verify persistence
        store2 = TowerStore(tmp, target_url="http://localhost:65535")
        assert store2.stats.n_sessions == 50
        rec = store2.load_session("sess_00000025")
        assert rec["bot_segment"].segment_id == "bot_demo"
        assert len(rec["dynamic_tokens"]) == 1000
        # Verify amortization sanity
        amort = store2.amortization_summary()
        assert amort["amortization_ratio"] > 1.0
        print(f"  tower_store: 50 sessions persist, "
              f"amortization {amort['amortization_ratio']:.4f}x")
    finally:
        shutil.rmtree(tmp)


def test_example_01():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "01_basic_hierarchy.py")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"example 01 failed:\n{result.stderr}"
    # Sanity-check key terms appear in the output
    assert "Amortization ratio" in result.stdout
    assert "Asymptotic limit" in result.stdout
    print("  example 01 (cache trie demo): OK")


def test_example_02():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "02_quantized_bases.py")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"example 02 failed:\n{result.stderr}"
    assert "Depth-adaptive quant" in result.stdout
    print("  example 02 (quantization): OK")


def main():
    section("Smoke tests (no model download required)")
    test_cache_trie()
    test_ac_codec()
    test_fast_codec()
    test_tower_store()
    test_example_01()
    test_example_02()
    print("\nAll smoke tests passed.")
    print("\nNext steps:")
    print("  - See Theorem 4.1 in action:          python benchmarks/tower_amortization.py")
    print("  - Run AC compression on a real LM:    python examples/03_ac_compress.py")
    print("  - Run end-to-end demo on llama-server: python examples/04_macbook_demo.py")
    print("  - Empirical 80,000x ratio:            python benchmarks/amortization.py")
    print("  - Measure thaw latency:               python benchmarks/thaw_latency.py")
    print("  - Demo for Scoble podcast:            bash demo/setup.sh")


if __name__ == "__main__":
    main()
