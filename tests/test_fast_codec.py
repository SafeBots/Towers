"""Test FastPLTEncoder using a real (random-weight) GPT-2 architecture.

We build a tiny GPT-2 from scratch (~150K params) with random weights.
This isn't a useful language model, but it's a real transformer with
proper past_key_values support, so the codec round-trip test is valid.

For real compression numbers you need a TRAINED model; this just verifies
the encoder/decoder logic is correct.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel

from tower_runner.fast_codec import FastPLTEncoder


class TinyTokenizer:
    """Minimum tokenizer-shaped interface FastPLTEncoder needs."""
    def __init__(self, vocab_size=256):
        self.bos_token_id = 0
        self.vocab_size = vocab_size

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def main():
    print("FastPLTEncoder smoke test (real GPT-2 architecture)")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    # Build a tiny but real GPT-2 with proper past_key_values support
    config = GPT2Config(
        vocab_size=256,
        n_positions=512,
        n_embd=64,
        n_layer=2,
        n_head=4,
    )
    model = GPT2LMHeadModel(config).eval()
    tokenizer = TinyTokenizer(vocab_size=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: tiny GPT-2 ({n_params:,} parameters, random init)")

    enc = FastPLTEncoder(model, tokenizer, device="cpu", max_chunk=256)

    # Generate test input
    n_test = 100
    tokens = np.random.randint(0, 256, size=n_test).tolist()
    print(f"\nInput: {n_test} random tokens (V=256)")

    t0 = time.time()
    encoded, stats = enc.encode_tokens(tokens)
    encode_secs = time.time() - t0
    print(f"\nEncode:")
    print(f"  Wall clock: {encode_secs:.2f}s ({n_test/encode_secs:.1f} tok/s)")
    print(f"  {stats.summary()}")

    t0 = time.time()
    decoded = enc.decode_tokens(encoded, n_test)
    decode_secs = time.time() - t0
    print(f"\nDecode:")
    print(f"  Wall clock: {decode_secs:.2f}s ({n_test/decode_secs:.1f} tok/s)")

    if decoded == tokens:
        print(f"  Round-trip: OK ({n_test} tokens recovered exactly)")
    else:
        n_match = sum(1 for a, b in zip(decoded, tokens) if a == b)
        first_mismatch = next(
            (i for i, (a, b) in enumerate(zip(decoded, tokens)) if a != b),
            len(decoded),
        )
        print(f"  Round-trip: FAILED")
        print(f"    {n_match}/{n_test} positions match ({n_match/n_test*100:.1f}%)")
        print(f"    First mismatch at position {first_mismatch}")
        if first_mismatch < len(tokens):
            print(f"    expected {tokens[first_mismatch]}, got {decoded[first_mismatch]}")
        sys.exit(1)

    print(f"\nNotes:")
    print(f"  bpt is high (~{stats.bits_per_token:.1f}) because random init can't predict")
    print(f"  much better than uniform. A real trained model gets ~3-6 bpt on text.")
    print(f"  What matters here: round-trip works, and the codec achieves the")
    print(f"  entropy floor of WHATEVER model you give it ({stats.overhead_factor:.3f}x overhead).")


if __name__ == "__main__":
    main()
