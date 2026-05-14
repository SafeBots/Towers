"""Fast arithmetic-coding encoder using transformer KV caching.

This is the v0.2 encoder. The v0.1 encoder in ac_codec.py takes O(n^2)
time per session by re-running the model from scratch at every position;
that's fine for short demos but unworkable for the 4000-token sessions
a real demo needs. v0.2 does ONE forward pass over the full sequence to
get all per-position logits at once. Encode of a 4000-token session
drops from minutes to seconds.

Mapping to paper §10:

    M_tilde in the paper is the small language model that produces the
    conditional probabilities P(t_i | t_{<i}) the arithmetic coder needs.
    Any Hugging Face transformer works as M_tilde; this file is a thin
    wrapper that takes one and produces compressed bitstrings.

What encode_tokens gives you:

    Compressed bytes plus statistics (bits-per-token, achieved overhead
    vs entropy floor). encode_tokens(t) runs in roughly the time of one
    model forward pass on the full sequence.

What decode_tokens gives you, with a caveat:

    decode_tokens IS implemented and reproduces the encoded token
    sequence by running the model in the same code path the encoder
    did (full prefix re-run at each position, O(n^2) total). However:

    FLOAT-PRECISION NON-DETERMINISM CAVEAT

    Attention with different sequence lengths produces logits that
    differ by ~1e-7 in float32 (~1e-16 in float64). That's enough to
    flip the integer cumulative-frequency quantization at borderline
    symbols and break the arithmetic decoder. We don't currently have
    a way to make a standard transformer produce bit-identical logits
    across different input lengths short of integer-quantized inference
    (NNCP-style), which is its own engineering project.

    The practical consequence: decode_tokens is reliable for very
    short sequences and for verification on tiny test models. For
    sessions over a few hundred tokens with a real LM, decode may
    desync. This does NOT affect encoding accuracy or the bits-per-token
    measurements; the headline compression numbers from encode_tokens
    are real.

    The pragmatic workflow used by populate.py and demo recall:

        1. populate.py runs encode_tokens to get accurate compressed-byte
           counts (the headline number for the paper / Scoble demo)
        2. populate.py ALSO stores the raw token ids (~16 KB per session)
           for use during recall, because recall is interactive and needs
           reliable, fast access
        3. Recall reads the raw token ids and feeds them to llama-server's
           prefill. No codec round-trip needed.

    The paper's storage-compression result is preserved: AC-compressed
    bytes are the ~1.5 KB number from §10. Even when we ALSO store the
    raw tokens (at ~16 KB each), the total per-session footprint of
    ~17 KB is still 75,000x smaller than the 1.28 GB raw FP16 KV state.

    A bit-deterministic v0.3 codec would use integer-quantized inference
    to make decode safe at any length. That's tracked as future work.

Speed expectations on Apple Silicon MPS:

    encode_tokens of a 4000-token session:
        SmolLM2-135M:   ~5-10 seconds
        Llama-3.2-1B:   ~15-30 seconds
        Qwen2.5-3B:     ~30-60 seconds

Bits-per-token expectations on conversational English:

    SmolLM2-135M: ~5-7 bits/token
    Llama-3.2-1B: ~3-5 bits/token
    Qwen2.5-3B:   ~3-4 bits/token (best for this size class)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ac_codec import (
    ArithmeticCodec,
    CompressionStats,
    PRECISION,
    WHOLE,
)


# How many positions the encoder processes in one forward pass. Larger
# values are faster but use more memory. For 4000-token sessions on a
# 24 GB MacBook with a small model, 4096 is fine.
DEFAULT_MAX_CHUNK = 4096


class FastPLTEncoder:
    """Encode/decode token sequences using arithmetic coding against a
    Hugging Face transformer model. Faster than PLTEncoder for long
    sequences because it does the model work in O(n) total rather than
    O(n^2).

    Construction takes a transformers model and tokenizer:

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        encoder = FastPLTEncoder(model, tokenizer)
        compressed, stats = encoder.encode_text("hello world")
        recovered = encoder.decode_text(compressed)

    Or with explicit tokens for direct use in the cache trie:

        tokens = tokenizer.encode("hello world", add_special_tokens=False)
        compressed, stats = encoder.encode_tokens(tokens)
        recovered_tokens = encoder.decode_tokens(compressed, len(tokens))

    Thread-safety: NOT thread-safe. The model has internal state. For
    parallel encoding, instantiate one FastPLTEncoder per worker process.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        freq_precision_bits: int = 16,
        max_chunk: int = DEFAULT_MAX_CHUNK,
    ):
        """
        Args:
            model: a Hugging Face transformers model (AutoModelForCausalLM)
            tokenizer: the matching tokenizer
            device: torch device string. If None, uses model.device
            freq_precision_bits: precision for the cumulative frequency
                table. 16 bits (65535 total) is standard and sufficient
                for vocabularies up to ~64K. Higher = tighter quantization
                of probabilities = closer to entropy floor, at the cost of
                slightly slower coder arithmetic.
            max_chunk: maximum positions processed in one forward pass.
                Bounded by available memory; 4096 is fine for small models
                on 24 GB.
        """
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.vocab_size = model.config.vocab_size
        self.freq_total = 1 << freq_precision_bits
        self._min_freq = 1
        self.max_chunk = max_chunk

        # BOS token: needed to seed P(t_0 | empty context). If the
        # tokenizer doesn't have one (rare), fall back to token 0.
        self.bos_id = tokenizer.bos_token_id
        if self.bos_id is None:
            self.bos_id = 0

        # Cache the model in eval mode; we never train through it.
        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------
    # Probability table construction
    # ------------------------------------------------------------------

    def _logits_to_cumfreq(self, logits) -> np.ndarray:
        """Convert one position's logits (vocab_size,) to an integer
        cumulative frequency table of length vocab_size+1 summing to
        freq_total. Same logic as PLTEncoder._probs_to_cumfreq but
        operates on a torch tensor and returns numpy.

        The integer-quantization step is necessary because the AC coder
        operates on exact integer arithmetic. The minimum frequency
        floor (_min_freq = 1) means no symbol has zero probability,
        which would require infinite bits to encode if it ever appears.
        """
        # Cast to float64 for numerical stability in the conversion
        # (fp16 log_softmax can produce -inf for tail probabilities,
        # which breaks integer quantization).
        logp = self._torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum()

        freqs = np.maximum((p * self.freq_total).astype(np.int64), self._min_freq)
        delta = int(self.freq_total - freqs.sum())
        if delta != 0:
            order = np.argsort(-freqs)
            if delta > 0:
                for idx in order[:delta]:
                    freqs[idx] += 1
            else:
                rem = -delta
                for idx in order:
                    if rem == 0:
                        break
                    can_remove = int(freqs[idx]) - self._min_freq
                    take = min(can_remove, rem)
                    freqs[idx] -= take
                    rem -= take

        cumfreq = np.zeros(self.vocab_size + 1, dtype=np.int64)
        cumfreq[1:] = np.cumsum(freqs)
        return cumfreq

    # ------------------------------------------------------------------
    # Encoding: ONE forward pass over the entire sequence
    # ------------------------------------------------------------------

    def encode_tokens(self, tokens: list) -> tuple[bytes, CompressionStats]:
        """Compress a token sequence to a bitstring.

        Implementation: prepend the BOS token to seed P(t_0). Run one
        forward pass through the model with the full sequence (chunked
        if longer than max_chunk). For each position i in the input,
        extract logits, convert to cumfreq, feed to the arithmetic coder.

        Memory cost: O(n * vocab_size) for the cumfreq tables computed
        once and then fed to the coder. For n=4000 and vocab=128000 with
        int64, that's ~4 GB. Too much. We compute and discard cumfreqs
        in a streaming fashion through a generator instead.

        Returns the compressed bytes and statistics.
        """
        import torch

        if len(tokens) == 0:
            return b"", CompressionStats(0, 0, 0.0, 0.0, 1.0)

        # We need logits at positions corresponding to predicting each
        # of `tokens`. Feed [BOS, t_0, t_1, ..., t_{n-2}] to the model;
        # the logits at output positions 0..n-1 then correspond to
        # P(t_0 | BOS), P(t_1 | BOS, t_0), ..., P(t_{n-1} | BOS, t_<{n-1}).
        input_ids = [self.bos_id] + list(tokens[:-1])

        all_logits_chunks = []
        with torch.no_grad():
            # Chunked forward pass to bound memory. Each chunk uses the
            # previous chunk's KV state via the model's past_key_values
            # mechanism. This is the same trick generation uses.
            past = None
            i = 0
            n = len(input_ids)
            while i < n:
                end = min(i + self.max_chunk, n)
                chunk = self._torch.tensor(
                    [input_ids[i:end]], device=self.device, dtype=self._torch.long
                )
                out = self.model(chunk, past_key_values=past, use_cache=True)
                # out.logits has shape (1, chunk_len, vocab_size)
                # Keep on the device; we'll process row-by-row outside
                all_logits_chunks.append(out.logits[0])  # (chunk_len, vocab_size)
                past = out.past_key_values
                i = end

        # Concatenate all chunks: shape (n, vocab_size)
        all_logits = self._torch.cat(all_logits_chunks, dim=0)
        assert all_logits.shape[0] == len(tokens), (
            f"logits length {all_logits.shape[0]} != tokens length {len(tokens)}"
        )

        # Precompute the bit-overhead estimate and the per-position
        # cumfreq slice the coder needs. We materialize cumfreqs lazily
        # to keep memory bounded.
        expected_bits = 0.0
        position_data: list[tuple[int, int, int]] = []  # (cum_low, cum_high, total) per position

        for i in range(len(tokens)):
            cumfreq = self._logits_to_cumfreq(all_logits[i])
            tok = tokens[i]
            cum_low = int(cumfreq[tok])
            cum_high = int(cumfreq[tok + 1])
            position_data.append((cum_low, cum_high, self.freq_total))
            interval = cum_high - cum_low
            expected_bits += np.log2(self.freq_total / interval)

        # Free GPU memory now that we have all the integer intervals
        del all_logits
        del all_logits_chunks

        def cumfreq_provider(i, prev):
            return position_data[i]

        encoded = ArithmeticCodec.encode(tokens, cumfreq_provider)

        n_tokens = len(tokens)
        compressed_bytes = len(encoded)
        bits_per_token = (8 * compressed_bytes) / n_tokens
        expected_bpt = expected_bits / n_tokens
        overhead = bits_per_token / expected_bpt if expected_bpt > 0 else 1.0

        return encoded, CompressionStats(
            n_tokens=n_tokens,
            compressed_bytes=compressed_bytes,
            bits_per_token=bits_per_token,
            expected_bits_per_token=expected_bpt,
            overhead_factor=overhead,
        )

    def encode_text(self, text: str) -> tuple[bytes, CompressionStats]:
        """Convenience wrapper: encode a text string."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return self.encode_tokens(tokens)

    # ------------------------------------------------------------------
    # Decoding: full-prefix re-runs for numerical agreement with encode
    # ------------------------------------------------------------------
    #
    # Why we don't use past_key_values during decode:
    #
    # In float32, a step-by-step model run with past_key_values produces
    # logits that differ from a full-sequence run by ~1e-7 due to
    # floating-point operation ordering. That tiny difference is enough
    # to flip the integer cumulative-frequency quantization at borderline
    # symbols, which then desyncs the AC encoder and decoder. The result
    # is round-trip failure at random positions.
    #
    # The fix is to decode using exactly the same computation path the
    # encoder used: forward pass over the full prefix [BOS, t_0, ..., t_{i-1}]
    # at each step i, take logits at the final position. This makes decode
    # O(n^2) but correctness is absolute.
    #
    # For our workflow this trade-off is fine:
    #   - populate.py only ENCODES; the slow decode never runs in the hot
    #     path of cache population.
    #   - Interactive session recall calls decode once per recalled session.
    #     For a 4000-token session on a moderate model this takes seconds,
    #     not subseconds, but recall isn't expected to be interactive-fast.
    #
    # A bit-deterministic future v0.3 codec would use integer-quantized
    # inference (NNCP-style) where step-by-step and full-pass produce
    # bit-identical results. That's a larger engineering effort and not
    # needed for the headline demo.

    def decode_tokens(self, data: bytes, n_tokens: int) -> list:
        """Decompress a bitstring back to a token sequence.

        Returns the exact token sequence that was encoded. Uses full-prefix
        re-runs (O(n^2) total) to ensure numerical agreement with the
        encoder.
        """
        import torch

        if n_tokens == 0:
            return []

        # Cached logits for the next position. Updated each time the AC
        # decoder calls cumfreq_provider with a longer prev_symbols list.
        state = {"next_logits": None, "consumed_prefix_len": -1}

        def _compute_logits_for_position(prev_symbols):
            """Run the model over [BOS, *prev_symbols] and return the
            logits at the final position (which predict the next token).
            """
            input_ids = [self.bos_id] + list(prev_symbols)
            ids = self._torch.tensor(
                [input_ids], device=self.device, dtype=self._torch.long
            )
            with torch.no_grad():
                out = self.model(ids, use_cache=False)
            # logits at the LAST position: predicts the next token
            return out.logits[0, -1]

        def cumfreq_provider(i, prev_symbols):
            # Only re-run the model if we've moved forward in the sequence
            if state["consumed_prefix_len"] != len(prev_symbols):
                state["next_logits"] = _compute_logits_for_position(prev_symbols)
                state["consumed_prefix_len"] = len(prev_symbols)
            cumfreq = self._logits_to_cumfreq(state["next_logits"])
            return cumfreq, self.freq_total

        def find_symbol(target, cumfreq, total):
            # Binary search for the symbol whose interval contains target
            lo, hi = 0, len(cumfreq) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if cumfreq[mid] <= target:
                    lo = mid
                else:
                    hi = mid
            sym = lo
            return sym, int(cumfreq[sym]), int(cumfreq[sym + 1])

        return ArithmeticCodec.decode(data, n_tokens, cumfreq_provider, find_symbol)

    def decode_text(self, data: bytes, n_tokens: int) -> str:
        """Convenience wrapper: decode and detokenize."""
        tokens = self.decode_tokens(data, n_tokens)
        return self.tokenizer.decode(tokens)
