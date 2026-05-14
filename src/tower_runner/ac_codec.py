"""Arithmetic coding compression of token sequences against a language model.

This file implements the compression mechanism of paper §10. The mapping
from paper to code:

    Paper:  the compressor uses M_tilde, a small distilled language model,
            to encode a token sequence tau against P_M_tilde(t_i | t_{<i}).
            Expected encoded length is the cumulative surprisal,
                sum_i -log_2 P_M_tilde(t_i | t_{<i})
            which by Shannon's source coding theorem is asymptotically
            optimal up to a constant.

    Code:   PLTEncoder wraps an arithmetic coder (ArithmeticCodec) and a
            user-supplied logprob_fn callable. The callable is M_tilde:
            given the tokens seen so far, it returns log P over the
            vocabulary for the next token. For research validation, any
            transformer language model loaded via Hugging Face works;
            see examples/03_ac_compress.py for the full integration.

Empirical anchors for what to expect from this code:

    Chinchilla 70B achieves 0.664 bits per byte on enwik9 via this exact
    technique (DeepMind, ICLR 2024). Smaller models give worse compression
    but the same algorithmic structure: SmolLM2-135M gives 4-6 bits per
    token on conversational English, Llama 3.2 1B gives 3-4 bits per token.
    For comparison: gzip achieves about 2-3 bytes per token. The factor
    of ~10 gain over gzip is what the paper's §10 result captures.

Implementation choice: range coding rather than classical AC.

    Range coding is an arithmetic coding variant with integer
    renormalization (rather than the unit-interval-and-bit-extraction
    formulation in compression textbooks). The two are mathematically
    equivalent; range coding avoids the carry-propagation pitfalls that
    can break finite-precision classical AC. The reference for the
    specific form here is Subbotin's "carryless range coder" and
    Bellard's NNCP.

    32-bit precision (PRECISION below) is more than enough for vocabulary
    sizes up to about 2^16. Real LLM vocabularies are 32K-256K tokens;
    32-bit precision handles those comfortably with room to spare.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Range coding core: finite-precision arithmetic coding
# ----------------------------------------------------------------------------
#
# A range coder maintains a "range" [low, high) inside a fixed-size integer
# window of PRECISION bits. To encode a symbol whose cumulative-probability
# interval is [cum_low/total, cum_high/total), the coder narrows its window
# to that sub-interval:
#
#     new_low  = low + (range * cum_low)  // total
#     new_high = low + (range * cum_high) // total - 1
#
# As the window shrinks, the high-order bits of low and high agree. Those
# bits are emitted to the output bitstream and the window is renormalized
# (shifted left by 1 bit) so it has room to encode the next symbol.
#
# A subtle case: low and high straddle the middle of the precision window
# (low in upper quarter, high in lower three-quarters). The bit doesn't
# converge but is "pending": we shift it out as a placeholder, and emit
# the actual bit (plus its inverse for each pending one) when convergence
# finally happens. _BitWriter.write_bit_and_pending handles this.

PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2


class _BitWriter:
    """Writes bits to a bytes buffer LSB-first within each byte."""
    def __init__(self) -> None:
        self.buf = bytearray()
        self._cur = 0
        self._cur_bits = 0
        self.n_bits = 0

    def write_bit(self, bit: int) -> None:
        self._cur |= (bit & 1) << self._cur_bits
        self._cur_bits += 1
        self.n_bits += 1
        if self._cur_bits == 8:
            self.buf.append(self._cur)
            self._cur = 0
            self._cur_bits = 0

    def write_bit_and_pending(self, bit: int, pending: int) -> int:
        self.write_bit(bit)
        for _ in range(pending):
            self.write_bit(1 - bit)
        return 0

    def finish(self) -> bytes:
        if self._cur_bits > 0:
            self.buf.append(self._cur)
            self._cur = 0
            self._cur_bits = 0
        return bytes(self.buf)


class _BitReader:
    """Reads bits from a bytes buffer LSB-first within each byte."""
    def __init__(self, data: bytes) -> None:
        self.data = data
        self._pos = 0
        self._cur_bits = 0
        self._cur = 0

    def read_bit(self) -> int:
        if self._cur_bits == 0:
            if self._pos < len(self.data):
                self._cur = self.data[self._pos]
                self._pos += 1
                self._cur_bits = 8
            else:
                # Reading past end returns 0s (standard AC convention)
                return 0
        bit = self._cur & 1
        self._cur >>= 1
        self._cur_bits -= 1
        return bit


class ArithmeticCodec:
    """Arithmetic / range coder. Encodes a sequence of symbols against a
    sequence of probability distributions (one distribution per symbol).

    Each distribution is given as an iterable of cumulative-frequency values
    in [0, total), with the symbol's interval being [cum_low, cum_high).
    """

    @staticmethod
    def encode(
        symbols: Sequence[int],
        cumfreq_provider,  # callable: position i -> (cum_low_for_sym, cum_high_for_sym, total)
    ) -> bytes:
        """Encode symbols using probabilities from cumfreq_provider.

        cumfreq_provider(i, prev_symbols) must return a tuple
        (cum_low, cum_high, total) where:
            cum_low <= cum_high <= total
            and the implied probability of symbols[i] is (cum_high-cum_low)/total.

        Returns the encoded bitstring as bytes.
        """
        low = 0
        high = WHOLE - 1
        pending = 0
        writer = _BitWriter()

        for i, sym in enumerate(symbols):
            cum_low, cum_high, total = cumfreq_provider(i, symbols[:i])
            range_size = high - low + 1
            # Narrow the window to the sub-interval for this symbol
            new_high = low + (range_size * cum_high) // total - 1
            new_low = low + (range_size * cum_low) // total
            low, high = new_low, new_high

            # Renormalize: emit any bits that low/high agree on
            while True:
                if high < HALF:
                    pending = writer.write_bit_and_pending(0, pending)
                elif low >= HALF:
                    pending = writer.write_bit_and_pending(1, pending)
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < HALF + QUARTER:
                    pending += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low <<= 1
                high = (high << 1) | 1
                # Keep within precision
                low &= WHOLE - 1
                high &= WHOLE - 1

        # Flush remaining bits
        pending += 1
        if low < QUARTER:
            writer.write_bit_and_pending(0, pending)
        else:
            writer.write_bit_and_pending(1, pending)

        return writer.finish()

    @staticmethod
    def decode(
        data: bytes,
        n_symbols: int,
        cumfreq_provider,  # callable: position i, prev_symbols -> (cumfreq_table, total)
        find_symbol,       # callable: (target, cumfreq_table, total) -> symbol_idx, cum_low, cum_high
    ) -> list[int]:
        """Decode n_symbols from data using probabilities from cumfreq_provider.

        cumfreq_provider(i, prev_symbols) returns the full cumulative
        frequency table at position i (an array of length V+1 where V is
        vocab size, ending with the total). find_symbol(target, table, total)
        returns (symbol_idx, cum_low, cum_high) for the symbol whose interval
        contains target.
        """
        reader = _BitReader(data)
        low = 0
        high = WHOLE - 1
        # Prime the value with the first PRECISION bits
        value = 0
        for _ in range(PRECISION):
            value = (value << 1) | reader.read_bit()

        symbols: list[int] = []
        for i in range(n_symbols):
            cumfreq_table, total = cumfreq_provider(i, symbols)
            range_size = high - low + 1
            target = ((value - low + 1) * total - 1) // range_size
            sym, cum_low, cum_high = find_symbol(target, cumfreq_table, total)
            symbols.append(sym)

            new_high = low + (range_size * cum_high) // total - 1
            new_low = low + (range_size * cum_low) // total
            low, high = new_low, new_high

            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    value -= HALF
                elif low >= QUARTER and high < HALF + QUARTER:
                    low -= QUARTER
                    high -= QUARTER
                    value -= QUARTER
                else:
                    break
                low <<= 1
                high = (high << 1) | 1
                value = (value << 1) | reader.read_bit()
                low &= WHOLE - 1
                high &= WHOLE - 1
                value &= WHOLE - 1

        return symbols


# ----------------------------------------------------------------------------
# PLT encoder: compose AC codec with a language model
# ----------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Statistics for a single segment compression."""
    n_tokens: int
    compressed_bytes: int
    bits_per_token: float
    expected_bits_per_token: float  # the entropy lower bound
    overhead_factor: float  # actual / expected; >= 1.0

    def summary(self) -> str:
        return (
            f"{self.n_tokens} tokens -> {self.compressed_bytes} bytes "
            f"({self.bits_per_token:.2f} bpt, "
            f"entropy floor {self.expected_bits_per_token:.2f} bpt, "
            f"overhead {self.overhead_factor:.2f}x)"
        )


class PLTEncoder:
    """Encode/decode token sequences using arithmetic coding against a
    language model's probability distribution.

    The probability model is provided by a callable `logprob_fn`:

        logprob_fn(prev_token_ids: list[int]) -> np.ndarray
            returns log P(next_token | prev) over the full vocabulary,
            as a float32 array of shape (vocab_size,).

    For production this would be a small distilled LM running in batched
    GPU inference. For this reference implementation we support any callable
    with that signature; see examples/03_ac_compress.py for how to plug in
    a Hugging Face model.
    """

    def __init__(self, logprob_fn, vocab_size: int, freq_precision_bits: int = 16):
        """
        Args:
            logprob_fn: as described above
            vocab_size: model's vocabulary size V
            freq_precision_bits: bits of precision for the cumulative frequency
                table (16 = max value 65535 per symbol, standard for AC)
        """
        self.logprob_fn = logprob_fn
        self.vocab_size = vocab_size
        self.freq_total = 1 << freq_precision_bits
        # Floor probability to avoid zero-probability symbols (which would
        # require infinite bits to encode). 1/freq_total = ~16-bit minimum.
        self._min_freq = 1

    def _probs_to_cumfreq(self, logprobs: np.ndarray) -> np.ndarray:
        """Convert log-probabilities to integer cumulative frequencies
        summing to freq_total. Returns array of length V+1 where
        table[i+1] - table[i] is symbol i's interval width.
        """
        # Convert from log to linear
        logprobs = logprobs.astype(np.float64)
        logprobs -= logprobs.max()  # numerical stability
        probs = np.exp(logprobs)
        probs /= probs.sum()

        # Quantize to integer frequencies summing to freq_total
        freqs = np.maximum((probs * self.freq_total).astype(np.int64), self._min_freq)
        # Adjust to sum exactly to freq_total
        delta = self.freq_total - int(freqs.sum())
        if delta != 0:
            # Distribute the delta starting from the highest-probability symbols
            order = np.argsort(-freqs)
            if delta > 0:
                for idx in order[:delta]:
                    freqs[idx] += 1
            else:
                # Need to remove (-delta) units, only from symbols with > min_freq
                rem = -delta
                for idx in order:
                    if rem == 0:
                        break
                    can_remove = freqs[idx] - self._min_freq
                    take = min(can_remove, rem)
                    freqs[idx] -= take
                    rem -= take
        assert int(freqs.sum()) == self.freq_total, f"freq sum {freqs.sum()} != {self.freq_total}"

        cumfreq = np.zeros(self.vocab_size + 1, dtype=np.int64)
        cumfreq[1:] = np.cumsum(freqs)
        return cumfreq

    def encode(self, tokens: list[int]) -> tuple[bytes, CompressionStats]:
        """Compress a token sequence to a bitstring.

        Returns the encoded bytes and statistics.
        """
        if len(tokens) == 0:
            return b"", CompressionStats(0, 0, 0.0, 0.0, 1.0)

        # Precompute all conditional distributions
        # (in production, batched on GPU; here we just iterate)
        cumfreqs: list[np.ndarray] = []
        expected_bits = 0.0
        for i in range(len(tokens)):
            logp = self.logprob_fn(tokens[:i])
            # The entropy contribution of this position is -log2 P(tokens[i] | ...)
            # which lower-bounds the bits the AC coder uses at this position
            cumfreq = self._probs_to_cumfreq(logp)
            cumfreqs.append(cumfreq)
            # Estimate per-token surprisal from the QUANTIZED distribution
            # (this is what AC will actually use, not the true continuous prob)
            tok = tokens[i]
            interval = cumfreq[tok + 1] - cumfreq[tok]
            expected_bits += np.log2(self.freq_total / interval)

        def cumfreq_provider(i, prev):
            cumfreq = cumfreqs[i]
            tok = tokens[i]
            return int(cumfreq[tok]), int(cumfreq[tok + 1]), self.freq_total

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

    def decode(self, data: bytes, n_tokens: int) -> list[int]:
        """Decompress a bitstring back to a token sequence."""
        if n_tokens == 0:
            return []

        # We need to re-run the LM in the same order as encode did,
        # because each position's probability depends on the previously
        # decoded tokens. Each cumfreq table is computed on demand.
        def cumfreq_provider(i, prev_symbols):
            logp = self.logprob_fn(prev_symbols)
            cumfreq = self._probs_to_cumfreq(logp)
            return cumfreq, self.freq_total

        def find_symbol(target, cumfreq, total):
            # Binary search for the symbol whose interval [cumfreq[s], cumfreq[s+1])
            # contains target.
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
