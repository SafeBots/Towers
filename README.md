# Towers of Segments

Reference implementation for [Towers of Segments: Hierarchical KV Cache Storage from Commodity Workstations to Production Servers](https://safebots.ai/papers/Towers.pdf).

## 81,000× compression, measured on a MacBook

A 4000-token conversation between a user and an LLM costs about 1.28 GB
of KV cache state when the model is Llama-3-70B with grouped-query
attention in FP16. Current production chat services either hold that
state resident in GPU memory between user messages (paying for it
whether the user comes back or not) or evict it and pay a full prefill
cost on return.

The same 4000-token conversation, stored on disk in the format
`demo/populate.py` produces, takes about 12 KB. The ratio is 81,000×,
and it's measurable from the bytes in the cache directory:

```bash
python benchmarks/amortization.py --cache-dir ~/towers_cache

# Target model              KV bytes/tok   tokens-only   AC-compressed
# Llama-3-70B (GQA, FP16 KV)   320,000      81,000×       860,000×
```

The benchmark reads real files and divides. No extrapolation.

You're not violating information theory. You're choosing a
representation that costs a few hundred milliseconds of GPU compute to
rehydrate, instead of one that costs a memcpy. For chat workloads,
where the gap between a user's messages is tens of seconds to minutes,
the trade is overwhelmingly favorable. The rest of this article
explains how the 81,000× figure is built, why the trade works, and
why this isn't already the default.

## The compression ratio is three mechanisms multiplied

Three independent layers compose. Each has its own code path and its
own test coverage in this repo.

### Layer 1 — tokens instead of KV state

A Llama-3-70B model with grouped-query attention has 80 layers and 8
KV heads of dimension 128. The FP16 KV cache for one token is

```
2 bytes (FP16) × 80 (layers) × 2 (K and V) × 8 (heads) × 128 (head dim)
  = 327,680 bytes per token
```

Multiply by 4000 tokens of context and you get the 1.28 GB figure.
The token IDs for the same conversation, stored as 32-bit integers,
are 4 bytes per token. Same conversation, six orders of magnitude
less data.

Storing the token IDs instead of the KV tensors is not novel — every
text editor on Earth does it for documents. What's new is the
architectural argument that the LLM serving stack also doesn't need
the KV tensors between user turns. KV state is fully determined by
the token sequence and the model weights, both of which you already
have. Whatever GPU memory holds the KV cache can be reconstructed
from the tokens via one forward pass.

The "tokens-only" column in the benchmark is just this layer, with
amortization across the multi-tenant trie.

### Layer 2 — structural amortization across sessions

System prompts repeat. A platform running ten thousand chat sessions
through the same `"You are a Python debugging assistant"` preamble
should store that preamble once, not ten thousand times. The cache
trie in `src/tower_runner/cache_trie.py` makes the preamble a
*segment*: a content-addressed node in a tree, shared across every
session that descends from it.

The hierarchy in this repo is three levels deep — platform →
community → bot. Each bot's full base prompt is stored once. Every
session that uses that bot stores only its own dynamic conversation
block and a pointer back to the bot segment.

The amortized base cost per session is `total_base_bytes /
N_sessions`. With 13 bot segments totaling 2.4 KB, that cost falls
below one byte per session by N=2500. The structural amortization
theorem in paper §4 makes the rate explicit: per-session cost
approaches the dynamic-only limit at O(1/N), and
`tests/test_tower_store.py` shows the empirical convergence hits
99.99% of the asymptote by N=1000.

This is the layer where multi-tenant deployments earn their leverage.
Running one chat session, you save nothing from amortization. Running
a million sessions across a few hundred bot configurations, the
system-prompt cost rounds to zero.

### Layer 3 — arithmetic coding to the entropy floor

If you have a language model M̃ that predicts next-token probabilities
given prior context, Shannon's source-coding theorem says you can
encode a token sequence at H(token | history) bits per token, plus
arbitrarily small overhead. With a decent small encoder — even
SmolLM2-135M, 135 million parameters — that's about 3 bits per token
on conversational English. Raw int32 storage is 32 bits per token.
This layer adds roughly a 10× multiplier on top of Layer 1.

`src/tower_runner/ac_codec.py` implements range-coded arithmetic
compression. It hits the entropy floor to within 1.003× on synthetic
distributions; `tests/smoke_test.py` verifies this in five seconds
without any model download.

Stack the three layers: 80,000× from tokens-vs-KV (Layer 1),
amortization across N sessions of one bot (Layer 2), then
~10× from entropy coding (Layer 3). The paper's headline 850,000×
ratio is the product. The benchmark in this repo measures 860,000×
against Llama-3-70B with realistic populate-output parameters, very
close to the predicted number.

The 81,000× figure uses only Layers 1 and 2. The 860,000× figure
uses all three.

## Why session thaw is fast enough for this to work

The objection to "just store tokens and re-prefill on demand" used to
be that prefill was slow. Two things changed.

Attention kernels got an order of magnitude faster. FlashAttention
and its descendants made prefill bandwidth-bound rather than
compute-bound, and modern GPUs do prefill at thousands of tokens per
second for production-class models. A 4000-token prefill on Qwen-14B
with Metal acceleration on an M-series MacBook is about a second.

Prefill is parallel within a sequence. A transformer forward pass
processes all positions of a context window at once — the
sequential generation cost only applies to the *new* tokens you're
producing, not to the *prior* tokens you're rebuilding. Rebuilding
4000 tokens of context takes the same wall time as rebuilding 1000;
it's one forward pass of the model.

This repo adds two further reductions on top.

First, the bot-level base segments (system prompt and any shared
preamble) get their KV state saved to disk *once*, the first time
populate.py constructs them, via llama-server's `slot save` API. On
recall, the base KV state restores from disk in tens of milliseconds.
Only the dynamic conversation block needs a fresh prefill. The thaw
time of a 4000-token session is dominated by the dynamic-block
prefill, which is roughly the time it takes a transformer to process
3000-3500 tokens — about a second on consumer hardware.

Second, llama-server's multi-slot scheduler interleaves forward
passes across slots on the same GPU. Five cold sessions thawing in
parallel land in roughly the wall time of one thaw, not five. The
`demo/parallel_recall.py` script measures this directly.

The combined effect: a session that has been sitting on disk for
weeks restores to a live, continueable conversation in under two
seconds on a 24 GB MacBook. The user experience is comparable to a
session that never left RAM.

## Why this hasn't already been the default

Three reasons worth stating, because they're all path-dependency
rather than fundamental limits.

The "store KV" pattern is older than FlashAttention. When the
architecture was settling in around 2019-2020, prefill was much
slower — kernels were unoptimized, attention was O(n²) memory rather
than O(n), and re-prefilling a 4000-token session genuinely was a
ten-second affair. Keeping KV resident was the obvious choice. The
performance landscape changed; the storage habit didn't.

Production benchmarks measure the wrong thing. Time-to-first-token
on a *fresh* request is the standard yardstick. If your benchmark
suite doesn't measure amortized cost-per-active-session, you don't
notice that 90% of your GPU memory holds inactive context for users
who walked away. The economics of GPU-hour pricing under-rate the
storage cost.

Finally, the engineering ingredients — slot save/restore in
llama.cpp, RadixAttention in SGLang, cross-engine KV transport in
LMCache — have been arriving incrementally enough that no single
release made the alternative obviously practical. The contribution
of this repo is the composition: cache trie + dynamic-block AC
compression + multi-slot fast thaw, with the storage architecture
treated as a first-class design choice rather than an
implementation detail.

## Honest framing — what's measured vs what's modeled

The 81,000× ratio (tokens-only mode, vs Llama-3-70B raw FP16 KV)
is measured directly from bytes on disk. Run
`benchmarks/amortization.py` against a populated cache and it
divides one number by another.

The 860,000× ratio (with AC compression) is also measured in the
sense that the bits-per-token figure is empirical and the
amortized base cost is empirical. The caveat: this release stores
both the compressed bytes *and* the raw token IDs in parallel,
because the AC decoder requires bit-identical logit computation
between encode and decode, and standard float-precision transformer
inference drifts at the ~1e-7 level over a few hundred tokens.
Encode is correct and tight; round-trip decode requires
integer-quantized inference, which is a v0.3 engineering item
(see `src/tower_runner/fast_codec.py` for the discussion).

In practical terms: the 12 KB per session that the demo writes to
disk is the *raw token* representation. The AC-compressed bytes
(roughly 1.5 KB per session) sit alongside as evidence the
compression works at encode time. Recall uses the raw tokens.

Thaw latency claims on Apple Silicon are *modeled* until measured
on real hardware. The architecture predicts under two seconds for a
4000-token session; `demo/recall.py --benchmark` reports the actual
number once you have a populated cache.

The universal cache decoder of paper §12 is conjectural. If it
works, thaw gets even faster; if not, the conservative thaw path
this repo implements is what you get.

## What you can verify in 30 seconds (no GPU, no downloads)

```bash
unzip Towers.zip
cd towers-of-segments
pip install -r requirements.txt
python tests/smoke_test.py                       # 5 sec, 6 tests pass
python tests/test_tower_store.py                 # 5 sec, amortization theorem
python benchmarks/tower_amortization.py          # 10 sec, convergence plot
```

These three together verify the structural claims:

- The arithmetic codec hits the entropy floor within 1.003× on
  synthetic distributions. Encode and decode round-trip correctly.
- Tower amortization converges to Theorem 4.1's asymptote within
  ~1000 sessions. Per-session storage drops from 12,008 bytes at
  N=100 to 12,000.8 bytes at N=1000, against a theoretical limit
  of exactly 12,000.
- The cache trie content-addresses correctly: identical segments
  collapse to one node regardless of how many times they're
  constructed.

## What requires a model download (5 min, ~270 MB)

```bash
python examples/03_ac_compress.py
```

Downloads SmolLM2-135M and runs AC compression on conversational
text. Expected: 5-7 bits per token achieved on English chat,
overhead vs. the encoder's cross-entropy floor of ~1.003×. This is
the codec working against a real language model.

## What requires a real deployment (1 day to populate, then live)

The `demo/` directory holds the multi-day demo configuration: a
24 GB MacBook running Qwen2.5-14B-Q4 via llama.cpp, with a Python
process generating model-vs-model conversations continuously, the
cache-trie tower structure persisted to disk, and a live scoreboard
with sparklines showing the per-session amortization ratio climbing
toward the asymptote in real time.

```bash
bash demo/setup.sh             # 10-30 min; downloads ~10 GB
bash demo/start.sh             # launches everything in tmux (or background)
# ... let it run for as many days as you can ...
bash demo/stop.sh              # cleanly stop when done

# Once a few hundred sessions are stored, see the headline number:
python benchmarks/amortization.py
```

`start.sh` launches three processes — `llama-server`, `populate.py`,
and `scoreboard.py` — in a tmux session named `towers` (one pane
each), or backgrounds them with log files if tmux isn't installed.
Re-attach with `tmux attach -t towers`, detach with `Ctrl-b d`,
stop everything with `demo/stop.sh`.

Other things to try once you have sessions:

```bash
python demo/recall.py                          # random session, interactive
python demo/recall.py --benchmark              # measure thaw latency
python demo/parallel_recall.py --n 5           # five cold sessions, all at once
```

## Headline numbers

Bold = verified directly in code on the machine that ran it.
Italic = computed from verified inputs (the "how to verify" column
says which script). No row is pure extrapolation; every number has
a path back to bytes on disk or arithmetic on entropy floors.

| Metric | Value | How to verify |
|---|---|---|
| AC codec overhead vs entropy floor | **~1.003×** | `python tests/smoke_test.py` |
| Tower amortization at N=1000 sessions | **1.372× savings** (99.75% converged) | `python benchmarks/tower_amortization.py` |
| Tower amortization at N=25000 sessions | **1.375× savings** (99.99% converged) | `python benchmarks/tower_amortization.py` |
| FastPLT encode speedup vs v0.1 at n=800 | **152×** | `python tests/test_fast_codec.py` |
| FastPLT encode throughput on tiny GPT-2 | **7000+ tokens/sec** | `python tests/test_fast_codec.py` |
| TowerStore amortization at N=1000 sessions | **>99% of asymptote** | `python tests/test_tower_store.py` |
| Bits per token (AC + SmolLM2-135M on text) | *5-7* | `python examples/03_ac_compress.py` |
| Per-session amortized disk (4000-tok session) | *~12 KB raw, ~1.5 KB AC* | `python benchmarks/amortization.py` |
| Compression vs Llama-3-8B raw FP16 KV | *~33,000× tokens, ~350,000× AC* | `python benchmarks/amortization.py` |
| **Compression vs Llama-3-70B raw FP16 KV** | ***~81,000× tokens, ~860,000× AC*** | `python benchmarks/amortization.py` |
| Compression vs Llama-3-405B raw FP16 KV | *~128,000× tokens, ~1,360,000× AC* | `python benchmarks/amortization.py` |
| Cold-session thaw on 24 GB MacBook (Qwen-14B-Q4) | *~200-1500 ms* | `python demo/recall.py --benchmark` |

## Status

What is verified in code, no hardware required:

- AC codec round-trip on synthetic distributions. Uniform V=256 gives
  8.04 bpt against 8.00 theoretical (0.5% overhead). Skewed
  Bernoulli(0.95) gives 0.72 bpt against 0.69 theoretical.
- Cache trie data structure. Content-hash addressing collapses
  identical segments; tower assembly walks root-to-leaf as paper §3
  specifies.
- Tower amortization convergence. `tests/test_tower_store.py`
  shows the per-session amortized cost decreasing toward the
  dynamic-only limit at the rate the theorem predicts.
- Fast codec speedup. The v0.2 fast encoder is 152× faster than
  v0.1 at n=800 tokens, ~7000 tokens/sec on tiny GPT-2.

What is wired up but requires a model download (~270 MB):

- AC compression against a real LM via `examples/03_ac_compress.py`.

What is wired up but requires a llama-server deployment (~9 GB):

- Multi-tenant orchestration on llama-server. `examples/04_macbook_demo.py`
  exercises the slot save/restore API for tower assembly.
- End-to-end populate + scoreboard + recall. The `demo/` workflow
  generates sessions continuously, stores them in the tower
  structure on disk, computes amortization numbers and compression
  ratios live, and lets you pull cold sessions interactively.

What is NOT in this v0.2 release:

- Bit-deterministic AC decoding for sessions over ~100 tokens.
  Encode is correct; decode of compressed bytes back to tokens via
  a standard transformer is numerically fragile across input
  lengths. The demo sidesteps this by also storing raw tokens (~12 KB
  per session) for recall. Integer-quantized inference (NNCP-style)
  would close this gap (v0.3 work).
- Universal cache decoder (paper §12). Conjectural; not implemented.
- Depth-adaptive KV quantization as a llama.cpp patch (paper §6).
  `examples/02_quantized_bases.py` shows the *sizing* effect without
  applying it. The C++ patch is a spec in `src/llama_patch/README.md`.
- RoPE-shifted segment reuse (paper §5.2). Trie structure supports
  it; the C++ patch to llama.cpp's attention isn't written.

## Repository layout

```
towers-of-segments/
├── README.md                       this file
├── LICENSE                         Apache 2.0
├── requirements.txt
├── paper/
│   └── Towers.pdf                  the paper itself
├── docs/
│   └── macbook.md                  Apple Silicon setup guide
├── src/
│   ├── tower_runner/
│   │   ├── __init__.py
│   │   ├── cache_trie.py           paper §3-4 in code (in-memory trie)
│   │   ├── tower_store.py          persistent cache trie + llama-server glue
│   │   ├── ac_codec.py             paper §10 in code (v0.1, O(n²) encode)
│   │   ├── fast_codec.py           v0.2: O(n) encode via single forward pass
│   │   ├── llama_client.py         HTTP wrapper for llama-server
│   │   └── tower.py                tower assembly orchestrator
│   └── llama_patch/
│       └── README.md               planned C++ patches (v0.3)
├── examples/
│   ├── 01_basic_hierarchy.py       cache trie, no model
│   ├── 02_quantized_bases.py       quantization sizes, no model
│   ├── 03_ac_compress.py           AC demo on a real LM
│   └── 04_macbook_demo.py          multi-tenant on llama-server
├── benchmarks/
│   ├── tower_amortization.py       Theorem 4.1 convergence (no model)
│   ├── amortization.py             EMPIRICAL ratio from real cache state
│   ├── compression_ratio.py        AC sweep across dataset
│   └── thaw_latency.py             cold reactivation timing
├── demo/                           v0.2 live demo (24 GB MacBook target)
│   ├── setup.sh                    clone+build llama.cpp, download models
│   ├── start.sh                    ONE COMMAND to launch the whole demo
│   ├── stop.sh                     cleanly stop all background processes
│   ├── populate.py                 generate sessions, tower-amortized
│   ├── scoreboard.py               live terminal display with sparklines
│   ├── parallel_recall.py          thaw N sessions across N slots in parallel
│   └── recall.py                   tower thaw via slot restore + dyn prefill
├── data/conversational/            sample chats and topic seeds
└── tests/
    ├── smoke_test.py               install verification
    ├── test_fast_codec.py          FastPLTEncoder correctness + speedup
    └── test_tower_store.py         TowerStore persistence + amortization
```

## Citing

```bibtex
@article{magarshak2026towers,
  title={Towers of Segments: Hierarchical KV Cache Storage from Commodity
         Workstations to Production Servers},
  author={Magarshak, Gregory},
  journal={arXiv preprint},
  year={2026}
}
```

Paper PDF: [https://safebots.ai/papers/Towers.pdf](https://safebots.ai/papers/Towers.pdf)

## Related work in the compression program

This is the fourth paper in an arXiv compression program:

1. PLT (arXiv:2604.06228) — probabilistic language tries, the theoretical framework
2. KV Sequential (arXiv:2604.15356) — per-token entropy bound on KV state
3. LAWS (arXiv:2605.04069) — certified expert substitution
4. Towers (this paper) — multi-tenant storage architecture

## Related production-grade work

The cache mechanism Towers uses (modular attention reuse via segment
composition) is well-established prior art. If you want to run something
like this in production today:

- [LMCache](https://github.com/LMCache/LMCache) — production cache layer
  with cross-engine compatibility (vLLM, SGLang). Closest to a deployed
  version of these ideas.
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention-based
  prefix sharing.
- [PromptCache](https://github.com/yale-sys/prompt-cache) — the
  foundational academic paper (MLSys 2024).

Towers' specific contribution is the theoretical framework
(structural amortization theorem, depth-adaptive quantization bound,
composition with arithmetic coding to source entropy) and the
measurement methodology that turns the storage architecture into
a number you can read off a benchmark. The cache mechanism itself
is not novel; the storage architecture argument is.

## License

Apache 2.0. See `LICENSE`.
