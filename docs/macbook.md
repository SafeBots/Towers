# Running Towers of Segments on a MacBook

The paper makes a specific claim about commodity Apple hardware: a 64 GB
MacBook with a 4 TB SSD can hold 2.7 billion compressed sessions and serve
multi-tenant LLM workloads at scale. This document is the practical guide
for actually running the demos on Apple Silicon.

## What runs where on Apple Silicon

llama.cpp's Metal backend is the default on macOS and runs the model's
matrix math on the M-series GPU via MPS (Metal Performance Shaders). The
unified memory architecture means model weights and KV cache live in one
memory pool that both the CPU and GPU can access without copies.

What llama.cpp does NOT use is the Apple Neural Engine. The ANE is
fixed-function hardware for specific neural-network shapes (mostly small,
mostly inference-optimized) and is not a general-purpose matmul
accelerator. The GPU is.

For Hugging Face transformers (what `example 03` uses to encode tokens
against a small LM), the corresponding backend is PyTorch MPS. This has
been stable since PyTorch 2.1 for the operations a small transformer
needs. Larger / older models occasionally hit a missing-op error; the
workaround is to set `PYTORCH_ENABLE_MPS_FALLBACK=1` which transparently
runs missing ops on CPU.

## RAM tiers and what each can do

The repo's demos scale with RAM. Here's what each tier can run:

### 16 GB MacBook (M1 / M2 / M3 base)

Comfortable:
- `examples/01_basic_hierarchy.py` (no model required)
- `examples/02_quantized_bases.py` (no model required)
- `examples/03_ac_compress.py --preset tiny` (SmolLM2-135M, ~270 MB)
- Multi-tenant demo against `llama-server` running a 1B-class model at Q4
  (e.g., Llama-3.2-1B-Instruct-Q4: ~700 MB on disk, ~1.5 GB resident)

Possible but tight:
- `examples/03_ac_compress.py --preset standard` (Llama-3.2-1B FP16, ~2.5 GB)

### 24-32 GB MacBook (M1 Pro / M2 Pro / M3 Pro)

Add:
- `examples/03_ac_compress.py --preset best` (Qwen2.5-3B FP16, ~6 GB)
- Multi-tenant demo against llama-server running a 7-8B-class model at Q4
  (e.g., Llama-3-8B-Instruct-Q4: ~4.5 GB on disk)

### 64 GB MacBook (M3 Max / M4 Max)

Add:
- Multi-tenant demo against a 70B-class model at Q4 (~40 GB resident).
  This is what the paper's headline claim is about: a 70B model with
  Towers compression on a single laptop.

### 128 GB MacBook (M3 Ultra)

Add:
- 70B at Q6 / Q8 quantization (better quality, more RAM)
- Multiple 8B models loaded simultaneously
- Headroom for the cache trie's hot pool to hold many warm sessions

## Recommended model choices

### For the small probability model (M_tilde in the paper, paper §10.5)

The arithmetic coder needs a callable that gives next-token log-probabilities.
The smaller and better-trained this model is, the cheaper encoding/decoding
is and the higher the compression ratio.

Three reasonable choices, depending on your RAM and the kind of text you
plan to compress:

**SmolLM2-135M (Hugging Face TB).** ~270 MB on disk. Trained on a
curated mix; surprisingly good for its size on English text. The default
in `example 03 --preset tiny`. Use this for the quick demo; the
bits/token number will be in the 4-6 range on conversational English.

**Llama-3.2-1B-Instruct (Meta).** ~2.5 GB FP16, ~700 MB at Q4. Better
language model, gives lower bits/token (typically 3-4 on the same text).
The default in `--preset standard`. Note: requires Hugging Face auth
(accept the license at huggingface.co/meta-llama/Llama-3.2-1B).

**Qwen2.5-3B (Alibaba).** ~6 GB FP16, ~2 GB at Q4. Excellent per-parameter
quality across natural language and code. No HF auth required. Best
compression of the three; slower because the model is bigger. The
default in `--preset best`.

For research validation: SmolLM2 is fine and proves the mechanism works.
For numbers you'd put in a paper or blog post: Llama-3.2-1B or Qwen2.5-3B
will give more impressive figures.

### For the target model (the one Towers serves at inference time)

This is the 70B-class model in the paper's main scenario. On a 64 GB+
MacBook, the realistic targets are:

**Llama-3.3-70B-Instruct at Q4** is the standard. ~40 GB on disk, ~42 GB
resident with KV cache headroom. Inference at maybe 6-10 tok/s on M3
Max, faster on M4 Max. GGUF builds are widely available.

**Qwen2.5-72B-Instruct at Q4** is comparable size, often better on code
and reasoning, no auth required.

**DeepSeek-V3 / R1** are 671B-parameter MoE models. With only ~37B active
per token they can theoretically fit at heavy quantization but in practice
need server-class memory; not a MacBook fit.

## Setting up llama.cpp on a MacBook (build with Metal)

```bash
# Clone and build
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Metal is the default on macOS. The build will detect it automatically.
cmake -B build
cmake --build build --config Release -j

# Verify Metal is enabled (look for "ggml_metal_init" in startup logs)
./build/bin/llama-server --version
```

Download a model. Use a GGUF Q4 build from Hugging Face. For example:

```bash
# Llama 3.2 1B (small, fast, good for first test)
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir ./models

# Llama 3.3 70B (the real target, needs 64+ GB)
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  Llama-3.3-70B-Instruct-Q4_K_M.gguf --local-dir ./models
```

Start llama-server with the slot API enabled:

```bash
mkdir -p /tmp/towers_slots

# For small model on any MacBook
./build/bin/llama-server \
    -m ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -c 8192 \
    --parallel 4 \
    --slots \
    --slot-save-path /tmp/towers_slots

# For 70B on a 64+ GB MacBook
./build/bin/llama-server \
    -m ./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
    -c 8192 \
    --parallel 2 \
    -ngl 999 \
    --slots \
    --slot-save-path /tmp/towers_slots
```

Key flags:
- `-c 8192` is the context window per slot. Towers compose multiple
  segments; make sure this is large enough for your platform + community
  + bot + dynamic prompt sizes summed.
- `--parallel N` is the number of concurrent slots. Each tower assembly
  uses one slot. For multi-tenant scenarios you want this set to the
  expected concurrent user count, bounded by RAM.
- `-ngl 999` (n-gpu-layers) offloads all layers to Metal. Critical for
  speed on Apple Silicon; the default on macOS is already this but
  explicit is safer.
- `--slots` enables the `/slots/{id}?action=save|restore` API that the
  paper's tower assembly relies on.
- `--slot-save-path` is the directory where slot files are persisted.
  This is the "disk tier" of the paper's cache hierarchy.

Then in another terminal:

```bash
python examples/04_macbook_demo.py
```

## Performance expectations on Apple Silicon

These are ballpark numbers. Your mileage will vary.

| Operation | M3 Pro 18 GB | M3 Max 64 GB | M4 Max 128 GB |
|---|---|---|---|
| AC encode, 256 tokens, SmolLM2 | ~10-20 s | ~5-10 s | ~3-7 s |
| AC encode, 256 tokens, Llama-3.2-1B | ~60-90 s | ~30-50 s | ~20-35 s |
| llama-server prefill, 4000 tok, 8B Q4 | ~3-5 s | ~2-3 s | ~1-2 s |
| llama-server prefill, 4000 tok, 70B Q4 | n/a (OOM) | ~30-60 s | ~20-40 s |
| Slot save, 4000 tok | ~50-150 ms | ~30-100 ms | ~20-80 ms |
| Slot restore, 4000 tok | ~30-100 ms | ~20-80 ms | ~15-60 ms |

The AC encode/decode numbers are dominated by running the probability
model forward at every position (O(n^2) in the v0.1 implementation).
A production version would use proper KV-caching during encoding to
make this O(n); we accept the higher cost to keep the v0.1 codec
small and easy to read.

The llama-server prefill numbers are real Metal-accelerated inference.
The 4000-token figure matches the paper's session-length assumption.
At the 70B level, prefill is the dominant cost of cold-tier thaw,
which is exactly what the paper's §10.5 says.

## Known issues and workarounds

**`OSError: cannot open shared object file` when loading a model.** On
some Python distributions, missing `libomp` causes this. Install via
`brew install libomp` then re-run.

**MPS missing-op errors during encoding.** Some Hugging Face models use
ops not yet supported on MPS (typically obscure attention variants).
Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to transparently fall back to CPU
for those ops, or use `--device cpu` to run everything on CPU.

**llama-server slot save failing with code 400.** Make sure llama-server
was started with both `--slots` and `--slot-save-path`. The save endpoint
is disabled by default for security reasons.

**Very slow on first run.** First-time tokenizer downloads from Hugging
Face are ~50-500 MB depending on the model. Subsequent runs use the
cached copy in `~/.cache/huggingface/`.

## When NOT to use Towers on a MacBook

Towers compresses the *cache state*. The compute cost of running the
target model still scales with model size. For real-time generation
on a 70B model, even a fast MacBook will give you ~6-10 tok/s, which
is slow if you need to generate hundreds of tokens. The architecture
makes storage cheap; it does not make inference fast.

If your bottleneck is generation throughput rather than session
storage cost, Towers buys you nothing. Use a smaller model, or
deploy to GPU infrastructure with high memory bandwidth.

Towers' value proposition is specifically for multi-tenant deployments
where many sessions sit idle most of the time, are occasionally
reactivated, and the dominant cost is keeping all that session state
available somewhere. The MacBook story is "your laptop can be the
storage tier for millions of sessions", not "your laptop is faster
than an A100 at inference."
