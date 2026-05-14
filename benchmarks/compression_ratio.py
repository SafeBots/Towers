"""Benchmark: measure compression ratio of AC-against-LM on conversational data.

Reproduces the paper's headline ~3 bits/token / 850,000x compression number.

Usage:
    python benchmarks/compression_ratio.py
    python benchmarks/compression_ratio.py --model HuggingFaceTB/SmolLM2-360M
    python benchmarks/compression_ratio.py --dataset data/conversational/chats.jsonl

Outputs:
    - benchmarks/results/compression.json: full results
    - benchmarks/results/compression.png: bits/token distribution plot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tower_runner.ac_codec import PLTEncoder


def load_dataset(path: Path | None) -> list[str]:
    """Load conversational texts. If no path given, use built-in samples."""
    if path is None or not path.exists():
        return [
            "Hello, how can I help you today?",
            "I have a question about Python decorators. They confuse me.",
            "Decorators are functions that take a function and return a modified function. The @ syntax is just sugar for f = decorator(f).",
            "Could you give me a concrete example with a use case?",
            "Sure. The classic example is timing: @timed wraps a function to log how long it took. The decorator definition is `def timed(f): def wrapper(*a, **kw): t = time.time(); r = f(*a, **kw); print(time.time() - t); return r; return wrapper`. Apply with @timed above any function.",
            "I'm trying to optimize a Postgres query that does a full table scan. The EXPLAIN shows Seq Scan on a 50M row table.",
            "First check if there's a usable index on the WHERE clause column. If yes but Postgres ignores it, your filter probably has low selectivity. Try `set enable_seqscan = off;` then re-run EXPLAIN to compare.",
            "Got it. What if there's no index yet and creating one is expensive?",
            "Consider partial indexes (only index rows that match a common predicate), or BRIN indexes if data is naturally ordered (e.g., timestamps). Both are much cheaper to maintain than full B-trees.",
        ]
    docs: list[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if isinstance(d, str):
                docs.append(d)
            elif isinstance(d, dict) and "text" in d:
                docs.append(d["text"])
    else:
        docs.append(path.read_text())
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to a .jsonl or .txt file of conversational samples")
    parser.add_argument("--max-tokens-per-doc", type=int, default=512)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "compression.json")
    args = parser.parse_args()

    print(f"Loading dataset...")
    docs = load_dataset(args.dataset)
    print(f"  {len(docs)} documents")

    print(f"Loading probability model: {args.model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    bos_id = tokenizer.bos_token_id or 0

    @torch.no_grad()
    def logprob_fn(prev):
        ids = torch.tensor([prev if prev else [bos_id]], device=device)
        out = model(ids)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        return logp

    encoder = PLTEncoder(logprob_fn, vocab_size=vocab_size)
    results = []
    bits_per_token = []

    for i, doc in enumerate(docs):
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        if len(tokens) > args.max_tokens_per_doc:
            tokens = tokens[:args.max_tokens_per_doc]
        if len(tokens) < 5:
            continue
        t0 = time.time()
        encoded, stats = encoder.encode(tokens)
        encode_secs = time.time() - t0

        decoded = encoder.decode(encoded, len(tokens))
        roundtrip = (decoded == tokens)

        result = {
            "doc_idx": i,
            "n_tokens": stats.n_tokens,
            "compressed_bytes": stats.compressed_bytes,
            "bits_per_token": stats.bits_per_token,
            "entropy_floor_bpt": stats.expected_bits_per_token,
            "overhead_factor": stats.overhead_factor,
            "encode_secs": encode_secs,
            "roundtrip_ok": roundtrip,
        }
        results.append(result)
        bits_per_token.append(stats.bits_per_token)
        print(f"  doc {i+1}/{len(docs)}: {stats.summary()} "
              f"[{'OK' if roundtrip else 'FAIL'}]")

    bpt = np.array(bits_per_token)
    summary = {
        "model": args.model,
        "device": device,
        "n_docs": len(results),
        "mean_bits_per_token": float(bpt.mean()),
        "median_bits_per_token": float(np.median(bpt)),
        "min_bits_per_token": float(bpt.min()),
        "max_bits_per_token": float(bpt.max()),
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results to {args.out}")

    print(f"\n{'=' * 60}")
    print(f"Aggregate (across {len(results)} documents):")
    print(f"  Mean bits/token:   {summary['mean_bits_per_token']:.2f}")
    print(f"  Median bits/token: {summary['median_bits_per_token']:.2f}")
    print(f"  Range:             {summary['min_bits_per_token']:.2f} – {summary['max_bits_per_token']:.2f}")
    print(f"{'=' * 60}")
    print()
    print("Projected per-token KV compression ratios:")
    mean_bpt = summary["mean_bits_per_token"]
    for label, fp16_bytes in [("8B", 32_000), ("70B (GQA)", 320_000), ("405B", 1_000_000)]:
        ratio = (fp16_bytes * 8) / mean_bpt
        print(f"  vs {label} FP16 KV: {ratio:>14,.0f}x")

    # Optional: histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(bpt, bins=20, edgecolor="black")
        ax.set_xlabel("Bits per token")
        ax.set_ylabel("Number of documents")
        ax.set_title(f"AC compression w/ {args.model}: mean {bpt.mean():.2f} bpt")
        ax.axvline(bpt.mean(), color="red", linestyle="--", label=f"mean {bpt.mean():.2f}")
        ax.legend()
        plot_path = args.out.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Plot: {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
