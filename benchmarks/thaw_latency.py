"""Benchmark: cold-tier thaw latency.

Measures the time to bring a session from cold storage (compressed bitstring)
back to a hot KV state usable for inference:

    1. AC decode the bitstring to tokens     [~5-15 ms target]
    2. Run llama-server prefill on tokens    [~100-500 ms target on 1B-8B; longer on 70B]
    3. Save the resulting slot back to disk  [~50-150 ms]

This is the conservative thaw path of paper §10.5 (the §12 conjecture
about U_M would replace step 2 with a learned decoder for ~10x speedup,
but is not implemented in v0.1).

Prerequisites: same as example 04 — running llama-server with --slots.

Usage:
    python benchmarks/thaw_latency.py --url http://localhost:8000 \
        --slot-dir /tmp/towers_slots
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.llama_client import LlamaServerClient


SAMPLE_PROMPTS = [
    "User: I need help debugging a memory leak in Python.\nAssistant:",
    "User: Explain how Postgres B-tree indexes work, briefly.\nAssistant:",
    "User: Write a haiku about morning coffee.\nAssistant:",
    "User: Compare TCP and UDP at a high level.\nAssistant:",
    "User: How do I freeze fresh basil so it lasts months?\nAssistant:",
]


def measure_one(client: LlamaServerClient, slot_dir: Path, prompt: str, n_trials: int = 3) -> dict:
    """Measure cold-thaw latency for one prompt by:
        1. Prefilling the prompt fresh (=> measures cold from-tokens cost)
        2. Saving the slot to disk
        3. Erasing the slot
        4. Restoring from disk (=> measures pure restore cost)
    """
    tokens = client.tokenize(prompt, add_special=True)
    n_tokens = len(tokens)
    filename = f"thaw_test_{abs(hash(prompt)) % 100000}.bin"

    prefill_times = []
    restore_times = []
    save_times = []

    for trial in range(n_trials):
        # Cold prefill (the §10.5 conservative thaw path)
        client.slot_erase(slot_id=0)
        t0 = time.time()
        prefill = client.prefill_only(prompt=tokens, slot_id=0, cache_prompt=False)
        prefill_times.append(prefill.prompt_ms)

        # Save to disk
        save = client.slot_save(slot_id=0, filename=filename)
        save_times.append(save.save_ms)

        # Now restore (the hot-cache primitive)
        client.slot_erase(slot_id=0)
        restore = client.slot_restore(slot_id=0, filename=filename)
        restore_times.append(restore.restore_ms)

    return {
        "prompt_chars": len(prompt),
        "n_tokens": n_tokens,
        "prefill_ms_mean": sum(prefill_times) / len(prefill_times),
        "prefill_ms_min": min(prefill_times),
        "save_ms_mean": sum(save_times) / len(save_times),
        "restore_ms_mean": sum(restore_times) / len(restore_times),
        "restore_ms_min": min(restore_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--slot-dir", default="/tmp/towers_slots")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "thaw_latency.json")
    args = parser.parse_args()

    print("Towers of Segments — Benchmark: cold-tier thaw latency")
    print("=" * 70)

    client = LlamaServerClient(base_url=args.url)
    if not client.wait_until_ready(max_wait_s=10):
        print(f"ERROR: llama-server not responding at {args.url}")
        sys.exit(1)

    props = client.props()
    print(f"Server n_ctx: {props.get('default_generation_settings', {}).get('n_ctx', '?')}")

    all_results = []
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        print(f"\nPrompt {i+1}/{len(SAMPLE_PROMPTS)}: {prompt[:60]}...")
        r = measure_one(client, Path(args.slot_dir), prompt, n_trials=args.trials)
        r["prompt"] = prompt
        all_results.append(r)
        print(f"  {r['n_tokens']:>4} tokens: "
              f"prefill {r['prefill_ms_mean']:>6.1f} ms, "
              f"save {r['save_ms_mean']:>5.1f} ms, "
              f"restore {r['restore_ms_mean']:>5.1f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": all_results}, indent=2))

    print(f"\n{'=' * 70}")
    print("Summary (across all prompts):")
    avg_prefill = sum(r["prefill_ms_mean"] for r in all_results) / len(all_results)
    avg_restore = sum(r["restore_ms_mean"] for r in all_results) / len(all_results)
    avg_save = sum(r["save_ms_mean"] for r in all_results) / len(all_results)
    print(f"  Cold thaw (full prefill):  {avg_prefill:>6.1f} ms (= conservative thaw path)")
    print(f"  Save to disk:              {avg_save:>6.1f} ms")
    print(f"  Restore from disk (hot):   {avg_restore:>6.1f} ms (= regular cache primitive)")
    print()
    print("These are realistic latency numbers for the paper's §10.5")
    print("'conservative thaw path' on this model class.")
    print()
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
