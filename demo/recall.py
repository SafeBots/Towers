"""recall.py — Interactive session recall with fast tower thaw.

The headline magic trick for the Scoble demo. Pulls a random (or named)
session from the cache, thaws it via tower-based slot restore + dynamic
prefill, and drops into an interactive prompt.

What thaw_session does and why it's fast:

    A naive "restore full conversation" would prefill the entire 4000
    tokens from scratch — that's ~5-15 seconds on a 24 GB MacBook with
    Qwen-14B-Q4. Too slow for a live demo to feel magical.

    With tower amortization:
        1. The bot's KV state (system prompt of ~150-300 tokens) is
           already saved to disk by populate.py.
        2. We restore that KV file directly into a llama-server slot.
           This is a memcpy + a small amount of GPU upload: ~10-100 ms
           depending on context length and disk speed.
        3. We then prefill only the dynamic block (~3000-3500 tokens).
           Prefill IS parallel within a sequence (transformer attention
           processes all positions in one shot), so this is one forward
           pass on Qwen-14B: maybe ~500-1500 ms on M-series.
        4. Total: under 2 seconds for a 4000-token session, often well
           under 1 second.

What the demo viewer sees:

    - The thaw timing in console: base restore + dynamic prefill, two
      milliseconds-resolution numbers.
    - The full conversation appearing in the terminal, having come from
      a 12 KB file on disk.
    - Live continuation that's indistinguishable from a never-paused
      conversation.

Usage:

    python demo/recall.py                          # random session
    python demo/recall.py --session sess_00001234  # specific session
    python demo/recall.py --interactive            # pick from recent
    python demo/recall.py --show-only              # show content, no chat
    python demo/recall.py --benchmark              # cycle 10 sessions, print stats
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests
import numpy as np

from tower_runner.tower_store import TowerStore


def print_session_summary(meta: dict, dyn_path: Path, bin_path: Path,
                          bot_seg, tower) -> None:
    """Pretty-print session metadata and tower structure."""
    dyn_size = dyn_path.stat().st_size if dyn_path.exists() else 0
    bin_size = bin_path.stat().st_size if bin_path.exists() else 0
    print(f"Session {meta['session_id']}")
    print(f"  Created:        {meta['created_at']}")
    print(f"  Topic:          {meta.get('topic', '(unknown)')}")
    print(f"  Bot triple:     "
          f"{meta.get('platform_id','?')}/{meta.get('community_id','?')}/{meta.get('bot_id','?')}")
    print(f"  Tower (root → bot):")
    for s in tower:
        print(f"    {s.segment_id:<55} {s.level.name:<10} {s.length:>5} tok")
    print(f"  Dynamic block:  {meta['n_dynamic_tokens']:>6} tokens")
    print(f"  On-disk size:")
    print(f"    Dynamic tokens (raw): {dyn_size:>7,} B")
    print(f"    AC-compressed:        {bin_size:>7,} B  "
          f"({meta['bits_per_token']:.2f} bpt)")
    base_total = sum(s.length for s in tower) * 4
    print(f"    Bot base (shared):    {base_total:>7,} B  "
          f"(amortized across all sessions of this bot)")


def show_conversation(target_url: str, slot_id: int, dynamic_tokens: list[int]) -> None:
    """Detokenize the dynamic tokens using llama-server and print."""
    # llama-server's /detokenize takes the token IDs
    r = requests.post(
        f"{target_url}/detokenize",
        json={"tokens": dynamic_tokens},
        timeout=60,
    )
    r.raise_for_status()
    text = r.json().get("content", "")
    print(text)


def continue_conversation(target_url: str, slot_id: int, user_message: str,
                          max_predict: int = 300) -> tuple[str, dict]:
    """Send a user message and stream back the assistant's reply.

    cache_prompt=True so the recalled state stays in slot KV; only the
    new user-message tokens get prefilled before generation.
    """
    full_prompt = f"\n\nUser: {user_message}\n\nAssistant:"
    t0 = time.time()
    r = requests.post(
        f"{target_url}/completion",
        json={
            "prompt": full_prompt,
            "id_slot": slot_id,
            "n_predict": max_predict,
            "temperature": 0.7,
            "cache_prompt": True,
            "stop": ["\n\nUser:", "\n\nAssistant:"],
        },
        timeout=300,
    )
    r.raise_for_status()
    elapsed_ms = (time.time() - t0) * 1000
    data = r.json()
    return data.get("content", "").strip(), {
        "total_ms": elapsed_ms,
        "tokens_predicted": data.get("tokens_predicted", 0),
        "prompt_ms": data.get("timings", {}).get("prompt_ms", 0.0),
        "predict_ms": data.get("timings", {}).get("predict_ms", 0.0),
    }


def run_thaw_benchmark(store: TowerStore, target_url: str, slot_id: int,
                       n_trials: int = 10) -> None:
    """Cycle N random sessions through thaw; print latency stats."""
    sessions = store.list_sessions()
    if len(sessions) < n_trials:
        print(f"Need at least {n_trials} sessions for benchmark; have {len(sessions)}")
        return

    print(f"\nThaw benchmark: {n_trials} random sessions")
    print("=" * 70)
    print(f"{'session':<20} {'cache':>6} {'base_restore':>14} "
          f"{'base_prefill':>14} {'dyn_prefill':>14} {'total':>10}")

    total_ms_list = []
    dyn_ms_list = []
    for sid in random.sample(sessions, n_trials):
        try:
            timings = store.thaw_session(slot_id=slot_id, session_id=sid)
            cache_hit = "yes" if timings["cache_hit"] else "no"
            print(f"{sid:<20} {cache_hit:>6} "
                  f"{timings['base_restore_ms']:>10.0f} ms "
                  f"{timings['base_prefill_ms']:>10.0f} ms "
                  f"{timings['dynamic_prefill_ms']:>10.0f} ms "
                  f"{timings['total_ms']:>7.0f} ms")
            total_ms_list.append(timings['total_ms'])
            dyn_ms_list.append(timings['dynamic_prefill_ms'])
        except Exception as e:
            print(f"{sid:<20} thaw failed: {e}")

    if total_ms_list:
        total_ms_list.sort()
        print()
        print(f"  Total thaw latency:")
        print(f"    Median:  {total_ms_list[len(total_ms_list)//2]:>7.0f} ms")
        print(f"    Mean:    {sum(total_ms_list)/len(total_ms_list):>7.0f} ms")
        print(f"    Min/Max: {total_ms_list[0]:>7.0f} / {total_ms_list[-1]:>7.0f} ms")
        print(f"  Of which dynamic-block prefill (most of the cost):")
        print(f"    Median:  {sorted(dyn_ms_list)[len(dyn_ms_list)//2]:>7.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="Recall a session from the cache")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "towers_cache")
    parser.add_argument("--target-url", default="http://localhost:8000")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--session", default=None,
                        help="Specific session_id (default: random)")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--show-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Cycle N sessions through thaw and print latency stats")
    parser.add_argument("--benchmark-trials", type=int, default=10)
    args = parser.parse_args()

    if not args.cache_dir.exists():
        print(f"Cache dir {args.cache_dir} not found.")
        sys.exit(1)

    store = TowerStore(args.cache_dir, target_url=args.target_url)
    sessions = store.list_sessions()
    if not sessions:
        print(f"No sessions in {args.cache_dir}/sessions/. Run populate.py first.")
        sys.exit(1)

    if args.benchmark:
        run_thaw_benchmark(store, args.target_url, args.slot_id,
                          n_trials=args.benchmark_trials)
        return

    if args.session:
        if f"{args.session}" not in sessions:
            print(f"Session {args.session} not found.")
            sys.exit(1)
        session_id = args.session
    elif args.interactive:
        recent = sessions[-20:]
        print("Recent sessions:")
        for i, sid in enumerate(reversed(recent)):
            meta = json.loads((store.sessions_dir / f"{sid}.json").read_text())
            print(f"  [{i}] {sid} - {meta.get('community_id','?')}/{meta.get('bot_id','?')}: "
                  f"{meta.get('topic','?')}")
        choice = input("\nSelect [0-19]: ").strip()
        try:
            idx = int(choice)
            session_id = list(reversed(recent))[idx]
        except (ValueError, IndexError):
            print("Invalid choice.")
            sys.exit(1)
    else:
        session_id = random.choice(sessions)

    # Load and summarize
    session = store.load_session(session_id)
    meta = session["metadata"]
    bot_seg = session["bot_segment"]
    tower = session["tower"]

    print("=" * 70)
    print_session_summary(
        meta,
        store.sessions_dir / f"{session_id}.tokens",
        store.sessions_dir / f"{session_id}.bin",
        bot_seg, tower,
    )
    print()
    print("=" * 70)
    print("Thawing tower (slot restore + dynamic prefill)...")

    timings = store.thaw_session(slot_id=args.slot_id, session=session)
    print(f"  base restore (slot KV from disk): {timings['base_restore_ms']:>8.0f} ms"
          f"  ({timings['base_tokens']} base tokens, cache hit: {timings['cache_hit']})")
    print(f"  base prefill (if no cache):       {timings['base_prefill_ms']:>8.0f} ms")
    print(f"  dynamic prefill (one fwd pass):   {timings['dynamic_prefill_ms']:>8.0f} ms"
          f"  ({timings['dynamic_tokens']} dynamic tokens)")
    print(f"  total wall:                       {timings['total_ms']:>8.0f} ms")
    print()

    print("=" * 70)
    print("Recalled conversation:")
    print("=" * 70)
    show_conversation(args.target_url, args.slot_id, session["dynamic_tokens"])
    print()
    print("=" * 70)

    if args.show_only:
        return

    print("Continue the conversation. Type your message; the model responds.")
    print("Type 'exit' or Ctrl-D to end.")
    print()

    try:
        while True:
            user_msg = input("You: ").strip()
            if not user_msg or user_msg.lower() in {"exit", "quit"}:
                break
            print(f"\nAssistant: ", end="", flush=True)
            response, t = continue_conversation(
                args.target_url, args.slot_id, user_msg
            )
            print(response)
            print(f"\n  [{t['tokens_predicted']} tok in {t['predict_ms']:.0f} ms, "
                  f"prefill {t['prompt_ms']:.0f} ms]\n")
    except EOFError:
        print()


if __name__ == "__main__":
    main()
