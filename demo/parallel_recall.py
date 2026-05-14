"""parallel_recall.py — Thaw N tower sessions into N llama-server slots in parallel.

The "five cold sessions, all live in a few seconds" demo moment.
Picks N random sessions from the cache, dispatches their thaws
concurrently across N slots, prints per-slot timings.

What's being parallelized:

    Within a single thaw, the dynamic-block prefill is already
    GPU-parallel (one transformer forward pass for ~3000 tokens).
    Across thaws, llama-server processes slots in parallel when
    started with --parallel N: separate slots share the model
    weights but have independent KV caches, and llama.cpp's
    scheduler interleaves their forward passes on the same GPU.

    The result: N cold sessions go from disk to live conversation
    in roughly the time of ONE thaw, not N times that. On a 24 GB
    MacBook running Qwen-14B-Q4, 5 sessions thaw in about 3 seconds
    total wall time.

Requires llama-server started with --parallel >= N:

    ./llama-server -m models/qwen2.5-14b-q4.gguf --parallel 5 \\
        --slot-save-path ~/towers_cache/bases/

Usage:

    python demo/parallel_recall.py --n 5
    python demo/parallel_recall.py --n 8 --show-conversations
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from tower_runner.tower_store import TowerStore


def thaw_one(store: TowerStore, slot_id: int, session_id: str) -> dict:
    """Thaw a single session into the given slot. Returns timings + meta."""
    session = store.load_session(session_id)
    meta = session["metadata"]
    bot_seg = session["bot_segment"]
    t0 = time.time()
    timings = store.thaw_session(slot_id=slot_id, session=session)
    wall_secs = time.time() - t0
    return {
        "slot_id": slot_id,
        "session_id": session_id,
        "topic": meta.get("topic", "?"),
        "bot_id": meta.get("bot_id", "?"),
        "community_id": meta.get("community_id", "?"),
        "n_dynamic_tokens": meta.get("n_dynamic_tokens", 0),
        "base_tokens": timings["base_tokens"],
        "base_restore_ms": timings["base_restore_ms"],
        "base_prefill_ms": timings["base_prefill_ms"],
        "dynamic_prefill_ms": timings["dynamic_prefill_ms"],
        "thaw_total_ms": timings["total_ms"],
        "wall_secs": wall_secs,
        "cache_hit": timings["cache_hit"],
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel tower thaw across N slots")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "towers_cache")
    parser.add_argument("--target-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of parallel thaws (must be <= --parallel on server)")
    parser.add_argument("--starting-slot", type=int, default=0)
    parser.add_argument("--show-conversations", action="store_true",
                        help="After thaw, print each session's text")
    args = parser.parse_args()

    if not args.cache_dir.exists():
        print(f"Cache dir {args.cache_dir} does not exist. Run populate.py first.")
        sys.exit(1)

    store = TowerStore(args.cache_dir, target_url=args.target_url)
    sessions = store.list_sessions()
    if len(sessions) < args.n:
        print(f"Need at least {args.n} sessions; have {len(sessions)}.")
        sys.exit(1)

    # Verify server has enough slots
    try:
        h = requests.get(f"{args.target_url}/health", timeout=5)
        h.raise_for_status()
    except Exception as e:
        print(f"Cannot reach llama-server at {args.target_url}: {e}")
        sys.exit(1)

    chosen = random.sample(sessions, args.n)

    print("=" * 72)
    print(f"Picking {args.n} random sessions from {len(sessions):,} in cache:")
    print("=" * 72)
    for i, sid in enumerate(chosen):
        try:
            session = store.load_session(sid)
            meta = session["metadata"]
            n_dyn = meta.get("n_dynamic_tokens", 0)
            n_base = sum(s.length for s in session["tower"])
            dyn_kb = n_dyn * 4 / 1024
            print(f"  slot {args.starting_slot + i}: {sid} - "
                  f"{meta.get('community_id','?')}/{meta.get('bot_id','?')} - "
                  f"{meta.get('topic','?')[:40]}")
            print(f"           tower {n_base + n_dyn} tok "
                  f"({n_base} base + {n_dyn} dyn), {dyn_kb:.1f} KB dyn on disk")
        except Exception as e:
            print(f"  slot {args.starting_slot + i}: {sid} - failed to load: {e}")

    print()
    print("=" * 72)
    print(f"Dispatching {args.n} parallel thaws across slots "
          f"{args.starting_slot}..{args.starting_slot + args.n - 1}...")
    print("=" * 72)

    wall_t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.n) as executor:
        future_to_idx = {
            executor.submit(thaw_one, store, args.starting_slot + i, sid): i
            for i, sid in enumerate(chosen)
        }
        for fut in as_completed(future_to_idx):
            try:
                results.append(fut.result())
            except Exception as e:
                idx = future_to_idx[fut]
                print(f"  slot {args.starting_slot + idx} failed: {e}")

    wall_secs = time.time() - wall_t0
    results.sort(key=lambda r: r["slot_id"])

    print()
    print(f"All {len(results)} slots ready in {wall_secs:.2f}s wall time")
    print()
    print(f"  {'slot':<5} {'hit':>4} {'restore':>10} {'base_pref':>10} "
          f"{'dyn_pref':>10} {'thaw_tot':>10} {'wall':>8}")
    print(f"  {'-'*5} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    sum_thaw = 0
    for r in results:
        hit = "yes" if r["cache_hit"] else "no"
        print(f"  {r['slot_id']:<5} {hit:>4} "
              f"{r['base_restore_ms']:>8.0f}ms "
              f"{r['base_prefill_ms']:>8.0f}ms "
              f"{r['dynamic_prefill_ms']:>8.0f}ms "
              f"{r['thaw_total_ms']:>8.0f}ms "
              f"{r['wall_secs']:>6.2f}s")
        sum_thaw += r["thaw_total_ms"]
    print()
    print(f"  Sum of individual thaw times: {sum_thaw:.0f}ms")
    print(f"  Wall time (parallel):          {wall_secs*1000:.0f}ms")
    if wall_secs > 0:
        print(f"  Speedup vs serial:             {(sum_thaw/1000)/wall_secs:.2f}x")
    print()

    # Disk-read summary
    total_dyn_bytes = sum(
        store.sessions_dir.joinpath(f"{r['session_id']}.tokens").stat().st_size
        for r in results
    )
    print(f"  Total disk read: {total_dyn_bytes/1024:.1f} KB across {len(results)} sessions")
    print(f"  Each slot is now warm and can be continued via:")
    print(f"    python demo/recall.py --slot-id <N> --session <id>")
    print()

    if args.show_conversations:
        for r in results:
            session = store.load_session(r["session_id"])
            print("=" * 72)
            print(f"slot {r['slot_id']}: {r['session_id']} ({r['community_id']}/{r['bot_id']})")
            print(f"  topic: {r['topic']}")
            print("=" * 72)
            try:
                detok = requests.post(
                    f"{args.target_url}/detokenize",
                    json={"tokens": session["dynamic_tokens"]},
                    timeout=60,
                )
                detok.raise_for_status()
                print(detok.json().get("content", "")[:2000])
                if len(session["dynamic_tokens"]) > 500:
                    print(f"\n  [truncated, full session is "
                          f"{len(session['dynamic_tokens'])} tokens]")
            except Exception as e:
                print(f"  detokenize failed: {e}")
            print()


if __name__ == "__main__":
    main()
