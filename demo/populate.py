"""populate.py — Generate model-vs-model conversations with tower amortization.

This is the v0.2 populator. It uses the cache trie (cache_trie.py) to
share platform/community/bot prefixes across sessions of the same triple.
Per-session storage drops from ~16 KB (full prompt) to ~12 KB (dynamic
block only), and recall thaw is much faster because the bot KV state is
pre-saved on disk via llama-server's slot save API.

Per-session lifecycle:

    1. Pick a (platform, community, bot) triple at random
    2. Restore the bot's pre-saved KV state into a llama-server slot
       (one-time prefill at script startup; subsequent uses restore from disk)
    3. Generate a model-vs-model conversation by alternating User/Assistant
       roles against the server, with the bot's system prompt already cached
    4. Tokenize the GENERATED CONVERSATION ONLY (not the system prompt)
    5. Compress the dynamic-block tokens via FastPLTEncoder
    6. Save: dynamic tokens + AC bytes + metadata pointing at the bot segment

The cache trie grows by a fixed number of base segments (one per
(platform, community, bot) triple in DEFAULT_TRIPLES) and then is stable.
After that, every new session only adds a dynamic block.

Why amortization matters for the demo headline:

    Each session stores ~12 KB of dynamic-block tokens.
    A 70B model's raw FP16 KV cache for the same conversation: ~1.28 GB.
    Empirical ratio: 1.28 GB / 12 KB = roughly 106,000x.
    AC compression on the dynamic block adds another ~5-10x on top
    (the actual bits-per-token measurement gates this).

Usage:

    bash demo/setup.sh                    # download model, build llama.cpp
    python demo/populate.py               # leave running for days
    python demo/scoreboard.py             # in another terminal: watch it grow
"""

import argparse
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _require_imports():
    global np, torch, requests, AutoTokenizer, AutoModelForCausalLM
    global FastPLTEncoder, TowerStore, SegmentLevel
    import numpy as np
    import torch
    import requests
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tower_runner.fast_codec import FastPLTEncoder
    from tower_runner.tower_store import TowerStore
    from tower_runner.cache_trie import SegmentLevel


# Default tower configuration. Each row defines one (platform, community,
# bot) triple. populate.py picks one at random per session.
DEFAULT_TRIPLES = [
    ("magarshak", "tech", "py-helper",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community discusses software engineering and debugging.",
     "You are a Python debugging assistant. Suggest concrete steps and code."),
    ("magarshak", "tech", "rust-helper",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community discusses software engineering and debugging.",
     "You are a Rust mentor for Python programmers. Compare and explain."),
    ("magarshak", "tech", "sql-helper",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community discusses software engineering and debugging.",
     "You are a SQL performance specialist. Focus on EXPLAIN plans."),
    ("magarshak", "tech", "devops-helper",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community discusses software engineering and debugging.",
     "You are a Kubernetes and infrastructure helper."),
    ("magarshak", "cooking", "recipe-assistant",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community shares recipes and kitchen techniques.",
     "You are a recipe assistant. Ask about dietary restrictions when relevant."),
    ("magarshak", "cooking", "technique-coach",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community shares recipes and kitchen techniques.",
     "You explain cooking techniques in detail and adapt to skill level."),
    ("magarshak", "writing", "poetry-coach",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community covers creative and technical writing.",
     "You help write and revise poetry. Be specific and constructive."),
    ("magarshak", "writing", "essay-coach",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community covers creative and technical writing.",
     "You give feedback on essays and short prose. Tighten and clarify."),
    ("magarshak", "science", "explainer",
     "You are an assistant on the Magarshak platform. Be concise and honest.",
     "This community explains scientific concepts in plain English.",
     "You explain scientific concepts simply but correctly. Use analogies."),
]


def ensure_all_segments(store, tokenize_fn, target_url, slot_id_for_save):
    """Build the cache trie's PLATFORM/COMMUNITY/BOT layers from
    DEFAULT_TRIPLES.

    For each unique bot triple, prefill the full base prompt into the
    slot and save the slot's KV state to disk. After this completes,
    every later session can thaw the bot's KV state in milliseconds
    rather than seconds.

    Returns dict (platform_id, community_id, bot_id) -> bot Segment.
    """
    import requests

    bot_segs = {}
    platforms: dict = {}
    communities: dict = {}
    bots: dict = {}

    for plat_id, comm_id, bot_id, plat_sys, comm_sys, bot_sys in DEFAULT_TRIPLES:
        platforms.setdefault(plat_id, plat_sys)
        communities.setdefault((plat_id, comm_id), comm_sys)
        bots[(plat_id, comm_id, bot_id)] = (plat_sys, comm_sys, bot_sys)

    plat_segs = {}
    for plat_id, plat_sys in platforms.items():
        plat_segs[plat_id] = store.ensure_segment(
            parent=None,
            tokens=tokenize_fn(plat_sys + "\n\n"),
            level=SegmentLevel.PLATFORM,
            segment_id=f"platform_{plat_id}",
        )

    comm_segs = {}
    for (plat_id, comm_id), comm_sys in communities.items():
        comm_segs[(plat_id, comm_id)] = store.ensure_segment(
            parent=plat_segs[plat_id],
            tokens=tokenize_fn(comm_sys + "\n\n"),
            level=SegmentLevel.COMMUNITY,
            segment_id=f"community_{plat_id}_{comm_id}",
        )

    for (plat_id, comm_id, bot_id), (plat_sys, comm_sys, bot_sys) in bots.items():
        seg = store.ensure_segment(
            parent=comm_segs[(plat_id, comm_id)],
            tokens=tokenize_fn(bot_sys + "\n\n"),
            level=SegmentLevel.BOT,
            segment_id=f"bot_{plat_id}_{comm_id}_{bot_id}",
        )
        bot_segs[(plat_id, comm_id, bot_id)] = seg

        # Pre-save the slot KV state for this bot's full base tower
        kv_filename = f"{seg.segment_id}.kv"
        kv_path = store.bases_dir / kv_filename
        if not kv_path.exists() or seg.slot_file is None:
            tower = store.trie.tower_for(seg)
            full_tokens = []
            for s in tower:
                full_tokens.extend(s.tokens)
            try:
                requests.post(
                    f"{target_url}/slots/{slot_id_for_save}?action=erase",
                    timeout=30,
                )
                r = requests.post(
                    f"{target_url}/completion",
                    json={
                        "prompt": full_tokens,
                        "id_slot": slot_id_for_save,
                        "n_predict": 0,
                        "temperature": 0.0,
                        "cache_prompt": False,
                    },
                    timeout=600,
                )
                r.raise_for_status()
                r = requests.post(
                    f"{target_url}/slots/{slot_id_for_save}?action=save",
                    json={"filename": kv_filename},
                    timeout=60,
                )
                r.raise_for_status()
                seg.slot_file = kv_filename
                store._save_trie()
                print(f"  Saved base KV for {seg.segment_id} "
                      f"({sum(s.length for s in tower)} tok)")
            except Exception as e:
                print(f"  Warning: could not pre-save KV for {seg.segment_id}: {e}")

    return bot_segs


def generate_dynamic_block(target_url, slot_id, bot_seg, topic,
                           tokens_target, temperature=0.8, max_turn_tokens=250):
    """Generate a model-vs-model conversation in the given slot.

    Assumes the bot's base KV state is already restored into the slot.
    Returns (text_without_system_prompt, n_turns).
    """
    import requests

    # Restore bot KV state
    if bot_seg.slot_file:
        try:
            requests.post(
                f"{target_url}/slots/{slot_id}?action=restore",
                json={"filename": bot_seg.slot_file},
                timeout=60,
            ).raise_for_status()
        except Exception:
            pass

    first_user_msg = f"I want to talk about how to {topic}. Where do I start?"
    conversation_lines = [f"User: {first_user_msg}"]
    approx_tokens = len(first_user_msg) // 4
    turn_count = 1
    next_role = "Assistant"

    while approx_tokens < tokens_target and turn_count < 30:
        history = "\n\n".join(conversation_lines)
        prompt_text = f"{history}\n\n{next_role}:"
        try:
            r = requests.post(
                f"{target_url}/completion",
                json={
                    "prompt": prompt_text,
                    "id_slot": slot_id,
                    "n_predict": max_turn_tokens,
                    "temperature": temperature,
                    "cache_prompt": True,
                    "stop": ["\n\nUser:", "\n\nAssistant:", "</s>", "<|endoftext|>"],
                },
                timeout=180,
            )
            r.raise_for_status()
            turn_text = r.json().get("content", "").strip()
        except Exception as e:
            raise RuntimeError(f"server completion failed: {e}")

        if not turn_text:
            break

        conversation_lines.append(f"{next_role}: {turn_text}")
        approx_tokens += len(turn_text) // 4
        turn_count += 1
        next_role = "User" if next_role == "Assistant" else "Assistant"

    return "\n\n".join(conversation_lines), turn_count


_STOP = False
def _handle_sigint(signum, frame):
    global _STOP
    _STOP = True
    print("\n\nGraceful shutdown. Finishing current session...", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate conversations with cache-trie tower amortization."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "towers_cache")
    parser.add_argument("--target-url", default="http://localhost:8000")
    parser.add_argument("--encoder-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--topics-file", type=Path,
        default=Path(__file__).parent.parent / "data" / "conversational" / "topics.txt"
    )
    parser.add_argument("--tokens-per-session", type=int, default=4000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--slot-id", type=int, default=0)
    args = parser.parse_args()

    _require_imports()
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"populate.py (v0.2 tower amortization)")
    print(f"  cache-dir:          {args.cache_dir}")
    print(f"  target-url:         {args.target_url}")
    print(f"  encoder-model:      {args.encoder_model}")
    print(f"  tokens-per-session: {args.tokens_per_session}")
    print(f"  slot-id:            {args.slot_id}")
    print()

    try:
        h = requests.get(f"{args.target_url}/health", timeout=10)
        h.raise_for_status()
    except Exception as e:
        print(f"ERROR: cannot reach llama-server at {args.target_url}")
        print(f"  {e}\n  Start it first; see docs/macbook.md.")
        sys.exit(1)

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading encoder {args.encoder_model} on {device}...")
    enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    enc_dtype = torch.float16 if device != "cpu" else torch.float32
    enc_model = AutoModelForCausalLM.from_pretrained(
        args.encoder_model, torch_dtype=enc_dtype
    ).to(device).eval()
    encoder = FastPLTEncoder(enc_model, enc_tokenizer, device=device)

    def target_tokenize(text):
        r = requests.post(f"{args.target_url}/tokenize",
                         json={"content": text}, timeout=60)
        r.raise_for_status()
        return r.json()["tokens"]

    print(f"Building cache trie from {len(DEFAULT_TRIPLES)} bot triples...")
    store = TowerStore(args.cache_dir, target_url=args.target_url)
    bot_segs = ensure_all_segments(
        store, target_tokenize, args.target_url, slot_id_for_save=args.slot_id
    )
    print(f"  {len(bot_segs)} bot segments ready")
    print(f"  Existing: {store.stats.n_sessions:,} sessions, "
          f"{store.stats.total_dynamic_compressed_bytes/1e6:.1f} MB compressed")

    topics = [t.strip() for t in args.topics_file.read_text().splitlines() if t.strip()]
    print(f"  {len(topics)} topics loaded")

    print("\n" + "=" * 70)
    print("Generation loop. Ctrl-C for graceful shutdown.")
    print("=" * 70 + "\n")

    n_generated = 0
    start_time = time.time()

    while not _STOP:
        if args.max_sessions and n_generated >= args.max_sessions:
            print(f"\nReached --max-sessions={args.max_sessions}; stopping.")
            break

        triple = random.choice(list(bot_segs.keys()))
        bot_seg = bot_segs[triple]
        topic = random.choice(topics)
        plat_id, comm_id, bot_id = triple

        try:
            t0 = time.time()
            dyn_text, n_turns = generate_dynamic_block(
                target_url=args.target_url,
                slot_id=args.slot_id,
                bot_seg=bot_seg,
                topic=topic,
                tokens_target=args.tokens_per_session,
            )
            generate_secs = time.time() - t0
        except Exception as e:
            print(f"  [generation failed: {e}] - retrying...", flush=True)
            time.sleep(5)
            continue

        try:
            dyn_tokens = enc_tokenizer.encode(dyn_text, add_special_tokens=False)
            if len(dyn_tokens) < 100:
                print(f"  [skipping {topic}: only {len(dyn_tokens)} tokens]", flush=True)
                continue
            t0 = time.time()
            encoded, stats = encoder.encode_tokens(dyn_tokens)
            encode_secs = time.time() - t0
        except Exception as e:
            print(f"  [encoding failed: {e}] - retrying...", flush=True)
            time.sleep(5)
            continue

        record = store.add_session(
            bot_seg=bot_seg,
            dynamic_tokens=dyn_tokens,
            compressed_bytes=encoded,
            metadata={
                "topic": topic,
                "platform_id": plat_id,
                "community_id": comm_id,
                "bot_id": bot_id,
                "n_turns": n_turns,
            },
            encode_secs=encode_secs,
            generate_secs=generate_secs,
            bits_per_token=stats.bits_per_token,
        )

        n_generated += 1
        elapsed = time.time() - start_time
        rate = n_generated / elapsed * 60 if elapsed > 0 else 0
        if n_generated % args.print_every == 0:
            print(
                f"  #{n_generated} {record['session_id']} "
                f"({comm_id}/{bot_id}): {topic[:35]:<35} | "
                f"{record['n_dynamic_tokens']} dyn tok, "
                f"{record['compressed_bytes']} B "
                f"({stats.bits_per_token:.2f} bpt) | "
                f"gen {generate_secs:.0f}s + enc {encode_secs:.0f}s | "
                f"{rate:.1f}/min | total {store.stats.n_sessions:,}",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\nGenerated {n_generated} sessions in {elapsed/60:.1f} min")
    amort = store.amortization_summary()
    print(f"\nFinal amortization summary:")
    print(f"  Sessions:                   {amort['n_sessions']:>10,}")
    print(f"  Base segments:              {amort['n_base_segments']:>10}")
    print(f"  Naive bytes/session:        {amort['naive_bytes_per_session']:>10,.0f}")
    print(f"  Amortized bytes/session:    {amort['amortized_bytes_per_session']:>10,.0f}")
    print(f"  Amortization ratio:         {amort['amortization_ratio']:>10.2f}x")
    print(f"  AC bytes/session (dynamic): {amort['compressed_bytes_per_session']:>10,.0f}")


if __name__ == "__main__":
    main()
