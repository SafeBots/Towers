"""Persistent storage for the cache trie.

This is the bridge between the in-memory CacheTrie data structure in
cache_trie.py and the on-disk layout that populate.py / recall.py use.
It also wraps llama-server's slot save/restore API so that base segments
can be re-materialized to GPU memory in milliseconds rather than seconds.

What the disk layout looks like:

    {cache_dir}/
        trie.json                          Trie structure: segments, parents, levels
        bases/
            seg_<id>.tokens                Raw tokens for a base segment
            seg_<id>.kv                    llama-server slot-save file (optional)
            seg_<id>.bin                   AC-compressed tokens (optional)
        sessions/
            sess_NNNNNNNN.tokens           DYNAMIC-BLOCK tokens only (not the full prompt)
            sess_NNNNNNNN.bin              AC-compressed dynamic block
            sess_NNNNNNNN.json             Metadata, including base_segment_id pointer
        index.json                         Aggregate stats

Why this is faster than the naive "store full prompt per session":

    Each session stores ~3000 tokens of dynamic block instead of ~4000
    tokens of full prompt. About a 25% reduction in per-session bytes.

    More importantly, thaw becomes ~10-50ms (slot restore from disk)
    plus ~few hundred ms (dynamic-block prefill of ~3000 tokens), instead
    of ~5-15 seconds for a cold 4000-token prefill. This is the
    "one parallel prefill pass" cost the paper claims.

Why this is faster than "always cache_prompt=True":

    llama-server's cache_prompt=True only matches a prefix that's
    currently resident in some slot's KV state. After a few hundred
    sessions cycle through 4 slots, your base segments get evicted.
    Slot save/restore makes the base segments persistent on disk, so
    a thaw always pays the disk-read cost (cheap) rather than the
    base-prefill cost (expensive).

Threading and atomicity:

    Single-writer assumption: only one populate.py runs against one
    cache_dir at a time. The trie.json is rewritten on each new segment;
    if populate.py is killed mid-write we use temp-file-and-rename for
    atomic replacement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .cache_trie import CacheTrie, Segment, SegmentLevel


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via temp-and-rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.rename(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.rename(tmp, path)


def _serialize_trie(trie: CacheTrie) -> dict:
    """Convert a CacheTrie to a JSON-serializable dict.

    We store segments as a flat list with parent references by segment_id.
    Tokens are not stored inline because they may be large; they live in
    separate files referenced by segment_id.
    """
    segments_out = []
    for seg in trie.all_segments():
        segments_out.append({
            "segment_id": seg.segment_id,
            "level": seg.level.name,
            "parent_id": seg.parent.segment_id if seg.parent else None,
            "n_tokens": seg.length,
            "source_position": seg.source_position,
            "has_kv_file": seg.slot_file is not None,
        })
    return {
        "root_segment_id": trie.root.segment_id if trie.root else None,
        "segments": segments_out,
    }


def _deserialize_trie(data: dict, bases_dir: Path) -> CacheTrie:
    """Reconstruct a CacheTrie from a JSON dict. Tokens are loaded from
    the bases_dir.
    """
    trie = CacheTrie()
    # First pass: load tokens for each segment
    segs_by_id: dict[str, Segment] = {}
    raw_segs = data["segments"]
    # Process in topological order: roots first, then by depth
    remaining = list(raw_segs)
    while remaining:
        progress = False
        still_remaining = []
        for s in remaining:
            if s["parent_id"] is None or s["parent_id"] in segs_by_id:
                tokens_path = bases_dir / f"{s['segment_id']}.tokens"
                if tokens_path.exists():
                    tokens = np.frombuffer(tokens_path.read_bytes(), dtype=np.int32).tolist()
                else:
                    tokens = []  # for compatibility with old caches
                parent = segs_by_id.get(s["parent_id"])
                seg = Segment(
                    segment_id=s["segment_id"],
                    tokens=tokens,
                    level=SegmentLevel[s["level"]],
                    parent=parent,
                    source_position=s["source_position"],
                    slot_file=(s["segment_id"] + ".kv") if s["has_kv_file"] else None,
                )
                trie.add(seg)
                segs_by_id[s["segment_id"]] = seg
                progress = True
            else:
                still_remaining.append(s)
        if not progress:
            raise RuntimeError(f"Could not resolve {len(still_remaining)} segments")
        remaining = still_remaining
    return trie


# ----------------------------------------------------------------------------
# llama-server slot API helpers
# ----------------------------------------------------------------------------

def _server_slot_save(target_url: str, slot_id: int, filename: str) -> dict:
    """Tell llama-server to save the current KV state of slot_id to
    {slot_save_path}/{filename}. Returns timing info.

    The server's --slot-save-path option determines the root directory
    these files are written to (typically /tmp/towers_slots).
    """
    r = requests.post(
        f"{target_url}/slots/{slot_id}?action=save",
        json={"filename": filename},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _server_slot_restore(target_url: str, slot_id: int, filename: str) -> dict:
    """Restore a saved KV state into slot_id. Returns timing info."""
    r = requests.post(
        f"{target_url}/slots/{slot_id}?action=restore",
        json={"filename": filename},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _server_slot_erase(target_url: str, slot_id: int) -> None:
    """Erase a slot's KV state. Used to prepare a slot for a fresh prefill."""
    try:
        r = requests.post(
            f"{target_url}/slots/{slot_id}?action=erase",
            timeout=30,
        )
        r.raise_for_status()
    except Exception:
        pass  # erase is best-effort


def _server_prefill(target_url: str, slot_id: int, tokens: list[int],
                    cache_prompt: bool = False) -> dict:
    """Prefill tokens into a slot without generating. Returns timings."""
    t0 = time.time()
    r = requests.post(
        f"{target_url}/completion",
        json={
            "prompt": tokens,
            "id_slot": slot_id,
            "n_predict": 0,
            "temperature": 0.0,
            "cache_prompt": cache_prompt,
        },
        timeout=600,
    )
    r.raise_for_status()
    elapsed_ms = (time.time() - t0) * 1000
    data = r.json()
    return {
        "total_ms": elapsed_ms,
        "prompt_ms": data.get("timings", {}).get("prompt_ms", 0.0),
        "n_prefill": data.get("timings", {}).get("prompt_n", len(tokens)),
    }


# ----------------------------------------------------------------------------
# Tower store
# ----------------------------------------------------------------------------

@dataclass
class TowerStoreStats:
    """Aggregate stats for the cache as a whole."""
    n_sessions: int = 0
    n_base_segments: int = 0
    total_base_token_bytes: int = 0
    total_dynamic_token_bytes: int = 0
    total_dynamic_compressed_bytes: int = 0
    started_at: str = ""
    last_updated_at: Optional[str] = None


class TowerStore:
    """Persistent cache-trie storage with llama-server integration.

    Usage in populate.py:

        store = TowerStore(cache_dir, target_url)
        bot_seg = store.ensure_bot_segment(
            platform_id="magarshak",
            community_id="tech",
            bot_id="py-helper",
            base_text=full_system_prompt,
            tokenize_fn=tokenizer.encode,
        )
        # bot_seg.slot_file is now persisted to disk

        # Generate a session that builds on this bot segment
        session_record = store.add_session(
            bot_seg=bot_seg,
            dynamic_tokens=conversation_token_list,
            metadata={"topic": "...", "n_turns": 12},
            compressed_bytes=ac_compressed_bitstring,
            encode_secs=encode_time,
            generate_secs=gen_time,
        )

    Usage in recall.py:

        store = TowerStore(cache_dir, target_url)
        session = store.load_session("sess_00000123")
        # session has bot_seg + dynamic_tokens + metadata

        # Fast thaw: restore bot KV state (~10ms), then prefill dynamic block
        timings = store.thaw_session(slot_id=0, session=session)
        # Slot 0 is now warm with the full session ready to continue
    """

    def __init__(
        self,
        cache_dir: Path,
        target_url: str = "http://localhost:8000",
    ):
        self.cache_dir = Path(cache_dir)
        self.target_url = target_url
        self.bases_dir = self.cache_dir / "bases"
        self.sessions_dir = self.cache_dir / "sessions"
        self.trie_path = self.cache_dir / "trie.json"
        self.index_path = self.cache_dir / "index.json"

        self.bases_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize trie
        if self.trie_path.exists():
            self.trie = _deserialize_trie(
                json.loads(self.trie_path.read_text()), self.bases_dir
            )
        else:
            self.trie = CacheTrie()

        # Load or initialize stats
        if self.index_path.exists():
            self.stats = TowerStoreStats(**json.loads(self.index_path.read_text()))
        else:
            from datetime import datetime
            self.stats = TowerStoreStats(started_at=datetime.utcnow().isoformat() + "Z")
            self._save_index()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_trie(self) -> None:
        data = _serialize_trie(self.trie)
        _atomic_write_text(self.trie_path, json.dumps(data, indent=2))

    def _save_index(self) -> None:
        from datetime import datetime
        self.stats.last_updated_at = datetime.utcnow().isoformat() + "Z"
        _atomic_write_text(self.index_path, json.dumps(asdict(self.stats), indent=2))

    def _save_base_tokens(self, segment_id: str, tokens: list[int]) -> int:
        """Write a base segment's tokens to disk. Returns bytes written."""
        path = self.bases_dir / f"{segment_id}.tokens"
        if path.exists():
            return path.stat().st_size
        arr = np.array(tokens, dtype=np.int32)
        _atomic_write_bytes(path, arr.tobytes())
        return path.stat().st_size

    # ------------------------------------------------------------------
    # Base segment management
    # ------------------------------------------------------------------

    def ensure_segment(
        self,
        parent: Optional[Segment],
        tokens: list[int],
        level: SegmentLevel,
        segment_id: Optional[str] = None,
        slot_id_for_kv_save: Optional[int] = None,
    ) -> Segment:
        """Add a segment to the trie if it doesn't already exist.

        If slot_id_for_kv_save is given AND this is a new segment, AND the
        slot already has the corresponding KV state materialized, we call
        the server's slot-save endpoint to persist the KV state to disk.
        On subsequent thaws of any session with this segment as ancestor,
        we restore from this file rather than re-prefilling.

        Returns the segment (possibly deduplicated to an existing one).
        """
        seg = self.trie.extend(parent, tokens, level, segment_id=segment_id)

        # If this is a fresh segment, persist its tokens to disk
        tokens_path = self.bases_dir / f"{seg.segment_id}.tokens"
        if not tokens_path.exists():
            n_bytes = self._save_base_tokens(seg.segment_id, tokens)
            self.stats.n_base_segments += 1
            self.stats.total_base_token_bytes += n_bytes

        # Try to persist the KV state via slot save
        if slot_id_for_kv_save is not None and seg.slot_file is None:
            kv_filename = f"{seg.segment_id}.kv"
            kv_path = self.bases_dir / kv_filename
            if not kv_path.exists():
                try:
                    _server_slot_save(self.target_url, slot_id_for_kv_save, kv_filename)
                    seg.slot_file = kv_filename
                except Exception as e:
                    # Slot save isn't critical; thaw falls back to prefill
                    pass

        self._save_trie()
        self._save_index()
        return seg

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _next_session_id(self) -> str:
        return f"sess_{self.stats.n_sessions + 1:08d}"

    def add_session(
        self,
        bot_seg: Segment,
        dynamic_tokens: list[int],
        compressed_bytes: bytes,
        metadata: dict,
        encode_secs: float = 0.0,
        generate_secs: float = 0.0,
        bits_per_token: float = 0.0,
    ) -> dict:
        """Store a new session, atomically.

        The session record only stores the DYNAMIC BLOCK tokens; the
        system-prompt/community/bot tokens live in the trie and are
        shared across all sessions for that bot.
        """
        from datetime import datetime

        session_id = self._next_session_id()
        raw_token_bytes = len(dynamic_tokens) * 4  # int32

        record = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "bot_segment_id": bot_seg.segment_id,
            "n_dynamic_tokens": len(dynamic_tokens),
            "base_n_tokens": bot_seg.source_position + bot_seg.length,
            "dynamic_token_bytes": raw_token_bytes,
            "compressed_bytes": len(compressed_bytes),
            "bits_per_token": bits_per_token,
            "encode_secs": encode_secs,
            "generate_secs": generate_secs,
            **metadata,
        }

        # Atomic write
        base = self.sessions_dir / session_id
        arr = np.array(dynamic_tokens, dtype=np.int32)
        _atomic_write_bytes(base.with_suffix(".tokens"), arr.tobytes())
        _atomic_write_bytes(base.with_suffix(".bin"), compressed_bytes)
        _atomic_write_text(base.with_suffix(".json"), json.dumps(record, indent=2))

        # Update stats
        self.stats.n_sessions += 1
        self.stats.total_dynamic_token_bytes += raw_token_bytes
        self.stats.total_dynamic_compressed_bytes += len(compressed_bytes)
        self._save_index()

        return record

    def load_session(self, session_id: str) -> dict:
        """Load a session record. Returns a dict with metadata, the bot
        segment, and the dynamic tokens.
        """
        base = self.sessions_dir / session_id
        if not base.with_suffix(".json").exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        meta = json.loads(base.with_suffix(".json").read_text())
        dynamic_tokens = np.frombuffer(
            base.with_suffix(".tokens").read_bytes(), dtype=np.int32
        ).tolist()
        bot_seg = self.trie.by_id.get(meta["bot_segment_id"])
        if bot_seg is None:
            raise RuntimeError(
                f"Session {session_id} references unknown base segment "
                f"{meta['bot_segment_id']}"
            )
        return {
            "metadata": meta,
            "bot_segment": bot_seg,
            "dynamic_tokens": dynamic_tokens,
            "tower": self.trie.tower_for(bot_seg),
        }

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.sessions_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Thaw: restore a session into a llama-server slot
    # ------------------------------------------------------------------

    def thaw_session(
        self,
        slot_id: int,
        session_id: Optional[str] = None,
        session: Optional[dict] = None,
    ) -> dict:
        """Restore a session into the given llama-server slot.

        Strategy:
            1. If the bot segment has a saved KV file: restore it (~10-50ms)
            2. Otherwise: prefill the full base tower (one-time cost; we
               then save the KV file for next time)
            3. Prefill the dynamic block on top of the base state

        Returns timing breakdown:
            {
                "base_restore_ms": ...,    # 0 if base wasn't pre-saved
                "base_prefill_ms": ...,    # 0 if base was restored from disk
                "dynamic_prefill_ms": ..., # always present
                "total_ms": ...,
                "base_tokens": ...,
                "dynamic_tokens": ...,
                "cache_hit": True/False,   # True if we used slot restore
            }
        """
        if session is None:
            session = self.load_session(session_id)

        bot_seg = session["bot_segment"]
        dynamic_tokens = session["dynamic_tokens"]
        tower = session["tower"]
        base_tokens_total = sum(s.length for s in tower)

        timings = {
            "base_restore_ms": 0.0,
            "base_prefill_ms": 0.0,
            "dynamic_prefill_ms": 0.0,
            "total_ms": 0.0,
            "base_tokens": base_tokens_total,
            "dynamic_tokens": len(dynamic_tokens),
            "cache_hit": False,
        }
        wall_t0 = time.time()

        # Erase the slot to start fresh
        _server_slot_erase(self.target_url, slot_id)

        # Try slot restore first
        kv_filename = f"{bot_seg.segment_id}.kv"
        kv_path = self.bases_dir / kv_filename
        if kv_path.exists() and bot_seg.slot_file is not None:
            t0 = time.time()
            try:
                _server_slot_restore(self.target_url, slot_id, kv_filename)
                timings["base_restore_ms"] = (time.time() - t0) * 1000
                timings["cache_hit"] = True
            except Exception:
                # Fall through to prefill
                pass

        # If restore failed or wasn't available, prefill the base tower
        if not timings["cache_hit"]:
            full_base_tokens: list[int] = []
            for s in tower:
                full_base_tokens.extend(s.tokens)
            if full_base_tokens:
                t0 = time.time()
                prefill_info = _server_prefill(
                    self.target_url, slot_id, full_base_tokens, cache_prompt=True
                )
                timings["base_prefill_ms"] = prefill_info["total_ms"]

                # Save the resulting KV state for next time
                try:
                    _server_slot_save(self.target_url, slot_id, kv_filename)
                    bot_seg.slot_file = kv_filename
                    self._save_trie()
                except Exception:
                    pass

        # Now prefill the dynamic block on top
        if dynamic_tokens:
            t0 = time.time()
            # cache_prompt=True so the base state stays in the slot
            r = requests.post(
                f"{self.target_url}/completion",
                json={
                    "prompt": dynamic_tokens,
                    "id_slot": slot_id,
                    "n_predict": 0,
                    "temperature": 0.0,
                    "cache_prompt": True,
                },
                timeout=600,
            )
            r.raise_for_status()
            timings["dynamic_prefill_ms"] = (time.time() - t0) * 1000

        timings["total_ms"] = (time.time() - wall_t0) * 1000
        return timings

    # ------------------------------------------------------------------
    # Amortization reporting
    # ------------------------------------------------------------------

    def amortization_summary(self) -> dict:
        """Compute the structural-amortization numbers for the current cache.

        Returns:
            {
                "n_sessions": N,
                "n_base_segments": M,
                "base_token_bytes_total": ...,
                "dynamic_token_bytes_total": ...,
                "amortized_bytes_per_session": ...,
                "naive_bytes_per_session": ...,   # what it would cost without sharing
                "amortization_ratio": ...,
                "compressed_bytes_per_session": ...,
            }
        """
        n_sessions = self.stats.n_sessions
        if n_sessions == 0:
            return {
                "n_sessions": 0,
                "n_base_segments": self.stats.n_base_segments,
                "amortized_bytes_per_session": 0.0,
                "naive_bytes_per_session": 0.0,
                "amortization_ratio": 1.0,
            }

        avg_base_bytes_per_session = self.stats.total_base_token_bytes / n_sessions
        avg_dynamic_bytes = self.stats.total_dynamic_token_bytes / n_sessions
        amortized = avg_base_bytes_per_session + avg_dynamic_bytes

        # Naive: each session would store its full prompt (base + dynamic)
        # As n_sessions grows, the base_token_bytes is fixed (M segments)
        # while dynamic grows linearly. Naive would have base*N + dynamic*N.
        # avg_dynamic_bytes is the per-session dynamic cost.
        # avg base bytes per session NAIVELY: total_base_token_bytes (no sharing)
        # which equals (total_base_token_bytes); this is the "what one session
        # would pay" cost. But naive means EVERY session pays it.
        # Total amortized: total_base + N * avg_dynamic
        # Total naive:     N * (avg_base_per_segment_chain + avg_dynamic)
        # where avg_base_per_segment_chain = ... actually we need the average
        # base tokens PER SESSION (not per segment).
        # For simplicity: assume each session's base tower is roughly
        # total_base / M tokens long.

        if self.stats.n_base_segments > 0:
            avg_base_chain_bytes = self.stats.total_base_token_bytes / self.stats.n_base_segments
        else:
            avg_base_chain_bytes = 0
        naive_bytes_per_session = avg_base_chain_bytes + avg_dynamic_bytes

        ratio = naive_bytes_per_session / amortized if amortized > 0 else 1.0
        compressed_per_session = self.stats.total_dynamic_compressed_bytes / n_sessions

        return {
            "n_sessions": n_sessions,
            "n_base_segments": self.stats.n_base_segments,
            "base_token_bytes_total": self.stats.total_base_token_bytes,
            "dynamic_token_bytes_total": self.stats.total_dynamic_token_bytes,
            "amortized_bytes_per_session": amortized,
            "naive_bytes_per_session": naive_bytes_per_session,
            "amortization_ratio": ratio,
            "compressed_bytes_per_session": compressed_per_session,
        }
