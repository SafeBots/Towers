"""Tower assembly: compose cached segments into a session's KV state.

This file implements the tower assembly procedure of paper §4. The mapping
from paper to code:

    Paper:  Given a leaf segment in the cache trie, walk root to leaf,
            materialize each segment's KV state in order. The final state
            is the prefilled context for that session: the language model
            can then generate continuations from here.

    Code:   TowerRunner.assemble(trie, leaf, slot_id) does this against a
            running llama-server. It returns a Tower object describing
            what was restored from disk versus newly computed.

Implementation strategy for v0.1:

    Use llama-server's existing slot save/restore HTTP API. Each segment
    is stored as a slot save file (one .bin per content-hashed segment).
    Tower assembly works by:

        1. Finding the deepest ancestor whose slot file already exists
        2. Restoring that slot (an O(read-from-disk) operation, very fast)
        3. For remaining segments below it, prefilling them through
           llama-server with cache_prompt=True. This tells the server to
           reuse the just-restored prefix and only compute the new tokens.
        4. Optionally saving the resulting leaf state to disk so the next
           access for this session is a pure restore.

    No C++ patches are needed for this v0.1 implementation. The slot
    save/restore API has been in llama-server since 2023; we just orchestrate
    it from Python.

Cold-tier thaw (paper §10.5, the conservative thaw path):

    When a segment's KV state has been evicted from the hot tier and only
    the compressed bitstring remains on disk, the thaw procedure is:

        a) AC-decode the bitstring back to tokens (~10 ms for a 4000-token
           segment with parallel chunk decoding)
        b) Run one parallel prefill pass through the target model on the
           decoded tokens (~100-500 ms depending on model size and hardware)
        c) Save the resulting slot back to disk so the segment is "warm"
           again

    Total: ~100-500 ms per cold reactivation. This is the conservative
    thaw path that always works; the paper's §12 conjectured universal
    decoder would replace step (b) with a learned function for ~10x
    speedup, but that's research work not implemented here.

Concurrent access:

    Each slot_id in llama-server can only hold one tower at a time. For
    multi-tenant workloads with concurrent users, use multiple slots
    (llama-server --parallel N) and assign one slot per user session.
    This module is single-slot per call; the caller is responsible for
    pooling slots across requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cache_trie import CacheTrie, Segment
from .llama_client import LlamaServerClient, CompletionResult


@dataclass
class TowerAssemblyStats:
    """Latency / size breakdown of one tower assembly."""
    leaf_segment_id: str
    chain_length: int               # number of segments in the chain
    n_restored_segments: int        # how many were already cached on disk
    n_prefilled_segments: int       # how many had to be prefilled from tokens
    total_tokens: int               # total tokens in the tower
    restore_ms: float = 0.0
    prefill_ms: float = 0.0
    cache_hit_tokens: int = 0       # tokens reused from llama-server's prefix cache

    @property
    def total_ms(self) -> float:
        return self.restore_ms + self.prefill_ms

    def summary(self) -> str:
        return (
            f"tower[{self.leaf_segment_id}]: "
            f"{self.chain_length} segments, {self.total_tokens} tokens, "
            f"{self.n_restored_segments} restored + {self.n_prefilled_segments} prefilled, "
            f"{self.restore_ms:.0f}ms restore + {self.prefill_ms:.0f}ms prefill "
            f"= {self.total_ms:.0f}ms total"
        )


@dataclass
class Tower:
    """A fully-materialized tower for one session.

    The tower has been assembled in slot_id of the llama-server. Further
    completion calls with the same slot_id will reuse this KV state.
    """
    leaf: Segment
    chain: list[Segment]
    slot_id: int
    stats: TowerAssemblyStats


class TowerRunner:
    """Orchestrates tower assembly using a llama-server.

    Manages the mapping between (segment in the cache trie) and
    (slot save file on disk). Coordinates the assembly of towers by walking
    the trie and issuing the appropriate save/restore/prefill calls.

    Args:
        client: a LlamaServerClient connected to a running llama-server
        slot_dir: the directory the server was started with via
            --slot-save-path. Used only for filename construction; the
            server actually reads/writes the files.
    """

    def __init__(self, client: LlamaServerClient, slot_dir: str | Path = "/tmp/slots"):
        self.client = client
        self.slot_dir = Path(slot_dir)

    def _slot_file_for(self, segment: Segment) -> str:
        """Deterministic filename for a segment's slot save file.
        Uses content hash so identical segments map to the same file
        (structural deduplication on disk).
        """
        return f"seg_{segment.content_hash[:16]}.bin"

    def _materialize_segment(
        self,
        segment: Segment,
        slot_id: int,
        all_tokens_so_far: list[int],
        force_prefill: bool = False,
    ) -> tuple[float, float, bool]:
        """Ensure segment's KV state is in slot `slot_id`.

        Returns (restore_ms, prefill_ms, was_restored).
        """
        slot_file = self._slot_file_for(segment)
        slot_file_path = self.slot_dir / slot_file

        # Try to restore from disk if a slot file exists
        if not force_prefill and slot_file_path.exists():
            res = self.client.slot_restore(slot_id=slot_id, filename=slot_file)
            # The restore brings the slot's KV cache to the state after
            # processing all tokens up through this segment. No further
            # action needed.
            return (res.restore_ms, 0.0, True)

        # Prefill: process the cumulative token sequence through this segment.
        # We use cache_prompt=True so any already-cached prefix in the slot
        # is reused (this is how restoring a parent and then prefilling a
        # child works: the parent's tokens are cache-hit, only the child's
        # new tokens are computed).
        prefill = self.client.prefill_only(
            prompt=all_tokens_so_far,
            slot_id=slot_id,
            cache_prompt=True,
        )
        prefill_ms = prefill.prompt_ms

        # Save the resulting slot to disk so future assemblies can restore
        save = self.client.slot_save(slot_id=slot_id, filename=slot_file)
        segment.slot_file = slot_file
        return (0.0, prefill_ms, False)

    def assemble(
        self,
        trie: CacheTrie,
        leaf: Segment,
        slot_id: int = 0,
        persist_intermediate: bool = True,
    ) -> Tower:
        """Assemble the tower for a leaf segment in slot `slot_id`.

        Walks root -> leaf, restoring the deepest cached ancestor first
        (skipping intermediate restores if a deeper ancestor is cached),
        then prefilling the remainder.
        """
        chain = trie.tower_for(leaf)

        # Find the deepest ancestor with an existing slot file on disk.
        # Restore that one, then prefill the rest in a single call.
        deepest_cached_idx = -1
        for i, seg in enumerate(chain):
            if (self.slot_dir / self._slot_file_for(seg)).exists():
                deepest_cached_idx = i

        restore_ms = 0.0
        prefill_ms = 0.0
        n_restored = 0
        n_prefilled = 0

        if deepest_cached_idx >= 0:
            # Restore the deepest cached ancestor
            anchor = chain[deepest_cached_idx]
            slot_file = self._slot_file_for(anchor)
            res = self.client.slot_restore(slot_id=slot_id, filename=slot_file)
            restore_ms += res.restore_ms
            n_restored = 1
        else:
            # Nothing cached; we'll prefill from scratch
            self.client.slot_erase(slot_id=slot_id)

        # Prefill any remaining segments (those after deepest_cached_idx)
        remaining = chain[deepest_cached_idx + 1:]
        if remaining:
            # Concatenate all tokens through the leaf. cache_prompt=True
            # ensures the restored prefix is reused, only new tokens prefilled.
            all_tokens: list[int] = []
            for seg in chain:
                all_tokens.extend(seg.tokens)

            prefill = self.client.prefill_only(
                prompt=all_tokens,
                slot_id=slot_id,
                cache_prompt=True,
            )
            prefill_ms = prefill.prompt_ms
            n_prefilled = len(remaining)

            if persist_intermediate:
                # Save slot states for each newly-computed segment so future
                # assemblies can restore from any point in the chain.
                # llama-server's slot files contain the state up through the
                # current position, so saving here gives us the leaf's state.
                slot_file = self._slot_file_for(leaf)
                self.client.slot_save(slot_id=slot_id, filename=slot_file)
                leaf.slot_file = slot_file

        leaf.access_count += 1

        stats = TowerAssemblyStats(
            leaf_segment_id=leaf.segment_id,
            chain_length=len(chain),
            n_restored_segments=n_restored,
            n_prefilled_segments=n_prefilled,
            total_tokens=sum(s.length for s in chain),
            restore_ms=restore_ms,
            prefill_ms=prefill_ms,
        )
        return Tower(leaf=leaf, chain=chain, slot_id=slot_id, stats=stats)

    def warm_cache(self, trie: CacheTrie, slot_id: int = 0) -> dict[str, float]:
        """Pre-materialize every segment in the trie by walking BFS from
        root. Useful for setup before benchmarking.

        Returns a dict segment_id -> total_ms.
        """
        timings: dict[str, float] = {}
        # BFS: root first, then children
        visited: set[str] = set()
        queue: list[Segment] = [trie.root] if trie.root else []
        children_of: dict[str, list[Segment]] = {}
        for seg in trie.all_segments():
            if seg.parent is not None:
                children_of.setdefault(seg.parent.content_hash, []).append(seg)

        while queue:
            seg = queue.pop(0)
            if seg.content_hash in visited:
                continue
            visited.add(seg.content_hash)

            # Materialize this segment via assembly to its position
            tower = self.assemble(trie, seg, slot_id=slot_id)
            timings[seg.segment_id] = tower.stats.total_ms

            for child in children_of.get(seg.content_hash, []):
                queue.append(child)

        return timings
