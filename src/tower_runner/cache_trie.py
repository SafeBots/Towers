"""Cache trie data structure for hierarchical multi-tenant KV cache.

This file implements the cache trie of paper §3 ("Formal Model") and the
structural amortization storage accounting of paper §4 (Theorem 4.1).

Mapping from paper to code:

    Paper:  T_C is a rooted directed tree whose nodes are segments sigma_i
            and whose edges encode the extension relation. A *tower* for a
            session is the unique root-to-leaf path in T_C corresponding to
            that session's full prompt.

    Code:   CacheTrie holds the tree; Segment is a single node. tower_for(leaf)
            returns the chain of segments from root to leaf. Segments are
            content-addressed by SHA256 of (parent_hash || tokens), so
            identical segments map to the same node (this is the structural
            deduplication that makes amortization work).

The four-level hierarchy used throughout the paper:

    Root: platform base B_perm
        |
        +-- Community A base (B_sess for A)
        |       |
        |       +-- Bot 1 base (B_bot,1)
        |       |       +-- Session 1 dynamic (B_dyn for sess 1)
        |       |       +-- Session 2 dynamic (B_dyn for sess 2)
        |       +-- Bot 2 base
        |               +-- Session 3 dynamic
        |
        +-- Community B base
                +-- ...

Why this matters (Theorem 4.1 of the paper):

    Per-session storage in a deployment of size N converges from O(L)
    (the naive per-session cost, where every tower stores its full
    prefix) toward |B_dyn| (the dynamic block only) at rate O(1/N).
    The intuition: as the deployment grows, more sessions share the
    same platform/community/bot bases, and those bases are stored
    once per deployment rather than once per session.

    For the paper's headline scenario (70B model, 4000-token sessions):
    naive storage is 1.28 GB/session; with cache trie amortization at
    deployment scale, it approaches 0.8 GB/session for the dynamic
    block. That's a factor of ~1.6x from structural amortization alone,
    before any compression. The big gains come when you stack arithmetic
    coding (paper §10) on top.

Byte-stability prerequisite (paper §2.4):

    The structural amortization works because the segment-construction
    pipeline is deterministic: the same tokens at the same path produce
    the same KV state every time. The paper formalizes this as
    byte-stability of M, the language model. In code, content_hash
    enforces this by content-addressing each segment so that two
    independently-constructed segments with the same tokens and parent
    collapse to the same node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Optional


class SegmentLevel(Enum):
    """The 4-level hierarchy from the paper.

    PLATFORM: the deployment-wide base (B_perm). Shared across all communities.
    COMMUNITY: a community-level prompt (B_sess). Shared across all bots in a community.
    BOT: a bot-specific prompt (B_bot). Shared across all sessions for that bot.
    DYNAMIC: per-session conversational content (B_dyn). Not shared.
    """
    PLATFORM = 0
    COMMUNITY = 1
    BOT = 2
    DYNAMIC = 3


@dataclass
class Segment:
    """A single segment in the cache trie.

    Corresponds to the paper's Definition 3.1 of a segment:
    sigma = (tau_sigma, pi_sigma, KV_sigma) where tau is the token sequence,
    pi is the source position at which it was processed, and KV is the
    cached state. Here we represent tokens as a list and store the slot
    filename rather than the raw KV state (llama.cpp manages that).

    Attributes:
        segment_id: stable identifier (typically derived from content hash)
        tokens: the token sequence as a list of ints
        level: position in the hierarchy
        parent: the parent segment, or None for the platform base
        source_position: the position offset at which this segment was originally
            prefilled (paper's pi_sigma). For root segments this is 0.
        slot_file: filename under llama-server's --slot-save-path where the
            KV state is persisted. None if not yet materialized.
        compressed_bytes: arithmetic-coded bitstring (paper §10).
            None if not yet compressed.
        access_count: how many times this segment has been used in a tower
            assembly. Used for hot/warm/cold tier decisions.
    """
    segment_id: str
    tokens: list[int]
    level: SegmentLevel
    parent: Optional[Segment] = None
    source_position: int = 0
    slot_file: Optional[str] = None
    compressed_bytes: Optional[bytes] = None
    access_count: int = 0

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def content_hash(self) -> str:
        """SHA256 of (parent_hash || tokens). Used to verify byte-stability
        of the segment-construction pipeline.

        Two segments with the same parent and same tokens have the same hash
        and represent the same node in the cache trie.
        """
        h = sha256()
        if self.parent is not None:
            h.update(self.parent.content_hash.encode())
        h.update(b"|")
        # Serialize tokens deterministically
        for t in self.tokens:
            h.update(t.to_bytes(4, "little"))
        return h.hexdigest()

    def __repr__(self) -> str:
        return (
            f"Segment(id={self.segment_id!r}, level={self.level.name}, "
            f"n_tokens={self.length}, parent={self.parent.segment_id if self.parent else None!r})"
        )


@dataclass
class CacheTrie:
    """The cache trie data structure.

    Maintains the segment hierarchy and supports the operations from the paper:
        - extend(parent, tokens, level): add a child segment
        - tower_for(leaf): walk from root to a leaf and return the chain
        - depth(segment): hierarchy depth
        - all_segments(): iterate all known segments
        - storage_bytes(compression='raw'): compute total storage under a
          compression model (raw/quantized/arithmetic)
    """
    root: Optional[Segment] = None
    by_id: dict[str, Segment] = field(default_factory=dict)
    by_content_hash: dict[str, Segment] = field(default_factory=dict)

    def add(self, segment: Segment) -> Segment:
        """Add a segment to the trie. If an equivalent segment (same
        content_hash) already exists, return that existing segment instead
        (structural deduplication).
        """
        existing = self.by_content_hash.get(segment.content_hash)
        if existing is not None:
            return existing

        if segment.parent is None:
            if self.root is not None and self.root.content_hash != segment.content_hash:
                raise ValueError(
                    f"Cache trie already has a different root "
                    f"({self.root.segment_id!r}); cannot add second root "
                    f"({segment.segment_id!r})"
                )
            self.root = segment

        self.by_id[segment.segment_id] = segment
        self.by_content_hash[segment.content_hash] = segment
        return segment

    def extend(
        self,
        parent: Segment,
        tokens: list[int],
        level: SegmentLevel,
        segment_id: Optional[str] = None,
    ) -> Segment:
        """Create a child segment extending the parent.

        The source_position of the child is parent.source_position + parent.length.
        If a segment with the same content_hash already exists, returns it
        instead (deduplication).
        """
        if segment_id is None:
            # Auto-derive a short id from content
            h = sha256()
            if parent is not None:
                h.update(parent.content_hash.encode())
            for t in tokens:
                h.update(t.to_bytes(4, "little"))
            segment_id = f"seg_{h.hexdigest()[:8]}"

        child = Segment(
            segment_id=segment_id,
            tokens=tokens,
            level=level,
            parent=parent,
            source_position=(parent.source_position + parent.length) if parent else 0,
        )
        return self.add(child)

    def tower_for(self, leaf: Segment) -> list[Segment]:
        """Walk from root to leaf, returning the chain of segments that
        compose this session's tower (paper Definition 3.2).
        """
        chain: list[Segment] = []
        cur: Optional[Segment] = leaf
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def depth(self, segment: Segment) -> int:
        d = 0
        cur = segment
        while cur.parent is not None:
            d += 1
            cur = cur.parent
        return d

    def all_segments(self):
        return self.by_content_hash.values()

    def total_tokens_naive(self, leaves: list[Segment]) -> int:
        """The naive total tokens if every session were stored independently
        (no structural sharing). Sum of |tower| for each leaf.
        """
        return sum(sum(s.length for s in self.tower_for(leaf)) for leaf in leaves)

    def total_tokens_shared(self) -> int:
        """The deduplicated total tokens stored under structural amortization
        (paper Theorem 4.1). Each segment counted once regardless of how many
        towers reference it.
        """
        return sum(s.length for s in self.all_segments())

    def amortization_ratio(self, leaves: list[Segment]) -> float:
        """Ratio of naive storage to amortized storage. As deployment size grows
        (number of leaves with shared bases), this approaches |B_dyn|/L per
        Theorem 4.1.
        """
        naive = self.total_tokens_naive(leaves)
        shared = self.total_tokens_shared()
        if shared == 0:
            return 0.0
        return naive / shared


# Helpful constructors for common deployment shapes

def make_simple_deployment(
    platform_tokens: list[int],
    community_prompts: dict[str, list[int]],
    bots_per_community: dict[str, dict[str, list[int]]],
    sessions: dict[str, list[int]],
) -> tuple[CacheTrie, list[Segment]]:
    """Build a cache trie representing a multi-tenant deployment.

    Args:
        platform_tokens: the platform-base prompt tokens
        community_prompts: dict mapping community_id -> community-base tokens
        bots_per_community: dict community_id -> {bot_id -> bot-base tokens}
        sessions: dict session_id -> dynamic-content tokens. Session IDs must
            be formatted as "{community_id}/{bot_id}/{session_id}".

    Returns:
        (trie, list_of_session_leaf_segments)
    """
    trie = CacheTrie()
    platform = trie.extend(None, platform_tokens, SegmentLevel.PLATFORM,
                            segment_id="platform")
    trie.add(platform)
    trie.root = platform

    community_segs: dict[str, Segment] = {}
    for cid, ctoks in community_prompts.items():
        community_segs[cid] = trie.extend(
            platform, ctoks, SegmentLevel.COMMUNITY, segment_id=f"community/{cid}"
        )

    bot_segs: dict[tuple[str, str], Segment] = {}
    for cid, bots in bots_per_community.items():
        for bid, btoks in bots.items():
            bot_segs[(cid, bid)] = trie.extend(
                community_segs[cid], btoks, SegmentLevel.BOT,
                segment_id=f"bot/{cid}/{bid}",
            )

    session_leaves: list[Segment] = []
    for sid, stoks in sessions.items():
        cid, bid, sname = sid.split("/", 2)
        leaf = trie.extend(
            bot_segs[(cid, bid)], stoks, SegmentLevel.DYNAMIC,
            segment_id=f"session/{sid}",
        )
        session_leaves.append(leaf)

    return trie, session_leaves
