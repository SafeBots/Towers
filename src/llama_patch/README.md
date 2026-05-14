# llama.cpp patches (future work)

This directory will hold patches against llama.cpp that the paper proposes
but that are not required for v0.1 of this repo. The v0.1 demos work
against stock llama-server via the existing slot save/restore API.

## Planned patches

### `rope_shift.patch` (paper §5.2)

Apply RoPE rotation deltas when slot-restoring K vectors at a non-original
position. Current llama.cpp slot save/restore assumes slots restore at
the same position they were saved. The paper proposes restoring a segment
into different position offsets so that the same cached state can be
reused at different positions within a tower (e.g., the same community
base appearing after platform bases of different sizes).

Difficulty: ~100-200 lines of C++. Mostly a math change in the K-vector
restoration path; the V vectors are position-independent and need no
change. Needs unit testing against monolithic prefill at the same
position for correctness.

### `per_segment_quant.patch` (paper §6)

Allow per-segment quantization levels rather than the model-wide
quantization llama.cpp uses today. Segments at the base of the
hierarchy (high `level` value in the cache trie) get more aggressive
quantization (Q4); recent dynamic segments stay at Q8.

Difficulty: moderate. The attention kernel needs to handle mixed-quant
K and V tensors. Easiest with an extra dequantize step in the prefill
path, accepting some performance loss for the v0.2 release.

### `slot_metadata.patch` (paper §3)

Add a metadata blob to each slot save file recording the segment's
level, parent hash, source position, and bit-width. This lets the
runner verify byte-stability across restarts and detect cache trie
corruption.

Difficulty: small. Append a metadata struct to the slot file format,
preserve backward compatibility with un-tagged saves.

## Why these aren't in v0.1

v0.1 is the research validation release. The headline numbers
(compression ratio, end-to-end multi-tenant feasibility) all work with
stock llama.cpp because the paper's storage-compression result depends
on the AC codec and the cache trie structure, not on KV-state-level
quantization details. The patches above are quality-of-life and
performance optimizations for v0.2+.

If you want to contribute a patch, open an issue first describing the
intended approach. We'll review against the paper's correctness
requirements (depth-adaptive error bound, byte-stability) before
merging.
