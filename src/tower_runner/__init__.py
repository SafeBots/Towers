"""Towers of Segments: reference implementation accompanying the paper.

See README.md and paper/Towers.pdf for details.
"""

__version__ = "0.1.0"

# Modules are imported lazily by examples/benchmarks to keep the package
# usable even when optional deps (torch/transformers) aren't installed.

__all__ = [
    "cache_trie",
    "tower",
    "ac_codec",
    "llama_client",
]
