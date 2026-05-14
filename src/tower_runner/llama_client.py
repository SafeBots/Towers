"""Thin HTTP wrapper around llama-server's slot and completion endpoints.

Requires llama-server started with `--slots --slot-save-path <dir>` flags.

The slot save/restore API:
    POST /slots/{id}?action=save
    POST /slots/{id}?action=restore
    POST /slots/{id}?action=erase

These let us persist a slot's KV cache to disk and restore it to a slot,
which is exactly the storage primitive Towers needs. We build on this
without any C++ patches.

Documentation:
    https://github.com/ggml-org/llama.cpp/tree/master/tools/server
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # Caller will get a clear error when they actually use it


@dataclass
class SlotSaveResult:
    """Result of POST /slots/{id}?action=save."""
    id_slot: int
    filename: str
    n_saved: int      # tokens saved
    n_written: int    # bytes written
    save_ms: float    # time taken


@dataclass
class SlotRestoreResult:
    """Result of POST /slots/{id}?action=restore."""
    id_slot: int
    filename: str
    n_restored: int
    n_read: int
    restore_ms: float


@dataclass
class CompletionResult:
    """Result of POST /completion or /v1/completions."""
    content: str
    tokens_predicted: int
    prompt_n: int           # length of prompt that was prefilled
    prompt_ms: float        # prefill time (ms)
    predict_ms: float       # generation time (ms)
    cache_n: int            # how many tokens came from cache


class LlamaServerClient:
    """Synchronous HTTP client for llama-server.

    Example:
        client = LlamaServerClient(base_url='http://localhost:8000')
        client.health()
        result = client.completion('Hello, world', slot_id=0, n_predict=50)
        print(result.content)
        save = client.slot_save(slot_id=0, filename='session_42.bin')
        print(f'Saved {save.n_saved} tokens in {save.save_ms:.1f}ms')
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 600.0):
        if requests is None:
            raise ImportError(
                "The `requests` package is required for LlamaServerClient. "
                "Install it with: pip install requests"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ----- Health / model info -----

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def props(self) -> dict:
        """Return server properties including n_ctx, total_slots, chat_template."""
        r = requests.get(f"{self.base_url}/props", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def slots(self) -> list[dict]:
        """List current slot status (requires --slots and --metrics)."""
        r = requests.get(f"{self.base_url}/slots", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ----- Tokenization -----

    def tokenize(self, content: str, add_special: bool = True) -> list[int]:
        """Convert text to token ids."""
        r = requests.post(
            f"{self.base_url}/tokenize",
            json={"content": content, "add_special": add_special},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["tokens"]

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token ids back to text."""
        r = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["content"]

    # ----- Slot save / restore (the cache primitives Towers uses) -----

    def slot_save(self, slot_id: int, filename: str) -> SlotSaveResult:
        """Persist a slot's KV cache to disk."""
        r = requests.post(
            f"{self.base_url}/slots/{slot_id}",
            params={"action": "save"},
            json={"filename": filename},
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()
        return SlotSaveResult(
            id_slot=d["id_slot"],
            filename=d["filename"],
            n_saved=d["n_saved"],
            n_written=d["n_written"],
            save_ms=d.get("timings", {}).get("save_ms", 0.0),
        )

    def slot_restore(self, slot_id: int, filename: str) -> SlotRestoreResult:
        """Restore a previously-saved KV cache into a slot."""
        r = requests.post(
            f"{self.base_url}/slots/{slot_id}",
            params={"action": "restore"},
            json={"filename": filename},
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()
        return SlotRestoreResult(
            id_slot=d["id_slot"],
            filename=d["filename"],
            n_restored=d["n_restored"],
            n_read=d["n_read"],
            restore_ms=d.get("timings", {}).get("restore_ms", 0.0),
        )

    def slot_erase(self, slot_id: int) -> dict:
        """Clear a slot's KV cache."""
        r = requests.post(
            f"{self.base_url}/slots/{slot_id}",
            params={"action": "erase"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ----- Completion -----

    def completion(
        self,
        prompt: str | list[int],
        slot_id: Optional[int] = None,
        n_predict: int = 128,
        temperature: float = 0.0,
        cache_prompt: bool = True,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> CompletionResult:
        """Run a completion. If slot_id is given, uses that specific slot.

        cache_prompt=True is critical for tower assembly: it tells llama-server
        to reuse any matching prefix already in the slot's KV cache rather
        than re-prefilling from scratch.
        """
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": cache_prompt,
        }
        if slot_id is not None:
            body["id_slot"] = slot_id
        if stop is not None:
            body["stop"] = stop
        body.update(kwargs)

        r = requests.post(
            f"{self.base_url}/completion",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()
        timings = d.get("timings", {})
        return CompletionResult(
            content=d.get("content", ""),
            tokens_predicted=d.get("tokens_predicted", 0),
            prompt_n=timings.get("prompt_n", 0),
            prompt_ms=timings.get("prompt_ms", 0.0),
            predict_ms=timings.get("predict_ms", 0.0),
            cache_n=d.get("tokens_cached", 0),
        )

    # ----- Convenience: time a prefill-only operation -----

    def prefill_only(
        self,
        prompt: str | list[int],
        slot_id: Optional[int] = None,
        cache_prompt: bool = True,
    ) -> CompletionResult:
        """Prefill without generating any new tokens. Useful for measuring
        the time to materialize a tower.
        """
        return self.completion(
            prompt=prompt,
            slot_id=slot_id,
            n_predict=0,
            temperature=0.0,
            cache_prompt=cache_prompt,
        )

    # ----- Health-check helper for examples -----

    def wait_until_ready(self, max_wait_s: float = 30.0) -> bool:
        """Poll /health until the server responds 'ok'."""
        start = time.time()
        while time.time() - start < max_wait_s:
            try:
                h = self.health()
                if h.get("status") == "ok":
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False
