"""scoreboard.py — Live terminal scoreboard with sparklines and bars.

This is the screen that goes on the demo display during the podcast.
Updates every couple seconds, shows the cache growing in real time:

    - Sessions and disk bytes ticking up
    - Sparklines for sessions/min and disk used (last ~60 datapoints)
    - Bar chart showing your-storage vs equivalent-raw-KV
    - The headline ratios in bold against several target models

History is tracked across runs in {cache_dir}/scoreboard_history.json
so sparklines retain their data when you restart the scoreboard.

Run alongside populate.py:

    python demo/scoreboard.py                          # default 2s refresh
    python demo/scoreboard.py --refresh-secs 1.0       # faster updates
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tower_runner.tower_store import TowerStore


# ANSI styling
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


TARGET_KV_BYTES_PER_TOKEN = [
    ("Qwen2.5-14B  (GQA)",  192_000),
    ("Llama-3-8B   (GQA)",  131_000),
    ("Llama-3-70B  (GQA)",  320_000),
    ("Llama-3-405B (GQA)",  504_000),
]


# ----------------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------------

SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def human_bytes(n: float, decimals: int = 2) -> str:
    for unit, lim in [("B", 1024), ("KB", 1024**2), ("MB", 1024**3),
                      ("GB", 1024**4), ("TB", 1024**5)]:
        if abs(n) < lim:
            return f"{n / (lim/1024):.{decimals}f} {unit}"
    return f"{n / 1024**5:.{decimals}f} PB"


def human_count(n: float) -> str:
    """Format a count: 1234 -> '1,234'; 1234567 -> '1.23M'."""
    if n < 10_000:
        return f"{n:,.0f}"
    if n < 1_000_000:
        return f"{n/1000:.1f}K"
    if n < 1_000_000_000:
        return f"{n/1_000_000:.2f}M"
    return f"{n/1_000_000_000:.2f}B"


def sparkline(values: list, width: int = 30) -> str:
    """Render a list of numeric values as a Unicode sparkline."""
    if not values:
        return " " * width
    # Downsample if too long
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(len(values)-1, int(i * step))] for i in range(width)]
    else:
        sampled = list(values) + [None] * (width - len(values))
    # Compute scaling over non-None values
    real = [v for v in sampled if v is not None]
    if not real:
        return " " * width
    vmin, vmax = min(real), max(real)
    span = max(vmax - vmin, 1e-9)
    out = []
    for v in sampled:
        if v is None:
            out.append(" ")
        else:
            idx = min(8, max(0, int((v - vmin) / span * 8)))
            out.append(SPARK_BLOCKS[idx])
    return "".join(out)


def comparison_bar(small_bytes: float, big_bytes: float, label_small: str,
                   label_big: str, width: int = 60) -> str:
    """Two-line proportional bar showing the disparity. Because the
    ratio is enormous (~80,000×), the small bar is shown at fixed
    minimum width with its size labeled; the big bar fills the full
    width.
    """
    if big_bytes <= 0:
        return ""
    # Small bar: at least 1 char, but show its true proportion
    proportion = small_bytes / big_bytes if big_bytes > 0 else 0
    if proportion < 1 / width:
        small_w = 1
        prefix_small = DIM + "▏" + RESET
    else:
        small_w = max(1, int(width * proportion))
        prefix_small = GREEN + ("▮" * small_w) + RESET
    big_w = width
    prefix_big = MAGENTA + ("▮" * big_w) + RESET

    ratio = big_bytes / small_bytes if small_bytes > 0 else 0
    s = []
    s.append(f"  {label_small:<32} {prefix_small}{' ' * (width - small_w)}  "
             f"{human_bytes(small_bytes):>10}")
    s.append(f"  {label_big:<32} {prefix_big}  "
             f"{BOLD}{human_bytes(big_bytes):>10}{RESET}  "
             f"{BOLD}{ratio:>10,.0f}x{RESET}")
    return "\n".join(s)


# ----------------------------------------------------------------------------
# History tracking
# ----------------------------------------------------------------------------

class History:
    """Tracks recent metrics for sparklines. Persists to disk so sparklines
    survive scoreboard restarts."""

    def __init__(self, path: Path, max_points: int = 120):
        self.path = path
        self.max_points = max_points
        self.records: list[dict] = []
        if path.exists():
            try:
                self.records = json.loads(path.read_text())
                if len(self.records) > max_points:
                    self.records = self.records[-max_points:]
            except Exception:
                self.records = []

    def add(self, timestamp: float, n_sessions: int, disk_bytes: float,
            equiv_kv_bytes_70b: float) -> None:
        self.records.append({
            "t": timestamp,
            "n": n_sessions,
            "disk": disk_bytes,
            "kv70b": equiv_kv_bytes_70b,
        })
        if len(self.records) > self.max_points:
            self.records = self.records[-self.max_points:]
        try:
            self.path.write_text(json.dumps(self.records))
        except Exception:
            pass

    def sessions_series(self) -> list:
        return [r["n"] for r in self.records]

    def rate_series(self) -> list:
        """Sessions per minute, computed from deltas."""
        rates = []
        for i in range(1, len(self.records)):
            dn = self.records[i]["n"] - self.records[i-1]["n"]
            dt = self.records[i]["t"] - self.records[i-1]["t"]
            if dt > 0:
                rates.append(dn / dt * 60.0)
        return rates

    def disk_series(self) -> list:
        return [r["disk"] for r in self.records]

    def kv_series(self) -> list:
        return [r["kv70b"] for r in self.records]


# ----------------------------------------------------------------------------
# Frame rendering
# ----------------------------------------------------------------------------

def get_disk_stats(path: Path) -> tuple[int, int]:
    stat = os.statvfs(str(path))
    return stat.f_bavail * stat.f_frsize, stat.f_blocks * stat.f_frsize


def render_frame(store: TowerStore, history: History,
                 last_count: int, last_time: float
                 ) -> tuple[int, float]:
    now = time.time()
    s = store.stats
    n = s.n_sessions

    # Total disk used (estimate including overhead)
    total_disk = (
        s.total_dynamic_token_bytes
        + s.total_dynamic_compressed_bytes
        + s.total_base_token_bytes
        + n * 500
    )

    # Rate
    if last_time and now - last_time > 0:
        rate = (n - last_count) / (now - last_time) * 60.0
    else:
        rate = 0.0

    # Per-session averages
    avg_dyn_bytes = s.total_dynamic_token_bytes / max(n, 1)
    avg_compressed = s.total_dynamic_compressed_bytes / max(n, 1)
    avg_base_bytes_amort = s.total_base_token_bytes / max(n, 1)
    avg_per_session = avg_dyn_bytes + avg_base_bytes_amort
    avg_compressed_per_session = avg_compressed + avg_base_bytes_amort

    avg_dyn_tokens = avg_dyn_bytes / 4
    if s.n_base_segments > 0:
        avg_base_chain_tokens = (s.total_base_token_bytes / s.n_base_segments) / 4
    else:
        avg_base_chain_tokens = 0
    avg_total_tokens = avg_dyn_tokens + avg_base_chain_tokens

    # Equivalent raw KV bytes for several targets
    kv_equiv = {}
    for label, bpt in TARGET_KV_BYTES_PER_TOKEN:
        kv_equiv[label] = n * avg_total_tokens * bpt

    # Disk stats
    disk_free, disk_total = get_disk_stats(store.cache_dir)
    if avg_per_session > 0:
        proj_total = int(disk_total / avg_per_session)
        proj_free = int(disk_free / avg_per_session)
    else:
        proj_total = proj_free = 0

    # Record history
    history.add(now, n, total_disk, kv_equiv["Llama-3-70B  (GQA)"])

    # ----- render -----
    lines = []
    lines.append(CLEAR)
    lines.append(f"{BOLD}Towers of Segments — live cache scoreboard{RESET}")
    lines.append("=" * 78)
    lines.append(f"  {DIM}Updated {datetime.now().isoformat(timespec='seconds')}    "
                 f"Cache: {store.cache_dir}{RESET}")
    lines.append("")

    # Big numbers panel
    lines.append(f"  {BOLD}Sessions stored:{RESET}  {BOLD}{CYAN}{n:>10,}{RESET}    "
                 f"Base segments: {s.n_base_segments}    "
                 f"Total on disk: {BOLD}{human_bytes(total_disk)}{RESET}")
    rate_str = f"{rate:.1f}/min" if rate > 0 else "starting..."
    lines.append(f"  Generation rate:  {BOLD}{GREEN}{rate_str:>10}{RESET}    "
                 f"({rate*60:.0f}/hr, {rate*60*24:.0f}/day at current pace)")
    lines.append("")

    # Sparklines
    spark_sessions = sparkline(history.sessions_series(), width=50)
    spark_rate = sparkline(history.rate_series(), width=50)
    lines.append(f"  {DIM}Last {len(history.records)} frames:{RESET}")
    lines.append(f"  Sessions:   {CYAN}{spark_sessions}{RESET}  "
                 f"{DIM}(0 to {human_count(max(history.sessions_series()) if history.sessions_series() else 0)}){RESET}")
    rate_series = history.rate_series()
    if rate_series:
        lines.append(f"  Rate/min:   {GREEN}{spark_rate}{RESET}  "
                     f"{DIM}({min(rate_series):.1f} to {max(rate_series):.1f}){RESET}")
    lines.append("")

    # Storage breakdown
    lines.append(f"  {BOLD}Storage breakdown:{RESET}")
    lines.append(f"    Dynamic tokens (raw int32):  {human_bytes(s.total_dynamic_token_bytes):>14}")
    lines.append(f"    AC-compressed dynamic:        {human_bytes(s.total_dynamic_compressed_bytes):>14}")
    lines.append(f"    Base segments (SHARED):       {human_bytes(s.total_base_token_bytes):>14}  "
                 f"{DIM}({s.n_base_segments} segs / {n:,} sessions = "
                 f"{human_bytes(avg_base_bytes_amort)} per session){RESET}")
    lines.append("")

    # Per-session amortized
    lines.append(f"  {BOLD}Per-session amortized cost:{RESET}")
    lines.append(f"    Raw tokens mode:    {BOLD}{human_bytes(avg_per_session):>10}{RESET}    "
                 f"({avg_dyn_tokens:,.0f} tokens / session)")
    lines.append(f"    AC-compressed mode: {BOLD}{human_bytes(avg_compressed_per_session):>10}{RESET}    "
                 f"({avg_compressed:,.0f} compressed bytes / session)")
    lines.append("")

    # The comparison bar — the visual moment
    if n > 0:
        lines.append(f"  {BOLD}Your cache vs equivalent raw FP16 KV cache (Llama-3-70B):{RESET}")
        bar = comparison_bar(
            small_bytes=total_disk,
            big_bytes=kv_equiv["Llama-3-70B  (GQA)"],
            label_small="Your cache (real disk):",
            label_big="Would need (raw FP16 KV):",
            width=50,
        )
        lines.append(bar)
        lines.append("")

    # Ratio table
    if n > 0:
        lines.append(f"  {BOLD}Empirical compression ratios (measured, not projected):{RESET}")
        for label, bpt in TARGET_KV_BYTES_PER_TOKEN:
            raw_kv = avg_total_tokens * bpt
            r_raw = raw_kv / avg_per_session if avg_per_session > 0 else 0
            r_comp = raw_kv / avg_compressed_per_session if avg_compressed_per_session > 0 else 0
            highlight = BOLD + YELLOW if "70B" in label else ""
            lines.append(f"    {highlight}vs {label:<22}{RESET if highlight else ''}   "
                         f"raw-tokens: {highlight}{r_raw:>10,.0f}x{RESET if highlight else ''}    "
                         f"AC-comp: {highlight}{r_comp:>10,.0f}x{RESET if highlight else ''}")
        lines.append("")

    # Disk projection
    lines.append(f"  {DIM}Disk: {human_bytes(disk_free)} free of {human_bytes(disk_total)} total. "
                 f"At current per-session size, this drive holds "
                 f"{proj_total:,} sessions ({proj_free:,} in free space).{RESET}")
    lines.append("")
    lines.append(f"  {DIM}Press Ctrl-C to stop scoreboard. populate.py keeps running.{RESET}")

    print("\n".join(lines))

    return n, now


def main():
    parser = argparse.ArgumentParser(description="Live cache scoreboard with charts")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "towers_cache")
    parser.add_argument("--refresh-secs", type=float, default=2.0)
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI styling")
    args = parser.parse_args()

    if args.no_color:
        global BOLD, DIM, GREEN, YELLOW, CYAN, MAGENTA, RESET
        BOLD = DIM = GREEN = YELLOW = CYAN = MAGENTA = RESET = ""

    if not args.cache_dir.exists():
        print(f"Cache dir {args.cache_dir} does not exist.")
        print(f"Start populate.py first, or use --cache-dir to point elsewhere.")
        return

    history = History(args.cache_dir / "scoreboard_history.json")

    last_count = 0
    last_time = 0.0
    try:
        while True:
            store = TowerStore(args.cache_dir, target_url="http://localhost:65535")
            last_count, last_time = render_frame(
                store, history, last_count, last_time
            )
            time.sleep(args.refresh_secs)
    except KeyboardInterrupt:
        print("\nScoreboard stopped. populate.py continues unaffected.")


if __name__ == "__main__":
    main()
