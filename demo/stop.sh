#!/usr/bin/env bash
# demo/stop.sh — Stop all demo processes started by demo/start.sh.
#
# Tries graceful SIGTERM first, then SIGKILL after a few seconds.
# Also kills the tmux session if one exists.

set -uo pipefail

CACHE_DIR="${CACHE_DIR:-$HOME/towers_cache}"
PID_FILE="$CACHE_DIR/.towers.pids"
TMUX_SESSION="towers"

echo "Towers of Segments — stopping demo processes"

# Kill tmux session if present
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "  Killing tmux session '$TMUX_SESSION'..."
    tmux kill-session -t "$TMUX_SESSION"
  fi
fi

# Kill PIDs from pidfile
if [ -f "$PID_FILE" ]; then
  while read -r pid; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "  Sending SIGTERM to PID $pid..."
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done < "$PID_FILE"

  # Wait up to 8 seconds for graceful shutdown
  sleep 3
  while read -r pid; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "  Sending SIGKILL to PID $pid..."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done < "$PID_FILE"

  rm -f "$PID_FILE"
fi

# Best-effort cleanup of any leftover llama-server / populate processes
for proc in llama-server "populate.py"; do
  pids=$(pgrep -f "$proc" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  Found stray $proc process(es): $pids — killing..."
    kill -TERM $pids 2>/dev/null || true
    sleep 1
    kill -KILL $pids 2>/dev/null || true
  fi
done

echo "Stopped."
