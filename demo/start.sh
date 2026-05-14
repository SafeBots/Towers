#!/usr/bin/env bash
# demo/start.sh — Launch the whole demo with one command.
#
# Starts (or attaches to) three processes:
#   1. llama-server (the target model backend, port 8000)
#   2. populate.py  (generates sessions continuously)
#   3. scoreboard.py (live terminal display)
#
# With tmux installed: creates a 3-pane session named "towers"
#   - left:  llama-server logs (tail of log file)
#   - top-r: populate.py output
#   - bot-r: scoreboard.py
# Attach later with: tmux attach -t towers
# Detach with: Ctrl-b d
#
# Without tmux: backgrounds llama-server + populate with log files
#   under {cache_dir}/logs/, runs scoreboard in the foreground.
#
# Stop everything with: bash demo/stop.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

CACHE_DIR="${CACHE_DIR:-$HOME/towers_cache}"
MODEL_PATH="${MODEL_PATH:-$HOME/towers_models/qwen2.5-14b-instruct-q4_k_m.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
TARGET_URL="${TARGET_URL:-http://localhost:8000}"
N_PARALLEL="${N_PARALLEL:-4}"     # number of slots for parallel recall
SLOT_SAVE_PATH="$CACHE_DIR/bases"
LOG_DIR="$CACHE_DIR/logs"
PID_FILE="$CACHE_DIR/.towers.pids"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$CACHE_DIR" "$LOG_DIR" "$SLOT_SAVE_PATH"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

echo "Towers of Segments — demo launcher"
echo "  cache dir:    $CACHE_DIR"
echo "  model path:   $MODEL_PATH"
echo "  llama-server: $LLAMA_SERVER_BIN"
echo "  target URL:   $TARGET_URL"
echo "  slots:        $N_PARALLEL (for parallel_recall.py)"
echo

errors=0
if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "  ERROR: llama-server not found at $LLAMA_SERVER_BIN"
  echo "         Run demo/setup.sh first to build llama.cpp."
  errors=$((errors + 1))
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "  ERROR: model GGUF not found at $MODEL_PATH"
  echo "         Run demo/setup.sh first to download the model."
  errors=$((errors + 1))
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "  ERROR: python3 not found in PATH"
  errors=$((errors + 1))
fi

if [ $errors -gt 0 ]; then
  echo
  echo "Fix the errors above, then re-run."
  exit 1
fi

# Check if anything is already running
if [ -f "$PID_FILE" ]; then
  if read -r -a pids < "$PID_FILE"; then
    alive=0
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [ $alive -gt 0 ]; then
      echo "WARNING: $alive process(es) from a previous run still alive."
      echo "  PIDs: ${pids[*]}"
      echo "  Run: bash demo/stop.sh    to stop them first."
      exit 1
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Process launch
# ---------------------------------------------------------------------------

LLAMA_LOG="$LOG_DIR/llama-server.log"
POPULATE_LOG="$LOG_DIR/populate.log"

llama_cmd=(
  "$LLAMA_SERVER_BIN"
  -m "$MODEL_PATH"
  --host 0.0.0.0
  --port 8000
  --parallel "$N_PARALLEL"
  --slot-save-path "$SLOT_SAVE_PATH"
  --ctx-size 8192
  --batch-size 256
  --n-gpu-layers 99
)

populate_cmd=(
  python3 "$REPO_ROOT/demo/populate.py"
  --cache-dir "$CACHE_DIR"
  --target-url "$TARGET_URL"
  --tokens-per-session 4000
)

scoreboard_cmd=(
  python3 "$REPO_ROOT/demo/scoreboard.py"
  --cache-dir "$CACHE_DIR"
)

# ---------------------------------------------------------------------------
# tmux mode
# ---------------------------------------------------------------------------

start_tmux() {
  local session="towers"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists. Attaching..."
    tmux attach -t "$session"
    return
  fi

  echo "Starting tmux session '$session'..."

  # Window 1: llama-server (left pane), populate (top-right), scoreboard (bot-right)
  tmux new-session -d -s "$session" -n main \
    "${llama_cmd[@]} 2>&1 | tee '$LLAMA_LOG'"

  # Wait for the server to come up before starting populate
  tmux split-window -h -t "$session:main" \
    "echo 'Waiting for llama-server at $TARGET_URL...'; \
     for i in {1..60}; do \
       if curl -sf $TARGET_URL/health >/dev/null 2>&1; then \
         echo 'llama-server is up.'; break; \
       fi; sleep 2; \
     done; \
     ${populate_cmd[*]} 2>&1 | tee '$POPULATE_LOG'"

  tmux split-window -v -t "$session:main.1" \
    "echo 'Waiting for first session...'; \
     for i in {1..120}; do \
       if [ -f '$CACHE_DIR/index.json' ]; then break; fi; sleep 1; \
     done; \
     ${scoreboard_cmd[*]}"

  # Resize panes: 40% left, 60% right (split into top/bottom)
  tmux select-layout -t "$session:main" main-vertical
  tmux resize-pane -t "$session:main.0" -x 50

  echo
  echo "Session 'towers' started. Attaching..."
  echo "  Detach with Ctrl-b d. Re-attach with: tmux attach -t towers"
  echo "  Stop everything with: bash demo/stop.sh"
  echo
  sleep 1
  tmux attach -t "$session"
}

# ---------------------------------------------------------------------------
# Plain backgrounding mode (no tmux)
# ---------------------------------------------------------------------------

start_plain() {
  echo "tmux not found; using plain backgrounding with log files."
  echo

  echo "Starting llama-server (log: $LLAMA_LOG)..."
  nohup "${llama_cmd[@]}" >"$LLAMA_LOG" 2>&1 &
  LLAMA_PID=$!
  echo "  llama-server PID: $LLAMA_PID"
  echo "$LLAMA_PID" > "$PID_FILE"

  # Wait for server health
  echo -n "Waiting for llama-server "
  for i in {1..60}; do
    if curl -sf "$TARGET_URL/health" >/dev/null 2>&1; then
      echo " up."
      break
    fi
    echo -n "."
    sleep 2
  done

  if ! curl -sf "$TARGET_URL/health" >/dev/null 2>&1; then
    echo
    echo "ERROR: llama-server did not come up. Check $LLAMA_LOG"
    exit 1
  fi

  echo "Starting populate.py (log: $POPULATE_LOG)..."
  nohup "${populate_cmd[@]}" >"$POPULATE_LOG" 2>&1 &
  POPULATE_PID=$!
  echo "  populate.py PID: $POPULATE_PID"
  echo "$POPULATE_PID" >> "$PID_FILE"

  echo
  echo "Background processes started."
  echo "  Logs: tail -f $LOG_DIR/{llama-server,populate}.log"
  echo "  Stop: bash demo/stop.sh"
  echo
  echo "Starting scoreboard in foreground (Ctrl-C to exit scoreboard; others keep running)..."
  echo
  sleep 2
  exec "${scoreboard_cmd[@]}"
}

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if command -v tmux >/dev/null 2>&1 && [ -z "${NO_TMUX:-}" ]; then
  start_tmux
else
  start_plain
fi
