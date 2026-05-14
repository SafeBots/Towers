#!/usr/bin/env bash
# setup.sh — One-shot setup for the Towers demo on a MacBook
#
# This script downloads llama.cpp, builds it with Metal acceleration,
# downloads the Qwen2.5-14B-Q4 model GGUF, downloads the SmolLM2-135M
# encoder, sets up the cache directory, and prints the next-step
# command to start populate.py.
#
# Expected runtime: 10-30 minutes depending on download speed.
# Disk required: ~12 GB for models + llama.cpp build.
#
# Usage:
#   bash demo/setup.sh
#
# After this finishes:
#   1. Start llama-server in one terminal (the script prints the command)
#   2. Start populate.py in another terminal
#   3. Optionally start scoreboard.py in a third
#
set -euo pipefail

INSTALL_DIR="${TOWERS_DIR:-$HOME/towers_demo}"
CACHE_DIR="${TOWERS_CACHE_DIR:-$HOME/towers_cache}"
TARGET_MODEL_REPO="bartowski/Qwen2.5-14B-Instruct-GGUF"
TARGET_MODEL_FILE="Qwen2.5-14B-Instruct-Q4_K_M.gguf"
TARGET_MODEL_SIZE_MB=9000

echo "================================================================"
echo "Towers of Segments demo setup"
echo "================================================================"
echo "  Install dir: $INSTALL_DIR"
echo "  Cache dir:   $CACHE_DIR"
echo "  Target:      Qwen2.5-14B-Instruct-Q4_K_M (about $TARGET_MODEL_SIZE_MB MB)"
echo ""

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "WARNING: This script is designed for macOS. It may work on Linux"
    echo "but Metal acceleration won't apply."
    echo ""
fi

# Check available RAM (macOS specific)
if [[ "$(uname)" == "Darwin" ]]; then
    RAM_BYTES=$(sysctl -n hw.memsize)
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    echo "  Detected RAM: ${RAM_GB} GB"
    if [[ $RAM_GB -lt 24 ]]; then
        echo ""
        echo "WARNING: Qwen-14B-Q4 needs ~12 GB of RAM at runtime."
        echo "With $RAM_GB GB you may have issues. Consider Llama-3-8B instead."
        read -p "Continue with Qwen-14B-Q4 anyway? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy] ]]; then
            echo "Aborting. Edit setup.sh to use a smaller model."
            exit 1
        fi
    fi
fi

mkdir -p "$INSTALL_DIR"
mkdir -p "$CACHE_DIR/sessions"

# --- Step 1: clone and build llama.cpp ---
LLAMA_DIR="$INSTALL_DIR/llama.cpp"
if [[ -d "$LLAMA_DIR" ]]; then
    echo "llama.cpp already exists at $LLAMA_DIR (skipping clone)"
    cd "$LLAMA_DIR"
    git pull --quiet || true
else
    echo ""
    echo "[1/4] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
fi

echo ""
echo "[2/4] Building llama.cpp (Metal enabled by default on macOS)..."
cmake -B build -DBUILD_SHARED_LIBS=OFF > /dev/null
cmake --build build --config Release -j 4 --target llama-server llama-cli > /dev/null
echo "       Build OK."

# --- Step 3: download target model ---
MODEL_PATH="$INSTALL_DIR/models/$TARGET_MODEL_FILE"
mkdir -p "$INSTALL_DIR/models"

if [[ -f "$MODEL_PATH" ]]; then
    echo ""
    echo "[3/4] Target model already downloaded: $MODEL_PATH"
else
    echo ""
    echo "[3/4] Downloading $TARGET_MODEL_FILE (~$TARGET_MODEL_SIZE_MB MB)..."
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "  Installing huggingface-cli..."
        pip install --user huggingface_hub
    fi
    huggingface-cli download "$TARGET_MODEL_REPO" "$TARGET_MODEL_FILE" \
        --local-dir "$INSTALL_DIR/models"
fi

# --- Step 4: pre-download the encoder model ---
echo ""
echo "[4/4] Pre-caching the encoder model (SmolLM2-135M, ~270 MB)..."
python3 - <<'PYEOF'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(f"  Cached: {m.config.vocab_size} vocab, "
      f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M params")
PYEOF

# --- Done ---
echo ""
echo "================================================================"
echo "Setup complete."
echo "================================================================"
echo ""
echo "To launch everything with one command (tmux recommended):"
echo ""
echo "    export MODEL_PATH=$MODEL_PATH"
echo "    export LLAMA_SERVER_BIN=$LLAMA_DIR/build/bin/llama-server"
echo "    export CACHE_DIR=$CACHE_DIR"
echo "    bash demo/start.sh"
echo ""
echo "    Detach from tmux with Ctrl-b d; reattach with 'tmux attach -t towers'."
echo "    Stop everything with: bash demo/stop.sh"
echo ""
echo "If you'd rather drive the three processes manually, the equivalent"
echo "commands are:"
echo ""
echo "  TERMINAL 1 — llama-server:"
echo "    $LLAMA_DIR/build/bin/llama-server \\"
echo "        -m $MODEL_PATH \\"
echo "        -c 8192 --parallel 4 -ngl 999 --slots \\"
echo "        --slot-save-path $CACHE_DIR/bases"
echo ""
echo "  TERMINAL 2 — generator:"
echo "    python demo/populate.py --cache-dir $CACHE_DIR"
echo ""
echo "  TERMINAL 3 — scoreboard:"
echo "    python demo/scoreboard.py --cache-dir $CACHE_DIR"
echo ""
echo "Once you have a hundred or so sessions, the empirical headline:"
echo ""
echo "    python benchmarks/amortization.py --cache-dir $CACHE_DIR"
echo ""
echo "And to recall a session interactively:"
echo ""
echo "    python demo/recall.py --cache-dir $CACHE_DIR"
echo ""
