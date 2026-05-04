#!/usr/bin/env bash
# One-command bootstrap + launch for youtube-to-markdown.
#
# Idempotent: safe to run on every start. Steps that already passed are skipped.
#
# 1. Check ffmpeg/ffprobe (can't safely auto-install system tools)
# 2. Install uv if missing (with confirmation prompt)
# 3. uv sync (installs Python deps; no-op if already in sync)
# 4. yt2md doctor (verifies the rest — Node 20+ for yt-dlp, ANTHROPIC_API_KEY,
#    YouTube cookies, etc.). If anything blocks, print its output and bail.
# 5. yt2md serve (launches the local web reader on http://localhost:7682)
#
# Usage:  ./run.sh

set -euo pipefail

cd "$(dirname "$0")"

bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }

bold "==> yt2md: bootstrap + launch"

# ---- 1. ffmpeg / ffprobe ----
missing_system=()
for tool in ffmpeg ffprobe; do
    command -v "$tool" >/dev/null 2>&1 || missing_system+=("$tool")
done
if [ ${#missing_system[@]} -gt 0 ]; then
    red "Missing system tools: ${missing_system[*]}"
    case "$(uname -s)" in
        Darwin) echo "  Install with: brew install ffmpeg" ;;
        Linux)  echo "  Install with: sudo apt install ffmpeg   (Debian/Ubuntu)"
                echo "             or: sudo dnf install ffmpeg   (Fedora)" ;;
        *)      echo "  Install from https://ffmpeg.org/download.html" ;;
    esac
    exit 1
fi
green "✓ ffmpeg + ffprobe"

# ---- 2. uv ----
if ! command -v uv >/dev/null 2>&1; then
    yellow "uv (Python toolchain) not found."
    read -r -p "Install uv now via the official installer? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS])
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            ;;
        *)
            red "Install uv yourself, then re-run ./run.sh"
            echo "  See https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
            ;;
    esac
fi
green "✓ uv ($(uv --version | awk '{print $2}'))"

# ---- 3. Python deps ----
bold "==> Syncing Python deps (uv sync — fast no-op if already in sync)"
uv sync

# ---- 4. Doctor (Node, API key, cookies, settings sanity) ----
bold "==> Running doctor"
if ! uv run yt2md doctor; then
    red "Doctor flagged blocking issues. Fix the items above, then re-run ./run.sh."
    echo
    echo "Common fixes:"
    echo "  • Node 20+ missing: brew install deno  (or  brew install node@20)"
    echo "  • ANTHROPIC_API_KEY missing: yt2md prompts on first interactive run,"
    echo "    or set it in ~/yt2md/.env directly:"
    echo "      echo 'ANTHROPIC_API_KEY=sk-ant-…' >> ~/yt2md/.env"
    exit 1
fi

# ---- 5. Launch the web reader ----
bold "==> Launching reader on http://localhost:7682/"
exec uv run yt2md serve
