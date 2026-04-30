#!/usr/bin/env bash
# Onboarding script for video-to-deck.
# Checks for required tools (uv, ffmpeg, ffprobe), installs uv if absent,
# then syncs Python dependencies into a local .venv via uv.
#
# Usage:  ./setup.sh

set -euo pipefail

cd "$(dirname "$0")"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }

bold "==> video-to-deck setup"

# ---- 1. ffmpeg / ffprobe (system dep, can't safely auto-install) ----
missing_system=()
for tool in ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        missing_system+=("$tool")
    fi
done

if [ ${#missing_system[@]} -gt 0 ]; then
    red "Missing system tools: ${missing_system[*]}"
    case "$(uname -s)" in
        Darwin)  echo "  Install with:  brew install ffmpeg" ;;
        Linux)   echo "  Install with:  sudo apt install ffmpeg   (Debian/Ubuntu)"
                 echo "             or: sudo dnf install ffmpeg   (Fedora)" ;;
        *)       echo "  Install ffmpeg from https://ffmpeg.org/download.html" ;;
    esac
    exit 1
fi
green "✓ ffmpeg + ffprobe found"

# ---- 2. uv (Python toolchain) ----
if ! command -v uv >/dev/null 2>&1; then
    yellow "uv not found."
    read -r -p "Install uv now via the official installer? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS])
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # Make uv visible in this shell session.
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            ;;
        *)
            red "Install uv yourself, then re-run ./setup.sh"
            echo "  See https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
            ;;
    esac
fi
green "✓ uv $(uv --version | awk '{print $2}') found"

# ---- 3. Sync project dependencies ----
bold "==> Syncing dependencies (uv sync)"
uv sync

# ---- 4. Done ----
bold "==> Ready"
cat <<'EOF'

Run the converter with:

    uv run video_to_deck.py input.mp4 transcript.srt -o output.pptx

See README.md for all options.
EOF
