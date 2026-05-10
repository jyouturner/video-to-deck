#!/usr/bin/env bash
#
# yt2md installer — turn long YouTube videos into readable digests + slides.
#
#   curl -fsSL https://raw.githubusercontent.com/jyouturner/youtube-to-markdown/main/install.sh | bash
#
# This script does four things:
#   1. Verifies you're on macOS (the only currently-supported platform).
#   2. Verifies Homebrew is installed (prints a link and exits if not — we
#      don't auto-install Homebrew because that's a bigger trust ask than
#      this script should own).
#   3. brew installs ffmpeg, node, and uv (no-ops if already present).
#   4. uv tool installs yt2md from this repo's main branch.
#
# Re-running this script upgrades yt2md to the latest commit on main.
# Source you're about to run: https://github.com/jyouturner/youtube-to-markdown/blob/main/install.sh

set -euo pipefail

REPO_URL="https://github.com/jyouturner/youtube-to-markdown"
INSTALL_SOURCE="git+${REPO_URL}"
PACKAGE_NAME="youtube-to-markdown"

# ANSI helpers (only when stdout is a tty — keeps logs clean when piped).
if [[ -t 1 ]]; then
    BOLD=$(printf '\033[1m')
    DIM=$(printf '\033[2m')
    RED=$(printf '\033[31m')
    GREEN=$(printf '\033[32m')
    YELLOW=$(printf '\033[33m')
    CYAN=$(printf '\033[36m')
    RESET=$(printf '\033[0m')
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi

ok()      { printf "  %s✓%s %s\n" "$GREEN" "$RESET" "$1"; }
info()    { printf "  %s%s%s\n"   "$DIM"   "$1"    "$RESET"; }
heading() { printf "\n%s%s%s\n"   "$BOLD"  "$1"    "$RESET"; }
fail()    {
    printf "\n  %s✗%s %s\n" "$RED" "$RESET" "$1" >&2
    if [[ -n "${2-}" ]]; then
        printf "  %s→ %s%s\n\n" "$DIM" "$2" "$RESET" >&2
    fi
    exit 1
}

heading "yt2md installer"
info "Source: $REPO_URL"

# 1. macOS check. The pipeline relies on macOS-default tooling (open for the
# OAuth flow, brew for system deps, the Apple Silicon CPU / hw-decode path).
# Linux works in principle but isn't tested; Windows users should use WSL.
heading "1/4  Platform"
if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "This installer only supports macOS." \
         "Linux/WSL: install uv (https://docs.astral.sh/uv/), then run \"uv tool install $INSTALL_SOURCE\""
fi
ok "macOS $(sw_vers -productVersion 2>/dev/null || echo) on $(uname -m)"

# 2. Homebrew. We won't curl-bash brew silently — it needs sudo and writes
# into /usr/local or /opt/homebrew. Asking the user to install it themselves
# is a one-time cost most Mac users have paid; doing it covertly inside this
# script would be a big escalation in trust.
heading "2/4  Homebrew"
if ! command -v brew >/dev/null 2>&1; then
    fail "Homebrew not found on PATH." \
         "Install it from https://brew.sh and re-run this script."
fi
ok "found at $(command -v brew)"

# 3. System deps via brew. brew install is idempotent — already-installed
# formulae print "Warning: ... is already installed" and return 0. Loop +
# pre-check makes the output cleaner.
heading "3/4  System dependencies"
for pkg in ffmpeg node uv; do
    if brew list --formula 2>/dev/null | grep -qx "$pkg"; then
        ok "$pkg (already installed)"
    else
        info "brew installing $pkg…"
        if ! brew install "$pkg"; then
            fail "brew install $pkg failed." \
                 "Try running \"brew install $pkg\" directly to see the full error."
        fi
        ok "$pkg installed"
    fi
done

# 4. yt2md itself, via uv tool install. --reinstall makes the script
# upgrade-by-default on subsequent runs — same command works for fresh
# install and update.
heading "4/4  yt2md"
if uv tool list 2>/dev/null | grep -q "^${PACKAGE_NAME} "; then
    info "yt2md already installed — upgrading to latest main…"
    uv tool install --reinstall "$INSTALL_SOURCE"
else
    info "uv tool installing from $INSTALL_SOURCE…"
    uv tool install "$INSTALL_SOURCE"
fi
ok "yt2md installed at $(command -v yt2md 2>/dev/null || echo "(uv tool dir)")"

# Done. Print the next step prominently — `yt2md serve` is the entry point.
heading "Done"
printf "\n  Run %s%syt2md serve%s to start the local reader.\n\n" \
    "$BOLD" "$CYAN" "$RESET"
printf "  First run opens a setup page in your browser asking for either:\n"
printf "    • an Anthropic API key  (https://console.anthropic.com/settings/keys), or\n"
printf "    • a Claude.ai Pro/Max login via the bundled Claude Code sandbox.\n\n"
printf "  ${DIM}Re-run this curl-bash command to upgrade yt2md to the latest version.${RESET}\n\n"
