# youtube-to-markdown

Turn a YouTube video (or local MP4 + SRT) into a Markdown digest you can read in 2–3
minutes. Topic-segmented summary, one representative frame per topic embedded as
`<img>` tags. Optional PowerPoint export for the same content.

## Quick start

```bash
# Install (requires `uv` and `ffmpeg`)
uv tool install --from git+https://github.com/<your-handle>/youtube-to-markdown yt2md

# Run on a YouTube URL — that's it
yt2md "https://youtu.be/nWzXyjXCoCE"
```

First run prompts for an Anthropic API key
([get one here](https://console.anthropic.com/settings/keys)) and offers to save it
to `~/.config/youtube-to-markdown/.env` so future runs find it automatically.

Output: `<video-id>_digest.md` plus a `<video-id>_digest_images/` folder, in your
current directory.

## How it works

1. **Fetch** — given a YouTube URL, download mp4 + auto-captions via `yt-dlp`,
   cached under `./downloads/<video-id>/`.
2. **Frame extraction** — `ffmpeg` scene detection + periodic interval sampling
   (every 20s by default). The interval pass is what makes screen recordings work;
   gradual scrolling rarely trips a scene cut on its own.
3. **Dedup** — perceptual-hash (`imagehash.phash`) compared only against the *previous
   kept frame*. Drops identical-looking neighbors, preserves recurring views (e.g.
   returning to the same editor 5 minutes later).
4. **Topic segmentation** — Claude reads the timestamped transcript and returns 5–12
   topic sections (title, summary, key points), each anchored to a real timestamp.
5. **Vision-aware frame picking** — for each topic, Claude looks at the candidate
   frames in that topic's time window and picks the most illustrative one.
6. **Render** — Markdown digest with `<img>` tags inline, plus the chosen frames in
   a sibling `_images/` folder. Optionally also a PowerPoint deck (one slide per
   kept frame, full transcript in speaker notes).

## Install

Two prerequisites: `uv` (https://docs.astral.sh/uv/) and `ffmpeg` (e.g.
`brew install ffmpeg` on macOS).

```bash
uv tool install --from git+https://github.com/<your-handle>/youtube-to-markdown yt2md
```

This installs both `yt2md` (short) and `youtube-to-markdown` (long form) — the
two are aliases for the same command.

For local development instead:
```bash
git clone <repo>
cd youtube-to-markdown
./setup.sh
```

## Usage

```bash
# Default — digest with vision-picked frames, from a YouTube URL
yt2md "https://youtu.be/nWzXyjXCoCE"

# Local files
yt2md input.mp4 transcript.srt

# Also build a PowerPoint deck
yt2md "https://youtu.be/..." --deck

# Just the deck, no API call (no key needed)
yt2md "https://youtu.be/..." --deck-only

# Cheaper digest (skip the vision pass)
yt2md "https://youtu.be/..." --no-vision
```

Output (defaults):
- `<video-name>_digest.md` — readable digest (overview + 5–12 topic sections)
- `<video-name>_digest_images/topic_NN.jpg` — one frame per topic, embedded as `<img>` tags
- `<video-name>_deck.pptx` — only when `--deck` is set

YouTube downloads cache to `./downloads/<video-id>/` and re-runs on the same URL
skip the download.

## API key

The digest needs `ANTHROPIC_API_KEY`. The tool looks in this order:
1. The shell environment
2. `.env` in the current directory
3. `~/.config/youtube-to-markdown/.env`

If none of those have it, the first interactive run prompts for the key and offers
to save it to `~/.config/youtube-to-markdown/.env` so you never have to think about
it again. Get a key at https://console.anthropic.com/settings/keys.

For non-interactive contexts (CI, cron) just export `ANTHROPIC_API_KEY` directly.

## Cost

Default settings on a 20-min video: ~$0.16
- Digest: ~$0.05 (Sonnet 4.6, ~9k input + ~2k output tokens)
- Vision pass: ~$0.11 (Sonnet 4.6, ~30 frames as input)

`--no-vision` drops it to ~$0.05. `--digest-model claude-opus-4-7` raises quality
~3–5× the cost. `--deck-only` skips the API entirely.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `-o`, `--output` | `<video-name>_digest.md` | Digest output path. |
| `--deck [PATH]` | off | Also write a PowerPoint deck (default path: `<video-name>_deck.pptx`). |
| `--deck-only` | off | Skip the digest entirely. No API key required. |
| `--no-vision` | off | Skip vision-based frame picking; cheaper but less illustrative. |
| `--digest-model` | `claude-sonnet-4-6` | Claude model for the digest. `claude-opus-4-7` for higher quality. |
| `--scene-threshold` | `0.2` | ffmpeg scene-cut sensitivity. `0.1` = many cuts, `0.5` = only major cuts. |
| `--interval` | `20` | Also sample one frame every N seconds. Set to `0` to disable. |
| `--hash-distance` | `4` | Perceptual-hash dedup threshold (compared to previous kept frame only). Higher = more aggressive dedup. |
| `--keep-frames` | off | Copy extracted frames to `./frames_<videoname>/` for inspection. |
| `--downloads-dir` | `./downloads` | Where to cache YouTube downloads. |

### Tuning

- **Deck too sparse?** Lower `--interval` (e.g. `10`) or `--scene-threshold` (e.g. `0.15`).
- **Deck too dense / lots of similar slides?** Raise `--hash-distance` (e.g. `6`) or
  raise `--interval` (e.g. `40`).
- **Pure scene-cut mode** (no periodic sampling): `--interval 0`.

## Example

```bash
$ yt2md "https://youtu.be/nWzXyjXCoCE"
[0/5] Fetching YouTube video: https://youtu.be/nWzXyjXCoCE
      using cached downloads/nWzXyjXCoCE/
[1/5] Extracting frames (scene threshold=0.2, interval=20.0s)...
      18 scene + 62 interval = 80 candidate frames
[2/5] Deduping consecutive near-identical frames (hash distance <= 4)...
      68 unique frames
[3/5] Parsing SRT: nWzXyjXCoCE.en.srt
      506 transcript segments
[4/5] Aligning transcript to frames...
[5/5] Skipping deck (use --deck to also write one)
[+] Generating digest with claude-sonnet-4-6 -> nWzXyjXCoCE_digest.md
      9 topics  |  input: 7600 tokens (cache write) |  output: 1891 tokens
[+] Vision-picking frames with claude-sonnet-4-6...
      vision-selected 9/9 topics  |  input: 30928 tokens  |  output: 512 tokens
      Digest written. Images in nWzXyjXCoCE_digest_images/

Done. Wrote: nWzXyjXCoCE_digest.md
```

## Project layout

```
youtube_to_markdown.py    main script
setup.sh            one-shot onboarding (checks ffmpeg, installs uv, syncs deps)
pyproject.toml      project metadata + dependencies (used by uv)
uv.lock             pinned resolution (commit this to share an exact env)
.python-version     pinned Python version for uv
```
