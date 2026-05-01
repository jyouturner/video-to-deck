# youtube-to-markdown

Read YouTube without watching. Subscribe to channels and get Markdown digests
of new videos automatically — topic-segmented summaries, frames embedded as
`<img>` tags, plus a weekly cross-cutting meta-digest. Or digest one video on
demand. Runs entirely on your Mac.

## Two modes

### Subscription mode (the recommended daily-use path)

Set up once, runs forever. New videos get digested in the background; a
weekly meta-digest weaves the week's videos into one cross-cutting summary.

```bash
# Install (requires `uv` and `ffmpeg`)
brew install ffmpeg uv
uv tool install git+https://github.com/jyouturner/youtube-to-markdown
uv tool install yt-dlp

# Subscribe to a channel
yt2md watch add https://www.youtube.com/@LennysPodcast/videos

# Schedule both jobs on your Mac (polling every 6h, meta-digest weekly)
yt2md schedule install
```

That's it. Outputs land in `~/yt2md/digests/<video-id>/digest.md` (one per
video) and `~/yt2md/meta/YYYY-WW.md` (one per week). Read them in Finder,
Obsidian, or whatever you like.

### One-off mode (digest a single video)

Saw a video and want a digest of just that one? Skip the subscription:

```bash
yt2md "https://youtu.be/nWzXyjXCoCE"
```

Output lands in the current directory: `<video-id>_digest.md` + a sibling
`_images/` folder.

---

First run prompts for an Anthropic API key
([get one here](https://console.anthropic.com/settings/keys)) and offers to
save it to `~/yt2md/.env` so future runs find it
automatically.

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

## Subscription mode — reference

```bash
yt2md watch add <CHANNEL_URL>      # subscribe to a channel
yt2md watch list                    # see what you're watching
yt2md watch remove <CHANNEL_URL>    # unsubscribe
yt2md watch run                     # poll once now (no scheduler needed)

yt2md meta run                      # generate a meta-digest now

yt2md schedule install              # poll every 6h + meta weekly via launchd
yt2md schedule status               # check job status
yt2md schedule uninstall            # remove both jobs

yt2md serve                         # local web reader at http://localhost:7682
```

### Reading the digests

The simplest reader is `yt2md serve` — a small local web app at
`http://localhost:7682/` that lists every digest and meta-digest, renders
them with embedded images, and rewrites cross-references so meta-digests
link to their source digests. Auto-opens a browser tab. Ctrl-C to stop.

You can also browse `~/yt2md/digests/` and `~/yt2md/meta/` directly in
Finder, Obsidian, VS Code, or any other Markdown reader — the files are
plain `.md`.

**Everything lives in one visible directory: `~/yt2md/`** (override with
`YT2MD_DATA=/path/to/dir`). To inspect or remove all of the tool's data,
look in or delete that one folder. Layout:

- `~/yt2md/channels.txt` — your subscriptions
- `~/yt2md/state.json` — last-seen video IDs per channel
- `~/yt2md/.env` — API key (auto-saved on first run)
- `~/yt2md/digests/<video-id>/digest.md` — one per video
- `~/yt2md/meta/YYYY-WW.md` — weekly cross-cutting synthesis
- `~/yt2md/logs/{poll,meta}.log` — job logs
- `~/yt2md/downloads/<video-id>/` — yt-dlp cache (re-runs skip downloaded videos)

(The launchd plists are the one exception — macOS requires those at
`~/Library/LaunchAgents/com.youtube-to-markdown.{poll,meta}.plist`. They're
tiny pointer files; `yt2md schedule uninstall` removes them.)

**Behavior notes:**
- First run on a new channel just *seeds* state without backfilling — only videos posted *after* you add the channel get digested.
- Meta-digest synthesizes the past 7 days when at least 2 digests are present; otherwise skips.
- Schedule defaults: poll every 6 hours; meta-digest Sundays at 9am local time.

**Why local instead of a cloud runner?** YouTube's "sign in to confirm you're
not a bot" wall fires on datacenter IPs (GitHub Actions, Claude Code remote
routines), so video downloads need a residential IP. Running on your Mac
sidesteps the wall entirely. Tradeoff: the Mac has to be powered on for jobs
to fire — easy to satisfy with a normally-used laptop.

## One-off mode — reference

```bash
yt2md "https://youtu.be/..."                 # YouTube URL
yt2md input.mp4 input.srt                    # local files
yt2md "https://youtu.be/..." --deck          # also write a PowerPoint deck
yt2md "https://youtu.be/..." --deck-only     # only the deck (no API call needed)
yt2md "https://youtu.be/..." --no-vision     # cheaper, skip vision frame picking
```

Output (defaults) lands in the current directory:
- `<video-name>_digest.md` — overview + 5–12 topic sections
- `<video-name>_digest_images/topic_NN.jpg` — one frame per topic, embedded as `<img>` tags
- `<video-name>_deck.pptx` — only when `--deck` is set

YouTube downloads cache to `./downloads/<video-id>/` and re-runs on the same URL skip the download.

## API key

The digest needs `ANTHROPIC_API_KEY`. The tool looks in this order:
1. The shell environment
2. `./.env` in the current directory (per-project override)
3. `~/yt2md/.env` (default, used by scheduled jobs)

If none of those have it, the first interactive run prompts for the key and offers
to save it to `~/yt2md/.env` so you never have to think about
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
