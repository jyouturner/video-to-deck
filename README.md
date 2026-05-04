# youtube-to-markdown

Read YouTube without watching. A local web app that turns videos into
Markdown digests — topic-segmented summaries with embedded frames — so you
can skim what was said in 30 seconds instead of sitting through 30 minutes.

Subscribe to channels and new videos get auto-digested in the background.
Drop in a one-off URL when you want a single digest. Click "Discuss with
experts" on any digest for a panel-of-experts deep-dive. Runs entirely on
your Mac.

## Quickstart

```bash
# Get the code
git clone https://github.com/jyouturner/youtube-to-markdown
cd youtube-to-markdown

# One-command bootstrap + launch
./run.sh
```

`run.sh` is idempotent — it checks ffmpeg, installs uv if missing (with a
prompt), syncs Python deps, runs `yt2md doctor` to verify the rest, and
launches the web reader at `http://localhost:7682/`. Re-run it any time to
restart; passing checks become no-ops.

If you'd rather drive it manually:

```bash
brew install ffmpeg uv      # one-time system deps
uv sync                     # install Python deps
uv run yt2md doctor         # verify everything
uv run yt2md serve          # launch reader (runs the in-process scheduler too)
``` Everything —
adding subscriptions, submitting one-off digests, generating panel
discussions, configuring models — happens through that UI. First run prompts
for an Anthropic API key
([get one here](https://console.anthropic.com/settings/keys)).

## Prerequisites

| | What | Why |
| --- | --- | --- |
| **System** | `ffmpeg` + `ffprobe` | frame extraction |
| **System** | `uv` (or `pip`) | Python toolchain |
| **System** | Node 20+ (or Deno) | yt-dlp's n-challenge JavaScript solver |
| **Config** | Anthropic API key | digest + panel-discussion LLM calls |
| **Config (recommended)** | YouTube login in a local browser | yt-dlp passes its cookies through; many videos now require a logged-in session |

`yt2md doctor` checks all of these and prints a punch list with fix hints.

If you only have Node 18 (old `/usr/local/bin/node`), the doctor and `serve`
auto-detect newer versions installed via nvm / fnm / asdf / volta and prepend
them to PATH. yt-dlp marks Node <20 as "unsupported" for the n-challenge.

## How it works

1. **Fetch** — `yt-dlp` downloads mp4 + auto-captions, with cookies passed
   through from your browser. If captions don't exist (e.g. some non-English
   videos), [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
   transcribes locally.
2. **Frame extraction** — `ffmpeg` scene detection + periodic interval sampling
   (every 20s by default).
3. **Dedup** — perceptual-hash clustering keeps the *last* frame of each
   near-identical run, so animated reveals settle on the fully-built diagram
   instead of a blank slide.
4. **Topic segmentation** — Claude reads the timestamped transcript and
   returns 5–12 topic sections (title, summary, key points), anchored to real
   timestamps.
5. **Vision-aware frame picking** — for each topic, Claude looks at the
   candidate frames *and the per-topic transcript slice* and picks the most
   illustrative one. The transcript context lets it ground picks on phrases
   like "as you can see here."
6. **Render** — Markdown digest with `<img>` tags, in either the source
   language (default — match the transcript) or English.
7. **(Optional) Discuss with experts** — click the button on any digest and
   Claude infers 3–5 domain-relevant experts (a neuroscientist for a brain
   talk, a hardware engineer for a chip-design talk, etc.) and runs a
   1500–2500-word panel discussion that surfaces what the speaker glossed
   over, brings contrary readings, and connects to adjacent domains.

## The web UI

Once `yt2md serve` is running, the sidebar gives you:

- **Digests** — every per-video digest, with unread markers
- **Subscriptions** — manage watched channels
- **One-off digest** — paste a URL; the digest runs in the background and
  lands in the sidebar when ready (with live progress on `/one-off` and
  `/activity`)
- **Schedule** — polling cadence + "Run now" buttons
- **Activity** — every completed run with timings, token usage, outcome.
  Survives server restarts.
- **Settings** — model choice (digest, panel, Whisper), output language
  (auto / English), browser to extract YouTube cookies from

Each digest page has a top toolbar:

- **Discuss with experts** — generates `panel.md` next to the digest
- **Delete digest** — wipes the rendered output, frames, and cached video

## Where things live

Everything is under `~/yt2md/` (override with `YT2MD_DATA=/path/to/dir`):

- `~/yt2md/.env` — API key + env-style overrides (auto-saved on first run)
- `~/yt2md/settings.json` — model + language + cookies preferences
- `~/yt2md/channels.txt` — subscriptions
- `~/yt2md/state.json` — last-seen video IDs per channel
- `~/yt2md/schedule.json` — polling interval
- `~/yt2md/schedule_state.json` — last-run timestamps
- `~/yt2md/library.db` — read state, run history (SQLite)
- `~/yt2md/digests/<video-id>/digest.md` — one digest per video
- `~/yt2md/digests/<video-id>/panel.md` — panel discussion (when generated)
- `~/yt2md/digests/<video-id>/digest_images/` — frames embedded in the digest
- `~/yt2md/digests/<video-id>/downloads/` — yt-dlp / Whisper cache
- `~/yt2md/logs/{poll,oneoff}.log` — job logs
- `~/yt2md/logs/runs.jsonl` — append-only structured run history

## Behavior notes

- **First run on a new channel** seeds state without backfilling — only
  videos posted *after* you subscribe get digested.
- **Permanent failures** (members-only / private / deleted videos) are
  marked seen on first failure so polling stops cycling on them.
  Transient failures (network, 5xx) keep retrying.
- **Schedule pauses while serve is down**, then catches up on missed slots
  the next time you start `yt2md serve`. Trade-off versus launchctl: one
  fewer execution context to debug, env/PATH match what just worked in your
  shell.
- **Why local instead of a cloud runner?** YouTube's "sign in to confirm
  you're not a bot" wall fires on datacenter IPs, so video downloads need a
  residential IP. Running on your Mac sidesteps the wall entirely.

## CLI reference

The CLI is small now — `serve` is the primary entry point and the web UI
covers everything else. The remaining commands:

```bash
yt2md doctor                          # check prerequisites and config
yt2md serve                           # local web reader at :7682
yt2md watch add <CHANNEL_URL>         # subscribe to a channel (or use the web UI)
yt2md watch list                      # see subscriptions
yt2md watch remove <CHANNEL_URL>      # unsubscribe
yt2md watch run                       # poll once now
yt2md "https://youtu.be/..."          # one-off digest from CLI (or use the web UI)
```

The single-video flow accepts `--digest-model`, `--whisper-model`,
`--cookies-from-browser`, `--digest-language`, etc. — most map to settings
the web UI exposes; check `yt2md --help` for the full list.

## Cost (defaults)

Per 20-min English video, with Sonnet 4.6 + vision: ~$0.16
- Digest pass: ~$0.05 (~9k input + ~2k output)
- Vision frame-picking: ~$0.11 (~30 frames as input)

Per panel discussion (Opus 4.7, on demand only): ~$0.30 — one click cost.
Pick `--digest-language=en` if you want English output for non-English
videos (slightly cheaper than Whisper-then-translate-yourself).

## Project layout

```
youtube_to_markdown.py    main script (CLI + web app + scheduler)
setup.sh                  one-shot onboarding (checks ffmpeg, installs uv, syncs deps)
pyproject.toml            project metadata + Python dependencies
uv.lock                   pinned resolution
.python-version           pinned Python version for uv
README.md                 this file
```
