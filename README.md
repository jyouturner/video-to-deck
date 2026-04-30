# video-to-deck

Turn a screen-recording MP4 + SRT transcript into a PowerPoint deck. Each slide is one
representative frame from the video, captioned with the transcript text spoken during
that segment; the full transcript is also placed in speaker notes.

## How it works

1. **Frame extraction** — combines `ffmpeg` scene detection with periodic interval
   sampling (one frame every N seconds). The interval pass is what makes screen
   recordings work; gradual scrolling/typing rarely trips a scene cut on its own.
2. **Dedup** — perceptual-hash (`imagehash.phash`) compared only against the *previous
   kept frame*. Identical-looking neighbors are dropped, but recurring views (e.g.
   returning to the same editor 5 minutes later) are preserved.
3. **Transcript alignment** — the SRT is parsed into segments; each frame's slide
   covers the window `[t_i, t_{i+1})` and collects every segment whose midpoint falls
   inside it.
4. **Deck build** — one slide per kept frame, image on top, transcript snippet
   below, full transcript in speaker notes.

## Setup

One command:

```bash
./setup.sh
```

This checks for `ffmpeg`/`ffprobe`, offers to install [`uv`](https://docs.astral.sh/uv/)
if it's missing, then runs `uv sync` to materialize the Python environment.

If you'd rather do it by hand: install `ffmpeg` (`brew install ffmpeg` on macOS) and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run `uv sync`.

## Usage

```bash
uv run video_to_deck.py input.mp4 transcript.srt -o output.pptx
```

uv resolves deps into `.venv` on first run; subsequent runs reuse it.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `-o`, `--output` | `output.pptx` | Output file path. |
| `--scene-threshold` | `0.2` | ffmpeg scene-cut sensitivity. `0.1` = many cuts, `0.5` = only major cuts. |
| `--interval` | `20` | Also sample one frame every N seconds. Set to `0` to disable. |
| `--hash-distance` | `4` | Perceptual-hash dedup threshold (compared to previous kept frame only). Higher = more aggressive dedup. |
| `--keep-frames` | off | Copy extracted frames to `./frames_<videoname>/` for inspection. |

### Tuning

- **Deck too sparse?** Lower `--interval` (e.g. `10`) or `--scene-threshold` (e.g. `0.15`).
- **Deck too dense / lots of similar slides?** Raise `--hash-distance` (e.g. `6`) or
  raise `--interval` (e.g. `40`).
- **Pure scene-cut mode** (no periodic sampling): `--interval 0`.

## Example

```bash
uv run video_to_deck.py Agent_Harness.mp4 Agent_Harness.srt -o Agent_Harness_deck.pptx
```

```
[1/5] Extracting frames (scene threshold=0.2, interval=20.0s)...
      18 scene + 62 interval = 80 candidate frames
[2/5] Deduping consecutive near-identical frames (hash distance <= 4)...
      69 unique frames
[3/5] Parsing SRT: Agent_Harness.srt
      506 transcript segments
[4/5] Aligning transcript to frames...
[5/5] Building deck -> Agent_Harness_deck.pptx

Done. 69 content slides + 1 title slide written to Agent_Harness_deck.pptx
```

## Project layout

```
video_to_deck.py    main script
setup.sh            one-shot onboarding (checks ffmpeg, installs uv, syncs deps)
pyproject.toml      project metadata + dependencies (used by uv)
uv.lock             pinned resolution (commit this to share an exact env)
.python-version     pinned Python version for uv
```
