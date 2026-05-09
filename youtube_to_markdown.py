#!/usr/bin/env python3
"""
youtube-to-markdown — Turn a YouTube video (or local MP4 + SRT) into a readable
Markdown digest with embedded frame images. Optional PowerPoint export.

Default flow (digest):
  yt2md "https://youtu.be/..."
  yt2md input.mp4 transcript.srt

Also build a deck:
  yt2md "https://youtu.be/..." --deck

Just the deck (no API key needed):
  yt2md input.mp4 transcript.srt --deck-only

Requirements:
  System:  ffmpeg, ffprobe
  API:     ANTHROPIC_API_KEY (prompted on first run unless --deck-only)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ---------- Step 1: Frame extraction (scene detection + periodic sampling) ----------

def extract_scene_frames(video: Path, out_dir: Path, threshold: float = 0.2) -> List[Tuple[Path, float]]:
    """Run ffmpeg scene detection. Returns list of (frame_path, timestamp_seconds)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-vf", f"select='eq(n,0)+gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-q:v", "3",
        str(out_dir / "scene_%04d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")

    timestamps = [float(m) for m in re.findall(r"pts_time:([\d.]+)", proc.stderr)]
    frame_files = sorted(out_dir.glob("scene_*.jpg"))
    if len(frame_files) != len(timestamps):
        n = min(len(frame_files), len(timestamps))
        frame_files, timestamps = frame_files[:n], timestamps[:n]

    return list(zip(frame_files, timestamps))


def extract_interval_frames(video: Path, out_dir: Path, interval: float, duration: float) -> List[Tuple[Path, float]]:
    """Sample one frame every `interval` seconds. Returns (frame_path, timestamp) list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if interval <= 0:
        return []

    fps = 1.0 / interval
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "3",
        str(out_dir / "interval_%04d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg interval sampling failed:\n{proc.stderr}")

    frame_files = sorted(out_dir.glob("interval_*.jpg"))
    # ffmpeg's fps filter places the first frame at t≈interval/2; reconstruct timestamps.
    timestamps = [min(duration, interval * (i + 0.5)) for i in range(len(frame_files))]
    return list(zip(frame_files, timestamps))


def merge_frames(*frame_lists: List[Tuple[Path, float]]) -> List[Tuple[Path, float]]:
    """Merge multiple frame lists, sort by timestamp."""
    combined: List[Tuple[Path, float]] = []
    for fl in frame_lists:
        combined.extend(fl)
    combined.sort(key=lambda x: x[1])
    return combined


# ---------- Step 2: Perceptual-hash dedup (consecutive only) ----------

def dedupe_frames(frames: List[Tuple[Path, float]], hash_distance: int = 4) -> List[Tuple[Path, float]]:
    """Cluster runs of near-identical consecutive frames; keep the LAST of each cluster.

    Animated slide reveals (bullets / diagram elements appearing over time) and slow
    pans both hit this case: each intermediate frame is similar to its neighbor, but
    only the final frame shows the fully-revealed / settled state. Keeping the last
    frame of the cluster preserves that state rather than the partial opening one.

    Discrete scene changes (dist > threshold between neighbors) break the cluster,
    so recurring views — e.g. switching to an editor and back to slides — still
    survive as separate kept frames.
    """
    if not frames:
        return []

    import imagehash
    from PIL import Image

    hashed: List[Tuple[Path, float, "imagehash.ImageHash"]] = []
    for path, ts in frames:
        with Image.open(path) as im:
            hashed.append((path, ts, imagehash.phash(im)))

    clusters: List[List[Tuple[Path, float, "imagehash.ImageHash"]]] = [[hashed[0]]]
    for i in range(1, len(hashed)):
        if (hashed[i][2] - hashed[i - 1][2]) <= hash_distance:
            clusters[-1].append(hashed[i])
        else:
            clusters.append([hashed[i]])

    kept: List[Tuple[Path, float]] = []
    for cluster in clusters:
        for path, _, _ in cluster[:-1]:
            path.unlink(missing_ok=True)
        kept.append((cluster[-1][0], cluster[-1][1]))
    return kept


# ---------- Step 3: SRT parser ----------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _srt_time_to_seconds(t: str) -> float:
    """'00:01:23,456' -> 83.456"""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(srt_path: Path) -> List[TranscriptSegment]:
    """Parse a .srt file. Tolerant of BOM, CRLF, dot-vs-comma ms separator, simple markup."""
    text = srt_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n", text.strip())

    segments: List[TranscriptSegment] = []
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_line_idx = next((i for i, ln in enumerate(lines) if time_re.search(ln)), None)
        if time_line_idx is None:
            continue
        m = time_re.search(lines[time_line_idx])
        start = _srt_time_to_seconds(m.group(1).replace(".", ","))
        end = _srt_time_to_seconds(m.group(2).replace(".", ","))
        body = " ".join(lines[time_line_idx + 1:])
        body = re.sub(r"<[^>]+>", "", body)         # <i>, <b>, <font>
        body = re.sub(r"\{[^}]+\}", "", body).strip()  # {\an8}, etc.
        if body:
            segments.append(TranscriptSegment(start=start, end=end, text=body))

    return segments


# ---------- Step 4: Align transcript to frames ----------

def assign_transcript_to_frames(
    frames: List[Tuple[Path, float]],
    segments: List[TranscriptSegment],
    video_duration: float,
) -> List[Tuple[Path, float, float, str]]:
    """
    For each frame at time t_i, the slide covers [t_i, t_{i+1}).
    Collect transcript segments whose midpoint falls in that window.
    Returns: list of (frame_path, slide_start, slide_end, transcript_text).
    """
    results = []
    for i, (path, start) in enumerate(frames):
        end = frames[i + 1][1] if i + 1 < len(frames) else video_duration
        chunk = " ".join(
            seg.text for seg in segments
            if start <= (seg.start + seg.end) / 2 < end
        )
        results.append((path, start, end, chunk))
    return results


def get_video_duration(video: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video),
    ])
    return float(json.loads(out)["format"]["duration"])


# ---------- Step 5: Build the .pptx ----------

def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _shorten(text: str, max_chars: int = 280) -> str:
    """Trim a transcript chunk to a slide-friendly length on a sentence boundary."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer to end on sentence punctuation, then on a word boundary
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx >= max_chars * 0.5:
            return cut[: idx + 1].strip()
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > 0 else cut).strip() + "…"


def build_deck(slides_data, output: Path, video_name: str) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from PIL import Image

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Title slide
    title_slide = prs.slides.add_slide(blank_layout)
    tb = title_slide.shapes.add_textbox(Inches(1), Inches(3), Inches(11.33), Inches(1.5))
    p = tb.text_frame.paragraphs[0]
    p.text = video_name
    p.font.size = Pt(40)
    p.font.bold = True
    sub = title_slide.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11.33), Inches(0.5))
    sub.text_frame.paragraphs[0].text = f"{len(slides_data)} slides extracted from video"
    sub.text_frame.paragraphs[0].font.size = Pt(18)

    # Layout: image on top (~5"), transcript snippet below (~1.6"), footer at bottom.
    img_top_in = 0.3
    img_max_h_in = 5.0
    img_max_w_in = 12.33
    text_top_in = 5.5
    text_h_in = 1.6
    footer_top_in = 7.15

    for idx, (frame_path, start, end, transcript) in enumerate(slides_data, 1):
        slide = prs.slides.add_slide(blank_layout)

        with Image.open(frame_path) as im:
            iw, ih = im.size
        scale = min(img_max_w_in / (iw / 96), img_max_h_in / (ih / 96))
        disp_w = (iw / 96) * scale
        disp_h = (ih / 96) * scale
        left = (13.33 - disp_w) / 2

        slide.shapes.add_picture(
            str(frame_path),
            Inches(left), Inches(img_top_in),
            width=Inches(disp_w), height=Inches(disp_h),
        )

        # On-slide transcript snippet
        snippet = _shorten(transcript, 280) if transcript else ""
        if snippet:
            tx = slide.shapes.add_textbox(Inches(0.5), Inches(text_top_in),
                                          Inches(12.33), Inches(text_h_in))
            tf = tx.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            para.text = snippet
            para.font.size = Pt(14)

        # Footer with time range + slide number
        footer = slide.shapes.add_textbox(Inches(0.3), Inches(footer_top_in), Inches(12.7), Inches(0.3))
        fp = footer.text_frame.paragraphs[0]
        fp.text = f"{format_timestamp(start)} – {format_timestamp(end)}   |   Slide {idx}"
        fp.font.size = Pt(10)

        if transcript:
            slide.notes_slide.notes_text_frame.text = transcript

    prs.save(str(output))


# ---------- Config / API key handling ----------

# Everything lives in one visible directory under HOME. Override with YT2MD_DATA.
# Layout under that directory:
#   .env, channels.txt, state.json, digests/, meta/, downloads/, logs/

DEFAULT_DATA_DIR = Path.home() / "yt2md"


def get_data_dir() -> Path:
    return Path(os.environ.get("YT2MD_DATA", str(DEFAULT_DATA_DIR))).expanduser()


def env_file() -> Path:
    return get_data_dir() / ".env"


def load_env_files() -> None:
    """Populate os.environ from .env files. Real env vars always win.

    Order (lowest priority first; later loads do NOT override earlier-set keys):
      1. Real env vars (from the shell)
      2. CWD/.env (project-local override)
      3. <data dir>/.env (default: ~/yt2md/.env)
    """
    from dotenv import load_dotenv

    load_dotenv()  # CWD/.env, only fills in missing
    e = env_file()
    if e.exists():
        load_dotenv(e)


def set_env_var(name: str, value: str) -> Path:
    """Persist <name>=<value> to ~/yt2md/.env, preserving other entries.

    Uses dotenv.set_key for round-trip safe updates (vs. naive overwrite, which
    would clobber co-resident keys). Also updates os.environ so the running
    process sees the new value immediately. Returns the .env path.
    """
    from dotenv import set_key

    e = env_file()
    e.parent.mkdir(parents=True, exist_ok=True)
    if not e.exists():
        e.touch()
    try:
        os.chmod(e, 0o600)
    except OSError:
        pass
    set_key(str(e), name, value, quote_mode="never")
    os.environ[name] = value
    return e


API_KEY_COST_NOTE = (
    "Anthropic bills your API key per request (separate from any Claude.ai "
    "subscription). Rough costs: a 30-min digest is ~$0.03 with "
    "<code>claude-sonnet-4-6</code> (default), ~$0.15 with "
    "<code>claude-opus-4-7</code>. The panel discussion adds one Opus call "
    "(~$0.10). Add a payment method at "
    '<a href="https://console.anthropic.com/settings/billing" target="_blank" '
    'rel="noopener">console.anthropic.com/settings/billing</a>.'
)


def validate_api_key(key: str) -> Optional[str]:
    """Send a 1-token request to the cheapest model to verify auth. Returns
    None on success, otherwise a short human-readable error string.
    """
    import anthropic

    try:
        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "ok"}],
        )
        return None
    except anthropic.AuthenticationError:
        return "key rejected by Anthropic (authentication failed)."
    except anthropic.PermissionDeniedError as e:
        return f"key authenticated but lacks permission: {e}"
    except anthropic.APIConnectionError as e:
        return f"could not reach Anthropic: {e}"
    except Exception as e:
        return f"unexpected error: {type(e).__name__}: {e}"


def ensure_api_key() -> None:
    """Make sure ANTHROPIC_API_KEY is set, prompting + saving on first run if interactive."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return

    e = env_file()
    msg = (
        "ANTHROPIC_API_KEY is not set.\n"
        "Get a key from: https://console.anthropic.com/settings/keys"
    )
    if not sys.stdin.isatty():
        sys.exit(
            f"{msg}\n"
            "Then either export it (`export ANTHROPIC_API_KEY=...`) or save it via:\n"
            f"  mkdir -p {e.parent} && echo 'ANTHROPIC_API_KEY=sk-ant-...' >> {e}"
        )

    print(msg)
    key = input("Paste your API key (or press Enter to abort): ").strip()
    if not key:
        sys.exit("Aborted.")

    save = input(
        f"Save it to {e} so future runs find it automatically? [Y/n] "
    ).strip().lower()
    if save in ("", "y", "yes"):
        set_env_var("ANTHROPIC_API_KEY", key)
        print(f"      saved to {e}")
    else:
        os.environ["ANTHROPIC_API_KEY"] = key


# ---------- YouTube fetch ----------

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


DEFAULT_WHISPER_MODEL = "medium"


def _whisper_secs_to_srt(secs: float) -> str:
    """Convert seconds to SRT-style HH:MM:SS,mmm timestamp."""
    if secs < 0:
        secs = 0.0
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int(round((secs - int(secs)) * 1000))
    if ms == 1000:  # rounding can push us a full ms over
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _transcribe_with_whisper(
    media_path: Path,
    out_dir: Path,
    video_id: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
) -> Tuple[Path, str]:
    """Transcribe a media file with faster-whisper, write an SRT, return (srt_path, lang).

    Model weights are downloaded on first use to ~/.cache/huggingface and reused
    afterwards. Detected language is used as the SRT filename suffix so the
    existing cache-by-glob logic in fetch_youtube picks it up on re-runs.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "Whisper transcription needed but faster-whisper is not installed. "
            "Run `uv sync` (or `pip install faster-whisper`) and try again."
        ) from e

    print(f"      loading whisper model '{model_name}' (first run downloads weights)")
    # int8 keeps memory low and runs well on CPU; faster-whisper picks Metal/CUDA
    # automatically when device='auto'.
    model = WhisperModel(model_name, device="auto", compute_type="int8")

    print(f"      transcribing audio with whisper ({media_path.name})...")
    segments_iter, info = model.transcribe(
        str(media_path),
        beam_size=5,
        vad_filter=True,  # cuts long silences so the transcript stays tight
    )

    lang = info.language or "und"
    srt_path = out_dir / f"{video_id}.{lang}.srt"

    n = 0
    with srt_path.open("w") as fh:
        for seg in segments_iter:
            n += 1
            text = seg.text.strip().replace("\n", " ")
            if not text:
                continue
            fh.write(
                f"{n}\n"
                f"{_whisper_secs_to_srt(seg.start)} --> {_whisper_secs_to_srt(seg.end)}\n"
                f"{text}\n\n"
            )

    print(
        f"      whisper: {n} segments, language='{lang}' "
        f"(prob={info.language_probability:.2f})"
    )
    return srt_path, lang


def _ensure_js_runtime_available() -> Optional[str]:
    """Find a JS runtime usable for yt-dlp's n-challenge solver.

    yt-dlp accepts deno / node / bun. As of 2026, it marks Node <20 as
    'unsupported' — silently failing the n-challenge and producing only
    storyboard formats. So for Node we collect all candidates (PATH match +
    common version-manager locations: nvm, fnm, asdf, volta), version-rank
    them, and prepend the dir of the highest version to os.environ['PATH']
    so yt-dlp's internal lookups pick the right one.
    """
    # deno / bun: trust the first PATH match (no version-rank needed).
    for rt in ("deno", "bun"):
        if shutil.which(rt):
            return rt

    home = Path.home()
    seen: set = set()
    node_candidates: List[Path] = []

    def _add(p: Optional[Path]):
        if p and p.is_file() and str(p) not in seen:
            seen.add(str(p))
            node_candidates.append(p)

    path_node = shutil.which("node")
    if path_node:
        _add(Path(path_node))
    for p in sorted((home / ".nvm" / "versions" / "node").glob("*/bin/node"), reverse=True):
        _add(p)
    for p in sorted(
        (home / ".local" / "share" / "fnm" / "node-versions").glob("*/installation/bin/node"),
        reverse=True,
    ):
        _add(p)
    for p in sorted((home / ".asdf" / "installs" / "nodejs").glob("*/bin/node"), reverse=True):
        _add(p)
    _add(home / ".volta" / "bin" / "node")

    best_path: Optional[Path] = None
    best_version: Tuple[int, ...] = (0, 0, 0)
    for c in node_candidates:
        try:
            r = subprocess.run(
                [str(c), "--version"], capture_output=True, text=True, timeout=5
            )
            v = tuple(int(p) for p in r.stdout.strip().lstrip("v").split(".")[:3])
        except Exception:
            continue
        if v > best_version:
            best_version = v
            best_path = c

    if best_path is None:
        return None

    # Prepend its dir so yt-dlp's shutil.which lookups pick this one.
    bin_dir = str(best_path.parent)
    current = shutil.which("node")
    if current != str(best_path):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        print(
            f"[yt2md] using node v{'.'.join(map(str, best_version))} at "
            f"{best_path} for yt-dlp"
        )

    if best_version < (20, 0, 0):
        print(
            f"[yt2md] warning: node v{'.'.join(map(str, best_version))} is below "
            "yt-dlp's required v20+ for YouTube JS challenges. Install a newer "
            "node (`nvm install 20`) or deno (`brew install deno`).",
            file=sys.stderr,
        )

    return "node"


def _pick_caption_lang(info: dict) -> Optional[Tuple[str, bool]]:
    """Choose best caption track from a yt-dlp info dict.

    Returns (lang_code, is_manual) or None if no captions exist at all.

    Priority:
      1. Manual English (any en-* code)
      2. Manual matching the audio language (info['language'])
      3. Any manual track
      4. Auto matching the audio language
      5. Auto English
      6. Any auto track

    Manual is preferred over auto everywhere. Within auto, we prefer the
    original audio language over English: YouTube's auto-EN on a non-English
    video is translation-of-auto-caption (double degradation), whereas
    Claude translating the auto-caption-of-original-audio is single degradation
    and produces better digests.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    audio_lang = (info.get("language") or "").lower()

    def first_starts_with(d, prefix):
        if not prefix:
            return None
        for k in d:
            if k.lower().startswith(prefix):
                return k
        return None

    def first_any(d):
        return next(iter(d), None)

    for picker, source, is_manual in (
        (lambda d: first_starts_with(d, "en"), manual, True),
        (lambda d: first_starts_with(d, audio_lang), manual, True),
        (first_any, manual, True),
        (lambda d: first_starts_with(d, audio_lang), auto, False),
        (lambda d: first_starts_with(d, "en"), auto, False),
        (first_any, auto, False),
    ):
        k = picker(source)
        if k:
            return (k, is_manual)
    return None


def fetch_youtube(
    url: str,
    cache_root: Path,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    allow_whisper: bool = True,
    cookies_from_browser: Optional[str] = None,
) -> dict:
    """Download mp4 + best-available SRT from YouTube. Cached by video ID under cache_root.

    Returns a dict with:
      mp4 (Path), srt (Path), lang (str),
      title (str), webpage_url (str),
      download_secs (float), whisper_secs (float),
      used_whisper (bool), whisper_model (Optional[str]).

    Falls back to local Whisper transcription when YouTube has no captions.
    Set allow_whisper=False to fail fast instead of falling back.
    """
    import yt_dlp
    import time as _time

    cache_root.mkdir(parents=True, exist_ok=True)

    # YouTube increasingly requires logged-in cookies to bypass the bot challenge.
    # yt-dlp accepts `cookiesfrombrowser` as a tuple — single-item is the simplest
    # form (no profile / domain filter).
    cookie_opt: dict = {}
    if cookies_from_browser:
        cookie_opt["cookiesfrombrowser"] = (cookies_from_browser,)

    # YouTube's "n challenge" obfuscates real format URLs behind a JavaScript
    # function that yt-dlp must execute to deobfuscate. Without a JS runtime +
    # the challenge solver scripts, only thumbnail storyboards come back.
    rt = _ensure_js_runtime_available()
    yt_dlp_runtime_opt: dict = {}
    if rt is not None:
        yt_dlp_runtime_opt["js_runtimes"] = {rt: {}}
        yt_dlp_runtime_opt["remote_components"] = ["ejs:github"]

    base_opts = {**cookie_opt, **yt_dlp_runtime_opt}

    # Probe first to get the video ID for stable cache layout. We use the same
    # format selector as the download below so probe doesn't reject videos whose
    # default yt-dlp selector ('bestvideo*+bestaudio') happens to match nothing
    # (some YouTube videos return a format pool that the default doesn't span).
    probe_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        **base_opts,
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    video_id = info["id"]
    title = info.get("title") or video_id
    webpage_url = info.get("webpage_url") or url
    out_dir = cache_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f"{video_id}.mp4"

    # Cache hit: lang is in the filename (works for legacy *.en.srt too).
    existing_srt = next(iter(out_dir.glob(f"{video_id}.*.srt")), None)
    if mp4_path.exists() and existing_srt is not None:
        lang = existing_srt.stem[len(video_id) + 1:]
        print(f"      using cached {out_dir}/ (lang: {lang})")
        return {
            "mp4": mp4_path, "srt": existing_srt, "lang": lang,
            "title": title, "webpage_url": webpage_url,
            "download_secs": 0.0, "whisper_secs": 0.0,
            "used_whisper": False, "whisper_model": None,
        }

    picked = _pick_caption_lang(info)

    if picked is not None:
        picked_lang, _is_manual = picked
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [picked_lang],
            "subtitlesformat": "srt/vtt/best",
            "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
            "quiet": True,
            "no_warnings": True,
            **base_opts,
        }
    else:
        # No captions of any kind. Download just the mp4 and transcribe locally.
        if not allow_whisper:
            raise RuntimeError(
                f"No subtitles available for {url} in any language and "
                "Whisper fallback is disabled."
            )
        print("      no captions on YouTube; will transcribe with Whisper after download")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            **base_opts,
        }

    download_t0 = _time.monotonic()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    download_secs = _time.monotonic() - download_t0

    candidates_mp4 = list(out_dir.glob(f"{video_id}.*"))
    mp4_found = next((p for p in candidates_mp4 if p.suffix in (".mp4", ".mkv", ".webm")), None)
    if mp4_found and mp4_found != mp4_path:
        mp4_found.rename(mp4_path)
    elif not mp4_path.exists():
        raise RuntimeError(f"yt-dlp finished but no video file found in {out_dir}")

    if picked is not None:
        # Re-derive lang from the filename yt-dlp actually wrote (it normalizes
        # codes like en-US -> en, so the picked code may differ from the result).
        srt_found = next(iter(out_dir.glob(f"{video_id}.*.srt")), None)
        if srt_found is None:
            raise RuntimeError(
                f"yt-dlp claimed '{picked[0]}' subtitles for {url} but produced no SRT."
            )
        lang = srt_found.stem[len(video_id) + 1:]
        return {
            "mp4": mp4_path, "srt": srt_found, "lang": lang,
            "title": title, "webpage_url": webpage_url,
            "download_secs": download_secs, "whisper_secs": 0.0,
            "used_whisper": False, "whisper_model": None,
        }

    whisper_t0 = _time.monotonic()
    srt_path, lang = _transcribe_with_whisper(mp4_path, out_dir, video_id, model_name=whisper_model)
    whisper_secs = _time.monotonic() - whisper_t0
    return {
        "mp4": mp4_path, "srt": srt_path, "lang": lang,
        "title": title, "webpage_url": webpage_url,
        "download_secs": download_secs, "whisper_secs": whisper_secs,
        "used_whisper": True, "whisper_model": whisper_model,
    }


# ---------- Markdown digest (transcript-primary, LLM-summarized) ----------

DIGEST_SYSTEM_PROMPT = (
    "You distill video transcripts into readable digests so a reader can grasp "
    "the essence of a video without watching it.\n\n"
    "Given a timestamped transcript, identify the natural topic segments. "
    "Return 5–12 sections depending on length and content density (favor fewer "
    "for short videos, more for long talks).\n\n"
    "For each topic:\n"
    "- title: a short, descriptive heading (not a question, not a teaser)\n"
    "- start_time: the timestamp in SECONDS where the topic begins, taken from "
    "the transcript's bracketed timestamps\n"
    "- summary: 2–4 sentences of informative prose distilling what is actually "
    "said. Concrete claims, names, numbers, and reasoning — not vague paraphrases.\n"
    "- key_points: 3–6 short bullet points capturing the most useful takeaways. "
    "Bullets should add detail beyond the summary, not restate it.\n\n"
    "Also produce a 2–3 sentence overview of the entire video.\n\n"
    "Write for a reader, not a viewer. Skip filler ('in this video I'll show you'); "
    "go straight to the substance."
)


def generate_digest(
    segments: List[TranscriptSegment],
    video_title: str,
    model: str,
    source_lang: str = "en",
    output_language: str = "auto",
):
    """Call Claude to segment the transcript into topics. Returns a parsed VideoDigest.

    source_lang is the BCP-47 language code of the transcript (e.g. 'en', 'zh-Hans').
    output_language: 'auto' (write in source language) or 'en' (force English).
    """
    import anthropic
    from pydantic import BaseModel
    from typing import List as TList

    class Topic(BaseModel):
        title: str
        start_time: float
        summary: str
        key_points: TList[str]

    class VideoDigest(BaseModel):
        title: str
        overview: str
        topics: TList[Topic]

    transcript = "\n".join(
        f"[{format_timestamp(seg.start)}] {seg.text}" for seg in segments
    )

    is_english_source = (source_lang or "").lower().startswith("en")
    lang_note = ""
    if not is_english_source:
        if output_language == "en":
            lang_note = (
                f"NOTE: This transcript is in language code '{source_lang}', not English. "
                "Translate to English while distilling. Title, overview, topic titles, "
                "summaries, and key points must all be written in English regardless of "
                "the source language. Preserve proper nouns (people, places, products) in "
                "their original form when there is no established English rendering.\n\n"
            )
        else:  # "auto" — match the source language
            lang_note = (
                f"NOTE: This transcript is in language code '{source_lang}'. Write the "
                "digest in the SAME language as the transcript. Title, overview, topic "
                "titles, summaries, and key points must all be in the source language. "
                "Preserve proper nouns and established technical terms in their original "
                "form (including English technical jargon when the field uses it that way).\n\n"
            )

    user_text = (
        f"{lang_note}"
        f"Video title: {video_title}\n\n"
        f"Total duration: {format_timestamp(segments[-1].end if segments else 0)}\n\n"
        f"Timestamped transcript:\n\n{transcript}"
    )

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=DIGEST_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": user_text,
                "cache_control": {"type": "ephemeral"},
            }],
        }],
        output_format=VideoDigest,
    )
    return response.parsed_output, response.usage


DEFAULT_PANEL_MODEL = "claude-opus-4-7"


PANEL_SYSTEM_PROMPT = (
    "You facilitate a panel of domain experts critically analyzing video content. "
    "Read the digest and transcript carefully, then:\n\n"
    "1. Infer 3–5 experts whose perspectives would best illuminate this material. "
    "Choose them from the actual domain of the video — a neuroscientist for a brain "
    "talk, a hardware engineer + an ML practitioner for a chip-design talk, a historian "
    "of science + a contemporary researcher for a science-history piece. Avoid generic "
    "labels (\"a thoughtful generalist\"); make each expert's specialty concrete enough "
    "that their angle on this material is distinct.\n\n"
    "2. Run a 1500–2500 word panel discussion in markdown. Open with one short paragraph "
    "introducing each panelist (name, role, one credential or claim-to-relevance). Then "
    "the discussion proper, with each turn labeled by the speaker's name.\n\n"
    "Goals for the discussion:\n"
    "- Surface what the speaker glossed over, hand-waved, or assumed without arguing.\n"
    "- Bring contrary readings — where would a competing school of thought disagree?\n"
    "- Connect to adjacent domains the speaker didn't mention.\n"
    "- Examine concrete claims (numbers, names, mechanisms) for how robust they actually are.\n"
    "- Synthesize, but don't paper over disagreements: if two panelists land in "
    "different places, leave them there.\n\n"
    "Style: skip restating the digest — the reader already read it. Open directly with "
    "the moderator framing the first question. No conclusion-summary at the end; let "
    "the discussion close naturally."
)


def generate_panel_discussion(
    digest_md_text: str,
    segments: List["TranscriptSegment"],
    model: str,
    source_lang: str = "en",
    output_language: str = "auto",
):
    """Call Claude to simulate a panel of domain-relevant experts discussing a video.
    Returns (markdown_text, usage).

    source_lang / output_language follow the same convention as generate_digest:
    'auto' writes the panel in the transcript's language; 'en' forces English.

    Costs ~one Opus call per click (≈ 4–8k input + 2–4k output tokens). Output is
    one markdown document the caller writes to digests/<id>/panel.md.
    """
    import anthropic

    transcript_str = "\n".join(
        f"[{format_timestamp(seg.start)}] {seg.text}" for seg in segments
    )

    is_english_source = (source_lang or "").lower().startswith("en")
    lang_directive = ""
    if not is_english_source and output_language == "auto":
        lang_directive = (
            f"\n\nIMPORTANT: The transcript is in language code '{source_lang}'. "
            "Write the entire panel discussion in the SAME language — expert names "
            "(transliterated when appropriate), credentials, the moderator's "
            "questions, every speaker's turns. Preserve proper nouns and technical "
            "terms in their original form when the field uses them that way."
        )
    # output_language == "en" with non-English source: rely on the existing
    # system prompt (no explicit translate directive needed; English is the
    # default Claude output style for this prompt).

    user_text = (
        "## Existing digest (the reader has already seen this)\n\n"
        f"{digest_md_text}\n\n"
        "## Full timestamped transcript\n\n"
        f"{transcript_str}"
        f"{lang_directive}\n\n"
        "Now: introduce the panelists, then run the discussion."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=PANEL_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{
                "type": "text", "text": user_text,
                "cache_control": {"type": "ephemeral"},
            }],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return text, response.usage


def _transcript_slice(
    segments: List["TranscriptSegment"],
    topic_start: float,
    topic_end: float,
) -> str:
    """Render the transcript segments inside [topic_start, topic_end) as one timestamped string.

    Used by vision_pick_frames to ground picks against what the narrator is saying
    at a candidate frame's timestamp. Truncates very long topics to keep token cost
    bounded — first 30 + last 30 segments with a marker between, which captures the
    narrator's framing at start and conclusion at end.
    """
    in_window = [s for s in segments if topic_start <= s.start < topic_end]
    if len(in_window) > 70:
        head = in_window[:30]
        tail = in_window[-30:]
        in_window = head + [None] + tail  # type: ignore[list-item]
    lines: List[str] = []
    for seg in in_window:
        if seg is None:
            lines.append("[…]")
            continue
        lines.append(f"[{format_timestamp(seg.start)}] {seg.text}")
    return "\n".join(lines)


def _candidates_for_topic(
    topic_start: float,
    topic_end: float,
    frames: List[Tuple[Path, float]],
    max_per_topic: int = 5,
    overlap_pre: float = 5.0,
) -> List[Tuple[Path, float]]:
    """Frames whose timestamp falls in [start - overlap, end). Downsample to max_per_topic."""
    in_window = [
        (p, t) for p, t in frames if (topic_start - overlap_pre) <= t < topic_end
    ]
    if len(in_window) <= max_per_topic:
        return in_window
    # Even spacing across the window
    step = len(in_window) / max_per_topic
    return [in_window[int(i * step)] for i in range(max_per_topic)]


def _encode_frame_for_vision(path: Path, max_long_edge: int = 1024) -> str:
    """Resize a frame to <= max_long_edge on the long side and return base64-encoded JPEG bytes."""
    import base64
    import io
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_long_edge / max(w, h)
        if scale < 1:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def vision_pick_frames(
    digest,
    frames: List[Tuple[Path, float]],
    video_duration: float,
    model: str,
    segments: Optional[List["TranscriptSegment"]] = None,
):
    """Use Claude's vision to pick the best frame per topic from in-window candidates.

    Returns a dict {topic_index -> chosen_frame_path}, plus the API usage object.
    Topics with no in-window candidates are omitted (caller falls back to timestamp-based pick).

    If `segments` is provided, the per-topic transcript slice is included so vision
    can ground picks on what the narrator is saying at each candidate's timestamp
    (e.g. "speaker says 'as you can see in this diagram' at 04:23 → frame at 04:23").
    """
    import anthropic
    from pydantic import BaseModel
    from typing import List as TList

    class TopicChoice(BaseModel):
        topic_index: int
        candidate_index: int
        rationale: str

    class FrameChoices(BaseModel):
        choices: TList[TopicChoice]

    topics = digest.topics

    # Build per-topic candidate lists and a flat list of (topic_idx, cand_idx, path, ts)
    per_topic: List[List[Tuple[Path, float]]] = []
    per_topic_transcript: List[str] = []
    for i, topic in enumerate(topics):
        end = topics[i + 1].start_time if i + 1 < len(topics) else video_duration
        per_topic.append(_candidates_for_topic(topic.start_time, end, frames))
        if segments:
            per_topic_transcript.append(
                _transcript_slice(segments, topic.start_time, end)
            )
        else:
            per_topic_transcript.append("")

    # Build the message: text intro -> for each topic, label + summary + numbered candidate images
    content: list = []
    intro = (
        "For each topic below, pick the candidate frame that best illustrates what the "
        "narrator is discussing. Prefer frames showing the most informative visual content "
        "(diagrams, code, distinctive UI) over generic framing or talking-head shots.\n\n"
        "When multiple candidates show the same scene at different stages of an animation "
        "or progressive reveal — bullets appearing one at a time, diagram elements being "
        "added, code typed line by line — prefer the LATEST candidate in the sequence. "
        "The final state shows the most complete information; partial/early states omit "
        "content the narrator goes on to add.\n\n"
        "Use the per-topic transcript to ground your pick: when the narrator says things "
        "like \"as you can see here\" or refers to a specific element at a specific "
        "moment, prefer the candidate whose timestamp is closest to that mention.\n\n"
        "Return one choice per topic that has candidates.\n\n"
        f"Total topics: {len(topics)}\n"
    )
    content.append({"type": "text", "text": intro})

    for ti, topic in enumerate(topics):
        cands = per_topic[ti]
        if not cands:
            content.append({
                "type": "text",
                "text": f"\n--- Topic {ti} (no candidates available — skip) ---\n"
                        f"Title: {topic.title}\n",
            })
            continue
        header_parts = [
            f"\n--- Topic {ti} ---",
            f"Title: {topic.title}",
            f"Summary: {topic.summary}",
        ]
        if per_topic_transcript[ti]:
            header_parts.append("Transcript:\n" + per_topic_transcript[ti])
        header_parts.append(f"Candidates ({len(cands)} frames):\n")
        content.append({"type": "text", "text": "\n".join(header_parts)})
        for ci, (path, ts) in enumerate(cands):
            content.append({
                "type": "text",
                "text": f"Candidate {ci} (at {format_timestamp(ts)}):",
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_frame_for_vision(path),
                },
            })

    system = (
        "You select illustrative frames for a video digest. You will be shown a list of "
        "topics, each with the topic's title, summary, the transcript spoken during that "
        "topic (timestamped), and a small set of candidate frames (also timestamped). "
        "For each topic with candidates, return the (topic_index, candidate_index) pair "
        "that best illustrates the topic, with a one-sentence rationale. Use the "
        "transcript to ground your choice on what the narrator is saying when each "
        "candidate frame was captured. Skip topics that say 'no candidates available'."
    )

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=FrameChoices,
    )

    chosen: dict = {}
    for choice in response.parsed_output.choices:
        if 0 <= choice.topic_index < len(topics):
            cands = per_topic[choice.topic_index]
            if 0 <= choice.candidate_index < len(cands):
                chosen[choice.topic_index] = cands[choice.candidate_index][0]
    return chosen, response.usage


def _pick_topic_frame(
    topic_start: float,
    topic_end: float,
    candidates: List[Tuple[Path, float]],
    used: set,
) -> Optional[Tuple[Path, float]]:
    """Pick the best frame for a topic: prefer a frame inside [start, end), else closest."""
    in_window = [(p, t) for p, t in candidates if topic_start <= t < topic_end and p not in used]
    if in_window:
        midpoint = (topic_start + min(topic_end, in_window[-1][1] + 1)) / 2
        return min(in_window, key=lambda x: abs(x[1] - midpoint))
    available = [(p, t) for p, t in candidates if p not in used]
    if not available:
        return None
    return min(available, key=lambda x: abs(x[1] - topic_start))


def write_markdown_digest(
    digest,
    candidate_frames: List[Tuple[Path, float]],
    video_duration: float,
    output_md: Path,
    images_dir: Path,
    vision_picks: Optional[dict] = None,
    video_title: Optional[str] = None,
    video_url: Optional[str] = None,
) -> None:
    """Render the digest as Markdown with <img> tags. Copies frames into images_dir.

    If vision_picks is provided, prefer those mappings; fall back to timestamp-based picks
    for any topic not covered.

    video_title (when provided) overrides the LLM-generated title so the digest's
    heading matches the original YouTube title — readers can map it back to the
    source. video_url renders a "Watch on YouTube" link directly under the title.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    used: set = set()
    topic_images: List[Optional[Path]] = []
    topics = digest.topics
    for i, topic in enumerate(topics):
        # Vision pick takes priority
        if vision_picks and i in vision_picks:
            src_path = vision_picks[i]
            used.add(src_path)
        else:
            end = topics[i + 1].start_time if i + 1 < len(topics) else video_duration
            pick = _pick_topic_frame(topic.start_time, end, candidate_frames, used)
            if pick is None:
                topic_images.append(None)
                continue
            src_path, _ = pick
            used.add(src_path)
        dest = images_dir / f"topic_{i + 1:02d}.jpg"
        shutil.copy(src_path, dest)
        topic_images.append(dest)

    rel_dir = images_dir.name  # render as "images/topic_NN.jpg"
    lines: List[str] = []
    heading = video_title if video_title else digest.title
    lines.append(f"# {heading}")
    lines.append("")
    if video_url:
        lines.append(f"**Watch on YouTube:** <{video_url}>")
        lines.append("")
    lines.append(digest.overview)
    lines.append("")
    for i, topic in enumerate(topics):
        ts = format_timestamp(topic.start_time)
        lines.append(f"## {topic.title}  <sub>*{ts}*</sub>")
        lines.append("")
        if topic_images[i] is not None:
            # Use the topic title as alt text — accessible to screen readers and
            # readable as a fallback if the image fails to load.
            alt = topic.title.replace('"', "'")
            lines.append(f'<img src="{rel_dir}/{topic_images[i].name}" alt="{alt}" width="800">')
            lines.append("")
        lines.append(topic.summary)
        lines.append("")
        for kp in topic.key_points:
            lines.append(f"- {kp}")
        if topic.key_points:
            lines.append("")

    output_md.write_text("\n".join(lines).rstrip() + "\n")


# ---------- Subcommands: watch / meta / serve ----------

LATEST_LIMIT = 10
MAX_NEW_PER_RUN = 3


def channels_file() -> Path:
    return get_data_dir() / "channels.txt"


def state_file() -> Path:
    return get_data_dir() / "state.json"


def read_channels() -> List[str]:
    p = channels_file()
    if not p.exists():
        return []
    return [
        line.strip()
        for line in p.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_channels(channels: List[str]) -> None:
    get_data_dir().mkdir(parents=True, exist_ok=True)
    body = "# YouTube channels to watch. One URL per line. Lines starting with # are ignored.\n"
    body += "\n".join(channels) + ("\n" if channels else "")
    channels_file().write_text(body)


def load_state() -> dict:
    p = state_file()
    if not p.exists():
        return {"channels": {}}
    return json.loads(p.read_text())


def save_state(state: dict) -> None:
    get_data_dir().mkdir(parents=True, exist_ok=True)
    state_file().write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


# ---- watch subcommands ----

def normalize_channel_url(url: str) -> str:
    """Light normalization for YouTube channel URLs.

    Accepts: '@handle', 'youtube.com/@handle', 'https://www.youtube.com/@handle/videos'.
    Always returns a fully-qualified URL. Adds '/videos' to bare @handle URLs so
    yt-dlp targets the videos tab specifically.
    """
    url = url.strip()
    if url.startswith("@"):
        url = f"https://www.youtube.com/{url}"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    m = re.match(r"^(https?://(?:www\.)?youtube\.com/@[^/]+)/?$", url)
    if m:
        url = m.group(1) + "/videos"
    return url


def cmd_watch_add(args) -> int:
    url = normalize_channel_url(args.url)
    if not is_url(url):
        sys.exit(f"Not a URL: {url}")
    channels = read_channels()
    if url in channels:
        print(f"Already watching: {url}")
        return 0
    channels.append(url)
    write_channels(channels)
    print(f"Added: {url}")
    return 0


def cmd_watch_list(args) -> int:
    channels = read_channels()
    if not channels:
        print("No channels configured. Add one with: yt2md watch add <URL>")
        return 0
    print(f"Watching {len(channels)} channel(s):")
    for ch in channels:
        print(f"  {ch}")
    print(f"\nConfig: {channels_file()}")
    print(f"State:  {state_file()}")
    print(f"Data:   {get_data_dir()}")
    return 0


def cmd_watch_remove(args) -> int:
    url = args.url.strip()
    channels = read_channels()
    if url not in channels:
        sys.exit(f"Not in list: {url}")
    channels = [c for c in channels if c != url]
    write_channels(channels)
    print(f"Removed: {url}")
    return 0


def _list_channel_videos(url: str, limit: int = LATEST_LIMIT) -> List[str]:
    out = subprocess.check_output(
        ["yt-dlp", "--flat-playlist", "--playlist-end", str(limit), "--print", "%(id)s", url],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _digest_video(video_id: str, output_dir: Path) -> Tuple[int, str]:
    """Run yt2md on a video. Returns (exit_code, combined_stdout_stderr).

    Streams output to the parent's stdout in real time (so poll.log captures
    it as it happens) AND collects it into a buffer the caller can scan for
    permanent-failure patterns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    digest_path = output_dir / "digest.md"
    yt2md = shutil.which("yt2md") or sys.argv[0]
    proc = subprocess.Popen(
        [yt2md, f"https://youtu.be/{video_id}", "-o", str(digest_path)],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chunks: list = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


# Substrings (case-insensitive) that indicate a video can never be digested by
# this account — gating it forever, deleted, etc. Matches mark the video
# as seen so polling stops re-trying it. Network/transient errors don't match
# and continue to retry next cycle.
_PERMANENT_FAILURE_PATTERNS = (
    "members-only",
    "join this channel",
    "private video",
    "video unavailable",
    "this video is no longer available",
    "removed by the uploader",
    "removed for violating",
    "sign in to confirm your age",
    "video has been removed",
)


def _is_permanent_failure(output: str) -> bool:
    low = output.lower()
    return any(p in low for p in _PERMANENT_FAILURE_PATTERNS)


def cmd_watch_run(args) -> int:
    ensure_api_key()
    channels = read_channels()
    if not channels:
        print("No channels configured. Add one with: yt2md watch add <URL>")
        return 0

    data_dir = get_data_dir()
    digests_dir = data_dir / "digests"
    state = load_state()
    any_failures = False

    for channel_url in channels:
        print(f"--- {channel_url}")
        seen = set(state["channels"].get(channel_url, {}).get("seen", []))
        latest_ids = _list_channel_videos(channel_url)

        if not seen:
            print(f"  first run, seeding state with {len(latest_ids)} videos (no backfill)")
            state["channels"][channel_url] = {"seen": sorted(latest_ids)}
            continue

        new_ids = [vid for vid in latest_ids if vid not in seen][:MAX_NEW_PER_RUN]
        if not new_ids:
            print("  no new videos")
            continue

        print(f"  {len(new_ids)} new: {new_ids}")
        for vid in reversed(new_ids):
            print(f"  processing {vid}...")
            rc, output = _digest_video(vid, digests_dir / vid)
            if rc == 0:
                seen.add(vid)
                state["channels"][channel_url] = {"seen": sorted(seen)}
                save_state(state)
            elif _is_permanent_failure(output):
                # Permanent: mark seen so polling stops cycling on it.
                # Wipe the partial dir (mp4 download, empty digest, etc.) to
                # keep digests/ tidy.
                print(f"  PERMANENTLY UNAVAILABLE: {vid} — marking seen and wiping partial dir")
                shutil.rmtree(digests_dir / vid, ignore_errors=True)
                seen.add(vid)
                state["channels"][channel_url] = {"seen": sorted(seen)}
                save_state(state)
            else:
                print(f"  FAILED on {vid} (transient — will retry next poll)", file=sys.stderr)
                any_failures = True

    save_state(state)
    return 1 if any_failures else 0


# ---- in-process scheduler ----
#
# Cadenced background runner for `yt2md watch run` (subscription poll). Lives
# inside `yt2md serve`: a daemon thread ticks every ~30s, fires due jobs as
# detached subprocesses, tracks pid + exit code in schedule_state.json for
# the /schedule UI.
#
# Tradeoff: scheduling pauses while serve is down — for an interactively-used
# reader this is fine; missed slots fire on next start (catch-up semantics).

DEFAULT_SCHEDULE_CONFIG = {
    "poll_interval_hours": 6,
}

_SCHED_TICK_SECS = 30
_scheduler_jobs: dict = {}
_scheduler_thread = None
_scheduler_lock_obj = None


def _schedule_lock():
    global _scheduler_lock_obj
    if _scheduler_lock_obj is None:
        import threading
        _scheduler_lock_obj = threading.Lock()
    return _scheduler_lock_obj


def _schedule_config_file() -> Path:
    return get_data_dir() / "schedule.json"


def _schedule_state_file() -> Path:
    return get_data_dir() / "schedule_state.json"


def load_schedule_config() -> dict:
    p = _schedule_config_file()
    if not p.exists():
        return dict(DEFAULT_SCHEDULE_CONFIG)
    try:
        cfg = json.loads(p.read_text())
        merged = dict(DEFAULT_SCHEDULE_CONFIG)
        merged.update({k: v for k, v in cfg.items() if k in DEFAULT_SCHEDULE_CONFIG})
        return merged
    except Exception:
        return dict(DEFAULT_SCHEDULE_CONFIG)


def save_schedule_config(cfg: dict) -> None:
    get_data_dir().mkdir(parents=True, exist_ok=True)
    _schedule_config_file().write_text(json.dumps(cfg, indent=2) + "\n")


def _load_schedule_state() -> dict:
    p = _schedule_state_file()
    default = {"poll": {}}
    if not p.exists():
        return default
    try:
        s = json.loads(p.read_text())
        return {"poll": s.get("poll") or {}}
    except Exception:
        return default


def _save_schedule_state(state: dict) -> None:
    p = _schedule_state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(p)


# ---- user-tunable settings (model + tooling choices) ----
#
# Lives at ~/yt2md/settings.json. Editable via the /settings page; flows into
# every subprocess spawn (one-off, scheduled poll) as YT2MD_* env vars, so the
# digest CLI's argparse defaults pick them up. The /digests/<id>/discuss route
# reads settings directly (it runs in-process).

DEFAULT_SETTINGS = {
    "digest_model": "claude-sonnet-4-6",
    "panel_model": "claude-opus-4-7",
    "whisper_model": DEFAULT_WHISPER_MODEL,
    "cookies_from_browser": "",
    # "auto" = write the digest in the same language as the transcript;
    # "en" = always translate to English. Applies to both the per-video digest
    # and the panel discussion.
    "digest_language": "auto",
}


def _settings_file() -> Path:
    return get_data_dir() / "settings.json"


def load_settings() -> dict:
    p = _settings_file()
    if not p.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        s = json.loads(p.read_text())
        merged = dict(DEFAULT_SETTINGS)
        merged.update({k: v for k, v in s.items() if k in DEFAULT_SETTINGS})
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s: dict) -> None:
    get_data_dir().mkdir(parents=True, exist_ok=True)
    tmp = _settings_file().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, indent=2) + "\n")
    tmp.replace(_settings_file())


def _settings_to_env(settings: dict) -> dict:
    """Map settings to YT2MD_* env vars for subprocess invocation. Empty values
    are dropped so the subprocess sees only meaningful overrides."""
    out: dict = {}
    if settings.get("digest_model"):
        out["YT2MD_DIGEST_MODEL"] = settings["digest_model"]
    if settings.get("whisper_model"):
        out["YT2MD_WHISPER_MODEL"] = settings["whisper_model"]
    if settings.get("cookies_from_browser"):
        out["YT2MD_COOKIES_FROM_BROWSER"] = settings["cookies_from_browser"]
    if settings.get("panel_model"):
        out["YT2MD_PANEL_MODEL"] = settings["panel_model"]
    if settings.get("digest_language"):
        out["YT2MD_DIGEST_LANGUAGE"] = settings["digest_language"]
    return out


def _format_schedule_summary(cfg: dict) -> str:
    """Human-readable description of the schedule config."""
    poll = cfg["poll_interval_hours"]
    poll_str = f"every {poll} hour{'s' if poll != 1 else ''}"
    return f"polling {poll_str}"


def _compute_next_poll(cfg: dict, last_started_at: Optional[float]) -> float:
    """Timestamp of the next scheduled poll. First-run convention: fire ASAP."""
    interval = max(60.0, float(cfg.get("poll_interval_hours", 6)) * 3600.0)
    if last_started_at is None:
        import time as _t
        return _t.time()
    return last_started_at + interval


def _fire_scheduled_job(kind: str) -> Optional[subprocess.Popen]:
    """Spawn yt2md as a subprocess for the given kind. Returns the running
    Popen or None if already running / yt2md not on PATH. Caller holds lock."""
    if kind != "poll":
        return None
    existing = _scheduler_jobs.get(kind)
    if existing is not None and existing.poll() is None:
        return existing
    yt2md_path = shutil.which("yt2md")
    if not yt2md_path:
        print(f"[scheduler] yt2md not on PATH; cannot fire {kind}", file=sys.stderr)
        return None
    args = [yt2md_path, "watch", "run"]
    log_path = get_data_dir() / "logs" / f"{kind}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import time as _t
    log_fd = open(log_path, "a")
    log_fd.write(f"\n===== {_t.strftime('%Y-%m-%d %H:%M:%S')} {kind} run =====\n")
    log_fd.flush()
    proc = subprocess.Popen(
        args,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        env={**os.environ, **_settings_to_env(load_settings()), "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    log_fd.close()
    _scheduler_jobs[kind] = proc

    state = _load_schedule_state()
    state[kind] = {
        **(state.get(kind) or {}),
        "last_started_at": _t.time(),
        "last_pid": proc.pid,
        "last_exit_code": None,
        "last_finished_at": None,
    }
    _save_schedule_state(state)
    return proc


def _reap_scheduled_job(kind: str) -> None:
    """Capture exit code if the kind's subprocess has exited. Caller holds lock."""
    proc = _scheduler_jobs.get(kind)
    if proc is None:
        return
    rc = proc.poll()
    if rc is None:
        return
    import time as _t
    state = _load_schedule_state()
    state[kind] = {
        **(state.get(kind) or {}),
        "last_finished_at": _t.time(),
        "last_exit_code": rc,
    }
    _save_schedule_state(state)
    del _scheduler_jobs[kind]


def _scheduler_tick() -> None:
    with _schedule_lock():
        _reap_scheduled_job("poll")
        cfg = load_schedule_config()
        state = _load_schedule_state()
        import time as _t
        now = _t.time()
        if "poll" not in _scheduler_jobs:
            if now >= _compute_next_poll(cfg, (state.get("poll") or {}).get("last_started_at")):
                _fire_scheduled_job("poll")


def _scheduler_loop() -> None:
    import time as _t
    while True:
        try:
            _scheduler_tick()
        except Exception as e:
            print(f"[scheduler] tick error: {e}", file=sys.stderr)
        _t.sleep(_SCHED_TICK_SECS)


def start_scheduler() -> None:
    """Start the daemon scheduler thread. Idempotent."""
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return
    import threading
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    print("[scheduler] started (in-process; ticks every "
          f"{_SCHED_TICK_SECS}s)")


def _cleanup_legacy_launchd() -> None:
    """Best-effort one-time removal of the old launchctl plists from
    ~/Library/LaunchAgents — so the user doesn't end up with duplicate
    scheduling after this migration. Silent if nothing is present."""
    launchd_dir = Path.home() / "Library" / "LaunchAgents"
    removed = []
    for label in ("com.youtube-to-markdown.poll", "com.youtube-to-markdown.meta"):
        plist = launchd_dir / f"{label}.plist"
        if plist.exists():
            subprocess.run(
                ["launchctl", "bootout", f"gui/{os.getuid()}", str(plist)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                plist.unlink()
                removed.append(label)
            except OSError:
                pass
    if removed:
        print(f"[scheduler] removed legacy launchd plists: {', '.join(removed)}")


def _scheduler_status_summary(kind: str, state: dict) -> Tuple[str, str]:
    """English summary + dot class for /schedule and /channels surfaces."""
    s = state.get(kind) or {}
    if kind in _scheduler_jobs:
        return ("Running now.", "dot-on")
    last_started = s.get("last_started_at")
    last_exit = s.get("last_exit_code")
    if last_started is None:
        return ("Set up — first run hasn't happened yet.", "dot-warn")
    if last_exit not in (0, None):
        return (f"Last run failed (exit code {last_exit}). Check the log.", "dot-warn")
    import datetime as dt
    age = dt.datetime.now() - dt.datetime.fromtimestamp(last_started)
    age_secs = int(age.total_seconds())
    if age_secs < 60:
        age_str = f"{age_secs}s ago"
    elif age_secs < 3600:
        age_str = f"{age_secs // 60}m ago"
    elif age_secs < 86400:
        age_str = f"{age_secs // 3600}h ago"
    else:
        age_str = f"{age_secs // 86400}d ago"
    return (f"Healthy — last run {age_str} (clean).", "dot-on")


def _format_next_run(ts: float) -> str:
    """Render an upcoming-run timestamp as 'in 2h 15m' / 'in 3 days' / 'overdue'."""
    import time as _t
    delta = ts - _t.time()
    if delta < 0:
        return "due now"
    if delta < 60:
        return f"in {int(delta)}s"
    if delta < 3600:
        return f"in {int(delta // 60)}m"
    if delta < 86400:
        h = int(delta // 3600)
        m = int((delta % 3600) // 60)
        return f"in {h}h {m}m"
    return f"in {int(delta // 86400)}d"


def _tail_log(path: Path, n: int = 20) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        return f"(error reading log: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"




# ---- one-off digest jobs (in-memory tracking; cheap) ----

# Module-level dict: PID -> {"video_id": str, "started": float, "url": str, "proc": Popen}.
# Lost on server restart, which is fine — the digest still completes (detached
# subprocess) and shows up in the sidebar when done.
_oneoff_jobs: dict = {}

# Recent failures, most-recent-first, capped at _ONEOFF_FAILURE_CAP.
# Lost on server restart (matches _oneoff_jobs lifecycle).
_oneoff_failures: list = []
_ONEOFF_FAILURE_CAP = 20


_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    """Pull a YouTube video ID from common URL forms. Returns '' if not found."""
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else ""


def _extract_last_error(log_path: Path, video_id: str) -> str:
    """Pull the most relevant error line from oneoff.log for a given video_id.

    The log uses '===== {ts} starting {video_id} ({url}) =====' as section
    markers. We bound the section, then prefer 'RuntimeError: ...' / similar
    summary lines over the bare 'Traceback' header.
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return ""
    marker = f"starting {video_id}"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    next_idx = text.find("\n===== ", idx + len(marker))
    section = text[idx:next_idx if next_idx > 0 else len(text)]
    candidates = [
        ln.strip() for ln in section.splitlines()
        if ln.strip()
        and not ln.lstrip().startswith("[download")
        and ("Error" in ln or "Traceback" in ln)
    ]
    for ln in reversed(candidates):
        if ":" in ln and not ln.startswith("Traceback"):
            return ln
    return candidates[-1] if candidates else ""


# Pipeline stage markers, in pipeline order. The status endpoint scans the log
# section for the LAST matching substring to determine the current stage.
# Reordering here changes which stage is reported when two markers happen to
# match — keep this list in pipeline order (later entries override earlier).
_ONEOFF_STAGE_MARKERS: list = [
    ("starting",            "starting "),  # section header is always present
    ("downloading",         "[0/5] Fetching YouTube video"),
    ("downloading",         "[download]"),
    ("loading whisper",     "loading whisper model"),
    ("transcribing",        "transcribing audio with whisper"),
    ("extracting frames",   "[1/5] Extracting frames"),
    ("deduping frames",     "[2/5] Deduping"),
    ("parsing transcript",  "[3/5] Parsing SRT"),
    ("aligning",            "[4/5] Aligning"),
    ("building deck",       "[5/5] Building deck"),
    ("digesting",           "[+] Generating digest"),
    ("vision pass",         "[+] Vision-picking"),
    ("writing digest",      "Digest written"),
]


def _describe_job_stage(log_text: str, video_id: str) -> str:
    """Return a short human-readable label for the latest stage of a job.

    Scans the log section bounded by the start marker for video_id (or EOF /
    next start marker) and returns the most pipeline-advanced stage whose
    substring marker appears in that section.
    """
    marker = f"starting {video_id}"
    idx = log_text.rfind(marker)
    if idx < 0:
        return "starting"
    next_idx = log_text.find("\n===== ", idx + len(marker))
    section = log_text[idx:next_idx if next_idx > 0 else len(log_text)]
    current = "starting"
    for label, needle in _ONEOFF_STAGE_MARKERS:
        if needle in section:
            current = label
    return current


def _extract_run_summary(log_text: str, video_id: str) -> Optional[dict]:
    """Pull the last `[summary] {...}` JSON line emitted by the pipeline for a job.

    The pipeline prints one line of the form `[summary] {...}` on successful
    completion. Returns the parsed dict, or None if absent / malformed.
    """
    import json as _json
    marker = f"starting {video_id}"
    idx = log_text.rfind(marker)
    if idx < 0:
        return None
    next_idx = log_text.find("\n===== ", idx + len(marker))
    section = log_text[idx:next_idx if next_idx > 0 else len(log_text)]
    last = None
    for ln in section.splitlines():
        s = ln.lstrip()
        if s.startswith("[summary] "):
            last = s[len("[summary] "):]
    if not last:
        return None
    try:
        return _json.loads(last)
    except Exception:
        return None


def _runs_jsonl_path() -> Path:
    return get_data_dir() / "logs" / "runs.jsonl"


def _record_run(row: dict) -> None:
    """Persist a run completion to library.db AND append a JSONL line.

    Two stores by design: SQLite for the activity UI's queries; JSONL for
    `tail -f` debugging and downstream scripts. Both reflect the same data.
    """
    import json as _json
    cols = (
        "video_id", "url", "source", "started_at", "ended_at", "duration_secs",
        "exit_code", "success", "stage_reached", "error", "source_lang",
        "used_whisper", "whisper_model",
        "download_secs", "whisper_secs", "frames_secs", "digest_secs", "vision_secs",
        "digest_input_tokens", "digest_output_tokens",
        "digest_cache_read_tokens", "digest_cache_creation_tokens",
        "digest_path",
    )
    placeholders = ", ".join("?" for _ in cols)
    values = tuple(row.get(c) for c in cols)
    try:
        with _library_connect() as conn:
            conn.execute(
                f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
    except Exception as e:
        # DB failure shouldn't bring down the reaper. JSONL still gets written.
        print(f"[runs] db insert failed: {e}", file=sys.stderr)

    jsonl = _runs_jsonl_path()
    try:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with jsonl.open("a") as fh:
            fh.write(_json.dumps(row) + "\n")
    except OSError as e:
        print(f"[runs] jsonl append failed: {e}", file=sys.stderr)


def _recent_runs(limit: int = 100) -> list:
    """Read the last `limit` runs from SQLite, newest first."""
    try:
        with _library_connect() as conn:
            conn.row_factory = __import__("sqlite3").Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _build_run_row(info: dict, exit_code: int, log_text: str) -> dict:
    """Translate a finished job + its log into a row dict suitable for runs table."""
    import time as _t
    video_id = info["video_id"]
    started = info["started"]
    ended = _t.time()
    summary = _extract_run_summary(log_text, video_id)
    success = exit_code == 0 and summary is not None
    error = "" if success else _extract_last_error(_oneoff_log_path(), video_id)
    stage = _describe_job_stage(log_text, video_id)
    timings = (summary or {}).get("timings") or {}
    tokens = (summary or {}).get("tokens") or {}
    return {
        "video_id": video_id,
        "url": info.get("url"),
        "source": "oneoff",
        "started_at": started,
        "ended_at": ended,
        "duration_secs": ended - started,
        "exit_code": exit_code,
        "success": 1 if success else 0,
        "stage_reached": stage,
        "error": error or None,
        "source_lang": (summary or {}).get("source_lang"),
        "used_whisper": 1 if (summary or {}).get("used_whisper") else 0,
        "whisper_model": (summary or {}).get("whisper_model"),
        "download_secs": timings.get("download"),
        "whisper_secs": timings.get("whisper"),
        "frames_secs": timings.get("frames"),
        "digest_secs": timings.get("digest"),
        "vision_secs": timings.get("vision"),
        "digest_input_tokens": tokens.get("input"),
        "digest_output_tokens": tokens.get("output"),
        "digest_cache_read_tokens": tokens.get("cache_read"),
        "digest_cache_creation_tokens": tokens.get("cache_creation"),
        "digest_path": (summary or {}).get("digest_path"),
    }


def _oneoff_log_path() -> Path:
    return get_data_dir() / "logs" / "oneoff.log"


def _record_oneoff_failure(info: dict, exit_code: int) -> None:
    import time as _t
    log_path = _oneoff_log_path()
    last_error = _extract_last_error(log_path, info["video_id"]) if log_path.exists() else ""
    _oneoff_failures.insert(0, {
        "video_id": info["video_id"],
        "url": info["url"],
        "exit_code": exit_code,
        "started": info["started"],
        "ended": _t.time(),
        "error": last_error,
    })
    del _oneoff_failures[_ONEOFF_FAILURE_CAP:]


def _list_active_oneoff_jobs() -> list:
    """Return one-off jobs whose subprocesses are still alive.

    Side effect: jobs that have exited are removed from _oneoff_jobs;
    non-zero exits are appended to _oneoff_failures.
    """
    active = []
    for pid in list(_oneoff_jobs.keys()):
        info = _oneoff_jobs[pid]
        proc = info.get("proc")
        if proc is None:
            # legacy entries with no Popen handle — fall back to kill probe
            try:
                os.kill(pid, 0)
                active.append({"pid": pid, **{k: v for k, v in info.items() if k != "proc"}})
            except (ProcessLookupError, PermissionError):
                del _oneoff_jobs[pid]
            continue
        rc = proc.poll()
        if rc is None:
            active.append({"pid": pid, **{k: v for k, v in info.items() if k != "proc"}})
            continue
        del _oneoff_jobs[pid]
        # Treat any non-zero exit, or zero-exit-with-no-digest, as a failure.
        digest_path = get_data_dir() / "digests" / info["video_id"] / "digest.md"
        try:
            log_text = _oneoff_log_path().read_text(errors="replace")
        except OSError:
            log_text = ""
        if rc != 0 or not digest_path.exists():
            _record_oneoff_failure(info, rc)
        _record_run(_build_run_row(info, rc, log_text))
    return active


def _list_recent_oneoff_failures() -> list:
    """Return a copy of recent failures (most recent first)."""
    return list(_oneoff_failures)


# ---- read-state library (SQLite-backed) ----

def _library_path() -> Path:
    return get_data_dir() / "library.db"


def _library_connect():
    """Open (and lazily migrate) the read-state SQLite database."""
    import sqlite3
    get_data_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_library_path())
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS digest_reads (
            digest_id TEXT PRIMARY KEY,
            opened_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            url TEXT,
            source TEXT NOT NULL,           -- 'oneoff' | 'poll' | 'meta'
            started_at REAL NOT NULL,
            ended_at REAL NOT NULL,
            duration_secs REAL NOT NULL,
            exit_code INTEGER NOT NULL,
            success INTEGER NOT NULL,
            stage_reached TEXT,
            error TEXT,
            source_lang TEXT,
            used_whisper INTEGER NOT NULL DEFAULT 0,
            whisper_model TEXT,
            download_secs REAL,
            whisper_secs REAL,
            frames_secs REAL,
            digest_secs REAL,
            vision_secs REAL,
            digest_input_tokens INTEGER,
            digest_output_tokens INTEGER,
            digest_cache_read_tokens INTEGER,
            digest_cache_creation_tokens INTEGER,
            digest_path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
    """)
    return conn


def _mark_digest_read(digest_id: str) -> None:
    import time
    with _library_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO digest_reads(digest_id, opened_at) VALUES (?, ?)",
            (digest_id, int(time.time())),
        )


def _read_digest_ids() -> set:
    with _library_connect() as conn:
        rows = conn.execute("SELECT digest_id FROM digest_reads").fetchall()
        return {r[0] for r in rows}


# ---- serve subcommand (local reader UI) ----

SERVE_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} — yt2md</title>
{% if base_href %}<base href="{{ base_href }}">{% endif %}
<script>
(function() {
  const stored = localStorage.getItem('yt2md-theme');
  if (stored && stored !== 'auto') document.documentElement.setAttribute('data-theme', stored);
})();
</script>
<style>
:root {
  --bg: #fafaf7;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #b65a2c;
  --unread: #2563eb;
  --border: #e5e3dc;
  --sidebar-bg: #f0eee6;
  --code-bg: #ececea;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1a1a1a;
    --fg: #e8e8e8;
    --muted: #999;
    --accent: #d97a4d;
    --unread: #60a5fa;
    --border: #2e2e2e;
    --sidebar-bg: #141414;
    --code-bg: #232323;
  }
}
:root[data-theme="dark"] {
  --bg: #1a1a1a;
  --fg: #e8e8e8;
  --muted: #999;
  --accent: #d97a4d;
  --unread: #60a5fa;
  --border: #2e2e2e;
  --sidebar-bg: #141414;
  --code-bg: #232323;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  display: flex;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
}
aside {
  width: 320px;
  flex-shrink: 0;
  height: 100vh;
  overflow-y: auto;
  padding: 20px 16px;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
}
aside h1 { margin: 0 0 16px; font-size: 18px; }
aside h1 a { color: var(--fg); text-decoration: none; }
aside h2 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 20px 0 8px;
  font-weight: 600;
}
aside ul { list-style: none; padding: 0; margin: 0; }
aside li { margin: 0 0 3px; }
aside li a {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  padding: 6px 8px;
  border-radius: 4px;
  color: var(--fg);
  text-decoration: none;
  font-size: 13px;
  line-height: 1.35;
}
aside li a:hover { background: rgba(0,0,0,0.05); }
@media (prefers-color-scheme: dark) {
  aside li a:hover { background: rgba(255,255,255,0.05); }
}
aside li.active a { background: var(--accent); color: white; }
aside li.unread a { font-weight: 600; }
aside .empty { color: var(--muted); font-size: 13px; padding: 6px 8px; }
aside .unread-count {
  display: inline-block; padding: 2px 7px; background: var(--unread); color: white;
  border-radius: 10px; font-size: 11px; font-weight: 600;
  text-transform: none; letter-spacing: 0; margin-left: 4px;
}
aside .unread-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--unread); margin-right: 6px; vertical-align: middle;
}
aside .meta-card.unread { border-color: var(--unread); }
aside .meta-card.unread .week { font-weight: 700; }
/* When an item is the currently-viewed one, suppress the unread signals
   to avoid two competing color cues. */
aside li.active .unread-dot,
aside .meta-card.active .unread-dot { display: none; }
aside .meta-card.active.unread { border-color: var(--accent); }
aside .meta-card {
  display: block; padding: 10px 12px; border-radius: 4px;
  border: 1px solid var(--border); margin-bottom: 6px;
  text-decoration: none; color: var(--fg);
}
aside .meta-card:hover { border-color: var(--accent); }
aside .meta-card.active { background: var(--accent); color: white; border-color: var(--accent); }
aside .meta-card .week { font-weight: 600; font-size: 13px; line-height: 1.2; }
aside .meta-card .count {
  color: var(--muted); font-size: 11px; margin-top: 2px;
  text-transform: uppercase; letter-spacing: 0.04em;
}
aside .meta-card.active .count { color: white; opacity: 0.85; }
main {
  flex: 1;
  height: 100vh;
  overflow-y: auto;
  padding: 32px 48px 80px;
}
main .reader {
  max-width: 720px;
  margin: 0 auto;
}
main h1 { font-size: 28px; line-height: 1.25; margin-top: 0; }
main h2 { font-size: 22px; line-height: 1.3; margin-top: 32px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
main h3 { font-size: 17px; }
main img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 4px; }
main a { color: var(--accent); }
main code { background: var(--code-bg); padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
main pre { background: var(--code-bg); padding: 12px; border-radius: 4px; overflow-x: auto; }
main pre code { background: none; padding: 0; }
main blockquote { border-left: 3px solid var(--border); padding-left: 16px; color: var(--muted); margin-left: 0; }
main ul, main ol { padding-left: 24px; }
main hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
main sub { color: var(--muted); font-size: 0.85em; }
.empty-state { color: var(--muted); margin-top: 80px; text-align: center; }
.meta-info { color: var(--muted); font-size: 13px; margin-top: -12px; margin-bottom: 24px; }
.featured-eyebrow {
  color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.06em; margin-bottom: 24px;
}
.featured-eyebrow a { color: var(--accent); text-decoration: none; font-weight: 600; }
.cta {
  display: inline-block; background: var(--accent); color: white;
  padding: 12px 20px; border-radius: 4px; text-decoration: none;
  font-weight: 500; margin-top: 12px;
}
.cta:hover { opacity: 0.9; }
.recent-list { list-style: none; padding: 0; margin: 12px 0; }
.recent-list li { padding: 6px 0; border-bottom: 1px solid var(--border); }
.recent-list a { color: var(--fg); text-decoration: none; }
.recent-list a:hover { color: var(--accent); }
.add-form { display: flex; gap: 8px; margin: 24px 0; }
.add-form input[type="text"] {
  flex: 1; padding: 10px 12px; font-size: 14px;
  border: 1px solid var(--border); border-radius: 4px;
  background: var(--bg); color: var(--fg);
}
.add-form button, .channel-list button {
  padding: 10px 16px; font-size: 14px;
  border: 1px solid var(--accent); border-radius: 4px;
  background: var(--accent); color: white; cursor: pointer;
}
.channel-list { list-style: none; padding: 0; }
.channel-list li {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px;
  margin-bottom: 8px; gap: 12px;
}
.channel-list .url { flex: 1; word-break: break-all; font-size: 14px; }
.channel-list button {
  background: transparent; color: var(--muted);
  border-color: var(--border); padding: 6px 12px; font-size: 12px;
}
.channel-list button:hover { color: var(--accent); border-color: var(--accent); }
.flash { padding: 10px 14px; border-radius: 4px; margin: 16px 0;
  background: var(--code-bg); border-left: 3px solid var(--accent); }
.next-step {
  padding: 14px 18px; border-radius: 6px; margin: 16px 0 24px;
  background: var(--sidebar-bg); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); font-size: 14px; line-height: 1.5;
}
.next-step strong { color: var(--accent); }
.next-step a { color: var(--accent); }
.schedule-form {
  background: var(--sidebar-bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 16px 20px; margin: 16px 0;
}
.schedule-fields {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px 20px; margin-bottom: 16px;
}
.schedule-fields label {
  display: flex; flex-direction: column; gap: 4px;
  font-size: 12px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.04em;
}
.schedule-fields input, .schedule-fields select {
  padding: 8px 10px; font-size: 14px; border: 1px solid var(--border);
  border-radius: 4px; background: var(--bg); color: var(--fg);
  font-family: inherit; text-transform: none; letter-spacing: normal;
}
.schedule-fields .suffix {
  position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
  color: var(--muted); font-size: 12px; pointer-events: none;
}
.status-table { width: 100%; border-collapse: collapse; font-size: 13px;
  margin: 12px 0 16px; }
.status-table td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
.status-table td:first-child { color: var(--muted); width: 140px; }
.status-table td:last-child { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.activity-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }
.activity-table th, .activity-table td {
  text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.activity-table th { color: var(--muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.04em; }
.activity-table td a { color: var(--accent); text-decoration: none; }
.activity-table td a:hover { text-decoration: underline; }
.activity-table .ok { color: #4a9f56; font-weight: 600; }
.activity-table .fail { color: #d04545; font-weight: 600; }
.activity-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
.activity-error { font-family: ui-monospace, "SF Mono", Menlo, monospace; word-break: break-word; }
.activity-stages, .activity-tokens { font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px; white-space: nowrap; }
.filter-row { display: flex; gap: 8px; margin: 12px 0 4px; flex-wrap: wrap; }
.filter-chip {
  padding: 5px 12px; border: 1px solid var(--border); border-radius: 999px;
  font-size: 12px; color: var(--fg); text-decoration: none; background: var(--bg);
}
.filter-chip:hover { border-color: var(--accent); }
.filter-chip.active { background: var(--accent); color: white; border-color: var(--accent); }
.filter-chip-count { opacity: 0.7; }
.delete-btn {
  padding: 6px 14px; font-size: 13px; cursor: pointer;
  background: transparent; color: #b13030;
  border: 1px solid #b13030; border-radius: 4px;
}
.delete-btn:hover { background: #b13030; color: white; }
.digest-actions { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.digest-toolbar { margin-top: -8px; margin-bottom: 8px; }
.digest-toolbar .delete-btn { margin-left: auto; }  /* push delete to the far right */
.discuss-btn {
  padding: 8px 16px; font-size: 13px; cursor: pointer;
  background: var(--accent); color: white;
  border: 1px solid var(--accent); border-radius: 4px;
  text-decoration: none; display: inline-block;
}
.discuss-btn:hover { opacity: 0.9; }
.discuss-btn-secondary {
  padding: 6px 12px; font-size: 12px; cursor: pointer;
  background: transparent; color: var(--fg);
  border: 1px solid var(--border); border-radius: 4px;
}
.discuss-btn-secondary:hover { border-color: var(--accent); }
.job-block { border: 1px solid var(--border); border-radius: 4px; padding: 16px 20px;
  margin: 16px 0; background: var(--sidebar-bg); }
.job-block h3 { margin: 0 0 8px; font-size: 15px; }
.job-actions { display: flex; gap: 8px; margin: 8px 0 16px; flex-wrap: wrap; }
.job-actions button {
  padding: 8px 14px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--border); border-radius: 4px;
  background: var(--bg); color: var(--fg);
}
.job-actions button.primary { background: var(--accent); color: white; border-color: var(--accent); }
.job-actions button:hover { border-color: var(--accent); }
.job-summary { font-size: 15px; margin: 0 0 12px; }
details summary {
  cursor: pointer; color: var(--muted); font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.04em;
  padding: 4px 0; user-select: none;
}
details summary:hover { color: var(--accent); }
details[open] summary { margin-bottom: 8px; }
.log-block {
  background: var(--code-bg); padding: 12px; border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px; line-height: 1.4; max-height: 240px;
  overflow: auto; white-space: pre-wrap; word-break: break-all;
}
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle; }
.dot-on { background: #4caf50; }
.dot-off { background: #999; }
.dot-warn { background: #d97a4d; }

/* Accessibility: skip to content for keyboard users */
.skip-link {
  position: absolute; left: -1000px; top: 0; padding: 8px 12px;
  background: var(--accent); color: white; text-decoration: none;
  border-radius: 0 0 4px 0; z-index: 100;
}
.skip-link:focus { left: 0; }

/* Form labels (visually hidden, exposed to screen readers) */
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}

/* Sidebar header (yt2md title + theme toggle inline) */
.sidebar-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 16px;
}
.sidebar-header h1 { margin: 0; }
.theme-toggle {
  background: transparent; border: 1px solid var(--border); border-radius: 4px;
  padding: 4px 8px; font-size: 14px; cursor: pointer; color: var(--fg);
  font-family: inherit; line-height: 1;
}
.theme-toggle:hover { border-color: var(--accent); }

/* Mobile: stack sidebar above main, slimmer padding */
@media (max-width: 720px) {
  body { flex-direction: column; }
  aside { width: 100%; height: auto; max-height: 40vh; border-right: none; border-bottom: 1px solid var(--border); }
  main { height: auto; padding: 24px 20px 60px; }
  main .reader { max-width: 100%; }
}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
<aside>
  <div class="sidebar-header">
    <h1><a href="/">yt2md</a></h1>
    <button class="theme-toggle" type="button" onclick="cycleTheme()" aria-label="Cycle theme: auto / light / dark">🌓</button>
  </div>

  <nav aria-label="Per-video digests">
  <h2>Digests {% if unread_digest_count %}<span class="unread-count">{{ unread_digest_count }} new</span>{% else %}({{ digests|length }}){% endif %}</h2>
  <ul>
    {% for d in digests %}
    <li{% if current == 'digest:' + d.id %} class="active"{% endif %}{% if d.unread %} class="unread"{% endif %}>
      <a href="/digests/{{ d.id }}/">{% if d.unread %}<span class="unread-dot" aria-label="unread"></span>{% endif %}{{ d.title }}</a>
    </li>
    {% else %}
    <li class="empty">none yet</li>
    {% endfor %}
  </ul>
  </nav>

  <nav aria-label="Manage">
  <h2>Manage</h2>
  <ul>
    <li{% if current == 'channels' %} class="active"{% endif %}><a href="/channels">Subscriptions ({{ channel_count }})</a></li>
    <li{% if current == 'one-off' %} class="active"{% endif %}><a href="/one-off">One-off digest</a></li>
    <li{% if current == 'schedule' %} class="active"{% endif %}><a href="/schedule">Schedule</a></li>
    <li{% if current == 'activity' %} class="active"{% endif %}><a href="/activity">Activity</a></li>
    <li{% if current == 'settings' %} class="active"{% endif %}><a href="/settings">Settings</a></li>
  </ul>
  </nav>
</aside>
<main id="main-content" tabindex="-1">
  <div class="reader">
    {{ body|safe }}
  </div>
</main>
<script>
function applyTheme() {
  const stored = localStorage.getItem('yt2md-theme') || 'auto';
  const root = document.documentElement;
  if (stored === 'auto') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', stored);
  const btn = document.querySelector('.theme-toggle');
  if (btn) {
    const icons = {auto: '🌓', light: '☀️', dark: '🌙'};
    btn.textContent = icons[stored];
    btn.title = 'Theme: ' + stored + ' (click to cycle)';
  }
}
function cycleTheme() {
  const cur = localStorage.getItem('yt2md-theme') || 'auto';
  const next = {auto: 'light', light: 'dark', dark: 'auto'}[cur];
  localStorage.setItem('yt2md-theme', next);
  applyTheme();
}
applyTheme();
</script>
</body>
</html>
"""


def _list_digests(digests_dir: Path) -> List[dict]:
    if not digests_dir.exists():
        return []
    results = []
    for d in sorted(digests_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        digest_md = d / "digest.md"
        if not d.is_dir() or not digest_md.exists():
            continue
        title = d.name
        try:
            for line in digest_md.read_text().splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        except Exception:
            pass
        results.append({"id": d.name, "title": title, "mtime": d.stat().st_mtime})
    return results


def _render_markdown(text: str) -> str:
    import markdown as md_lib
    html = md_lib.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    # Rewrite cross-references to other digests (e.g. ../digests/X/digest.md) into view URLs.
    html = re.sub(r'href="[^"]*digests/([^"/]+)/digest\.md"', r'href="/digests/\1/"', html)
    return html


def cmd_serve(args) -> int:
    try:
        from flask import Flask, render_template_string, abort, send_from_directory
    except ImportError:
        sys.exit(
            "Flask is required for the reader. Reinstall with:\n"
            "  uv tool install --reinstall git+https://github.com/jyouturner/youtube-to-markdown"
        )

    data_dir = get_data_dir()
    digests_dir = data_dir / "digests"

    app = Flask(__name__)
    # Disable Flask's default request logging — keep stdout clean.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def page(body: str, *, title: str, current: str, base_href: str = None):
        # Annotate listing items with read state so the sidebar can show "new" markers.
        digests = _list_digests(digests_dir)
        try:
            read_digests = _read_digest_ids()
        except Exception:
            read_digests = set()
        for d in digests:
            d["unread"] = d["id"] not in read_digests
        unread_digest_count = sum(1 for d in digests if d["unread"])

        # Persistent banner above every page when the API key is missing. Reading
        # cached digests still works without it; only generation does, so we
        # warn rather than gate. The Setup page is the one place this is hidden
        # (the page itself IS the configuration UI).
        if current != "setup" and not os.environ.get("ANTHROPIC_API_KEY"):
            banner = (
                '<div class="flash" style="border-left-color: #c00;">'
                '<strong>API key not configured.</strong> '
                'Anthropic API access is required to generate digests and panel '
                'discussions. Reading existing digests still works. '
                '<a href="/setup">Set it up →</a>'
                '</div>'
            )
            body = banner + body

        return render_template_string(
            SERVE_PAGE_TEMPLATE,
            body=body,
            title=title,
            current=current,
            base_href=base_href,
            digests=digests,
            channel_count=len(read_channels()),
            unread_digest_count=unread_digest_count,
        )

    def _require_api_key_or_redirect():
        """Helper for action endpoints: returns a redirect Response if no key,
        else None. Use as: r = _require_api_key_or_redirect(); if r: return r."""
        from flask import redirect
        if os.environ.get("ANTHROPIC_API_KEY"):
            return None
        return redirect("/setup?msg=API+key+required+to+run+this+action.")

    @app.route("/")
    def home():
        from flask import redirect
        digests = _list_digests(digests_dir)
        channels = read_channels()

        # True first-run (no key, no digests, no channels): land directly on
        # the setup page so the user isn't asked to subscribe before they can
        # generate anything. Skip when there are existing digests — those
        # should still be readable even without a key.
        if (not digests and not channels
                and not os.environ.get("ANTHROPIC_API_KEY")):
            return redirect("/setup")

        # Empty states first — show a real CTA, not a list of zero items.
        if not digests:
            if not channels:
                body = (
                    "<h1>Welcome to yt2md</h1>"
                    "<p>You haven't subscribed to any channels yet.</p>"
                    '<p><a class="cta" href="/channels">Add your first channel →</a></p>'
                )
            else:
                body = (
                    "<h1>Polling is set up</h1>"
                    f"<p>You're watching {len(channels)} channel(s). Your first digest "
                    "will appear after the next polling run (every few hours, or "
                    'fire one now from the <a href="/schedule">Schedule</a> page).</p>'
                )
            return page(body, title="yt2md", current="home")

        # Featured content: the latest digest's body, with an eyebrow link.
        featured = digests[0]
        featured_md = (digests_dir / featured["id"] / "digest.md").read_text()
        body = (
            f'<p class="featured-eyebrow">Latest digest · '
            f'<a href="/digests/{featured["id"]}/">{featured["title"]}</a></p>'
        )
        body += _render_markdown(featured_md)
        base_href = f"/digests/{featured['id']}/"

        # No "More digests" footer here — sidebar is the navigation surface;
        # showing the same list twice is just noise.

        return page(body, title="Home", current="home", base_href=base_href)

    @app.route("/channels", methods=["GET"])
    def channels_page():
        from flask import request
        channels = read_channels()
        digests = _list_digests(digests_dir)
        sched_state = _load_schedule_state()
        poll_has_run = bool((sched_state.get("poll") or {}).get("last_started_at"))
        flash = request.args.get("msg", "")

        body = "<h1>Subscriptions</h1>"
        if flash:
            body += f'<div class="flash">{flash}</div>'

        # "What's next" guidance — adapts to current state. Only shown until the
        # user has at least one digest, then disappears.
        next_step = None
        if not channels:
            next_step = (
                "Paste a YouTube channel URL below to get started. "
                "Each new video on this channel will be auto-digested."
            )
        elif not poll_has_run:
            next_step = (
                "You\'re subscribed but the scheduler hasn\'t fired its first poll yet. "
                'It runs every few hours — or trigger one now from the '
                '<a href="/schedule">Schedule page</a>.'
            )
        elif not digests:
            next_step = (
                "Polling fires every 6 hours. Your first digest will land after the next run "
                '— or fire one now from the <a href="/schedule">Schedule page</a>.'
            )
        if next_step:
            body += f'<div class="next-step"><strong>Next step:</strong> {next_step}</div>'

        body += (
            '<form method="post" action="/channels" class="add-form">'
            '<label for="channel-url" class="sr-only">YouTube channel URL</label>'
            '<input id="channel-url" type="text" name="url" '
            'placeholder="https://www.youtube.com/@channel/videos  (or @handle)" '
            'autofocus required>'
            '<button type="submit">Add</button>'
            '</form>'
        )
        if channels:
            body += '<ul class="channel-list">'
            for ch in channels:
                body += (
                    '<li>'
                    f'<span class="url">{ch}</span>'
                    '<form method="post" action="/channels/remove" style="margin:0;">'
                    f'<input type="hidden" name="url" value="{ch}">'
                    '<button type="submit">Remove</button>'
                    '</form>'
                    '</li>'
                )
            body += '</ul>'
        else:
            body += "<p class='empty-state'>No subscriptions yet. Paste a YouTube channel URL above.</p>"
        body += (
            "<p class='meta-info' style='margin-top:32px'>"
            "Stored in <code>~/yt2md/channels.txt</code>."
            "</p>"
        )
        return page(body, title="Subscriptions", current="channels")

    @app.route("/channels", methods=["POST"])
    def channels_add():
        from flask import request, redirect
        url = normalize_channel_url(request.form.get("url", ""))
        if not is_url(url):
            return redirect("/channels?msg=Not+a+valid+URL")
        channels = read_channels()
        if url in channels:
            return redirect(f"/channels?msg=Already+watching+{url}")
        channels.append(url)
        write_channels(channels)
        return redirect(f"/channels?msg=Added+{url}")

    @app.route("/channels/remove", methods=["POST"])
    def channels_remove():
        from flask import request, redirect
        url = request.form.get("url", "").strip()
        channels = [c for c in read_channels() if c != url]
        write_channels(channels)
        return redirect(f"/channels?msg=Removed+{url}")

    @app.route("/schedule")
    def schedule_page():
        from flask import request
        from html import escape as h
        import time as _t
        flash = request.args.get("msg", "")
        cfg = load_schedule_config()
        sched_state = _load_schedule_state()

        body = "<h1>Schedule</h1>"
        if flash:
            body += f'<div class="flash">{h(flash)}</div>'

        body += '<form method="post" action="/schedule/save" class="schedule-form">'
        body += '<div class="schedule-fields">'
        body += '<label>Polling interval'
        body += f'  <input type="number" name="poll_hours" value="{cfg["poll_interval_hours"]}" min="0.1" step="0.1" required>'
        body += '  <span class="suffix">hours</span>'
        body += '</label>'
        body += '</div>'  # schedule-fields

        body += '<button type="submit" class="primary">Save</button>'
        body += '</form>'

        body += (
            f'<p class="meta-info">Current schedule: {h(_format_schedule_summary(cfg))}. '
            'Scheduling runs inside this server — pauses while it\'s down, '
            'catches up on missed slots when you start it again.</p>'
        )

        for kind, friendly, desc in [
            ("poll", "Polling", "fires <code>yt2md watch run</code>"),
        ]:
            sentence, dot_class = _scheduler_status_summary(kind, sched_state)
            s = sched_state.get(kind) or {}
            next_at = _compute_next_poll(cfg, s.get("last_started_at"))

            body += f'<div class="job-block"><h3><span class="dot {dot_class}"></span>{friendly}</h3>'
            body += f'<p class="meta-info" style="margin: 0 0 12px;">{desc}</p>'
            body += f'<p class="job-summary">{sentence}</p>'
            body += (f'<p class="meta-info">Next run: {h(_format_next_run(next_at))} '
                     f'({h(_t.strftime("%Y-%m-%d %H:%M", _t.localtime(next_at)))}).</p>')
            running = kind in _scheduler_jobs
            disabled = " disabled" if running else ""
            running_label = " (already running)" if running else ""
            body += (
                f'<div class="job-actions">'
                f'<form method="post" action="/schedule/run/{kind}" style="margin:0;">'
                f'<button type="submit"{disabled}>Run now{running_label}</button>'
                f'</form></div>'
            )

            body += '<details><summary>Diagnostics</summary>'
            body += '<table class="status-table">'
            for k in ("last_started_at", "last_finished_at", "last_exit_code", "last_pid"):
                v = s.get(k)
                if v is None:
                    continue
                if k.endswith("_at"):
                    v = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(v))
                body += f'<tr><td>{k.replace("_", " ")}</td><td>{h(str(v))}</td></tr>'
            body += '</table></details>'

            log_path = data_dir / "logs" / f"{kind}.log"
            body += '<details style="margin-top: 8px;"><summary>Recent log (last 20 lines)</summary>'
            body += f'<div class="log-block">{h(_tail_log(log_path, 20))}</div>'
            body += '</details>'
            body += '</div>'

        body += '<p class="meta-info">Refresh the page to see updated status after a run.</p>'
        return page(body, title="Schedule", current="schedule")

    @app.route("/schedule/save", methods=["POST"])
    def schedule_save():
        from flask import redirect, request
        cfg = load_schedule_config()
        try:
            if request.form.get("poll_hours"):
                cfg["poll_interval_hours"] = float(request.form["poll_hours"])
        except Exception as e:
            return redirect(f"/schedule?msg=Invalid+input:+{e}")
        save_schedule_config(cfg)
        return redirect("/schedule?msg=Saved+(scheduler+picks+up+within+30s)")

    @app.route("/schedule/run/<job>", methods=["POST"])
    def schedule_run(job):
        from flask import redirect
        gate = _require_api_key_or_redirect()
        if gate is not None:
            return gate
        if job != "poll":
            abort(404)
        with _schedule_lock():
            existing = _scheduler_jobs.get(job)
            if existing is not None and existing.poll() is None:
                return redirect(f"/schedule?msg={job}+is+already+running")
            proc = _fire_scheduled_job(job)
        if proc is None:
            return redirect(f"/schedule?msg=Failed+to+fire+{job}+(yt2md+not+on+PATH)")
        return redirect(
            f"/schedule?msg=Started+{job}+(pid+{proc.pid}).+Refresh+for+status."
        )

    @app.route("/settings", methods=["GET"])
    def settings_page():
        from flask import request
        from html import escape as h
        flash = request.args.get("msg", "")
        s = load_settings()
        # Show the EFFECTIVE value (settings.json with .env fallback) so the
        # form matches what the system is actually using. On Save we persist
        # whatever the user submits, which becomes the new canonical value.
        for key, env_name in (
            ("digest_model", "YT2MD_DIGEST_MODEL"),
            ("panel_model", "YT2MD_PANEL_MODEL"),
            ("whisper_model", "YT2MD_WHISPER_MODEL"),
            ("cookies_from_browser", "YT2MD_COOKIES_FROM_BROWSER"),
            ("digest_language", "YT2MD_DIGEST_LANGUAGE"),
        ):
            if not s.get(key) and os.environ.get(env_name):
                s[key] = os.environ[env_name]

        whisper_choices = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
        cookie_choices = ("", "chrome", "firefox", "safari", "brave", "edge",
                          "chromium", "opera", "vivaldi")

        body = "<h1>Settings</h1>"
        if flash:
            body += f'<div class="flash">{h(flash)}</div>'
        body += (
            '<p class="meta-info">Stored in <code>~/yt2md/settings.json</code> '
            '(API key in <code>~/yt2md/.env</code>). '
            'New values take effect on the next one-off submit / scheduled poll / '
            '"Discuss with experts" click — no restart required.</p>'
        )

        body += '<form method="post" action="/settings/save" class="schedule-form">'
        body += '<div class="schedule-fields" style="grid-template-columns: 1fr;">'

        cur_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if cur_key:
            cur_state = (
                f'<span style="color: var(--accent);">set</span> '
                f'(<code>{h(cur_key[:7])}…{h(cur_key[-4:])}</code>)'
            )
        else:
            cur_state = '<span style="color: #c00;">not set</span>'
        body += (
            '<label>Anthropic API key'
            '  <input type="password" name="anthropic_api_key" '
            '    placeholder="sk-ant-... (leave blank to keep current)" '
            '    autocomplete="off">'
            '  <span class="suffix" style="display:block;">'
            f'    Current: {cur_state}. '
            f'    {API_KEY_COST_NOTE} '
            '    Get a key at '
            '    <a href="https://console.anthropic.com/settings/keys" target="_blank" '
            '       rel="noopener">console.anthropic.com/settings/keys</a>. '
            '    The key is validated on save with a 1-token test call.'
            '  </span>'
            '</label>'
        )

        body += (
            '<label>Digest model'
            f'  <input type="text" name="digest_model" value="{h(s["digest_model"])}" required>'
            '  <span class="suffix" style="display:block;">'
            'Anthropic model ID for the per-video digest. e.g. <code>claude-sonnet-4-6</code>, '
            '<code>claude-opus-4-7</code>, <code>claude-haiku-4-5-20251001</code>.'
            '</span>'
            '</label>'
        )

        body += (
            '<label>Panel-discussion model'
            f'  <input type="text" name="panel_model" value="{h(s["panel_model"])}" required>'
            '  <span class="suffix" style="display:block;">'
            'Used when you click "Discuss with experts" on a digest. Multi-perspective '
            'synthesis benefits from a stronger reasoning model — Opus is the default.'
            '</span>'
            '</label>'
        )

        body += '<label>Whisper model'
        body += '  <select name="whisper_model">'
        for w in whisper_choices:
            sel = ' selected' if s["whisper_model"] == w else ''
            body += f'    <option value="{w}"{sel}>{w}</option>'
        body += '  </select>'
        body += (
            '  <span class="suffix" style="display:block;">'
            'Local STT fallback when YouTube has no captions. Larger = better quality, '
            'slower, bigger first-run download. <code>medium</code> is a good default.'
            '</span>'
            '</label>'
        )

        body += '<label>Digest language'
        body += '  <select name="digest_language">'
        for code, label in (
            ("auto", "auto — match the transcript's language"),
            ("en", "en — always English"),
        ):
            sel = ' selected' if s.get("digest_language", "auto") == code else ''
            body += f'    <option value="{code}"{sel}>{label}</option>'
        body += '  </select>'
        body += (
            '  <span class="suffix" style="display:block;">'
            'Applies to both the per-video digest and the panel discussion. '
            '<code>auto</code> writes in the source language (e.g. Chinese for a Chinese-language video). '
            '<code>en</code> forces English regardless.'
            '</span>'
            '</label>'
        )

        body += '<label>Cookies from browser'
        body += '  <select name="cookies_from_browser">'
        for c in cookie_choices:
            sel = ' selected' if s.get("cookies_from_browser", "") == c else ''
            label = "(none)" if c == "" else c
            body += f'    <option value="{c}"{sel}>{label}</option>'
        body += '  </select>'
        body += (
            '  <span class="suffix" style="display:block;">'
            'YouTube increasingly requires logged-in cookies. Pick the browser you\'re '
            'signed into YouTube on; yt-dlp extracts cookies on each run. Leave as '
            '"(none)" if you only digest publicly-accessible videos.'
            '</span>'
            '</label>'
        )

        body += '</div>'  # schedule-fields
        body += '<button type="submit" class="primary">Save</button>'
        body += '</form>'
        return page(body, title="Settings", current="settings")

    @app.route("/settings/save", methods=["POST"])
    def settings_save():
        from flask import redirect, request
        from urllib.parse import quote_plus
        s = load_settings()
        for key in ("digest_model", "panel_model", "whisper_model",
                    "cookies_from_browser", "digest_language"):
            v = request.form.get(key)
            if v is not None:
                s[key] = v.strip()
        save_settings(s)

        # API key is stored separately (.env, not settings.json) so non-secret
        # config can be checked into a shared settings file without leaking it.
        new_key = (request.form.get("anthropic_api_key") or "").strip()
        if new_key:
            err = validate_api_key(new_key)
            if err:
                return redirect(
                    f"/settings?msg=Settings+saved+but+API+key+rejected:+{quote_plus(err)}"
                )
            set_env_var("ANTHROPIC_API_KEY", new_key)
            return redirect("/settings?msg=Saved+(API+key+validated).")
        return redirect("/settings?msg=Saved.")

    @app.route("/setup", methods=["GET"])
    def setup_page():
        from flask import request
        from html import escape as h
        flash = request.args.get("msg", "")

        cur_key = os.environ.get("ANTHROPIC_API_KEY", "")
        already_set = bool(cur_key)

        body = "<h1>Set up your Anthropic API key</h1>"
        if flash:
            body += f'<div class="flash">{h(flash)}</div>'

        if already_set:
            body += (
                '<div class="flash" style="border-left-color: var(--accent);">'
                f'<strong>A key is already configured</strong> '
                f'(<code>{h(cur_key[:7])}…{h(cur_key[-4:])}</code>). '
                'Submit a new one below to replace it, or '
                '<a href="/">go to your library</a>.'
                '</div>'
            )
        else:
            body += (
                '<p class="meta-info">yt2md uses your own Anthropic API key to '
                'generate digests and panel discussions. The key is stored '
                'locally in <code>~/yt2md/.env</code> and never leaves your '
                'machine except when calling Anthropic.</p>'
            )

        body += (
            '<div class="next-step">'
            '<strong>Steps</strong>'
            '<ol style="margin: 8px 0 0 20px; padding: 0;">'
            '<li>Sign in to <a href="https://console.anthropic.com/settings/keys" '
            'target="_blank" rel="noopener">console.anthropic.com/settings/keys</a> '
            '(create an account if you don\'t have one).</li>'
            '<li>Add a payment method at '
            '<a href="https://console.anthropic.com/settings/billing" '
            'target="_blank" rel="noopener">Billing</a> — note: your Claude.ai '
            'subscription does not cover API usage; they\'re billed separately.</li>'
            '<li>Click <strong>Create Key</strong>, copy the value (starts with '
            '<code>sk-ant-</code>), and paste it below.</li>'
            '</ol>'
            f'<p style="margin-top: 12px;">{API_KEY_COST_NOTE}</p>'
            '</div>'
        )

        body += '<form method="post" action="/setup/save" class="schedule-form">'
        body += '<div class="schedule-fields" style="grid-template-columns: 1fr;">'
        body += (
            '<label>Anthropic API key'
            '  <input type="password" name="anthropic_api_key" '
            '    placeholder="sk-ant-..." autocomplete="off" autofocus required>'
            '  <span class="suffix" style="display:block;">'
            '    Validated with a 1-token test call before being saved.'
            '  </span>'
            '</label>'
        )
        body += '</div>'
        body += '<button type="submit" class="primary">Save and validate</button>'
        body += '</form>'

        return page(body, title="Set up", current="setup")

    @app.route("/setup/save", methods=["POST"])
    def setup_save():
        from flask import redirect, request
        from urllib.parse import quote_plus
        new_key = (request.form.get("anthropic_api_key") or "").strip()
        if not new_key:
            return redirect("/setup?msg=Paste+a+key+first.")
        err = validate_api_key(new_key)
        if err:
            return redirect(f"/setup?msg=Key+rejected:+{quote_plus(err)}")
        set_env_var("ANTHROPIC_API_KEY", new_key)
        return redirect("/?msg=API+key+saved.+You+can+now+generate+digests.")

    @app.route("/one-off", methods=["GET"])
    def one_off_page():
        from flask import request
        from html import escape as h
        flash = request.args.get("msg", "")
        active = _list_active_oneoff_jobs()  # also reaps exited subprocesses
        failures = _list_recent_oneoff_failures()

        body = "<h1>One-off digest</h1>"
        if flash:
            body += f'<div class="flash">{h(flash)}</div>'
        body += (
            '<p class="meta-info">Paste a YouTube video URL. The digest runs in the '
            'background and lands in your library when complete (1–25 min depending on '
            'video length and the vision pass). You can close this tab — it keeps running.</p>'
        )
        body += (
            '<form method="post" action="/one-off" class="add-form">'
            '<label for="video-url" class="sr-only">YouTube video URL</label>'
            '<input id="video-url" type="text" name="url" '
            'placeholder="https://youtu.be/... (or full watch URL)" '
            'autofocus required>'
            '<button type="submit">Digest</button>'
            '</form>'
        )

        import time as _t
        # Read the log once so per-job stage lookups don't hit disk repeatedly.
        log_path = data_dir / "logs" / "oneoff.log"
        try:
            log_text = log_path.read_text(errors="replace")
        except OSError:
            log_text = ""

        # Always render the section containers so the polling JS can target them
        # (display:none hides them when empty).
        active_hidden = "" if active else " style='display:none'"
        body += f"<section id='oneoff-active-section'{active_hidden}>"
        body += "<h2>In progress</h2>"
        body += "<ul id='oneoff-active-list' class='channel-list'>"
        for j in active:
            elapsed = int(_t.time() - j["started"])
            stage = _describe_job_stage(log_text, j["video_id"])
            body += (
                '<li>'
                f'<span class="url"><strong>{h(j["video_id"])}</strong> · '
                f'{h(j["url"])}</span>'
                f'<span style="color: var(--muted); font-size: 12px;">'
                f'{elapsed//60}m {elapsed%60}s · {h(stage)}</span>'
                '</li>'
            )
        body += "</ul></section>"

        failures_hidden = "" if failures else " style='display:none'"
        body += f"<section id='oneoff-failures-section'{failures_hidden}>"
        body += "<h2>Recent failures</h2>"
        body += "<ul id='oneoff-failures-list' class='channel-list'>"
        for f in failures:
            ago = int(_t.time() - f["ended"])
            if ago < 60:
                when = f"{ago}s ago"
            elif ago < 3600:
                when = f"{ago // 60}m ago"
            else:
                when = f"{ago // 3600}h ago"
            err = f["error"] or f"exit code {f['exit_code']}"
            body += (
                '<li>'
                f'<span class="url"><strong>{h(f["video_id"])}</strong> · '
                f'{h(f["url"])}</span>'
                f'<span style="color: var(--muted); font-size: 12px;">{when}</span>'
                f'<div style="color: var(--muted); font-size: 13px; margin-top: 4px;">{h(err)}</div>'
                '</li>'
            )
        body += "</ul>"
        body += (
            "<p class='meta-info'>"
            f"Full output: <code>{h(str(log_path))}</code>"
            "</p>"
        )
        body += "</section>"

        body += (
            "<p class='meta-info' style='margin-top: 32px;'>"
            "One-off digests share the same library as subscription mode. "
            "They appear in the sidebar's <strong>Digests</strong> section once ready."
            "</p>"
        )

        # Polling: refresh in-progress + failures every 2s without reloading the page.
        body += """
<script>
(function () {
  const activeSection  = document.getElementById('oneoff-active-section');
  const activeList     = document.getElementById('oneoff-active-list');
  const failSection    = document.getElementById('oneoff-failures-section');
  const failList       = document.getElementById('oneoff-failures-list');
  if (!activeSection || !failSection) return;

  const fmtElapsed = s => `${Math.floor(s/60)}m ${s%60}s`;
  const fmtAgo = s => s < 60 ? `${s}s ago` : (s < 3600 ? `${Math.floor(s/60)}m ago` : `${Math.floor(s/3600)}h ago`);
  const esc = s => { const d = document.createElement('div'); d.textContent = String(s ?? ''); return d.innerHTML; };

  function render(data) {
    const active = data.active || [];
    activeSection.style.display = active.length ? '' : 'none';
    activeList.innerHTML = active.map(j => `
      <li>
        <span class="url"><strong>${esc(j.video_id)}</strong> &middot; ${esc(j.url)}</span>
        <span style="color: var(--muted); font-size: 12px;">${esc(fmtElapsed(j.elapsed_secs))} &middot; ${esc(j.stage)}</span>
      </li>
    `).join('');

    const failures = data.failures || [];
    failSection.style.display = failures.length ? '' : 'none';
    failList.innerHTML = failures.map(f => `
      <li>
        <span class="url"><strong>${esc(f.video_id)}</strong> &middot; ${esc(f.url)}</span>
        <span style="color: var(--muted); font-size: 12px;">${esc(fmtAgo(f.ago_secs))}</span>
        <div style="color: var(--muted); font-size: 13px; margin-top: 4px;">${esc(f.error || ('exit code ' + f.exit_code))}</div>
      </li>
    `).join('');
  }

  async function poll() {
    try {
      const res = await fetch('/one-off/status', { cache: 'no-store' });
      if (res.ok) render(await res.json());
    } catch (_) { /* transient — try again next tick */ }
  }

  poll();
  setInterval(poll, 2000);
})();
</script>
"""
        return page(body, title="One-off digest", current="one-off")

    @app.route("/one-off/status")
    def one_off_status():
        from flask import jsonify
        import time as _t
        active = _list_active_oneoff_jobs()  # also reaps any just-exited subprocesses
        failures = _list_recent_oneoff_failures()
        log_path = data_dir / "logs" / "oneoff.log"
        try:
            log_text = log_path.read_text(errors="replace")
        except OSError:
            log_text = ""
        now = _t.time()
        return jsonify({
            "active": [
                {
                    "video_id": j["video_id"],
                    "url": j["url"],
                    "started": j["started"],
                    "elapsed_secs": int(now - j["started"]),
                    "stage": _describe_job_stage(log_text, j["video_id"]),
                }
                for j in active
            ],
            "failures": [
                {
                    "video_id": f["video_id"],
                    "url": f["url"],
                    "started": f["started"],
                    "ended": f["ended"],
                    "ago_secs": int(now - f["ended"]),
                    "exit_code": f["exit_code"],
                    "error": f["error"],
                }
                for f in failures
            ],
        })

    @app.route("/one-off", methods=["POST"])
    def one_off_submit():
        from flask import redirect, request
        import time as _t
        gate = _require_api_key_or_redirect()
        if gate is not None:
            return gate
        url = request.form.get("url", "").strip()
        if not url:
            return redirect("/one-off?msg=URL+is+required")

        video_id = extract_video_id(url)
        if not video_id:
            return redirect(
                f"/one-off?msg=Couldn%27t+extract+a+YouTube+video+ID+from:+{url}"
            )

        # Already in library? Send them straight to the existing digest.
        existing = digests_dir / video_id / "digest.md"
        if existing.exists():
            return redirect(f"/digests/{video_id}/")

        # Already in progress? Show the page with a message rather than re-firing.
        for active in _list_active_oneoff_jobs():
            if active["video_id"] == video_id:
                return redirect(f"/one-off?msg=Already+digesting+{video_id}")

        # Fire and forget. start_new_session=True detaches the child so it survives
        # if the web server is killed. stdout/stderr go to oneoff.log.
        digest_path = digests_dir / video_id / "digest.md"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = data_dir / "logs" / "oneoff.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_path, "a")
        log_fd.write(f"\n===== {_t.strftime('%Y-%m-%d %H:%M:%S')} starting {video_id} ({url}) =====\n")
        log_fd.flush()

        yt2md_path = shutil.which("yt2md")
        if not yt2md_path:
            log_fd.close()
            return redirect("/one-off?msg=yt2md+not+on+PATH")

        try:
            proc = subprocess.Popen(
                [yt2md_path, url, "-o", str(digest_path)],
                cwd=digest_path.parent,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env={**os.environ, **_settings_to_env(load_settings())},
                start_new_session=True,
            )
        finally:
            # Subprocess holds its own copy of the fd; safe for us to close.
            log_fd.close()

        _oneoff_jobs[proc.pid] = {
            "video_id": video_id,
            "started": _t.time(),
            "url": url,
            "proc": proc,
        }
        return redirect(
            f"/one-off?msg=Started+digesting+{video_id}+(check+sidebar+in+a+few+minutes)"
        )

    @app.route("/activity")
    def activity_page():
        from flask import request
        from html import escape as h
        import time as _t
        flt = request.args.get("status", "all")  # all | success | failed
        rows = _recent_runs(limit=200)
        if flt == "success":
            rows = [r for r in rows if r.get("success")]
        elif flt == "failed":
            rows = [r for r in rows if not r.get("success")]

        body = "<h1>Activity</h1>"
        body += '<p class="meta-info">Every completed one-off digest, success or failure. Persists across server restarts.</p>'

        # In-progress section — live one-off jobs the reaper hasn't logged yet.
        # Hidden by default; populated by the polling JS at the bottom of the page.
        active_now = _list_active_oneoff_jobs()
        log_path = data_dir / "logs" / "oneoff.log"
        try:
            log_text = log_path.read_text(errors="replace")
        except OSError:
            log_text = ""
        active_hidden = "" if active_now else " style='display:none'"
        body += f"<section id='activity-active-section'{active_hidden}>"
        body += "<h2>In progress</h2>"
        body += "<ul id='activity-active-list' class='channel-list'>"
        for j in active_now:
            elapsed = int(_t.time() - j["started"])
            stage = _describe_job_stage(log_text, j["video_id"])
            body += (
                '<li>'
                f'<span class="url"><strong>{h(j["video_id"])}</strong> · '
                f'{h(j["url"])}</span>'
                f'<span style="color: var(--muted); font-size: 12px;">'
                f'{elapsed//60}m {elapsed%60}s · {h(stage)}</span>'
                '</li>'
            )
        body += "</ul></section>"

        # Polling JS for the in-progress section — emitted here so both the
        # "no runs yet" early return and the full table path include it.
        active_poll_js = """
<script>
(function () {
  const section = document.getElementById('activity-active-section');
  const list = document.getElementById('activity-active-list');
  if (!section || !list) return;
  let prevCount = list.children.length;
  const fmtElapsed = s => `${Math.floor(s/60)}m ${s%60}s`;
  const esc = s => { const d = document.createElement('div'); d.textContent = String(s ?? ''); return d.innerHTML; };
  async function poll() {
    try {
      const res = await fetch('/one-off/status', { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      const active = data.active || [];
      section.style.display = active.length ? '' : 'none';
      list.innerHTML = active.map(j => `
        <li>
          <span class="url"><strong>${esc(j.video_id)}</strong> &middot; ${esc(j.url)}</span>
          <span style="color: var(--muted); font-size: 12px;">${esc(fmtElapsed(j.elapsed_secs))} &middot; ${esc(j.stage)}</span>
        </li>
      `).join('');
      if (active.length < prevCount) {
        window.location.reload();
        return;
      }
      prevCount = active.length;
    } catch (_) { /* try again */ }
  }
  poll();
  setInterval(poll, 2000);
})();
</script>
"""

        # Filter chips
        chip = lambda v, label, count: (
            f'<a href="/activity?status={v}" class="filter-chip'
            + (' active' if flt == v else '')
            + f'">{h(label)} <span class="filter-chip-count">({count})</span></a>'
        )
        all_runs = _recent_runs(limit=200)
        n_all = len(all_runs)
        n_ok = sum(1 for r in all_runs if r.get("success"))
        n_fail = n_all - n_ok
        body += "<div class='filter-row'>"
        body += chip("all", "All", n_all)
        body += chip("success", "Success", n_ok)
        body += chip("failed", "Failed", n_fail)
        body += "</div>"

        if not rows:
            body += "<p class='meta-info'>No runs recorded yet. Submit a one-off digest from the <a href='/one-off'>One-off digest</a> page.</p>"
            body += active_poll_js
            return page(body, title="Activity", current="activity")

        def _fmt_ago(secs: float) -> str:
            secs = int(secs)
            if secs < 60: return f"{secs}s ago"
            if secs < 3600: return f"{secs//60}m ago"
            if secs < 86400: return f"{secs//3600}h ago"
            return f"{secs//86400}d ago"

        def _fmt_dur(secs) -> str:
            if secs is None: return "—"
            secs = float(secs)
            if secs < 60: return f"{secs:.1f}s"
            return f"{int(secs)//60}m {int(secs)%60}s"

        def _fmt_int(n) -> str:
            if n is None: return "—"
            n = int(n)
            return f"{n:,}"

        now = _t.time()
        body += "<table class='activity-table'>"
        body += (
            "<thead><tr>"
            "<th>When</th><th>Video</th><th>Outcome</th>"
            "<th>Duration</th><th>Stages</th><th>Tokens</th>"
            "</tr></thead><tbody>"
        )
        for r in rows:
            video_id = r.get("video_id") or ""
            url = r.get("url") or ""
            ago = _fmt_ago(now - (r.get("started_at") or now))
            dur = _fmt_dur(r.get("duration_secs"))
            success = r.get("success")
            if success:
                outcome = "<span class='ok'>✓ done</span>"
                if r.get("digest_path"):
                    title = f'<a href="/digests/{h(video_id)}/">{h(video_id)}</a>'
                else:
                    title = h(video_id)
            else:
                stage = r.get("stage_reached") or "?"
                outcome = f"<span class='fail'>✗ failed at {h(stage)}</span>"
                title = h(video_id)
            # Stage breakdown
            parts = []
            for label, key in [
                ("dl", "download_secs"),
                ("whisper", "whisper_secs"),
                ("frames", "frames_secs"),
                ("digest", "digest_secs"),
                ("vision", "vision_secs"),
            ]:
                v = r.get(key)
                if v is not None and v > 0.05:
                    parts.append(f"{label} {v:.1f}s")
            stages_cell = h(", ".join(parts)) if parts else "—"
            # Tokens (digest only)
            tin = r.get("digest_input_tokens")
            tout = r.get("digest_output_tokens")
            cache = r.get("digest_cache_read_tokens") or 0
            if tin or tout:
                tokens_cell = f"in {_fmt_int(tin)} · out {_fmt_int(tout)}"
                if cache:
                    tokens_cell += f" · cache {_fmt_int(cache)}"
                tokens_cell = h(tokens_cell)
            else:
                tokens_cell = "—"

            body += "<tr>"
            body += f"<td title='{h(url)}'>{h(ago)}</td>"
            body += f"<td>{title}"
            extras = []
            if r.get("source_lang"): extras.append(f"lang: {h(r['source_lang'])}")
            if r.get("used_whisper"):
                wm = r.get("whisper_model") or "?"
                extras.append(f"whisper: {h(wm)}")
            if extras:
                body += f"<div class='activity-meta'>{' · '.join(extras)}</div>"
            body += "</td>"
            body += f"<td>{outcome}"
            err = r.get("error")
            if err and not success:
                body += f"<div class='activity-meta activity-error'>{h(err)}</div>"
            body += "</td>"
            body += f"<td>{h(dur)}</td>"
            body += f"<td class='activity-stages'>{stages_cell}</td>"
            body += f"<td class='activity-tokens'>{tokens_cell}</td>"
            body += "</tr>"
        body += "</tbody></table>"
        body += (
            "<p class='meta-info' style='margin-top: 24px;'>"
            f"Raw log: <code>{h(str(_oneoff_log_path()))}</code><br>"
            f"JSONL: <code>{h(str(_runs_jsonl_path()))}</code>"
            "</p>"
        )
        body += active_poll_js
        return page(body, title="Activity", current="activity")

    @app.route("/digests/<video_id>/")
    def view_digest(video_id):
        from html import escape as h
        digest_md = digests_dir / video_id / "digest.md"
        if not digest_md.exists():
            abort(404)
        try:
            _mark_digest_read(video_id)
        except Exception:
            pass  # never block reading on a DB error
        rendered = _render_markdown(digest_md.read_text())

        # Build the action toolbar. Lives at the TOP of the page so the user
        # doesn't have to scroll past the whole digest to find these. Discuss
        # POST is synchronous (~60–120s for the Opus call); the JS handler
        # disables the button on submit and swaps its label so the wait is
        # visible. The form still posts normally; the browser navigates to the
        # panel page on redirect.
        panel_md = digests_dir / video_id / "panel.md"
        panel_exists = panel_md.exists()
        discuss_submit_js = (
            "this.querySelector('button').disabled=true;"
            "this.querySelector('button').textContent='Generating panel… (~60–120s)';"
        )
        toolbar = "<div class='digest-actions digest-toolbar'>"
        if panel_exists:
            toolbar += (
                f"<a class='discuss-btn' href='/digests/{h(video_id)}/panel/'>"
                "View panel discussion</a>"
                f"<form method='post' action='/digests/{h(video_id)}/discuss' "
                f"style='display:inline;' onsubmit=\"{discuss_submit_js}\">"
                "<button type='submit' class='discuss-btn-secondary' "
                "title='Re-runs the panel; replaces existing panel.md'>Regenerate</button>"
                "</form>"
            )
        else:
            toolbar += (
                f"<form method='post' action='/digests/{h(video_id)}/discuss' "
                f"style='display:inline;' onsubmit=\"{discuss_submit_js}\">"
                "<button type='submit' class='discuss-btn' "
                "title='Generates a panel-of-experts discussion (~60–120s, one Opus 4.7 call)'>"
                "Discuss with experts</button>"
                "</form>"
            )
        toolbar += (
            f"<form method='post' action='/digests/{h(video_id)}/delete' style='display:inline;' "
            "onsubmit=\"return confirm('Delete this digest? "
            "The rendered output, frames, and cached video will be wiped.');\">"
            "<button type='submit' class='delete-btn' "
            "title='Wipes digest.md, frames, and cached video. Re-submit via One-off digest to regenerate.'>"
            "Delete digest</button>"
            "</form>"
        )
        toolbar += "</div>"

        body = (
            toolbar
            + "<hr style='margin: 16px 0 32px; border: none; border-top: 1px solid var(--border);'>"
            + rendered
        )
        return page(body, title=video_id, current=f"digest:{video_id}",
                    base_href=f"/digests/{video_id}/")

    @app.route("/digests/<video_id>/discuss", methods=["POST"])
    def generate_panel(video_id):
        from flask import redirect
        gate = _require_api_key_or_redirect()
        if gate is not None:
            return gate
        digest_md = digests_dir / video_id / "digest.md"
        if not digest_md.exists():
            abort(404)
        # Find the cached SRT to feed into the panel prompt. The download dir
        # holds whichever language track was picked (en / zh-Hans / etc.).
        srt_dir = digests_dir / video_id / "downloads" / video_id
        srt_files = list(srt_dir.glob("*.srt")) if srt_dir.exists() else []
        if not srt_files:
            return redirect(
                f"/digests/{video_id}/?msg=No+cached+transcript.+Re-digest+the+video+first."
            )
        # Filename pattern: <video_id>.<lang>.srt — pull the lang back out so
        # the panel can be written in the same language as the digest.
        srt_path = srt_files[0]
        lang = srt_path.stem[len(video_id) + 1:] if srt_path.stem.startswith(video_id + ".") else "en"
        s = load_settings()
        try:
            segments = parse_srt(srt_path)
            text, _usage = generate_panel_discussion(
                digest_md.read_text(),
                segments,
                model=s.get("panel_model") or DEFAULT_PANEL_MODEL,
                source_lang=lang,
                output_language=s.get("digest_language") or "auto",
            )
        except Exception as e:
            return redirect(f"/digests/{video_id}/?msg=Panel+generation+failed:+{e}")
        (digests_dir / video_id / "panel.md").write_text(text)
        return redirect(f"/digests/{video_id}/panel/")

    @app.route("/digests/<video_id>/panel/")
    def view_panel(video_id):
        panel_md = digests_dir / video_id / "panel.md"
        if not panel_md.exists():
            abort(404)
        rendered = _render_markdown(panel_md.read_text())
        nav = (
            f'<p class="meta-info" style="margin-top:0;">'
            f'<a href="/digests/{video_id}/">← Back to digest</a></p>'
        )
        return page(nav + rendered, title=f"Panel · {video_id}",
                    current=f"digest:{video_id}",
                    base_href=f"/digests/{video_id}/panel/")

    @app.route("/digests/<video_id>/delete", methods=["POST"])
    def delete_digest(video_id):
        from flask import redirect
        target = digests_dir / video_id
        if not target.exists():
            abort(404)
        # Wipe artifacts + cached source media. shutil handles missing children.
        shutil.rmtree(target, ignore_errors=True)
        # Clear read-state row.
        try:
            with _library_connect() as conn:
                conn.execute("DELETE FROM digest_reads WHERE digest_id = ?", (video_id,))
        except Exception:
            pass
        return redirect("/?msg=Deleted+" + video_id)

    @app.route("/digests/<video_id>/digest_images/<path:filename>")
    def digest_image(video_id, filename):
        return send_from_directory(digests_dir / video_id / "digest_images", filename)

    @app.errorhandler(404)
    def not_found(e):
        return page(
            "<h1>Not found</h1><p>That digest doesn't exist yet.</p>",
            title="404", current="home",
        ), 404

    url = f"http://127.0.0.1:{args.port}/"
    print(f"yt2md reader: {url}")
    print(f"Data dir: {data_dir}")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print(f"API key: set ({api_key[:7]}…{api_key[-4:]})")
    else:
        print(f"API key: (not set — first-run setup at {url}setup)")
    # Surface the YouTube prerequisites at startup so the user notices missing
    # cookies / JS runtime before a one-off digest fails 30s later.
    cookies_browser = os.environ.get("YT2MD_COOKIES_FROM_BROWSER")
    print(f"Cookies: {cookies_browser or '(none — set YT2MD_COOKIES_FROM_BROWSER for paywalled YouTube)'}")
    js_rt = _ensure_js_runtime_available()
    print(f"JS runtime: {js_rt or '(not found — yt-dlp n-challenge will fail; install deno or node)'}")
    _cleanup_legacy_launchd()
    start_scheduler()
    print("Press Ctrl-C to stop.\n")

    if not args.no_browser:
        import webbrowser
        import threading
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


def cmd_doctor(args) -> int:
    """Diagnose prerequisites and config. Prints check/X per item with fix hints.

    Exits 0 if everything looks usable, 1 if any blocking issue was found.
    Designed to be the first thing a new user runs after install — gives a
    concrete punch list instead of failing mid-pipeline 30s into the first
    digest.
    """
    OK, WARN, FAIL = "\033[32m✓\033[0m", "\033[33m!\033[0m", "\033[31m✗\033[0m"
    blocking: list = []
    advisory: list = []

    def ok(msg: str) -> None:
        print(f"  {OK} {msg}")

    def warn(msg: str, hint: Optional[str] = None) -> None:
        print(f"  {WARN} {msg}")
        if hint:
            print(f"      → {hint}")
        advisory.append(msg)

    def fail(msg: str, hint: Optional[str] = None) -> None:
        print(f"  {FAIL} {msg}")
        if hint:
            print(f"      → {hint}")
        blocking.append(msg)

    print("\nyt2md doctor — checking prerequisites and config\n")

    print("System tools:")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            ok(f"{tool} on PATH")
        else:
            fail(f"{tool} not found",
                 "macOS: `brew install ffmpeg` · Linux: `apt install ffmpeg`")

    if shutil.which("uv"):
        try:
            v = subprocess.run(["uv", "--version"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
            ok(f"uv ({v})")
        except Exception:
            ok("uv on PATH")
    else:
        warn("uv not on PATH",
             "Recommended: install via `curl -LsSf https://astral.sh/uv/install.sh | sh`")

    rt = _ensure_js_runtime_available()
    if rt:
        rt_path = shutil.which(rt) or "(unknown)"
        try:
            ver = subprocess.run([rt, "--version"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except Exception:
            ver = "?"
        ok(f"JS runtime: {rt} {ver} ({rt_path})")
    else:
        fail("No JS runtime found (needed for yt-dlp's n-challenge solver)",
             "`brew install deno` (simplest) or `nvm install 20`")

    print("\nPython packages:")
    try:
        import yt_dlp  # type: ignore
        ok(f"yt-dlp {yt_dlp.version.__version__}")
    except ImportError:
        fail("yt-dlp not installed", "Run `uv sync` from the project directory")
    try:
        import faster_whisper  # type: ignore  # noqa: F401
        ok("faster-whisper installed")
    except ImportError:
        warn("faster-whisper not installed",
             "Whisper fallback won't work for captionless videos. `uv sync` to install.")
    try:
        import anthropic  # type: ignore  # noqa: F401
        ok("anthropic SDK installed")
    except ImportError:
        fail("anthropic SDK not installed", "Run `uv sync`")

    print("\nAPI / auth:")
    load_env_files()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        ok(f"ANTHROPIC_API_KEY set ({key[:10]}…{key[-4:]})")
    else:
        fail("ANTHROPIC_API_KEY not set",
             f"Get a key at https://console.anthropic.com/settings/keys; first run "
             f"of yt2md will prompt and save it to {get_data_dir() / '.env'}")

    settings = load_settings()
    cookies = settings.get("cookies_from_browser") or os.environ.get("YT2MD_COOKIES_FROM_BROWSER", "")
    if cookies:
        ok(f"YouTube cookies: from browser '{cookies}'")
    else:
        warn("YouTube cookies not configured",
             "Many videos now require login. Set in /settings or as "
             "YT2MD_COOKIES_FROM_BROWSER=firefox in ~/yt2md/.env")

    print("\nConfig:")
    data_dir = get_data_dir()
    print(f"  data dir: {data_dir}")
    if _settings_file().exists():
        ok(f"settings.json exists")
    else:
        warn("settings.json not yet created — defaults in effect",
             f"Open http://localhost:7682/settings to configure (after `yt2md serve`)")
    print(f"  digest model: {settings.get('digest_model')}")
    print(f"  panel model:  {settings.get('panel_model')}")
    print(f"  whisper model: {settings.get('whisper_model')}")
    print(f"  digest language: {settings.get('digest_language')}")

    cfg = load_schedule_config()
    print(f"  schedule: {_format_schedule_summary(cfg)}")

    digests_dir = data_dir / "digests"
    n_digests = len([d for d in digests_dir.iterdir() if (d / "digest.md").exists()]) if digests_dir.exists() else 0
    n_channels = len(read_channels())
    print(f"  library: {n_digests} digest(s), {n_channels} channel(s)")

    print()
    if blocking:
        print(f"{FAIL} {len(blocking)} blocking issue{'s' if len(blocking) != 1 else ''}; "
              "fix the items marked above.")
        if advisory:
            print(f"{WARN} {len(advisory)} advisory item{'s' if len(advisory) != 1 else ''} "
                  "(non-blocking but worth setting up).")
        return 1
    if advisory:
        n = len(advisory)
        print(f"{OK} Core requirements met. {n} advisory item"
              f"{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} optional.")
    else:
        print(f"{OK} All checks passed. Run `yt2md serve` to start the reader.")
    return 0


# ---- subcommand dispatcher ----

def _subcommand_main(argv: List[str]) -> int:
    """Handle yt2md {watch,serve} ..."""
    ap = argparse.ArgumentParser(prog="yt2md", description="yt2md subcommands")
    sub = ap.add_subparsers(dest="cmd", required=True)

    watch = sub.add_parser("watch", help="Manage watched channels and run polling")
    watch_sub = watch.add_subparsers(dest="watch_cmd", required=True)
    p = watch_sub.add_parser("add", help="Add a channel URL"); p.add_argument("url")
    p.set_defaults(func=cmd_watch_add)
    p = watch_sub.add_parser("list", help="List watched channels"); p.set_defaults(func=cmd_watch_list)
    p = watch_sub.add_parser("remove", help="Remove a channel URL"); p.add_argument("url")
    p.set_defaults(func=cmd_watch_remove)
    p = watch_sub.add_parser("run", help="Poll all channels and digest new videos")
    p.set_defaults(func=cmd_watch_run)

    serve = sub.add_parser("serve", help="Start a local web reader (also runs the in-process scheduler)")
    serve.add_argument("--port", type=int, default=7682, help="Port (default: 7682)")
    serve.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open a browser tab on start")
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="Check prerequisites and config; print a punch list")
    doctor.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    return args.func(args)


# ---------- Main ----------

def main():
    # Subcommand dispatch — short-circuit the single-video flow when the user
    # invokes yt2md watch / serve / doctor.
    if len(sys.argv) > 1 and sys.argv[1] in ("watch", "serve", "doctor"):
        load_env_files()
        sys.exit(_subcommand_main(sys.argv[1:]))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Input MP4 file path OR a YouTube URL (auto-downloads mp4 + SRT)")
    ap.add_argument("srt", nargs="?", default=None,
                    help="SRT transcript file (required if 'video' is a local path; "
                         "ignored for URLs — fetched automatically)")
    ap.add_argument("-o", "--output", type=Path, default=None, metavar="PATH",
                    help="Digest output path (default: <video-name>_digest.md)")
    ap.add_argument("--deck", nargs="?", const="__default__", default=None, metavar="PATH",
                    help="Also write a PowerPoint deck. With no path, defaults to "
                         "<video-name>_deck.pptx.")
    ap.add_argument("--deck-only", action="store_true",
                    help="Skip the digest entirely — only build the deck. No API key needed.")
    ap.add_argument("--no-vision", action="store_true",
                    help="Disable vision-based frame picking for the digest. Cheaper but the "
                         "frames may be less illustrative.")
    ap.add_argument("--digest-model",
                    default=os.environ.get("YT2MD_DIGEST_MODEL") or "claude-sonnet-4-6",
                    help="Claude model for the digest (default: claude-sonnet-4-6). "
                         "Use claude-opus-4-7 for the highest-quality summarization.")
    ap.add_argument("--scene-threshold", type=float, default=0.2,
                    help="Scene-detection sensitivity, 0.1=lots of frames, 0.5=only major changes (default: 0.2)")
    ap.add_argument("--interval", type=float, default=20.0,
                    help="Also sample one frame every N seconds (0 to disable). Useful for "
                         "screen recordings where gradual changes don't trip scene detection. (default: 20)")
    ap.add_argument("--hash-distance", type=int, default=4,
                    help="Perceptual hash dedup threshold; lower = stricter. Compared only against "
                         "the previous kept frame, so recurring views are preserved. (default: 4)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Keep extracted frames in ./frames_<videoname>/ instead of cleaning up")
    ap.add_argument("--downloads-dir", type=Path, default=Path("downloads"),
                    help="Where to cache YouTube downloads (default: ./downloads)")
    ap.add_argument("--source-lang", default=None, metavar="CODE",
                    help="BCP-47 language code of a local SRT (e.g. 'zh-Hans'). "
                         "Drives the digest's output language when --digest-language=auto. "
                         "Ignored for URLs — yt-dlp / Whisper picks the track and the "
                         "lang is read from the file.")
    ap.add_argument("--digest-language",
                    default=os.environ.get("YT2MD_DIGEST_LANGUAGE") or "auto",
                    choices=["auto", "en"],
                    help="Output language for the digest + panel discussion. "
                         "'auto' (default) writes in the transcript's language. "
                         "'en' forces English regardless of source.")
    ap.add_argument("--whisper-model",
                    default=os.environ.get("YT2MD_WHISPER_MODEL") or DEFAULT_WHISPER_MODEL,
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    help=f"faster-whisper model used as fallback when a YouTube "
                         f"video has no captions (default: {DEFAULT_WHISPER_MODEL}).")
    ap.add_argument("--no-whisper", action="store_true",
                    help="Disable Whisper fallback; fail when a video has no captions.")
    ap.add_argument("--cookies-from-browser", default=os.environ.get("YT2MD_COOKIES_FROM_BROWSER"),
                    choices=["chrome", "firefox", "safari", "brave", "edge",
                             "chromium", "opera", "vivaldi"],
                    metavar="BROWSER",
                    help="Pass cookies from this browser to yt-dlp. Required when "
                         "YouTube returns 'Sign in to confirm you're not a bot'. "
                         "Defaults to $YT2MD_COOKIES_FROM_BROWSER if set.")
    args = ap.parse_args()

    load_env_files()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. Install with `brew install ffmpeg` or your package manager.")

    do_digest = not args.deck_only
    if do_digest:
        ensure_api_key()

    import time as _time
    timings: dict = {}
    fetch_meta: dict = {
        "used_whisper": False, "whisper_model": None,
    }
    video_title: Optional[str] = None
    video_url: Optional[str] = None

    if is_url(args.video):
        print(f"[0/5] Fetching YouTube video: {args.video}")
        result = fetch_youtube(
            args.video,
            args.downloads_dir,
            whisper_model=args.whisper_model,
            allow_whisper=not args.no_whisper,
            cookies_from_browser=args.cookies_from_browser,
        )
        video_path = result["mp4"]
        srt_path = result["srt"]
        source_lang = result["lang"]
        video_title = result.get("title")
        video_url = result.get("webpage_url")
        timings["download"] = round(result["download_secs"], 3)
        timings["whisper"] = round(result["whisper_secs"], 3)
        fetch_meta["used_whisper"] = result["used_whisper"]
        fetch_meta["whisper_model"] = result["whisper_model"]
        print(f"      mp4: {video_path}")
        print(f"      srt: {srt_path} (lang: {source_lang})")
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            sys.exit(f"Video not found: {video_path}")
        if args.srt is None:
            sys.exit("SRT path is required when 'video' is a local file.")
        srt_path = Path(args.srt)
        if not srt_path.exists():
            sys.exit(f"SRT not found: {srt_path}")
        # Local-file path: caller didn't tell us the language; assume English.
        # Override with --source-lang if you're digesting a non-English local SRT.
        source_lang = args.source_lang or "en"

    base = video_path.stem
    digest_path = args.output if args.output is not None else Path(f"{base}_digest.md")
    if args.deck == "__default__":
        deck_path = Path(f"{base}_deck.pptx")
    elif args.deck is not None:
        deck_path = Path(args.deck)
    else:
        deck_path = None
    if args.deck_only and deck_path is None:
        deck_path = Path(f"{base}_deck.pptx")

    workdir = Path(tempfile.mkdtemp(prefix="v2d_"))
    scene_dir = workdir / "scene"
    interval_dir = workdir / "interval"

    try:
        duration = get_video_duration(video_path)

        print(f"[1/5] Extracting frames "
              f"(scene threshold={args.scene_threshold}, interval={args.interval}s)...")
        _frames_t0 = _time.monotonic()
        scene_frames = extract_scene_frames(video_path, scene_dir, args.scene_threshold)
        interval_frames = extract_interval_frames(video_path, interval_dir, args.interval, duration)
        frames = merge_frames(scene_frames, interval_frames)
        print(f"      {len(scene_frames)} scene + {len(interval_frames)} interval = "
              f"{len(frames)} candidate frames")

        print(f"[2/5] Deduping consecutive near-identical frames (hash distance <= {args.hash_distance})...")
        frames = dedupe_frames(frames, args.hash_distance)
        print(f"      {len(frames)} unique frames")
        timings["frames"] = round(_time.monotonic() - _frames_t0, 3)

        print(f"[3/5] Parsing SRT: {srt_path.name}")
        segments = parse_srt(srt_path)
        print(f"      {len(segments)} transcript segments")

        print("[4/5] Aligning transcript to frames...")
        slides_data = assign_transcript_to_frames(frames, segments, duration)

        if deck_path is not None:
            print(f"[5/5] Building deck -> {deck_path}")
            build_deck(slides_data, deck_path, video_path.stem)
        else:
            print("[5/5] Skipping deck (use --deck to also write one)")

        usage = None
        if do_digest:
            digest_path.parent.mkdir(parents=True, exist_ok=True)
            images_dir = digest_path.parent / f"{digest_path.stem}_images"
            print(f"[+] Generating digest with {args.digest_model} -> {digest_path}")
            _digest_t0 = _time.monotonic()
            digest, usage = generate_digest(
                segments, video_title or video_path.stem, args.digest_model,
                source_lang=source_lang,
                output_language=args.digest_language,
            )
            timings["digest"] = round(_time.monotonic() - _digest_t0, 3)
            print(f"      {len(digest.topics)} topics  |  "
                  f"input: {usage.input_tokens} tokens "
                  f"(cache read: {getattr(usage, 'cache_read_input_tokens', 0)}, "
                  f"cache write: {getattr(usage, 'cache_creation_input_tokens', 0)})  |  "
                  f"output: {usage.output_tokens} tokens")

            vision_picks = None
            if not args.no_vision:
                print(f"[+] Vision-picking frames with {args.digest_model}...")
                _vision_t0 = _time.monotonic()
                vision_picks, v_usage = vision_pick_frames(
                    digest, frames, duration, args.digest_model, segments=segments
                )
                timings["vision"] = round(_time.monotonic() - _vision_t0, 3)
                print(f"      vision-selected {len(vision_picks)}/{len(digest.topics)} topics  |  "
                      f"input: {v_usage.input_tokens} tokens  |  "
                      f"output: {v_usage.output_tokens} tokens")

            write_markdown_digest(
                digest, frames, duration, digest_path, images_dir, vision_picks,
                video_title=video_title, video_url=video_url,
            )
            print(f"      Digest written. Images in {images_dir}/")

        if args.keep_frames:
            dest = Path.cwd() / f"frames_{video_path.stem}"
            dest.mkdir(parents=True, exist_ok=True)
            for d in (scene_dir, interval_dir):
                if d.exists():
                    shutil.copytree(d, dest, dirs_exist_ok=True)
            print(f"      frames saved to {dest}")

        outputs = []
        if do_digest:
            outputs.append(str(digest_path))
        if deck_path is not None:
            outputs.append(str(deck_path))
        print(f"\nDone. Wrote: {', '.join(outputs)}")

        # Structured one-line summary parsed by the web reaper. Keep this on
        # one line and as the LAST thing printed on success.
        summary = {
            "source_lang": source_lang,
            "used_whisper": fetch_meta["used_whisper"],
            "whisper_model": fetch_meta["whisper_model"],
            "timings": timings,
            "tokens": {
                "input": getattr(usage, "input_tokens", None) if usage else None,
                "output": getattr(usage, "output_tokens", None) if usage else None,
                "cache_read": getattr(usage, "cache_read_input_tokens", None) if usage else None,
                "cache_creation": getattr(usage, "cache_creation_input_tokens", None) if usage else None,
            } if usage else None,
            "digest_path": str(digest_path) if do_digest else None,
        }
        print("[summary] " + json.dumps(summary))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
