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
    """
    Drop a frame only if it's near-identical to the *previous kept* frame.
    Comparing only consecutively preserves recurring views (e.g., returning to an editor).
    """
    import imagehash
    from PIL import Image

    kept: List[Tuple[Path, float]] = []
    prev_hash = None
    for path, ts in frames:
        h = imagehash.phash(Image.open(path))
        if prev_hash is not None and (h - prev_hash) <= hash_distance:
            path.unlink(missing_ok=True)
            continue
        kept.append((path, ts))
        prev_hash = h
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
            f"  mkdir -p {e.parent} && echo 'ANTHROPIC_API_KEY=sk-ant-...' > {e}"
        )

    print(msg)
    key = input("Paste your API key (or press Enter to abort): ").strip()
    if not key:
        sys.exit("Aborted.")

    save = input(
        f"Save it to {e} so future runs find it automatically? [Y/n] "
    ).strip().lower()
    if save in ("", "y", "yes"):
        e.parent.mkdir(parents=True, exist_ok=True)
        e.write_text(f"ANTHROPIC_API_KEY={key}\n")
        try:
            os.chmod(e, 0o600)
        except OSError:
            pass
        print(f"      saved to {e}")
    os.environ["ANTHROPIC_API_KEY"] = key


# ---------- YouTube fetch ----------

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def fetch_youtube(url: str, cache_root: Path) -> Tuple[Path, Path]:
    """Download mp4 + English SRT from YouTube. Cached by video ID under cache_root.

    Returns (mp4_path, srt_path).
    """
    import yt_dlp

    cache_root.mkdir(parents=True, exist_ok=True)

    # Probe first to get the video ID for stable cache layout
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    video_id = info["id"]
    out_dir = cache_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f"{video_id}.mp4"
    srt_path = out_dir / f"{video_id}.en.srt"

    if mp4_path.exists() and srt_path.exists():
        print(f"      using cached {out_dir}/")
        return mp4_path, srt_path

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "srt/vtt/best",
        "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Resolve actual filenames yt-dlp produced (lang suffix can vary)
    candidates_mp4 = list(out_dir.glob(f"{video_id}.*"))
    mp4_found = next((p for p in candidates_mp4 if p.suffix in (".mp4", ".mkv", ".webm")), None)
    if mp4_found and mp4_found != mp4_path:
        mp4_found.rename(mp4_path)
    elif not mp4_path.exists():
        raise RuntimeError(f"yt-dlp finished but no video file found in {out_dir}")

    srt_found = next((p for p in out_dir.glob(f"{video_id}.*.srt")), None)
    if srt_found and srt_found != srt_path:
        srt_found.rename(srt_path)
    elif not srt_path.exists():
        raise RuntimeError(
            f"No English subtitles available for {url} (neither manual nor auto-captions)."
        )

    return mp4_path, srt_path


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
):
    """Call Claude to segment the transcript into topics. Returns a parsed VideoDigest."""
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

    user_text = (
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
):
    """Use Claude's vision to pick the best frame per topic from in-window candidates.

    Returns a dict {topic_index -> chosen_frame_path}, plus the API usage object.
    Topics with no in-window candidates are omitted (caller falls back to timestamp-based pick).
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
    for i, topic in enumerate(topics):
        end = topics[i + 1].start_time if i + 1 < len(topics) else video_duration
        per_topic.append(_candidates_for_topic(topic.start_time, end, frames))

    # Build the message: text intro -> for each topic, label + summary + numbered candidate images
    content: list = []
    intro = (
        "For each topic below, pick the candidate frame that best illustrates what the "
        "narrator is discussing. Prefer frames showing the most informative visual content "
        "(diagrams, code, distinctive UI) over generic framing or talking-head shots. "
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
        header = (
            f"\n--- Topic {ti} ---\n"
            f"Title: {topic.title}\n"
            f"Summary: {topic.summary}\n"
            f"Candidates ({len(cands)} frames):\n"
        )
        content.append({"type": "text", "text": header})
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
        "topics, each with a small set of candidate frames. For each topic with candidates, "
        "return the (topic_index, candidate_index) pair that best illustrates the topic, "
        "with a one-sentence rationale. Skip topics that say 'no candidates available'."
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
) -> None:
    """Render the digest as Markdown with <img> tags. Copies frames into images_dir.

    If vision_picks is provided, prefer those mappings; fall back to timestamp-based picks
    for any topic not covered.
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
    lines.append(f"# {digest.title}")
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


# ---------- Subcommands: watch / meta / schedule ----------

SCHEDULE_LABEL_POLL = "com.youtube-to-markdown.poll"
SCHEDULE_LABEL_META = "com.youtube-to-markdown.meta"
LATEST_LIMIT = 10
MAX_NEW_PER_RUN = 3
META_LOOKBACK_DAYS = 7
META_MIN_DIGESTS = 2
META_MODEL_DEFAULT = "claude-sonnet-4-6"


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


def _digest_video(video_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest_path = output_dir / "digest.md"
    yt2md = shutil.which("yt2md") or sys.argv[0]
    subprocess.run(
        [yt2md, f"https://youtu.be/{video_id}", "-o", str(digest_path)],
        check=True,
        cwd=output_dir,
    )


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
            try:
                _digest_video(vid, digests_dir / vid)
                seen.add(vid)
                state["channels"][channel_url] = {"seen": sorted(seen)}
                save_state(state)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED on {vid}: {e}", file=sys.stderr)
                any_failures = True

    save_state(state)
    return 1 if any_failures else 0


# ---- meta subcommand ----

META_SYSTEM_PROMPT = (
    "You synthesize multiple video digests into a single readable cross-cutting "
    "meta-digest. Match the tone of the source digests: direct, concrete, "
    "reader-first. Quote real claims, name real people and products, cite real "
    "numbers. This is synthesis, not summary of summaries — find the through-lines."
)

META_USER_PROMPT_TEMPLATE = """\
Below are {n} video digests added or modified in the last {days} days. Produce a \
~600–1000 word meta-digest with:

1. A one-line teaser per video.
2. 3–5 themes spanning multiple videos. Under each theme, weave together specific points across the digests, with backlinks to source digests using relative paths like `[Spiegel on distribution](../digests/-7Yol5vX5xw/digest.md)`.
3. An optional final "Standout single-video items" section for anything notable that didn't fit a theme.

Drop a theme rather than padding if it doesn't have substance. The output is the meta-digest itself — no preamble.

---

{digests}
"""


def _find_recent_digests(digests_dir: Path, lookback_days: int):
    """Return list of (video_id, digest_path) for digests modified in lookback window."""
    if not digests_dir.exists():
        return []
    import datetime as _dt
    cutoff = _dt.datetime.now().timestamp() - lookback_days * 86400
    results = []
    for video_dir in sorted(digests_dir.iterdir()):
        digest = video_dir / "digest.md"
        if not video_dir.is_dir() or not digest.exists():
            continue
        if digest.stat().st_mtime >= cutoff:
            results.append((video_dir.name, digest))
    return results


def cmd_meta_run(args) -> int:
    ensure_api_key()
    import datetime as _dt
    import anthropic

    data_dir = get_data_dir()
    digests_dir = data_dir / "digests"
    meta_dir = data_dir / "meta"

    recent = _find_recent_digests(digests_dir, args.lookback_days)
    if len(recent) < args.min_digests:
        print(
            f"Only {len(recent)} digest(s) modified in the last {args.lookback_days} days "
            f"(need at least {args.min_digests}). Skipping."
        )
        return 0

    sections = []
    for vid, path in recent:
        sections.append(
            f"## Video {vid}\n\nSource path: digests/{vid}/digest.md\n\n{path.read_text()}"
        )
    digests_blob = "\n\n---\n\n".join(sections)
    user_text = META_USER_PROMPT_TEMPLATE.format(
        n=len(recent), days=args.lookback_days, digests=digests_blob
    )

    iso_year, iso_week, _ = _dt.datetime.now().isocalendar()
    out_path = meta_dir / f"{iso_year}-W{iso_week:02d}.md"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing {len(recent)} digests with {args.model} -> {out_path}")
    for vid, _p in recent:
        print(f"  - {vid}")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        system=META_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{
            "type": "text",
            "text": user_text,
            "cache_control": {"type": "ephemeral"},
        }]}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    out_path.write_text(text.rstrip() + "\n")

    print(f"Wrote {out_path} ({len(text)} chars)")
    print(
        f"Tokens: input={response.usage.input_tokens}, "
        f"output={response.usage.output_tokens}, "
        f"cache_read={getattr(response.usage, 'cache_read_input_tokens', 0)}"
    )
    return 0


# ---- schedule subcommands ----

LAUNCHD_DIR = Path.home() / "Library" / "LaunchAgents"


def _launchd_path_value() -> str:
    """PATH for launchd-spawned processes (HOME-anchored, includes uv-tool bin and brew)."""
    home = str(Path.home())
    return ":".join([
        f"{home}/.local/bin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ])


DEFAULT_SCHEDULE_CONFIG = {
    "poll_interval_hours": 6,
    "meta_frequency": "weekly",   # "daily" or "weekly"
    "meta_weekday": 0,            # 0=Sun ... 6=Sat (only used when weekly)
    "meta_hour": 9,
    "meta_minute": 0,
}


def _schedule_config_file() -> Path:
    return get_data_dir() / "schedule.json"


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


def _format_schedule_summary(cfg: dict) -> str:
    """Human-readable description of the schedule config."""
    poll = cfg["poll_interval_hours"]
    poll_str = f"every {poll} hour{'s' if poll != 1 else ''}"
    weekday_names = ["Sundays", "Mondays", "Tuesdays", "Wednesdays",
                     "Thursdays", "Fridays", "Saturdays"]
    time_str = f"{cfg['meta_hour']:02d}:{cfg['meta_minute']:02d}"
    if cfg["meta_frequency"] == "daily":
        meta_str = f"daily at {time_str} local"
    else:
        meta_str = f"{weekday_names[cfg['meta_weekday']]} at {time_str} local"
    return f"polling {poll_str}, meta-digest {meta_str}"


def _make_poll_plist(yt2md_path: str, data_dir: Path, cfg: dict = None) -> str:
    cfg = cfg or DEFAULT_SCHEDULE_CONFIG
    interval_seconds = max(60, int(cfg["poll_interval_hours"] * 3600))
    log_path = data_dir / "logs" / "poll.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SCHEDULE_LABEL_POLL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{yt2md_path}</string>
        <string>watch</string>
        <string>run</string>
    </array>

    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{_launchd_path_value()}</string>
        <key>YT2MD_DATA</key>
        <string>{data_dir}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
"""


def _make_meta_plist(yt2md_path: str, data_dir: Path, cfg: dict = None) -> str:
    cfg = cfg or DEFAULT_SCHEDULE_CONFIG
    log_path = data_dir / "logs" / "meta.log"
    if cfg["meta_frequency"] == "daily":
        cal_inner = (
            f"        <key>Hour</key>\n        <integer>{int(cfg['meta_hour'])}</integer>\n"
            f"        <key>Minute</key>\n        <integer>{int(cfg['meta_minute'])}</integer>"
        )
    else:
        cal_inner = (
            f"        <key>Weekday</key>\n        <integer>{int(cfg['meta_weekday'])}</integer>\n"
            f"        <key>Hour</key>\n        <integer>{int(cfg['meta_hour'])}</integer>\n"
            f"        <key>Minute</key>\n        <integer>{int(cfg['meta_minute'])}</integer>"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SCHEDULE_LABEL_META}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{yt2md_path}</string>
        <string>meta</string>
        <string>run</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
{cal_inner}
    </dict>

    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{_launchd_path_value()}</string>
        <key>YT2MD_DATA</key>
        <string>{data_dir}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
"""


def _do_schedule_install(cfg: dict = None):
    """Install both launchd jobs using cfg (or saved config). Returns (success, messages)."""
    if cfg is None:
        cfg = load_schedule_config()
    yt2md_path = shutil.which("yt2md")
    if not yt2md_path:
        return False, ["yt2md not found on PATH (install with `uv tool install ...`)"]
    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            return False, [f"required tool '{tool}' is not on PATH"]

    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    LAUNCHD_DIR.mkdir(parents=True, exist_ok=True)
    save_schedule_config(cfg)

    poll_desc = f"polling every {cfg['poll_interval_hours']}h"
    if cfg["meta_frequency"] == "daily":
        meta_desc = f"meta-digest daily at {cfg['meta_hour']:02d}:{cfg['meta_minute']:02d}"
    else:
        weekday_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        meta_desc = (f"meta-digest {weekday_names[cfg['meta_weekday']]}s "
                     f"at {cfg['meta_hour']:02d}:{cfg['meta_minute']:02d}")

    plists = [
        (SCHEDULE_LABEL_POLL, _make_poll_plist(yt2md_path, data_dir, cfg), poll_desc),
        (SCHEDULE_LABEL_META, _make_meta_plist(yt2md_path, data_dir, cfg), meta_desc),
    ]
    messages = []
    for label, content, desc in plists:
        plist_path = LAUNCHD_DIR / f"{label}.plist"
        plist_path.write_text(content)
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False, [f"failed to load {label}: {result.stderr.strip()}"]
        messages.append(f"{label} — {desc}")
    return True, messages


def _do_schedule_uninstall():
    """Uninstall both launchd jobs. Returns list of (label, action_taken)."""
    results = []
    for label in (SCHEDULE_LABEL_POLL, SCHEDULE_LABEL_META):
        plist_path = LAUNCHD_DIR / f"{label}.plist"
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if plist_path.exists():
            plist_path.unlink()
            results.append((label, "removed"))
        else:
            results.append((label, "was not installed"))
    return results


_STATUS_CACHE: dict = {}  # label -> (expires_at, info_dict)
_STATUS_TTL = 30.0  # seconds


def _invalidate_status_cache() -> None:
    _STATUS_CACHE.clear()


def _job_status(label: str) -> dict:
    """Return loaded/state/runs/last_exit for a launchd label.

    Cached for ~30 seconds — `launchctl print` shells out and is the slowest
    thing the schedule page does. Mutating actions (install/uninstall/run-now)
    must call _invalidate_status_cache() so users see fresh data after acting.
    """
    import time
    now = time.time()
    cached = _STATUS_CACHE.get(label)
    if cached and cached[0] > now:
        return cached[1]

    plist_path = LAUNCHD_DIR / f"{label}.plist"
    info = {
        "label": label,
        "plist_exists": plist_path.exists(),
        "loaded": False,
        "state": None,
        "runs": None,
        "last_exit": None,
        "pid": None,
    }
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        info["loaded"] = True
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("state ="):
                info["state"] = line.split("=", 1)[1].strip()
            elif line.startswith("runs ="):
                info["runs"] = line.split("=", 1)[1].strip()
            elif line.startswith("last exit code ="):
                info["last_exit"] = line.split("=", 1)[1].strip()
            elif line.startswith("pid ="):
                info["pid"] = line.split("=", 1)[1].strip()

    _STATUS_CACHE[label] = (now + _STATUS_TTL, info)
    return info


def _job_summary(status: dict) -> Tuple[str, str]:
    """Translate a launchd status dict into a one-sentence English summary.

    Returns (sentence, dot_class) where dot_class is one of dot-on/dot-warn/dot-off.
    """
    if not status["loaded"]:
        return ("Schedule isn't installed yet.", "dot-off")
    if status["state"] == "running":
        pid = status.get("pid", "?")
        return (f"Running now (PID {pid}).", "dot-on")
    runs = status.get("runs")
    last = status.get("last_exit")
    if runs in (None, "0"):
        return ("Set up — first run hasn't happened yet.", "dot-warn")
    if last and last not in ("0", "(never exited)"):
        return (f"Last run failed (exit code {last}). Check the log.", "dot-warn")
    n = runs if runs else "?"
    return (f"Healthy — ran {n} time{'s' if n != '1' else ''}, last run was clean.", "dot-on")


def _tail_log(path: Path, n: int = 20) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        return f"(error reading log: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"


def cmd_schedule_install(args) -> int:
    cfg = load_schedule_config()
    if args.poll_hours is not None:
        cfg["poll_interval_hours"] = args.poll_hours
    if args.meta_frequency is not None:
        cfg["meta_frequency"] = args.meta_frequency
    if args.meta_time is not None:
        try:
            h, m = args.meta_time.split(":")
            cfg["meta_hour"] = int(h)
            cfg["meta_minute"] = int(m)
        except Exception:
            sys.exit(f"--meta-time must be HH:MM, got: {args.meta_time}")
    if args.meta_weekday is not None:
        cfg["meta_weekday"] = args.meta_weekday

    success, messages = _do_schedule_install(cfg)
    if not success:
        sys.exit(messages[0])
    data_dir = get_data_dir()
    print(f"Installed 2 launchd jobs (data dir = {data_dir})")
    for m in messages:
        print(f"  ✓ {m}")
    print(f"\nSchedule config saved to {_schedule_config_file()}")
    print(
        f"\nLogs:\n"
        f"  tail -f {data_dir/'logs'/'poll.log'}\n"
        f"  tail -f {data_dir/'logs'/'meta.log'}\n"
        f"\nUninstall: yt2md schedule uninstall"
    )
    return 0


def cmd_schedule_uninstall(args) -> int:
    for label, action in _do_schedule_uninstall():
        print(f"  {'✓' if action == 'removed' else '-'} {label} {action}")
    return 0


def cmd_schedule_status(args) -> int:
    for label in (SCHEDULE_LABEL_POLL, SCHEDULE_LABEL_META):
        info = _job_status(label)
        print(f"--- {label}")
        if not info["loaded"]:
            print("  not loaded")
            continue
        for k in ("state", "runs", "last_exit", "pid"):
            if info.get(k) is not None:
                print(f"  {k} = {info[k]}")
    return 0


# ---- one-off digest jobs (in-memory tracking; cheap) ----

# Module-level dict: PID -> {"video_id": str, "started": float, "url": str}.
# Lost on server restart, which is fine — the digest still completes (detached
# subprocess) and shows up in the sidebar when done.
_oneoff_jobs: dict = {}


_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    """Pull a YouTube video ID from common URL forms. Returns '' if not found."""
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else ""


def _list_active_oneoff_jobs() -> list:
    """Return one-off jobs whose subprocesses are still alive."""
    active = []
    for pid in list(_oneoff_jobs.keys()):
        try:
            os.kill(pid, 0)  # zero-signal probes liveness without killing
            info = _oneoff_jobs[pid]
            active.append({"pid": pid, **info})
        except (ProcessLookupError, PermissionError):
            del _oneoff_jobs[pid]
    return active


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
        CREATE TABLE IF NOT EXISTS meta_reads (
            week TEXT PRIMARY KEY,
            opened_at INTEGER NOT NULL
        );
    """)
    return conn


def _mark_digest_read(digest_id: str) -> None:
    import time
    with _library_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO digest_reads(digest_id, opened_at) VALUES (?, ?)",
            (digest_id, int(time.time())),
        )


def _mark_meta_read(week: str) -> None:
    import time
    with _library_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta_reads(week, opened_at) VALUES (?, ?)",
            (week, int(time.time())),
        )


def _read_digest_ids() -> set:
    with _library_connect() as conn:
        rows = conn.execute("SELECT digest_id FROM digest_reads").fetchall()
        return {r[0] for r in rows}


def _read_meta_weeks() -> set:
    with _library_connect() as conn:
        rows = conn.execute("SELECT week FROM meta_reads").fetchall()
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
aside li { margin: 0; }
aside li a {
  display: -webkit-box;
  -webkit-line-clamp: 2;
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

  <nav aria-label="Manage">
  <h2>Manage</h2>
  <ul>
    <li{% if current == 'channels' %} class="active"{% endif %}><a href="/channels">Subscriptions ({{ channel_count }})</a></li>
    <li{% if current == 'one-off' %} class="active"{% endif %}><a href="/one-off">One-off digest</a></li>
    <li{% if current == 'schedule' %} class="active"{% endif %}><a href="/schedule">Schedule</a></li>
  </ul>
  </nav>

  <nav aria-label="Weekly meta-digests">
  <h2>Meta-digests {% if unread_meta_count %}<span class="unread-count">{{ unread_meta_count }} new</span>{% else %}({{ metas|length }}){% endif %}</h2>
  {% for m in metas %}
  <a class="meta-card{% if current == 'meta:' + m.week %} active{% endif %}{% if m.unread %} unread{% endif %}" href="/meta/{{ m.week }}/">
    <div class="week">{% if m.unread %}<span class="unread-dot" aria-label="unread"></span>{% endif %}{{ m.date_range }}</div>
    <div class="count">{% if m.count %}covers {{ m.count }} video{{ 's' if m.count != 1 else '' }}{% else %}meta-digest{% endif %}</div>
  </a>
  {% else %}
  <p class="empty">none yet</p>
  {% endfor %}
  </nav>

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


def _iso_week_display(stem: str) -> str:
    """Convert '2026-W18' into a friendly date range like 'Apr 27 – May 3'."""
    import datetime as _dt
    try:
        year_str, week_str = stem.split("-W")
        year, week = int(year_str), int(week_str)
        jan4 = _dt.date(year, 1, 4)
        week1_monday = jan4 - _dt.timedelta(days=jan4.isoweekday() - 1)
        monday = week1_monday + _dt.timedelta(weeks=week - 1)
        sunday = monday + _dt.timedelta(days=6)
    except Exception:
        return stem
    # %-d not portable; use .day directly.
    if monday.year == sunday.year:
        return f"{monday.strftime('%b')} {monday.day} – {sunday.strftime('%b')} {sunday.day}"
    return (
        f"{monday.strftime('%b')} {monday.day}, {monday.year} – "
        f"{sunday.strftime('%b')} {sunday.day}, {sunday.year}"
    )


def _list_metas(meta_dir: Path) -> List[dict]:
    if not meta_dir.exists():
        return []
    results = []
    for f in sorted(meta_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Count unique digest backlinks to show "covers N videos" in the card.
        try:
            text = f.read_text()
            count = len(set(re.findall(r"digests/([^/)\"\s]+)/digest\.md", text)))
        except Exception:
            count = 0
        results.append({
            "week": f.stem,
            "date_range": _iso_week_display(f.stem),
            "mtime": f.stat().st_mtime,
            "count": count,
        })
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
    meta_dir = data_dir / "meta"

    app = Flask(__name__)
    # Disable Flask's default request logging — keep stdout clean.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def page(body: str, *, title: str, current: str, base_href: str = None):
        # Annotate listing items with read state so the sidebar can show "new" markers.
        digests = _list_digests(digests_dir)
        metas = _list_metas(meta_dir)
        try:
            read_digests = _read_digest_ids()
            read_metas = _read_meta_weeks()
        except Exception:
            read_digests, read_metas = set(), set()
        for d in digests:
            d["unread"] = d["id"] not in read_digests
        for m in metas:
            m["unread"] = m["week"] not in read_metas
        unread_digest_count = sum(1 for d in digests if d["unread"])
        unread_meta_count = sum(1 for m in metas if m["unread"])
        return render_template_string(
            SERVE_PAGE_TEMPLATE,
            body=body,
            title=title,
            current=current,
            base_href=base_href,
            digests=digests,
            metas=metas,
            channel_count=len(read_channels()),
            unread_digest_count=unread_digest_count,
            unread_meta_count=unread_meta_count,
        )

    @app.route("/")
    def home():
        digests = _list_digests(digests_dir)
        metas = _list_metas(meta_dir)
        channels = read_channels()

        # Empty states first — show a real CTA, not a list of zero items.
        if not digests and not metas:
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
                    "will appear after the next polling run (every 6 hours, or "
                    'fire one now from the <a href="/schedule">Schedule</a> page).</p>'
                )
            return page(body, title="yt2md", current="home")

        # Featured content: prefer the latest meta-digest, fall back to the latest digest.
        if metas:
            featured = metas[0]
            featured_md = (meta_dir / f"{featured['week']}.md").read_text()
            try:
                _mark_meta_read(featured["week"])
            except Exception:
                pass
            body = (
                f'<p class="featured-eyebrow">Weekly meta-digest · '
                f'<a href="/meta/{featured["week"]}/">{featured["date_range"]}</a> · '
                f'covers {featured["count"]} video{"s" if featured["count"] != 1 else ""}</p>'
            )
            body += _render_markdown(featured_md)
            base_href = f"/meta/{featured['week']}/"
        else:
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
        poll_status = _job_status(SCHEDULE_LABEL_POLL)
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
        elif not poll_status["loaded"]:
            next_step = (
                'You\'re subscribed but the polling schedule isn\'t set up yet. '
                '<a href="/schedule">Install the schedule</a> to start auto-digesting new videos every 6 hours.'
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
        flash = request.args.get("msg", "")
        cfg = load_schedule_config()
        poll_status = _job_status(SCHEDULE_LABEL_POLL)
        meta_status = _job_status(SCHEDULE_LABEL_META)
        any_loaded = poll_status["loaded"] or meta_status["loaded"]

        body = "<h1>Schedule</h1>"
        if flash:
            body += f'<div class="flash">{h(flash)}</div>'

        # Configurable schedule form — same form for install and reconfigure.
        weekday_opts = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        body += '<form method="post" action="/schedule/install" class="schedule-form">'
        body += '<div class="schedule-fields">'
        body += '<label>Polling interval'
        body += f'  <input type="number" name="poll_hours" value="{cfg["poll_interval_hours"]}" min="0.1" step="0.1" required>'
        body += '  <span class="suffix">hours</span>'
        body += '</label>'

        body += '<label>Meta-digest frequency'
        body += '  <select name="meta_frequency">'
        for opt in ("daily", "weekly"):
            sel = ' selected' if cfg["meta_frequency"] == opt else ''
            body += f'    <option value="{opt}"{sel}>{opt}</option>'
        body += '  </select>'
        body += '</label>'

        body += '<label class="weekday-label">Day of week'
        body += '  <select name="meta_weekday">'
        for i, name in enumerate(weekday_opts):
            sel = ' selected' if cfg["meta_weekday"] == i else ''
            body += f'    <option value="{i}"{sel}>{name}</option>'
        body += '  </select>'
        body += '</label>'

        body += '<label>Time'
        body += f'  <input type="time" name="meta_time" value="{cfg["meta_hour"]:02d}:{cfg["meta_minute"]:02d}" required>'
        body += '</label>'
        body += '</div>'  # schedule-fields

        action = "Save & reinstall" if any_loaded else "Install both jobs"
        body += f'<button type="submit" class="primary">{action}</button>'
        if any_loaded:
            body += (
                '<form method="post" action="/schedule/uninstall" style="display:inline; margin-left: 8px;">'
                '<button type="submit">Uninstall</button></form>'
            )
        body += '</form>'

        body += (
            f'<p class="meta-info">Current schedule: {h(_format_schedule_summary(cfg))}.</p>'
        )

        for label, status, run_key, friendly, desc in [
            (SCHEDULE_LABEL_POLL, poll_status, "poll", "Polling",
             "every 6 hours · fires <code>yt2md watch run</code>"),
            (SCHEDULE_LABEL_META, meta_status, "meta", "Meta-digest",
             "Sundays at 9am local · fires <code>yt2md meta run</code>"),
        ]:
            # English-language status sentence + dot color signaling actual health.
            sentence, dot_class = _job_summary(status)
            body += f'<div class="job-block"><h3><span class="dot {dot_class}"></span>{friendly}</h3>'
            body += f'<p class="meta-info" style="margin: 0 0 12px;">{desc}</p>'
            body += f'<p class="job-summary">{sentence}</p>'

            if status["loaded"]:
                body += (
                    f'<div class="job-actions">'
                    f'<form method="post" action="/schedule/run/{run_key}" style="margin:0;">'
                    f'<button type="submit">Run now</button>'
                    f'</form></div>'
                )

            # Disclosure for the diagnostic dump.
            body += '<details><summary>Diagnostics</summary>'
            body += '<table class="status-table">'
            body += f'<tr><td>plist exists</td><td>{status["plist_exists"]}</td></tr>'
            body += f'<tr><td>loaded</td><td>{status["loaded"]}</td></tr>'
            for k in ("state", "runs", "last_exit", "pid"):
                v = status.get(k)
                if v is not None:
                    body += f'<tr><td>{k.replace("_", " ")}</td><td>{h(str(v))}</td></tr>'
            body += '</table></details>'

            log_path = data_dir / "logs" / f"{run_key}.log"
            body += '<details style="margin-top: 8px;"><summary>Recent log (last 20 lines)</summary>'
            body += f'<div class="log-block">{h(_tail_log(log_path, 20))}</div>'
            body += '</details>'
            body += '</div>'

        body += '<p class="meta-info">Refresh the page to see updated status after a run.</p>'
        return page(body, title="Schedule", current="schedule")

    @app.route("/schedule/install", methods=["POST"])
    def schedule_install():
        from flask import redirect, request
        cfg = load_schedule_config()
        # Form fields override the saved config (when present).
        try:
            if request.form.get("poll_hours"):
                cfg["poll_interval_hours"] = float(request.form["poll_hours"])
            if request.form.get("meta_frequency") in ("daily", "weekly"):
                cfg["meta_frequency"] = request.form["meta_frequency"]
            if request.form.get("meta_weekday"):
                cfg["meta_weekday"] = int(request.form["meta_weekday"])
            if request.form.get("meta_time"):
                h_, m_ = request.form["meta_time"].split(":")
                cfg["meta_hour"] = int(h_)
                cfg["meta_minute"] = int(m_)
        except Exception as e:
            return redirect(f"/schedule?msg=Invalid+input:+{e}")
        success, messages = _do_schedule_install(cfg)
        _invalidate_status_cache()
        msg = ("Installed: " + "; ".join(messages)) if success else f"Install failed: {messages[0]}"
        return redirect(f"/schedule?msg={msg}")

    @app.route("/schedule/uninstall", methods=["POST"])
    def schedule_uninstall():
        from flask import redirect
        results = _do_schedule_uninstall()
        _invalidate_status_cache()
        msg = "Uninstalled: " + "; ".join(f"{lbl} ({act})" for lbl, act in results)
        return redirect(f"/schedule?msg={msg}")

    @app.route("/schedule/run/<job>", methods=["POST"])
    def schedule_run(job):
        from flask import redirect
        label_map = {"poll": SCHEDULE_LABEL_POLL, "meta": SCHEDULE_LABEL_META}
        if job not in label_map:
            abort(404)
        label = label_map[job]
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
            capture_output=True, text=True,
        )
        _invalidate_status_cache()
        if result.returncode != 0:
            return redirect(f"/schedule?msg=Failed+to+trigger+{job}:+{result.stderr.strip()}")
        return redirect(
            f"/schedule?msg=Triggered+{job}+(refresh+in+a+few+seconds+to+see+log+update)"
        )

    @app.route("/one-off", methods=["GET"])
    def one_off_page():
        from flask import request
        from html import escape as h
        flash = request.args.get("msg", "")
        active = _list_active_oneoff_jobs()

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

        if active:
            body += "<h2>In progress</h2><ul class='channel-list'>"
            import time as _t
            for j in active:
                elapsed = int(_t.time() - j["started"])
                body += (
                    '<li>'
                    f'<span class="url"><strong>{h(j["video_id"])}</strong> · '
                    f'{h(j["url"])}</span>'
                    f'<span style="color: var(--muted); font-size: 12px;">{elapsed//60}m {elapsed%60}s</span>'
                    '</li>'
                )
            body += "</ul>"

        body += (
            "<p class='meta-info' style='margin-top: 32px;'>"
            "One-off digests share the same library as subscription mode. "
            "They appear in the sidebar's <strong>Digests</strong> section once ready."
            "</p>"
        )
        return page(body, title="One-off digest", current="one-off")

    @app.route("/one-off", methods=["POST"])
    def one_off_submit():
        from flask import redirect, request
        import time as _t
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
                start_new_session=True,
            )
        finally:
            # Subprocess holds its own copy of the fd; safe for us to close.
            log_fd.close()

        _oneoff_jobs[proc.pid] = {
            "video_id": video_id,
            "started": _t.time(),
            "url": url,
        }
        return redirect(
            f"/one-off?msg=Started+digesting+{video_id}+(check+sidebar+in+a+few+minutes)"
        )

    @app.route("/digests/<video_id>/")
    def view_digest(video_id):
        digest_md = digests_dir / video_id / "digest.md"
        if not digest_md.exists():
            abort(404)
        try:
            _mark_digest_read(video_id)
        except Exception:
            pass  # never block reading on a DB error
        html = _render_markdown(digest_md.read_text())
        return page(html, title=video_id, current=f"digest:{video_id}",
                    base_href=f"/digests/{video_id}/")

    @app.route("/digests/<video_id>/digest_images/<path:filename>")
    def digest_image(video_id, filename):
        return send_from_directory(digests_dir / video_id / "digest_images", filename)

    @app.route("/meta/<week>/")
    def view_meta(week):
        meta_md = meta_dir / f"{week}.md"
        if not meta_md.exists():
            abort(404)
        try:
            _mark_meta_read(week)
        except Exception:
            pass
        html = _render_markdown(meta_md.read_text())
        return page(html, title=f"Meta-digest · {_iso_week_display(week)}",
                    current=f"meta:{week}", base_href="/meta/")

    @app.errorhandler(404)
    def not_found(e):
        return page(
            "<h1>Not found</h1><p>That digest or meta-digest doesn't exist yet.</p>",
            title="404", current="home",
        ), 404

    url = f"http://127.0.0.1:{args.port}/"
    print(f"yt2md reader: {url}")
    print(f"Data dir: {data_dir}")
    print("Press Ctrl-C to stop.\n")

    if not args.no_browser:
        import webbrowser
        import threading
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


# ---- subcommand dispatcher ----

def _subcommand_main(argv: List[str]) -> int:
    """Handle yt2md {watch,meta,schedule} ..."""
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

    meta = sub.add_parser("meta", help="Generate cross-cutting meta-digests")
    meta_sub = meta.add_subparsers(dest="meta_cmd", required=True)
    p = meta_sub.add_parser("run", help="Synthesize a meta-digest of recent digests")
    p.add_argument("--lookback-days", type=int, default=META_LOOKBACK_DAYS,
                   help=f"Window in days (default: {META_LOOKBACK_DAYS})")
    p.add_argument("--min-digests", type=int, default=META_MIN_DIGESTS,
                   help=f"Skip if fewer than this many recent digests (default: {META_MIN_DIGESTS})")
    p.add_argument("--model", default=META_MODEL_DEFAULT,
                   help=f"Claude model (default: {META_MODEL_DEFAULT})")
    p.set_defaults(func=cmd_meta_run)

    serve = sub.add_parser("serve", help="Start a local web reader for digests")
    serve.add_argument("--port", type=int, default=7682, help="Port (default: 7682)")
    serve.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open a browser tab on start")
    serve.set_defaults(func=cmd_serve)

    schedule = sub.add_parser("schedule", help="Install/uninstall launchd jobs")
    sched_sub = schedule.add_subparsers(dest="schedule_cmd", required=True)
    p = sched_sub.add_parser("install", help="Install both launchd jobs (saves config to ~/yt2md/schedule.json)")
    p.add_argument("--poll-hours", type=float, default=None, metavar="N",
                   help="Polling interval in hours (default: 6, or last saved value)")
    p.add_argument("--meta-frequency", choices=["daily", "weekly"], default=None,
                   help="Meta-digest cadence (default: weekly)")
    p.add_argument("--meta-weekday", type=int, choices=range(7), default=None, metavar="N",
                   help="Day of week for weekly meta-digest: 0=Sun ... 6=Sat (default: 0)")
    p.add_argument("--meta-time", type=str, default=None, metavar="HH:MM",
                   help="Time of day for meta-digest (default: 09:00)")
    p.set_defaults(func=cmd_schedule_install)
    p = sched_sub.add_parser("uninstall", help="Remove both launchd jobs")
    p.set_defaults(func=cmd_schedule_uninstall)
    p = sched_sub.add_parser("status", help="Show launchd job status")
    p.set_defaults(func=cmd_schedule_status)

    args = ap.parse_args(argv)
    return args.func(args)


# ---------- Main ----------

def main():
    # Subcommand dispatch — short-circuit the single-video flow when the user
    # invokes yt2md watch / meta / schedule.
    if len(sys.argv) > 1 and sys.argv[1] in ("watch", "meta", "schedule", "serve"):
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
    ap.add_argument("--digest-model", default="claude-sonnet-4-6",
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
    args = ap.parse_args()

    load_env_files()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. Install with `brew install ffmpeg` or your package manager.")

    do_digest = not args.deck_only
    if do_digest:
        ensure_api_key()

    if is_url(args.video):
        print(f"[0/5] Fetching YouTube video: {args.video}")
        video_path, srt_path = fetch_youtube(args.video, args.downloads_dir)
        print(f"      mp4: {video_path}")
        print(f"      srt: {srt_path}")
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            sys.exit(f"Video not found: {video_path}")
        if args.srt is None:
            sys.exit("SRT path is required when 'video' is a local file.")
        srt_path = Path(args.srt)
        if not srt_path.exists():
            sys.exit(f"SRT not found: {srt_path}")

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
        scene_frames = extract_scene_frames(video_path, scene_dir, args.scene_threshold)
        interval_frames = extract_interval_frames(video_path, interval_dir, args.interval, duration)
        frames = merge_frames(scene_frames, interval_frames)
        print(f"      {len(scene_frames)} scene + {len(interval_frames)} interval = "
              f"{len(frames)} candidate frames")

        print(f"[2/5] Deduping consecutive near-identical frames (hash distance <= {args.hash_distance})...")
        frames = dedupe_frames(frames, args.hash_distance)
        print(f"      {len(frames)} unique frames")

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

        if do_digest:
            digest_path.parent.mkdir(parents=True, exist_ok=True)
            images_dir = digest_path.parent / f"{digest_path.stem}_images"
            print(f"[+] Generating digest with {args.digest_model} -> {digest_path}")
            digest, usage = generate_digest(segments, video_path.stem, args.digest_model)
            print(f"      {len(digest.topics)} topics  |  "
                  f"input: {usage.input_tokens} tokens "
                  f"(cache read: {getattr(usage, 'cache_read_input_tokens', 0)}, "
                  f"cache write: {getattr(usage, 'cache_creation_input_tokens', 0)})  |  "
                  f"output: {usage.output_tokens} tokens")

            vision_picks = None
            if not args.no_vision:
                print(f"[+] Vision-picking frames with {args.digest_model}...")
                vision_picks, v_usage = vision_pick_frames(
                    digest, frames, duration, args.digest_model
                )
                print(f"      vision-selected {len(vision_picks)}/{len(digest.topics)} topics  |  "
                      f"input: {v_usage.input_tokens} tokens  |  "
                      f"output: {v_usage.output_tokens} tokens")

            write_markdown_digest(digest, frames, duration, digest_path, images_dir, vision_picks)
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
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
