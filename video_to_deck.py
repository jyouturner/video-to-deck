#!/usr/bin/env python3
"""
video_to_deck.py — Turn a screen-recording MP4 + SRT transcript into a PowerPoint deck.

Pipeline:
  1. FFmpeg scene detection -> candidate frames + timestamps
  2. Perceptual-hash dedup -> drop near-identical frames
  3. Parse SRT transcript -> timestamped segments
  4. Align transcript chunks to each frame's time window
  5. Build .pptx with frame as visual + transcript as speaker notes

Usage:
    python video_to_deck.py input.mp4 transcript.srt -o output.pptx
    python video_to_deck.py input.mp4 transcript.srt --scene-threshold 0.25

Requirements:
    System:  ffmpeg, ffprobe
    Python:  pip install imagehash Pillow python-pptx
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


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


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="Input MP4 file")
    ap.add_argument("srt", type=Path, help="SRT transcript file")
    ap.add_argument("-o", "--output", type=Path, default=Path("output.pptx"))
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
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    if not args.srt.exists():
        sys.exit(f"SRT not found: {args.srt}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    workdir = Path(tempfile.mkdtemp(prefix="v2d_"))
    scene_dir = workdir / "scene"
    interval_dir = workdir / "interval"

    try:
        duration = get_video_duration(args.video)

        print(f"[1/5] Extracting frames "
              f"(scene threshold={args.scene_threshold}, interval={args.interval}s)...")
        scene_frames = extract_scene_frames(args.video, scene_dir, args.scene_threshold)
        interval_frames = extract_interval_frames(args.video, interval_dir, args.interval, duration)
        frames = merge_frames(scene_frames, interval_frames)
        print(f"      {len(scene_frames)} scene + {len(interval_frames)} interval = "
              f"{len(frames)} candidate frames")

        print(f"[2/5] Deduping consecutive near-identical frames (hash distance <= {args.hash_distance})...")
        frames = dedupe_frames(frames, args.hash_distance)
        print(f"      {len(frames)} unique frames")

        print(f"[3/5] Parsing SRT: {args.srt.name}")
        segments = parse_srt(args.srt)
        print(f"      {len(segments)} transcript segments")

        print("[4/5] Aligning transcript to frames...")
        slides_data = assign_transcript_to_frames(frames, segments, duration)

        print(f"[5/5] Building deck -> {args.output}")
        build_deck(slides_data, args.output, args.video.stem)

        if args.keep_frames:
            dest = Path.cwd() / f"frames_{args.video.stem}"
            dest.mkdir(parents=True, exist_ok=True)
            for d in (scene_dir, interval_dir):
                if d.exists():
                    shutil.copytree(d, dest, dirs_exist_ok=True)
            print(f"      frames saved to {dest}")

        print(f"\nDone. {len(slides_data)} content slides + 1 title slide written to {args.output}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
