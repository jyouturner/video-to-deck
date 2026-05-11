# tests

A small kit of regression checks. Not eval — eval would need a corpus of
examples; here we have one curated case per video, used to verify the
pipeline still picks the slides a human reviewer thought it should.

## `compare_slides.py`

Compare a pipeline-generated deck against a hand-curated reference deck.

The reference fixture under `fixtures/<video_id>.pptx` is the maintainer's
ground truth for which slides the deck should contain for that video. The
script computes perceptual-hash similarity between each reference slide
and the candidate's slides, and reports recall + precision + a per-slide
match table.

### Workflow

1. **Regenerate the deck** for the fixture's video. From the web reader:
   open `http://localhost:7682/digests/<video_id>/`, delete the existing
   `slides.pptx` (or use the Generate slides button if it doesn't exist),
   wait for the async job to finish.
2. **Run the comparison**:
   ```bash
   uv run python tests/compare_slides.py \
       tests/fixtures/igO8iyca2_g.pptx \
       ~/yt2md/digests/igO8iyca2_g/slides.pptx
   ```
3. **Read the output.** Each reference slide gets one line in the table:
   the nearest candidate slide, the pHash distance, and a ✓/✗ flag. The
   script exits 0 if recall ≥ 90% (configurable with `--min-recall`).

### Current fixtures

- `fixtures/igO8iyca2_g.pptx` — Anthropic talk "Running an AI-native
  engineering org" (28 min). 32 slides total (1 title + 31 content).
  This is the case used to investigate the `global_phash_cluster`
  false-positive that was dropping template-similar slides; it's the
  primary regression check for the slide-extraction pipeline.

### When to run

After changing anything in:

- `extract_scene_frames` / `extract_interval_frames` / `extract_scene_and_interval_frames`
- `dedupe_frames` / `global_phash_cluster`
- `_render_classification_grids` / `classify_slides_via_grids`
- `assign_transcript_to_frames` / `build_deck`

These are the slide-pipeline functions. Anything else (digest, panel,
takeaway) doesn't affect deck output.

### What the test does NOT cover

- **LLM noise** — the vision classifier is non-deterministic. A failing
  run doesn't necessarily mean a regression; re-run once to confirm.
- **New videos** — this only verifies behavior on the one curated video.
  For a true eval you'd need a few diverse fixtures (deck-heavy talk,
  talking-head interview, mostly-visual demo, etc.).
- **Quality of synthesis** (digest / panel / takeaway). Those are LLM
  outputs and don't have ground truth.
