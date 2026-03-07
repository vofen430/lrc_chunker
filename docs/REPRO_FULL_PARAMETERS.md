# Project Reproduction Parameters

## Scope

This document reconstructs the full reproducible parameter set for the current lyrics-alignment and motion-preview pipeline discussed in this project session.

Pipeline:

1. LRC preprocessing
2. Audio-text forced alignment
3. Word-timing refinement
4. M0 text/timing feature extraction
5. M1 audio feature benchmark and visualization
6. Multi-video comparison rendering

Primary objective:

- Input: song audio (`wav/mp3`) + LRC
- Output: chunk-level timing, word-level timing, motion-preview features, and comparison videos


## Libraries And Roles

### Alignment / Chunking

- Library: `stable-ts`
- Python import: `stable_whisper`
- Upstream model family: `OpenAI Whisper`
- Models used in this project:
  - `small.en`
  - `medium.en`
  - `large-v3`
  - `turbo`

Role:

- Forced alignment from known lyric text to song audio
- Produce segment and word timestamps
- Provide initial segmentation primitives used before custom chunk DP


### Denoising / Vocal Isolation

- Library: `demucs`
- Model priority used:
  - `htdemucs`
  - `mdx_extra_q`
  - `mdx_q`
  - `mdx_extra`
  - `hdemucs_mmi`

Role:

- Reduce accompaniment interference during alignment
- Provide vocal stem for comparison and refinement


### Voice Activity Detection

- Provider: `Silero VAD`
- Usage path: enabled internally through `stable-ts`

Role:

- Improve non-vocal boundary estimation
- Help `adjust_gaps()` and alignment boundary stability


### Audio Features

- Library: `librosa`

Role:

- Onset detection
- Beat tracking
- RMS/loudness-like proxy features
- Boundary and rhythm evaluation features


### Rendering / Video

- Libraries:
  - `matplotlib`
  - `imageio`
  - `ffmpeg`

Role:

- Render single-preview parameter videos
- Mux original audio
- Compose comparison videos


### Runtime

- `torch`
- Python environments discussed in the project:
  - `Python 3.10`
  - also tested historically on `Python 3.13`


## Stage 1: LRC Preprocessing

Implementation intent:

- Expand multi-timestamp LRC rows
- Remove metadata rows
- Group same-timestamp rows
- Keep primary language row
- Trim head/tail non-lyric rows

Core heuristics:

- Same-timestamp row selection: prefer higher ASCII ratio, then longer text
- Head/tail trimming: keep rows that look like real lyrics, remove credits/info rows

No exposed runtime CLI knobs were relied on here beyond the input files themselves.


## Stage 2: Alignment And Chunking

Script:

- `lyrics_chunker_baseline.py`

Primary libraries:

- `stable-ts`
- `demucs`

Default reproducible alignment / chunking parameters:

- `model`: one of `small.en`, `medium.en`, `large-v3`, `turbo`
- `language`: `en`
- `vad_threshold`: `0.35`
- `max_gap`: `0.35`
- `merge_gap`: `0.12`
- `max_chars`: `42`
- `max_words`: `6`
- `max_dur`: `3.2`
- `hard_max_chunk_dur`: `6.0`
- `rhythm_weight`: `2.8`
- `hard_line_breaks`: `true`
- `emphasize_long_words`: `true`
- `long_word_single_threshold`: `0.78`
- `long_word_bonus`: `2.6`
- `apply_clamp_max`: `true`
- `denoiser_requested`: `auto`
- `denoiser_effective`: typically `demucs`
- `demucs_model_priority`:
  - `htdemucs`
  - `mdx_extra_q`
  - `mdx_q`
  - `mdx_extra`
  - `hdemucs_mmi`

Chunk strategy:

- Whisper alignment result
- `clamp_max()`
- punctuation split
- gap split
- length split
- duration split
- merge by gap
- `adjust_gaps()`
- custom semantic-audio DP segmentation

Custom chunk-DP controls:

- semantic boundary preference
- audio gap preference
- rhythm preference
- hard line-break enforcement
- strict max-word enforcement

Typical command form:

```powershell
python lyrics_chunker_baseline.py "Memories - Conan Gray.wav" "Memories - Conan Gray.lrc" `
  --model small.en `
  --language en `
  --vad-threshold 0.35 `
  --max-gap 0.35 `
  --merge-gap 0.12 `
  --max-chars 42 `
  --max-words 6 `
  --max-dur 3.2 `
  --hard-max-chunk-dur 6.0 `
  --rhythm-weight 2.8 `
  --denoiser auto
```


## Stage 3: Word Timing Refinement

Script:

- `word_timing_refine.py`

Purpose:

- Do not change chunk text/sequence
- Fix early first-word starts
- Compress overlong function words
- Keep most words stable

Audio feature source:

- `librosa` onset detection on:
  - original mix
  - optional Demucs vocal stem

Built-in profiles:

- `mild`
- `balanced`
- `slow_attack`
- `aggressive`
- `rap_snap`

### Default `slow_attack` profile

- `start_shift_max`: `0.40`
- `start_back_max`: `0.03`
- `boundary_shift_max`: `0.12`
- `min_word_dur`: `0.10`
- `func_max_dur`: `0.52`
- `func_ratio_max`: `2.1`
- `keep_weight`: `0.35`
- `onset_weight`: `1.00`
- `prefer_future_penalty`: `1.55`
- `force_forward_min_gap`: `0.05`
- `use_lrc_anchors`: `true`
- `lrc_anchor_span_words`: `1`
- `lrc_anchor_max_ratio`: `0.15`

### `rap_snap` profile added in this project

- `start_shift_max`: `0.24`
- `start_back_max`: `0.00`
- `boundary_shift_max`: `0.20`
- `min_word_dur`: `0.07`
- `func_max_dur`: `0.34`
- `func_ratio_max`: `1.45`
- `keep_weight`: `0.40`
- `onset_weight`: `1.00`
- `prefer_future_penalty`: `1.10`
- `force_forward_min_gap`: `0.05`

### `small.en` rap-snap run actually used

- `profile`: `rap_snap`
- `start_shift_max`: `0.28`
- `start_back_max`: `0.00`
- `boundary_shift_max`: `0.22`
- `min_word_dur`: `0.06`
- `func_max_dur`: `0.30`
- `func_ratio_max`: `1.35`
- `keep_weight`: `0.35`
- `onset_weight`: `1.00`
- `prefer_future_penalty`: `1.20`
- `force_forward_min_gap`: `0.03`

Other default CLI parameters:

- `sr`: `22050`
- `hop_length`: `256`
- `early_thr`: `0.05`
- `func_long_thr`: `0.55`

Typical command form:

```powershell
python word_timing_refine.py "chunking.json" `
  --audio-mix "Memories - Conan Gray.wav" `
  --sr 22050 `
  --hop-length 256
```

Rap-snap command form:

```powershell
python word_timing_refine.py "chunking.json" `
  --audio-mix "Memories - Conan Gray.wav" `
  --audio-vocals "vocals.wav" `
  --profile rap_snap `
  --start-shift-max 0.28 `
  --start-back-max 0.0 `
  --boundary-shift-max 0.22 `
  --min-word-dur 0.06 `
  --func-max-dur 0.30 `
  --func-ratio-max 1.35 `
  --keep-weight 0.35 `
  --prefer-future-penalty 1.2 `
  --force-forward-min-gap 0.03
```


## Stage 4: M0 Feature Extraction

Script:

- `motion_m0_extract.py`

Purpose:

- Convert chunking JSON into normalized text/timing features for motion stages

Default parameters:

- `min_repaired_word_dur`: `0.04`

Output fields per chunk:

- `chunk_id`
- `start`
- `end`
- `text`
- `chunk_duration`
- `word_count`
- `word_count_fallback_text_tokenized`
- `words_per_second`
- `min_word_dur`
- `gap_to_next`
- `word_durations`
- `word_indices`
- word records with raw and effective durations

Validation checks:

- chunk duration positive
- effective word duration positive
- max visible chunks at any time `<= 1`
- low unmatched word ratio
- no negative chunk gaps

Typical command form:

```powershell
python motion_m0_extract.py "chunking_refined.json" -o "features_text_timing.json" --report "validation_report_m0.json"
```


## Stage 5: M1 Audio Feature Benchmark

Script:

- `motion_m1_demucs_benchmark.py`

Purpose:

- Run audio feature extraction on:
  - original mix
  - Demucs vocals
- Compare proxy timing quality
- Render parameter preview video

Default audio-analysis parameters:

- `sr`: `22050`
- `onset-hop-length`: `512`
- `proximity-window-sec`: `0.20`
- `lufs-window-sec`: `0.40`

Default proxy metric windows:

- `chunk-onset-window`: `0.16`
- `chunk-beat-window`: `0.12`
- `word-onset-window`: `0.12`
- `word-beat-window`: `0.10`

Default video render parameters:

- `video-fps`: `8`
- `video-max-seconds`: `90.0`

Typical benchmark command:

```powershell
python motion_m1_demucs_benchmark.py "features_text_timing.json" `
  --audio "Memories - Conan Gray.wav" `
  --vocals-path "vocals.wav" `
  --run-dir "artifacts/benchmarks/run_name" `
  --video-max-seconds 60 `
  --video-fps 8 `
  --mux-original-audio
```

Proxy metrics produced:

- `mean_boundary_score`
- `p90_boundary_score`
- `chunk_onset_hit_rate`
- `chunk_beat_hit_rate`
- `word_onset_hit_rate`
- `word_beat_hit_rate`
- `overall_proxy_accuracy`


## Stage 6: Video Rendering Variants Used In This Project

Single-preview variants added during the session:

- base parameter preview
- white-text parameter preview
- white-text + timer preview

Comparison video variants added during the session:

- four-model comparison
- white-note comparison
- white-note + global timer comparison

Visualization behavior now includes:

- one-line / one-chunk display
- active-word highlight in yellow
- chart labels in white
- comparison-note banner
- local timer in single-preview videos
- global timer in comparison videos


## Model Comparison Run Used

Run group:

- `small.en`
- `medium.en`
- `large-v3`
- `turbo`

Shared settings for that ablation:

- same audio
- same LRC
- same chunking hyperparameters
- same word-refine baseline profile unless otherwise specified
- same M1 benchmark windows
- same 60-second preview duration


## Reproduction Order

Recommended exact order:

1. Generate or prepare vocal stem with `demucs`
2. Run `lyrics_chunker_baseline.py`
3. Run `word_timing_refine.py`
4. Run `motion_m0_extract.py`
5. Run `motion_m1_demucs_benchmark.py`
6. Compose multi-video comparison if needed


## Canonical Inputs

Audio used in this project session:

- `Memories - Conan Gray.wav`

Lyrics used in this project session:

- `examples/Memories - Conan Gray.lrc`

Optional vocal stem used in comparison/refinement:

- Demucs-separated `vocals.wav`


## Output Families

Alignment outputs:

- `chunking_*.json`

Word-refined outputs:

- `*_wordref_*.json`
- `word_timing_refine_report_*.json`

M0 outputs:

- `features_text_timing_*.json`
- `validation_report_m0_*.json`

M1 outputs:

- `features_audio_fast_mix*.json`
- `features_audio_fast_demucs_vocals*.json`
- `validation_report_m1_mix*.json`
- `validation_report_m1_demucs_vocals*.json`
- `comparison_report_m1_demucs*.json`
- `m1_demucs_parameter_preview*.mp4`

Comparison outputs:

- `compare_4models*.mp4`
- timing summaries
- proxy summaries


## Notes On Accuracy Work

Current project focus areas identified during the session:

- first-word early-start correction
- short-word overextension reduction
- keep most word spans stable while allowing sparse corrections
- optional manual anchor enforcement with human timestamps as highest standard

If manual anchors are used, they should be interpreted as:

- human-annotated word start times
