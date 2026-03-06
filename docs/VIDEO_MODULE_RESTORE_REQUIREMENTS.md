# Video Module Restore Requirements

## Goal

Restore the original video-preview module from remembered behavior, while separating uncertain visual details from confirmed functional behavior.


## Confirmed Functional Behavior

These capabilities were explicitly implemented in the earlier project session and should be treated as restored requirements:

1. Render a single-preview video that shows only one chunk at a time.
2. Highlight the active word inside the current chunk.
3. Show a subtitle information area with:
   - current absolute playback time
   - current chunk id
   - current chunk time span
   - current chunk text
   - current active word text and its word span
4. Render a line chart for mix-vs-demucs parameter comparison.
5. Render a bar chart for current-chunk parameter values.
6. Render orb-style beat/emotion indicators.
7. Support white-text rendering for dark backgrounds.
8. Support a timer display on single-preview videos.
9. Support muxing original music back into the rendered video.
10. Support 2x2 comparison composition for four model outputs.
11. Support a global timer and a white comparison-note banner on the 2x2 comparison video.


## Confirmed Data Inputs

The restored module should accept precomputed JSON artifacts rather than recompute alignment:

1. M0 feature JSON containing:
   - chunk timing
   - chunk text
   - word timing list per chunk
2. M1 mix feature JSON
3. M1 demucs-vocals feature JSON
4. Original mix audio path
5. Vocal stem audio path


## Confirmed Visual Semantics

These meanings were stable in the earlier implementation:

1. Yellow text is active-word emphasis, not chunk label.
2. White chunk text is the currently visible subtitle chunk.
3. The line chart compares:
   - mix boundary score
   - demucs boundary score
   - mix arousal proxy
   - demucs arousal proxy
4. The bar chart compares the current chunk on:
   - beat proximity
   - onset proximity
   - boundary score
   - arousal proxy
5. The orb panel uses:
   - pulse size for beat-near timing
   - color for arousal proxy


## Known Uncertain Details

The following details were remembered only at a functional level, not pixel-perfect:

1. Exact font family choices.
2. Exact title wording in every panel.
3. Exact line thickness, alpha, and spacing values.
4. Exact orb radius scaling constants.
5. Exact banner height used in each comparison variant.
6. Exact color hex values for some secondary text.
7. Whether every historical variant used local timer only, global timer only, or both.


## Requirement For Uncertain Details

For the items above, restoration should prioritize:

1. Preserving the original information architecture.
2. Preserving the semantic meaning of each visual region.
3. Keeping the output legible on pause-frame inspection.
4. Avoiding fake historical precision when exact constants are unknown.


## Expected CLI Surface

The restored module should provide at least two operations:

1. `single`
   - Render one preview video from M0 + M1 mix + M1 demucs + audio.
2. `quad`
   - Combine four rendered videos into one comparison video.


## Validation Expectations

After restoration, the module should be considered acceptable if:

1. It can render a 60-second preview from existing artifacts.
2. The active-word highlight updates according to word timestamps.
3. The timer can be used to pause and inspect subjective word-boundary perception.
4. The 2x2 comparison video can be generated with audio muxed in.

