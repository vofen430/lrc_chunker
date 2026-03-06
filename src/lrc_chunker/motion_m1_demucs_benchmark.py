#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reconstructed video module for M1 mix-vs-demucs preview rendering.

This file restores the parts that were clearly established during the
project session:
  - single-line / single-chunk subtitle preview
  - active-word highlight
  - mix vs demucs parameter timeline
  - current-chunk parameter bars
  - pulse orbs driven by beat + arousal proxy
  - local timer on single preview
  - optional comparison note banner
  - ffmpeg audio mux
  - 2x2 comparison video composition with global timer

It intentionally does not try to recreate unknown styling details exactly.
Those are documented separately in docs/VIDEO_MODULE_RESTORE_REQUIREMENTS.md.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class M0Chunk:
    chunk_id: int
    start: float
    end: float
    text: str
    words: List[Dict[str, object]]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _nearest_distance(sorted_times: Sequence[float], t: float) -> float:
    arr = np.asarray(sorted_times, dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    i = int(np.searchsorted(arr, t))
    best = float("inf")
    if 0 <= i < arr.size:
        best = min(best, abs(float(arr[i]) - t))
    if i - 1 >= 0:
        best = min(best, abs(float(arr[i - 1]) - t))
    return best


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_m0(path: Path) -> List[M0Chunk]:
    data = _load_json(path)
    chunks: List[M0Chunk] = []
    for i, raw in enumerate(data.get("chunks", [])):
        chunks.append(
            M0Chunk(
                chunk_id=_safe_int(raw.get("chunk_id"), i),
                start=_safe_float(raw.get("start"), 0.0),
                end=_safe_float(raw.get("end"), _safe_float(raw.get("start"), 0.0)),
                text=str(raw.get("text") or "").strip(),
                words=list(raw.get("words") or []),
            )
        )
    chunks.sort(key=lambda x: (x.chunk_id, x.start, x.end))
    return chunks


def _chunk_rows_by_id(path: Path) -> Dict[int, Dict[str, object]]:
    data = _load_json(path)
    out: Dict[int, Dict[str, object]] = {}
    for i, row in enumerate(data.get("chunks", [])):
        out[_safe_int(row.get("chunk_id"), i)] = row
    return out


def _format_timecode(t: float) -> str:
    t = max(0.0, float(t))
    mm = int(t // 60.0)
    ss = int(t % 60.0)
    ms = int(round((t - int(t)) * 1000.0))
    if ms >= 1000:
        ss += 1
        ms = 0
    if ss >= 60:
        mm += 1
        ss -= 60
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def _find_active_chunk_idx(chunks: List[M0Chunk], t: float, start_idx: int) -> int:
    n = len(chunks)
    i = max(0, min(start_idx, max(0, n - 1)))
    while i + 1 < n and t >= chunks[i].end - EPS:
        i += 1
    while i > 0 and t < chunks[i].start - EPS:
        i -= 1
    return i


def _active_word_index(words: List[Dict[str, object]], t: float) -> int:
    for i, w in enumerate(words):
        ws = _safe_float(w.get("start"), 0.0)
        we = _safe_float(w.get("end"), ws)
        if ws - EPS <= t < we - EPS:
            return i
    return -1


def _subtitle_tokens(chunk: M0Chunk) -> List[str]:
    toks = [str(w.get("text") or "").strip() for w in chunk.words]
    toks = [t for t in toks if t]
    if toks:
        return toks
    return [t for t in chunk.text.split() if t]


def _draw_word_highlight_row(ax, chunk: M0Chunk, active_word_idx: int, y: float = 0.52) -> List[object]:
    tokens = _subtitle_tokens(chunk)
    if not tokens:
        return []

    char_unit = 0.0175
    space_unit = 0.013
    widths = [max(0.05, min(0.24, len(t) * char_unit)) for t in tokens]
    total_w = sum(widths) + max(0, len(tokens) - 1) * space_unit
    x = max(0.02, 0.5 - total_w / 2.0)

    artists: List[object] = []
    for i, tok in enumerate(tokens):
        is_active = i == active_word_idx
        txt = ax.text(
            x,
            y,
            tok,
            fontsize=29 if is_active else 28,
            color="#FFD54F" if is_active else "#F5F7FA",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontweight="bold" if is_active else "normal",
        )
        artists.append(txt)
        x += widths[i] + space_unit
    return artists


def _style_axis_for_dark(ax) -> None:
    ax.tick_params(colors="#E8EEF4", labelcolor="#E8EEF4")
    ax.xaxis.label.set_color("#E8EEF4")
    ax.yaxis.label.set_color("#E8EEF4")
    ax.title.set_color("#F5F7FA")
    for spine in ax.spines.values():
        spine.set_color("#607080")


def _style_legend_for_dark(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("#111827")
    frame.set_edgecolor("#7B8A9A")
    frame.set_alpha(0.92)
    for txt in legend.get_texts():
        txt.set_color("#F5F7FA")


def _row_val(row: Dict[int, Dict[str, object]], chunk_id: int, key: str) -> float:
    return _safe_float(row.get(chunk_id, {}).get(key), 0.0)


def _arousal_proxy(chunk_row: Dict[str, object]) -> float:
    return max(
        0.0,
        min(
            1.0,
            0.6 * _safe_float(chunk_row.get("onset_density_norm"), 0.0)
            + 0.4 * _safe_float(chunk_row.get("loudness_lufs_norm"), 0.0),
        ),
    )


def _extract_onset_and_beats(audio_path: Path, sr: int, hop_length: int) -> Tuple[List[float], List[float]]:
    import librosa  # type: ignore

    y, use_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=use_sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=use_sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=use_sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=use_sr,
        hop_length=hop_length,
        units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=use_sr, hop_length=hop_length)
    return [float(x) for x in onset_times.tolist()], [float(x) for x in beat_times.tolist()]


def render_single_preview(
    *,
    m0_path: Path,
    m1_mix_path: Path,
    m1_demucs_path: Path,
    audio_mix_path: Path,
    audio_vocals_path: Path,
    out_video_path: Path,
    fps: int = 8,
    max_seconds: float = 60.0,
    comparison_note: str = "",
) -> None:
    import imageio.v2 as imageio  # type: ignore
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import patches  # type: ignore
    from matplotlib.cm import get_cmap  # type: ignore

    chunks = _load_m0(m0_path)
    mix_rows = _chunk_rows_by_id(m1_mix_path)
    dmx_rows = _chunk_rows_by_id(m1_demucs_path)
    _, beat_mix = _extract_onset_and_beats(audio_mix_path, sr=22050, hop_length=512)
    _, beat_dmx = _extract_onset_and_beats(audio_vocals_path, sr=22050, hop_length=512)

    if not chunks:
        raise ValueError("No chunks available for preview.")

    duration = min(max_seconds, max(c.end for c in chunks)) if max_seconds > 0 else max(c.end for c in chunks)
    times = np.arange(0.0, duration, 1.0 / max(1, fps), dtype=np.float64)
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    chunk_mid = np.array([0.5 * (c.start + c.end) for c in chunks], dtype=np.float64)
    mix_boundary = np.array(
        [
            0.5
            * (_row_val(mix_rows, c.chunk_id, "beat_proximity") + _row_val(mix_rows, c.chunk_id, "onset_proximity"))
            for c in chunks
        ],
        dtype=np.float64,
    )
    dmx_boundary = np.array(
        [
            0.5
            * (_row_val(dmx_rows, c.chunk_id, "beat_proximity") + _row_val(dmx_rows, c.chunk_id, "onset_proximity"))
            for c in chunks
        ],
        dtype=np.float64,
    )
    mix_arousal = np.array([_arousal_proxy(mix_rows.get(c.chunk_id, {})) for c in chunks], dtype=np.float64)
    dmx_arousal = np.array([_arousal_proxy(dmx_rows.get(c.chunk_id, {})) for c in chunks], dtype=np.float64)

    summary_mix = float(np.mean(mix_boundary)) if len(mix_boundary) else 0.0
    summary_dmx = float(np.mean(dmx_boundary)) if len(dmx_boundary) else 0.0

    cmap = get_cmap("turbo")
    fig = plt.figure(figsize=(12, 6.8), dpi=160)
    fig.patch.set_facecolor("#0F1115")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.2, 1.0], width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.2)
    ax_sub = fig.add_subplot(gs[0, :])
    ax_line = fig.add_subplot(gs[1, :])
    ax_bar = fig.add_subplot(gs[2, 0])
    ax_orb = fig.add_subplot(gs[2, 1])

    ax_sub.set_facecolor("#171A21")
    ax_line.set_facecolor("#151821")
    ax_bar.set_facecolor("#151821")
    ax_orb.set_facecolor("#151821")
    ax_sub.set_axis_off()

    if comparison_note.strip():
        ax_sub.text(0.0, 0.995, comparison_note.strip(), fontsize=12, color="#F5F7FA", transform=ax_sub.transAxes, va="top")
    sub_title = ax_sub.text(0.0, 0.90, "", fontsize=13, color="#C9D1D9", transform=ax_sub.transAxes, va="top")
    sub_timer = ax_sub.text(
        0.995,
        0.90,
        "",
        fontsize=22,
        color="#F5F7FA",
        transform=ax_sub.transAxes,
        va="top",
        ha="right",
        fontweight="bold",
        bbox={"facecolor": "#0A0E14", "edgecolor": "#DDE6EF", "alpha": 0.78, "boxstyle": "round,pad=0.30"},
    )
    sub_chunk_text = ax_sub.text(0.0, 0.22, "", fontsize=13, color="#9FB3C8", transform=ax_sub.transAxes, va="top")
    sub_word = ax_sub.text(0.0, 0.08, "", fontsize=14, color="#F0C419", transform=ax_sub.transAxes, va="top")
    active_word_artists: List[object] = []

    ax_line.set_title("M1 Parameter Timeline (Mix vs Demucs Vocals)", color="#F5F7FA")
    ax_line.set_xlim(0.0, duration)
    ax_line.set_ylim(0.0, 1.05)
    ax_line.set_xlabel("Time (s)", color="#E8EEF4")
    ax_line.set_ylabel("Normalized value", color="#E8EEF4")
    ax_line.grid(alpha=0.25, linestyle="--", color="#6F7F8F")
    ax_line.plot(chunk_mid, mix_boundary, color="#4FC3F7", linewidth=1.8, label="Mix boundary score")
    ax_line.plot(chunk_mid, dmx_boundary, color="#FFB74D", linewidth=1.8, label="Demucs boundary score")
    ax_line.plot(chunk_mid, mix_arousal, color="#4FC3F7", linewidth=1.2, linestyle=":", label="Mix arousal proxy")
    ax_line.plot(chunk_mid, dmx_arousal, color="#FFB74D", linewidth=1.2, linestyle=":", label="Demucs arousal proxy")
    for bt in beat_mix:
        if 0 <= bt <= duration:
            ax_line.axvline(bt, color="#90A4AE", alpha=0.06, linewidth=0.6)
    playhead = ax_line.axvline(0.0, color="#EF5350", linewidth=2.0, alpha=0.9)
    _style_axis_for_dark(ax_line)
    _style_legend_for_dark(ax_line.legend(loc="upper right", fontsize=8))

    cats = ["beat_prox", "onset_prox", "boundary", "arousal_proxy"]
    x = np.arange(len(cats))
    width = 0.35
    bars_mix = ax_bar.bar(x - width / 2, np.zeros(len(cats)), width, color="#4FC3F7", label="Mix")
    bars_dmx = ax_bar.bar(x + width / 2, np.zeros(len(cats)), width, color="#FFB74D", label="Demucs")
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_xticks(x, cats, rotation=15)
    ax_bar.set_title("Current Chunk Parameters", color="#F5F7FA")
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--", color="#6F7F8F")
    _style_axis_for_dark(ax_bar)
    _style_legend_for_dark(ax_bar.legend(loc="upper right", fontsize=8))

    ax_orb.set_xlim(0.0, 1.0)
    ax_orb.set_ylim(0.0, 1.0)
    ax_orb.set_aspect("equal")
    ax_orb.set_axis_off()
    ax_orb.set_title("Pulse Orbs (Beat) + Color (Emotion Proxy)", color="#F5F7FA")
    mix_orb = patches.Circle((0.30, 0.56), radius=0.16, color=cmap(0.2), alpha=0.95)
    dmx_orb = patches.Circle((0.72, 0.56), radius=0.16, color=cmap(0.2), alpha=0.95)
    ax_orb.add_patch(mix_orb)
    ax_orb.add_patch(dmx_orb)
    mix_label = ax_orb.text(0.30, 0.20, "", ha="center", va="center", fontsize=10, color="#E0F7FA")
    dmx_label = ax_orb.text(0.72, 0.20, "", ha="center", va="center", fontsize=10, color="#FFF3E0")
    ax_orb.text(
        0.5,
        0.93,
        f"Boundary Mean  Mix={summary_mix:.3f}  Demucs={summary_dmx:.3f}  Delta={summary_dmx - summary_mix:+.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color="#ECEFF1",
    )

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video_path), fps=fps, codec="libx264", quality=8)

    idx = 0
    n_frames = len(times)
    for fi, t in enumerate(times):
        idx = _find_active_chunk_idx(chunks, float(t), idx)
        c = chunks[idx]
        active_w = _active_word_index(c.words, float(t))
        row_mix = mix_rows.get(c.chunk_id, {})
        row_dmx = dmx_rows.get(c.chunk_id, {})

        for artist in active_word_artists:
            try:
                artist.remove()
            except Exception:
                pass
        active_word_artists = _draw_word_highlight_row(ax_sub, c, active_w, y=0.54)

        sub_title.set_text(f"t={t:6.2f}s   chunk={c.chunk_id} [{c.start:.2f},{c.end:.2f}]   single-line/single-chunk preview")
        sub_timer.set_text(f"TIME {_format_timecode(float(t))}")
        sub_chunk_text.set_text("chunk: " + textwrap.shorten(c.text, width=120, placeholder="..."))
        if 0 <= active_w < len(c.words):
            w = c.words[active_w]
            sub_word.set_text(
                f"active word: {str(w.get('text') or '').strip()}   [{_safe_float(w.get('start'), 0.0):.2f}, {_safe_float(w.get('end'), 0.0):.2f}]"
            )
        else:
            sub_word.set_text("active word:")

        playhead.set_xdata([t, t])
        mix_vals = [
            _safe_float(row_mix.get("beat_proximity"), 0.0),
            _safe_float(row_mix.get("onset_proximity"), 0.0),
            0.5 * (_safe_float(row_mix.get("beat_proximity"), 0.0) + _safe_float(row_mix.get("onset_proximity"), 0.0)),
            _arousal_proxy(row_mix),
        ]
        dmx_vals = [
            _safe_float(row_dmx.get("beat_proximity"), 0.0),
            _safe_float(row_dmx.get("onset_proximity"), 0.0),
            0.5 * (_safe_float(row_dmx.get("beat_proximity"), 0.0) + _safe_float(row_dmx.get("onset_proximity"), 0.0)),
            _arousal_proxy(row_dmx),
        ]
        for b, v in zip(bars_mix, mix_vals):
            b.set_height(max(0.0, min(1.0, v)))
        for b, v in zip(bars_dmx, dmx_vals):
            b.set_height(max(0.0, min(1.0, v)))

        d_mix = _nearest_distance(beat_mix, float(t))
        d_dmx = _nearest_distance(beat_dmx, float(t))
        pulse_mix = max(0.0, 1.0 - (d_mix / 0.15)) if math.isfinite(d_mix) else 0.0
        pulse_dmx = max(0.0, 1.0 - (d_dmx / 0.15)) if math.isfinite(d_dmx) else 0.0
        mix_orb.set_radius(0.13 + 0.09 * pulse_mix)
        dmx_orb.set_radius(0.13 + 0.09 * pulse_dmx)
        mix_orb.set_color(cmap(max(0.0, min(1.0, mix_vals[-1]))))
        dmx_orb.set_color(cmap(max(0.0, min(1.0, dmx_vals[-1]))))
        mix_label.set_text(f"Mix\nB:{mix_vals[2]:.2f}\nA:{mix_vals[-1]:.2f}")
        dmx_label.set_text(f"Demucs\nB:{dmx_vals[2]:.2f}\nA:{dmx_vals[-1]:.2f}")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        writer.append_data(np.ascontiguousarray(rgba[:, :, :3]))

        if fi % 50 == 0 or fi == n_frames - 1:
            print(f"[video] frame {fi + 1}/{n_frames}  t={t:.2f}s")

    writer.close()
    plt.close(fig)


def mux_audio_with_ffmpeg(video_path: Path, audio_path: Path, out_path: Path, duration_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{max(0.1, float(duration_sec)):.3f}",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def compose_quad_video(
    *,
    inputs: List[Path],
    labels: List[str],
    out_raw_path: Path,
    out_mux_path: Optional[Path] = None,
    audio_source_for_mux: Optional[Path] = None,
    note: str = "",
    fps: int = 8,
    max_seconds: float = 60.0,
) -> None:
    import imageio.v2 as imageio  # type: ignore
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    if len(inputs) != 4 or len(labels) != 4:
        raise ValueError("compose_quad_video requires exactly 4 inputs and 4 labels.")
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    readers = [imageio.get_reader(str(p)) for p in inputs]
    meta = readers[0].get_meta_data()
    fps = int(round(meta.get("fps", fps)))

    w_cell, h_cell = 960, 544
    banner_h = 92
    w_out, h_out = 1920, 1184

    try:
        font_main = ImageFont.truetype("arial.ttf", 31)
        font_label = ImageFont.truetype("arial.ttf", 27)
        font_timer = ImageFont.truetype("consola.ttf", 38)
    except Exception:
        font_main = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_timer = ImageFont.load_default()

    frame_count = min([r.count_frames() for r in readers])
    max_frames = min(frame_count, int(fps * max_seconds))
    out_raw_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_raw_path), fps=fps, codec="libx264", quality=8, macro_block_size=16)

    for i in range(max_frames):
        frames = [r.get_data(i) for r in readers]
        cells = [np.asarray(Image.fromarray(f).resize((w_cell, h_cell), resample=Image.BILINEAR), dtype=np.uint8) for f in frames]

        canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        canvas[:banner_h, :, :] = np.array([10, 13, 20], dtype=np.uint8)
        y0 = banner_h
        canvas[y0 : y0 + h_cell, 0:w_cell] = cells[0]
        canvas[y0 : y0 + h_cell, w_cell:w_out] = cells[1]
        canvas[y0 + h_cell : y0 + 2 * h_cell, 0:w_cell] = cells[2]
        canvas[y0 + h_cell : y0 + 2 * h_cell, w_cell:w_out] = cells[3]

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        if note.strip():
            draw.text((22, 16), note.strip(), fill=(245, 247, 250), font=font_main)
        t = i / max(1, fps)
        draw.text((1260, 44), f"GLOBAL TIME  {_format_timecode(t)}", fill=(245, 247, 250), font=font_timer)
        draw.text((26, banner_h + 18), labels[0], fill=(245, 247, 250), font=font_label)
        draw.text((w_cell + 26, banner_h + 18), labels[1], fill=(245, 247, 250), font=font_label)
        draw.text((26, banner_h + h_cell + 18), labels[2], fill=(245, 247, 250), font=font_label)
        draw.text((w_cell + 26, banner_h + h_cell + 18), labels[3], fill=(245, 247, 250), font=font_label)

        writer.append_data(np.asarray(pil, dtype=np.uint8))
        if i % 50 == 0 or i == max_frames - 1:
            print(f"[quad] frame {i + 1}/{max_frames} t={t:.2f}s")

    writer.close()
    for r in readers:
        r.close()

    if out_mux_path and audio_source_for_mux:
        mux_audio_with_ffmpeg(out_raw_path, audio_source_for_mux, out_mux_path, duration_sec=max_seconds)


def _single_parser(subparsers) -> None:
    p = subparsers.add_parser("single", help="render one single-preview parameter video")
    p.add_argument("m0", type=str)
    p.add_argument("m1_mix", type=str)
    p.add_argument("m1_demucs", type=str)
    p.add_argument("audio_mix", type=str)
    p.add_argument("audio_vocals", type=str)
    p.add_argument("-o", "--output", type=str, required=True)
    p.add_argument("--mux-output", type=str, default="")
    p.add_argument("--comparison-note", type=str, default="")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=60.0)


def _quad_parser(subparsers) -> None:
    p = subparsers.add_parser("quad", help="compose four single-preview videos into one 2x2 comparison video")
    p.add_argument("--in1", type=str, required=True)
    p.add_argument("--in2", type=str, required=True)
    p.add_argument("--in3", type=str, required=True)
    p.add_argument("--in4", type=str, required=True)
    p.add_argument("--label1", type=str, required=True)
    p.add_argument("--label2", type=str, required=True)
    p.add_argument("--label3", type=str, required=True)
    p.add_argument("--label4", type=str, required=True)
    p.add_argument("-o", "--output", type=str, required=True)
    p.add_argument("--mux-output", type=str, default="")
    p.add_argument("--audio-source", type=str, default="")
    p.add_argument("--note", type=str, default="")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=60.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconstructed video module for mix-vs-demucs preview rendering.")
    subparsers = p.add_subparsers(dest="cmd", required=True)
    _single_parser(subparsers)
    _quad_parser(subparsers)
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.cmd == "single":
        out_path = Path(args.output).expanduser().resolve()
        render_single_preview(
            m0_path=Path(args.m0).expanduser().resolve(),
            m1_mix_path=Path(args.m1_mix).expanduser().resolve(),
            m1_demucs_path=Path(args.m1_demucs).expanduser().resolve(),
            audio_mix_path=Path(args.audio_mix).expanduser().resolve(),
            audio_vocals_path=Path(args.audio_vocals).expanduser().resolve(),
            out_video_path=out_path,
            fps=int(args.fps),
            max_seconds=float(args.max_seconds),
            comparison_note=args.comparison_note,
        )
        if args.mux_output:
            mux_audio_with_ffmpeg(
                out_path,
                Path(args.audio_mix).expanduser().resolve(),
                Path(args.mux_output).expanduser().resolve(),
                duration_sec=float(args.max_seconds),
            )
        return 0

    if args.cmd == "quad":
        compose_quad_video(
            inputs=[
                Path(args.in1).expanduser().resolve(),
                Path(args.in2).expanduser().resolve(),
                Path(args.in3).expanduser().resolve(),
                Path(args.in4).expanduser().resolve(),
            ],
            labels=[args.label1, args.label2, args.label3, args.label4],
            out_raw_path=Path(args.output).expanduser().resolve(),
            out_mux_path=Path(args.mux_output).expanduser().resolve() if args.mux_output else None,
            audio_source_for_mux=Path(args.audio_source).expanduser().resolve() if args.audio_source else None,
            note=args.note,
            fps=int(args.fps),
            max_seconds=float(args.max_seconds),
        )
        return 0

    raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

