from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .motion_m1_demucs_benchmark import compose_quad_video_with_audio, render_single_preview_with_audio
from .utils import artifact_name_prefix, clamp, find_payload_vocals_path, mean, percentile, read_json, safe_stem, write_json


DEFAULTS = {
    "sr": 22050,
    "onset_hop_length": 512,
    "proximity_window_sec": 0.20,
    "lufs_window_sec": 0.40,
    "chunk_onset_window": 0.16,
    "chunk_beat_window": 0.12,
    "word_onset_window": 0.12,
    "word_beat_window": 0.10,
    "video_fps": 8,
    "video_max_seconds": 60.0,
}


def _nearest_distance(sorted_times: Sequence[float], t: float) -> float:
    if not sorted_times:
        return float("inf")
    arr = np.asarray(sorted_times, dtype=np.float64)
    i = int(np.searchsorted(arr, t))
    best = float("inf")
    if 0 <= i < arr.size:
        best = min(best, abs(float(arr[i]) - t))
    if i - 1 >= 0:
        best = min(best, abs(float(arr[i - 1]) - t))
    return best


def _require_librosa():
    try:
        import librosa  # type: ignore
    except Exception as exc:
        raise RuntimeError("M1 benchmark requires librosa; install the pinned conda Python 3.8 environment first.") from exc
    return librosa


def _load_audio_features(audio_path: str, sr: int, hop_length: int) -> Dict[str, object]:
    librosa = _require_librosa()
    y, use_sr = librosa.load(audio_path, sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=use_sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=use_sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
    )
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=use_sr,
        hop_length=hop_length,
        units="frames",
    )
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=use_sr, hop_length=hop_length)
    return {
        "y": y,
        "sr": use_sr,
        "onset_times": [float(x) for x in librosa.frames_to_time(onset_frames, sr=use_sr, hop_length=hop_length)],
        "beat_times": [float(x) for x in librosa.frames_to_time(beat_frames, sr=use_sr, hop_length=hop_length)],
        "rms": np.asarray(rms, dtype=np.float64),
        "frame_times": np.asarray(frame_times, dtype=np.float64),
    }


def _slice_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= float(start)) & (times < float(end))


def _hit_rate(timestamps: Sequence[float], anchors: Sequence[float], window: float) -> float:
    if not timestamps:
        return 0.0
    hits = sum(1 for t in timestamps if _nearest_distance(anchors, float(t)) <= window)
    return hits / max(1, len(timestamps))


def _extract_rows(m0_payload: dict, audio_path: str, source_name: str, cfg: Dict[str, float]) -> Tuple[dict, dict]:
    audio = _load_audio_features(audio_path, sr=int(cfg["sr"]), hop_length=int(cfg["onset_hop_length"]))
    chunks = m0_payload.get("chunks", []) or []
    raw_rows: List[dict] = []
    loudness_raw: List[float] = []
    density_raw: List[float] = []
    boundary_scores: List[float] = []
    chunk_onset_hits: List[float] = []
    chunk_beat_hits: List[float] = []
    word_onset_hits: List[float] = []
    word_beat_hits: List[float] = []

    onset_times = audio["onset_times"]
    beat_times = audio["beat_times"]
    frame_times = audio["frame_times"]
    rms = audio["rms"]

    for idx, chunk in enumerate(chunks):
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        duration = max(0.001, end - start)
        word_starts = [float(word.get("start", 0.0)) for word in chunk.get("words", []) or []]
        d_onset = _nearest_distance(onset_times, start)
        d_beat = _nearest_distance(beat_times, start)
        onset_prox = max(0.0, 1.0 - d_onset / cfg["chunk_onset_window"])
        beat_prox = max(0.0, 1.0 - d_beat / cfg["chunk_beat_window"])
        boundary = 0.5 * (onset_prox + beat_prox)
        mask = _slice_mask(frame_times, start, end)
        rms_mean = float(np.mean(rms[mask])) if np.any(mask) else 0.0
        local_onsets = [t for t in onset_times if start <= t < end]
        onset_density = len(local_onsets) / duration
        word_onset = _hit_rate(word_starts, onset_times, cfg["word_onset_window"])
        word_beat = _hit_rate(word_starts, beat_times, cfg["word_beat_window"])

        row = {
            "chunk_id": int(chunk.get("chunk_id", idx)),
            "start": start,
            "end": end,
            "text": str(chunk.get("text") or "").strip(),
            "beat_proximity": beat_prox,
            "onset_proximity": onset_prox,
            "boundary_score": boundary,
            "word_onset_hit_rate": word_onset,
            "word_beat_hit_rate": word_beat,
            "chunk_onset_hit": 1.0 if d_onset <= cfg["chunk_onset_window"] else 0.0,
            "chunk_beat_hit": 1.0 if d_beat <= cfg["chunk_beat_window"] else 0.0,
            "loudness_raw": rms_mean,
            "onset_density_raw": onset_density,
        }
        raw_rows.append(row)
        loudness_raw.append(rms_mean)
        density_raw.append(onset_density)
        boundary_scores.append(boundary)
        chunk_onset_hits.append(row["chunk_onset_hit"])
        chunk_beat_hits.append(row["chunk_beat_hit"])
        word_onset_hits.append(word_onset)
        word_beat_hits.append(word_beat)

    loud_lo = percentile(loudness_raw, 0.05)
    loud_hi = percentile(loudness_raw, 0.95)
    density_hi = max(1e-6, percentile(density_raw, 0.90))
    for row in raw_rows:
        if loud_hi > loud_lo:
            loud_norm = (row["loudness_raw"] - loud_lo) / (loud_hi - loud_lo)
        else:
            loud_norm = 0.0
        row["loudness_lufs_norm"] = clamp(loud_norm, 0.0, 1.0)
        row["onset_density_norm"] = clamp(row["onset_density_raw"] / density_hi, 0.0, 1.0)

    overall_proxy_accuracy = mean(
        [
            mean(boundary_scores),
            mean(chunk_onset_hits),
            mean(chunk_beat_hits),
            mean(word_onset_hits),
            mean(word_beat_hits),
        ]
    )

    feature_payload = {
        "meta": {
            **m0_payload.get("meta", {}),
            "stage": "m1",
            "source": source_name,
            "audio_path": audio_path,
            "params": cfg,
        },
        "chunks": raw_rows,
    }
    report = {
        "source": source_name,
        "audio_path": audio_path,
        "chunk_count": len(raw_rows),
        "mean_boundary_score": mean(boundary_scores),
        "p90_boundary_score": percentile(boundary_scores, 0.90),
        "chunk_onset_hit_rate": mean(chunk_onset_hits),
        "chunk_beat_hit_rate": mean(chunk_beat_hits),
        "word_onset_hit_rate": mean(word_onset_hits),
        "word_beat_hit_rate": mean(word_beat_hits),
        "overall_proxy_accuracy": overall_proxy_accuracy,
    }
    return feature_payload, report


def run_benchmark(args) -> int:
    m0_path = Path(args.m0).expanduser().resolve()
    m0_payload = read_json(m0_path)
    vocals_path = args.vocals_path or find_payload_vocals_path(m0_payload)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = artifact_name_prefix(run_dir=str(run_dir), audio_path=str(Path(args.audio).expanduser().resolve()), fallback=str(m0_path))

    cfg = {
        "sr": float(args.sr),
        "onset_hop_length": float(args.onset_hop_length),
        "proximity_window_sec": float(args.proximity_window_sec),
        "lufs_window_sec": float(args.lufs_window_sec),
        "chunk_onset_window": float(args.chunk_onset_window),
        "chunk_beat_window": float(args.chunk_beat_window),
        "word_onset_window": float(args.word_onset_window),
        "word_beat_window": float(args.word_beat_window),
        "video_fps": float(args.video_fps),
        "video_max_seconds": float(args.video_max_seconds),
    }

    mix_payload, mix_report = _extract_rows(m0_payload, args.audio, "mix", cfg)
    mix_json = run_dir / f"{prefix}_features_audio_fast_mix.json"
    mix_report_json = run_dir / f"{prefix}_validation_report_m1_mix.json"
    write_json(mix_json, mix_payload)
    write_json(mix_report_json, mix_report)
    print(f"[m1] wrote {mix_json}")
    print(f"[m1] wrote {mix_report_json}")

    vocals_json = None
    vocals_report_json = None
    comparison_json = None
    mux_video = None
    if vocals_path:
        vocals_payload, vocals_report = _extract_rows(m0_payload, vocals_path, "demucs_vocals", cfg)
        vocals_json = run_dir / f"{prefix}_features_audio_fast_demucs_vocals.json"
        vocals_report_json = run_dir / f"{prefix}_validation_report_m1_demucs_vocals.json"
        comparison_json = run_dir / f"{prefix}_comparison_report_m1_demucs.json"
        write_json(vocals_json, vocals_payload)
        write_json(vocals_report_json, vocals_report)
        write_json(
            comparison_json,
            {
                "mix": mix_report,
                "demucs_vocals": vocals_report,
                "delta_overall_proxy_accuracy": vocals_report["overall_proxy_accuracy"] - mix_report["overall_proxy_accuracy"],
                "delta_mean_boundary_score": vocals_report["mean_boundary_score"] - mix_report["mean_boundary_score"],
            },
        )
        print(f"[m1] wrote {vocals_json}")
        print(f"[m1] wrote {vocals_report_json}")
        print(f"[m1] wrote {comparison_json}")

        mux_video = run_dir / f"{prefix}_m1_demucs_parameter_preview.mp4"
        render_single_preview_with_audio(
            m0_path=m0_path,
            m1_mix_path=mix_json,
            m1_demucs_path=vocals_json,
            audio_mix_path=Path(args.audio).expanduser().resolve(),
            audio_vocals_path=Path(vocals_path).expanduser().resolve(),
            out_video_path=mux_video,
            fps=int(args.video_fps),
            max_seconds=float(args.video_max_seconds),
            comparison_note=args.comparison_note,
        )
        print(f"[m1] wrote {mux_video}")

    return 0


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


def _benchmark_parser(subparsers) -> None:
    p = subparsers.add_parser("benchmark", help="extract M1 audio features and optionally render the preview video")
    p.add_argument("m0", type=str)
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--vocals-path", type=str, default="")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--sr", type=int, default=int(DEFAULTS["sr"]))
    p.add_argument("--onset-hop-length", type=int, default=int(DEFAULTS["onset_hop_length"]))
    p.add_argument("--proximity-window-sec", type=float, default=float(DEFAULTS["proximity_window_sec"]))
    p.add_argument("--lufs-window-sec", type=float, default=float(DEFAULTS["lufs_window_sec"]))
    p.add_argument("--chunk-onset-window", type=float, default=float(DEFAULTS["chunk_onset_window"]))
    p.add_argument("--chunk-beat-window", type=float, default=float(DEFAULTS["chunk_beat_window"]))
    p.add_argument("--word-onset-window", type=float, default=float(DEFAULTS["word_onset_window"]))
    p.add_argument("--word-beat-window", type=float, default=float(DEFAULTS["word_beat_window"]))
    p.add_argument("--video-fps", type=int, default=int(DEFAULTS["video_fps"]))
    p.add_argument("--video-max-seconds", type=float, default=float(DEFAULTS["video_max_seconds"]))
    p.add_argument("--comparison-note", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 benchmark and preview pipeline.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _benchmark_parser(subparsers)
    _single_parser(subparsers)
    _quad_parser(subparsers)
    return parser


def _normalize_argv(argv: Sequence[str] | None) -> List[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return argv
    if argv[0] in {"-h", "--help"}:
        return argv
    if argv[0] not in {"benchmark", "single", "quad"}:
        return ["benchmark", *argv]
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(_normalize_argv(argv))

    if args.cmd == "benchmark":
        return run_benchmark(args)

    if args.cmd == "single":
        out_path = Path(args.output).expanduser().resolve()
        final_path = Path(args.mux_output).expanduser().resolve() if args.mux_output else out_path
        render_single_preview_with_audio(
            m0_path=Path(args.m0).expanduser().resolve(),
            m1_mix_path=Path(args.m1_mix).expanduser().resolve(),
            m1_demucs_path=Path(args.m1_demucs).expanduser().resolve(),
            audio_mix_path=Path(args.audio_mix).expanduser().resolve(),
            audio_vocals_path=Path(args.audio_vocals).expanduser().resolve(),
            out_video_path=final_path,
            fps=int(args.fps),
            max_seconds=float(args.max_seconds),
            comparison_note=args.comparison_note,
        )
        return 0

    if args.cmd == "quad":
        if not args.audio_source:
            raise ValueError("quad requires --audio-source so only muxed output is generated.")
        final_path = Path(args.mux_output).expanduser().resolve() if args.mux_output else Path(args.output).expanduser().resolve()
        compose_quad_video_with_audio(
            inputs=[
                Path(args.in1).expanduser().resolve(),
                Path(args.in2).expanduser().resolve(),
                Path(args.in3).expanduser().resolve(),
                Path(args.in4).expanduser().resolve(),
            ],
            labels=[args.label1, args.label2, args.label3, args.label4],
            out_video_path=final_path,
            audio_source_for_mux=Path(args.audio_source).expanduser().resolve(),
            note=args.note,
            fps=int(args.fps),
            max_seconds=float(args.max_seconds),
        )
        return 0

    raise ValueError(f"unknown command: {args.cmd}")


