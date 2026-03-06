from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

from .utils import FUNCTION_WORDS, median, read_json, safe_stem, tokenize, write_json


PROFILES: Dict[str, Dict[str, float]] = {
    "mild": {
        "start_shift_max": 0.08,
        "start_back_max": 0.02,
        "boundary_shift_max": 0.08,
        "min_word_dur": 0.10,
        "func_max_dur": 0.55,
        "func_ratio_max": 2.4,
        "keep_weight": 0.85,
        "onset_weight": 1.0,
        "prefer_future_penalty": 0.25,
        "force_forward_min_gap": 0.05,
    },
    "balanced": {
        "start_shift_max": 0.14,
        "start_back_max": 0.03,
        "boundary_shift_max": 0.12,
        "min_word_dur": 0.10,
        "func_max_dur": 0.52,
        "func_ratio_max": 2.1,
        "keep_weight": 0.70,
        "onset_weight": 1.00,
        "prefer_future_penalty": 0.45,
        "force_forward_min_gap": 0.05,
    },
    "aggressive": {
        "start_shift_max": 0.18,
        "start_back_max": 0.03,
        "boundary_shift_max": 0.16,
        "min_word_dur": 0.08,
        "func_max_dur": 0.40,
        "func_ratio_max": 1.8,
        "keep_weight": 0.52,
        "onset_weight": 1.0,
        "prefer_future_penalty": 0.80,
        "force_forward_min_gap": 0.04,
    },
    "rap_snap": {
        "start_shift_max": 0.24,
        "start_back_max": 0.00,
        "boundary_shift_max": 0.20,
        "min_word_dur": 0.07,
        "func_max_dur": 0.34,
        "func_ratio_max": 1.45,
        "keep_weight": 0.40,
        "onset_weight": 1.00,
        "prefer_future_penalty": 1.10,
        "force_forward_min_gap": 0.05,
    },
}


def _iter_words(payload: dict) -> Iterable[dict]:
    for chunk in payload.get("chunks", []):
        for word in chunk.get("words", []) or []:
            yield word


def _load_onsets(*audio_paths: str, sr: int, hop_length: int) -> List[float]:
    valid_paths = [path for path in audio_paths if path]
    if not valid_paths:
        return []
    try:
        import librosa  # type: ignore
    except Exception:
        return []

    onset_times: List[float] = []
    for audio_path in valid_paths:
        y, use_sr = librosa.load(audio_path, sr=sr, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=use_sr, hop_length=hop_length)
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=use_sr,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
        )
        onset_times.extend(float(t) for t in librosa.frames_to_time(frames, sr=use_sr, hop_length=hop_length))
    return sorted(set(round(t, 3) for t in onset_times))


def _nearest_candidates(onsets: List[float], t: float, start_back_max: float, start_shift_max: float) -> List[float]:
    cands = [t]
    for onset in onsets:
        diff = onset - t
        if -start_back_max <= diff <= start_shift_max:
            cands.append(onset)
    return sorted(set(cands))


def _is_function_word(text: str) -> bool:
    toks = [tok.lower() for tok in tokenize(text) if any(ch.isalnum() for ch in tok)]
    return bool(toks) and all(tok in FUNCTION_WORDS for tok in toks)


def refine_payload(
    payload: dict,
    *,
    profile: str,
    audio_mix: str = "",
    audio_vocals: str = "",
    sr: int = 22050,
    hop_length: int = 256,
    early_thr: float = 0.05,
    func_long_thr: float = 0.55,
    overrides: Dict[str, float] | None = None,
) -> tuple[dict, dict]:
    params = dict(PROFILES[profile])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = float(value)

    refined = deepcopy(payload)
    onsets = _load_onsets(audio_mix, audio_vocals, sr=sr, hop_length=hop_length)
    changes = 0
    total_shift = 0.0
    durations: List[float] = [max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0))) for w in _iter_words(payload)]
    median_word_dur = max(params["min_word_dur"], median(durations) or params["min_word_dur"])

    for chunk in refined.get("chunks", []):
        prev_end = float(chunk.get("start", 0.0))
        words = chunk.get("words", []) or []
        for idx, word in enumerate(words):
            orig_start = float(word.get("start", 0.0))
            orig_end = float(word.get("end", orig_start))
            cands = _nearest_candidates(onsets, orig_start, params["start_back_max"], params["start_shift_max"])
            best_start = orig_start
            best_score = float("inf")
            for cand in cands:
                shift = cand - orig_start
                score = params["keep_weight"] * abs(shift)
                if shift > 0:
                    score -= params["prefer_future_penalty"] * shift
                score += params["onset_weight"] * abs(cand - orig_start)
                if score < best_score:
                    best_score = score
                    best_start = cand

            if abs(best_start - orig_start) < early_thr:
                best_start = orig_start
            best_start = max(prev_end, best_start)

            dur = max(params["min_word_dur"], orig_end - orig_start)
            if _is_function_word(str(word.get("text") or "")) and dur >= func_long_thr:
                dur = min(dur, params["func_max_dur"], median_word_dur * params["func_ratio_max"])
            best_end = max(best_start + params["min_word_dur"], best_start + dur)

            if idx + 1 < len(words):
                next_start = float(words[idx + 1].get("start", best_end))
                best_end = min(best_end, next_start + params["boundary_shift_max"])

            best_end = max(best_start + params["min_word_dur"], best_end)
            if abs(best_start - orig_start) > 1e-6 or abs(best_end - orig_end) > 1e-6:
                changes += 1
                total_shift += abs(best_start - orig_start)

            word["start"] = round(best_start, 3)
            word["end"] = round(best_end, 3)
            prev_end = float(word["end"]) + params["force_forward_min_gap"]

        if words:
            chunk["start"] = float(words[0]["start"])
            chunk["end"] = float(words[-1]["end"])

    chunks = refined.get("chunks", []) or []
    for idx in range(len(chunks) - 1):
        current = chunks[idx]
        nxt = chunks[idx + 1]
        current_words = current.get("words", []) or []
        next_words = nxt.get("words", []) or []
        if not current_words or not next_words:
            continue
        next_start = float(next_words[0].get("start", nxt.get("start", 0.0)))
        if float(current.get("end", 0.0)) > next_start:
            current_words[-1]["end"] = round(max(float(current_words[-1].get("start", next_start)), next_start), 3)
            current["end"] = float(current_words[-1]["end"])
            nxt["start"] = next_start

    report = {
        "profile": profile,
        "params": params,
        "audio_mix": audio_mix,
        "audio_vocals": audio_vocals,
        "sr": sr,
        "hop_length": hop_length,
        "onset_count": len(onsets),
        "words_changed": changes,
        "mean_abs_start_shift": round(total_shift / max(1, changes), 4),
        "median_word_duration_before": round(median_word_dur, 4),
    }
    refined.setdefault("meta", {})
    refined["meta"]["word_refine"] = report
    return refined, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine word-level timestamps while keeping lyric sequence stable.")
    parser.add_argument("chunking_json", type=str)
    parser.add_argument("--audio-mix", type=str, default="")
    parser.add_argument("--audio-vocals", type=str, default="")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--early-thr", type=float, default=0.05)
    parser.add_argument("--func-long-thr", type=float, default=0.55)
    parser.add_argument("--start-shift-max", type=float, default=None)
    parser.add_argument("--start-back-max", type=float, default=None)
    parser.add_argument("--boundary-shift-max", type=float, default=None)
    parser.add_argument("--min-word-dur", type=float, default=None)
    parser.add_argument("--func-max-dur", type=float, default=None)
    parser.add_argument("--func-ratio-max", type=float, default=None)
    parser.add_argument("--keep-weight", type=float, default=None)
    parser.add_argument("--onset-weight", type=float, default=None)
    parser.add_argument("--prefer-future-penalty", type=float, default=None)
    parser.add_argument("--force-forward-min-gap", type=float, default=None)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("-o", "--output", type=str, default="")
    parser.add_argument("--report", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = read_json(Path(args.chunking_json))
    refined, report = refine_payload(
        payload,
        profile=args.profile,
        audio_mix=args.audio_mix,
        audio_vocals=args.audio_vocals,
        sr=int(args.sr),
        hop_length=int(args.hop_length),
        early_thr=float(args.early_thr),
        func_long_thr=float(args.func_long_thr),
        overrides={
            "start_shift_max": args.start_shift_max,
            "start_back_max": args.start_back_max,
            "boundary_shift_max": args.boundary_shift_max,
            "min_word_dur": args.min_word_dur,
            "func_max_dur": args.func_max_dur,
            "func_ratio_max": args.func_ratio_max,
            "keep_weight": args.keep_weight,
            "onset_weight": args.onset_weight,
            "prefer_future_penalty": args.prefer_future_penalty,
            "force_forward_min_gap": args.force_forward_min_gap,
        },
    )

    stem = safe_stem(args.chunking_json)
    out_path = Path(args.output) if args.output else Path(args.artifacts_dir) / "refinement" / f"{stem}_wordref_{args.profile}.json"
    report_path = Path(args.report) if args.report else Path(args.artifacts_dir) / "refinement" / f"word_timing_refine_report_{stem}_{args.profile}.json"
    write_json(out_path, refined)
    write_json(report_path, report)
    print(f"[word-refine] wrote {out_path}")
    print(f"[word-refine] wrote {report_path}")
    return 0

