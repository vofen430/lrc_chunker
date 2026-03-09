from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .display_timing import apply_chunk_display_timing
from .utils import FUNCTION_WORDS, find_payload_vocals_path, median, percentile, read_json, safe_stem, tokenize, write_json


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
        "breath_gap_min": 0.14,
        "breath_scan_max": 0.22,
        "breath_flatness_max": 0.55,
        "breath_rms_ratio": 0.38,
        "breath_min_delay": 0.03,
        "min_chunk_dur": 0.04,
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
        "breath_gap_min": 0.12,
        "breath_scan_max": 0.24,
        "breath_flatness_max": 0.58,
        "breath_rms_ratio": 0.35,
        "breath_min_delay": 0.035,
        "min_chunk_dur": 0.04,
    },
    "slow_attack": {
        "start_shift_max": 0.40,
        "start_back_max": 0.03,
        "boundary_shift_max": 0.12,
        "min_word_dur": 0.10,
        "func_max_dur": 0.52,
        "func_ratio_max": 2.1,
        "keep_weight": 0.35,
        "onset_weight": 1.00,
        "prefer_future_penalty": 1.55,
        "force_forward_min_gap": 0.05,
        "breath_gap_min": 0.12,
        "breath_scan_max": 0.60,
        "breath_flatness_max": 0.45,
        "breath_rms_ratio": 0.70,
        "breath_min_delay": 0.10,
        "min_chunk_dur": 0.04,
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
        "breath_gap_min": 0.10,
        "breath_scan_max": 0.26,
        "breath_flatness_max": 0.62,
        "breath_rms_ratio": 0.32,
        "breath_min_delay": 0.03,
        "min_chunk_dur": 0.04,
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
        "breath_gap_min": 0.08,
        "breath_scan_max": 0.18,
        "breath_flatness_max": 0.68,
        "breath_rms_ratio": 0.28,
        "breath_min_delay": 0.02,
        "min_chunk_dur": 0.04,
    },
}


def _iter_words(payload: dict) -> Iterable[dict]:
    for chunk in payload.get("chunks", []):
        for word in chunk.get("words", []) or []:
            yield word


def _sync_top_level_words(refined: dict) -> None:
    # Keep payload["words"] consistent with the per-chunk timings used downstream.
    refined["words"] = [deepcopy(word) for word in _iter_words(refined)]


def _line_timestamp_map(payload: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for raw in payload.get("lines", []) or []:
        try:
            line_id = int(raw.get("line_id"))
            timestamp = float(raw.get("timestamp"))
        except (TypeError, ValueError):
            continue
        out[line_id] = timestamp
    return out


def _line_first_word_positions(payload: dict) -> Dict[int, tuple[int, int]]:
    out: Dict[int, tuple[int, int]] = {}
    for chunk_idx, chunk in enumerate(payload.get("chunks", []) or []):
        for word_idx, word in enumerate(chunk.get("words", []) or []):
            try:
                line_id = int(word.get("line_id"))
            except (TypeError, ValueError):
                continue
            if line_id in out:
                continue
            if not str(word.get("text") or "").strip():
                continue
            out[line_id] = (chunk_idx, word_idx)
    return out


def _line_word_refs(payload: dict) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for word in _iter_words(payload):
        try:
            line_id = int(word.get("line_id"))
        except (TypeError, ValueError):
            continue
        out.setdefault(line_id, []).append(word)
    return out


def _load_refine_signals(*audio_paths: str, sr: int, hop_length: int) -> Dict[str, object]:
    valid_paths = [path for path in audio_paths if path]
    if not valid_paths:
        return {
            "onsets": [],
            "frame_times": np.asarray([], dtype=np.float64),
            "rms": np.asarray([], dtype=np.float64),
            "flatness": np.asarray([], dtype=np.float64),
            "global_rms_ref": 0.0,
        }
    try:
        import librosa  # type: ignore
    except Exception:
        return {
            "onsets": [],
            "frame_times": np.asarray([], dtype=np.float64),
            "rms": np.asarray([], dtype=np.float64),
            "flatness": np.asarray([], dtype=np.float64),
            "global_rms_ref": 0.0,
        }

    onset_times: List[float] = []
    analysis_path = valid_paths[0]
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
    y, use_sr = librosa.load(analysis_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=use_sr, hop_length=hop_length)
    non_silent = [float(v) for v in rms if float(v) > 1e-6]
    return {
        "onsets": sorted(set(round(t, 3) for t in onset_times)),
        "frame_times": np.asarray(frame_times, dtype=np.float64),
        "rms": np.asarray(rms, dtype=np.float64),
        "flatness": np.asarray(flatness, dtype=np.float64),
        "global_rms_ref": percentile(non_silent, 0.35) if non_silent else 0.0,
    }


def _nearest_candidates(onsets: List[float], t: float, start_back_max: float, start_shift_max: float) -> List[float]:
    cands = [t]
    for onset in onsets:
        diff = onset - t
        if -start_back_max <= diff <= start_shift_max:
            cands.append(onset)
    return sorted(set(cands))


def _lrc_anchor_candidates(
    onsets: List[float],
    target: float,
    current_start: float,
    start_back_max: float,
    anchor_window: float,
) -> List[float]:
    cands = [current_start, target]
    for onset in onsets:
        diff = onset - target
        if -start_back_max <= diff <= anchor_window:
            cands.append(onset)
    return sorted(set(cands))


def _select_lrc_anchor_target(
    onsets: List[float],
    current_start: float,
    line_timestamp: float,
    *,
    start_back_max: float,
    anchor_window: float,
    anchor_weight: float,
    keep_weight: float,
) -> float:
    candidates = _lrc_anchor_candidates(onsets, line_timestamp, current_start, start_back_max, anchor_window)
    best = current_start
    best_score = float("inf")
    for cand in candidates:
        score = anchor_weight * abs(cand - line_timestamp) + keep_weight * abs(cand - current_start)
        if score < best_score:
            best_score = score
            best = cand
    return best


def _is_function_word(text: str) -> bool:
    toks = [tok.lower() for tok in tokenize(text) if any(ch.isalnum() for ch in tok)]
    return bool(toks) and all(tok in FUNCTION_WORDS for tok in toks)


def _word_char_weight(text: str) -> float:
    text = str(text or "").strip()
    if not text:
        return 1.0
    tokens = tokenize(text)
    if not tokens:
        return 1.0
    weight = 0.0
    for token in tokens:
        if any("\u4e00" <= ch <= "\u9fff" for ch in token):
            weight += sum(1.0 for ch in token if "\u4e00" <= ch <= "\u9fff")
            continue
        alnum = sum(1 for ch in token if ch.isalnum())
        if alnum > 0:
            weight += float(alnum)
        else:
            weight += 0.5
    return max(1.0, weight)


def _chunk_display_bounds(chunk: dict) -> Tuple[float, float]:
    start = float(chunk.get("display_start", chunk.get("start", 0.0)))
    end = float(chunk.get("display_end", chunk.get("end", start)))
    return start, max(start, end)


def _clear_simulated_timing(word: dict) -> None:
    for key in ("simulated_start", "simulated_end", "simulated_midpoint", "simulated_duration"):
        word.pop(key, None)


def _repair_zero_duration_words_in_chunk(
    chunk: dict,
    *,
    zero_word_epsilon: float,
    min_word_dur: float,
    global_median_word_dur: float,
    max_word_dur: float,
    function_word_ratio: float,
    inter_word_safety_gap: float,
) -> tuple[int, int, int]:
    words = chunk.get("words", []) or []
    if not words:
        return 0, 0, 0

    for word in words:
        raw_start = float(word.get("start", 0.0))
        raw_end = float(word.get("end", raw_start))
        raw_duration = max(0.0, raw_end - raw_start)
        flagged_for_simulation = bool(word.get("_needs_simulated_word_timing"))
        if raw_duration > zero_word_epsilon and not flagged_for_simulation:
            _clear_simulated_timing(word)
            word["timing_source"] = "alignment"
            word["timing_confidence"] = float(word.get("confidence", 1.0) or 1.0)

    valid_indices = [
        idx
        for idx, word in enumerate(words)
        if max(0.0, float(word.get("end", 0.0)) - float(word.get("start", 0.0))) > zero_word_epsilon
        and not bool(word.get("_needs_simulated_word_timing"))
    ]
    valid_durations = [
        max(0.0, float(words[idx].get("end", 0.0)) - float(words[idx].get("start", 0.0)))
        for idx in valid_indices
    ]
    valid_char_weights = [_word_char_weight(words[idx].get("text", "")) for idx in valid_indices]
    chunk_median_word_dur = max(min_word_dur, median(valid_durations) or global_median_word_dur or min_word_dur)
    char_rate = (
        sum(valid_durations) / max(1.0, sum(valid_char_weights))
        if valid_durations and sum(valid_char_weights) > 0
        else 0.0
    )
    duration_ceiling = max(min_word_dur, min(max_word_dur, 2.5 * chunk_median_word_dur))
    display_start, display_end = _chunk_display_bounds(chunk)

    zero_indices = [
        idx
        for idx, word in enumerate(words)
        if bool(word.get("_needs_simulated_word_timing"))
        or max(0.0, float(word.get("end", 0.0)) - float(word.get("start", 0.0))) <= zero_word_epsilon
    ]
    if not zero_indices:
        return 0, 0, 0

    groups: List[Tuple[int, int]] = []
    group_start = zero_indices[0]
    prev_idx = zero_indices[0]
    for idx in zero_indices[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
            continue
        groups.append((group_start, prev_idx))
        group_start = idx
        prev_idx = idx
    groups.append((group_start, prev_idx))

    repaired_words = 0
    repaired_groups = 0
    compressed_words = 0
    for lo, hi in groups:
        repaired_groups += 1
        left_idx = lo - 1 if lo - 1 >= 0 else None
        right_idx = hi + 1 if hi + 1 < len(words) else None

        left_edge = display_start
        if left_idx is not None:
            left_edge = max(left_edge, float(words[left_idx].get("end", words[left_idx].get("start", display_start))))
        right_edge = display_end
        if right_idx is not None:
            right_edge = min(right_edge, float(words[right_idx].get("start", display_end)))
        if right_edge < left_edge:
            right_edge = left_edge

        group_words = words[lo : hi + 1]
        estimates: List[float] = []
        for word in group_words:
            char_weight = _word_char_weight(word.get("text", ""))
            estimated = char_rate * char_weight if char_rate > 0 else chunk_median_word_dur
            if _is_function_word(str(word.get("text") or "")):
                estimated = min(estimated, chunk_median_word_dur * function_word_ratio)
            estimated = max(min_word_dur, min(duration_ceiling, estimated))
            estimates.append(float(estimated))

        total_gap_budget = inter_word_safety_gap * max(0, len(group_words) - 1)
        available_duration = max(0.0, right_edge - left_edge - total_gap_budget)
        total_estimated = sum(estimates)
        if total_estimated <= 0:
            estimates = [min_word_dur for _ in group_words]
            total_estimated = sum(estimates)

        scale = 1.0
        if available_duration > 0 and total_estimated > available_duration:
            scale = available_duration / total_estimated
            compressed_words += len(group_words)

        fitted_durations = [max(1e-3, est * scale) for est in estimates]
        fitted_total = sum(fitted_durations)
        usable_duration = max(0.0, right_edge - left_edge - total_gap_budget)
        lead_slack = max(0.0, usable_duration - fitted_total) * 0.5
        cursor = left_edge + lead_slack
        if available_duration <= 0.0:
            cursor = left_edge

        for idx, word in enumerate(group_words):
            sim_start = cursor
            sim_end = sim_start + fitted_durations[idx]
            if right_idx is not None and idx == len(group_words) - 1:
                sim_end = min(sim_end, right_edge)
            sim_end = max(sim_start + 1e-3, sim_end)
            sim_mid = 0.5 * (sim_start + sim_end)
            word["simulated_start"] = round(sim_start, 3)
            word["simulated_end"] = round(sim_end, 3)
            word["simulated_midpoint"] = round(sim_mid, 3)
            word["simulated_duration"] = round(max(1e-3, sim_end - sim_start), 3)
            word["timing_source"] = "simulated_from_chunk_context"
            word["timing_confidence"] = 0.35
            word.pop("_needs_simulated_word_timing", None)
            repaired_words += 1
            cursor = sim_end + inter_word_safety_gap

    for word in words:
        word.pop("_needs_simulated_word_timing", None)

    return repaired_words, repaired_groups, compressed_words


def _find_voiced_start(
    frame_times: np.ndarray,
    rms: np.ndarray,
    flatness: np.ndarray,
    start: float,
    search_end: float,
    global_rms_ref: float,
    params: Dict[str, float],
) -> Optional[float]:
    if frame_times.size == 0 or rms.size == 0 or flatness.size == 0:
        return None
    mask = (frame_times >= float(start)) & (frame_times <= float(search_end))
    if not np.any(mask):
        return None
    local_times = frame_times[mask]
    local_rms = rms[mask]
    local_flatness = flatness[mask]
    local_peak = float(np.max(local_rms)) if local_rms.size else 0.0
    if local_peak <= 0.0:
        return None
    rms_floor = max(float(global_rms_ref) * float(params["breath_rms_ratio"]), local_peak * float(params["breath_rms_ratio"]))
    flatness_max = float(params["breath_flatness_max"])
    for t, amp, flat in zip(local_times, local_rms, local_flatness):
        if float(amp) >= rms_floor and float(flat) <= flatness_max:
            return float(t)
    return None


def _apply_breath_guard(
    best_start: float,
    orig_end: float,
    prev_end: float,
    idx: int,
    frame_times: np.ndarray,
    rms: np.ndarray,
    flatness: np.ndarray,
    global_rms_ref: float,
    params: Dict[str, float],
) -> float:
    gap_before = max(0.0, best_start - prev_end)
    if idx > 0 and gap_before < float(params["breath_gap_min"]):
        return best_start
    search_end = min(orig_end, best_start + float(params["breath_scan_max"]))
    voiced_start = _find_voiced_start(frame_times, rms, flatness, best_start, search_end, global_rms_ref, params)
    if voiced_start is None:
        return best_start
    if voiced_start - best_start < float(params["breath_min_delay"]):
        return best_start
    return min(voiced_start, orig_end - float(params["min_word_dur"]))


def _apply_line_anchor_warp(
    words: List[dict],
    target_start: float,
    *,
    anchor_span_words: int,
    anchor_max_ratio: float,
    min_delta: float,
) -> float:
    if not words:
        return 0.0
    first_start = float(words[0].get("start", 0.0))
    raw_delta = float(target_start) - first_start
    if abs(raw_delta) < min_delta:
        return 0.0

    if len(words) == 1:
        orig_start = float(words[0].get("start", 0.0))
        orig_end = float(words[0].get("end", orig_start))
        words[0]["start"] = round(orig_start + raw_delta, 3)
        words[0]["end"] = round(orig_end + raw_delta, 3)
        return raw_delta

    span_words = max(1, min(anchor_span_words, len(words)))
    if span_words < len(words):
        region_end = float(words[span_words].get("start", float(words[-1].get("end", first_start))))
    else:
        region_end = float(words[-1].get("end", first_start))
    region_len = max(0.20, region_end - first_start)
    max_delta = region_len * max(0.10, min(anchor_max_ratio, 0.95))
    delta = max(-max_delta, min(max_delta, raw_delta))
    if abs(delta) < min_delta:
        return 0.0

    originals = [
        (
            float(word.get("start", 0.0)),
            float(word.get("end", float(word.get("start", 0.0)))),
        )
        for word in words
    ]

    def warp(t: float) -> float:
        if t <= first_start:
            return t + delta
        if t >= region_end:
            return t
        ratio = (t - first_start) / region_len
        return t + delta * (1.0 - ratio)

    for word, (orig_start, orig_end) in zip(words, originals):
        if orig_start < region_end:
            word["start"] = round(warp(orig_start), 3)
        if orig_end < region_end:
            word["end"] = round(warp(orig_end), 3)
    return float(words[0].get("start", first_start)) - first_start


def refine_payload(
    payload: dict,
    *,
    profile: str,
    audio_mix: str = "",
    audio_vocals: str = "",
    use_lrc_anchors: bool = False,
    lrc_anchor_window: float = 0.18,
    lrc_anchor_weight: float = 3.5,
    lrc_anchor_keep_weight: float = 0.30,
    lrc_anchor_min_delta: float = 0.04,
    lrc_anchor_span_words: int = 1,
    lrc_anchor_max_ratio: float = 0.15,
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
    signals = _load_refine_signals(audio_vocals, audio_mix, sr=sr, hop_length=hop_length)
    onsets = list(signals["onsets"])
    frame_times = np.asarray(signals["frame_times"], dtype=np.float64)
    rms = np.asarray(signals["rms"], dtype=np.float64)
    flatness = np.asarray(signals["flatness"], dtype=np.float64)
    global_rms_ref = float(signals["global_rms_ref"])
    changes = 0
    total_shift = 0.0
    breath_guard_moves = 0
    lrc_anchor_moves = 0
    repaired_zero_words = 0
    repaired_zero_word_groups = 0
    compressed_simulated_words = 0
    durations: List[float] = [max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0))) for w in _iter_words(payload)]
    median_word_dur = max(params["min_word_dur"], median(durations) or params["min_word_dur"])
    line_timestamps = _line_timestamp_map(payload) if use_lrc_anchors else {}

    for chunk_idx, chunk in enumerate(refined.get("chunks", [])):
        chunk_start = float(chunk.get("start", 0.0))
        prev_end = chunk_start
        words = chunk.get("words", []) or []
        for idx, word in enumerate(words):
            orig_start = float(word.get("start", 0.0))
            orig_end = float(word.get("end", orig_start))
            word["_needs_simulated_word_timing"] = (orig_end - orig_start) <= 0.01
            prev_anchor = chunk_start if idx == 0 else prev_end
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
            best_start = max(prev_anchor, best_start)
            guarded_start = _apply_breath_guard(
                best_start,
                orig_end,
                prev_anchor,
                idx,
                frame_times,
                rms,
                flatness,
                global_rms_ref,
                params,
            )
            if guarded_start > best_start + 1e-6:
                breath_guard_moves += 1
            best_start = max(prev_anchor, guarded_start)

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

    if use_lrc_anchors:
        for line_id, words in _line_word_refs(refined).items():
            if line_id not in line_timestamps or not words:
                continue
            target_start = _select_lrc_anchor_target(
                onsets,
                float(words[0].get("start", 0.0)),
                line_timestamps[line_id],
                start_back_max=float(params["start_back_max"]),
                anchor_window=float(lrc_anchor_window),
                anchor_weight=float(lrc_anchor_weight),
                keep_weight=float(lrc_anchor_keep_weight),
            )
            applied = _apply_line_anchor_warp(
                words,
                target_start,
                anchor_span_words=int(lrc_anchor_span_words),
                anchor_max_ratio=float(lrc_anchor_max_ratio),
                min_delta=float(lrc_anchor_min_delta),
            )
            if abs(applied) > 1e-6:
                lrc_anchor_moves += 1

    for chunk in refined.get("chunks", []):
        words = chunk.get("words", []) or []
        if words:
            chunk["start"] = float(words[0]["start"])
            chunk["end"] = float(words[-1]["end"])

    original_chunks = payload.get("chunks", []) or []
    original_positive_indices = [
        idx
        for idx, chunk in enumerate(original_chunks)
        if float(chunk.get("end", chunk.get("start", 0.0))) > float(chunk.get("start", 0.0))
    ]
    original_zero_indices = {
        idx
        for idx, chunk in enumerate(original_chunks)
        if float(chunk.get("end", chunk.get("start", 0.0))) <= float(chunk.get("start", 0.0))
    }

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

    dropped_zero_chunks = 0
    interior_zero_chunks_fixed = 0
    if original_positive_indices:
        keep_lo = original_positive_indices[0]
        keep_hi = original_positive_indices[-1]
        dropped_zero_chunks = keep_lo + max(0, len(chunks) - 1 - keep_hi)
        chunks = chunks[keep_lo : keep_hi + 1]
        refined["chunks"] = chunks
        original_zero_indices = {idx - keep_lo for idx in original_zero_indices if keep_lo <= idx <= keep_hi}
    else:
        dropped_zero_chunks = len(chunks)
        refined["chunks"] = []
        chunks = []
        original_zero_indices = set()

    min_chunk_dur = float(params.get("min_chunk_dur", params["min_word_dur"]))
    for idx, chunk in enumerate(chunks):
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        if end > start and idx not in original_zero_indices:
            continue
        target_end = start + min_chunk_dur
        if idx + 1 < len(chunks):
            next_chunk = chunks[idx + 1]
            next_start = float(next_chunk.get("start", target_end))
            if next_start > start:
                target_end = min(target_end, next_start)
            else:
                next_chunk["start"] = round(target_end, 3)
        chunk["end"] = round(max(start + min_chunk_dur, target_end), 3)
        interior_zero_chunks_fixed += 1

    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = idx

    apply_chunk_display_timing(refined)

    for chunk in refined.get("chunks", []):
        fixed_words, fixed_groups, compressed_words = _repair_zero_duration_words_in_chunk(
            chunk,
            zero_word_epsilon=0.01,
            min_word_dur=float(params["min_word_dur"]),
            global_median_word_dur=median_word_dur,
            max_word_dur=0.42,
            function_word_ratio=0.90,
            inter_word_safety_gap=0.005,
        )
        repaired_zero_words += fixed_words
        repaired_zero_word_groups += fixed_groups
        compressed_simulated_words += compressed_words

    report = {
        "profile": profile,
        "params": params,
        "audio_mix": audio_mix,
        "audio_vocals": audio_vocals,
        "sr": sr,
        "hop_length": hop_length,
        "onset_count": len(onsets),
        "breath_guard_moves": breath_guard_moves,
        "use_lrc_anchors": use_lrc_anchors,
        "lrc_anchor_moves": lrc_anchor_moves,
        "lrc_anchor_window": lrc_anchor_window,
        "lrc_anchor_weight": lrc_anchor_weight,
        "lrc_anchor_keep_weight": lrc_anchor_keep_weight,
        "lrc_anchor_min_delta": lrc_anchor_min_delta,
        "lrc_anchor_span_words": lrc_anchor_span_words,
        "lrc_anchor_max_ratio": lrc_anchor_max_ratio,
        "words_changed": changes,
        "mean_abs_start_shift": round(total_shift / max(1, changes), 4),
        "median_word_duration_before": round(median_word_dur, 4),
        "dropped_zero_chunks": dropped_zero_chunks,
        "interior_zero_chunks_fixed": interior_zero_chunks_fixed,
        "simulated_zero_words": repaired_zero_words,
        "simulated_zero_word_groups": repaired_zero_word_groups,
        "compressed_simulated_words": compressed_simulated_words,
    }
    _sync_top_level_words(refined)
    refined.setdefault("meta", {})
    refined["meta"]["word_refine"] = report
    return refined, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine word-level timestamps while keeping lyric sequence stable.")
    parser.add_argument("chunking_json", type=str)
    parser.add_argument("--audio-mix", type=str, default="")
    parser.add_argument("--audio-vocals", type=str, default="")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="slow_attack")
    parser.add_argument("--use-lrc-anchors", dest="use_lrc_anchors", action="store_true", default=True)
    parser.add_argument("--no-lrc-anchors", dest="use_lrc_anchors", action="store_false")
    parser.add_argument("--lrc-anchor-window", type=float, default=0.18)
    parser.add_argument("--lrc-anchor-weight", type=float, default=4.5)
    parser.add_argument("--lrc-anchor-keep-weight", type=float, default=0.20)
    parser.add_argument("--lrc-anchor-min-delta", type=float, default=0.04)
    parser.add_argument("--lrc-anchor-span-words", type=int, default=4)
    parser.add_argument("--lrc-anchor-max-ratio", type=float, default=0.35)
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
    audio_vocals = args.audio_vocals or find_payload_vocals_path(payload)
    refined, report = refine_payload(
        payload,
        profile=args.profile,
        audio_mix=args.audio_mix,
        audio_vocals=audio_vocals,
        use_lrc_anchors=bool(args.use_lrc_anchors),
        lrc_anchor_window=float(args.lrc_anchor_window),
        lrc_anchor_weight=float(args.lrc_anchor_weight),
        lrc_anchor_keep_weight=float(args.lrc_anchor_keep_weight),
        lrc_anchor_min_delta=float(args.lrc_anchor_min_delta),
        lrc_anchor_span_words=int(args.lrc_anchor_span_words),
        lrc_anchor_max_ratio=float(args.lrc_anchor_max_ratio),
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

