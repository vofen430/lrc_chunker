from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def _line_token_count(text: str) -> int:
    from .utils import tokenize

    return sum(1 for tok in tokenize(text or "") if any(ch.isalnum() for ch in tok))


def _audio_duration_from_payload(payload: dict) -> Optional[float]:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    audio_path = str(meta.get("audio_path") or "").strip() if isinstance(meta, dict) else ""
    if not audio_path:
        return None
    path = Path(audio_path).expanduser()
    if not path.is_file():
        return None
    try:
        import soundfile as sf  # type: ignore

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def _chunk_line_id(chunk: dict) -> Optional[int]:
    line_ids = chunk.get("line_ids", []) or []
    if line_ids:
        try:
            return int(line_ids[0])
        except (TypeError, ValueError):
            pass
    for word in chunk.get("words", []) or []:
        try:
            return int(word.get("line_id"))
        except (TypeError, ValueError):
            continue
    return None


def _line_interval_end(lines: List[dict], idx: int, audio_duration: Optional[float]) -> float:
    line = lines[idx]
    start = float(line.get("timestamp", 0.0))
    if idx + 1 < len(lines):
        return float(lines[idx + 1].get("timestamp", start))
    fallback = start + max(2.0, 0.48 * max(1, _line_token_count(str(line.get("text") or ""))))
    if audio_duration is not None:
        return max(start, min(float(audio_duration), fallback))
    return fallback


def apply_chunk_display_timing(payload: dict, *, min_chunk_weight: float = 0.12) -> dict:
    lines = list(payload.get("lines", []) or [])
    chunks = list(payload.get("chunks", []) or [])
    if not lines or not chunks:
        return payload

    audio_duration = _audio_duration_from_payload(payload)
    chunks_by_line: Dict[int, List[dict]] = {}
    for chunk in chunks:
        line_id = _chunk_line_id(chunk)
        if line_id is None:
            chunk["display_start"] = float(chunk.get("start", 0.0))
            chunk["display_end"] = float(chunk.get("end", chunk.get("start", 0.0)))
            continue
        chunks_by_line.setdefault(line_id, []).append(chunk)

    for idx, line in enumerate(lines):
        try:
            line_id = int(line.get("line_id"))
        except (TypeError, ValueError):
            continue
        line_chunks = chunks_by_line.get(line_id, [])
        if not line_chunks:
            continue
        interval_start = float(line.get("timestamp", 0.0))
        interval_end = _line_interval_end(lines, idx, audio_duration)
        if idx + 1 >= len(lines):
            acoustic_end = max(
                float(chunk.get("end", chunk.get("start", interval_start)))
                for chunk in line_chunks
            )
            interval_end = max(interval_end, acoustic_end, interval_start + max(min_chunk_weight, 1e-3))
        interval_duration = max(interval_end - interval_start, max(min_chunk_weight, 1e-3))

        weights: List[float] = []
        for chunk in line_chunks:
            acoustic_start = float(chunk.get("start", interval_start))
            acoustic_end = float(chunk.get("end", acoustic_start))
            acoustic_duration = max(0.0, acoustic_end - acoustic_start)
            weights.append(max(acoustic_duration, float(min_chunk_weight)))

        total_weight = sum(weights) or float(len(line_chunks))
        cursor = interval_start
        for chunk_idx, chunk in enumerate(line_chunks):
            if chunk_idx == len(line_chunks) - 1:
                display_start = cursor
                display_end = interval_end
            else:
                chunk_display_duration = interval_duration * (weights[chunk_idx] / total_weight)
                display_start = cursor
                display_end = min(interval_end, display_start + chunk_display_duration)
            chunk["display_start"] = round(display_start, 3)
            chunk["display_end"] = round(max(display_start, display_end), 3)
            cursor = float(chunk["display_end"])

    return payload
