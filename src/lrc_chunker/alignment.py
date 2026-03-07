from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .lrc import reference_text
from .models import Chunk, LyricLine, WordTiming
from .utils import safe_stem


@dataclass
class AlignmentConfig:
    model: str = "small.en"
    language: str = "en"
    vad_threshold: float = 0.35
    denoiser: str = "auto"
    denoiser_effective: str = "none"
    denoiser_output_path: str = ""
    max_word_dur: float = 3.0
    alignment_backend: str = "stable_ts"
    allow_lrc_fallback: bool = False
    only_voice_freq: bool = False


def _is_word_token(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _flatten_stable_result(result_obj: object) -> List[Dict[str, float]]:
    if hasattr(result_obj, "to_dict"):
        payload = result_obj.to_dict()
    else:
        payload = result_obj
    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    flat: List[Dict[str, float]] = []
    for seg in segments:
        for word in seg.get("words", []) or []:
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            flat.append(
                {
                    "text": text,
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", word.get("start", 0.0))),
                    "probability": float(word.get("probability", 1.0)),
                }
            )
    return flat


def _assign_aligned_words_to_lines(lines: Sequence[LyricLine], aligned_words: Sequence[Dict[str, float]]) -> Tuple[List[WordTiming], List[Dict[str, object]]]:
    from .utils import tokenize

    line_word_tokens: List[List[str]] = []
    for line in lines:
        toks = [tok for tok in tokenize(line.text) if _is_word_token(tok)]
        line_word_tokens.append(toks)

    word_items = [w for w in aligned_words if _is_word_token(str(w["text"]))]
    word_cursor = 0
    out_words: List[WordTiming] = []
    line_records: List[Dict[str, object]] = []

    for line, line_tokens in zip(lines, line_word_tokens):
        current: List[WordTiming] = []
        for token in line_tokens:
            if word_cursor >= len(word_items):
                break
            item = word_items[word_cursor]
            current.append(
                WordTiming(
                    text=token,
                    start=float(item["start"]),
                    end=float(item["end"]),
                    line_id=line.line_id,
                    confidence=float(item.get("probability", 1.0)),
                    source="stable_ts",
                    index=len(out_words),
                )
            )
            out_words.append(current[-1])
            word_cursor += 1

        if current:
            start = current[0].start
            end = current[-1].end
        else:
            start = line.timestamp
            end = line.timestamp
        line_records.append(
            {
                "line_id": line.line_id,
                "text": line.text,
                "start": start,
                "end": end,
                "timestamp": line.timestamp,
                "word_count": len(current),
            }
        )

    return out_words, line_records


def fallback_align_from_lrc(lines: Sequence[LyricLine], audio_path: Optional[str] = None) -> Tuple[List[WordTiming], List[Dict[str, object]], str]:
    from .utils import tokenize

    words: List[WordTiming] = []
    records: List[Dict[str, object]] = []
    for idx, line in enumerate(lines):
        toks = [tok for tok in tokenize(line.text) if _is_word_token(tok)]
        next_ts = lines[idx + 1].timestamp if idx + 1 < len(lines) else line.timestamp + max(2.0, 0.48 * max(1, len(toks)))
        span = max(0.30, next_ts - line.timestamp)
        step = span / max(1, len(toks))
        line_words: List[WordTiming] = []
        for j, tok in enumerate(toks):
            start = line.timestamp + j * step
            end = line.timestamp + (j + 1) * step
            word = WordTiming(
                text=tok,
                start=start,
                end=end,
                line_id=line.line_id,
                confidence=0.0,
                source="lrc_fallback",
                index=len(words),
            )
            words.append(word)
            line_words.append(word)
        records.append(
            {
                "line_id": line.line_id,
                "text": line.text,
                "timestamp": line.timestamp,
                "start": line_words[0].start if line_words else line.timestamp,
                "end": line_words[-1].end if line_words else line.timestamp,
                "word_count": len(line_words),
            }
        )
    return words, records, "lrc_fallback"


def _normalize_denoiser_name(config: AlignmentConfig) -> Optional[str]:
    requested = str(config.denoiser or "").strip().lower()
    if requested in {"", "none", "off", "false", "0"}:
        return None
    if requested != "auto":
        return requested
    try:
        from stable_whisper.audio import get_denoiser_func  # type: ignore

        get_denoiser_func("demucs", "access")()
    except Exception:
        return None
    return "demucs"


def align_lyrics(audio_path: str, lines: Sequence[LyricLine], config: AlignmentConfig) -> Tuple[List[WordTiming], List[Dict[str, object]], str]:
    if config.alignment_backend == "lrc":
        return fallback_align_from_lrc(lines, audio_path)

    try:
        import stable_whisper  # type: ignore
        from stable_whisper.alignment import align as stable_align  # type: ignore
        from stable_whisper.default import set_global_overwrite_permission  # type: ignore
    except Exception:
        if config.allow_lrc_fallback:
            return fallback_align_from_lrc(lines, audio_path)
        raise

    try:
        set_global_overwrite_permission(True)
        model = stable_whisper.load_model(config.model)
        denoiser_name = _normalize_denoiser_name(config)
        denoiser_options: Optional[Dict[str, object]] = None
        if denoiser_name:
            denoiser_options = {}
            if config.denoiser_output_path:
                save_path = Path(config.denoiser_output_path).expanduser().resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                denoiser_options["save_path"] = str(save_path)
        config.denoiser_effective = denoiser_name or "none"
        result = stable_align(
            model,
            audio_path,
            reference_text(list(lines)),
            language=config.language,
            max_word_dur=config.max_word_dur,
            denoiser=denoiser_name,
            denoiser_options=denoiser_options,
            vad=True,
            vad_threshold=config.vad_threshold,
            only_voice_freq=config.only_voice_freq,
            original_split=False,
        )
        flat = _flatten_stable_result(result)
        if not flat:
            raise RuntimeError("stable-ts returned no aligned words")
        words, line_records = _assign_aligned_words_to_lines(lines, flat)
        return words, line_records, "stable_ts"
    except Exception:
        config.denoiser_effective = "none"
        if config.allow_lrc_fallback:
            return fallback_align_from_lrc(lines, audio_path)
        raise


def build_alignment_payload(
    *,
    audio_path: str,
    lrc_path: str,
    lines: Sequence[LyricLine],
    words: Sequence[WordTiming],
    chunks: Sequence[Chunk],
    config: AlignmentConfig,
    backend_used: str,
) -> Dict[str, object]:
    return {
        "meta": {
            "audio_path": str(Path(audio_path)),
            "lrc_path": str(Path(lrc_path)),
            "alignment_backend_requested": config.alignment_backend,
            "alignment_backend_used": backend_used,
            "model": config.model,
            "language": config.language,
            "vad_threshold": config.vad_threshold,
            "denoiser_requested": config.denoiser,
            "denoiser_effective": config.denoiser_effective,
            "denoiser_output_path": config.denoiser_output_path,
            "max_word_dur": config.max_word_dur,
        },
        "lines": [line.to_dict() for line in lines],
        "words": [word.to_dict() for word in words],
        "chunks": [chunk.to_dict() for chunk in chunks],
    }


def default_alignment_output(artifacts_dir: str, audio_path: str, model: str) -> Path:
    stem = safe_stem(audio_path)
    return Path(artifacts_dir) / "alignment" / f"chunking_{stem}_{model}.json"
