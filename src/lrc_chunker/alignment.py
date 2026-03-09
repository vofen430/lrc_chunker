from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .display_timing import apply_chunk_display_timing
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
    qwen_aligner_checkpoint: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    qwen_device: str = "cuda:0"
    qwen_dtype: str = "bfloat16"
    qwen_attn_implementation: str = ""
    align_window_target_seconds: float = 30.0
    align_window_max_seconds: float = 170.0
    align_window_max_lines: int = 1
    align_window_pre_roll: float = 1.0
    align_window_post_roll: float = 1.0


@dataclass
class ForcedAlignmentWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    source: str = "alignment"


@dataclass
class AlignmentWindow:
    start: float
    end: float
    lines: List[LyricLine]


def _is_word_token(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _line_token_count(text: str) -> int:
    from .utils import tokenize

    return sum(1 for tok in tokenize(text) if _is_word_token(tok))


def _flatten_stable_result(result_obj: object) -> List[ForcedAlignmentWord]:
    if hasattr(result_obj, "to_dict"):
        payload = result_obj.to_dict()
    else:
        payload = result_obj
    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    flat: List[ForcedAlignmentWord] = []
    for seg in segments:
        for word in seg.get("words", []) or []:
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            flat.append(
                ForcedAlignmentWord(
                    text=text,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", word.get("start", 0.0))),
                    confidence=float(word.get("probability", 1.0)),
                    source="stable_ts",
                )
            )
    return flat


def _coerce_alignment_word(raw: Any, *, source: str) -> Optional[ForcedAlignmentWord]:
    if raw is None:
        return None
    if isinstance(raw, ForcedAlignmentWord):
        return raw
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("word") or "").strip()
        if not text:
            return None
        start_raw = raw.get("start_time", raw.get("start", 0.0))
        end_raw = raw.get("end_time", raw.get("end", start_raw))
        confidence_raw = raw.get("confidence", raw.get("probability", 1.0))
    else:
        text = str(getattr(raw, "text", getattr(raw, "word", "")) or "").strip()
        if not text:
            return None
        start_raw = getattr(raw, "start_time", getattr(raw, "start", 0.0))
        end_raw = getattr(raw, "end_time", getattr(raw, "end", start_raw))
        confidence_raw = getattr(raw, "confidence", getattr(raw, "probability", 1.0))
    return ForcedAlignmentWord(
        text=text,
        start=float(start_raw),
        end=float(end_raw),
        confidence=float(confidence_raw),
        source=source,
    )


def _flatten_qwen_alignment(result_obj: object) -> List[ForcedAlignmentWord]:
    items = result_obj
    if hasattr(items, "to_dict"):
        items = items.to_dict()
    if not isinstance(items, list):
        items = [items]
    flat: List[ForcedAlignmentWord] = []
    for chunk in items:
        if hasattr(chunk, "items") and not isinstance(chunk, dict):
            for entry in getattr(chunk, "items") or []:
                word = _coerce_alignment_word(entry, source="qwen_forced_aligner")
                if word is not None:
                    flat.append(word)
            continue
        if isinstance(chunk, dict) and isinstance(chunk.get("items"), list):
            for entry in chunk.get("items") or []:
                word = _coerce_alignment_word(entry, source="qwen_forced_aligner")
                if word is not None:
                    flat.append(word)
            continue
        if isinstance(chunk, list):
            for entry in chunk:
                word = _coerce_alignment_word(entry, source="qwen_forced_aligner")
                if word is not None:
                    flat.append(word)
            continue
        word = _coerce_alignment_word(chunk, source="qwen_forced_aligner")
        if word is not None:
            flat.append(word)
    return flat


def _assign_aligned_words_to_lines(lines: Sequence[LyricLine], aligned_words: Sequence[ForcedAlignmentWord]) -> Tuple[List[WordTiming], List[Dict[str, object]]]:
    from .utils import tokenize

    line_word_tokens: List[List[str]] = []
    for line in lines:
        toks = [tok for tok in tokenize(line.text) if _is_word_token(tok)]
        line_word_tokens.append(toks)

    word_items = [w for w in aligned_words if _is_word_token(str(w.text))]
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
                    start=float(item.start),
                    end=float(item.end),
                    line_id=line.line_id,
                    confidence=float(item.confidence),
                    source=item.source,
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


def _estimate_line_end(lines: Sequence[LyricLine], idx: int) -> float:
    line = lines[idx]
    if idx + 1 < len(lines):
        return max(line.timestamp, float(lines[idx + 1].timestamp))
    return float(line.timestamp) + max(2.0, 0.48 * max(1, _line_token_count(line.text)))


def _audio_duration_seconds(audio_path: str) -> float:
    try:
        import soundfile as sf  # type: ignore

        return float(sf.info(audio_path).duration)
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        return float(librosa.get_duration(path=audio_path))
    except Exception:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        raise RuntimeError(f"Unable to determine audio duration for {audio_path}")


def _build_alignment_windows(lines: Sequence[LyricLine], audio_duration: float, config: AlignmentConfig) -> List[AlignmentWindow]:
    if not lines:
        return []

    target_seconds = float(config.align_window_target_seconds or 0.0)
    max_seconds = float(config.align_window_max_seconds or 0.0)
    max_lines = max(1, int(config.align_window_max_lines or 1))
    if target_seconds > 0.0:
        pre_roll = max(0.0, float(config.align_window_pre_roll))
        post_roll = max(0.0, float(config.align_window_post_roll))
        windows: List[AlignmentWindow] = []
        start_idx = 0

        while start_idx < len(lines):
            if start_idx >= len(lines) - 1:
                end_idx = start_idx
            else:
                target_time = float(lines[start_idx].timestamp) + target_seconds
                boundary_idx = min(
                    range(start_idx + 1, len(lines)),
                    key=lambda idx: abs(float(lines[idx].timestamp) - target_time),
                )
                end_idx = max(start_idx, boundary_idx - 1)

            window_start = max(0.0, float(lines[start_idx].timestamp) - pre_roll)
            window_end = min(audio_duration, _estimate_line_end(lines, end_idx) + post_roll)
            if max_seconds > 0.0 and window_end - window_start > max_seconds:
                fallback_end_idx = start_idx
                while fallback_end_idx + 1 <= end_idx:
                    candidate_end = min(audio_duration, _estimate_line_end(lines, fallback_end_idx + 1) + post_roll)
                    if candidate_end - window_start > max_seconds:
                        break
                    fallback_end_idx += 1
                end_idx = fallback_end_idx
                window_end = min(audio_duration, _estimate_line_end(lines, end_idx) + post_roll)

            windows.append(
                AlignmentWindow(
                    start=window_start,
                    end=window_end,
                    lines=list(lines[start_idx : end_idx + 1]),
                )
            )
            start_idx = end_idx + 1

        return windows

    if max_seconds <= 0.0:
        return [
            AlignmentWindow(
                start=0.0,
                end=max(audio_duration, _estimate_line_end(lines, len(lines) - 1)),
                lines=list(lines),
            )
        ]

    pre_roll = max(0.0, float(config.align_window_pre_roll))
    post_roll = max(0.0, float(config.align_window_post_roll))
    windows: List[AlignmentWindow] = []
    start_idx = 0

    while start_idx < len(lines):
        window_start = max(0.0, float(lines[start_idx].timestamp) - pre_roll)
        end_idx = start_idx
        while end_idx + 1 < len(lines):
            if end_idx - start_idx + 1 >= max_lines:
                break
            candidate_end = min(audio_duration, _estimate_line_end(lines, end_idx + 1) + post_roll)
            if candidate_end - window_start > max_seconds:
                break
            end_idx += 1

        window_end = min(audio_duration, _estimate_line_end(lines, end_idx) + post_roll)
        if window_end <= window_start:
            window_end = min(audio_duration, window_start + max(1.0, post_roll))
        windows.append(
            AlignmentWindow(
                start=window_start,
                end=window_end,
                lines=list(lines[start_idx : end_idx + 1]),
            )
        )
        start_idx = end_idx + 1

    return windows


def _load_audio_window(audio_path: str, start: float, end: float):
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(audio_path)
        sr = int(info.samplerate)
        start_frame = max(0, int(round(start * sr)))
        end_frame = max(start_frame + 1, int(round(end * sr)))
        audio, _ = sf.read(
            audio_path,
            start=start_frame,
            frames=end_frame - start_frame,
            dtype="float32",
            always_2d=True,
        )
        return audio.mean(axis=1), sr
    except Exception:
        import librosa  # type: ignore

        audio, sr = librosa.load(audio_path, sr=None, mono=True, offset=start, duration=max(0.0, end - start))
        return audio, int(sr)


def _prepare_alignment_audio(audio_path: str, config: AlignmentConfig) -> str:
    denoiser_name = _normalize_denoiser_name(config)
    config.denoiser_effective = denoiser_name or "none"
    if denoiser_name is None and not config.only_voice_freq:
        return audio_path

    if config.denoiser_output_path:
        save_path = Path(config.denoiser_output_path).expanduser().resolve()
        if save_path.is_file():
            return str(save_path)
    else:
        save_path = None

    try:
        from stable_whisper.audio import prep_audio  # type: ignore
        from stable_whisper.audio.output import save_audio_tensor  # type: ignore
    except Exception:
        if denoiser_name is None and config.only_voice_freq:
            return audio_path
        raise

    processed = prep_audio(
        audio_path,
        denoiser=denoiser_name,
        denoiser_options={},
        only_voice_freq=config.only_voice_freq,
        verbose=False,
        sr=16000,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio_tensor(processed.cpu(), str(save_path), 16000, verbose=False)
        return str(save_path)
    return audio_path


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


def _normalize_qwen_language(language: str) -> str:
    key = str(language or "").strip().lower()
    if not key:
        return "English"
    mapping = {
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "chinese": "Chinese",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "it": "Italian",
        "italian": "Italian",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "russian": "Russian",
        "es": "Spanish",
        "spanish": "Spanish",
    }
    return mapping.get(key, language)


def _resolve_torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name or "").strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported qwen dtype: {dtype_name}")
    return mapping[key]


def _align_with_qwen_forced_aligner(audio_path: str, lines: Sequence[LyricLine], config: AlignmentConfig) -> Tuple[List[WordTiming], List[Dict[str, object]], str]:
    try:
        from qwen_asr import Qwen3ForcedAligner  # type: ignore
    except Exception:
        if config.allow_lrc_fallback:
            return fallback_align_from_lrc(lines, audio_path)
        raise

    try:
        dtype = _resolve_torch_dtype(config.qwen_dtype)
        prepared_audio_path = _prepare_alignment_audio(audio_path, config)
        backend_kwargs: Dict[str, object] = {
            "dtype": dtype,
            "device_map": config.qwen_device,
        }
        if config.qwen_attn_implementation:
            backend_kwargs["attn_implementation"] = config.qwen_attn_implementation
        model = Qwen3ForcedAligner.from_pretrained(
            config.qwen_aligner_checkpoint,
            **backend_kwargs,
        )
        audio_duration = _audio_duration_seconds(prepared_audio_path)
        windows = _build_alignment_windows(lines, audio_duration, config)
        all_words: List[WordTiming] = []
        all_line_records: List[Dict[str, object]] = []
        for window in windows:
            window_audio = _load_audio_window(prepared_audio_path, window.start, window.end)
            result = model.align(
                audio=window_audio,
                text=reference_text(list(window.lines)),
                language=_normalize_qwen_language(config.language),
            )
            flat = _flatten_qwen_alignment(result)
            if not flat:
                raise RuntimeError(
                    f"Qwen forced aligner returned no aligned words for window {window.start:.3f}-{window.end:.3f}s"
                )
            shifted = [
                ForcedAlignmentWord(
                    text=item.text,
                    start=max(0.0, float(item.start) + window.start),
                    end=max(0.0, float(item.end) + window.start),
                    confidence=float(item.confidence),
                    source=item.source,
                )
                for item in flat
            ]
            window_words, window_records = _assign_aligned_words_to_lines(window.lines, shifted)
            all_words.extend(window_words)
            all_line_records.extend(window_records)
        if not all_words:
            raise RuntimeError("Qwen forced aligner returned no aligned words")
        words, line_records = all_words, all_line_records
        return words, line_records, "qwen_forced_aligner"
    except Exception:
        if config.allow_lrc_fallback:
            return fallback_align_from_lrc(lines, audio_path)
        raise


def align_lyrics(audio_path: str, lines: Sequence[LyricLine], config: AlignmentConfig) -> Tuple[List[WordTiming], List[Dict[str, object]], str]:
    if config.alignment_backend == "lrc":
        return fallback_align_from_lrc(lines, audio_path)
    if config.alignment_backend == "qwen_forced_aligner":
        return _align_with_qwen_forced_aligner(audio_path, lines, config)

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
    payload = {
        "meta": {
            "audio_path": str(Path(audio_path)),
            "lrc_path": str(Path(lrc_path)),
            "alignment_backend_requested": config.alignment_backend,
            "alignment_backend_used": backend_used,
            "model": config.model,
            "language": config.language,
            "qwen_aligner_checkpoint": config.qwen_aligner_checkpoint,
            "qwen_device": config.qwen_device,
            "qwen_dtype": config.qwen_dtype,
            "qwen_attn_implementation": config.qwen_attn_implementation,
            "align_window_target_seconds": config.align_window_target_seconds,
            "align_window_max_seconds": config.align_window_max_seconds,
            "align_window_max_lines": config.align_window_max_lines,
            "align_window_pre_roll": config.align_window_pre_roll,
            "align_window_post_roll": config.align_window_post_roll,
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
    return apply_chunk_display_timing(payload)


def default_alignment_output(artifacts_dir: str, audio_path: str, model: str) -> Path:
    stem = safe_stem(audio_path)
    return Path(artifacts_dir) / "alignment" / f"chunking_{stem}_{model}.json"
