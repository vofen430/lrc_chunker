from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .models import Chunk, WordTiming
from .utils import clamp, normalize_ws


PUNCT_BOUNDARY_RE = re.compile(r"[.!?,;:]+$")


@dataclass
class ChunkingConfig:
    max_gap: float = 0.35
    merge_gap: float = 0.12
    max_chars: int = 42
    max_words: int = 6
    max_dur: float = 3.2
    hard_max_chunk_dur: float = 6.0
    confident_boundary_threshold: float = 0.98
    confident_boundary_max_span: float = 0.20
    use_confident_boundary_gate: bool = True
    rhythm_weight: float = 2.8
    hard_line_breaks: bool = True
    emphasize_long_words: bool = True
    long_word_single_threshold: float = 0.78
    long_word_bonus: float = 2.6
    apply_clamp_max: bool = True


def _text(words: Sequence[WordTiming]) -> str:
    return normalize_ws(" ".join(word.text for word in words))


def _split_needed(current: List[WordTiming], nxt: WordTiming, config: ChunkingConfig) -> bool:
    if not current:
        return False
    prev = current[-1]
    gap = max(0.0, nxt.start - prev.end)
    duration = max(0.0, nxt.end - current[0].start)
    proposed = _text([*current, nxt])

    def confident_boundary() -> bool:
        if not config.use_confident_boundary_gate:
            return True
        boundary_span = max(prev.end, nxt.end) - min(prev.start, nxt.start)
        return (
            min(float(prev.confidence), float(nxt.confidence)) >= float(config.confident_boundary_threshold)
            and boundary_span <= float(config.confident_boundary_max_span)
        )

    if config.hard_line_breaks and nxt.line_id != prev.line_id:
        return confident_boundary()
    if gap > config.max_gap:
        return confident_boundary()
    if len(current) + 1 > config.max_words:
        return True
    if duration > config.max_dur:
        return confident_boundary()
    if PUNCT_BOUNDARY_RE.search(prev.text or "") and gap >= config.merge_gap:
        return confident_boundary()
    return False


def _clamp_long_words(words: Sequence[WordTiming], hard_max_chunk_dur: float) -> List[WordTiming]:
    out: List[WordTiming] = []
    for word in words:
        start = float(word.start)
        end = float(word.end)
        if end < start:
            end = start
        if end - start > hard_max_chunk_dur:
            end = start + hard_max_chunk_dur
        out.append(
            WordTiming(
                text=word.text,
                start=start,
                end=end,
                line_id=word.line_id,
                confidence=word.confidence,
                source=word.source,
                index=word.index,
            )
        )
    return out


def _chunk_score(words: Sequence[WordTiming], next_word: WordTiming, config: ChunkingConfig) -> Dict[str, float]:
    if not words:
        return {"boundary_bonus": 0.0, "gap_bonus": 0.0, "long_word_bonus": 0.0}
    last = words[-1]
    gap = max(0.0, next_word.start - last.end)
    boundary_bonus = 0.75 if PUNCT_BOUNDARY_RE.search(last.text or "") else 0.0
    line_bonus = 1.0 if next_word.line_id != last.line_id else 0.0
    gap_bonus = clamp(gap / max(config.max_gap, 1e-6), 0.0, 1.0) * config.rhythm_weight
    long_bonus = 0.0
    if config.emphasize_long_words and len(words) == 1 and words[0].end - words[0].start >= config.long_word_single_threshold:
        long_bonus = config.long_word_bonus
    return {
        "boundary_bonus": boundary_bonus + line_bonus,
        "gap_bonus": gap_bonus,
        "long_word_bonus": long_bonus,
    }


def _finalize_chunk(chunk_id: int, words: Sequence[WordTiming], config: ChunkingConfig, next_word: WordTiming | None) -> Chunk:
    start = words[0].start
    end = words[-1].end
    if next_word is not None and next_word.start < end:
        end = max(start, 0.5 * (end + next_word.start))
    scores = _chunk_score(words, next_word or words[-1], config) if words else {}
    return Chunk(
        chunk_id=chunk_id,
        start=start,
        end=max(start, end),
        text=_text(words),
        words=list(words),
        line_ids=sorted({word.line_id for word in words if word.line_id is not None}),
        scores=scores,
        flags={
            "word_count": len(words),
            "char_count": len(_text(words)),
        },
    )


def build_chunks(words: Sequence[WordTiming], config: ChunkingConfig) -> List[Chunk]:
    base_words = _clamp_long_words(words, config.hard_max_chunk_dur) if config.apply_clamp_max else list(words)
    if not base_words:
        return []

    chunks: List[Chunk] = []
    current: List[WordTiming] = []
    for word in base_words:
        if current and _split_needed(current, word, config):
            chunks.append(_finalize_chunk(len(chunks), current, config, word))
            current = []
        current.append(word)

    if current:
        chunks.append(_finalize_chunk(len(chunks), current, config, None))

    merged: List[Chunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        prev = merged[-1]
        gap = max(0.0, chunk.start - prev.end)
        merged_text = normalize_ws(f"{prev.text} {chunk.text}")
        merged_words = [*prev.words, *chunk.words]
        merged_dur = merged_words[-1].end - merged_words[0].start
        if (
            gap <= config.merge_gap
            and len(merged_words) <= config.max_words
            and merged_dur <= config.max_dur
        ):
            merged[-1] = Chunk(
                chunk_id=prev.chunk_id,
                start=merged_words[0].start,
                end=merged_words[-1].end,
                text=merged_text,
                words=merged_words,
                line_ids=sorted({word.line_id for word in merged_words if word.line_id is not None}),
                scores={"merged_gap": gap},
                flags={"merged": True, "word_count": len(merged_words), "char_count": len(merged_text)},
            )
        else:
            chunk.chunk_id = len(merged)
            merged.append(chunk)

    for idx, chunk in enumerate(merged):
        chunk.chunk_id = idx
        if idx + 1 < len(merged):
            next_chunk = merged[idx + 1]
            if next_chunk.start < chunk.end:
                midpoint = 0.5 * (chunk.words[-1].end + next_chunk.words[0].start)
                chunk.end = max(chunk.start, midpoint)
                next_chunk.start = max(chunk.end, next_chunk.start)
    return merged
