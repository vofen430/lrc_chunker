from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .models import Chunk, WordTiming
from .utils import FUNCTION_WORDS, clamp, normalize_ws


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
    chunker_model: str = "semantic_dp"
    min_chunk_dur: float = 0.30
    short_gap_block: float = 0.12
    hard_overlap_block: float = 0.03
    semantic_weight: float = 1.0
    embedding_weight: float = 2.4
    gap_weight: float = 0.30
    length_weight: float = 0.65
    line_start_anchor: bool = True
    line_start_anchor_tolerance: float = 1.0
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_instruction: str = (
        "Decide whether a lyric phrase boundary before the target word is natural for karaoke subtitle chunking."
    )
    embedding_local_only: bool = True


class _SemanticBoundaryEmbedder:
    _instances: Dict[tuple[str, str, bool], Optional["_SemanticBoundaryEmbedder"]] = {}
    _load_failures: set[tuple[str, str, bool]] = set()

    def __init__(self, config: ChunkingConfig):
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = str(config.embedding_model_path).strip()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=bool(config.embedding_local_only),
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=bool(config.embedding_local_only),
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer.padding_side = "left"
        self._doc_cache: Dict[str, List[float]] = {}
        self._query_cache: Dict[str, List[float]] = {}
        instruction = str(config.embedding_instruction).strip()
        self.good_prototype = self._encode_query(
            instruction,
            "A natural boundary between two complete lyric phrases for karaoke subtitles.",
        )
        self.bad_prototype = self._encode_query(
            instruction,
            "An unnatural split that breaks a lyric phrase in the middle for karaoke subtitles.",
        )

    @classmethod
    def get(cls, config: ChunkingConfig) -> Optional["_SemanticBoundaryEmbedder"]:
        key = (
            str(config.embedding_model_path).strip(),
            str(config.embedding_instruction).strip(),
            bool(config.embedding_local_only),
        )
        if key in cls._instances:
            return cls._instances[key]
        try:
            cls._instances[key] = cls(config)
        except Exception:
            cls._instances[key] = None
            if key not in cls._load_failures:
                warnings.warn(
                    f'Failed to load embedding model "{config.embedding_model_path}". '
                    "Semantic chunking will fall back to heuristic-only scoring.",
                    stacklevel=2,
                )
                cls._load_failures.add(key)
        return cls._instances[key]

    def _last_token_pool(self, hidden, attention_mask):
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[self._torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def _normalize(self, tensor):
        return self._torch.nn.functional.normalize(tensor, p=2, dim=-1)

    def _encode(self, text: str, *, is_query: bool) -> List[float]:
        cache = self._query_cache if is_query else self._doc_cache
        if text in cache:
            return cache[text]
        encoded = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            outputs = self.model(**encoded)
            pooled = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = self._normalize(pooled)[0].detach().cpu().tolist()
        cache[text] = normalized
        return normalized

    def _encode_query(self, instruction: str, query: str) -> List[float]:
        text = f"Instruct: {instruction}\nQuery: {query}"
        return self._encode(text, is_query=True)

    def _encode_document(self, text: str) -> List[float]:
        return self._encode(text, is_query=False)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    def score_boundary(self, left_text: str, right_text: str) -> float:
        candidate = self._encode_document(f"{left_text} || {right_text}")
        return clamp(
            self._cosine(candidate, self.good_prototype) - self._cosine(candidate, self.bad_prototype),
            -1.0,
            1.0,
        )


def _text(words: Sequence[WordTiming]) -> str:
    return normalize_ws(" ".join(word.text for word in words))


def _gap(prev: WordTiming, nxt: WordTiming) -> float:
    return max(0.0, float(nxt.start) - float(prev.end))


def _raw_gap(prev: WordTiming, nxt: WordTiming) -> float:
    return float(nxt.start) - float(prev.end)


def _is_function_word(word: WordTiming) -> bool:
    return str(word.text or "").strip().lower() in FUNCTION_WORDS


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


def _legacy_split_needed(current: List[WordTiming], nxt: WordTiming, config: ChunkingConfig) -> bool:
    if not current:
        return False
    prev = current[-1]
    gap = _gap(prev, nxt)
    duration = max(0.0, nxt.end - current[0].start)

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


def _legacy_chunk_score(words: Sequence[WordTiming], next_word: WordTiming, config: ChunkingConfig) -> Dict[str, float]:
    if not words:
        return {"boundary_bonus": 0.0, "gap_bonus": 0.0, "long_word_bonus": 0.0}
    last = words[-1]
    gap_bonus = clamp(_gap(last, next_word) / max(config.max_gap, 1e-6), 0.0, 1.0) * config.rhythm_weight
    boundary_bonus = 0.75 if PUNCT_BOUNDARY_RE.search(last.text or "") else 0.0
    line_bonus = 1.0 if next_word.line_id != last.line_id else 0.0
    long_bonus = 0.0
    if config.emphasize_long_words and len(words) == 1 and words[0].end - words[0].start >= config.long_word_single_threshold:
        long_bonus = config.long_word_bonus
    return {
        "boundary_bonus": boundary_bonus + line_bonus,
        "gap_bonus": gap_bonus,
        "long_word_bonus": long_bonus,
    }


def _make_chunk(
    chunk_id: int,
    words: Sequence[WordTiming],
    *,
    start_override: float | None = None,
    end_override: float | None = None,
    scores: Optional[Dict[str, float]] = None,
    flags: Optional[Dict[str, object]] = None,
) -> Chunk:
    text = _text(words)
    start = float(start_override if start_override is not None else words[0].start)
    end = float(end_override if end_override is not None else words[-1].end)
    return Chunk(
        chunk_id=chunk_id,
        start=max(0.0, start),
        end=max(max(0.0, start), end),
        text=text,
        words=list(words),
        line_ids=sorted({word.line_id for word in words if word.line_id is not None}),
        scores=dict(scores or {}),
        flags={
            "word_count": len(words),
            "char_count": len(text),
            **(flags or {}),
        },
    )


def _legacy_build_chunks(words: Sequence[WordTiming], config: ChunkingConfig) -> List[Chunk]:
    if not words:
        return []

    chunks: List[Chunk] = []
    current: List[WordTiming] = []
    for word in words:
        if current and _legacy_split_needed(current, word, config):
            next_word = word
            scores = _legacy_chunk_score(current, next_word, config)
            end = current[-1].end
            if next_word.start < end:
                end = max(current[0].start, 0.5 * (end + next_word.start))
            chunks.append(_make_chunk(len(chunks), current, end_override=end, scores=scores))
            current = []
        current.append(word)

    if current:
        scores = _legacy_chunk_score(current, current[-1], config)
        chunks.append(_make_chunk(len(chunks), current, scores=scores))

    merged: List[Chunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        prev = merged[-1]
        gap = max(0.0, chunk.start - prev.end)
        merged_words = [*prev.words, *chunk.words]
        merged_dur = merged_words[-1].end - merged_words[0].start
        if gap <= config.merge_gap and len(merged_words) <= config.max_words and merged_dur <= config.max_dur:
            merged[-1] = _make_chunk(
                prev.chunk_id,
                merged_words,
                scores={"merged_gap": gap},
                flags={"merged": True},
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


def _group_words_by_line(words: Sequence[WordTiming]) -> List[List[WordTiming]]:
    if not words:
        return []
    groups: List[List[WordTiming]] = []
    current: List[WordTiming] = [words[0]]
    for word in words[1:]:
        if word.line_id != current[-1].line_id:
            groups.append(current)
            current = [word]
            continue
        current.append(word)
    groups.append(current)
    return groups


def _line_boundary_score(words: Sequence[WordTiming], split_idx: int, config: ChunkingConfig) -> Dict[str, float]:
    prev = words[split_idx]
    nxt = words[split_idx + 1]
    raw_gap = _raw_gap(prev, nxt)
    gap = max(0.0, raw_gap)
    long_gap_threshold = max(config.max_gap * 0.75, config.short_gap_block * 2.0, 0.24)
    heuristic = 0.0
    if PUNCT_BOUNDARY_RE.search(prev.text or ""):
        heuristic += 0.35
    if _is_function_word(prev):
        heuristic -= 0.05
    if _is_function_word(nxt):
        heuristic -= 0.05
    if str(nxt.text or "").strip()[:1].isupper():
        heuristic += 0.05

    embedding = 0.0
    embedder = _SemanticBoundaryEmbedder.get(config)
    if embedder is not None:
        embedding = float(embedder.score_boundary(_text(words[: split_idx + 1]), _text(words[split_idx + 1 :])))

    gap_score = clamp(gap / max(config.max_gap, 1e-6), 0.0, 2.0)
    short_gap_penalty = 0.0
    if gap < config.short_gap_block:
        short_gap_penalty = -0.35 * (1.0 - clamp(gap / max(config.short_gap_block, 1e-6), 0.0, 1.0))
    long_gap_bonus = 0.35 if gap >= long_gap_threshold else 0.0
    return {
        "heuristic": heuristic,
        "embedding": embedding,
        "gap": gap_score,
        "short_gap_penalty": short_gap_penalty,
        "long_gap_bonus": long_gap_bonus,
        "combined": (
            config.semantic_weight * heuristic
            + config.embedding_weight * embedding
            + config.gap_weight * gap_score
            + long_gap_bonus
            + short_gap_penalty
        ),
    }


def _word_count_bonus(word_count: int, boundary_support: float, config: ChunkingConfig) -> float:
    long_gap_threshold = max(config.max_gap * 0.75, config.short_gap_block * 2.0, 0.24)
    very_long_gap_threshold = max(config.max_gap * 1.10, config.short_gap_block * 2.8, 0.38)
    if word_count == 1:
        bonus = -0.75
        if boundary_support >= long_gap_threshold:
            bonus += 0.95
        if boundary_support >= very_long_gap_threshold:
            bonus += 0.35
        return bonus
    if word_count == 2:
        bonus = -0.2
        if boundary_support >= long_gap_threshold:
            bonus += 0.95
        return bonus
    if word_count == 3:
        return 1.5
    if word_count == 4:
        return 1.35
    if word_count == 5:
        return -0.45
    return -1.0 - max(0, word_count - 6) * 0.45


def _chunk_penalty(words: Sequence[WordTiming], start: float, end: float, config: ChunkingConfig) -> float:
    text = _text(words)
    duration = max(0.0, end - start)
    penalty = 0.0
    if len(words) > config.max_words:
        penalty -= (len(words) - config.max_words) * 0.5
    if len(text) > config.max_chars:
        penalty -= (len(text) - config.max_chars) * 0.04
    if duration > config.max_dur:
        penalty -= (duration - config.max_dur) * 1.2
    if _is_function_word(words[-1]):
        penalty -= 0.15
    if len(words) > 1 and _is_function_word(words[0]):
        penalty -= 0.10
    return penalty


def _chunk_score(
    line_words: Sequence[WordTiming],
    start_idx: int,
    end_idx: int,
    boundary_scores: Sequence[float],
    line_anchor_start: Optional[float],
    config: ChunkingConfig,
) -> Dict[str, float]:
    words = line_words[start_idx : end_idx + 1]
    chunk_start = float(line_anchor_start) if start_idx == 0 and line_anchor_start is not None else words[0].start
    chunk_end = words[-1].end
    gap_before = 0.0 if start_idx == 0 else _gap(line_words[start_idx - 1], line_words[start_idx])
    gap_after = 0.0 if end_idx == len(line_words) - 1 else _gap(line_words[end_idx], line_words[end_idx + 1])
    boundary_support = max(gap_before, gap_after)
    length_bonus = _word_count_bonus(len(words), boundary_support, config)
    boundary_bonus = 0.8 if end_idx == len(line_words) - 1 else boundary_scores[end_idx]
    penalty = _chunk_penalty(words, chunk_start, chunk_end, config)
    total = boundary_bonus + config.length_weight * length_bonus + penalty
    return {
        "semantic_boundary": boundary_bonus,
        "length_bonus": config.length_weight * length_bonus,
        "penalty": penalty,
        "total": total,
    }


def _is_valid_chunk(
    line_words: Sequence[WordTiming],
    start_idx: int,
    end_idx: int,
    line_anchor_start: Optional[float],
    config: ChunkingConfig,
    *,
    enforce_min_duration: bool,
) -> bool:
    words = line_words[start_idx : end_idx + 1]
    chunk_start = float(line_anchor_start) if start_idx == 0 and line_anchor_start is not None else words[0].start
    duration = max(0.0, words[-1].end - chunk_start)
    if duration > config.hard_max_chunk_dur:
        return False
    if enforce_min_duration and duration < config.min_chunk_dur:
        return False
    if end_idx < len(line_words) - 1 and _raw_gap(line_words[end_idx], line_words[end_idx + 1]) < -config.hard_overlap_block:
        return False
    return True


def _dp_chunk_line(
    line_words: Sequence[WordTiming],
    config: ChunkingConfig,
    line_timestamp: Optional[float],
) -> List[Chunk]:
    if not line_words:
        return []

    boundary_scores = [
        _line_boundary_score(line_words, idx, config)["combined"] for idx in range(len(line_words) - 1)
    ]
    line_anchor_delta = None
    line_anchor_start = None
    line_anchor_suppressed = False
    if config.line_start_anchor and line_timestamp is not None:
        line_anchor_delta = abs(float(line_words[0].start) - float(line_timestamp))
        if line_anchor_delta <= float(config.line_start_anchor_tolerance):
            line_anchor_start = float(line_timestamp)
        else:
            line_anchor_suppressed = True

    def solve(enforce_min_duration: bool) -> List[tuple[int, int, Dict[str, float]]]:
        best_scores = [-float("inf")] * (len(line_words) + 1)
        previous: List[tuple[int, Dict[str, float]] | None] = [None] * (len(line_words) + 1)
        best_scores[0] = 0.0

        for start_idx in range(len(line_words)):
            if best_scores[start_idx] == -float("inf"):
                continue
            for end_idx in range(start_idx, len(line_words)):
                if not _is_valid_chunk(
                    line_words,
                    start_idx,
                    end_idx,
                    line_anchor_start,
                    config,
                    enforce_min_duration=enforce_min_duration,
                ):
                    continue
                score_details = _chunk_score(line_words, start_idx, end_idx, boundary_scores, line_anchor_start, config)
                candidate_score = best_scores[start_idx] + score_details["total"]
                if candidate_score > best_scores[end_idx + 1]:
                    best_scores[end_idx + 1] = candidate_score
                    previous[end_idx + 1] = (start_idx, score_details)

        if best_scores[-1] == -float("inf"):
            return []

        spans: List[tuple[int, int, Dict[str, float]]] = []
        cursor = len(line_words)
        while cursor > 0:
            state = previous[cursor]
            if state is None:
                return []
            start_idx, score_details = state
            spans.append((start_idx, cursor - 1, score_details))
            cursor = start_idx
        spans.reverse()
        return spans

    spans = solve(True) or solve(False)
    if not spans:
        return [
            _make_chunk(
                0,
                line_words,
                start_override=line_anchor_start,
                flags={
                    "chunker_model": "semantic_dp",
                    "line_chunk_index": 0,
                    "line_anchor_suppressed": line_anchor_suppressed,
                    "line_anchor_delta": line_anchor_delta,
                },
            )
        ]

    line_chunks: List[Chunk] = []
    for local_idx, (start_idx, end_idx, score_details) in enumerate(spans):
        chunk_words = line_words[start_idx : end_idx + 1]
        start_override = None
        if local_idx == 0 and line_anchor_start is not None:
            start_override = float(line_anchor_start)
        line_chunks.append(
            _make_chunk(
                local_idx,
                chunk_words,
                start_override=start_override,
                scores=score_details,
                flags={
                    "chunker_model": "semantic_dp",
                    "line_chunk_index": local_idx,
                    "line_anchor_suppressed": bool(line_anchor_suppressed and local_idx == 0),
                    "line_anchor_delta": line_anchor_delta if local_idx == 0 else None,
                },
            )
        )
    return line_chunks


def _semantic_build_chunks(
    words: Sequence[WordTiming],
    config: ChunkingConfig,
    *,
    line_timestamps: Optional[Dict[int, float]],
) -> List[Chunk]:
    groups = _group_words_by_line(words if config.hard_line_breaks else words)
    out: List[Chunk] = []
    for group in groups:
        line_id = group[0].line_id if group else None
        line_timestamp = line_timestamps.get(line_id) if (line_timestamps and line_id is not None) else None
        for chunk in _dp_chunk_line(group, config, line_timestamp):
            chunk.chunk_id = len(out)
            out.append(chunk)
    return out


def build_chunks(
    words: Sequence[WordTiming],
    config: ChunkingConfig,
    *,
    line_timestamps: Optional[Dict[int, float]] = None,
) -> List[Chunk]:
    base_words = _clamp_long_words(words, config.hard_max_chunk_dur) if config.apply_clamp_max else list(words)
    if not base_words:
        return []
    if config.chunker_model == "legacy":
        return _legacy_build_chunks(base_words, config)
    if config.chunker_model != "semantic_dp":
        raise ValueError(f"Unsupported chunker_model: {config.chunker_model}")
    return _semantic_build_chunks(base_words, config, line_timestamps=line_timestamps)
