from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import LyricLine
from .utils import ascii_ratio, clamp, looks_like_lyric_text, normalize_ws


TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
META_RE = re.compile(r"^\[(ar|ti|al|by|offset|length|re|ve):", re.IGNORECASE)
EDGE_METADATA_RE = re.compile(
    r"^(?:"
    r"\[(?:by|ar|ti|al|offset|length|re|ve):"
    r"|by[:：]"
    r"|作词\b"
    r"|作曲\b"
    r"|编曲\b"
    r"|词[:：]"
    r"|曲[:：]"
    r"|录音\b"
    r"|混音\b"
    r"|母带\b"
    r"|翻译\b"
    r"|字幕\b"
    r"|封面\b"
    r"|制作人\b"
    r"|监制\b"
    r"|composer[:：]"
    r"|lyrics?[:：]"
    r"|written by\b"
    r"|composed by\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LrcParseConfig:
    metadata_head_window_seconds: float = 45.0
    metadata_tail_window_seconds: float = 45.0
    metadata_embedding_model_path: str = "./Qwen3-Embedding-0.6B"
    metadata_embedding_local_only: bool = True
    use_metadata_embedding: bool = True


class _LyricMetadataEmbedder:
    _instances: Dict[tuple[str, bool], Optional["_LyricMetadataEmbedder"]] = {}
    _load_failures: set[tuple[str, bool]] = set()

    def __init__(self, config: LrcParseConfig):
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = str(config.metadata_embedding_model_path).strip()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=bool(config.metadata_embedding_local_only),
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=bool(config.metadata_embedding_local_only),
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer.padding_side = "left"
        self._cache: Dict[str, List[float]] = {}
        self._lyric_prototype = self._encode("A normal sung lyric line in an LRC karaoke file.")
        self._metadata_prototype = self._encode(
            "A metadata or credit line in an LRC file, such as composer, lyricist, arranger, uploader, or by-line."
        )

    @classmethod
    def get(cls, config: LrcParseConfig) -> Optional["_LyricMetadataEmbedder"]:
        if not bool(config.use_metadata_embedding):
            return None
        key = (
            str(config.metadata_embedding_model_path).strip(),
            bool(config.metadata_embedding_local_only),
        )
        if key in cls._instances:
            return cls._instances[key]
        try:
            cls._instances[key] = cls(config)
        except Exception:
            cls._instances[key] = None
            if key not in cls._load_failures:
                warnings.warn(
                    f'Failed to load metadata embedding model "{config.metadata_embedding_model_path}". '
                    "LRC metadata filtering will fall back to heuristic-only scoring.",
                    stacklevel=2,
                )
                cls._load_failures.add(key)
        return cls._instances[key]

    def _last_token_pool(self, hidden, attention_mask):
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[self._torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def _normalize(self, tensor):
        return self._torch.nn.functional.normalize(tensor, p=2, dim=-1)

    def _encode(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        encoded = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            outputs = self.model(**encoded)
            pooled = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = self._normalize(pooled)[0].detach().cpu().tolist()
        self._cache[text] = normalized
        return normalized

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    def lyric_score(self, text: str) -> float:
        encoded = self._encode(text)
        return clamp(
            self._cosine(encoded, self._lyric_prototype) - self._cosine(encoded, self._metadata_prototype),
            -1.0,
            1.0,
        )


def parse_lrc_timestamp(match: Tuple[str, str, str]) -> float:
    mm, ss, frac = match
    frac = frac or "0"
    if len(frac) == 1:
        frac = frac + "00"
    elif len(frac) == 2:
        frac = frac + "0"
    return int(mm) * 60.0 + int(ss) + int(frac[:3]) / 1000.0


def _looks_like_edge_metadata(text: str) -> bool:
    text = normalize_ws(text)
    if not text:
        return False
    if EDGE_METADATA_RE.match(text):
        return True
    lower = text.lower()
    if lower.startswith("[") and lower.endswith("]") and "by:" in lower:
        return True
    return False


def _is_edge_timestamp(timestamp: float, first_timestamp: float, last_timestamp: float, config: LrcParseConfig) -> bool:
    return (
        timestamp <= first_timestamp + max(0.0, float(config.metadata_head_window_seconds))
        or timestamp >= last_timestamp - max(0.0, float(config.metadata_tail_window_seconds))
    )


def _text_selection_score(
    text: str,
    *,
    timestamp: float,
    first_timestamp: float,
    last_timestamp: float,
    config: LrcParseConfig,
) -> float:
    text = normalize_ws(text)
    if not text:
        return -10.0
    edge_region = _is_edge_timestamp(timestamp, first_timestamp, last_timestamp, config)
    metadata_like = _looks_like_edge_metadata(text)
    heuristic = 0.35 if looks_like_lyric_text(text) else -0.55
    if metadata_like:
        heuristic -= 1.35
        if edge_region:
            heuristic -= 1.1
    elif edge_region:
        heuristic += 0.15
    embedder = _LyricMetadataEmbedder.get(config)
    embedding = float(embedder.lyric_score(text)) if embedder is not None else 0.0
    return heuristic + embedding + 0.01 * len(text)


def _best_group_text(texts: List[str], *, timestamp: float, first_timestamp: float, last_timestamp: float, config: LrcParseConfig) -> str:
    return max(
        texts,
        key=lambda t: (
            _text_selection_score(
                t,
                timestamp=timestamp,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                config=config,
            ),
            1 if looks_like_lyric_text(t) else 0,
            ascii_ratio(t),
            len(normalize_ws(t)),
        ),
    )


def parse_lrc(path: str, config: Optional[LrcParseConfig] = None) -> List[LyricLine]:
    config = config or LrcParseConfig()
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    grouped: Dict[float, List[Tuple[int, str]]] = defaultdict(list)

    for raw_index, raw in enumerate(raw_lines):
        line = raw.strip("\ufeff").strip()
        if not line:
            continue
        if META_RE.match(line):
            continue
        timestamps = [parse_lrc_timestamp(m) for m in TIMESTAMP_RE.findall(line)]
        text = normalize_ws(TIMESTAMP_RE.sub("", line))
        if not timestamps:
            continue
        for ts in timestamps:
            grouped[round(ts, 3)].append((raw_index, text))

    rows: List[LyricLine] = []
    if not grouped:
        return rows
    ordered_timestamps = sorted(grouped)
    first_timestamp = float(ordered_timestamps[0])
    last_timestamp = float(ordered_timestamps[-1])
    for timestamp in ordered_timestamps:
        items = grouped[timestamp]
        texts = [text for _, text in items if text]
        if not texts:
            continue
        chosen = _best_group_text(
            texts,
            timestamp=float(timestamp),
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            config=config,
        )
        alts = [text for text in texts if text != chosen]
        rows.append(
            LyricLine(
                line_id=len(rows),
                timestamp=float(timestamp),
                text=chosen,
                raw_index=min(idx for idx, _ in items),
                alternatives=alts,
            )
        )

    while rows and _looks_like_edge_metadata(rows[0].text):
        rows.pop(0)
    while rows and _looks_like_edge_metadata(rows[-1].text):
        rows.pop()

    for idx, row in enumerate(rows):
        row.line_id = idx
    return rows


def reference_text(lines: List[LyricLine]) -> str:
    return "\n".join(line.text for line in lines)
