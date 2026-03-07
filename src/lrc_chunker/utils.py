from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "up",
    "we",
    "when",
    "with",
    "you",
    "your",
}


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / max(1, len(text))


def normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


def token_word_count(text: str) -> int:
    return sum(1 for tok in tokenize(text) if re.search(r"\w", tok))


def looks_like_lyric_text(text: str) -> bool:
    text = normalize_ws(text)
    if not text:
        return False
    lower = text.lower()
    if lower.startswith(("作词", "作曲", "编曲", "词:", "曲:", "by:", "ar:", "ti:", "al:", "offset:")):
        return False
    if re.match(r"^\[(ar|ti|al|by|offset|length|re|ve):", lower):
        return False
    if all(ch in "()-[]{}.,!?/&'\" " for ch in text):
        return False
    return token_word_count(text) > 0 or len(text) >= 2


def safe_stem(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "artifact"


def first_existing_path(candidates: Iterable[object]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.is_file():
            return str(path.resolve())
    return ""


def find_payload_vocals_path(payload: dict) -> str:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    word_refine = meta.get("word_refine", {}) if isinstance(meta, dict) else {}
    return first_existing_path(
        [
            meta.get("denoiser_output_path"),
            meta.get("vocals_path"),
            word_refine.get("audio_vocals") if isinstance(word_refine, dict) else "",
        ]
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = clamp(q, 0.0, 1.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


