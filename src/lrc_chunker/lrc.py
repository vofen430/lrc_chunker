from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .models import LyricLine
from .utils import ascii_ratio, looks_like_lyric_text, normalize_ws


TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
META_RE = re.compile(r"^\[(ar|ti|al|by|offset|length|re|ve):", re.IGNORECASE)


def parse_lrc_timestamp(match: Tuple[str, str, str]) -> float:
    mm, ss, frac = match
    frac = frac or "0"
    if len(frac) == 1:
        frac = frac + "00"
    elif len(frac) == 2:
        frac = frac + "0"
    return int(mm) * 60.0 + int(ss) + int(frac[:3]) / 1000.0


def _best_group_text(texts: List[str]) -> str:
    return max(
        texts,
        key=lambda t: (
            1 if looks_like_lyric_text(t) else 0,
            ascii_ratio(t),
            len(normalize_ws(t)),
        ),
    )


def parse_lrc(path: str) -> List[LyricLine]:
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
    for timestamp in sorted(grouped):
        items = grouped[timestamp]
        texts = [text for _, text in items if text]
        if not texts:
            continue
        chosen = _best_group_text(texts)
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

    while rows and not looks_like_lyric_text(rows[0].text):
        rows.pop(0)
    while rows and not looks_like_lyric_text(rows[-1].text):
        rows.pop()

    for idx, row in enumerate(rows):
        row.line_id = idx
    return rows


def reference_text(lines: List[LyricLine]) -> str:
    return "\n".join(line.text for line in lines)
