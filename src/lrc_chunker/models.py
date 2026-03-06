from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LyricLine:
    line_id: int
    timestamp: float
    text: str
    raw_index: int = 0
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "timestamp": self.timestamp,
            "text": self.text,
            "raw_index": self.raw_index,
            "alternatives": list(self.alternatives),
        }


@dataclass
class WordTiming:
    text: str
    start: float
    end: float
    line_id: Optional[int] = None
    confidence: float = 1.0
    source: str = "alignment"
    index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "line_id": self.line_id,
            "confidence": self.confidence,
            "source": self.source,
            "index": self.index,
        }


@dataclass
class Chunk:
    chunk_id: int
    start: float
    end: float
    text: str
    words: List[WordTiming]
    line_ids: List[int] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "line_ids": list(self.line_ids),
            "scores": dict(self.scores),
            "flags": dict(self.flags),
            "words": [w.to_dict() for w in self.words],
        }
