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
    simulated_start: Optional[float] = None
    simulated_end: Optional[float] = None
    simulated_midpoint: Optional[float] = None
    simulated_duration: Optional[float] = None
    line_id: Optional[int] = None
    confidence: float = 1.0
    source: str = "alignment"
    timing_source: str = "alignment"
    timing_confidence: float = 1.0
    index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "simulated_start": self.simulated_start,
            "simulated_end": self.simulated_end,
            "simulated_midpoint": self.simulated_midpoint,
            "simulated_duration": self.simulated_duration,
            "line_id": self.line_id,
            "confidence": self.confidence,
            "source": self.source,
            "timing_source": self.timing_source,
            "timing_confidence": self.timing_confidence,
            "index": self.index,
        }


@dataclass
class Chunk:
    chunk_id: int
    start: float
    end: float
    text: str
    words: List[WordTiming]
    display_start: Optional[float] = None
    display_end: Optional[float] = None
    line_ids: List[int] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "display_start": self.display_start,
            "display_end": self.display_end,
            "text": self.text,
            "line_ids": list(self.line_ids),
            "scores": dict(self.scores),
            "flags": dict(self.flags),
            "words": [w.to_dict() for w in self.words],
        }
