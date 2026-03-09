from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .utils import mean, read_json, safe_stem, write_json


def extract_m0_features(payload: dict, min_repaired_word_dur: float = 0.04) -> tuple[dict, dict]:
    chunks_out: List[dict] = []
    negative_gaps = 0
    overlaps = 0
    unmatched = 0
    total_words = 0

    chunks = payload.get("chunks", []) or []
    for idx, chunk in enumerate(chunks):
        words = chunk.get("words", []) or []
        out_words: List[dict] = []
        raw_durs: List[float] = []
        eff_durs: List[float] = []
        for word_idx, word in enumerate(words):
            raw_start = float(word.get("start", 0.0))
            raw_end = float(word.get("end", raw_start))
            start = float(word.get("simulated_start", raw_start))
            end = float(word.get("simulated_end", word.get("end", start)))
            raw_dur = max(0.0, raw_end - raw_start)
            simulated_dur = max(0.0, end - start)
            eff_dur = max(min_repaired_word_dur, simulated_dur if simulated_dur > 0 else raw_dur)
            raw_durs.append(raw_dur)
            eff_durs.append(eff_dur)
            total_words += 1
            if not str(word.get("text") or "").strip():
                unmatched += 1
            out_words.append(
                {
                    "word_index": word_idx,
                    "text": str(word.get("text") or "").strip(),
                    "start": start,
                    "end": end,
                    "audio_start": raw_start,
                    "audio_end": raw_end,
                    "midpoint": float(word.get("simulated_midpoint", 0.5 * (start + end))),
                    "raw_duration": raw_dur,
                    "simulated_duration": float(word.get("simulated_duration", simulated_dur)),
                    "effective_duration": eff_dur,
                    "timing_source": str(word.get("timing_source", "alignment")),
                    "timing_confidence": float(word.get("timing_confidence", word.get("confidence", 1.0) or 1.0)),
                    "is_simulated": "simulated_start" in word or str(word.get("timing_source", "")) == "simulated_from_chunk_context",
                }
            )

        audio_start = float(chunk.get("start", 0.0))
        audio_end = float(chunk.get("end", audio_start))
        start = float(chunk.get("display_start", audio_start))
        end = float(chunk.get("display_end", max(start, audio_end)))
        duration = max(0.0, end - start)
        next_start = (
            float(chunks[idx + 1].get("display_start", chunks[idx + 1].get("start", end)))
            if idx + 1 < len(chunks)
            else end
        )
        gap_to_next = next_start - end
        if gap_to_next < 0:
            negative_gaps += 1
        if idx + 1 < len(chunks) and next_start < end:
            overlaps += 1

        tokens = [token for token in str(chunk.get("text") or "").split() if token]
        chunks_out.append(
            {
                "chunk_id": int(chunk.get("chunk_id", idx)),
                "start": start,
                "end": end,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "display_start": start,
                "display_end": end,
                "text": str(chunk.get("text") or "").strip(),
                "chunk_duration": duration,
                "word_count": len(words),
                "word_count_fallback_text_tokenized": len(tokens),
                "words_per_second": len(words) / duration if duration > 0 else 0.0,
                "min_word_dur": min(eff_durs) if eff_durs else 0.0,
                "gap_to_next": gap_to_next,
                "word_durations": raw_durs,
                "word_indices": list(range(len(words))),
                "words": out_words,
            }
        )

    report = {
        "chunk_count": len(chunks_out),
        "total_words": total_words,
        "max_visible_chunks_at_once": 1 if overlaps == 0 else 2,
        "low_unmatched_word_ratio": (unmatched / total_words) if total_words else 0.0,
        "no_negative_chunk_gaps": negative_gaps == 0,
        "positive_chunk_durations": all(chunk["chunk_duration"] >= 0.0 for chunk in chunks_out),
        "positive_effective_word_durations": all(
            word["effective_duration"] > 0.0 for chunk in chunks_out for word in chunk["words"]
        ),
        "mean_words_per_second": mean(chunk["words_per_second"] for chunk in chunks_out),
    }

    features = {
        "meta": {
            **payload.get("meta", {}),
            "stage": "m0",
            "min_repaired_word_dur": min_repaired_word_dur,
        },
        "chunks": chunks_out,
    }
    return features, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract normalized M0 text/timing features from refined chunk JSON.")
    parser.add_argument("chunking_json", type=str)
    parser.add_argument("--min-repaired-word-dur", type=float, default=0.04)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("-o", "--output", type=str, default="")
    parser.add_argument("--report", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = read_json(Path(args.chunking_json))
    features, report = extract_m0_features(payload, min_repaired_word_dur=float(args.min_repaired_word_dur))
    stem = safe_stem(args.chunking_json)
    out_path = Path(args.output) if args.output else Path(args.artifacts_dir) / "m0" / f"features_text_timing_{stem}.json"
    report_path = Path(args.report) if args.report else Path(args.artifacts_dir) / "m0" / f"validation_report_m0_{stem}.json"
    write_json(out_path, features)
    write_json(report_path, report)
    print(f"[m0] wrote {out_path}")
    print(f"[m0] wrote {report_path}")
    return 0
