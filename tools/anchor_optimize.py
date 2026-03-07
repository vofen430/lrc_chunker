#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lrc_chunker.alignment import AlignmentConfig, align_lyrics, build_alignment_payload
from lrc_chunker.chunking import ChunkingConfig, build_chunks
from lrc_chunker.lrc import parse_lrc
from lrc_chunker.utils import safe_stem, write_json
from lrc_chunker.word_refine import refine_payload


@dataclass(frozen=True)
class Anchor:
    text: str
    start: float


def _load_anchors(path: Path) -> List[Anchor]:
    anchors: List[Anchor] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in {"word", "text"}:
                continue
            text = row[0].strip()
            start = float(row[1].strip())
            anchors.append(Anchor(text=text, start=start))
    if not anchors:
        raise ValueError(f"no anchors found in {path}")
    return anchors


def _normalize_word(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch in {"'", "’"})


def _flatten_words(payload: dict) -> List[dict]:
    words: List[dict] = []
    for chunk in payload.get("chunks", []) or []:
        for word in chunk.get("words", []) or []:
            if str(word.get("text") or "").strip():
                words.append(word)
    return words


def _match_anchors(words: Sequence[dict], anchors: Sequence[Anchor]) -> List[Dict[str, object]]:
    matched: List[Dict[str, object]] = []
    cursor = 0
    for anchor in anchors:
        target = _normalize_word(anchor.text)
        candidates: List[Tuple[float, int, dict]] = []
        for idx in range(cursor, len(words)):
            word = words[idx]
            if _normalize_word(str(word.get("text") or "")) != target:
                continue
            err = abs(float(word.get("start", 0.0)) - anchor.start)
            candidates.append((err, idx, word))
        if not candidates:
            matched.append(
                {
                    "anchor_text": anchor.text,
                    "anchor_start": anchor.start,
                    "matched": False,
                }
            )
            continue
        _, idx, word = min(candidates, key=lambda item: item[0])
        cursor = idx + 1
        pred = float(word.get("start", 0.0))
        matched.append(
            {
                "anchor_text": anchor.text,
                "anchor_start": anchor.start,
                "matched": True,
                "matched_index": idx,
                "matched_text": str(word.get("text") or ""),
                "pred_start": pred,
                "abs_error": abs(pred - anchor.start),
                "signed_error": pred - anchor.start,
            }
        )
    return matched


def _score_payload(payload: dict, anchors: Sequence[Anchor]) -> Dict[str, object]:
    matched = _match_anchors(_flatten_words(payload), anchors)
    errors = [float(row["abs_error"]) for row in matched if row.get("matched")]
    return {
        "anchor_count": len(anchors),
        "matched_count": sum(1 for row in matched if row.get("matched")),
        "mae": sum(errors) / len(errors) if errors else 999.0,
        "max_error": max(errors) if errors else 999.0,
        "rows": matched,
    }


def _candidate_refines() -> List[Tuple[str, str, Dict[str, float]]]:
    return [
        ("balanced", "balanced", {}),
        ("slow_attack", "slow_attack", {}),
        ("aggressive", "aggressive", {}),
        ("rap_snap", "rap_snap", {}),
        ("balanced_tight_breath", "balanced", {"breath_flatness_max": 0.52, "breath_rms_ratio": 0.42, "breath_scan_max": 0.20}),
        ("balanced_forward", "balanced", {"start_shift_max": 0.18, "keep_weight": 0.55, "prefer_future_penalty": 0.80}),
        ("aggressive_tight_breath", "aggressive", {"breath_flatness_max": 0.50, "breath_rms_ratio": 0.45, "breath_scan_max": 0.18}),
        ("rap_snap_tight", "rap_snap", {"start_shift_max": 0.20, "prefer_future_penalty": 1.30, "breath_flatness_max": 0.58}),
    ]


def _candidate_alignments() -> List[Dict[str, object]]:
    return [
        {"name": "small", "model": "small.en", "vad_threshold": 0.35, "only_voice_freq": False},
        {"name": "medium", "model": "medium.en", "vad_threshold": 0.35, "only_voice_freq": False},
    ]


def _run_search(audio: Path, lrc: Path, anchors_path: Path, out_dir: Path) -> int:
    anchors = _load_anchors(anchors_path)
    lines = parse_lrc(str(lrc))
    if not lines:
        raise ValueError(f"no usable lyric rows found in {lrc}")

    summary_rows: List[Dict[str, object]] = []
    best_payload: dict | None = None
    best_summary: Dict[str, object] | None = None

    chunk_cfg = ChunkingConfig()

    for align_cfg in _candidate_alignments():
        align_name = str(align_cfg["name"])
        align_dir = out_dir / align_name
        align_dir.mkdir(parents=True, exist_ok=True)
        denoised_path = align_dir / f"{safe_stem(str(audio))}_{align_name}_demucs_vocals.wav"

        config = AlignmentConfig(
            model=str(align_cfg["model"]),
            language="en",
            vad_threshold=float(align_cfg["vad_threshold"]),
            denoiser="auto",
            denoiser_output_path=str(denoised_path),
            alignment_backend="stable_ts",
            allow_lrc_fallback=False,
            only_voice_freq=bool(align_cfg["only_voice_freq"]),
        )
        print(f"[align] model={config.model} only_voice_freq={config.only_voice_freq}")
        words, _, backend_used = align_lyrics(str(audio), lines, config)
        chunks = build_chunks(words, chunk_cfg)
        payload = build_alignment_payload(
            audio_path=str(audio),
            lrc_path=str(lrc),
            lines=lines,
            words=words,
            chunks=chunks,
            config=config,
            backend_used=backend_used,
        )
        align_json = align_dir / "alignment.json"
        write_json(align_json, payload)

        raw_score = _score_payload(payload, anchors)
        summary_rows.append(
            {
                "stage": "align",
                "align_name": align_name,
                "refine_name": "none",
                "mae": raw_score["mae"],
                "max_error": raw_score["max_error"],
                "matched_count": raw_score["matched_count"],
                "path": str(align_json),
            }
        )
        print(f"[score] {align_name}/none mae={raw_score['mae']:.3f} max={raw_score['max_error']:.3f}")

        vocals_path = str(denoised_path) if denoised_path.is_file() else ""
        for refine_name, profile, overrides in _candidate_refines():
            refined, report = refine_payload(
                payload,
                profile=profile,
                audio_mix=str(audio),
                audio_vocals=vocals_path,
                overrides=overrides,
            )
            score = _score_payload(refined, anchors)
            refine_json = align_dir / f"refine_{refine_name}.json"
            refine_report = align_dir / f"refine_{refine_name}_report.json"
            eval_json = align_dir / f"refine_{refine_name}_anchor_eval.json"
            write_json(refine_json, refined)
            write_json(refine_report, report)
            write_json(eval_json, score)
            row = {
                "stage": "refine",
                "align_name": align_name,
                "refine_name": refine_name,
                "mae": score["mae"],
                "max_error": score["max_error"],
                "matched_count": score["matched_count"],
                "path": str(refine_json),
            }
            summary_rows.append(row)
            print(f"[score] {align_name}/{refine_name} mae={score['mae']:.3f} max={score['max_error']:.3f}")
            if best_summary is None or float(row["mae"]) < float(best_summary["mae"]):
                best_summary = row
                best_payload = refined

    summary_rows.sort(key=lambda row: (float(row["mae"]), float(row["max_error"])))
    summary_path = out_dir / "anchor_search_summary.json"
    summary_csv = out_dir / "anchor_search_summary.csv"
    best_path = out_dir / "best_refined.json"
    write_json(summary_path, {"rows": summary_rows, "anchors_path": str(anchors_path)})
    if best_payload is not None:
        write_json(best_path, best_payload)
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "align_name", "refine_name", "mae", "max_error", "matched_count", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    if best_summary:
        print(f"[best] {best_summary['align_name']}/{best_summary['refine_name']} mae={float(best_summary['mae']):.3f} max={float(best_summary['max_error']):.3f}")
        print(f"[best] path={best_summary['path']}")
    print(f"[summary] wrote {summary_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search alignment/refinement settings against manual word-start anchors.")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--lrc", type=str, required=True)
    parser.add_argument("--anchors", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args(argv)
    return _run_search(
        audio=Path(args.audio).expanduser().resolve(),
        lrc=Path(args.lrc).expanduser().resolve(),
        anchors_path=Path(args.anchors).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
