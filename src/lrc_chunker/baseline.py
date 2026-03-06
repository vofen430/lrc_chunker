from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import AlignmentConfig, align_lyrics, build_alignment_payload, default_alignment_output
from .chunking import ChunkingConfig, build_chunks
from .lrc import parse_lrc
from .utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore lyric word timestamps and chunking from audio + LRC.")
    parser.add_argument("audio", type=str)
    parser.add_argument("lrc", type=str)
    parser.add_argument("--model", type=str, default="small.en")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--max-gap", type=float, default=0.35)
    parser.add_argument("--merge-gap", type=float, default=0.12)
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--max-dur", type=float, default=3.2)
    parser.add_argument("--hard-max-chunk-dur", type=float, default=6.0)
    parser.add_argument("--rhythm-weight", type=float, default=2.8)
    parser.add_argument("--hard-line-breaks", action="store_true", default=True)
    parser.add_argument("--no-hard-line-breaks", dest="hard_line_breaks", action="store_false")
    parser.add_argument("--emphasize-long-words", action="store_true", default=True)
    parser.add_argument("--no-emphasize-long-words", dest="emphasize_long_words", action="store_false")
    parser.add_argument("--long-word-single-threshold", type=float, default=0.78)
    parser.add_argument("--long-word-bonus", type=float, default=2.6)
    parser.add_argument("--apply-clamp-max", action="store_true", default=True)
    parser.add_argument("--no-apply-clamp-max", dest="apply_clamp_max", action="store_false")
    parser.add_argument("--denoiser", type=str, default="auto")
    parser.add_argument("--alignment-backend", choices=["stable_ts", "lrc"], default="stable_ts")
    parser.add_argument("--allow-lrc-fallback", action="store_true")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("-o", "--output", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lines = parse_lrc(args.lrc)
    if not lines:
        raise ValueError(f"No usable lyric rows found in {args.lrc}")

    align_config = AlignmentConfig(
        model=args.model,
        language=args.language,
        vad_threshold=float(args.vad_threshold),
        denoiser=args.denoiser,
        alignment_backend=args.alignment_backend,
        allow_lrc_fallback=bool(args.allow_lrc_fallback),
    )
    chunk_config = ChunkingConfig(
        max_gap=float(args.max_gap),
        merge_gap=float(args.merge_gap),
        max_chars=int(args.max_chars),
        max_words=int(args.max_words),
        max_dur=float(args.max_dur),
        hard_max_chunk_dur=float(args.hard_max_chunk_dur),
        rhythm_weight=float(args.rhythm_weight),
        hard_line_breaks=bool(args.hard_line_breaks),
        emphasize_long_words=bool(args.emphasize_long_words),
        long_word_single_threshold=float(args.long_word_single_threshold),
        long_word_bonus=float(args.long_word_bonus),
        apply_clamp_max=bool(args.apply_clamp_max),
    )

    words, _, backend_used = align_lyrics(args.audio, lines, align_config)
    chunks = build_chunks(words, chunk_config)
    payload = build_alignment_payload(
        audio_path=args.audio,
        lrc_path=args.lrc,
        lines=lines,
        words=words,
        chunks=chunks,
        config=align_config,
        backend_used=backend_used,
    )
    payload["meta"]["chunking"] = vars(args)

    out_path = Path(args.output) if args.output else default_alignment_output(args.artifacts_dir, args.audio, args.model)
    write_json(out_path, payload)
    print(f"[baseline] wrote {out_path}")
    return 0
