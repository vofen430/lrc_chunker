from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import AlignmentConfig, align_lyrics, build_alignment_payload, default_alignment_output
from .chunking import ChunkingConfig, build_chunks
from .lrc import LrcParseConfig, parse_lrc
from .utils import safe_stem, write_json


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
    parser.add_argument("--chunker-model", choices=["semantic_dp", "legacy"], default="semantic_dp")
    parser.add_argument("--min-chunk-dur", type=float, default=0.30)
    parser.add_argument("--short-gap-block", type=float, default=0.12)
    parser.add_argument("--hard-overlap-block", type=float, default=0.03)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--embedding-weight", type=float, default=2.4)
    parser.add_argument("--gap-weight", type=float, default=0.30)
    parser.add_argument("--length-weight", type=float, default=0.65)
    parser.add_argument("--line-start-anchor", action="store_true", default=True)
    parser.add_argument("--no-line-start-anchor", dest="line_start_anchor", action="store_false")
    parser.add_argument("--line-start-anchor-tolerance", type=float, default=1.0)
    parser.add_argument("--metadata-head-window-seconds", type=float, default=45.0)
    parser.add_argument("--metadata-tail-window-seconds", type=float, default=45.0)
    parser.add_argument("--metadata-use-embedding", action="store_true", default=True)
    parser.add_argument("--no-metadata-embedding", dest="metadata_use_embedding", action="store_false")
    parser.add_argument("--embedding-model-path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--embedding-instruction",
        type=str,
        default="Decide whether a lyric phrase boundary before the target word is natural for karaoke subtitle chunking.",
    )
    parser.add_argument("--embedding-local-only", action="store_true", default=True)
    parser.add_argument("--allow-embedding-download", dest="embedding_local_only", action="store_false")
    parser.add_argument("--denoiser", type=str, default="auto")
    parser.add_argument("--denoiser-output", type=str, default="")
    parser.add_argument("--only-voice-freq", action="store_true")
    parser.add_argument("--alignment-backend", choices=["stable_ts", "qwen_forced_aligner", "lrc"], default="stable_ts")
    parser.add_argument("--allow-lrc-fallback", action="store_true")
    parser.add_argument("--qwen-aligner-checkpoint", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--qwen-device", type=str, default="cuda:0")
    parser.add_argument("--qwen-dtype", type=str, default="bfloat16")
    parser.add_argument("--qwen-attn-implementation", type=str, default="")
    parser.add_argument("--align-window-target-seconds", type=float, default=30.0)
    parser.add_argument("--align-window-max-seconds", type=float, default=170.0)
    parser.add_argument("--align-window-max-lines", type=int, default=1)
    parser.add_argument("--align-window-pre-roll", type=float, default=1.0)
    parser.add_argument("--align-window-post-roll", type=float, default=1.0)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("-o", "--output", type=str, default="")
    return parser


def _default_denoiser_output(artifacts_dir: str, audio_path: str, denoiser: str) -> str:
    requested = str(denoiser or "").strip().lower()
    if requested in {"", "none", "off", "false", "0"}:
        return ""
    return str(Path(artifacts_dir) / "denoised" / f"{safe_stem(audio_path)}_demucs_vocals.wav")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lines = parse_lrc(
        args.lrc,
        LrcParseConfig(
            metadata_head_window_seconds=float(args.metadata_head_window_seconds),
            metadata_tail_window_seconds=float(args.metadata_tail_window_seconds),
            metadata_embedding_model_path=args.embedding_model_path,
            metadata_embedding_local_only=bool(args.embedding_local_only),
            use_metadata_embedding=bool(args.metadata_use_embedding),
        ),
    )
    if not lines:
        raise ValueError(f"No usable lyric rows found in {args.lrc}")

    align_config = AlignmentConfig(
        model=args.model,
        language=args.language,
        vad_threshold=float(args.vad_threshold),
        denoiser=args.denoiser,
        denoiser_output_path=args.denoiser_output or _default_denoiser_output(args.artifacts_dir, args.audio, args.denoiser),
        alignment_backend=args.alignment_backend,
        allow_lrc_fallback=bool(args.allow_lrc_fallback),
        only_voice_freq=bool(args.only_voice_freq),
        qwen_aligner_checkpoint=args.qwen_aligner_checkpoint,
        qwen_device=args.qwen_device,
        qwen_dtype=args.qwen_dtype,
        qwen_attn_implementation=args.qwen_attn_implementation,
        align_window_target_seconds=float(args.align_window_target_seconds),
        align_window_max_seconds=float(args.align_window_max_seconds),
        align_window_max_lines=int(args.align_window_max_lines),
        align_window_pre_roll=float(args.align_window_pre_roll),
        align_window_post_roll=float(args.align_window_post_roll),
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
        chunker_model=args.chunker_model,
        min_chunk_dur=float(args.min_chunk_dur),
        short_gap_block=float(args.short_gap_block),
        hard_overlap_block=float(args.hard_overlap_block),
        semantic_weight=float(args.semantic_weight),
        embedding_weight=float(args.embedding_weight),
        gap_weight=float(args.gap_weight),
        length_weight=float(args.length_weight),
        line_start_anchor=bool(args.line_start_anchor),
        line_start_anchor_tolerance=float(args.line_start_anchor_tolerance),
        embedding_model_path=args.embedding_model_path,
        embedding_instruction=args.embedding_instruction,
        embedding_local_only=bool(args.embedding_local_only),
    )

    words, _, backend_used = align_lyrics(args.audio, lines, align_config)
    chunks = build_chunks(words, chunk_config, line_timestamps={line.line_id: line.timestamp for line in lines})
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
