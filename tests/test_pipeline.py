from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrc_chunker.alignment import AlignmentConfig, _assign_aligned_words_to_lines, build_alignment_payload, fallback_align_from_lrc
from lrc_chunker.chunking import ChunkingConfig, build_chunks
from lrc_chunker.lrc import parse_lrc
from lrc_chunker.models import LyricLine
from lrc_chunker.motion_m0_extract import extract_m0_features
from lrc_chunker.utils import find_payload_vocals_path
from lrc_chunker.word_refine import PROFILES, _apply_breath_guard, build_parser as build_refine_parser, refine_payload


def test_parse_lrc_removes_head_metadata_and_prefers_primary_language(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.000][by:test_user]",
                "[00:00.000]作曲 : Someone",
                "[00:00.500]Hello from the other side",
                "[00:00.500]你好 来自另一边",
                "[00:03.000]I must have called a thousand times",
                "[00:03.000]我一定打过上千次电话",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_lrc(str(lrc_path))

    assert [row.text for row in rows] == [
        "Hello from the other side",
        "I must have called a thousand times",
    ]


def test_assign_aligned_words_to_lines_uses_lrc_tokens_for_word_text() -> None:
    lines = [
        LyricLine(line_id=0, timestamp=0.0, text="don't go"),
    ]
    aligned_words = [
        {"text": "dont", "start": 0.10, "end": 0.40, "probability": 0.9},
        {"text": "gooo", "start": 0.45, "end": 0.80, "probability": 0.8},
    ]

    words, records = _assign_aligned_words_to_lines(lines, aligned_words)

    assert [word.text for word in words] == ["don't", "go"]
    assert records[0]["word_count"] == 2


def test_fallback_pipeline_produces_non_overlapping_chunks(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.500]Hello from the other side",
                "[00:03.000]I must have called a thousand times",
                "[00:06.000]To tell you I'm sorry",
            ]
        ),
        encoding="utf-8",
    )

    lines = parse_lrc(str(lrc_path))
    words, _, backend_used = fallback_align_from_lrc(lines, audio_path="demo.wav")
    chunks = build_chunks(words, ChunkingConfig())
    payload = build_alignment_payload(
        audio_path="demo.wav",
        lrc_path=str(lrc_path),
        lines=lines,
        words=words,
        chunks=chunks,
        config=AlignmentConfig(alignment_backend="lrc"),
        backend_used=backend_used,
    )

    refined, _ = refine_payload(payload, profile="balanced")
    features, report = extract_m0_features(refined)

    assert features["chunks"]
    assert report["no_negative_chunk_gaps"] is True
    assert report["max_visible_chunks_at_once"] == 1


def test_payload_vocals_path_is_discovered_from_meta(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"fake")
    payload = {
        "meta": {
            "denoiser_output_path": str(vocals),
        }
    }

    assert find_payload_vocals_path(payload) == str(vocals.resolve())


def test_breath_guard_pushes_start_to_voiced_frame() -> None:
    params = dict(PROFILES["balanced"])
    frame_times = np.asarray([0.50, 0.56, 0.62, 0.68, 0.74], dtype=np.float64)
    rms = np.asarray([0.01, 0.015, 0.08, 0.11, 0.10], dtype=np.float64)
    flatness = np.asarray([0.92, 0.86, 0.34, 0.28, 0.30], dtype=np.float64)

    shifted = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        params,
    )

    assert shifted >= 0.62


def test_breath_guard_keeps_voiced_start_when_not_breathy() -> None:
    params = dict(PROFILES["balanced"])
    frame_times = np.asarray([0.50, 0.56, 0.62], dtype=np.float64)
    rms = np.asarray([0.08, 0.10, 0.11], dtype=np.float64)
    flatness = np.asarray([0.30, 0.28, 0.26], dtype=np.float64)

    kept = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        params,
    )

    assert kept == 0.50


def test_slow_attack_profile_waits_for_later_attack() -> None:
    balanced = dict(PROFILES["balanced"])
    slow_attack = dict(PROFILES["slow_attack"])
    frame_times = np.asarray([0.50, 0.56, 0.62, 0.68, 0.74, 0.80], dtype=np.float64)
    rms = np.asarray([0.03, 0.04, 0.06, 0.09, 0.13, 0.14], dtype=np.float64)
    flatness = np.asarray([0.40, 0.38, 0.32, 0.28, 0.24, 0.22], dtype=np.float64)

    balanced_shift = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        balanced,
    )
    slow_attack_shift = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        slow_attack,
    )

    assert balanced_shift == 0.62
    assert slow_attack_shift >= 0.74


def test_refine_payload_syncs_top_level_words() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "words": [
                    {"text": "hello", "start": 0.10, "end": 0.40},
                    {"text": "world", "start": 0.45, "end": 0.90},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 0.00, "end": 0.30},
            {"text": "world", "start": 0.35, "end": 0.70},
        ],
    }

    refined, _ = refine_payload(payload, profile="balanced")

    assert refined["words"] == refined["chunks"][0]["words"]


def test_refine_payload_can_anchor_line_first_word_to_lrc_timestamp() -> None:
    payload = {
        "meta": {},
        "lines": [
            {"line_id": 0, "timestamp": 1.0, "text": "hello world"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.6,
                "end": 1.8,
                "text": "hello world",
                "words": [
                    {"text": "hello", "start": 0.62, "end": 0.74, "line_id": 0},
                    {"text": "world", "start": 1.15, "end": 1.70, "line_id": 0},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 0.62, "end": 0.74, "line_id": 0},
            {"text": "world", "start": 1.15, "end": 1.70, "line_id": 0},
        ],
    }

    refined, report = refine_payload(
        payload,
        profile="balanced",
        use_lrc_anchors=True,
        lrc_anchor_window=0.30,
        lrc_anchor_weight=4.0,
        lrc_anchor_keep_weight=0.20,
        lrc_anchor_span_words=4,
        lrc_anchor_max_ratio=0.80,
    )

    assert refined["chunks"][0]["words"][0]["start"] == 1.0
    assert refined["chunks"][0]["words"][1]["start"] > 1.15
    assert refined["chunks"][0]["words"][1]["start"] - 1.15 < 1.0 - 0.62
    assert refined["chunks"][0]["words"][1]["end"] > refined["chunks"][0]["words"][1]["start"]
    assert report["lrc_anchor_moves"] == 1


def test_refine_cli_defaults_to_small_flow_profile_and_lrc_anchors() -> None:
    parser = build_refine_parser()
    args = parser.parse_args(["demo.json"])

    assert args.profile == "slow_attack"
    assert args.use_lrc_anchors is True
